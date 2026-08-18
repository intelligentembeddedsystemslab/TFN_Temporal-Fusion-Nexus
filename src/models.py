import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_observation_mask(x: Tensor, value_mask: Tensor | None) -> Tensor:
    """
    Build the sparsity-preserving network input from the feature-level mask M.

    Paper: a binary mask M in {0,1}^(B,T,F) marks observed features; state
    updates are applied only to observed features and missing ones are ignored,
    with no imputation. Missing entries are therefore zeroed (they contribute
    nothing to the gate pre-activations) and M is concatenated so that
    missingness itself remains informative to the network.

    Returns a tensor of shape (B, T, 2F) -- or (B, T, F) if no mask is given.
    """
    if value_mask is None:
        return x

    if x.shape != value_mask.shape:
        raise ValueError(f"value_mask shape {value_mask.shape} does not match x shape {x.shape}")

    m = value_mask.to(dtype=x.dtype)
    return torch.cat([x * m, m], dim=-1)


def encoder_input_size() -> int:
    """Input width of the encoder cells: F observed values plus F mask channels."""
    return 2 * len(CONFIG['ts_features'])


def build_notes_causal_attention_mask(
    timesteps: Tensor,
    notes_timesteps: Tensor,
    notes_mask: Tensor | None,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """
    Build a boolean cross-attention mask that blocks notes from the future.

    Returns:
        attn_mask: Boolean mask of shape (B * num_heads, L, S) with True=blocked.
        has_valid_causal_note: Boolean mask of shape (B, L) indicating which query
            timesteps have at least one non-padded note at or before the query time.
    """
    B, L = timesteps.shape
    _, S = notes_timesteps.shape

    t_i = timesteps.unsqueeze(-1)          # (B, L, 1)
    t_notes = notes_timesteps.unsqueeze(1) # (B, 1, S)
    future_mask = t_notes > t_i            # (B, L, S), True => block

    if notes_mask is not None:
        valid_notes = notes_mask.bool().unsqueeze(1).expand(-1, L, -1)
    else:
        valid_notes = torch.ones_like(future_mask, dtype=torch.bool)

    has_valid_causal_note = (~future_mask & valid_notes).any(dim=-1)  # (B, L)
    safe_mask = future_mask.clone()

    # MultiheadAttention cannot handle rows where every key is masked. For those
    # rows, temporarily unmask one real note to keep the softmax finite, then
    # zero the resulting attention output afterwards so no future note is used.
    if (~has_valid_causal_note).any():
        batch_idx, query_idx = (~has_valid_causal_note).nonzero(as_tuple=True)
        if notes_mask is not None:
            fallback_note_idx = notes_mask.bool().float().argmax(dim=-1)
        else:
            fallback_note_idx = torch.zeros(B, dtype=torch.long, device=timesteps.device)
        safe_mask[batch_idx, query_idx, fallback_note_idx[batch_idx]] = False

    attn_mask = safe_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B * num_heads, L, S)
    return attn_mask, has_valid_causal_note


class StaticEncoder(nn.Module):
    """
    Encodes static covariates with a variable-wise selection step.

    Each static variable -- categorical or numerical -- gets its own embedding,
    and a selection network produces one weight per variable so the model can
    decide which static covariates matter. This is what makes the downstream
    fusion "variable-wise" in the sense used by the paper.
    """

    def __init__(self, categorical_cardinalities):
        super(StaticEncoder, self).__init__()
        self.categorical_cols = CONFIG['static_categorical_cols']
        self.numerical_cols = CONFIG['static_numerical_cols']
        self.embedding_dim = CONFIG['static_embedding_dim']
        self.num_variables = len(self.categorical_cols) + len(self.numerical_cols)

        if len(categorical_cardinalities) != len(self.categorical_cols):
            raise ValueError("Length of categorical_cardinalities does not match number of categorical columns.")

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, self.embedding_dim)
            for num_categories in categorical_cardinalities
        ])

        # One projection per numerical variable, so variable identity survives
        # into the selection weights.
        self.numerical_processors = nn.ModuleList([
            nn.Linear(1, self.embedding_dim)
            for _ in self.numerical_cols
        ])

        flat_dim = self.embedding_dim * self.num_variables
        self.variable_selection = nn.Sequential(
            nn.Linear(flat_dim, flat_dim),
            nn.ReLU(),
            nn.Linear(flat_dim, self.num_variables),
        )

        self.fc = nn.Linear(self.embedding_dim, CONFIG['static_output_dim'])

    def forward(self, categorical_features, numerical_features):
        per_variable = []
        for i, embedding in enumerate(self.embeddings):
            per_variable.append(embedding(categorical_features[:, i]))
        for i, processor in enumerate(self.numerical_processors):
            per_variable.append(processor(numerical_features[:, i:i + 1]))

        stacked = torch.stack(per_variable, dim=1)          # (B, V, embedding_dim)
        flat = stacked.flatten(start_dim=1)                 # (B, V * embedding_dim)

        weights = torch.softmax(self.variable_selection(flat), dim=-1)  # (B, V)
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=1)         # (B, embedding_dim)

        out = F.relu(self.fc(selected))
        return out, weights


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class NotesEncoder(nn.Module):
    def __init__(self, model_name=None):
        super(NotesEncoder, self).__init__()
        model_name = model_name or CONFIG['notes_encoder_model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    def forward(self, input_texts):
        # input_texts: list of strings
        tokenized_input = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**tokenized_input)
            embeddings = average_pool(outputs.last_hidden_state, tokenized_input['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings  # Returns embeddings as a tensor


class VanillaTimeSeriesEncoder(nn.Module):
    # simple time series encoder
    # this uses a unidirectional LSTM to represent time sequences with its hidden states
    # in: time series up to timestep t, out: hidden states
    def __init__(self):
        super(VanillaTimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=encoder_input_size(), hidden_size=CONFIG['lstm_hidden_size'], num_layers=CONFIG['lstm_num_layers'], batch_first=True, bidirectional=False)

    def forward(self, x, elapsed_times=None, mask=None):
        # elapsed times and mask are never used here, in signature for consistency
        # x: input in shape (batch size, sequence length, input features)
        lstm_out, (hn, cn) = self.lstm(x)

        # return lstm_out: hidden states for each timestep of last layer
        # None for attn_weights
        return lstm_out, None


class TLSTMCell(nn.Module):
    """
    Time-aware LSTM cell (paper Eqs. 1-3).

      - Memory decomposition:
          C^S_{t-1} = tanh(W_decomp * C_{t-1} + b_decomp)          (Eq. 1)
          C^T_{t-1} = C_{t-1} - C^S_{t-1}                          (Eq. 2)
          C*_{t-1}  = C^T_{t-1} + C^S_{t-1} * g(dt)                (Eq. 3)

      - Time gate g(dt) = exp(-relu(softplus(w_decay) * dt + b_decay)).

        The paper writes g(dt) = exp(-W_decay * dt + b_decay). Taken literally
        that is unbounded: a negative W_decay gives exp(+|W| dt), which explodes
        on the multi-year gaps present in this cohort, and a positive b_decay
        makes g > 1 at dt = 0, amplifying memory across a zero-length gap. We
        therefore use the standard guarded form (as in GRU-D), which keeps
        g in (0, 1] and reduces to the paper's expression wherever that
        expression is well behaved. w_decay and b_decay are per-unit vectors,
        giving the feature-wise decay the paper describes ("time gates of size
        512" in Supplement B).

      - Then standard LSTM gating on the decayed memory.
    """

    def __init__(self, input_size, hidden_size):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Decomposition weights
        self.W_decomp = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_decomp = nn.Parameter(torch.Tensor(hidden_size))

        # Input gate
        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Ui = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.Wf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.Wo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate cell
        self.Wc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bc = nn.Parameter(torch.Tensor(hidden_size))

        # Time gate (learnable, per hidden unit)
        self.w_decay_raw = nn.Parameter(torch.Tensor(hidden_size))
        self.b_decay = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        decay_params = {id(self.w_decay_raw), id(self.b_decay)}
        for weight in self.parameters():
            if id(weight) not in decay_params:
                nn.init.uniform_(weight, -stdv, stdv)

        # Spread initial half-lives log-uniformly so the units start at
        # genuinely different timescales rather than collapsing onto one.
        # A uniform(-stdv, stdv) init here would make roughly half the decay
        # rates negative and blow up on long gaps, which is why these
        # parameters are excluded from the loop above.
        lo = math.log(CONFIG['decay_half_life_min_days'])
        hi = math.log(CONFIG['decay_half_life_max_days'])
        half_life = torch.exp(torch.empty(self.hidden_size).uniform_(lo, hi))
        rate = math.log(2.0) / half_life
        with torch.no_grad():
            # softplus^-1 so that softplus(w_decay_raw) == rate
            self.w_decay_raw.copy_(torch.log(torch.expm1(rate)))
            self.b_decay.zero_()

    def decay_half_lives(self) -> Tensor:
        """Learned half-life per hidden unit, in days. Useful for interpretation."""
        with torch.no_grad():
            return math.log(2.0) / F.softplus(self.w_decay_raw).clamp_min(1e-12)

    def map_elapse_time(self, t):
        """
        Maps elapsed time t -> per-unit decay factor in (0, 1].

        t shape: (batch_size, 1); returns (batch_size, hidden_size).
        """
        t = torch.nan_to_num(
            t.to(dtype=self.b_decay.dtype),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )
        t_clamped = torch.clamp(t, min=0.0, max=1e6)

        rate = F.softplus(self.w_decay_raw).unsqueeze(0)          # (1, H), >= 0
        decay_arg = rate * t_clamped + self.b_decay.unsqueeze(0)  # (B, H)
        return torch.exp(-F.relu(decay_arg))

    def forward(self, x_t, states, elapsed_time):
        """
        x_t: (batch_size, input_size) at the current step
        states: (h_prev, c_prev), each (batch_size, hidden_size)
        elapsed_time: (batch_size,) or (batch_size, 1)

        returns (h_t, c_t)
        """
        h_prev, c_prev = states

        if elapsed_time.dim() == 1:
            elapsed_time = elapsed_time.unsqueeze(1)

        # 1) Time gate
        T = self.map_elapse_time(elapsed_time)  # (B, hidden_size)

        # 2) Decompose and decay only the short-term component
        C_S = torch.tanh(c_prev @ self.W_decomp + self.b_decomp)  # Eq. 1
        C_T = c_prev - C_S                                        # Eq. 2
        c_prev_decayed = C_T + C_S * T                            # Eq. 3

        # 3) Standard LSTM gates
        i_t = torch.sigmoid(x_t @ self.Wi + h_prev @ self.Ui + self.bi)
        f_t = torch.sigmoid(x_t @ self.Wf + h_prev @ self.Uf + self.bf)
        o_t = torch.sigmoid(x_t @ self.Wo + h_prev @ self.Uo + self.bo)
        g_t = torch.tanh(x_t @ self.Wc + h_prev @ self.Uc + self.bc)

        # 4) Update cell and hidden state
        c_t = f_t * c_prev_decayed + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TimeAwareLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList([
            TLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, input_seq, elapsed_times, step_mask=None, initial_states=None):
        """
        input_seq: (batch_size, seq_length, input_size)
        elapsed_times: (batch_size, seq_length)
        step_mask: (batch_size, seq_length), True where the timestep is real.
            Padded steps carry the previous state forward untouched.
        initial_states: List of tuples [(h_0, c_0), ..., (h_n, c_n)]
        """
        batch_size, seq_length, _ = input_seq.size()

        if initial_states is None:
            h_t = [input_seq.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c_t = [input_seq.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_t, c_t = [list(s) for s in zip(*initial_states)]

        outputs = []

        for t in range(seq_length):
            x = input_seq[:, t, :]
            delta_t = elapsed_times[:, t]

            keep = None
            if step_mask is not None:
                keep = step_mask[:, t].unsqueeze(-1).to(dtype=input_seq.dtype)

            for layer in range(self.num_layers):
                h_new, c_new = self.lstm_cells[layer](x, (h_t[layer], c_t[layer]), delta_t)
                if keep is not None:
                    # Padded timesteps must not perturb the recurrent state.
                    h_new = keep * h_new + (1.0 - keep) * h_t[layer]
                    c_new = keep * c_new + (1.0 - keep) * c_t[layer]
                h_t[layer], c_t[layer] = h_new, c_new
                x = h_t[layer]

            outputs.append(h_t[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_length, hidden_size)
        return outputs, (h_t, c_t)


class TimeAwareAttentionEncoder(nn.Module):
    def __init__(self, use_temporal_attention=True):
        super(TimeAwareAttentionEncoder, self).__init__()
        self.input_size = encoder_input_size()
        self.use_temporal_attention = use_temporal_attention
        self.lstm = TimeAwareLSTM(input_size=self.input_size, hidden_size=CONFIG['lstm_hidden_size'], num_layers=CONFIG['lstm_num_layers'])
        self.attention = nn.MultiheadAttention(embed_dim=CONFIG['lstm_hidden_size'], num_heads=CONFIG['num_heads'], batch_first=True)
        assert CONFIG['lstm_hidden_size'] % CONFIG['num_heads'] == 0, "embed_dim must be divisible by num_heads"

        self.layer_norm_1 = nn.LayerNorm(CONFIG['lstm_hidden_size'])

        self.ff = nn.Sequential(
            nn.Linear(CONFIG['lstm_hidden_size'], 4 * CONFIG['lstm_hidden_size']),
            nn.ReLU(),
            nn.Linear(4 * CONFIG['lstm_hidden_size'], CONFIG['lstm_hidden_size']),
        )
        self.layer_norm_2 = nn.LayerNorm(CONFIG['lstm_hidden_size'])

    def forward(self, x, elapsed_times, mask=None):
        """
        x: Input sequence of shape (batch_size, seq_length, input_size)
        elapsed_times: Elapsed times of shape (batch_size, seq_length)
        """
        elapsed_times = torch.nan_to_num(
            elapsed_times.to(device=x.device, dtype=x.dtype),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).clamp_min(0.0)

        if mask is not None:
            mask = mask.bool().to(device=x.device)
            elapsed_times = elapsed_times.masked_fill(~mask, 0.0)

        lstm_out, (hn, cn) = self.lstm(x, elapsed_times, step_mask=mask)
        batch_size, seq_len, _ = lstm_out.size()

        # True = ignore
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        key_padding_mask = None
        if mask is not None:
            # mask: True=valid, False=pad -> key_padding_mask: True=ignore, False=keep
            key_padding_mask = ~mask.bool()
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
            lstm_out = lstm_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        attn_weights = None
        if self.use_temporal_attention:
            attn_output, attn_weights = self.attention(
                query=lstm_out,
                key=lstm_out,
                value=lstm_out,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
                attn_mask=causal_mask,
                is_causal=True,
            )
            lstm_out = self.layer_norm_1(lstm_out + attn_output)
            ff_out = self.ff(lstm_out)
            lstm_out = self.layer_norm_2(lstm_out + ff_out)

            if mask is not None:
                lstm_out = lstm_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return lstm_out, attn_weights


class TimeAwareEncoder(nn.Module):
    def __init__(self):
        super(TimeAwareEncoder, self).__init__()
        self.lstm = TimeAwareLSTM(input_size=encoder_input_size(), hidden_size=CONFIG['lstm_hidden_size'], num_layers=CONFIG['lstm_num_layers'])
        self.ff = nn.Linear(CONFIG['lstm_hidden_size'], len(CONFIG['ts_features']))

    def forward(self, x, elapsed_times, mask=None):
        lstm_out, (hn, cn) = self.lstm(x, elapsed_times, step_mask=mask)
        out = self.ff(lstm_out)
        return out


class TMLSTMDecoder(nn.Module):
    """
    Decoder used during training only (paper Sec. Methods; Supplement B).

    Given the Nexus representation at time t, it rolls forward H steps to
    reconstruct x_{t+1..t+H} at their actual observation times, supplying the
    multi-horizon term of L_recon (Eq. 7). Two TM-LSTM layers, hidden size 512.
    """

    def __init__(self, hidden_size, out_features, num_layers, horizon):
        super(TMLSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers
        self.horizon = horizon

        self.cells = nn.ModuleList([
            TLSTMCell(out_features if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.state_init = nn.Linear(hidden_size, 2 * num_layers * hidden_size)
        self.readout = nn.Linear(hidden_size, out_features)

    def forward(self, z, x_seed, future_elapsed):
        """
        z: (N, hidden_size) encoded representation at the decode origin
        x_seed: (N, out_features) last observed values at the origin
        future_elapsed: (N, H) gap from each future step to the previous one

        returns (N, H, out_features)
        """
        N = z.size(0)
        init = self.state_init(z).view(N, 2, self.num_layers, self.hidden_size)
        h_t = [init[:, 0, i, :] for i in range(self.num_layers)]
        c_t = [init[:, 1, i, :] for i in range(self.num_layers)]

        preds = []
        step_input = x_seed
        for step in range(self.horizon):
            delta_t = future_elapsed[:, step]
            x = step_input
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.cells[layer](x, (h_t[layer], c_t[layer]), delta_t)
                x = h_t[layer]
            pred = self.readout(x)
            preds.append(pred)
            step_input = pred  # autoregressive rollout

        return torch.stack(preds, dim=1)


class _MultiModalBase(nn.Module):
    """
    Shared trunk: time-aware encoding, variable-wise static gating, causal
    cross-attention over clinical notes, and the learnable Nexus transform.
    """

    def __init__(self, lstm_encoder, categorical_cardinalities=None, use_static=False, use_notes=False):
        super(_MultiModalBase, self).__init__()
        self.use_static = use_static
        self.use_notes = use_notes
        self.lstm_encoder = lstm_encoder

        hidden = CONFIG['lstm_hidden_size']

        if use_static:
            assert categorical_cardinalities is not None
            self.static_encoder = StaticEncoder(categorical_cardinalities)
            # Variable-wise gating: a sigmoid gate decides, per hidden unit and
            # per timestep, how much of the static signal to admit.
            self.static_gate = nn.Linear(hidden + CONFIG['static_output_dim'], hidden)
            self.static_candidate = nn.Linear(hidden + CONFIG['static_output_dim'], hidden)
            self.static_norm = nn.LayerNorm(hidden)

        if use_notes:
            self.downproj = nn.Linear(CONFIG['notes_embedding_dim'], hidden)
            self.cross_attention = nn.MultiheadAttention(embed_dim=hidden, num_heads=CONFIG['num_heads'], batch_first=True)
            self.layer_norm = nn.LayerNorm(hidden)

        # Sigma(.): the learnable transformation that yields the Nexus embedding.
        self.nexus = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.nexus_norm = nn.LayerNorm(hidden)

    def encode(self, x, elapsed_times=None, timesteps=None, notes_embeddings=None,
               notes_timesteps=None, static_features=None, mask=None, notes_mask=None,
               value_mask=None):
        """
        Returns (Z, attn_weights_tuple, static_encoding, static_variable_weights)
        where Z is the Nexus embedding of shape (B, T, hidden_size).
        """
        x = apply_observation_mask(x, value_mask)
        hidden_states, attn_weights = self.lstm_encoder(x, elapsed_times, mask)

        assert not torch.isnan(hidden_states).any(), "NaN detected in encoder output"

        static_encoding = None
        static_weights = None
        if self.use_static:
            assert static_features is not None
            static_encoding, static_weights = self.static_encoder(static_features[0], static_features[1])
            expanded_static = static_encoding.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            joined = torch.cat([hidden_states, expanded_static], dim=-1)

            gate = torch.sigmoid(self.static_gate(joined))
            candidate = self.static_candidate(joined)
            hidden_states = self.static_norm(hidden_states + gate * candidate)

        attn_weights_notes = None
        if self.use_notes and notes_embeddings is not None and notes_embeddings.size(1) > 0:
            assert not torch.isnan(notes_embeddings).any(), "NaN detected in notes embeddings"
            notes_embeddings = F.relu(self.downproj(notes_embeddings))

            padding_notes_mask = None
            has_valid_note = None
            if notes_mask is not None:
                padding_notes_mask = ~notes_mask.bool()
                has_valid_note = notes_mask.bool().any(dim=1)
                if (~has_valid_note).any():
                    padding_notes_mask = padding_notes_mask.clone()
                    padding_notes_mask[~has_valid_note, 0] = False

            attn_mask = None
            has_valid_causal_note = None
            if timesteps is not None and notes_timesteps is not None:
                attn_mask, has_valid_causal_note = build_notes_causal_attention_mask(
                    timesteps=timesteps,
                    notes_timesteps=notes_timesteps,
                    notes_mask=notes_mask,
                    num_heads=self.cross_attention.num_heads,
                )

            attn_output, attn_weights_notes = self.cross_attention(
                query=hidden_states,
                key=notes_embeddings,
                value=notes_embeddings,
                attn_mask=attn_mask,
                key_padding_mask=padding_notes_mask,
                is_causal=False,  # custom causal cross-attention mask is passed via attn_mask
            )

            if has_valid_causal_note is not None:
                valid_rows = has_valid_causal_note.unsqueeze(-1).to(attn_output.dtype)
                attn_output = attn_output * valid_rows
                attn_weights_notes = self._zero_invalid_weights(attn_weights_notes, has_valid_causal_note)
            elif has_valid_note is not None and (~has_valid_note).any():
                attn_output = attn_output * has_valid_note.view(-1, 1, 1).to(attn_output.dtype)
                attn_weights_notes = self._zero_invalid_weights(attn_weights_notes, has_valid_note)

            hidden_states = self.layer_norm(hidden_states + attn_output)

        # Sigma(.) -> Nexus
        Z = self.nexus_norm(hidden_states + self.nexus(hidden_states))
        if mask is not None:
            Z = Z.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        return Z, (attn_weights, attn_weights_notes), static_encoding, static_weights

    @staticmethod
    def _zero_invalid_weights(weights, valid):
        if weights is None:
            return None
        if weights.dim() == 3:
            if valid.dim() == 2:
                return weights * valid.unsqueeze(-1).to(weights.dtype)
            return weights * valid.view(-1, 1, 1).to(weights.dtype)
        if weights.dim() == 4:
            if valid.dim() == 2:
                return weights * valid.unsqueeze(1).unsqueeze(-1).to(weights.dtype)
            return weights * valid.view(-1, 1, 1, 1).to(weights.dtype)
        return weights


class MultiModal(_MultiModalBase):
    """
    Temporal Fusion Nexus.

    forward() returns:
        predictions: (B, T, H, F) reconstruction of the next H steps from each
            timestep, or None when future_elapsed is not supplied (inference).
        Z: (B, T, hidden_size) Nexus embedding -- the model's task-agnostic output.
        attn: (temporal_attention_weights, notes_cross_attention_weights)
        static_encoding: (B, static_output_dim) or None
        static_weights: (B, V) variable-selection weights or None
    """

    def __init__(self, lstm_encoder, categorical_cardinalities=None, use_static=False, use_notes=False):
        super(MultiModal, self).__init__(lstm_encoder, categorical_cardinalities, use_static, use_notes)
        self.decoder = TMLSTMDecoder(
            hidden_size=CONFIG['lstm_hidden_size'],
            out_features=len(CONFIG['ts_features']),
            num_layers=CONFIG['decoder_num_layers'],
            horizon=CONFIG['prediction_horizon'],
        )

    def forward(self, x, elapsed_times=None, timesteps=None, notes_embeddings=None,
                notes_timesteps=None, static_features=None, mask=None, notes_mask=None,
                value_mask=None, future_elapsed=None, decode_mask=None):
        """
        future_elapsed: (B, T, H) gaps between consecutive future observation
            times. Required for the reconstruction loss; omit at inference.
        decode_mask: optional (B, T) boolean selecting which timesteps to decode
            from. Positions outside the mask are never run through the decoder,
            which is what actually bounds decoder memory on long sequences.
            Predictions at those positions are returned as zeros and are
            excluded from the loss by the matching target mask.
        """
        Z, attn, static_encoding, static_weights = self.encode(
            x=x, elapsed_times=elapsed_times, timesteps=timesteps,
            notes_embeddings=notes_embeddings, notes_timesteps=notes_timesteps,
            static_features=static_features, mask=mask, notes_mask=notes_mask,
            value_mask=value_mask,
        )

        predictions = None
        if future_elapsed is not None:
            B, T, D = Z.shape
            H = self.decoder.horizon
            n_features = self.decoder.out_features
            seed = x if value_mask is None else x * value_mask.to(dtype=x.dtype)

            z_flat = Z.reshape(B * T, D)
            seed_flat = seed.reshape(B * T, n_features)
            elapsed_flat = future_elapsed.reshape(B * T, H)

            if decode_mask is None:
                predictions = self.decoder(z_flat, seed_flat, elapsed_flat).reshape(B, T, H, n_features)
            else:
                idx = decode_mask.reshape(-1).nonzero(as_tuple=True)[0]
                selected = self.decoder(z_flat[idx], seed_flat[idx], elapsed_flat[idx])
                dense = z_flat.new_zeros(B * T, H, n_features)
                predictions = dense.index_put((idx,), selected).reshape(B, T, H, n_features)

        return predictions, Z, attn, static_encoding, static_weights


class MultiModalVAE(_MultiModalBase):
    """
    Variational variant. NOT part of the published Temporal Fusion Nexus --
    the paper's Nexus is a deterministic embedding. Kept for exploratory work;
    do not use it to reproduce paper results.
    """

    def __init__(self, lstm_encoder, categorical_cardinalities=None, use_static=False, use_notes=False):
        super(MultiModalVAE, self).__init__(lstm_encoder, categorical_cardinalities, use_static, use_notes)
        hidden = CONFIG['lstm_hidden_size']
        self.mean_encoder = nn.Linear(hidden, hidden)
        self.logvar_encoder = nn.Linear(hidden, hidden)
        self.ff = nn.Linear(hidden, len(CONFIG['ts_features']))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, elapsed_times=None, timesteps=None, notes_embeddings=None,
                notes_timesteps=None, static_features=None, mask=None, notes_mask=None,
                value_mask=None):
        Z, attn, static_encoding, static_weights = self.encode(
            x=x, elapsed_times=elapsed_times, timesteps=timesteps,
            notes_embeddings=notes_embeddings, notes_timesteps=notes_timesteps,
            static_features=static_features, mask=mask, notes_mask=notes_mask,
            value_mask=value_mask,
        )

        mu = self.mean_encoder(Z)
        logvar = self.logvar_encoder(Z)
        z = self.reparameterize(mu, logvar)
        outputs = self.ff(z)

        return outputs, Z, attn, static_encoding, mu, logvar


### NEURAL NET CLASSIFIER

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # 1 output for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)
