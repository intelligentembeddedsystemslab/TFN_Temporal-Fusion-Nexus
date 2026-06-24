import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from config import CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_missing_value_sentinel(x: Tensor, value_mask: Tensor | None, sequence_mask: Tensor | None = None) -> Tensor:
    if value_mask is None:
        return x

    if x.shape != value_mask.shape:
        raise ValueError(f"value_mask shape {value_mask.shape} does not match x shape {x.shape}")

    missing_sentinel = torch.full_like(x, CONFIG['MISSING_TS_SENTINEL'])
    masked_x = torch.where(value_mask.bool(), x, missing_sentinel)

    # Keep batch padding at the padding value instead of turning padded rows into "all missing".
    if sequence_mask is not None:
        masked_x = masked_x * sequence_mask.unsqueeze(-1).to(dtype=x.dtype)

    return masked_x


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
    def __init__(self, categorical_cardinalities):
        super(StaticEncoder, self).__init__()
        self.categorical_cols = CONFIG['static_categorical_cols']
        self.embedding_dim = CONFIG['static_embedding_dim']
        self.numerical_dim = len(CONFIG['static_numerical_cols'])

        if len(categorical_cardinalities) != len(self.categorical_cols):
            raise ValueError("Length of categorical_cardinalities does not match number of categorical columns.")

        # Create embedding layers for each categorical column
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, self.embedding_dim)
            for num_categories in categorical_cardinalities
        ])

        # Define a linear layer for numerical features
        self.numerical_processor = nn.Sequential(
            nn.Linear(self.numerical_dim, self.embedding_dim),
            nn.ReLU()
        )

        # Define a final linear layer to combine all features
        total_embedding_dim = self.embedding_dim * len(self.categorical_cols) + self.embedding_dim
        self.fc = nn.Linear(total_embedding_dim, CONFIG['static_output_dim'])
        self.relu = nn.ReLU()

    def forward(self, categorical_features, numerical_features):
        # Apply embedding layers
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(categorical_features[:, i]))
        embedded = torch.cat(embedded, dim=1)  # Shape: [batch_size, embedding_dim * num_categorical]

        # Process numerical features
        numerical = self.numerical_processor(numerical_features)  # Shape: [batch_size, numerical_dim]

        # Concatenate all features
        combined = torch.cat([embedded, numerical], dim=1)  # Shape: [batch_size, total_embedding_dim]

        # Pass through fully connected layers
        out = self.fc(combined)
        out = self.relu(out)
        return out


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class NotesEncoder(nn.Module):
    def __init__(self, model_name="thenlper/gte-large"):
        super(NotesEncoder, self).__init__()
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
    # one linear layer is used to project back to feature space
    # in: time series up to timestep t, out: features for t+1
    def __init__(self):
        super(VanillaTimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=len(CONFIG['ts_features']), hidden_size=CONFIG['lstm_hidden_size'], num_layers=CONFIG['lstm_num_layers'], batch_first=True, bidirectional=False)

    def forward(self, x, elapsed_times=None, mask=None):
        # alpsed time and mask are never used here, in signature for consistency
        # x: input in shape (batch size, sequence length, input features)
        lstm_out, (hn, cn) = self.lstm(x)

        # return lstm_out: hidden states for each timestep of last layer (batch_size, sequence length, hidden size)
        # None for attn_weights
        return lstm_out, None

class MultiModalVAE(nn.Module):
    def __init__(self, lstm_encoder, categorical_cardinalities=None, use_static=False, use_notes=False):
        super(MultiModalVAE, self).__init__()
        self.use_static = use_static
        self.use_notes = use_notes
        self.lstm_encoder = lstm_encoder
        
        if use_static:
            assert(categorical_cardinalities is not None)
            self.static_encoder = StaticEncoder(categorical_cardinalities)
            self.static_fusion_layer = nn.Linear(CONFIG['lstm_hidden_size'] + CONFIG['static_output_dim'], CONFIG['lstm_hidden_size'])

        if use_notes:
            self.downproj = nn.Linear(CONFIG['notes_embedding_dim'], CONFIG['lstm_hidden_size'])
            self.cross_attention = nn.MultiheadAttention(embed_dim=CONFIG['lstm_hidden_size'], num_heads=CONFIG['num_heads'], batch_first=True)
            self.layer_norm = nn.LayerNorm(CONFIG['lstm_hidden_size'])
        
        self.mean_encoder = nn.Linear(CONFIG['lstm_hidden_size'], CONFIG['lstm_hidden_size'])
        self.logvar_encoder = nn.Linear(CONFIG['lstm_hidden_size'], CONFIG['lstm_hidden_size'])
        
        self.ff = nn.Linear(CONFIG['lstm_hidden_size'], len(CONFIG['ts_features']))

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, elapsed_times=None, timesteps=None, notes_embeddings=None, notes_timesteps=None, static_features=None, mask=None, notes_mask=None, value_mask=None):
        """
        x: Input sequence of shape (batch_size, seq_length, input_size)
        elapsed_times: Elapsed times between time series data points of shape (batch_size, seq_length)
        timesteps: absolute timesteps for the time series data
        notes_embeddings: embedded notes
        notes_timesteps: corresponding timesteps for each note's text 
        static_features: tuple of categorical and numerical static features
        mask: mask for the padding of sequences in batch. True where real data, false where padded
        notes_mask: mask for padded notes in batch
        value_mask: feature-level observation mask aligned with x. Missing values inside real timesteps are replaced by a sentinel.
        """

        x = inject_missing_value_sentinel(x, value_mask=value_mask, sequence_mask=mask)
        lstm_out, attn_weights = self.lstm_encoder(x, elapsed_times, mask)  # lstm_out: (batch_size, seq_length, lstm_hidden_size)
        
        assert not torch.isnan(lstm_out).any(), "NaN detected in LSTM output"

        static_encoding = None
        if self.use_static:
            assert(static_features is not None)
            static_encoding = self.static_encoder(static_features[0], static_features[1])  # shape (batch_size, static output dim)
            # fuse static encoding into hidden states
            expanded_static = static_encoding.unsqueeze(1).repeat(1, lstm_out.shape[1], 1)  # expand static encoding to shape (bs, seq len, static output dim)
            fused_input = torch.cat([lstm_out, expanded_static], dim=-1)
            fused_output = self.static_fusion_layer(fused_input)  # shape (B, T, hidden_size)

            # Skip connection + activation
            lstm_out = F.relu(fused_output + lstm_out)

            #lstm_out = torch.cat([lstm_out, expanded_static], dim=-1)  # shape (batch_size, seq_length, lstm_hidden_size + static output dim)
            #lstm_out = F.relu(self.static_fusion_layer(lstm_out)) # project back to (batch_size, seq_length, lstm_hidden_size)

        attn_weights_notes = None
        if self.use_notes and notes_embeddings is not None and notes_embeddings.size(1) > 0:
            assert not torch.isnan(notes_embeddings).any(), "NaN detected in notes embeddings"
            # downprojection of notes to match lstm hidden size
            notes_embeddings = F.relu(self.downproj(notes_embeddings))

            padding_notes_mask = None
            if notes_mask is not None:
                # For MultiheadAttention, key_padding_mask is (B, S) with True=ignore
                padding_notes_mask = ~notes_mask.bool()

            attn_mask = None
            has_valid_causal_note = None
            if timesteps is not None and notes_timesteps is not None:
                attn_mask, has_valid_causal_note = build_notes_causal_attention_mask(
                    timesteps=timesteps,
                    notes_timesteps=notes_timesteps,
                    notes_mask=notes_mask,
                    num_heads=self.cross_attention.num_heads,
                )


            # Let the time steps attend to the notes (only up to the same-time notes)
            # let the time series embeddings attend to the texts 
            attn_output, attn_weights_notes = self.cross_attention(
                query=lstm_out,  # hidden states from LSTM for each timestep
                key=notes_embeddings,  # notes_embeddings for different timesteps
                value=notes_embeddings,
                attn_mask=attn_mask,
                key_padding_mask=padding_notes_mask,
                is_causal=False # custom causal cross-attention mask is passed via attn_mask
            )

            if has_valid_causal_note is not None:
                valid_rows = has_valid_causal_note.unsqueeze(-1).to(attn_output.dtype)
                attn_output = attn_output * valid_rows

                if attn_weights_notes is not None:
                    if attn_weights_notes.dim() == 3:
                        attn_weights_notes = attn_weights_notes * has_valid_causal_note.unsqueeze(-1).to(attn_weights_notes.dtype)
                    elif attn_weights_notes.dim() == 4:
                        attn_weights_notes = attn_weights_notes * has_valid_causal_note.unsqueeze(1).unsqueeze(-1).to(attn_weights_notes.dtype)

            lstm_out = self.layer_norm(lstm_out + attn_output)

        # VAE encoding
        mu = self.mean_encoder(lstm_out)
        logvar = self.logvar_encoder(lstm_out)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        outputs = self.ff(z)
        
        # Return outputs and VAE parameters for loss calculation
        return outputs, lstm_out, (attn_weights, attn_weights_notes), static_encoding, mu, logvar



class MultiModal(nn.Module):
    def __init__(self, lstm_encoder, categorical_cardinalities=None, use_static=False, use_notes=False):
        super(MultiModal, self).__init__()
        self.use_static = use_static
        self.use_notes = use_notes

        self.lstm_encoder = lstm_encoder

        if use_static:
            assert(categorical_cardinalities is not None)
            self.static_encoder = StaticEncoder(categorical_cardinalities)
            self.static_fusion_layer = nn.Linear(CONFIG['lstm_hidden_size'] + CONFIG['static_output_dim'], CONFIG['lstm_hidden_size'])

        if use_notes:
            self.downproj = nn.Linear(CONFIG['notes_embedding_dim'], CONFIG['lstm_hidden_size'])
            self.cross_attention = nn.MultiheadAttention(embed_dim=CONFIG['lstm_hidden_size'], num_heads=CONFIG['num_heads'], batch_first=True)
            self.layer_norm = nn.LayerNorm(CONFIG['lstm_hidden_size'])

        self.ff = nn.Linear(CONFIG['lstm_hidden_size'], len(CONFIG['ts_features']))

    def forward(self, x, elapsed_times=None, timesteps=None, notes_embeddings=None, notes_timesteps=None, static_features=None, mask=None, notes_mask=None, value_mask=None):
        """
        x: Input sequence of shape (batch_size, seq_length, input_size)
        elapsed_times: Elapsed times between time series data points of shape (batch_size, seq_length)
        timesteps: absolute timesteps for the time series data
        notes_embeddings: embedded notes
        notes_timesteps: corresponding timesteps for each note's text 
        static_features: tuple of categorical and numerical static features
        mask: mask for the padding of sequences in batch. True where real data, false where padded
        notes_mask: mask for padded notes in batch
        value_mask: feature-level observation mask aligned with x. Missing values inside real timesteps are replaced by a sentinel.
        """

        x = inject_missing_value_sentinel(x, value_mask=value_mask, sequence_mask=mask)
        lstm_out, attn_weights = self.lstm_encoder(x, elapsed_times, mask)  # lstm_out: (batch_size, seq_length, lstm_hidden_size)
        
        assert not torch.isnan(lstm_out).any(), "NaN detected in LSTM output"

        static_encoding = None
        if self.use_static:
            assert(static_features is not None)
            static_encoding = self.static_encoder(static_features[0], static_features[1])  # shape (batch_size, static output dim)
            # fuse static encoding into hidden states
            expanded_static = static_encoding.unsqueeze(1).repeat(1, lstm_out.shape[1], 1)  # expand static encoding to shape (bs, seq len, static output dim)
            fused_input = torch.cat([lstm_out, expanded_static], dim=-1)
            fused_output = self.static_fusion_layer(fused_input)  # shape (B, T, hidden_size)

            # Skip connection + activation
            lstm_out = F.relu(fused_output + lstm_out)

            #lstm_out = torch.cat([lstm_out, expanded_static], dim=-1)  # shape (batch_size, seq_length, lstm_hidden_size + static output dim)
            #lstm_out = F.relu(self.static_fusion_layer(lstm_out)) # project back to (batch_size, seq_length, lstm_hidden_size)

        attn_weights_notes = None
        if self.use_notes and notes_embeddings is not None and notes_embeddings.size(1) > 0:
            assert not torch.isnan(notes_embeddings).any(), "NaN detected in notes embeddings"
            # downprojection of notes to match lstm hidden size
            notes_embeddings = F.relu(self.downproj(notes_embeddings))

            padding_notes_mask = None
            if notes_mask is not None:
                # For MultiheadAttention, key_padding_mask is (B, S) with True=ignore
                padding_notes_mask = ~notes_mask.bool()

                # prevent all-masked rows from producing NaN in softmax
                all_masked = padding_notes_mask.all(dim=1)  # (B,)
                if all_masked.any():
                    # valid_rows mask will later correctly zero out patients who had no real notes
                    padding_notes_mask = padding_notes_mask.clone()
                    padding_notes_mask[all_masked, 0] = False

            attn_mask = None
            has_valid_causal_note = None
            if timesteps is not None and notes_timesteps is not None:
                attn_mask, has_valid_causal_note = build_notes_causal_attention_mask(
                    timesteps=timesteps,
                    notes_timesteps=notes_timesteps,
                    notes_mask=notes_mask,
                    num_heads=self.cross_attention.num_heads,
                )


            # Let the time steps attend to the notes (only up to the same-time notes)
            # let the time series embeddings attend to the texts 
            attn_output, attn_weights_notes = self.cross_attention(
                query=lstm_out,  # hidden states from LSTM for each timestep
                key=notes_embeddings,  # notes_embeddings for different timesteps
                value=notes_embeddings,
                attn_mask=attn_mask,
                key_padding_mask=padding_notes_mask,
                is_causal=False # custom causal cross-attention mask is passed via attn_mask
            )
            assert not torch.isnan(attn_output).any(), "NaN after notes fusion"

            if has_valid_causal_note is not None:
                valid_rows = has_valid_causal_note.unsqueeze(-1).to(attn_output.dtype)
                attn_output = attn_output * valid_rows

                if attn_weights_notes is not None:
                    if attn_weights_notes.dim() == 3:
                        attn_weights_notes = attn_weights_notes * has_valid_causal_note.unsqueeze(-1).to(attn_weights_notes.dtype)
                    elif attn_weights_notes.dim() == 4:
                        attn_weights_notes = attn_weights_notes * has_valid_causal_note.unsqueeze(1).unsqueeze(-1).to(attn_weights_notes.dtype)

            lstm_out = self.layer_norm(lstm_out + attn_output)

        outputs = self.ff(lstm_out)  # (batch_size, seq_length, input_size)

        return outputs, lstm_out, (attn_weights, attn_weights_notes), static_encoding


class TimeAwareAttentionEncoder(nn.Module):
    def __init__(self, use_temporal_attention=True):
        super(TimeAwareAttentionEncoder, self).__init__()
        self.input_size = len(CONFIG['ts_features'])
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

        lstm_out, (hn, cn) = self.lstm(x, elapsed_times)
        batch_size, seq_len, _ = lstm_out.size()

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
            lstm_out = lstm_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

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
        self.lstm = TimeAwareLSTM(input_size=len(CONFIG['ts_features']), hidden_size=CONFIG['lstm_hidden_size'], num_layers=2)
        self.ff = nn.Linear(CONFIG['lstm_hidden_size'], len(CONFIG['ts_features']))

    def forward(self, x, elapsed_times):
        # x: input in shape (batch size, sequence length, input features)
        lstm_out, (hn, cn) = self.lstm(x, elapsed_times)
        # lstm_out: (batch size, sequence length, hidden_size), hidden state for each timestep from last layer
        out = self.ff(lstm_out) # project last hidden states to features
        # out: (batch size, seq len, input features) projection of last hidden states back to features
        return out

class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TimeAwareLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of time-aware LSTM cells
        self.lstm_cells = nn.ModuleList([
            TLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])


    def forward(self, input_seq, elapsed_times, initial_states=None):
        """
        input_seq: Tensor of shape (batch_size, seq_length, input_size)
        elapsed_times: Tensor of shape (batch_size, seq_length)
        initial_states: List of tuples [(h_0, c_0), ..., (h_n, c_n)]
        """
        batch_size, seq_length, _ = input_seq.size()
        input_seq = input_seq
        elapsed_times = elapsed_times

        # Initialize hidden and cell states if not provided
        if initial_states is None:
            h_t = [input_seq.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c_t = [input_seq.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_t, c_t = zip(*[(h, c) for h, c in initial_states])

        outputs = []

        for t in range(seq_length):
            x = input_seq[:, t, :]  # Current input
            delta_t = elapsed_times[:, t]  # Elapsed time since last observation

            for layer in range(self.num_layers):
                h, c = self.lstm_cells[layer](x, (h_t[layer], c_t[layer]), delta_t)
                h_t[layer], c_t[layer] = h, c
                x = h  # Input to the next layer is the hidden state
            outputs.append(h.unsqueeze(1))  # Collect output from the last layer

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_length, hidden_size)
        return outputs, (h_t, c_t)



class TLSTMCell(nn.Module):
    """
      - Decomposition step:
          C_ST = tanh(W_decomp * c_prev + b_decomp)
          C_ST_dis = T * C_ST
          c_prev = c_prev - C_ST + C_ST_dis

      - Then standard LSTM gating:
          i, f, o, g (candidate)
          c_t = f * c_prev + i * g
          h_t = o * tanh(c_t)

      - map_elapse_time(t)
        T = 1 / log(t + e)
        repeated across hidden dim
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

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def map_elapse_time(self, t):
        """
        Maps elapsed time t -> T.
        logic:
            T = 1 / log(t + e)
        t shape: (batch_size, 1)
        returns shape: (batch_size, hidden_size)
        """
        t = torch.nan_to_num(
            t.to(dtype=self.W_decomp.dtype),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )
        t_clamped = torch.clamp(t, min=0.0, max=1e6)
        T = 1.0 / torch.log(t_clamped + 2.7183)
        T = T.repeat(1, self.hidden_size)
        return T

    def forward(self, x_t, states, elapsed_time):
        """
        x_t: (batch_size, input_size) at the current step
        states: (h_prev, c_prev)
                h_prev: (batch_size, hidden_size)
                c_prev: (batch_size, hidden_size)
        elapsed_time: (batch_size,) or (batch_size,1)
                      the elapsed time since the last event

        returns (h_t, c_t)
        """
        h_prev, c_prev = states

        # Ensure elapsed_time is shape (B,1)
        if elapsed_time.dim() == 1:
            elapsed_time = elapsed_time.unsqueeze(1)

        # 1) Map elapsed time
        T = self.map_elapse_time(elapsed_time)  # (B, hidden_size)

        # 2) Decompose c_prev
        #    C_ST = tanh(c_prev * W_decomp + b_decomp)
        C_ST = torch.tanh(c_prev @ self.W_decomp + self.b_decomp)  # (B, hidden_size)
        C_ST_dis = T * C_ST  # (B, hidden_size)

        # c_prev_new = c_prev - C_ST + C_ST_dis
        c_prev_new = c_prev - C_ST + C_ST_dis

        # 3) Standard LSTM gates
        #    i, f, o, g (candidate)
        i_t = torch.sigmoid(x_t @ self.Wi + h_prev @ self.Ui + self.bi)
        f_t = torch.sigmoid(x_t @ self.Wf + h_prev @ self.Uf + self.bf)
        o_t = torch.sigmoid(x_t @ self.Wo + h_prev @ self.Uo + self.bo)
        g_t = torch.tanh(x_t @ self.Wc + h_prev @ self.Uc + self.bc)

        # 4) Update cell
        c_t = f_t * c_prev_new + i_t * g_t

        # 5) Update hidden
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


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
