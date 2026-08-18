import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import CONFIG

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def compute_mutual_information_appr(x, y):
    """
    Compute approximation of mutual information between two representations.
    Args:
        x (torch.Tensor): First representation of shape (B, D1)
        y (torch.Tensor): Second representation of shape (B, D2)
    Returns:
        torch.Tensor: MI estimate of shape (D1, D2)
    """
    if x.shape[0] < 2:
        return torch.zeros((x.shape[1], y.shape[1]), device=x.device, dtype=x.dtype)

    # Normalize inputs
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    
    # Compute normalized correlation as MI approximation
    # (B, D1) and (B, D2) -> (D1, D2)
    mi = torch.abs((x.T @ y) / (x.shape[0] - 1))
    return mi


def get_last_valid_step(sequence, mask=None):
    """
    Gather the last non-padded timestep for each sequence in the batch.

    Args:
        sequence (torch.Tensor): Tensor of shape (B, T, D).
        mask (torch.Tensor | None): Boolean-like tensor of shape (B, T) where
            True marks real timesteps. If None, the last batch position is used.
    """
    if sequence.dim() != 3:
        raise ValueError(f"sequence must have shape (B, T, D), got {sequence.shape}")

    if mask is None:
        return sequence[:, -1, :]

    if mask.shape != sequence.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match sequence shape {sequence.shape[:2]}")

    lengths = mask.long().sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("Each sequence must contain at least one valid timestep")

    last_indices = lengths - 1
    batch_indices = torch.arange(sequence.size(0), device=sequence.device)
    return sequence[batch_indices, last_indices]


def build_elapsed_times(timesteps, mask=None):
    """
    Build elapsed times aligned to each timestep.

    The first valid timestep has elapsed time 0. Subsequent valid timesteps get
    the gap from the previous timestep. Padded positions are zeroed.
    """
    if timesteps.dim() != 2:
        raise ValueError(f"timesteps must have shape (B, T), got {timesteps.shape}")

    elapsed = torch.zeros_like(timesteps, dtype=torch.float32)
    if timesteps.size(1) > 1:
        elapsed[:, 1:] = timesteps[:, 1:].float() - timesteps[:, :-1].float()

    elapsed = torch.nan_to_num(elapsed, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)

    if mask is not None:
        if mask.shape != timesteps.shape:
            raise ValueError(f"mask shape {mask.shape} does not match timesteps shape {timesteps.shape}")
        elapsed = elapsed.masked_fill(~mask.bool(), 0.0)

    return elapsed


def masked_mean_over_time(sequence, mask):
    """
    Compute the mean over valid timesteps only.
    """
    if sequence.dim() != 3:
        raise ValueError(f"sequence must have shape (B, T, D), got {sequence.shape}")
    if mask is None:
        return sequence.mean(dim=1)
    if mask.shape != sequence.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match sequence shape {sequence.shape[:2]}")

    weights = mask.unsqueeze(-1).to(sequence.dtype)
    counts = weights.sum(dim=1).clamp(min=1)
    return (sequence * weights).sum(dim=1) / counts


def get_last_valid_note_embedding(notes_embeddings, notes_mask=None):
    """
    Gather the last real note embedding per sample. Samples with no notes return
    an all-zero embedding so downstream auxiliary losses remain well-defined.
    """
    if notes_embeddings.dim() != 3:
        raise ValueError(
            f"notes_embeddings must have shape (B, S, D), got {notes_embeddings.shape}"
        )

    batch_size, num_notes, note_dim = notes_embeddings.shape
    if num_notes == 0:
        return torch.zeros((batch_size, note_dim), device=notes_embeddings.device, dtype=notes_embeddings.dtype)

    if notes_mask is None:
        return notes_embeddings[:, -1, :]

    if notes_mask.shape != notes_embeddings.shape[:2]:
        raise ValueError(
            f"notes_mask shape {notes_mask.shape} does not match note shape {notes_embeddings.shape[:2]}"
        )

    lengths = notes_mask.long().sum(dim=1)
    safe_indices = lengths.clamp(min=1) - 1
    batch_indices = torch.arange(batch_size, device=notes_embeddings.device)
    last_notes = notes_embeddings[batch_indices, safe_indices]

    no_notes = lengths == 0
    if no_notes.any():
        last_notes = last_notes.clone()
        last_notes[no_notes] = 0

    return last_notes

def modality_mi_loss(lstm_states, static_emb, notes_emb):
    B, D = lstm_states.shape
    if B < 2:
        return torch.zeros((), device=lstm_states.device, dtype=lstm_states.dtype)
    
    # Compute MI between each lstm dimension and each modality
    mi_static = compute_mutual_information_appr(lstm_states, static_emb)
    mi_notes = compute_mutual_information_appr(lstm_states, notes_emb)
    
    eps = 1e-8
    total_mi = mi_static.sum(dim=1) + mi_notes.sum(dim=1) + eps
    
    # Compute concentration scores
    static_concentration = (mi_static.sum(dim=1) / total_mi) ** 2
    notes_concentration = (mi_notes.sum(dim=1) / total_mi) ** 2
    
    # Remove negative sign - we want to maximize concentration
    concentration_loss = (static_concentration + notes_concentration).mean()
    
    # Subtract from 1 to convert to a loss (1 = max loss, 0 = perfect specialization)
    return 1.0 - concentration_loss

def compute_mutual_information(X, y):
    """
    Compute mutual information between features X and target y.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    Returns:
        mi_scores: numpy array of shape (n_features,)
    """
    return mutual_info_regression(X, y)

def _infer_is_categorical(n_factors):
    """
    Factors are assembled as static categorical, then static numerical, then
    time-series, so the leading len(static_categorical_cols) are categorical.
    """
    n_cat = len(CONFIG['static_categorical_cols'])
    if n_factors >= n_cat:
        return [True] * n_cat + [False] * (n_factors - n_cat)
    return [False] * n_factors


def _discretize(x, bins=20):
    edges = np.histogram_bin_edges(x, bins=bins)
    return np.digitize(x, edges[1:-1])


def _entropy(codes):
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _eta_squared(z, codes):
    """Correlation ratio: between-group variance over total variance."""
    total_var = z.var()
    if total_var <= 1e-12:
        return 0.0
    grand = z.mean()
    num = 0.0
    for c in np.unique(codes):
        g = z[codes == c]
        if g.size:
            num += g.size * (g.mean() - grand) ** 2
    return float(num / (z.size * total_var))


def compute_mig(representations, factors, is_cat=None, bins=20):
    """
    MIG (Chen et al. 2018): for each factor, the normalised gap between the
    highest and second-highest mutual information across single latent
    dimensions, averaged over factors.

    The gap is the point: a disentangled representation should carry a factor
    in one dimension, not spread across many.
    """
    from sklearn.metrics import mutual_info_score

    Z = np.asarray(representations)
    F = np.asarray(factors)
    if is_cat is None:
        is_cat = _infer_is_categorical(F.shape[1])

    Zd = np.stack([_discretize(Z[:, j], bins) for j in range(Z.shape[1])], axis=1)
    per_factor = []
    for k in range(F.shape[1]):
        fk = F[:, k].astype(int) if is_cat[k] else _discretize(F[:, k], bins)
        H = _entropy(fk)
        if H <= 1e-8:
            continue
        mi = np.array([mutual_info_score(Zd[:, j], fk) for j in range(Z.shape[1])])
        top = np.sort(mi)[::-1]
        per_factor.append((top[0] - top[1]) / H)
    if not per_factor:
        return float('nan'), np.array([])
    return float(np.mean(per_factor)), np.array(per_factor)


def compute_sap(representations, factors, is_cat=None):
    """
    SAP (Kumar et al. 2018): for each factor, the gap between the best and
    second-best SINGLE latent dimension at predicting it, averaged over factors.

    Numerical factors are scored by the R^2 of a one-predictor regression
    (equivalently the squared correlation); categorical factors by the
    correlation ratio eta^2, its categorical analogue.
    """
    Z = np.asarray(representations)
    F = np.asarray(factors)
    if is_cat is None:
        is_cat = _infer_is_categorical(F.shape[1])

    Zc = Z - Z.mean(axis=0, keepdims=True)
    Zs = Zc.std(axis=0) + 1e-12
    per_factor = []
    for k in range(F.shape[1]):
        if is_cat[k]:
            codes = F[:, k].astype(int)
            scores = np.array([_eta_squared(Z[:, j], codes) for j in range(Z.shape[1])])
        else:
            f = F[:, k] - F[:, k].mean()
            corr = (Zc * f[:, None]).mean(axis=0) / (Zs * (f.std() + 1e-12))
            scores = corr ** 2
        top = np.sort(scores)[::-1]
        per_factor.append(float(top[0] - top[1]))
    if not per_factor:
        return float('nan'), np.array([])
    return float(np.mean(per_factor)), np.array(per_factor)


def compute_dci(representations, factors, is_cat=None, n_estimators=50, max_depth=10,
                random_state=42, test_size=0.2):
    """
    DCI (Eastwood & Williams 2018): Disentanglement, Completeness, Informativeness.

    Built from an importance matrix R, where R[j, i] is the importance of latent
    dimension i for predicting factor j, taken from random-forest feature
    importances.

      D = sum_i rho_i * (1 - H_{n_factors}(P_i)),  P_i = R[:, i] / sum_j R[j, i]
      C = mean_j (1 - H_{n_latents}(P~_j)),        P~_j = R[j, :] / sum_i R[j, i]
      I = mean_j R^2 (or accuracy) of predicting factor j from all latents

    D asks whether each dimension captures one factor; C asks whether each
    factor is captured by one dimension. Both are gaps in disguise, which is
    what makes them meaningful where compute_sap_lenient is not.

    This mirrors the implementation in notebooks/interpret.ipynb, with one
    correction: categorical factors are handled with a classifier rather than a
    regressor on the raw label codes.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    Z = np.asarray(representations)
    F = np.asarray(factors)
    if is_cat is None:
        is_cat = _infer_is_categorical(F.shape[1])

    n_factors, n_latents = F.shape[1], Z.shape[1]
    R = np.zeros((n_factors, n_latents))
    scores = []

    for j in range(n_factors):
        y = F[:, j]
        if is_cat[j]:
            y = y.astype(int)
            if len(np.unique(y)) < 2:
                continue
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          random_state=random_state)
        try:
            Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=test_size,
                                                  random_state=random_state)
        except ValueError:
            continue
        model.fit(Xtr, ytr)
        R[j] = model.feature_importances_
        scores.append(float(model.score(Xte, yte)))

    total = R.sum()
    if total <= 0:
        return {'disentanglement': float('nan'), 'completeness': float('nan'),
                'informativeness': float('nan')}

    # Disentanglement: per-dimension importance entropy, weighted by dimension use
    rho = R.sum(axis=0) / total
    D_i = np.zeros(n_latents)
    for i in range(n_latents):
        col = R[:, i]
        s = col.sum()
        if s <= 0:
            continue
        p = col / s
        H = -(p * np.log(p + 1e-10)).sum() / np.log(n_factors)
        D_i[i] = 1.0 - H
    D = float((rho * D_i).sum())

    # Completeness: per-factor importance entropy across dimensions
    C_j = []
    for j in range(n_factors):
        row = R[j]
        s = row.sum()
        if s <= 0:
            continue
        p = row / s
        H = -(p * np.log(p + 1e-10)).sum() / np.log(n_latents)
        C_j.append(1.0 - H)
    C = float(np.mean(C_j)) if C_j else float('nan')

    I = float(np.mean(scores)) if scores else float('nan')
    return {'disentanglement': D, 'completeness': C, 'informativeness': I}


def compute_mig_lenient(representations, factors, group_size=3, overlap=1):
    """
    DEPRECATED -- kept only to reproduce previously published numbers.

    Scores overlapping GROUPS of dimensions rather than the standard top-1 vs
    top-2 single-dimension gap, which makes it substantially more permissive.
    Use compute_mig for the standard definition.

    Args:
        representations: numpy array of shape (n_samples, n_latent_dims)
        factors: numpy array of shape (n_samples, n_factors)
        group_size: size of dimension groups to consider
        overlap: number of dimensions that can overlap between groups
    """
    n_samples, n_latent = representations.shape
    n_factors = factors.shape[1]
    
    scaler = StandardScaler()
    representations_norm = scaler.fit_transform(representations)
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((n_factors, n_latent))
    for f in range(n_factors):
        mi_matrix[f] = compute_mutual_information(representations_norm, factors[:, f])
    
    factor_mig_scores = []
    for f in range(n_factors):
        sorted_indices = np.argsort(mi_matrix[f])[::-1]
        
        # Create overlapping groups
        groups = []
        for i in range(0, n_latent - group_size + 1, group_size - overlap):
            group = sorted_indices[i:i + group_size]
            if len(group) == group_size:
                groups.append(group)
        
        if not groups:
            factor_mig_scores.append(0)
            continue
            
        # Compute MI for each group
        group_mis = []
        for group in groups:
            group_mi = np.sum(mi_matrix[f][group])
            group_mis.append(group_mi)
        
        # Compare best group with second best
        if len(group_mis) > 1:
            sorted_group_mis = np.sort(group_mis)[::-1]
            gap = sorted_group_mis[0] - sorted_group_mis[1]
            normalized_score = gap / sorted_group_mis[0] if sorted_group_mis[0] > 0 else 0
        else:
            normalized_score = 1.0  # Only one group
            
        factor_mig_scores.append(normalized_score)
    
    mig_score = np.mean(factor_mig_scores)
    return mig_score, np.array(factor_mig_scores)

def compute_sap_lenient(representations, factors, top_k=5, threshold=0.1):
    """
    DEPRECATED -- kept only to reproduce previously published numbers.

    This does NOT measure disentanglement. It reports the absolute R^2 of the
    best top_k dimensions fitted jointly, i.e. how PREDICTABLE a factor is,
    rather than the standard best-vs-second-best gap, which is what separation
    means. A fully entangled representation, where every dimension encodes
    every factor, scores high here. Use compute_sap instead.

    Original note from the authors:
    "More lenient SAP calculation that's easier for models to achieve good
    scores. 1. Uses absolute R² values instead of gaps 2. Applies softer
    thresholding 3. More generous normalization"
    """
    n_samples, n_latent = representations.shape
    n_factors = factors.shape[1]
    
    scaler_rep = StandardScaler()
    scaler_fac = StandardScaler()
    representations_norm = scaler_rep.fit_transform(representations)
    factors_norm = scaler_fac.fit_transform(factors)
    
    factor_sap_scores = []
    for f in range(n_factors):
        # Get individual R² scores
        r2_scores = np.zeros(n_latent)
        for l in range(n_latent):
            # Simple linear regression for each dimension
            reg = LinearRegression()
            reg.fit(representations_norm[:, [l]], factors_norm[:, f])
            r2_scores[l] = r2_score(factors_norm[:, f], reg.predict(representations_norm[:, [l]]))
        
        # Get top-k dimensions
        top_k_indices = np.argsort(r2_scores)[::-1][:top_k]
        X_top = representations_norm[:, top_k_indices]
        
        # Fit model on top-k dimensions together
        reg_top = LinearRegression()
        reg_top.fit(X_top, factors_norm[:, f])
        top_r2 = r2_score(factors_norm[:, f], reg_top.predict(X_top))
        
        # Instead of comparing with next-k, just use the absolute R² value
        # Apply soft thresholding to make it easier to get higher scores
        score = max(0, (top_r2 - threshold) / (1 - threshold))
        factor_sap_scores.append(score)
    
    sap_score = np.mean(factor_sap_scores)
    return sap_score, np.array(factor_sap_scores)

def correlation_loss(z, reduction='sum'):
    """
    L_decorr (paper Eq. 8): penalise off-diagonal covariance of the embedding.

        L_decorr = sum_{i != j} Cov(Z)_{i,j}^2

    Args:
        z: representations of shape (N, D).
        reduction: how the D(D-1) off-diagonal terms are aggregated.
            'sum'  -- Eq. 8 exactly as printed.
            'dim'  -- divided by D.
            'mean' -- divided by D(D-1).

    Note on scale: with D = 512 the sum runs over 261,632 terms, so 'sum' is
    orders of magnitude larger than L_recon and, at the paper's lambda_decorr
    of 0.4, dominates the objective. The reduction is therefore exposed rather
    than fixed, since Eq. 8 does not say which one produced the reported
    results. See docs/paper_alignment.md.
    """
    N, D = z.shape
    if N < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    # Normalize to zero mean
    z = z - z.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov_matrix = (z.T @ z) / (N - 1)

    # Remove diagonal (force decorrelation)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    total = (off_diag ** 2).sum()

    if reduction == 'sum':
        return total
    if reduction == 'dim':
        return total / D
    if reduction == 'mean':
        return total / (D * (D - 1))
    raise ValueError(f"unknown reduction: {reduction}")

def balanced_correlation_loss(z, beta=0.5):
    """
    Enhanced decorrelation loss that balances between orthogonality and activity.
    Higher beta values (0-1) favor more distributed activations across dimensions.
    
    Args:
        z (torch.Tensor): Hidden state representations of shape (B, D).
        beta (float): Balance factor between decorrelation and activity distribution.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, D = z.shape
    if B < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    # Normalize to zero mean
    z = z - z.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov_matrix = (z.T @ z) / (B - 1)
    
    # Standard decorrelation loss: penalize off-diagonal elements
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    decorr_loss = (off_diag ** 2).sum()
    
    # Activity distribution loss: prevent a few dimensions from dominating
    diag_elements = torch.diag(cov_matrix)
    
    # Normalize diagonal elements (dimension variances)
    normalized_activities = diag_elements / (diag_elements.sum() + 1e-8)
    
    # Compute negative entropy to encourage uniform usage of dimensions
    # Higher entropy = more uniform distribution of activations
    activity_loss = (normalized_activities * torch.log(normalized_activities + 1e-8)).sum()
    
    # Combine both losses with the beta parameter
    return (1 - beta) * decorr_loss + beta * activity_loss


class FeatureDecoders(nn.Module):
    """
    Per-factor decoders behind L_disent (paper Eq. 9).

    Each generative factor f_d gets its own sparse soft mask M_d over the Nexus
    dimensions and its own small decoder, so that predicting f_d can only draw
    on the dimensions its mask selects. The L1 penalty on the masks is what
    pushes different factors onto different dimensions.
    """

    def __init__(self, hidden_dim, config, categorical_cardinalities=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_masks = len(config['static_categorical_cols']) + \
                        len(config['static_numerical_cols']) + \
                        len(config['ts_features'])

        self.dimension_masks = nn.Parameter(
            torch.randn(self.num_masks, hidden_dim),
            requires_grad=True
        )

        # Categorical factors are multi-class (blood group, underlying disease,
        # type of donation, ...), so each decoder emits one logit per class and
        # is scored with cross-entropy. Scoring them against the raw integer
        # label -- as a single logit would require -- would impose a spurious
        # ordinal structure on unordered categories.
        if categorical_cardinalities is None:
            raise ValueError(
                "categorical_cardinalities is required so categorical factors can be "
                "decoded as multi-class targets"
            )
        if len(categorical_cardinalities) != len(config['static_categorical_cols']):
            raise ValueError("Length of categorical_cardinalities does not match number of categorical columns.")

        self.categorical_cardinalities = list(categorical_cardinalities)
        self.cat_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, int(cardinality))
            )
            for cardinality in self.categorical_cardinalities
        ])

        self.num_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in config['static_numerical_cols']
        ])
        
        self.ts_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in config['ts_features']
        ])

    def forward(self, hidden_states):
        # hidden_states: (B, hidden_dim)
        masks = torch.sigmoid(self.dimension_masks)  # (num_masks, hidden_dim)
        
        predictions = {
            'categorical': [],
            'numerical': [],
            'ts': []
        }
        
        start_idx = 0
        
        # Each prediction will be (B, 1)
        for i, decoder in enumerate(self.cat_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['categorical'].append(decoder(masked_hidden))
            
        start_idx += len(self.cat_decoders)
        
        for i, decoder in enumerate(self.num_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['numerical'].append(decoder(masked_hidden))
            
        start_idx += len(self.num_decoders)
        
        for i, decoder in enumerate(self.ts_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['ts'].append(decoder(masked_hidden))
        
        return predictions, masks

def feature_prediction_loss(predictions, masks, cat_features, num_features, ts_features, alpha=None):
    """
    L_disent (paper Eq. 9): mean factor-prediction error plus an L1 penalty on
    the soft masks.

        L_disent = (1/D) sum_d (f_d - f_hat_d(Z))^2 + alpha * ||M_d||_1

    Categorical factors use cross-entropy rather than squared error; see the
    note in FeatureDecoders.
    """
    if alpha is None:
        alpha = CONFIG['alpha_disent']

    criterion_num = nn.MSELoss()

    loss = 0.0
    num_factors = 0

    # Categorical factors (multi-class cross-entropy)
    for i, pred in enumerate(predictions['categorical']):
        target = cat_features[:, i].long()
        loss = loss + F.cross_entropy(pred, target)
        num_factors += 1

    # Numerical static factors
    for i, pred in enumerate(predictions['numerical']):
        target = num_features[:, i].unsqueeze(1)  # (B, 1)
        loss = loss + criterion_num(pred, target)
        num_factors += 1

    # Time series factors
    for i, pred in enumerate(predictions['ts']):
        target = ts_features[:, i].unsqueeze(1)  # (B, 1)
        loss = loss + criterion_num(pred, target)
        num_factors += 1

    loss = loss / max(num_factors, 1)

    # L1 sparsity on the soft masks
    sparsity_loss = torch.mean(torch.sum(masks, dim=1) / masks.shape[1])

    return loss + alpha * sparsity_loss


def build_future_windows(timesteps, ts_values, value_mask, step_mask, horizon):
    """
    Gather the next `horizon` observations after every query timestep.

    Returns:
        future_elapsed: (B, T, H) gap between each future step and the one before
            it (the first is measured from the query timestep itself).
        future_values: (B, T, H, F) target values.
        target_mask: (B, T, H, F) 1 where the target is a real, observed value
            that also falls inside a valid (non-padded) timestep.
    """
    B, T, n_features = ts_values.shape
    H = horizon
    dev = ts_values.device

    future_values = ts_values.new_zeros(B, T, H, n_features)
    future_obs = value_mask.new_zeros(B, T, H, n_features)
    future_times = timesteps.new_zeros(B, T, H)
    future_step_valid = torch.zeros(B, T, H, dtype=torch.bool, device=dev)

    for h in range(1, H + 1):
        if h >= T:
            break
        future_values[:, :T - h, h - 1] = ts_values[:, h:]
        future_obs[:, :T - h, h - 1] = value_mask[:, h:]
        future_times[:, :T - h, h - 1] = timesteps[:, h:]
        if step_mask is not None:
            future_step_valid[:, :T - h, h - 1] = step_mask[:, h:].bool()
        else:
            future_step_valid[:, :T - h, h - 1] = True

    # Gaps between consecutive future observation times.
    prev_times = torch.cat([timesteps.unsqueeze(-1), future_times[:, :, :-1]], dim=-1)
    future_elapsed = (future_times - prev_times).to(dtype=ts_values.dtype)
    future_elapsed = torch.nan_to_num(future_elapsed, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
    future_elapsed = future_elapsed.masked_fill(~future_step_valid, 0.0)

    target_mask = future_obs.to(dtype=ts_values.dtype) * future_step_valid.unsqueeze(-1).to(dtype=ts_values.dtype)
    if step_mask is not None:
        # Only decode from query timesteps that are themselves real.
        target_mask = target_mask * step_mask.bool().view(B, T, 1, 1).to(dtype=ts_values.dtype)

    return future_elapsed, future_values, target_mask


def multistep_reconstruction_loss(predictions, targets, target_mask):
    """
    L_recon (paper Eq. 7): mean squared error over the H predicted steps,
    averaged over observed target entries only.

    predictions / targets / target_mask: (B, T, H, F)
    """
    squared_error = (predictions - targets) ** 2
    masked = squared_error * target_mask
    denom = target_mask.sum()
    if denom.item() == 0:
        return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    return masked.sum() / denom


def flatten_valid_timesteps(sequence, step_mask):
    """
    Collapse (B, T, D) to (N, D) keeping only real timesteps. Used to apply the
    decorrelation loss across the Nexus embedding rather than a single timestep.
    """
    if step_mask is None:
        return sequence.reshape(-1, sequence.shape[-1])
    return sequence[step_mask.bool()]
