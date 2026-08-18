CONFIG = {
    'static_categorical_cols': ['gender', 'underlying_disease', 'blood_group', 'gender_donor', 'donor_bloodgroup', 'type_of_donation'],

    'static_numerical_cols': ['age', 'number_dialyses', 'cold_ischemia_time', 'age_donor', 'pirche_score', 'mma_broad', 'mmb_broad', 'mmdr_broad','mm_broad'],

    'ts_features' : ['bp_sys', 'bp_dia', 'weight', 'urine_volume', 'hr', 'temperature', 'diuresis_time',
                     'creatinine', 'leukocyte', 'proteinuria', 'crphp', 'egfr', 'acr', 'Tacrolimus', 'Methylprednisolon', 'Ciclosporin'],

    'static_embedding_dim': 16, # embedding size of each feature, categorical or numerical
    'static_output_dim': 256, # final static embedding dimension after last ff layer
    'lstm_hidden_size': 512, # hidden size in lstms
    'lstm_num_layers': 2,
    'decoder_num_layers': 2, # TM-LSTM decoder, used during training only (Supplement B)
    'num_heads': 4, # both temporal self-attention and notes cross-attention (Supplement B)
    'PADDING_VAL': 0,
    'notes_embedding_dim': 1024, # embedding dim of clinical notes

    # --- Reconstruction objective (paper Eq. 7) ---
    'prediction_horizon': 10, # H, number of future steps averaged in L_recon

    # --- Loss weights (Supplement B) ---
    'lambda_decorr': 0.4,
    'lambda_disent': 0.2,
    'alpha_disent': 0.1, # L1 weight on the disentanglement soft masks

    # --- Optimisation (Supplement B) ---
    'learning_rate': 5e-4,
    'batch_size': 32,

    # --- TM-LSTM time gate (paper Eq. 3) ---
    # Per-unit decay rates are initialised so that half-lives are spread
    # log-uniformly across this range (in days), giving the 512 units genuinely
    # different timescales instead of collapsing onto one.
    'decay_half_life_min_days': 1.0,
    'decay_half_life_max_days': 365.0,

    # --- Text encoder ---
    # The paper's Med-GTE-hybrid-de. Note that finetune_gte.ipynb in this repo
    # only implements the contrastive stage; the released model below is the
    # dual-stage (contrastive + denoising) artifact the paper reports.
    'notes_encoder_model': 'MedAI-HS/med-gte-hybrid-de',
}
