## Temporal Fusion Nexus: A task-agnostic multi-modal embedding model for clinical narratives and irregular time series in post-kidney transplant care.

Summary: Multimodal modeling for kidney transplant patients on the NephroCAGE cohort. The code fuses irregular time-series vitals, static donor/recipient attributes, and free-text clinical notes into a shared embedding (the "Nexus"), which downstream heads use to predict graft loss, rejection, and mortality across multiple horizons.

## Repository Map

Library code (`src/`):
- `config.py`: feature lists, model dimensions, loss weights, optimiser settings, and the clinical-note encoder id.
- `preprocessing.py`: load raw NephroCAGE files, clean/merge static tables, build time series (vitals, labs, meds), attach note embeddings, and assemble the PyTorch `NephroCAGEDataset` plus `collate_fn`. Preprocessing artifacts (label encoders, scalers) are fitted on training patients only and reused for val/test.
- `models.py`: time-aware LSTM (TM-LSTM) encoder with a learnable per-unit time gate, causal temporal self-attention, variable-wise static gating, causal cross-attention over clinical notes, the learnable Nexus transform, and a TM-LSTM decoder used during training. Also `SimpleMLP` for lightweight classification heads.
- `utils.py`: loss terms (multi-step reconstruction, decorrelation, disentanglement) and representation diagnostics (MIG, SAP, DCI).

Command-line entry points (`src/`):
- `embed_notes.py`: generate clinical-note embeddings; supports sharding across GPUs.
- `train.py`: train the encoder with cross-validation and early stopping.
- `evaluate.py`: extract Nexus representations and score the downstream tasks.
- `evaluate_paper_protocol.py`: downstream evaluation following the notebook protocol, with configurable probes.
- `disentanglement.py`: MIG / SAP / DCI on trained checkpoints, each against a permutation null.
- `sweep_window.py`: sensitivity of downstream performance to the sampling window, plus a raw-feature baseline.

Notebooks (`notebooks/`) — see `notebooks/README.md` for the execution order:
- `preprocessing/`: per-modality preprocessing.
- `generate_embeddings/`: note-embedding generation and text-encoder fine-tuning.
- `dataset_pool_assignment.ipynb`: builds the persisted splits in `data/splits/`.
- `training.ipynb`, `classification.ipynb`: backbone training and classification heads.
- `results.ipynb`: plotting from stored result files.
- `misc/`: calibration, clustering, interpretation, visualisations, the VAE variant, and the user study.

Other:
- `data/splits/`: persisted patient-level splits so runs are comparable.
- `data/results/`: stored metrics and plot inputs; raw data are not included.
- `docs/paper_alignment.md`: a record of how the code corresponds to the manuscript, including the points where the manuscript underdetermines an implementation choice.

## Setup
1. Python 3.10 recommended.
2. Create an environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. The note encoder downloads `MedAI-HS/med-gte-hybrid-de` from Hugging Face; ensure access or pre-cache via `HF_HOME`.
4. A GPU is recommended for training and for note embedding.

## Data
NephroCAGE is not publicly available. We used NephroCAGE v1 raw files under `data/v1/` with the filenames:
- 1_BAseline_parameter_fertig2_HLA_Pirche_final.xlsx
- 2_donoparameter_final.xlsx
- 4_exams.csv
- 5 Biopsy_patho_kreuz_extensive.xlsx
- 6 Lab_cohort.csv
- 7_clinical_assessment.csv
- 8_Medikation.csv
- 9_Hospitalization.xlsx
- 10_HLA-DSA_timecourse.xlsx

## Preprocessing Workflow
- Use the notebooks in `notebooks/preprocessing/` for exploratory runs, or call helpers in `src/preprocessing.py`:
  - `get_dfs(project_path)` loads all raw tables.
  - `create_static_df`, `create_vitals_df`, `create_medication_df`, `create_notes_df` clean each modality.
  - `create_ts_data` merges vitals/labs/meds, computes eGFR, and aligns timelines.
  - `get_valid_patient_ids` and `split_patient_ids` build patient-level splits. `get_or_create_global_split` persists one split to disk so it is stable across notebooks; `create_dataset_splits` also accepts `patient_split_ids` when a caller supplies splits directly (used by the training and evaluation scripts).
  - `NephroCAGEDataset` packages static, time-series and note features with masks and reuses train-fitted preprocessing artifacts; `collate_fn` pads variable-length batches.
- `CONFIG` in `src/config.py` lists the feature sets and model dimensions.

## Modeling Overview
- **Time-series backbone**: `TimeAwareLSTM`, whose cell decomposes memory into short- and long-term components and decays the short-term part by a learnable, per-unit function of the elapsed time since the previous observation. `TimeAwareAttentionEncoder` adds causal temporal self-attention on top.
- **Missingness**: a feature-level binary mask marks which values were observed. Missing entries contribute nothing to the gate pre-activations, and the mask is supplied to the network so that missingness itself is informative. No imputation is performed.
- **Static features**: `StaticEncoder` embeds each static variable separately and learns per-variable selection weights; the result is admitted into the hidden state through a gate at every time step.
- **Clinical notes**: `NotesEncoder` embeds notes, and cross-attention lets each time step attend to notes, masked so that no note written after the current time is visible.
- **Nexus**: a learnable transformation of the fused representation. This embedding is what the downstream heads consume.
- **Decoder**: a TM-LSTM decoder, used during training only, reconstructs the next `prediction_horizon` observations at their actual observation times.
- **Objective**: reconstruction loss plus decorrelation and disentanglement terms, weighted by `lambda_decorr`, `lambda_disent` and `alpha_disent` in `CONFIG`.

## Training and Evaluation

Scripts (headless, loggable):
```bash
# 1. clinical-note embeddings
python src/embed_notes.py --project-root . --out data/embeddings/emb.npy

# 2. train the encoder
python src/train.py --project-root . --embeddings data/embeddings/emb.npy --out-dir runs/example

# 3. downstream evaluation
python src/evaluate_paper_protocol.py --project-root . --embeddings data/embeddings/emb.npy --run-dir runs/example

# 4. representation diagnostics
python src/disentanglement.py --project-root . --embeddings data/embeddings/emb.npy --runs runs/example
```

Notebooks: `training.ipynb` produces the backbone and `classification.ipynb` the heads. See `notebooks/README.md` for the intended order and for how splits and checkpointing are handled.

Both routes share `src/`, so a change to the model or preprocessing affects them identically.

## Notes on Evaluation
Downstream performance depends on how observation times are sampled for scoring — `min_history_days`, `max_days`, `max_samples_per_patient`, and which eligible timesteps are kept. These define the prediction task rather than the model, so they should be fixed in advance and reported. `src/sweep_window.py` reports performance across a grid of these settings, together with a raw-feature baseline. Event counts should be reported alongside any metric, since they govern how precisely it can be estimated.

## Repro Tips
- Cache Hugging Face models before running note-heavy steps.
- Run preprocessing first so downstream steps can load cleaned arrays and embeddings.
- Reuse a persisted split from `data/splits/` when comparing runs.
- When adding features, update `CONFIG` and preprocessing together so shapes stay aligned across the dataset, encoders, and heads.
