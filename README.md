# HuBERT Temporal Relevance Explanations for Speech Emotion Recognition

This repository contains a command-line research pipeline for explaining a
HuBERT speech emotion classifier with temporal relevance maps. The current
pipeline focuses on lightweight relevance-aware variants built on top of
gradient-weighted attention rollout, plus quantitative checks for sparsity,
deletion faithfulness, and class-specificity.

The main checkpoint used by the current scripts is:

```text
superb/hubert-base-superb-er
```

The model has four output classes:

```text
0 = neu
1 = hap
2 = ang
3 = sad
```

## What is implemented

The supported explanation modes are:

| Mode | Meaning |
| --- | --- |
| `rollout` | Gradient-weighted attention rollout through HuBERT attention maps. |
| `level3` | Rollout multiplied by lightweight head/pooling relevance for the target class. |
| `level3-contrastive` | Rollout multiplied by target-vs-contrastive head relevance. |
| `head-conditioned-rollout` | Uses head relevance as the rollout seed. |
| `contrastive-conditioned-rollout` | Uses contrastive head relevance as the rollout seed. |

Important conceptual note: this is **not** full Transformer LRP. The project
does not propagate relevance through every HuBERT Transformer block, GELU,
LayerNorm, residual branch, or attention internals. The current “Level 3”
implementation is a lightweight approximation:

```text
target class logit
<- classification head
<- temporal mean pooling
<- final HuBERT time tokens
```

The motivation is:

```text
attention rollout = temporal/token diffusion inside HuBERT
head/pooling relevance = which final HuBERT tokens support the class logit
combined score = relevance-aware temporal explanation
```

## Repository layout

```text
src/explainers/transformer_relevance/
  attention_extractor.py          # extracts HuBERT attentions and final hidden states
  gradient_attention.py           # computes gradients for target class explanations
  head_relevance.py               # lightweight head/pooling and contrastive relevance
  rollout.py                      # attention rollout utilities
  score_pipeline.py               # combines all supported explanation modes
  token_mapping.py                # maps token relevance to time intervals
  visualization.py                # timeline, spectrogram, and class heatmaps

src/evaluation/
  common.py                       # shared evaluation helpers
  ravdess.py                      # RAVDESS parsing and label mapping
  iemocap.py                      # IEMOCAP parsing and label mapping
  explanation_metrics.py          # sparsity and class-specificity metrics

scripts/
  run_relevance_explanation.py    # one-audio explanation
  evaluate_ravdess.py             # RAVDESS prediction baseline
  evaluate_iemocap.py             # IEMOCAP prediction baseline
  evaluate_deletion_faithfulness.py
  evaluate_class_specificity.py
  compare_class_specificity.py    # one-audio predicted vs runner-up heatmap
  summarize_explanation_tables.py # post-process existing metric CSVs

outputs/
  INDEX.md                        # index of local experiment folders

references/
  paper notes and project references
```

The older notebooks and `src/speech_xai_project/` are kept as legacy exploratory
material. The main reproducible workflow is now the command-line HuBERT
pipeline documented below.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Alternatively, install the project package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

All commands below assume you are in the repository root and use:

```bash
.venv/bin/python
```

If you do not have CUDA, use:

```bash
--device cpu
```

The first run may download the Hugging Face checkpoint. After the model is
cached locally, you can add:

```bash
--local-files-only
```

to force offline loading from the local Hugging Face cache.

## Data layout

Datasets are intentionally not committed to Git. Put them under `data/`.

Expected RAVDESS layout:

```text
data/ravdess/Audio_Speech_Actors_01-24/
  Actor_01/
  Actor_02/
  ...
```

Expected IEMOCAP layout:

```text
data/iemocap/archive/IEMOCAP_full_release/
  Session1/
  Session2/
  Session3/
  Session4/
  Session5/
```

IEMOCAP uses the standard four-class mapping:

```text
neu -> neu
hap -> hap
exc -> hap
ang -> ang
sad -> sad
```

The labels `fru`, `fea`, `dis`, `sur`, `oth`, and `xxx` are excluded from the
four-class IEMOCAP evaluation.

RAVDESS is treated as an external cross-corpus evaluation using the strict
four-class subset:

```text
neutral, happy, angry, sad
```

## Output policy

The scripts create timestamped output folders and avoid overwriting existing
research results.

Recommended structure:

```text
outputs/test_XX_dataset_or_scenario/
  run_01_prediction_baseline/
  run_02_faithfulness/
  run_03_class_specificity/
  run_04_metric_tables/
```

The file [outputs/INDEX.md](outputs/INDEX.md) documents the local experiment
folders used during development.

## Phase 1: run one-audio temporal explanations

Run a single explanation:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio path/to/audio.wav \
  --explanation-mode contrastive-conditioned-rollout \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_01_explanation
```

Available modes:

```text
rollout
level3
level3-contrastive
head-conditioned-rollout
contrastive-conditioned-rollout
```

You can force a target class:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio path/to/audio.wav \
  --explanation-mode level3 \
  --target-class 2 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_02_target_ang
```

For contrastive modes, you can also force the competitor class:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio path/to/audio.wav \
  --explanation-mode level3-contrastive \
  --target-class 2 \
  --contrast-class 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_03_ang_vs_sad
```

Each run writes:

```text
*_temporal.csv       # token/time relevance values
*_timeline.png       # waveform/spectrogram/relevance visualization
*_metadata.json      # source audio, class, mode, and output provenance
```

## Phase 2: evaluate prediction performance

These scripts produce the prediction CSVs used by the faithfulness and
class-specificity evaluations.

### RAVDESS smoke test

```bash
.venv/bin/python scripts/evaluate_ravdess.py \
  --max-samples 60 \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_02_ravdess_explanation_benchmark/run_02_external_smoke
```

### RAVDESS full external evaluation

```bash
.venv/bin/python scripts/evaluate_ravdess.py \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_02_ravdess_explanation_benchmark/run_03_external_full
```

### IEMOCAP smoke test

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --max-samples 4 \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_01_evaluation_smoke
```

### IEMOCAP full in-domain diagnostic

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full
```

IEMOCAP note: the public checkpoint is already fine-tuned for IEMOCAP-style
emotion recognition. Treat IEMOCAP results as in-domain diagnostics unless you
define a separate held-out protocol.

Prediction evaluation outputs:

```text
predictions.csv
metrics.json
confusion_matrix.csv
run_manifest.json
```

## Phase 3: deletion-faithfulness evaluation

Deletion faithfulness tests whether removing the most relevant time tokens
reduces the target class logit/probability more than removing bottom-k or random
tokens.

Use a `predictions.csv` produced by Phase 2.

Example for IEMOCAP:

```bash
.venv/bin/python scripts/evaluate_deletion_faithfulness.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --fractions 0.05,0.1,0.2,0.3 \
  --random-trials 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_05_faithfulness_full
```

Example for RAVDESS:

```bash
.venv/bin/python scripts/evaluate_deletion_faithfulness.py \
  --predictions-csv outputs/test_02_ravdess_explanation_benchmark/run_03_external_full/ravdess_external_eval_TIMESTAMP/predictions.csv \
  --dataset-name RAVDESS \
  --max-examples 8 \
  --fractions 0.05,0.1,0.2,0.3 \
  --random-trials 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_02_ravdess_explanation_benchmark/run_05_faithfulness_all_modes
```

Deletion outputs:

```text
audio_manifest.csv
selected_examples.csv
class_coverage.csv
deletion_records.csv
deletion_summary.csv
deletion_curves.png
sparsity_records.csv
sparsity_summary_by_method.csv
sparsity_summary_by_class.csv
robustness_by_class.csv
config.json
```

Key deletion interpretation:

- Stronger faithfulness: top-k deletion should decrease the target probability
  or logit more than bottom-k and random deletion.
- Negative deletion / bottom-k is a sanity check: deleting the least relevant
  regions should usually have a smaller effect.
- `robustness_by_class.csv` helps check whether behavior is stable across
  emotional classes.

## Phase 4: sparsity / concentration metrics

Sparsity is already computed inside the deletion-faithfulness script. The main
summary file is:

```text
sparsity_summary_by_method.csv
```

It reports:

```text
normalized_entropy
effective_tokens
gini
top_5_percent_mass
top_10_percent_mass
```

Interpretation:

- Higher normalized entropy = more diffuse explanation.
- Higher effective tokens = relevance spread across more time tokens.
- Higher Gini = more concentrated explanation.
- Higher top-5/top-10 mass = more relevance concentrated in the most relevant
  time tokens.

This table is useful for formalizing:

```text
rollout is diffuse
contrastive variants are more selective
```

## Phase 5: class-specificity evaluation

Class-specificity compares the relevance map for the predicted class with the
map for the runner-up class on the same audio.

Lower correlation means the method is more class-specific.

Run the batch metric:

```bash
.venv/bin/python scripts/evaluate_class_specificity.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_09_class_specificity_metrics_full
```

Outputs:

```text
audio_manifest.csv
selected_examples.csv
class_coverage.csv
class_specificity_records.csv
class_specificity_summary_by_method.csv
class_specificity_summary_by_class.csv
config.json
```

The main columns are:

```text
pearson_correlation
spearman_correlation
```

Interpretation:

- High positive correlation: predicted-class and runner-up maps are similar.
- Low or negative correlation: maps differ more strongly by target class.
- A contrastive method is expected to have lower correlation than plain rollout.

## Phase 6: one-audio predicted vs runner-up heatmap

For qualitative inspection of one audio:

```bash
.venv/bin/python scripts/compare_class_specificity.py \
  --audio path/to/audio.wav \
  --explanation-mode contrastive-conditioned-rollout \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_04_class_specificity_heatmap
```

Outputs:

```text
class_specificity_scores.csv
class_specificity_summary.csv
class_specificity_heatmap.png
metadata.json
```

The metadata records the exact audio path, predicted class, runner-up class, and
correlation metrics.

## Phase 7: post-process existing metric tables

If you already have `sparsity_records.csv` or `class_specificity_records.csv`,
you can generate compact summary tables without rerunning the model:

```bash
.venv/bin/python scripts/summarize_explanation_tables.py \
  --faithfulness-run outputs/test_03_iemocap_in_domain_benchmark/run_05_faithfulness_full/iemocap_deletion_faithfulness_TIMESTAMP \
  --class-specificity-run outputs/test_03_iemocap_in_domain_benchmark/run_09_class_specificity_metrics_full/iemocap_class_specificity_TIMESTAMP \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_10_metric_tables
```

This writes:

```text
sparsity_summary_by_method.csv
sparsity_summary_by_method.md
class_specificity_summary_by_method.csv
class_specificity_summary_by_method.md
config.json
```

## Recommended full experiment order

For a new dataset/scenario, run:

1. Prediction baseline:

   ```bash
   .venv/bin/python scripts/evaluate_iemocap.py \
     --batch-size 1 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_01_prediction_baseline
   ```

2. Deletion faithfulness and sparsity:

   ```bash
   .venv/bin/python scripts/evaluate_deletion_faithfulness.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --fractions 0.05,0.1,0.2,0.3 \
     --random-trials 3 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_02_faithfulness
   ```

3. Class-specificity:

   ```bash
   .venv/bin/python scripts/evaluate_class_specificity.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_03_class_specificity
   ```

4. Compact tables:

   ```bash
   .venv/bin/python scripts/summarize_explanation_tables.py \
     --faithfulness-run outputs/test_XX_iemocap/run_02_faithfulness/iemocap_deletion_faithfulness_TIMESTAMP \
     --class-specificity-run outputs/test_XX_iemocap/run_03_class_specificity/iemocap_class_specificity_TIMESTAMP \
     --output-root outputs/test_XX_iemocap/run_04_metric_tables
   ```

## CPU and runtime notes

The pipeline can run without CUDA, but some stages are slow on CPU.

Practical CPU recommendations:

- Use `--batch-size 1` for dataset evaluation.
- Start with `--max-samples 4` or `--max-examples 1`.
- Increase to `--max-examples 8`, then larger values if runtime is acceptable.
- Class-specificity is more expensive than single-target explanation because it
  computes maps for both predicted and runner-up classes.
- Deletion faithfulness can become expensive because it reruns the model on
  masked audio for each method, fraction, and random trial.

If Matplotlib warns that `~/.config/matplotlib` is not writable, the scripts
usually still work by using a temporary cache. To silence and speed this up:

```bash
mkdir -p /tmp/mplconfig
export MPLCONFIGDIR=/tmp/mplconfig
```

## Git and data policy

The repository `.gitignore` excludes local datasets, generated outputs, virtual
environments, caches, audio files, and large model artifacts.

Do not commit:

```text
data/
outputs/* large run folders
.venv/
model checkpoints
raw audio datasets
```

If you want to share final results, copy a small curated subset of tables or
figures into a dedicated tracked folder such as:

```text
docs/results/
```

or use Git LFS / external artifact storage for large files.

## Troubleshooting

### The model cannot be loaded with `--local-files-only`

Run once without `--local-files-only` so Hugging Face can download the model:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio path/to/audio.wav \
  --explanation-mode rollout \
  --device cpu
```

Then retry with `--local-files-only`.

### The evaluator cannot find RAVDESS or IEMOCAP

Check that the dataset root matches the expected layout. You can override the
default paths:

```bash
.venv/bin/python scripts/evaluate_ravdess.py \
  --ravdess-root /absolute/path/to/Audio_Speech_Actors_01-24 \
  --max-samples 10 \
  --device cpu
```

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --iemocap-root /absolute/path/to/IEMOCAP_full_release \
  --max-samples 4 \
  --device cpu
```

### How do I know which audio was used?

The important outputs store audio provenance:

- `predictions.csv` has an `audio_path` column.
- deletion/class-specificity runs save `audio_manifest.csv`.
- one-audio explanation outputs save `*_metadata.json`.
- class-specificity heatmaps save `metadata.json`.

### Which result should I report?

Use:

- `metrics.json` and `confusion_matrix.csv` for model performance.
- `deletion_summary.csv` for faithfulness.
- `sparsity_summary_by_method.csv` for concentration/selectivity.
- `class_specificity_summary_by_method.csv` for predicted-vs-runner-up map correlation.
- `robustness_by_class.csv` or `class_specificity_summary_by_class.csv` for emotion-wise behavior.
