# HuBERT Temporal Relevance Explanations for Speech Emotion Recognition

This repository contains a command-line research pipeline for explaining a
HuBERT speech emotion classifier with temporal relevance maps. The current
pipeline includes rollout-based relevance modes, LeGrad-inspired HuBERT modes,
and a duration-matched quantitative comparison against SpeechXAI/Pastor over
IEMOCAP and RAVDESS.

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
| `legrad_final_score_relu_attention_gradient_mean_layers_source_tokens` | LeGrad-inspired positive attention-gradient relevance using the final classifier score and mean layer aggregation. |
| `legrad_final_score_relu_attention_gradient_renormalized_hubert_layer_weighted_source_tokens` | Same final-score LeGrad-inspired relevance, aggregated with renormalized HuBERT learned layer weights. |
| `legrad_layer_local_score_relu_attention_gradient_mean_layers_source_tokens` | Layer-local score ablation with mean layer aggregation. |
| `legrad_layer_local_score_relu_attention_gradient_renormalized_hubert_layer_weighted_source_tokens` | Layer-local score ablation with renormalized HuBERT learned layer weights. |

Important conceptual note: this is **not** full Transformer LRP. The project
does not propagate relevance through every HuBERT Transformer block, GELU,
LayerNorm, residual branch, or attention internals. The current "Level 3"
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
  legrad_hubert.py                # LeGrad-inspired HuBERT temporal relevance variants
  rollout.py                      # attention rollout utilities
  score_pipeline.py               # combines all supported explanation modes
  token_mapping.py                # maps token relevance to time intervals
  visualization.py                # timeline, spectrogram, and class heatmaps

src/evaluation/
  common.py                       # shared evaluation helpers
  duration_matched_speechxai.py   # SpeechXAI/Pastor vs HuBERT shared-duration evaluation
  ravdess.py                      # RAVDESS parsing and label mapping
  iemocap.py                      # IEMOCAP parsing and label mapping
  explanation_metrics.py          # sparsity and class-specificity metrics

scripts/
  run_relevance_explanation.py    # one-audio explanation
  evaluate_duration_matched_speechxai.py # full SpeechXAI/Pastor comparison
  run_colab_duration_matched_eval.py # Colab setup and full GPU run helper
  evaluate_ravdess.py             # RAVDESS prediction baseline
  evaluate_iemocap.py             # IEMOCAP prediction baseline
  evaluate_deletion_faithfulness.py
  evaluate_insertion_faithfulness.py
  evaluate_truncated_rollout.py
  evaluate_hidden_gradient_relevance.py
  analyze_relevant_acoustic_features.py
  evaluate_class_specificity.py
  compare_class_specificity.py    # one-audio predicted vs runner-up heatmap
  summarize_explanation_tables.py # post-process existing metric CSVs

outputs/
  test_*/                          # local experiment folders

references/
  paper notes and project references
```

The notebooks are used for inspection and one-audio demonstrations. The main
reproducible full-dataset workflow is the command-line duration-matched
evaluator documented below.

## Reproducibility checklist

The full quantitative experiment uses only these external assets:

| Asset | Source | Required local location |
| --- | --- | --- |
| Project code | `https://github.com/MateusWiteck/gradient_based_speach_xai.git` | repository root |
| LeGrad reference code | `https://github.com/WalBouss/LeGrad.git` | `third_party/LeGrad` |
| SpeechXAI code | `https://github.com/elianap/SpeechXAI.git` | `third_party/SpeechXAI` |
| HuBERT classifier | Hugging Face `superb/hubert-base-superb-er` | Hugging Face cache |
| IEMOCAP audio/labels | USC IEMOCAP release: `https://sail.usc.edu/iemocap/` | `data/Session1` ... `data/Session5` |
| RAVDESS speech audio | Zenodo RAVDESS record: `https://zenodo.org/records/1188976` | `data/ravdess/Audio_Speech_Actors_01-24` |
| FFmpeg | system install or local executable | system `PATH` or `third_party/ffmpeg/ffmpeg.exe` |

Fresh clone:

```powershell
git clone https://github.com/MateusWiteck/gradient_based_speach_xai.git
cd gradient_based_speach_xai
git submodule update --init --recursive third_party/LeGrad
```

The current project configuration expects SpeechXAI as a local third-party
checkout. Reproduce the version used during development with:

```powershell
git clone https://github.com/elianap/SpeechXAI.git third_party\SpeechXAI
git -C third_party\SpeechXAI checkout 7c43d0ce90c82ca3d2f860534136f06d3640e8d0
```

The LeGrad submodule commit used in this workspace is:

```text
a9edb031422d5f5829d9e6238a1e21feb1972c92
```

IEMOCAP cannot be redistributed with this repository. Request/download it from
USC, extract the official release, and copy or symlink the `Session1` through
`Session5` folders directly under `data/`.

RAVDESS can be downloaded from Zenodo. The file used by this project is
`Audio_Speech_Actors_01-24.zip`. On Windows:

```powershell
New-Item -ItemType Directory -Force data\ravdess
Invoke-WebRequest `
  -Uri "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1" `
  -OutFile data\ravdess\Audio_Speech_Actors_01-24.zip
Expand-Archive `
  -Path data\ravdess\Audio_Speech_Actors_01-24.zip `
  -DestinationPath data\ravdess `
  -Force
```

Before a full run, verify that both dataset parsers see the expected records:

```powershell
@'
from collections import Counter
from scripts.evaluate_duration_matched_speechxai import collect_examples
from src.evaluation.iemocap import STANDARD_SESSION_IDS

examples = collect_examples(
    datasets=["iemocap", "ravdess"],
    iemocap_root="data",
    iemocap_sessions=STANDARD_SESSION_IDS,
    ravdess_root="data/ravdess/Audio_Speech_Actors_01-24",
)
print("total", len(examples))
print(Counter(example.dataset for example in examples))
'@ | .\.venv\Scripts\python.exe -
```

In the current local layout this prints `5531` IEMOCAP records and `672`
RAVDESS records.

Every full quantitative run writes a `config.json` containing the command-line
arguments, project/SpeechXAI/LeGrad git commits, dirty git status, Python
version, package versions, FFmpeg version, selected dataset roots, and output
file names. Treat a run as exactly reproducible only when the project repository
and third-party repositories are on recorded commits with no uncommitted changes
that affect the evaluation.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
```

`requirements.txt` pins `setuptools==70.3.0` because `ctranslate2`, used by
WhisperX/faster-whisper, imports the deprecated `pkg_resources` module. This
pin keeps normal runs free of the default setuptools 81 deprecation warning
until upstream removes that import.

Alternatively, install the project package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

All commands below assume you are in the repository root and use:

```bash
.venv/bin/python
```

On Windows, use the equivalent:

```powershell
.\.venv\Scripts\python.exe
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

Expected IEMOCAP layout for this workspace and the default config:

```text
data/
  Session1/
  Session2/
  Session3/
  Session4/
  Session5/
```

The IEMOCAP parser also accepts a nested release root such as
`data/iemocap/archive/IEMOCAP_full_release/` when `--iemocap-root` points to
`data/iemocap`.

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

## Full quantitative evaluation

Run the duration-matched SpeechXAI/Pastor vs HuBERT evaluation over all
compatible audios in both datasets with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_duration_matched_speechxai.py `
  --datasets iemocap,ravdess `
  --iemocap-root data `
  --ravdess-root data\ravdess\Audio_Speech_Actors_01-24 `
  --output-root outputs\test_06_duration_matched_full_both_datasets `
  --ks 1,2,3,5 `
  --random-trials 20
```

The same command for bash/zsh is:

```bash
.venv/bin/python scripts/evaluate_duration_matched_speechxai.py \
  --datasets iemocap,ravdess \
  --iemocap-root data \
  --ravdess-root data/ravdess/Audio_Speech_Actors_01-24 \
  --output-root outputs/test_06_duration_matched_full_both_datasets \
  --ks 1,2,3,5 \
  --random-trials 20
```

### Google Colab GPU cell

Paste the following cell into Google Colab after selecting a GPU runtime
(`Runtime -> Change runtime type -> GPU`). The cell fails immediately if CUDA is
not visible or if PyTorch is not using the GPU.

Before running it, create a Colab Secret named `KAGGLE_API_TOKEN`, paste a valid
Kaggle API token, and enable Notebook access for that secret. The helper script
downloads IEMOCAP from Kaggle, downloads RAVDESS from Zenodo, stores reusable
SpeechXAI word-alignment cache files in Google Drive, and writes results to
Google Drive.

```python
from pathlib import Path
import subprocess
import shutil
import sys

REPO_URL = "https://github.com/MateusWiteck/gradient_based_speach_xai.git"
PROJECT_BRANCH = "main"
PROJECT_DIR = Path("/content/gradient_based_speach_xai")


def run(command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


if PROJECT_DIR.exists() and not (PROJECT_DIR / ".git").exists():
    shutil.rmtree(PROJECT_DIR)

if not (PROJECT_DIR / ".git").exists():
    run(["git", "clone", "--branch", PROJECT_BRANCH, "--single-branch", REPO_URL, PROJECT_DIR])
else:
    run(["git", "remote", "set-url", "origin", REPO_URL], cwd=PROJECT_DIR)
    run(["git", "fetch", "--prune", "origin", PROJECT_BRANCH], cwd=PROJECT_DIR)
    run(["git", "checkout", "-B", PROJECT_BRANCH, f"origin/{PROJECT_BRANCH}"], cwd=PROJECT_DIR)
    run(["git", "reset", "--hard", f"origin/{PROJECT_BRANCH}"], cwd=PROJECT_DIR)

run(["git", "log", "-1", "--oneline"], cwd=PROJECT_DIR)

run(
    [
        sys.executable,
        "scripts/run_colab_duration_matched_eval.py",
        "--require-gpu",
    ],
    cwd=PROJECT_DIR,
)
```

The command intentionally does not use `--max-audios`; it evaluates every
selected IEMOCAP and RAVDESS record. Device selection is automatic. Add
`--device cpu` to force CPU or `--local-files-only` after the HuBERT checkpoint
is already cached. The default SpeechXAI Whisper model is `large-v2`, so a full
CPU run can take a long time.

Each run creates a timestamped folder under the selected `--output-root`, for
example:

```text
outputs/test_06_duration_matched_full_both_datasets/
  iemocap_ravdess_duration_matched_speechxai_YYYYMMDD_HHMMSS_ffffff/
```

The generated result files are:

```text
audio_manifest.csv              # selected audio provenance
original_predictions.csv        # unmasked HuBERT predictions
audio_progress.csv              # one synced checkpoint row after each audio
duration_matched_records.csv    # per-mask deletion records
duration_matched_summary.csv    # aggregate confidence-drop summary
speechxai_word_scores.csv       # SpeechXAI/Pastor word leave-one-out scores
selected_intervals.csv          # exact intervals masked by each method
method_catalog.csv              # method labels and descriptions
failures.csv                    # written only when an audio fails
config.json                     # full run configuration
```

`audio_progress.csv` is written after each audio finishes, is skipped, or fails.
It records how many rows were persisted to each table and can be used to audit
partial Colab runs without waiting for the final summary step.
If a run is interrupted with `Ctrl+C`, the current audio is marked as
`interrupted`, partial summary/config files are written, and the process exits
with code `130`.

The random baseline is random deletion by silence masking: random fixed-duration
audio bins are selected to match the SpeechXAI top-k duration, set to zero, and
the waveform length is preserved.

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

## Phase 4: insertion-faithfulness evaluation

Insertion faithfulness is the positive counterpart of deletion. It starts from
a silent waveform and copies back only the original regions selected by an
explanation. If the explanation is faithful, inserting top-ranked regions
should recover the target class probability/logit faster than inserting
bottom-ranked or random regions.

Example for IEMOCAP:

```bash
.venv/bin/python scripts/evaluate_insertion_faithfulness.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --fractions 0.05,0.1,0.2,0.3 \
  --random-trials 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_06_insertion_full
```

Insertion outputs:

```text
audio_manifest.csv
selected_examples.csv
class_coverage.csv
insertion_records.csv
insertion_summary.csv
insertion_auc_by_example.csv
insertion_auc_by_method.csv
insertion_robustness_by_class.csv
insertion_curves.png
config.json
```

Key insertion interpretation:

- Stronger faithfulness: top-k insertion should recover the target probability
  or logit faster than bottom-k and random insertion.
- `insertion_auc_by_method.csv` summarizes each curve; higher AUC/recovery is
  better.
- `insertion_robustness_by_class.csv` checks whether the behavior is stable
  across emotional classes.

## Phase 5: truncated rollout over the last k layers

Full rollout can become diffuse because it multiplies relevance through every
Transformer layer. A useful ablation is to run rollout using only the last
`k` layers, for example `1, 2, 4, 6`, and evaluate whether this reduces
diffusion while preserving deletion faithfulness.

Example:

```bash
.venv/bin/python scripts/evaluate_truncated_rollout.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --last-k-layers 1,2,4,6 \
  --fractions 0.05,0.1,0.2,0.3 \
  --random-trials 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_11_truncated_rollout
```

Outputs:

```text
audio_manifest.csv
selected_examples.csv
class_coverage.csv
deletion_records.csv
deletion_summary.csv
deletion_curves.png
sparsity_records.csv
sparsity_summary_by_method.csv
robustness_by_class.csv
config.json
```

The generated explanation modes are named:

```text
rollout-last-1
rollout-last-2
rollout-last-4
rollout-last-6
```

Interpretation:

- Lower entropy and higher Gini/top-k mass indicate less diffuse rollout.
- Higher top-deletion logit/probability drop indicates better faithfulness.
- The best `last-k` value is the one that improves concentration without
  destroying deletion faithfulness.

## Phase 6: gradient × hidden-state relevance

This diagnostic compares the existing linear head relevance with a local
gradient-based score at the final HuBERT token representation:

```text
score[t] = sum_d abs(H[t, d] * d logit_c / d H[t, d])
```

Run:

```bash
.venv/bin/python scripts/evaluate_hidden_gradient_relevance.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --fractions 0.05,0.1,0.2,0.3 \
  --random-trials 3 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_12_hidden_gradient
```

Outputs:

```text
head_vs_gradient_hidden_records.csv
head_vs_gradient_hidden_summary.csv
sparsity_records.csv
sparsity_summary_by_method.csv
deletion_records.csv
deletion_summary.csv
deletion_curves.png
robustness_by_class.csv
config.json
```

Interpretation:

- `head_vs_gradient_hidden_summary.csv` tells whether both maps are similar.
- `sparsity_summary_by_method.csv` shows whether gradient-hidden is more or
  less concentrated than linear head relevance.
- `deletion_summary.csv` checks whether either token score is more faithful
  under top-token deletion.

## Phase 7: acoustic features in relevant regions

This analysis connects temporal relevance maps to human-interpretable acoustic
concepts. For each selected audio, it extracts features from the top, bottom,
and optionally random relevance regions:

```text
pitch / F0
RMS energy
pause and silence statistics
speaking-rate proxy from energy-envelope peaks
```

Important note: the speaking-rate value is an acoustic proxy, not true words
per second. It does not use transcripts or forced alignment.

Run:

```bash
.venv/bin/python scripts/analyze_relevant_acoustic_features.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --fractions 0.05,0.1 \
  --modes rollout,level3,level3-contrastive,head-conditioned-rollout,contrastive-conditioned-rollout,head-relevance \
  --selection-types top,bottom,random \
  --random-trials 3 \
  --context-tokens 2 \
  --merge-gap-tokens 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_13_acoustic_features
```

You can include the gradient × hidden-state map too:

```bash
--modes rollout,level3-contrastive,contrastive-conditioned-rollout,head-relevance,gradient-hidden
```

Outputs:

```text
acoustic_segment_records.csv
acoustic_selection_records.csv
acoustic_full_audio_records.csv
acoustic_summary_by_method.csv
acoustic_summary_by_class.csv
acoustic_top_bottom_differences.csv
acoustic_top_bottom_differences_summary.csv
config.json
```

Interpretation:

- `acoustic_segment_records.csv` is the fine-grained table: one row per merged
  relevant excerpt.
- `acoustic_selection_records.csv` aggregates all top/bottom/random excerpts
  for one audio, method, and fraction.
- `acoustic_top_bottom_differences_summary.csv` is often the most useful table:
  it asks whether top regions have higher pitch, energy, voicing, or
  speaking-rate proxy than bottom regions.

## Phase 8: sparsity / concentration metrics

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

## Phase 9: class-specificity evaluation

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

## Phase 10: one-audio predicted vs runner-up heatmap

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

## Phase 11: post-process existing metric tables

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

3. Insertion faithfulness:

   ```bash
   .venv/bin/python scripts/evaluate_insertion_faithfulness.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --fractions 0.05,0.1,0.2,0.3 \
     --random-trials 3 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_03_insertion
   ```

4. Truncated rollout last-k ablation:

   ```bash
   .venv/bin/python scripts/evaluate_truncated_rollout.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --last-k-layers 1,2,4,6 \
     --fractions 0.05,0.1,0.2,0.3 \
     --random-trials 3 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_04_truncated_rollout
   ```

5. Gradient × hidden-state comparison:

   ```bash
   .venv/bin/python scripts/evaluate_hidden_gradient_relevance.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --fractions 0.05,0.1,0.2,0.3 \
     --random-trials 3 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_05_hidden_gradient
   ```

6. Acoustic features in relevant regions:

   ```bash
   .venv/bin/python scripts/analyze_relevant_acoustic_features.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --fractions 0.05,0.1 \
     --selection-types top,bottom,random \
     --random-trials 3 \
     --context-tokens 2 \
     --merge-gap-tokens 1 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_06_acoustic_features
   ```

7. Class-specificity:

   ```bash
   .venv/bin/python scripts/evaluate_class_specificity.py \
     --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
     --dataset-name IEMOCAP \
     --max-examples 8 \
     --device cpu \
     --local-files-only \
     --output-root outputs/test_XX_iemocap/run_07_class_specificity
   ```

8. Compact tables:

   ```bash
   .venv/bin/python scripts/summarize_explanation_tables.py \
     --faithfulness-run outputs/test_XX_iemocap/run_02_faithfulness/iemocap_deletion_faithfulness_TIMESTAMP \
     --class-specificity-run outputs/test_XX_iemocap/run_07_class_specificity/iemocap_class_specificity_TIMESTAMP \
     --output-root outputs/test_XX_iemocap/run_08_metric_tables
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
