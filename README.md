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
  evaluate_ravdess.py             # RAVDESS prediction baseline
  evaluate_iemocap.py             # IEMOCAP prediction baseline
  evaluate_deletion_faithfulness.py
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

Before running it, put IEMOCAP in Google Drive and edit `IEMOCAP_ROOT` so it
points to the directory that contains `Session1` through `Session5`. RAVDESS is
downloaded automatically when it is not already present.

```python
# Full duration-matched evaluation on Google Colab GPU.
# Edit this path before running. It must contain Session1, ..., Session5.
IEMOCAP_ROOT = "/content/drive/MyDrive/datasets/IEMOCAP_full_release"

from collections import Counter
from pathlib import Path
import os
import subprocess
import sys
import urllib.request
import zipfile


REPO_URL = "https://github.com/MateusWiteck/gradient_based_speach_xai.git"
PROJECT_DIR = Path("/content/gradient_based_speach_xai")
RAVDESS_ROOT = Path("/content/data/ravdess/Audio_Speech_Actors_01-24")
OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/gradient_based_speech_xai_outputs/"
    "test_06_duration_matched_full_both_datasets"
)
SPEECHXAI_COMMIT = "7c43d0ce90c82ca3d2f860534136f06d3640e8d0"


def run(command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


try:
    from google.colab import drive
except ImportError as error:
    raise RuntimeError("This cell is intended to run inside Google Colab.") from error

drive.mount("/content/drive")

# Hard GPU checks. These intentionally raise before the long evaluation starts.
run(["nvidia-smi"])
try:
    import torch
except ImportError:
    torch = None
if torch is not None and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Change the Colab runtime type to GPU.")

if not PROJECT_DIR.exists():
    run(["git", "clone", "--recursive", REPO_URL, PROJECT_DIR])

run(["git", "submodule", "update", "--init", "--recursive", "third_party/LeGrad"], cwd=PROJECT_DIR)

speechxai_dir = PROJECT_DIR / "third_party" / "SpeechXAI"
if not speechxai_dir.exists():
    run(["git", "clone", "https://github.com/elianap/SpeechXAI.git", speechxai_dir])
run(["git", "fetch", "origin"], cwd=speechxai_dir)
run(["git", "checkout", SPEECHXAI_COMMIT], cwd=speechxai_dir)

run([sys.executable, "-m", "pip", "install", "-U", "pip", "wheel"])
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=PROJECT_DIR)

cuda_check = (
    "import torch; "
    "assert torch.cuda.is_available(), 'CUDA is not available after dependency install'; "
    "print('CUDA device:', torch.cuda.get_device_name(0)); "
    "print('torch:', torch.__version__)"
)
run([sys.executable, "-c", cuda_check], cwd=PROJECT_DIR)

if not RAVDESS_ROOT.exists():
    ravdess_dir = RAVDESS_ROOT.parent
    ravdess_dir.mkdir(parents=True, exist_ok=True)
    ravdess_zip = ravdess_dir / "Audio_Speech_Actors_01-24.zip"
    if not ravdess_zip.exists():
        urllib.request.urlretrieve(
            "https://zenodo.org/records/1188976/files/"
            "Audio_Speech_Actors_01-24.zip?download=1",
            ravdess_zip,
        )
    with zipfile.ZipFile(ravdess_zip) as archive:
        archive.extractall(ravdess_dir)

iemocap_root = Path(IEMOCAP_ROOT)
if not (iemocap_root / "Session1").is_dir():
    raise FileNotFoundError(
        f"IEMOCAP_ROOT={iemocap_root} does not contain Session1. "
        "Edit IEMOCAP_ROOT before running the cell."
    )

sys.path.insert(0, str(PROJECT_DIR))
from scripts.evaluate_duration_matched_speechxai import collect_examples
from src.evaluation.iemocap import STANDARD_SESSION_IDS

examples = collect_examples(
    datasets=["iemocap", "ravdess"],
    iemocap_root=iemocap_root,
    iemocap_sessions=STANDARD_SESSION_IDS,
    ravdess_root=RAVDESS_ROOT,
)
counts = Counter(example.dataset for example in examples)
print("Selected records:", dict(sorted(counts.items())), "total=", len(examples))
if counts.get("IEMOCAP") != 5531 or counts.get("RAVDESS") != 672:
    raise RuntimeError(
        "Unexpected dataset counts. Expected IEMOCAP=5531 and RAVDESS=672. "
        f"Got {dict(counts)}."
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
run(
    [
        sys.executable,
        "scripts/evaluate_duration_matched_speechxai.py",
        "--datasets",
        "iemocap,ravdess",
        "--iemocap-root",
        iemocap_root,
        "--ravdess-root",
        RAVDESS_ROOT,
        "--output-root",
        OUTPUT_ROOT,
        "--ks",
        "1,2,3,5",
        "--random-trials",
        "20",
        "--device",
        "cuda",
        "--speechxai-compute-type",
        "float16",
        "--speechxai-batch-size",
        "1",
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
duration_matched_records.csv    # per-mask deletion records
duration_matched_summary.csv    # aggregate confidence-drop summary
speechxai_word_scores.csv       # SpeechXAI/Pastor word leave-one-out scores
selected_intervals.csv          # exact intervals masked by each method
method_catalog.csv              # method labels and descriptions
failures.csv                    # written only when an audio fails
config.json                     # full run configuration
```

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

The file [INDEX.md](INDEX.md) documents the local experiment folders used
during development.
