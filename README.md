# LeGrad-Inspired Temporal Speech Explanations

Course-project scaffold for comparing SpeechXAI with a LeGrad-inspired temporal attribution method on IEMOCAP speech emotion recognition.

The working plan is captured in [skill.md](skill.md). The repository is organized around diagnostic notebooks first, then explanation generation, masking, deletion evaluation, and final plots.

The fixed classifier is [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er), matching the IEMOCAP model loaded by the Pastor et al. SpeechXAI implementation.

## Repository Layout

- `notebooks/`: one notebook per project phase.
- `src/speech_xai_project/`: reusable helpers used by the notebooks.
- `configs/default.yaml`: paths, model id, SpeechXAI top-k values, random trials, and output file names.
- `data/`: local audio and metadata. Raw IEMOCAP files should stay uncommitted.
- `results/`: generated CSVs, plots, and debug artifacts.
- `scripts/`: optional command-line runners once notebook experiments are stable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Some dependencies, especially SpeechXAI itself, may need manual installation depending on the upstream repository state. Start with Notebook 01, then do not move beyond Notebook 03 until attention maps and attention gradients are confirmed.

## Expected Order

1. `01_inference_iemocap.ipynb`
2. `02_attention_access_test.ipynb`
3. `03_attention_gradient_test.ipynb`
4. `04_legrad_temporal_explanation.ipynb`
5. `05_speechxai_explanations.ipynb`
6. `06_unified_interval_masking.ipynb`
7. `07_deletion_evaluation.ipynb`
8. `08_qualitative_examples.ipynb`
9. `09_final_results_summary.ipynb`

## Corrected Quantitative Evaluation

The deletion evaluation is driven by SpeechXAI top-k words, not fixed percentages of audio duration.

For each audio and each `k` in `1, 2, 3, 5`:

1. Select the top-k SpeechXAI word intervals.
2. Compute their total duration `X`.
3. Mask exactly those SpeechXAI word intervals.
4. Mask LeGrad top time bins whose total duration is `X`.
5. Mask random time bins whose total duration is `X`, repeated over multiple random trials.

This is fairer because SpeechXAI is word-based while LeGrad is time-bin/token-based. The amount of removed audio is held fixed; only the selected regions differ.

## Generated Outputs

The main generated files are expected to be:

- `results/predictions.csv`
- `results/attention_debug.pkl`
- `results/gradient_debug.pkl`
- `results/legrad_explanations.csv`
- `results/speechxai_explanations.csv`
- `results/deletion_results.csv`
