# LeGrad-Inspired Temporal Speech Explanations

Course-project scaffold for comparing SpeechXAI with a LeGrad-inspired temporal attribution method on IEMOCAP speech emotion recognition.

The working plan is captured in [skill.md](skill.md). The repository is organized around diagnostic notebooks first, then explanation generation, masking, deletion evaluation, and final plots.

## Repository Layout

- `notebooks/`: one notebook per project phase.
- `src/speech_xai_project/`: reusable helpers used by the notebooks.
- `configs/default.yaml`: paths, model id, budgets, and output file names.
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

## Generated Outputs

The main generated files are expected to be:

- `results/predictions.csv`
- `results/attention_debug.pkl`
- `results/gradient_debug.pkl`
- `results/legrad_explanations.csv`
- `results/speechxai_explanations.csv`
- `results/deletion_results.csv`

