# LeGrad-Inspired Temporal Speech Explanations

Course-project scaffold for comparing SpeechXAI with a LeGrad-inspired temporal attribution method on IEMOCAP speech emotion recognition.

The working plan is captured in [skill.md](skill.md). Each notebook now has one clear role: validate a technical capability, demonstrate one explanation method, perform the complete quantitative evaluation, or produce qualitative examples.

The fixed classifier is [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er), matching the IEMOCAP model loaded by the Pastor et al. SpeechXAI implementation.

## Repository Layout

- `notebooks/`: focused diagnostics, single-example demonstrations, quantitative evaluation, and qualitative examples.
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

Some dependencies, especially SpeechXAI itself, may need manual installation depending on the upstream repository state. Start with Notebook 01, then do not implement the LeGrad example until Notebooks 02 and 03 confirm attention and gradient access.

## Expected Order

1. `01_speechxai_single_example.ipynb`
2. `02_attention_access_test.ipynb`
3. `03_attention_gradient_test.ipynb`
4. `04_legrad_temporal_explanation.ipynb`
5. `05_quantitative_evaluation.ipynb`
6. `06_qualitative_examples.ipynb`

Notebook 01 is the detailed single-audio SpeechXAI demonstration. Notebook 05 reuses the validated SpeechXAI and LeGrad workflows at evaluation scale and contains interval unification, duration-matched masking, deletion metrics, and aggregate result plots.

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
