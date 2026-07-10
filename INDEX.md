# Research output index

Outputs are grouped by **experimental scenario**, not by individual command. Within each scenario, runs are in chronological order. Existing result files were moved only; none were deleted or overwritten.

| Scenario folder | Included runs | Purpose |
| --- | --- | --- |
| `test_01_synthetic_sanity` | `run_01_sine`, `run_02_synthetic_speech` | Early controlled checks of rollout, Level 3, and contrastive relevance. |
| `test_02_ravdess_explanation_benchmark` | `run_01_legacy_single_audio_unverified` through `run_06_class_specificity` | One RAVDESS scenario: external baseline evaluation, deletion-faithfulness, concentration, robustness, and class-specificity analysis. |
| `test_03_iemocap_in_domain_benchmark` | `run_01_evaluation_smoke` through `run_08_class_specificity_metrics_smoke` | CPU-validated IEMOCAP workflow using the standard four-class mapping, including full prediction evaluation, faithfulness, sparsity tables, and class-specificity metrics. |
| `test_05_duration_matched_smoke` | one-audio smoke runs | Duration-matched SpeechXAI/Pastor vs HuBERT evaluation used to validate the shared-duration deletion pipeline. |
| `test_06_duration_matched_full_both_datasets` | timestamped full runs | Full quantitative evaluation over all compatible IEMOCAP and RAVDESS audios with SpeechXAI/Pastor, every HuBERT explanation mode, bottom controls, and random deletion by silence masking. |

## RAVDESS run order

| Run | What it contains |
| --- | --- |
| `run_01_legacy_single_audio_unverified` | Earlier single-audio explanation files made before audio provenance was stored. The source audio is intentionally not inferred. |
| `run_02_external_smoke` | Small 60-example external-prediction baseline. Its `predictions.csv` is the source of the initial faithfulness selections. |
| `run_03_external_full` | Full 672-example RAVDESS external-prediction baseline. |
| `run_04_faithfulness_smoke` | One-example, first-three-mode deletion-faithfulness smoke test. |
| `run_05_faithfulness_all_modes` | All five modes, with deletion, concentration, robustness, and class-coverage results. |
| `run_06_class_specificity` | Predicted-versus-runner-up class heatmap comparison. |

## IEMOCAP run order

IEMOCAP uses the official `EmoEvaluation` annotations. Its standard four-class
subset maps `neu -> neu`, `hap -> hap`, `exc -> hap`, `ang -> ang`, and
`sad -> sad`.

| Run | What it contains |
| --- | --- |
| `run_01_evaluation_smoke` | Four utterances, one per model class, validating the IEMOCAP parser and prediction evaluator on CPU. |
| `run_02_faithfulness_smoke` | One correctly classified IEMOCAP utterance evaluated across all five explanation modes. It retains session, dialogue, utterance, annotation, and audio provenance. |
| `run_03_class_specificity_smoke` | Predicted-versus-runner-up class heatmap for the same selected IEMOCAP utterance. |
| `run_04_evaluation_full` | Full IEMOCAP four-class prediction table, metrics, and confusion matrix. |
| `run_05_faithfulness_full` | Multi-example IEMOCAP deletion-faithfulness run with all five explanation modes and raw sparsity records. |
| `run_06_class_specificity` | Single-audio predicted-versus-runner-up class heatmap comparison. |
| `run_07_metric_tables` | Post-processed summary tables, including sparsity by method and optional class-specificity by method. |
| `run_08_class_specificity_metrics_smoke` | One-example CPU smoke test for the new predicted-vs-runner-up correlation metric across all explanation modes. |

The IEMOCAP checkpoint is already fine-tuned on IEMOCAP. Treat its results as
in-domain diagnostics unless an explicitly documented held-out protocol is used.

## Duration-matched SpeechXAI/Pastor comparison

The current quantitative comparison is owned by:

```text
scripts/evaluate_duration_matched_speechxai.py
```

Full run over both datasets:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_duration_matched_speechxai.py `
  --datasets iemocap,ravdess `
  --iemocap-root data `
  --ravdess-root data\ravdess\Audio_Speech_Actors_01-24 `
  --output-root outputs\test_06_duration_matched_full_both_datasets `
  --ks 1,2,3,5 `
  --random-trials 20
```

The command writes one timestamped folder named like:

```text
iemocap_ravdess_duration_matched_speechxai_YYYYMMDD_HHMMSS_ffffff
```

Expected files:

| File | Purpose |
| --- | --- |
| `audio_manifest.csv` | All selected IEMOCAP and RAVDESS audio records. |
| `original_predictions.csv` | Original HuBERT prediction and target confidence before masking. |
| `speechxai_word_scores.csv` | SpeechXAI/Pastor leave-one-out word scores. |
| `selected_intervals.csv` | Exact word, token, bottom-control, and random deletion by silence masking intervals. |
| `duration_matched_records.csv` | Per-mask deletion result rows. |
| `duration_matched_summary.csv` | Aggregated confidence-drop and flip-rate metrics. |
| `method_catalog.csv` | Method labels and descriptions. |
| `failures.csv` | Failed audio records, written only when failures occur. |
| `config.json` | Run configuration and output manifest. |

This full command intentionally has no `--max-audios` argument. Use
`--max-audios N` only for smoke tests.

Each full run stores reproducibility metadata in `config.json`, including the
command, Python/package versions, project commit, SpeechXAI commit, LeGrad
commit, dirty git status, platform, and FFmpeg version.

## Finding the audio used

New explanation CSV files contain `audio_path` and have a sibling `*_metadata.json`. Evaluation `predictions.csv` records an `audio_path` per row. Class-specificity stores it in `metadata.json`. The legacy run is the only exception.

## Future scenarios

Create one top-level scenario folder, then direct each related command into it. For example:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio data/test_speech.wav \
  --explanation-mode contrastive-conditioned-rollout \
  --output-root outputs/test_04_new_scenario/run_01_single_audio
```

Use `run_02_*`, `run_03_*`, and so on for the other experiments in that same scenario. The evaluator, deletion-faithfulness evaluator, and class-specificity scripts also accept `--output-root`.
