# Scripts

This directory contains the command-line scripts used to run the project's experiments and evaluations.

Run all commands from the repository root.

---

## 1. Single-audio explanation

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

Target class:

```bash
.venv/bin/python scripts/run_relevance_explanation.py \
  --audio path/to/audio.wav \
  --explanation-mode level3 \
  --target-class 2 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_02_target_ang
```

Contrastive target:

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

---

## 2. Prediction evaluation

### RAVDESS (smoke test)

```bash
.venv/bin/python scripts/evaluate_ravdess.py \
  --max-samples 60 \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_02_ravdess_explanation_benchmark/run_02_external_smoke
```

### RAVDESS (full)

```bash
.venv/bin/python scripts/evaluate_ravdess.py \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_02_ravdess_explanation_benchmark/run_03_external_full
```

### IEMOCAP (smoke test)

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --max-samples 4 \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_01_evaluation_smoke
```

### IEMOCAP (full)

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full
```

---

## 3. Deletion Faithfulness

### IEMOCAP

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

### RAVDESS

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

---

## 4. Insertion Faithfulness

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

---

## 5. Truncated Rollout

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

---

## 6. Hidden Gradient Relevance

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

---

## 7. Acoustic Feature Analysis

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

Gradient-hidden can also be included:

```bash
--modes rollout,level3-contrastive,contrastive-conditioned-rollout,head-relevance,gradient-hidden
```

---

## 8. Class Specificity

```bash
.venv/bin/python scripts/evaluate_class_specificity.py \
  --predictions-csv outputs/test_03_iemocap_in_domain_benchmark/run_04_evaluation_full/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_09_class_specificity_metrics_full
```

---

## 9. Predicted vs Runner-up Comparison

```bash
.venv/bin/python scripts/compare_class_specificity.py \
  --audio path/to/audio.wav \
  --explanation-mode contrastive-conditioned-rollout \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_04_single_audio/run_04_class_specificity_heatmap
```

---

## 10. Summarize Results

```bash
.venv/bin/python scripts/summarize_explanation_tables.py \
  --faithfulness-run outputs/test_03_iemocap_in_domain_benchmark/run_05_faithfulness_full/iemocap_deletion_faithfulness_TIMESTAMP \
  --class-specificity-run outputs/test_03_iemocap_in_domain_benchmark/run_09_class_specificity_metrics_full/iemocap_class_specificity_TIMESTAMP \
  --output-root outputs/test_03_iemocap_in_domain_benchmark/run_10_metric_tables
```

---

## Recommended execution order

1. Prediction baseline

```bash
.venv/bin/python scripts/evaluate_iemocap.py \
  --batch-size 1 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_XX_iemocap/run_01_prediction_baseline
```

2. Deletion faithfulness

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

3. Insertion faithfulness

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

4. Truncated rollout

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

5. Hidden gradient relevance

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

6. Acoustic feature analysis

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

7. Class specificity

```bash
.venv/bin/python scripts/evaluate_class_specificity.py \
  --predictions-csv outputs/test_XX_iemocap/run_01_prediction_baseline/iemocap_in_domain_eval_TIMESTAMP/predictions.csv \
  --dataset-name IEMOCAP \
  --max-examples 8 \
  --device cpu \
  --local-files-only \
  --output-root outputs/test_XX_iemocap/run_07_class_specificity
```

8. Summarize results

```bash
.venv/bin/python scripts/summarize_explanation_tables.py \
  --faithfulness-run outputs/test_XX_iemocap/run_02_faithfulness/iemocap_deletion_faithfulness_TIMESTAMP \
  --class-specificity-run outputs/test_XX_iemocap/run_07_class_specificity/iemocap_class_specificity_TIMESTAMP \
  --output-root outputs/test_XX_iemocap/run_08_metric_tables
```
