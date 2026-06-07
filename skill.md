# Project Working Plan

## Objective

Compare two explanation methods for a pretrained speech emotion
recognition classifier on IEMOCAP:

1. SpeechXAI word-level explanations.
2. A LeGrad-inspired temporal attribution method for wav2vec 2.0.

Both methods use the same classifier and are evaluated with the same
duration-matched silence-masking protocol. The main faithfulness metric
is the confidence drop for the class predicted from the original audio.

## Development Rules

- Notebooks are experiments with a clear technical question or
  deliverable. Avoid notebooks that only split one pipeline into
  artificial phases.
- Run every notebook after creating or modifying it. Also run the
  relevant validation or smoke test.
- If execution is blocked, report the exact missing dependency, data,
  credential, or failing command.
- Show intermediate evidence such as tensor shapes, predictions,
  interval tables, plots, and assertions.
- Keep reusable logic in `src/speech_xai_project/`; notebooks should
  explain and exercise that logic.
- Use explicit variable names such as `predictions_table` and
  `deletion_results_table`.
- Write explanations positively and directly.
- End each notebook with a concise status stating what was tested, what
  worked, what remains unresolved, generated files, and the next step.

## Dataset And Model

Dataset:

- Use IEMOCAP.
- Keep raw session data uncommitted.
- Configure paths through `configs/default.yaml`.

Classifier:

- Use `superb/wav2vec2-base-superb-er`, matching the model used by
  Pastor et al. for the SpeechXAI IEMOCAP experiment.
- Load it with Hugging Face
  `Wav2Vec2ForSequenceClassification`.
- Keep this classifier fixed for SpeechXAI, LeGrad-inspired
  explanations, masking, and evaluation.

## Notebook Sequence

### Notebook 01 - SpeechXAI Single Example

File: `notebooks/01_speechxai_single_example.ipynb`

Purpose:

Demonstrate the complete SpeechXAI process on one IEMOCAP utterance in
enough detail to understand and validate every stage.

Tasks:

- Load one waveform and run the fixed classifier.
- Use SpeechXAI/WhisperX to transcribe and force-align words.
- Explain how VAD, Whisper transcription, and wav2vec 2.0 forced
  alignment produce word timestamps.
- Reuse the same word-segment objects for attribution and plotting.
- Run SpeechXAI leave-one-out attribution for the original predicted
  class.
- Display lexical word intervals, repository removal intervals,
  alignment confidence, attribution values, and the original audio.
- Show where the repository's `-100 ms/+40 ms` removal intervals
  overlap.
- Validate timestamp order, bounds, score alignment, and consistency
  between project and SpeechXAI audio-loading paths.
- Demonstrate the duration-matched random baseline on this one example
  as a local sanity check.
- Generate no result files.

Success criterion:

One audio has a valid, interpretable, fully executed SpeechXAI
explanation with real timestamps, attribution values, visualizations,
and consistency checks.

### Notebook 02 - Attention Access Test

File: `notebooks/02_attention_access_test.ipynb`

Purpose:

Prove that the wav2vec 2.0 Transformer returns the attention tensors
needed by the LeGrad-inspired method.

Tasks:

- Inspect the classifier architecture.
- Run inference with attention output enabled.
- Report the number of layers and heads.
- Validate attention tensor dimensions and finite values.
- Visualize at least one layer/head matrix.

Success criterion:

The model returns one `[batch, heads, tokens, tokens]` attention tensor
per Transformer layer.

### Notebook 03 - Attention Gradient Test

File: `notebooks/03_attention_gradient_test.ipynb`

Purpose:

Prove that gradients of the original predicted-class logit with respect
to attention matrices are accessible and numerically meaningful.

Tasks:

- Reuse the working forward path from Notebook 02.
- Backpropagate from the original predicted-class logit.
- Retain or hook attention tensors and their gradients.
- Report tensor shapes, minimum, maximum, mean, and nonzero percentage.
- Test hooks or alternate forward paths if returned tensors do not
  retain gradients.

Success criterion:

At least one layer returns finite, nonzero
`d class_logit / d attention_matrix` values.

### Notebook 04 - LeGrad-Inspired Single Example

File: `notebooks/04_legrad_temporal_explanation.ipynb`

Purpose:

Implement and visualize the LeGrad-inspired temporal explanation on one
audio after attention and gradient access are validated.

Method:

```text
R_l = mean_heads(ReLU(G_l) * A_l)
R = mean_layers(R_l)
token_relevance = reduce one token-token dimension of R
```

Tasks:

- Compute attention-gradient relevance.
- State and justify the token-token reduction direction.
- Normalize relevance for visualization and preserve raw values where
  needed.
- Map Transformer tokens to time intervals using the feature encoder's
  temporal resolution.
- Plot waveform and temporal relevance.
- Produce a ranked interval table.
- Check that relevance is finite, nonzero, and nonconstant.

Success criterion:

One audio has a ranked, time-aligned LeGrad-inspired explanation that
can be passed to the quantitative evaluation.

### Notebook 05 - Quantitative Evaluation

File: `notebooks/05_quantitative_evaluation.ipynb`

Purpose:

Run the complete multi-audio comparison. This notebook owns batch
explanation generation, unified intervals, masking, deletion metrics,
and aggregate quantitative results.

Tasks:

- Select and document the controlled IEMOCAP evaluation set.
- Run the validated SpeechXAI workflow from Notebook 01 for every
  audio.
- Run the validated LeGrad workflow from Notebook 04 for the same
  audios.
- Represent both methods as `audio_id, start, end, score`.
- Cache transcriptions and explanations when useful for repeatability.
- For each `k` in `{1, 2, 3, 5}`, select the top-k SpeechXAI words.
- Compute `X` as the union duration of their actual repository removal
  intervals.
- Silence SpeechXAI regions, preserving waveform length.
- Select and silence LeGrad top bins with total duration `X`.
- Sample and silence random bins with total duration `X` for 20 trials.
- Optionally evaluate LeGrad-bottom using the same duration.
- Classify every masked waveform against the original predicted class.
- Record absolute confidence drop, relative confidence drop,
  prediction flip, and actual masked duration.
- Verify equal duration across methods for every audio and `k`.
- Save `results/deletion_results.csv`.
- Produce aggregate tables, deletion curves, boxplots, masked-duration
  checks, and flip-rate summaries.

Required result columns:

```text
audio_id
k
method
masked_duration
original_label
original_confidence
masked_confidence
confidence_drop
relative_confidence_drop
prediction_flipped
random_trial
```

Success criterion:

SpeechXAI, LeGrad-top, and repeated random masking are compared on the
same audios with the same classifier, target class, masking operation,
and occluded duration. The notebook contains the main quantitative
evidence for the report.

### Notebook 06 - Qualitative Examples

File: `notebooks/06_qualitative_examples.ipynb`

Purpose:

Create report-ready examples selected from Notebook 05's quantitative
results.

Choose three to five cases:

- SpeechXAI and LeGrad agree.
- LeGrad produces a larger confidence drop.
- SpeechXAI produces a larger confidence drop.
- Both methods are weak or random is competitive.

For each case show:

- waveform and playable audio;
- SpeechXAI words, attribution scores, and selected masks;
- LeGrad temporal relevance and selected bins;
- original confidence and masked confidence for each method;
- a concise interpretation grounded in the quantitative result.

Success criterion:

The notebook produces representative figures that explain agreements,
disagreements, and failure cases without duplicating aggregate analysis.

## Quantitative Evaluation Protocol

For each audio:

1. Classify the original waveform.
2. Fix the original predicted class as the evaluation target.
3. Generate SpeechXAI and LeGrad-inspired explanations.
4. Select the top-k SpeechXAI words.
5. Compute the union duration `X` of their effective removal masks.
6. Evaluate SpeechXAI after silencing those masks.
7. Evaluate LeGrad after silencing top bins totaling `X`.
8. Evaluate repeated random bins totaling `X`.
9. Measure:

```text
confidence_drop = p_original(target) - p_masked(target)
relative_drop = confidence_drop / p_original(target)
prediction_flipped = predicted_label_masked != target
```

Fairness constraints:

- Use the same audios and classifier for every method.
- Use the original predicted class as the target.
- Preserve waveform length with silence replacement.
- Match actual union duration, including overlap handling.
- Use the same `X` for SpeechXAI, LeGrad, random, and optional bottom
  conditions.

## Research Question

Which explanation method is more faithful to the classifier's
prediction under equal-duration deletion: SpeechXAI or the
LeGrad-inspired temporal attribution method?

## Required Limitations

- The proposed method is LeGrad-inspired, not an exact reproduction of
  vision LeGrad.
- Silence masking can create out-of-distribution audio.
- SpeechXAI and LeGrad use different explanation units.
- WhisperX timestamps are estimated alignments.
- SpeechXAI's released repository expands masks by 100 ms before and
  40 ms after each word, although the paper does not document this
  padding.
- IEMOCAP provides emotion labels rather than ground-truth
  explanations, so evaluation measures model faithfulness rather than
  human explanation correctness.
