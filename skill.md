You are my research/coding agent for a course project on speech explainability.

Project title:
LeGrad-Inspired Temporal Explanations for Speech Emotion Recognition on IEMOCAP, Compared with SpeechXAI

General objective:
Build an experimental pipeline that compares two explanation methods for a pretrained speech emotion recognition model on IEMOCAP:

1. SpeechXAI explanations
2. A LeGrad-inspired temporal attribution method for wav2vec2/Transformer speech models

The final evaluation should compare both methods using the same deletion/occlusion protocol: selected audio regions are replaced by silence while preserving the original audio length, and we measure the drop in classifier confidence for the original predicted class.

Important behavior:
For each project step, create a separate Jupyter notebook. Each notebook must be experimental and self-contained enough to test partial progress. Do not only write final scripts. I want to see intermediate outputs, sanity checks, plots, tensor shapes, example predictions, and small tests before moving to the next step.

When a notebook is created or modified for a project step, run that notebook before considering the step complete. Also run the relevant evaluation, validation, or smoke test for that step. If the notebook cannot be run because of a concrete blocker such as missing data, missing dependencies, or unavailable credentials, explicitly report the blocker and the exact command/check that failed.

Use simple, clear, incremental development. Prefer working minimal versions over complex abstractions. When there are multiple possible solutions, choose the easiest robust one.

Write explanations positively and directly. Describe what a method, result, or component is and what it does. Avoid contrastive phrasing such as "it is not X" when a direct description is sufficient.

Dataset:
Use IEMOCAP. If the full dataset path is not yet configured, write the code so the path can be easily changed in one variable at the top of the notebook.

Model:
Use the same pretrained emotion classifier as the Pastor et al. SpeechXAI paper:
superb/wav2vec2-base-superb-er

Load it with Hugging Face `Wav2Vec2ForSequenceClassification`. Keep this classifier fixed for SpeechXAI, LeGrad-inspired explanations, masking, and evaluation.

Do not continue blindly if attention gradients are impossible. First create a diagnostic notebook proving whether attentions and gradients are accessible.

Code style:
Use concise, readable code.
Use explicit variable names. Do not use names like df; use names like predictions_table, explanations_table, deletion_results.
Do not create unnecessary copies of datasets just to add columns.
For notebooks, import modules normally, for example:
import torch
import pandas as pd
import torchaudio
not from module import function unless it is clearly better.
Save intermediate CSV files when useful.
Each notebook should end with a short “Current status” markdown cell explaining what works, what failed, and what file was generated.

Expected notebook sequence:

Notebook 01 — Model inference on IEMOCAP
Goal:
Load a few IEMOCAP audio samples and run inference using the pretrained SUPERB wav2vec2-IEMOCAP emotion model used by Pastor et al.

Tasks:
- Define dataset/audio path variables.
- Load one audio file.
- Resample if needed.
- Run the pretrained model.
- Print predicted emotion and confidence scores.
- Run the same process for a small batch of audios.
- Save predictions to predictions.csv.

Show:
- Audio duration.
- Waveform plot for one sample.
- Predicted class.
- Confidence vector.
- Small table with audio_id, true_label if available, predicted_label, predicted_confidence.

Success criterion:
The classifier runs on at least a few IEMOCAP audios and produces predictions.

Notebook 02 — Attention access test
Goal:
Check whether the model can return internal Transformer attention maps.

Tasks:
- Try to run the model with output_attentions=True or equivalent.
- Inspect the internal model architecture.
- Print all relevant modules.
- Identify where the wav2vec2 encoder lives.
- Extract attention maps if possible.
- Print attention tensor shapes.

Show:
- Number of Transformer layers.
- Number of heads.
- Attention shape, ideally [batch, heads, tokens, tokens].
- A small visualization of one attention matrix from one head/layer.

Success criterion:
We can access attention maps from the speech Transformer.

If this fails:
- Explain why.
- Try fallback access through the underlying Hugging Face model.
- Document the fallback clearly.

Notebook 03 — Attention gradient test
Goal:
Check whether gradients with respect to attention maps can be computed.

Tasks:
- Run a forward pass.
- Select the score/logit/probability for the original predicted class.
- Backpropagate from that score.
- Capture gradients with respect to attention maps.
- Print gradient tensor shapes.
- Check whether gradients are nonzero.
- Plot or summarize gradient values.

Show:
- Predicted class.
- Selected class score.
- Attention tensor shape.
- Gradient tensor shape.
- Min, max, mean, and nonzero percentage of gradients.

Success criterion:
We can obtain ∂ class_score / ∂ attention_matrix for at least one layer.

If this fails:
- Try hooks.
- Try using logits instead of probabilities.
- Try disabling no_grad.
- Try setting model.train() only if needed for gradients, but avoid dropout effects if possible.
- Document the final working method or the reason it failed.

Notebook 04 — LeGrad-inspired temporal explanation
Goal:
Implement a simple LeGrad-inspired relevance curve over audio time.

Use this simple formulation:
For each layer l:
A_l = attention map
G_l = gradient of class score with respect to A_l
R_l = mean_over_heads(ReLU(G_l) * A_l)

Then:
R = mean_over_layers(R_l)

Convert token-token relevance into token relevance using:
token_relevance = sum over one attention dimension

Then map token relevance to time in seconds.

Tasks:
- Implement the relevance calculation.
- Normalize relevance scores.
- Map token indices to time intervals.
- Plot waveform with relevance over time.
- Extract top relevant intervals that can be selected later under a SpeechXAI-derived duration X.

Show:
- Token relevance curve.
- Waveform plus relevance overlay.
- Top intervals table with start, end, score.
- Sanity check: relevance values are not all zero or all equal.

Output:
Save legrad_explanations.csv with:
audio_id, start, end, score

Success criterion:
For one audio, produce a ranked list of relevant time intervals.

Notebook 05 — SpeechXAI explanations
Goal:
Run SpeechXAI on the same audios and extract segment-level explanations.

Tasks:
- Install/import/use the SpeechXAI codebase.
- Run SpeechXAI on the same model and same audio samples, if possible.
- Extract word-level or segment-level importance.
- Convert outputs to time intervals in seconds.
- Save results in the same format as LeGrad.

Use only the easiest main output for quantitative comparison:
word-level/audio-segment importance.

Do not make paralinguistic features the main evaluation unless the segment-level pipeline already works.

Show:
- Example SpeechXAI explanation.
- Table of segments with start, end, score.
- Plot waveform with SpeechXAI important intervals.

Output:
Save speechxai_explanations.csv with:
audio_id, start, end, score

Success criterion:
SpeechXAI produces ranked intervals for the same audios used by the LeGrad-like method.

Notebook 06 - Unified interval representation and masking
Goal:
Convert all explanation methods into a common time-interval format and implement silence masking using SpeechXAI top-k words as the shared duration source.

Tasks:
- Load legrad_explanations.csv.
- Load speechxai_explanations.csv.
- Use top-k SpeechXAI words to define the masking duration X.
- Use k values:
  1, 2, 3, 5
- For each audio and each k, select the top-k SpeechXAI word intervals.
- Compute X as the total duration of those SpeechXAI intervals.
- Select LeGrad top bins until their total duration is X.
- Select random bins until their total duration is X.
- Repeat random selection multiple times, for example 20 random trials.
- Optional: implement LeGrad-bottom using the same X.
- Implement silence masking while preserving audio length.

Important:
SpeechXAI determines the shared duration X. For a given audio and k, all compared methods must mask the same total duration X derived from the SpeechXAI top-k words.

Show:
- Original waveform.
- Masked waveform for SpeechXAI top-k.
- Masked waveform for LeGrad top bins with the same X.
- Masked waveform for random bins with the same X.
- Print total masked duration for each method to verify fairness.

Success criterion:
Given one audio and one k, the notebook creates comparable masked versions for:
- SpeechXAI-top
- LeGrad-top
- Random
- Optional LeGrad-bottom

Notebook 07 - Deletion/occlusion evaluation
Goal:
Evaluate faithfulness using confidence drop under the corrected SpeechXAI-top-k duration protocol.

For each audio:
1. Run original classifier.
2. Store original predicted class y_hat.
3. Store original confidence p_original(y_hat).
4. Run SpeechXAI and LeGrad-like explanations.
5. For each k in {1, 2, 3, 5}, select the top-k SpeechXAI words.
6. Compute X = total duration of those top-k SpeechXAI word intervals.
7. Mask those SpeechXAI intervals and evaluate confidence drop.
8. Select LeGrad top bins whose total duration is X and evaluate confidence drop.
9. Select random bins whose total duration is X, repeat multiple random trials, and evaluate confidence drop.
10. Measure:
drop = p_original(y_hat) - p_masked(y_hat)
relative_drop = drop / p_original(y_hat)
prediction_flipped = whether predicted label changed

Conditions:
- SpeechXAI-top
- LeGrad-top
- Random
- Optional LeGrad-bottom

Top-k values:
- 1
- 2
- 3
- 5

Output:
Save deletion_results.csv with:
audio_id, k, method, masked_duration, original_label, original_confidence, masked_confidence, drop, relative_drop, prediction_flipped, random_trial

Show:
- Results for one audio.
- Mean confidence drop table by k.
- Mean masked duration by k.
- Boxplot of drops by method.
- Deletion curve: k vs mean confidence drop.
- Flip rate table.

Success criterion:
We can compare whether SpeechXAI or LeGrad-like explanations cause larger confidence drops than repeated random masking under the same SpeechXAI-derived masked duration.

Notebook 08 — Qualitative examples
Goal:
Produce visual examples for the final report.

Tasks:
Choose 3 to 5 representative audios:
- One where SpeechXAI and LeGrad agree.
- One where LeGrad performs better in deletion.
- One where SpeechXAI performs better in deletion.
- One where both fail or random is competitive, if such case exists.

For each example show:
- Waveform.
- SpeechXAI highlighted intervals.
- LeGrad relevance curve.
- Top masked regions.
- Original confidence.
- Confidence after each masking condition.

Show concise interpretation for each example.

Success criterion:
The notebook produces report-ready figures and example explanations.

Notebook 09 — Final results summary
Goal:
Aggregate all results and generate final tables/plots.

Tasks:
- Load deletion_results.csv.
- Compute mean and standard deviation of confidence drop by method and k.
- Compute relative drop.
- Compute flip rate.
- Optionally compute class-wise results.
- Generate final plots.

Final tables:
1. Mean confidence drop by method and k.
2. Relative confidence drop by method and k.
3. Prediction flip rate by method.
4. Optional class-wise performance.

Final plots:
1. Deletion curve.
2. Boxplot of confidence drops.
3. Barplot of flip rates.
4. Example visualizations from Notebook 08.

Success criterion:
This notebook contains the main evidence needed for the final report.

Evaluation protocol:
Use the same audios for all methods.
Use the same classifier for all evaluations.
Use the same SpeechXAI-derived masked duration X for all methods for a given audio and k.
Use silence replacement while preserving original audio length.
Measure confidence drop for the original predicted class, not necessarily the new predicted class.
Random intervals must have the same total duration X as the SpeechXAI top-k word intervals.
Optional bottom-relevance intervals must also use the same duration X.

Main research question:
Which explanation method is more faithful to the classifier’s prediction under deletion-based evaluation: SpeechXAI or the LeGrad-inspired temporal attribution method?

Expected final claim:
We implemented a LeGrad-inspired temporal attribution method for wav2vec2-based speech emotion recognition and compared it with SpeechXAI on IEMOCAP. SpeechXAI word explanations and LeGrad time-bin explanations were evaluated under equal-duration silence masking, where SpeechXAI top-k words define the shared masked duration X. A faithful explanation should produce a larger confidence drop than repeated random masking with the same X.

Important limitations to mention:
- This is LeGrad-inspired, not an exact reproduction of LeGrad.
- Silence masking may introduce out-of-distribution artifacts.
- SpeechXAI and LeGrad explain different aspects of the model: SpeechXAI is more human-interpretable, while LeGrad is more model-internal.
- IEMOCAP labels are emotion labels, not ground-truth explanation labels, so evaluation is based on model faithfulness rather than human explanation ground truth.

Development rule:
Do not skip diagnostic notebooks. The early notebooks must prove:
1. The model works.
2. Attention maps are accessible.
3. Attention gradients are accessible.
Only after these tests should the LeGrad-like method be implemented.

At the end of each notebook, include:
- What was tested.
- What worked.
- What failed, if anything.
- What files were generated.
- What the next notebook should do.
