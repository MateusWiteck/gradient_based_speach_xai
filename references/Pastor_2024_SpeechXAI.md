# SpeechXAI: Method and Evaluation

## Reference

Eliana Pastor, Alkis Koudounas, Giuseppe Attanasio, Dirk Hovy, and
Elena Baralis. "Explaining Speech Classification Models via Word-Level
Audio Segments and Paralinguistic Features." EACL 2024.

- [Paper](https://aclanthology.org/2024.eacl-long.136/)
- [Repository](https://github.com/elianap/SpeechXAI)

## Objective

SpeechXAI explains a speech classifier using concepts that people can
inspect directly:

1. words and their corresponding audio intervals;
2. paralinguistic properties such as pitch, speaking rate, noise, and
   reverberation.

The method is model-agnostic. It does not require gradients or access to
internal model layers. It repeatedly modifies the waveform and measures
how the classifier output changes.

## Word-Level Explanation Method

### 1. Obtain word intervals

SpeechXAI begins with a transcript and word-level timestamps. When gold
timestamps are unavailable, the paper uses WhisperX to transcribe and
align the audio.

For an utterance, alignment produces non-overlapping segments:

```text
S = {(word_i, start_i, end_i)} for i = 1, ..., m
```

These intervals are the interpretable input features. Pauses and other
regions without aligned words are excluded from the word explanation.

Alignment quality is therefore part of explanation quality. Incorrect
transcription or boundaries can assign a perturbation effect to the
wrong word.

### 2. Perturb word segments

To mask a word, all waveform samples between its start and end times
are set to zero. The segment remains in place, preserving:

- total waveform length;
- the absolute timing of later words;
- the number of model input samples.

The paper chooses silence replacement because physically deleting a
word would shorten the signal and introduce a separate duration shift.

#### What the paper says about masking boundaries

The paper describes masking at the level of the word segments produced
from the alignment timestamps:

- Section 2.1, **Segment contribution** (PDF page 3), says that one or
  more segments are masked by zeroing their corresponding samples in
  the time domain.
- Footnote 2 on the same page says that zeroing is preferred to
  removing segments because removal would introduce effects caused by
  shorter recordings.
- Appendix B.1, **Word-Level Attribution** (PDF page 13), again
  describes selective zeroing, or silencing, of time-domain audio
  segments. For LIME, it says that the audio is split according to the
  timestamps derived from WhisperX.

The paper does **not** specify additional temporal padding before or
after a word. Its methodological description therefore identifies the
WhisperX word interval `[start, end]` as the conceptual segment being
masked.

#### Additional padding in the released implementation

The released SpeechXAI repository applies a wider interval than the
paper explicitly documents. In
`speechxai/explainers/utils_removal.py`, both `remove_word` and
`remove_specified_words` define:

```python
a, b = 100, 40
```

The pydub slices and replacement duration then use:

```text
effective start = word start - 100 ms
effective end   = word end + 40 ms
```

Consequently, the implementation silences the aligned word together
with 100 ms of preceding context and 40 ms of following context. This
is repository behavior rather than a padding rule stated in the paper.
It matters when reproducing published code because:

- the perturbation used to compute an attribution is wider than the
  lexical WhisperX interval shown for the word;
- padded intervals from neighboring words can overlap;
- the true deletion budget must use the union of effective padded
  intervals rather than summing their durations;
- the upstream pydub implementation does not clamp a negative
  `start - 100 ms` boundary explicitly, so words beginning within the
  first 100 ms require special care.

For this project, visualizations distinguish the lexical WhisperX
interval from the effective padded mask. Quantitative masking follows
the released implementation's effective interval so that the plotted
perturbation and the attribution computation describe the same audio
region.

### 3. Define the target output

The paper explains the probability assigned to the model's predicted
class. If the original waveform is `x`, classifier `f` predicts class
`k`, and `f_k(x)` is the corresponding probability, the explanation
asks which components support that observed prediction.

This is a faithfulness target. It explains the model's behavior even
when the predicted label is wrong.

### 4. Leave-One-Out attribution

Leave-One-Out (L1O) masks one word interval at a time:

```text
r_i = f_k(x) - f_k(x without word segment i)
```

Interpretation:

- a large positive `r_i` means silencing the word reduces target-class
  confidence, so the segment supports the prediction;
- a score near zero means the model is locally insensitive to that
  word;
- a negative score means silencing the word increases target-class
  confidence, so the segment may oppose the prediction.

Advantages:

- direct and easy to interpret;
- deterministic for a fixed classifier and waveform;
- requires one perturbed inference per word;
- naturally produces a word, interval, and score table.

Limitation:

- each word is tested independently;
- interactions between words are not represented;
- the silent replacement may itself be out of distribution.

### 5. LIME attribution

SpeechXAI also adapts Local Interpretable Model-Agnostic Explanations
(LIME) to word-aligned audio.

Each perturbed neighbor is represented by a binary vector:

```text
z in {0, 1}^m
```

Each component indicates whether a word segment is present or masked.
SpeechXAI samples many such combinations, synthesizes their masked
waveforms, obtains classifier predictions, weights samples by proximity
to the original all-present utterance, and fits a sparse local surrogate
model:

```text
f_k(x_z) approximately beta_0 + sum_i beta_i z_i
```

The learned coefficient `beta_i` becomes the word attribution.

Unlike L1O, LIME can observe samples with several missing words and can
partly represent effects that only appear when words are perturbed
together. Its explanation depends on the random neighborhood, sample
count, distance kernel, and surrogate regularization, which introduces
instability.

The released implementation adapts the `Lime-For-Time` library. Instead
of equal-width temporal chunks, it uses variable-duration word
intervals and silence masking.

## Paralinguistic Explanation Method

SpeechXAI separately measures sensitivity to transformations of the
whole waveform. The paper considers:

- pitch shifting;
- time stretching;
- additive white noise at multiple signal-to-noise ratios;
- reverberation.

For a transformation family `p`, several transformed signals are
created. The effect of one transformation is:

```text
r_p(x_tilde) = f_k(x) - f_k(x_tilde)
```

The final attribution for the property is the mean effect across its
tested transformation levels:

```text
r(x, p) = mean over transformed signals of r_p(x_tilde)
```

A large absolute score means the classifier is sensitive to that
property. A positive value means the perturbations reduce target-class
confidence on average; a negative value means they increase it.

These scores describe sensitivity to a controlled transformation. They
do not establish that the classifier internally represents a pure,
independent concept such as pitch. Transformations may produce coupled
acoustic effects, and third-party signal-processing quality matters.

## Evaluation

### Tasks, datasets, and models

The method is evaluated on:

- Fluent Speech Commands for English intent classification;
- ITALIC for Italian intent classification;
- IEMOCAP for English emotion recognition.

The explained models include fine-tuned wav2vec 2.0 and multilingual
XLS-R checkpoints. For the IEMOCAP analysis, the paper uses Session 1,
containing 942 utterances.

WhisperX provides transcription and word alignment when needed. The
reported word error rate is 1.72 on Fluent Speech Commands, 15.77 on
IEMOCAP, and 7.49 on ITALIC. The higher IEMOCAP error illustrates why
alignment and transcription errors must be treated as an explanation
uncertainty.

### Quantitative faithfulness

The paper adapts **comprehensiveness** and **sufficiency** from
token-level NLP explanation evaluation.

For a selected set of highly relevant words:

- **comprehensiveness** measures how much target confidence falls when
  the selected word segments are removed;
- **sufficiency** measures how much confidence changes when only the
  selected word segments are retained.

These metrics are evaluated over increasing percentages of selected
words and summarized across budgets. The analysis compares attribution
methods and random selection. The paper reports that LIME generally
provides stronger faithfulness results than Leave-One-Out, motivating
the use of LIME explanations in the human study.

The protocol evaluates selected word counts or percentages. Because
words have different durations, equal word counts do not guarantee
equal amounts of perturbed audio. This is the main reason the current
project uses duration-matched deletion when comparing SpeechXAI with a
time-bin method.

### Qualitative and global analysis

The paper shows:

- instance-level word relevance tables and heatmaps;
- sensitivity to individual paralinguistic transformations;
- dataset-level averages for recurring words and transformation types;
- IEMOCAP examples where lexical and paralinguistic evidence jointly
  help interpret an emotion prediction.

These examples demonstrate how the method can reveal whether a model
relies on expected command words, speaking rate, pitch, or recording
conditions.

### Human plausibility study

The authors conduct a user study comparing visualization formats and
asking participants to rate whether explanations agree with human
reasoning. Participants inspect explanations from English and Italian
intent-classification tasks.

The study evaluates:

- ability to identify important words;
- ability to compare word importance;
- ability to inspect several explanations;
- overall visualization preference;
- explanation plausibility on a four-point scale.

This evaluates plausibility and usability, not model faithfulness.
Humans may prefer an explanation that sounds reasonable even when it
does not accurately describe the classifier.

## Main Findings

- Word-level segments are understandable explanation units for speech
  tasks with meaningful lexical content.
- LIME can outperform independent Leave-One-Out perturbations in the
  paper's faithfulness evaluation.
- Paralinguistic perturbations expose model sensitivity that word-only
  explanations cannot show.
- Human participants generally find the explanations plausible, though
  preferred visualization formats depend on the task.
- Emotion recognition benefits from combining semantic and
  paralinguistic analysis.

## Limitations

- Explanations depend on transcription and word-alignment quality.
- Silence masking creates an artificial acoustic condition.
- L1O misses interactions among words.
- LIME is stochastic and sensitive to its neighborhood sample size.
- Word explanations are less suitable for tasks such as speaker or
  language identification.
- Paralinguistic transformations may introduce artifacts.
- The paper's main quantitative faithfulness evaluation focuses on
  word-level explanations; paralinguistic faithfulness remains less
  standardized.

## Relevance to This Project

SpeechXAI establishes the word-level side of the comparison:

```text
word | start time | end time | importance
```

The project adopts these central choices:

- explain the original predicted class;
- preserve waveform length;
- mask selected intervals with silence;
- measure the resulting confidence change.

The project changes the comparison budget. For each `k`, the top
SpeechXAI words define `X` seconds of audio. A LeGrad-like method and a
random baseline must each mask the same duration `X`. This preserves
SpeechXAI's native word ranking while making it comparable with
variable-count temporal bins.
