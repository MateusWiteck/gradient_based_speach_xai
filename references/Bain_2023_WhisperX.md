# WhisperX: Word-Level Timestamp Alignment

## Reference

Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. "WhisperX:
Time-Accurate Speech Transcription of Long-Form Audio." Interspeech
2023.

- [Interspeech paper](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.html)
- [PDF](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)
- [arXiv](https://arxiv.org/abs/2303.00747)
- [Repository](https://github.com/m-bain/whisperX)

## Relevance to This Project

SpeechXAI uses WhisperX to obtain the word text and timestamps that
define its interpretable audio regions. In this project, those regions
connect three objects:

```text
recognized word
<-> lexical audio interval [start, end]
<-> SpeechXAI attribution score
```

Timestamp quality therefore affects both interpretation and
evaluation. A boundary error changes which samples are silenced and
can assign the resulting confidence change to the wrong word.

## WhisperX Pipeline

WhisperX augments Whisper with three stages designed for long-form,
time-accurate transcription.

### 1. Voice activity detection

A voice activity detection (VAD) model identifies regions containing
speech. WhisperX uses these regions to:

- avoid processing long inactive periods;
- place chunk boundaries in regions with minimal speech activity;
- constrain later alignment to local audio segments;
- reduce dependence on Whisper's own timestamp tokens.

The VAD output is a sequence of active speech intervals with start and
end indexes. These are broader speech regions, not final word
boundaries.

### 2. Cut and merge

Active speech regions can be too long or too short for Whisper.
WhisperX cuts long regions near minima in voice activity and merges
neighboring short regions up to a target duration. The paper uses
approximately 30-second chunks, matching Whisper's training input
duration.

The resulting chunks can be transcribed independently in batches. This
improves throughput and limits timestamp drift across long recordings.

### 3. Whisper transcription

Whisper decodes each prepared audio chunk into text. WhisperX performs
this stage without relying on Whisper's timestamp decoding. The text
becomes the transcript that the alignment stage must place on the
audio timeline.

This distinction matters: Whisper generates the recognized words, but
the external alignment model generates the word boundaries used by
SpeechXAI.

### 4. Forced phoneme alignment

For an audio segment and its transcript, WhisperX uses an external
phoneme-recognition model, implemented with wav2vec 2.0 in the paper.
The process is:

1. determine the transcript symbols supported by the phoneme model;
2. run the phoneme model over the audio segment to obtain frame-level
   logits;
3. use Dynamic Time Warping to find an optimal temporal path through
   the transcript phonemes;
4. assign each word's start from its first aligned phoneme and its end
   from its last aligned phoneme.

The output used by SpeechXAI contains entries such as:

```python
{
    "word": "problem?",
    "start": 1.943,
    "end": 2.366,
    "score": 0.854,
}
```

The score describes alignment confidence. It is separate from the
emotion classifier and separate from the SpeechXAI attribution score.

## Does WhisperX Guarantee Non-Overlapping Words?

### Short answer

**The paper does not explicitly guarantee that every returned pair of
word intervals is non-overlapping.**

### Why intervals are normally non-overlapping

Within one successfully aligned segment, the forced-alignment path is
monotonic in time. Transcript phonemes are aligned in their original
order, and each word receives the span from its first to its last
phoneme. Under this construction, consecutive lexical word intervals
are expected to be chronologically ordered and non-overlapping. They
can have a gap between them or meet at a boundary.

The current WhisperX implementation follows the same principle. It
constructs ordered character segments from the alignment path, then
computes each word's start as the minimum character start and its end
as the maximum character end.

### Why this is not a formal guarantee

The paper:

- presents non-overlap as neither a theorem nor an explicit API
  invariant;
- evaluates timestamp quality using word-segmentation precision and
  recall with a 200 ms collar, rather than measuring overlap
  violations;
- assigns unsupported transcript phonemes the timestamp of the nearest
  following phoneme;
- depends on VAD segmentation, transcription correctness, phoneme-model
  coverage, and alignment success.

These details mean that downstream code should validate interval
ordering instead of assuming it solely from the paper.

Pastor et al. describe their resulting word segments as
non-overlapping, which is consistent with the normal WhisperX output.
That statement should be treated as an observed/required property of
the segments used by SpeechXAI, not as a formal guarantee established
by the WhisperX publication.

### Project policy

Notebook 01 validates that:

```text
start >= 0
end > start
word starts are chronologically ordered
end <= audio duration
```

It should additionally retain an explicit lexical-overlap check:

```text
end_i <= start_(i+1)
```

If that condition fails, the audio should be flagged for alignment
review before its attribution is included in the quantitative
evaluation.

The SpeechXAI removal masks are a separate issue. Even when WhisperX
lexical intervals do not overlap, the released SpeechXAI code expands
each mask by 100 ms before and 40 ms after the word. Those effective
removal regions can and do overlap. Notebook 01 visualizes that padding
overlap and measures deletion duration using the union of the masks.

## Evaluation in the Paper

WhisperX evaluates:

- transcription quality with word error rate;
- repetition and hallucination behavior;
- transcription speed;
- word-segmentation precision and recall.

A predicted word is considered correct for segmentation when its text
matches and its predicted interval overlaps the ground-truth word
within a 200 ms collar. On AMI and Switchboard, WhisperX substantially
improves word-segmentation precision and recall over timestamps derived
directly from Whisper.

These metrics demonstrate improved timestamp accuracy at dataset
level. They do not certify every individual boundary, so alignment
scores and interval validation remain important for SpeechXAI.

## Methodological Decisions for This Project

- Use SpeechXAI's `transcribe_audio`, which invokes WhisperX, rather
  than estimating word positions from transcript length.
- Call transcription once and pass the same word dictionaries to the
  explainer and visualization.
- Preserve the original lexical `[start, end]` interval separately
  from SpeechXAI's padded removal interval.
- Retain WhisperX alignment confidence as diagnostic metadata.
- Validate chronological order, bounds, positive duration, and lexical
  non-overlap before accepting an explanation.
- Treat WhisperX timestamps as model estimates with measurable
  uncertainty, not ground-truth annotations.
- Equalize evaluation budgets using the actual union duration of
  SpeechXAI removal masks.

## Limitations

- Transcription errors can remove, insert, or substitute words before
  alignment begins.
- Alignment quality depends on the language-specific phoneme model.
- Unsupported symbols and failed alignments require fallback behavior.
- VAD and chunk boundaries can affect transcription and alignment.
- A high alignment score does not prove that the boundary is exact.
- The paper's 200 ms evaluation collar permits timing deviations that
  may still matter for short-word perturbation experiments.

