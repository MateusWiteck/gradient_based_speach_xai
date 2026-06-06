# IEMOCAP: Dataset Summary

## Reference

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily
Mower, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S.
Narayanan. "IEMOCAP: Interactive Emotional Dyadic Motion Capture
Database." *Language Resources and Evaluation*, 2008.

- [Dataset website](https://sail.usc.edu/iemocap/)
- [Paper](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)

## Dataset Summary

IEMOCAP is a multimodal emotional-expression corpus recorded at the USC
Signal Analysis and Interpretation Laboratory. It contains approximately
12 hours of dyadic interaction from ten professional actors, organized
as five sessions with one female and one male actor per session.

The actors performed both scripted scenes and improvised scenarios.
Improvised scenarios were designed to elicit happiness, anger, sadness,
frustration, and neutral behavior, while the scripted material produced
a wider and sometimes more ambiguous range of emotions.

The complete corpus includes:

- audio and video recordings;
- utterance-level transcripts and time boundaries;
- categorical emotion annotations;
- dimensional annotations for valence, activation, and dominance;
- facial, head, and hand motion-capture information.

The paper reports 10,039 segmented conversational turns: 5,255 scripted
and 4,784 spontaneous. The average turn lasts approximately 4.5 seconds
and contains 11.4 words.

Categorical labels include anger, sadness, happiness, disgust, fear,
surprise, frustration, excitement, neutral, and other. Three evaluators
assessed each utterance, and the released labels preserve the
subjectivity and ambiguity of emotional perception.

## Use in This Project

This project uses IEMOCAP as the source of speech samples, transcripts,
and emotion labels. The selected SUPERB classifier used by Pastor et al.
uses a four-class mapping:

- anger;
- happiness/excitement;
- neutral;
- sadness.

Other IEMOCAP categories therefore need to be excluded or mapped
consistently when model accuracy is evaluated. For explanation
faithfulness, the project can evaluate the model's original predicted
class without requiring the ground-truth label to belong to this
four-class subset.

IEMOCAP supplies utterance-level timestamps and transcripts, but the
standard release does not provide a reliable word boundary for every
word. SpeechXAI therefore requires a separate word-alignment stage.

## Main Cautions

- The corpus contains only ten actors, so speaker-independent splitting
  is important.
- Emotion labels are subjective and sometimes ambiguous.
- Scripted and improvised speech have different distributions.
- Many studies use different class-merging and filtering rules, so
  results are comparable only when the label protocol is stated.
