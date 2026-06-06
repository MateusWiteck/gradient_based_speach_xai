# wav2vec 2.0: Architecture and Method

## Reference

Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli.
"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations." NeurIPS 2020.

- [Paper](https://arxiv.org/abs/2006.11477)
- [Original fairseq implementation](https://github.com/facebookresearch/fairseq)

## Central Idea

wav2vec 2.0 learns speech representations directly from unlabeled raw
audio and then adapts those representations to a labeled downstream
task. Its pretraining task resembles masked language modeling, but the
units are learned acoustic representations rather than written tokens.

The model:

1. converts the waveform into a shorter sequence of latent acoustic
   vectors;
2. masks spans in that latent sequence;
3. contextualizes the masked sequence with a Transformer;
4. discretizes the unmasked latent vectors into learned target units;
5. asks the contextual representation at each masked position to
   identify its correct quantized target among distractors.

This design combines continuous contextual inputs with discrete
prediction targets. The paper's ablations show that this combination is
more effective than feeding quantized vectors into the Transformer or
using continuous targets.

## Architecture

### 1. Raw waveform input

The input is a one-dimensional waveform:

```text
X = (x_1, x_2, ..., x_N)
```

The original model operates on 16 kHz audio. The waveform is normalized
before entering the feature encoder. No manually engineered
spectrogram, mel filter bank, or phonetic representation is required.

### 2. Convolutional feature encoder

A temporal convolutional network maps the waveform to latent speech
representations:

```text
f: X -> Z
Z = (z_1, z_2, ..., z_T)
```

Each convolutional block contains:

1. a one-dimensional temporal convolution;
2. normalization;
3. a GELU nonlinearity.

The architecture has seven convolutional blocks. Every block has 512
channels. Their kernel widths and strides are:

| Block | Kernel width | Stride |
| --- | ---: | ---: |
| 1 | 10 | 5 |
| 2 | 3 | 2 |
| 3 | 3 | 2 |
| 4 | 3 | 2 |
| 5 | 3 | 2 |
| 6 | 2 | 2 |
| 7 | 2 | 2 |

The combined stride is:

```text
5 * 2 * 2 * 2 * 2 * 2 * 2 = 320 samples
```

At 16 kHz, one output step occurs every 320 waveform samples, or about
20 ms. The output frequency is therefore approximately 49 to 50 latent
steps per second. Each latent vector has a receptive field of about 400
input samples, corresponding to roughly 25 ms of audio.

This distinction matters:

- **stride** describes the temporal spacing between neighboring latent
  vectors;
- **receptive field** describes how much waveform context contributes
  to one latent vector.

Neighboring latent vectors overlap in their waveform support because
the receptive field is wider than the stride.

The feature encoder performs learned downsampling. A several-second
waveform containing tens of thousands of samples becomes a manageable
sequence containing approximately 50 vectors per second.

### 3. Feature projection

The convolutional output is normalized and projected to the
Transformer's model dimension. Implementations may expose this as a
separate feature-projection module. This stage makes the encoder output
compatible with the hidden size expected by the context network.

### 4. Convolutional positional representation

Self-attention alone does not encode sequence order. wav2vec 2.0 adds a
learned convolutional positional representation to the latent sequence.
The paper uses a grouped convolution with:

- kernel size 128;
- 16 groups;
- GELU activation.

This differs from a table of absolute position embeddings. The
convolution describes local relative position and can naturally operate
on variable-length speech sequences.

### 5. Transformer context network

The context network maps latent speech features to contextualized
representations:

```text
g: Z -> C
C = (c_1, c_2, ..., c_T)
```

Each output `c_t` incorporates information from the entire utterance
through multi-head self-attention. The sequence length remains at the
approximately 20 ms latent resolution.

The paper presents two main configurations:

| Configuration | Transformer blocks | Hidden size | FFN size | Heads |
| --- | ---: | ---: | ---: | ---: |
| BASE | 12 | 768 | 3,072 | 8 |
| LARGE | 24 | 1,024 | 4,096 | 16 |

Each Transformer block contains multi-head self-attention, a
position-wise feed-forward network, residual connections, and
normalization. In simplified form:

```text
H'_l = H_(l-1) + MultiHeadAttention(H_(l-1))
H_l  = H'_l + FeedForward(H'_l)
```

For a head, self-attention is based on:

```text
A = softmax(Q K^T / sqrt(d_k))
O = A V
```

`A` is a token-to-token attention matrix. In wav2vec 2.0, its rows and
columns refer to temporal latent positions rather than words. These
attention tensors are the planned intermediate representation for the
LeGrad-inspired explanation in this project.

### 6. Quantization module

The quantizer creates discrete targets from convolutional latent
representations. It participates in self-supervised pretraining and is
not the sequence passed into the Transformer.

The model uses product quantization:

1. divide the code representation into `G` groups;
2. maintain a codebook of `V` entries for each group;
3. choose one entry from every group;
4. concatenate the selected entries;
5. apply a learned linear transformation to form `q_t`.

The paper uses:

- `G = 2` codebook groups;
- `V = 320` entries per group;
- up to `320^2 = 102,400` possible combined codewords.

Selections are made with a hard Gumbel-softmax during the forward pass.
A straight-through estimator allows gradients to train the encoder and
codebooks. The Gumbel temperature is gradually reduced during
pretraining, making assignments increasingly discrete.

The architectural separation is essential:

```text
continuous z_t -> masking -> Transformer -> contextual c_t
continuous z_t -> quantizer -> discrete target q_t
```

The Transformer receives rich continuous inputs. The contrastive task
uses discrete targets that suppress some low-level details and encourage
the model to recognize reusable acoustic units.

## Self-Supervised Pretraining

### Latent masking

Masking occurs after the convolutional feature encoder and before the
Transformer. Random latent positions are selected as starting points,
and a span of `M` consecutive positions is replaced with a learned mask
vector. Overlapping spans are allowed.

The main setup uses:

- start probability `p = 0.065`;
- span length `M = 10` latent steps.

Because spans overlap, this masks roughly 49% of latent positions. A
ten-step span covers about 200 ms by stride, while the paper reports an
average merged masked span of approximately 299 ms.

Masking latent vectors has two advantages:

- the Transformer must reconstruct contextual information instead of
  copying local acoustic detail;
- training is far cheaper than applying attention directly to every raw
  waveform sample.

The quantizer still sees the unmasked latent vector at the target
position, allowing it to produce the correct target `q_t`.

### Contrastive objective

For a masked position `t`, the Transformer output `c_t` must identify
the true quantized target `q_t` among `K` distractors sampled from other
masked positions in the same utterance.

The similarity is cosine similarity with temperature `kappa`:

```text
L_m(t) = -log(
    exp(sim(c_t, q_t) / kappa)
    /
    sum_q exp(sim(c_t, q) / kappa)
)
```

The paper uses `K = 100` distractors and `kappa = 0.1`.

Sampling distractors from the same utterance makes the task harder:
speaker identity and recording conditions cannot trivially distinguish
the positive target from negatives.

### Codebook diversity objective

Without regularization, the quantizer could collapse onto a small number
of codewords. A diversity loss encourages approximately uniform use of
entries in each codebook.

The total pretraining objective is:

```text
L = L_m + alpha * L_d
```

where `L_m` is the contrastive loss, `L_d` is the codebook diversity
loss, and the paper uses `alpha = 0.1`.

## Fine-Tuning

After pretraining, wav2vec 2.0 can be adapted to a labeled task by
placing a task-specific prediction head above the contextual
representations.

In the paper's speech-recognition experiments:

- the head predicts character or phoneme classes;
- training uses Connectionist Temporal Classification (CTC);
- the convolutional feature encoder remains frozen;
- initially only the output head is trained;
- the Transformer is unfrozen later;
- time and channel masking provide additional regularization.

Speech emotion recognition uses a different head and aggregation
strategy. A classifier must convert the sequence of contextual vectors
into one utterance-level representation and then produce emotion
logits. The SUPERB checkpoint used by this project learns a weighted
combination of wav2vec 2.0 representation levels followed by a
projection and four-class classification head. This downstream head is
separate from the original wav2vec 2.0 pretraining architecture.

## Evaluation in the Paper

The paper evaluates how effectively unlabeled pretraining reduces the
need for transcribed speech:

- pretraining data: LibriSpeech 960 hours or LibriVox 53,200 hours;
- labeled fine-tuning: 10 minutes, 1 hour, 10 hours, 100 hours, or the
  full 960 hours of LibriSpeech;
- primary ASR metric: word error rate;
- additional phoneme-recognition evaluation: TIMIT.

The strongest model pretrained on 53,200 hours and fine-tuned with only
10 minutes of labeled speech achieves 4.8/8.2 WER on LibriSpeech
test-clean/test-other with a language model. The results demonstrate
label efficiency, although language-model decoding contributes
substantially in the smallest labeled-data settings.

The ablations support several design choices:

- continuous Transformer inputs with quantized contrastive targets;
- nontrivial contiguous masking spans;
- a diversity penalty that maintains codebook use;
- same-utterance negative sampling;
- Gumbel noise for trainable discrete selection.

## Relevance to This Project

The emotion classifier being explained uses a wav2vec 2.0 encoder. The
parts most relevant to temporal explanation are:

- the feature encoder maps waveform samples to approximately 20 ms
  latent steps;
- Transformer tokens retain temporal order;
- each self-attention layer produces head-specific token-to-token
  attention matrices;
- gradients of an emotion score can potentially be computed with
  respect to those attention matrices;
- token relevance must be mapped back through the convolutional stride
  and receptive field to waveform time.

The quantizer is mainly a pretraining mechanism. It is generally absent
from the downstream inference path used for emotion classification, so
the planned LeGrad-inspired method should focus on contextual
Transformer tokens and attention maps.

## Practical Interpretation

wav2vec 2.0 tokens are acoustic time steps, not words. A high relevance
score for token `t` means a short temporal neighborhood contributes to
the emotion decision. Word-level SpeechXAI explanations therefore
cannot be compared to wav2vec 2.0 token relevance by equal token or
segment count. This project instead equalizes the total duration of
masked audio.
