# LeGrad: Method and Evaluation

## Reference

Walid Bousselham, Angie Boggust, Sofian Chaybouti, Hendrik Strobelt,
and Hilde Kuehne. "LeGrad: An Explainability Method for Vision
Transformers via Feature Formation Sensitivity." 2024.

- [Paper](https://arxiv.org/abs/2404.03214)
- [Repository](https://github.com/WalBouss/LeGrad)

## Objective

LeGrad explains a Vision Transformer by measuring how sensitive a
class score is to each self-attention map. It treats the gradient with
respect to attention as the explanation signal and aggregates that
signal across rows, heads, and Transformer layers.

The method is designed around two observations:

1. information is formed progressively across Transformer layers;
2. attention weights alone do not indicate whether changing an
   attention connection would affect the selected prediction.

LeGrad therefore uses class-specific attention gradients and
intermediate representations rather than only the last layer.

## Transformer Background

A Vision Transformer represents an image as patch tokens plus an
optional classification token:

```text
Z^0 = {z_cls, z_1, ..., z_n}
```

At layer `l`, multi-head self-attention produces one attention matrix
per head:

```text
A^l in R^(h x (n+1) x (n+1))
```

Each row describes how one query token combines information from all
key tokens. Layers then refine the token sequence through attention,
feed-forward networks, and residual connections.

The final image representation may be obtained from:

- the classification token;
- an average over tokens;
- a separate attentional pooler.

LeGrad is formulated to accommodate these feature-aggregation choices.

## LeGrad Method

### 1. Construct a class score at every layer

Let:

```text
Z^l = {z_0^l, z_1^l, ..., z_n^l}
```

be the output tokens from layer `l`. The paper first builds an
intermediate feature representation, such as the mean of all tokens:

```text
z_bar^l = (1 / (n + 1)) * sum_i z_i^l
```

A classifier or text embedding maps that feature to class scores:

```text
y_bar^l = z_bar^l C
```

For target class `c`, the scalar activation is:

```text
s^l = y_bar_c^l
```

Using an intermediate score for every layer allows the method to ask
how attention at that stage contributes to the class representation
formed at the same stage.

This differs from computing only a final output score and propagating
it through all layers. The layerwise construction makes relevance maps
more directly comparable and allows simple averaging.

### 2. Differentiate the score with respect to attention

For each layer:

```text
G^l = partial s^l / partial A^l
```

`G^l` has the same shape as the multi-head attention tensor:

```text
h x (n + 1) x (n + 1)
```

Each element measures the local sensitivity of the selected class score
to one attention probability. A large positive derivative means that a
small increase in that attention connection would increase the target
activation locally.

### 3. Keep positive sensitivity

LeGrad applies a ReLU:

```text
G_pos^l = ReLU(G^l)
```

Negative derivatives are discarded because the explanation is intended
to show positive support for the target score. The paper's ablation
shows that this choice improves perturbation performance.

### 4. Reduce query rows and heads

For each key token `j`, LeGrad averages the positive gradients over all
query positions `i` and heads:

```text
E_hat_j^l =
    (1 / (h * (n + 1)))
    * sum_over_heads_and_queries G_pos^l[head, i, j]
```

The result is a vector with one relevance value per token. This
reduction asks:

> Across all attention heads and all tokens gathering information,
> how sensitive is the target score to attention directed toward token
> `j`?

For a ViT, the classification-token entry is removed and the remaining
patch values are reshaped into the original patch grid.

### 5. Aggregate layers

The layerwise relevance vectors are averaged:

```text
E_bar = (1 / L) * sum_l E_hat^l
```

The merged vector is reshaped and min-max normalized to produce the
final heatmap.

Layer aggregation is central to LeGrad. Earlier layers can encode local
features while later layers contain increasingly class-specific and
global representations. The paper finds that larger models generally
benefit from including more layers, although performance eventually
plateaus.

### 6. Attentional-pooler adaptation

For models such as SigLIP, a learnable query attends to the final patch
tokens. LeGrad applies the attentional pooler to intermediate token
representations, derives a target activation from the pooled feature,
and differentiates with respect to the pooler's attention map.

This shows that the method's essential requirement is a differentiable
attention map associated with feature aggregation. It does not strictly
require a classification token.

## Important Methodological Clarification

Original LeGrad uses:

```text
ReLU(partial score / partial attention)
```

as the relevance signal. It does **not** define its core map as:

```text
attention * gradient
```

Gradient-weighted attention is closer to AttentionCAM or CheferCAM.
Any speech implementation that multiplies attention values by their
gradients should be described as a LeGrad-inspired variant and
evaluated against the gradient-only formulation.

## Evaluation

### Object segmentation

The paper converts explanation heatmaps into binary foreground masks
using a threshold of 0.5. On ImageNet-Segmentation, explanations are
evaluated using:

- mean Intersection over Union;
- pixel accuracy;
- mean average precision.

LeGrad is compared with LRP variants, attention rollout, raw attention,
GradCAM, CheferCAM, and TextSpan. It obtains the strongest reported
localization results in the main table.

This test primarily evaluates spatial alignment with annotated objects.
It is a localization/plausibility proxy rather than direct causal
faithfulness.

### Open-vocabulary localization

For vision-language models, arbitrary text prompts define the target
score. The OpenImagesV7 validation set supplies positive and negative
point annotations across thousands of object classes.

The metric is point-based mean IoU. LeGrad substantially improves over
the compared methods in this setting, suggesting that its maps preserve
fine-grained prompt-specific localization.

### Audio-prompted visual localization

The paper applies LeGrad to the image encoder of ImageBind. An audio or
speech prompt defines a target in the shared embedding space, and
LeGrad identifies image regions associated with that prompt.

ADE20K Sound-Prompted and Speech-Prompted datasets provide
segmentation evaluation. This experiment demonstrates multimodal use,
but LeGrad still explains visual Transformer patches. It does not
produce temporal explanations over an audio encoder.

### Perturbation-based faithfulness

The paper performs complementary erasure tests on ImageNet:

- **positive perturbation:** remove regions from most relevant to least
  relevant;
- **negative perturbation:** remove regions from least relevant to most
  relevant.

Model accuracy is tracked while erasing from 0% to 90% of image pixels,
and the area under the accuracy curve is measured.

Desired behavior:

- positive perturbation should destroy accuracy quickly, producing a
  low AUC;
- negative perturbation should preserve accuracy for longer, producing
  a high AUC.

Tests are computed for both predicted and ground-truth classes. This
separates explanation of the model's observed decision from
localization relative to the labeled class.

This perturbation design is the most directly relevant part for the
current speech project. It evaluates whether highly ranked regions
matter more to the target score than low-ranked regions.

### Efficiency

Runtime is measured over 1,000 images. LeGrad remains close to
single-layer gradient methods because it sums layer contributions
rather than multiplying relevance matrices through the network. It is
reported as much faster than LRP, TextSpan, and CheferCAM in the tested
setup.

### Ablations

The paper tests:

- positive-gradient clipping with ReLU;
- use of all layers versus only the final layer;
- how many layers should contribute for different model sizes;
- gradient distributions across pretrained models.

Both ReLU and all-layer aggregation improve perturbation scores.
Larger ViTs generally require more layers for best performance.
Layer-importance profiles differ even among models with the same
architecture, suggesting that pretraining changes where class-relevant
features form.

## Main Findings

- Attention-gradient sensitivity produces sharper and more
  class-specific maps than raw attention.
- Intermediate layers contribute useful explanatory information.
- Simple averaging makes the method scalable to large Transformers.
- The method supports classifiers, vision-language similarity scores,
  and attentional pooling.
- LeGrad performs strongly in both localization and perturbation
  evaluations.

## Limitations

- The method was designed and primarily validated for image patches.
- Min-max normalization removes absolute relevance scale.
- Positive-gradient clipping excludes inhibitory evidence.
- A gradient is a local sensitivity, not a finite causal intervention.
- Perturbation results depend on how image regions are erased.
- Intermediate class scores require a classifier or embedding mapping
  that can sensibly operate on every layer.

## Adaptation to Speech

A wav2vec 2.0 encoder has temporal tokens instead of image patches:

```text
Z^l = {z_1^l, ..., z_T^l}
A^l in R^(h x T x T)
```

A direct temporal adaptation would:

1. define the target as the original predicted emotion score;
2. expose each layer's attention tensor;
3. construct a target score from the corresponding intermediate tokens;
4. compute the gradient of that score with respect to attention;
5. apply ReLU;
6. average over query rows and heads to obtain one score per temporal
   key token;
7. average compatible layer scores;
8. map tokens to waveform time using the feature-encoder stride.

The active SUPERB classifier learns a weighted combination of all
wav2vec 2.0 representation levels before its projection and
classification head. Intermediate layer scores therefore require a
carefully justified projection. A simpler initial variant can
differentiate the final predicted-class score with respect to every
layer's attention tensor, but this is a modification of the paper's
layer-local score formulation.

## Evaluation Adaptation for This Project

SpeechXAI outputs word intervals, while temporal LeGrad outputs
approximately fixed-resolution time bins. Equal region counts would
create unequal perturbations. The proposed evaluation therefore uses a
duration budget:

1. select the top `k` SpeechXAI words;
2. calculate their combined duration `X`;
3. silence those exact word intervals;
4. select the highest LeGrad time bins totaling `X` seconds;
5. select random time bins totaling the same `X`;
6. compare confidence drop for the original predicted class.

This is analogous to LeGrad's positive perturbation evaluation, adapted
to preserve waveform length and to equalize the amount of removed
speech.
