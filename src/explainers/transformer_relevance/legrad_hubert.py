"""LeGrad-style temporal relevance for HuBERT sequence classification.

This module intentionally separates LeGrad-style relevance from the existing
gradient-times-attention rollout baselines.  The LeGrad-style variants use the
positive gradient of a class score with respect to attention probabilities:

    ReLU(d score / d attention)

The existing gradient-times-attention rollout baselines remain separate.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


TokenAxis = Literal["source_tokens", "query_tokens"]
LayerAggregation = Literal[
    "mean_layers",
    "renormalized_hubert_layer_weighted",
]

LEGRAD_FINAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS = (
    "legrad_final_score_relu_attention_gradient_mean_layers_source_tokens"
)
LEGRAD_FINAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS = (
    "legrad_final_score_relu_attention_gradient_"
    "renormalized_hubert_layer_weighted_source_tokens"
)
LEGRAD_LAYER_LOCAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS = (
    "legrad_layer_local_score_relu_attention_gradient_mean_layers_source_tokens"
)
LEGRAD_LAYER_LOCAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS = (
    "legrad_layer_local_score_relu_attention_gradient_"
    "renormalized_hubert_layer_weighted_source_tokens"
)

LEGRAD_HUBERT_EXPLANATION_MODES = (
    LEGRAD_FINAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS,
    LEGRAD_FINAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS,
    LEGRAD_LAYER_LOCAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS,
    LEGRAD_LAYER_LOCAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS,
)


def normalize_nonnegative_token_scores(
    scores: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize non-negative temporal scores with a uniform fallback."""
    if scores.ndim != 2:
        raise ValueError(f"Expected [batch, time] scores, got {tuple(scores.shape)}.")
    if not torch.isfinite(scores).all():
        raise ValueError("Token relevance must be finite before normalization.")

    scores = scores.clamp_min(0)
    if token_mask is not None:
        if token_mask.shape != scores.shape:
            raise ValueError(
                "token_mask must have shape [batch, time], got "
                f"{tuple(token_mask.shape)} for scores {tuple(scores.shape)}."
            )
        token_mask = token_mask.to(device=scores.device, dtype=scores.dtype)
        scores = scores * token_mask
        valid_counts = token_mask.sum(dim=-1, keepdim=True)
        uniform = token_mask / valid_counts.clamp_min(1)
        unmasked_uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
        uniform = torch.where(valid_counts > 0, uniform, unmasked_uniform)
    else:
        uniform = torch.full_like(scores, 1.0 / scores.shape[-1])

    score_sums = scores.sum(dim=-1, keepdim=True)
    return torch.where(score_sums > eps, scores / score_sums.clamp_min(eps), uniform)


def minmax_normalize_token_scores(
    scores: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-max normalize temporal scores for paper-style visualizations."""
    if scores.ndim != 2:
        raise ValueError(f"Expected [batch, time] scores, got {tuple(scores.shape)}.")
    if not torch.isfinite(scores).all():
        raise ValueError("Token relevance must be finite before min-max normalization.")

    scores = scores.clamp_min(0)
    if token_mask is None:
        score_min = scores.min(dim=-1, keepdim=True).values
        score_max = scores.max(dim=-1, keepdim=True).values
        score_range = score_max - score_min
        normalized = (scores - score_min) / score_range.clamp_min(eps)
        return torch.where(score_range > eps, normalized, torch.zeros_like(scores))

    if token_mask.shape != scores.shape:
        raise ValueError(
            "token_mask must have shape [batch, time], got "
            f"{tuple(token_mask.shape)} for scores {tuple(scores.shape)}."
        )
    valid_mask = token_mask.to(device=scores.device, dtype=torch.bool)
    valid_counts = valid_mask.sum(dim=-1, keepdim=True)
    masked_min_scores = scores.masked_fill(~valid_mask, float("inf"))
    masked_max_scores = scores.masked_fill(~valid_mask, -float("inf"))
    score_min = masked_min_scores.min(dim=-1, keepdim=True).values
    score_max = masked_max_scores.max(dim=-1, keepdim=True).values
    score_range = score_max - score_min
    normalized = (scores - score_min) / score_range.clamp_min(eps)
    normalized = torch.where(
        (valid_counts > 0) & (score_range > eps),
        normalized,
        torch.zeros_like(scores),
    )
    return normalized.masked_fill(~valid_mask, 0)


def masked_temporal_mean(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply HuBERT's temporal mean pooling rule to projected token states."""
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, time, dim], got "
            f"{tuple(hidden_states.shape)}."
        )
    if token_mask is None:
        return hidden_states.mean(dim=1)

    if token_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "token_mask must have shape [batch, time], got "
            f"{tuple(token_mask.shape)} for hidden states {tuple(hidden_states.shape)}."
        )
    token_mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    expanded_mask = token_mask.unsqueeze(-1)
    summed = (hidden_states * expanded_mask).sum(dim=1)
    counts = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return summed / counts


def feature_attention_mask_from_inputs(
    model,
    inputs: dict[str, torch.Tensor],
    token_count: int,
) -> torch.Tensor | None:
    """Return the reduced HuBERT feature mask used by sequence classification."""
    attention_mask = inputs.get("attention_mask")
    get_feature_mask = getattr(model, "_get_feature_vector_attention_mask", None)
    if attention_mask is None or get_feature_mask is None:
        return None
    return get_feature_mask(token_count, attention_mask)


def reduce_attention_gradient_to_token_relevance(
    attention_gradient: torch.Tensor,
    token_axis: TokenAxis = "source_tokens",
) -> torch.Tensor:
    """Convert an attention-gradient tensor to temporal token relevance.

    HuBERT attention gradients have shape [batch, heads, query_time, source_time].
    The default keeps source/key tokens, matching the original LeGrad reduction
    that averages over heads and query tokens.
    """
    if attention_gradient.ndim != 4:
        raise ValueError(
            "attention_gradient must have shape [batch, heads, query, source], "
            f"got {tuple(attention_gradient.shape)}."
        )

    positive_gradient = attention_gradient.clamp_min(0)
    token_matrix = positive_gradient.mean(dim=1)
    if token_axis == "source_tokens":
        return token_matrix.mean(dim=1)
    if token_axis == "query_tokens":
        return token_matrix.mean(dim=2)
    raise ValueError(f"Unknown token_axis: {token_axis!r}.")


def transformer_layer_mixture_weights(
    model,
    *,
    num_attention_layers: int,
    device,
    dtype,
    renormalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return HuBERT layer-mixture coefficients aligned to attention layers.

    The SUPERB HuBERT checkpoint stores 13 raw layer weights: H_0 plus the 12
    Transformer outputs.  Attention maps exist only for the Transformer layers,
    so this helper drops the H_0 coefficient and optionally renormalizes the
    remaining coefficients to sum to one.
    """
    layer_weights = getattr(model, "layer_weights", None)
    if layer_weights is None:
        return torch.full(
            (num_attention_layers,),
            1.0 / num_attention_layers,
            device=device,
            dtype=dtype,
        )

    coefficients = torch.softmax(layer_weights.detach().to(device=device, dtype=dtype), dim=-1)
    if coefficients.numel() == num_attention_layers + 1:
        coefficients = coefficients[1:]
    elif coefficients.numel() != num_attention_layers:
        raise ValueError(
            "Could not align model.layer_weights with attention layers: "
            f"{coefficients.numel()} coefficients for {num_attention_layers} attention layers."
        )

    if renormalize:
        coefficients = coefficients / coefficients.sum().clamp_min(eps)
    return coefficients


def aggregate_layer_relevance(
    layer_token_relevance: torch.Tensor,
    *,
    aggregation: LayerAggregation,
    model,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate [batch, layers, time] relevance into [batch, time]."""
    if layer_token_relevance.ndim != 3:
        raise ValueError(
            "layer_token_relevance must have shape [batch, layers, time], got "
            f"{tuple(layer_token_relevance.shape)}."
        )

    _, num_layers, _ = layer_token_relevance.shape
    if aggregation == "mean_layers":
        layer_weights = torch.full(
            (num_layers,),
            1.0 / num_layers,
            device=layer_token_relevance.device,
            dtype=layer_token_relevance.dtype,
        )
    elif aggregation == "renormalized_hubert_layer_weighted":
        layer_weights = transformer_layer_mixture_weights(
            model,
            num_attention_layers=num_layers,
            device=layer_token_relevance.device,
            dtype=layer_token_relevance.dtype,
            renormalize=True,
        )
    else:
        raise ValueError(f"Unknown layer aggregation: {aggregation!r}.")

    aggregated = (layer_token_relevance * layer_weights.view(1, -1, 1)).sum(dim=1)
    return (
        normalize_nonnegative_token_scores(aggregated, token_mask=token_mask),
        minmax_normalize_token_scores(aggregated, token_mask=token_mask),
        layer_weights,
    )


def _prepare_audio_inputs(
    processor,
    waveform,
    sampling_rate: int,
    device: str,
) -> dict[str, torch.Tensor]:
    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _classifier_logits_from_hidden_states(
    model,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    projector = getattr(model, "projector", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(projector, nn.Linear) or not isinstance(classifier, nn.Linear):
        raise RuntimeError(
            "LeGrad-HuBERT layer-local scores require linear model.projector "
            "and model.classifier modules."
        )

    projected = projector(hidden_states)
    pooled = masked_temporal_mean(projected, token_mask)
    return classifier(pooled)


def extract_legrad_hubert_relevance(
    model,
    processor,
    waveform,
    sampling_rate: int = 16000,
    device: str = "cuda",
    target_class: int | None = None,
    token_axis: TokenAxis = "source_tokens",
) -> dict:
    """Compute LeGrad-style HuBERT relevance variants for one utterance.

    Returns all currently supported LeGrad-HuBERT modes.  The layer-local modes
    use H_l -> projector -> masked temporal pooling -> classifier to build one
    score per Transformer layer.  The final-score modes use the checkpoint's
    real final emotion logit built from H_mix.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    inputs = _prepare_audio_inputs(processor, waveform, sampling_rate, device)
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )

    logits = outputs.logits
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("The model did not return differentiable attention maps.")
    if hidden_states is None or len(hidden_states) != len(attentions) + 1:
        raise RuntimeError(
            "Expected HuBERT hidden_states to contain H_0 plus one output per "
            "attention layer."
        )

    predicted_class = int(torch.argmax(logits, dim=-1).item())
    if target_class is None:
        target_class = predicted_class
    if not 0 <= target_class < logits.shape[-1]:
        raise ValueError(
            f"target_class must be in [0, {logits.shape[-1] - 1}], got {target_class}."
        )

    first_attention = attentions[0]
    token_mask = feature_attention_mask_from_inputs(
        model,
        inputs,
        token_count=first_attention.shape[-1],
    )

    final_score = logits[:, target_class].sum()
    final_score_grads = torch.autograd.grad(
        final_score,
        attentions,
        retain_graph=True,
        create_graph=False,
    )
    final_score_layer_relevance = torch.stack(
        [
            reduce_attention_gradient_to_token_relevance(grad, token_axis=token_axis)
            for grad in final_score_grads
        ],
        dim=1,
    ).detach()

    layer_local_relevances = []
    layer_local_scores = []
    for attention_index, attention in enumerate(attentions):
        # hidden_states[0] is H_0. Attention layer i produces H_{i+1}.
        layer_hidden_states = hidden_states[attention_index + 1]
        layer_logits = _classifier_logits_from_hidden_states(
            model,
            layer_hidden_states,
            token_mask,
        )
        layer_score = layer_logits[:, target_class].sum()
        layer_grad = torch.autograd.grad(
            layer_score,
            attention,
            retain_graph=True,
            create_graph=False,
        )[0]
        layer_local_relevances.append(
            reduce_attention_gradient_to_token_relevance(layer_grad, token_axis=token_axis)
        )
        layer_local_scores.append(layer_logits[:, target_class].detach())

    layer_local_score_layer_relevance = torch.stack(
        layer_local_relevances,
        dim=1,
    ).detach()
    layer_local_scores = torch.stack(layer_local_scores, dim=1)

    final_mean, final_mean_visual, mean_layer_weights = aggregate_layer_relevance(
        final_score_layer_relevance,
        aggregation="mean_layers",
        model=model,
        token_mask=token_mask,
    )
    (
        final_hubert_weighted,
        final_hubert_weighted_visual,
        hubert_layer_weights,
    ) = aggregate_layer_relevance(
        final_score_layer_relevance,
        aggregation="renormalized_hubert_layer_weighted",
        model=model,
        token_mask=token_mask,
    )
    layer_local_mean, layer_local_mean_visual, _ = aggregate_layer_relevance(
        layer_local_score_layer_relevance,
        aggregation="mean_layers",
        model=model,
        token_mask=token_mask,
    )
    (
        layer_local_hubert_weighted,
        layer_local_hubert_weighted_visual,
        _,
    ) = aggregate_layer_relevance(
        layer_local_score_layer_relevance,
        aggregation="renormalized_hubert_layer_weighted",
        model=model,
        token_mask=token_mask,
    )

    return {
        "scores": {
            LEGRAD_FINAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS: final_mean.detach(),
            LEGRAD_FINAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS: final_hubert_weighted.detach(),
            LEGRAD_LAYER_LOCAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS: layer_local_mean.detach(),
            LEGRAD_LAYER_LOCAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS: layer_local_hubert_weighted.detach(),
        },
        "minmax_visual_scores": {
            LEGRAD_FINAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS: final_mean_visual.detach(),
            LEGRAD_FINAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS: final_hubert_weighted_visual.detach(),
            LEGRAD_LAYER_LOCAL_SCORE_MEAN_LAYERS_SOURCE_TOKENS: layer_local_mean_visual.detach(),
            LEGRAD_LAYER_LOCAL_SCORE_HUBERT_WEIGHTED_SOURCE_TOKENS: layer_local_hubert_weighted_visual.detach(),
        },
        "target_class": target_class,
        "predicted_class": predicted_class,
        "logits": logits.detach(),
        "token_axis": token_axis,
        "feature_attention_mask": token_mask.detach() if token_mask is not None else None,
        "final_score_layer_relevance": final_score_layer_relevance,
        "layer_local_score_layer_relevance": layer_local_score_layer_relevance,
        "layer_local_scores": layer_local_scores,
        "mean_layer_weights": mean_layer_weights.detach(),
        "renormalized_hubert_layer_weights": hubert_layer_weights.detach(),
        "num_attention_layers": len(attentions),
    }
