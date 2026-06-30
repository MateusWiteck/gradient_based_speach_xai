"""Lightweight relevance propagation through an audio classification head.

This intentionally stops at the final HuBERT token representations.  It does
not attempt LRP through the HuBERT Transformer, activations, normalization, or
residual connections.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


PoolingType = Literal["mean"]


def infer_pooling_type(model) -> PoolingType:
    """Return the supported pooling type for the current HuBERT classifier.

    ``HubertForSequenceClassification`` pools its projected token features with
    a (masked) mean.  We intentionally fail for unfamiliar architectures
    rather than silently attributing relevance with the wrong pooling rule.
    """
    if (
        isinstance(getattr(model, "projector", None), nn.Linear)
        and isinstance(getattr(model, "classifier", None), nn.Linear)
        and hasattr(model, "hubert")
    ):
        return "mean"

    raise NotImplementedError(
        "Could not detect a supported pooling strategy. "
        "This lightweight explainer currently supports mean pooling only."
    )


def _effective_classifier_weight(model, hidden_dim: int, num_classes: int) -> torch.Tensor:
    """Return the target-logit linear weight expressed in HuBERT hidden space.

    The SUPERB HuBERT model uses ``H -> projector -> mean -> classifier``.
    Because the projector is linear, the relevant weight for a pre-projector
    token is ``classifier.weight @ projector.weight``.  Models whose classifier
    consumes ``H`` directly are also supported.
    """
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise RuntimeError(
            "Expected model.classifier to be torch.nn.Linear for head relevance."
        )

    if classifier.out_features != num_classes:
        raise ValueError(
            "Classifier output size does not match logits: "
            f"{classifier.out_features} != {num_classes}."
        )

    if classifier.in_features == hidden_dim:
        return classifier.weight.detach()

    projector = getattr(model, "projector", None)
    if (
        isinstance(projector, nn.Linear)
        and projector.in_features == hidden_dim
        and projector.out_features == classifier.in_features
    ):
        return (classifier.weight @ projector.weight).detach()

    raise RuntimeError(
        "Cannot express the classifier as a linear head over the supplied "
        "HuBERT hidden states. Expected either a direct linear classifier or "
        "a linear model.projector followed by model.classifier."
    )


def _normalize_nonnegative(
    scores: torch.Tensor,
    *,
    token_mask: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Normalize token scores, using absolute values and then uniform fallback."""
    finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    positive_scores = finite_scores.clamp_min(0)
    if token_mask is not None:
        positive_scores = positive_scores * token_mask

    positive_sum = positive_scores.sum(dim=-1, keepdim=True)
    absolute_scores = finite_scores.abs()
    if token_mask is not None:
        absolute_scores = absolute_scores * token_mask
    absolute_sum = absolute_scores.sum(dim=-1, keepdim=True)

    # Prefer positive evidence. If it is empty/unstable, use unsigned evidence.
    normalized = positive_scores / positive_sum.clamp_min(eps)
    normalized_abs = absolute_scores / absolute_sum.clamp_min(eps)
    normalized = torch.where(positive_sum > eps, normalized, normalized_abs)

    # A completely zero linear contribution has no preferred token. Keep the
    # output valid by assigning equal relevance to all valid tokens.
    if token_mask is None:
        uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
    else:
        valid_tokens = token_mask.sum(dim=-1, keepdim=True)
        masked_uniform = token_mask / valid_tokens.clamp_min(1)
        unmasked_uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
        uniform = torch.where(valid_tokens > 0, masked_uniform, unmasked_uniform)
    return torch.where(absolute_sum > eps, normalized, uniform)


def compute_head_relevance(
    last_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    target_class: int,
    model,
    pooling: PoolingType = "mean",
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized token relevance from the linear head and mean pooling.

    For mean pooling, the contribution of token ``t`` to target logit ``c`` is
    ``sum_j(H[t, j] * W[c, j]) / T``. ``W`` is the final classifier weight in
    the HuBERT hidden-state space; for the current SUPERB checkpoint this folds
    the linear projector into the classifier weight. Positive contributions are
    normalized first, with absolute-contribution normalization as a fallback.

    Args:
        last_hidden_states: HuBERT representations fed into the classification
            head, shaped ``[batch, time, hidden_dim]``.
        logits: Classification logits shaped ``[batch, classes]``.
        target_class: Class whose logit is explained.
        model: Model exposing a direct ``classifier`` linear layer, optionally
            preceded by a linear ``projector``.
        pooling: Only ``"mean"`` is supported.
        token_mask: Optional valid-token mask shaped ``[batch, time]``. It
            mirrors the checkpoint's masked mean pooling for padded batches.

    Returns:
        Non-negative tensor ``[batch, time]`` whose rows sum to one.
    """
    if last_hidden_states.ndim != 3:
        raise ValueError(
            "last_hidden_states must have shape [batch, time, hidden_dim], got "
            f"{tuple(last_hidden_states.shape)}."
        )
    if logits.ndim != 2 or logits.shape[0] != last_hidden_states.shape[0]:
        raise ValueError("logits must have shape [batch, classes] for the same batch.")
    if pooling != "mean":
        raise NotImplementedError("Only mean pooling is supported for Level 3 relevance.")
    if not 0 <= target_class < logits.shape[-1]:
        raise ValueError(
            f"target_class must be in [0, {logits.shape[-1] - 1}], got {target_class}."
        )

    batch_size, num_tokens, hidden_dim = last_hidden_states.shape
    if num_tokens == 0:
        raise ValueError("last_hidden_states must contain at least one time token.")

    if token_mask is not None:
        if token_mask.shape != (batch_size, num_tokens):
            raise ValueError(
                "token_mask must have shape [batch, time], got "
                f"{tuple(token_mask.shape)}."
            )
        token_mask = token_mask.to(
            device=last_hidden_states.device,
            dtype=last_hidden_states.dtype,
        )

    effective_weight = _effective_classifier_weight(
        model=model,
        hidden_dim=hidden_dim,
        num_classes=logits.shape[-1],
    ).to(device=last_hidden_states.device, dtype=last_hidden_states.dtype)
    target_weight = effective_weight[target_class]

    # [B, T, D] @ [D] -> [B, T]. Mean pooling distributes each token's
    # contribution equally; for masked pooling divide by valid tokens per row.
    token_contrib = torch.einsum("btd,d->bt", last_hidden_states, target_weight)
    if token_mask is None:
        divisor = float(num_tokens)
    else:
        divisor = token_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    token_contrib = token_contrib / divisor

    return _normalize_nonnegative(token_contrib, token_mask=token_mask, eps=eps)


def compute_contrastive_head_relevance(
    last_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    target_class: int,
    model,
    pooling: PoolingType = "mean",
    contrast_class: int | None = None,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attribute tokens that favour the target over a competing class.

    This is a lightweight contrastive extension of :func:`compute_head_relevance`.
    For target class ``c`` and competitor ``k``, each token receives the
    positive part of ``H[t] · (W[c] - W[k]) / T``. The class-bias difference is
    intentionally not assigned to a time token. If ``contrast_class`` is not
    supplied, the highest-logit non-target class is selected for each sample.

    Returns:
        A tuple ``(token_relevance, contrast_classes)``. ``token_relevance``
        has shape ``[batch, time]`` and sums to one per sample;
        ``contrast_classes`` has shape ``[batch]``.
    """
    if last_hidden_states.ndim != 3:
        raise ValueError(
            "last_hidden_states must have shape [batch, time, hidden_dim], got "
            f"{tuple(last_hidden_states.shape)}."
        )
    if logits.ndim != 2 or logits.shape[0] != last_hidden_states.shape[0]:
        raise ValueError("logits must have shape [batch, classes] for the same batch.")
    if pooling != "mean":
        raise NotImplementedError("Only mean pooling is supported for Level 3 relevance.")
    if logits.shape[-1] < 2:
        raise ValueError("Contrastive relevance requires at least two classes.")
    if not 0 <= target_class < logits.shape[-1]:
        raise ValueError(
            f"target_class must be in [0, {logits.shape[-1] - 1}], got {target_class}."
        )

    batch_size, num_tokens, hidden_dim = last_hidden_states.shape
    if num_tokens == 0:
        raise ValueError("last_hidden_states must contain at least one time token.")

    if token_mask is not None:
        if token_mask.shape != (batch_size, num_tokens):
            raise ValueError(
                "token_mask must have shape [batch, time], got "
                f"{tuple(token_mask.shape)}."
            )
        token_mask = token_mask.to(
            device=last_hidden_states.device,
            dtype=last_hidden_states.dtype,
        )

    if contrast_class is None:
        # The runner-up is the class with the highest logit except the target.
        candidate_logits = logits.detach().clone()
        candidate_logits[:, target_class] = -torch.inf
        contrast_classes = candidate_logits.argmax(dim=-1)
    else:
        if not 0 <= contrast_class < logits.shape[-1]:
            raise ValueError(
                f"contrast_class must be in [0, {logits.shape[-1] - 1}], "
                f"got {contrast_class}."
            )
        if contrast_class == target_class:
            raise ValueError("contrast_class must differ from target_class.")
        contrast_classes = torch.full(
            (batch_size,),
            contrast_class,
            device=last_hidden_states.device,
            dtype=torch.long,
        )

    effective_weight = _effective_classifier_weight(
        model=model,
        hidden_dim=hidden_dim,
        num_classes=logits.shape[-1],
    ).to(device=last_hidden_states.device, dtype=last_hidden_states.dtype)
    target_weight = effective_weight[target_class].unsqueeze(0)
    contrast_weight = effective_weight[contrast_classes]
    margin_weight = target_weight - contrast_weight

    # [B, T, D] · [B, D] -> [B, T]. This is the time-dependent portion of
    # the target-versus-competitor logit margin under mean pooling.
    token_contrib = torch.einsum("btd,bd->bt", last_hidden_states, margin_weight)
    if token_mask is None:
        divisor = float(num_tokens)
    else:
        divisor = token_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    token_contrib = token_contrib / divisor

    relevance = _normalize_nonnegative(token_contrib, token_mask=token_mask, eps=eps)
    return relevance, contrast_classes
