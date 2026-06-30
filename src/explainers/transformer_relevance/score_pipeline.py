"""Shared score construction for rollout and lightweight Level 3 variants."""

from __future__ import annotations

import torch

from src.explainers.transformer_relevance.head_relevance import (
    compute_contrastive_head_relevance,
    compute_head_relevance,
    infer_pooling_type,
)
from src.explainers.transformer_relevance.rollout import (
    compute_rollout_attention,
    rollout_from_token_relevance,
    rollout_to_temporal_relevance,
)


EXPLANATION_MODES = (
    "rollout",
    "level3",
    "level3-contrastive",
    "head-conditioned-rollout",
    "contrastive-conditioned-rollout",
)


def normalize_token_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize non-negative token scores with a valid zero-score fallback."""
    if scores.ndim != 2:
        raise ValueError(f"Expected [batch, time] scores, got {tuple(scores.shape)}.")
    if not torch.isfinite(scores).all():
        raise ValueError("Relevance scores must be finite before normalization.")

    scores = scores.clamp_min(0)
    score_sums = scores.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
    return torch.where(score_sums > eps, scores / score_sums.clamp_min(eps), uniform)


def compute_relevance_scores(
    model,
    base_result: dict,
    grad_result: dict,
    contrast_class: int | None = None,
) -> dict:
    """Build every supported temporal-relevance variant for one utterance.

    The existing Level 3 modes retain their product formulation. The two
    conditioned modes instead use the ordinary or contrastive head relevance
    as final-token seeds and propagate those seeds through the joint gradient ×
    attention rollout matrix.
    """
    target_class = grad_result["target_class"]
    head_input = base_result.get("head_input_hidden_states")
    if head_input is None:
        raise RuntimeError("Could not capture HuBERT states entering the classifier head.")

    joint_attention = compute_rollout_attention(
        grad_result["grad_attentions"],
        start_layer=0,
    )
    raw_rollout = rollout_to_temporal_relevance(joint_attention, strategy="mean")
    rollout_score = normalize_token_scores(raw_rollout)

    common_kwargs = {
        "last_hidden_states": head_input,
        "logits": base_result["logits"],
        "target_class": target_class,
        "model": model,
        "pooling": infer_pooling_type(model),
        "token_mask": base_result.get("feature_attention_mask"),
    }
    head_relevance = normalize_token_scores(compute_head_relevance(**common_kwargs))
    contrastive_head, contrast_classes = compute_contrastive_head_relevance(
        **common_kwargs,
        contrast_class=contrast_class,
    )
    contrastive_head = normalize_token_scores(contrastive_head)

    scores = {
        "rollout": rollout_score,
        "level3": normalize_token_scores(rollout_score * head_relevance),
        "level3-contrastive": normalize_token_scores(rollout_score * contrastive_head),
        "head-conditioned-rollout": rollout_from_token_relevance(
            joint_attention,
            head_relevance,
        ),
        "contrastive-conditioned-rollout": rollout_from_token_relevance(
            joint_attention,
            contrastive_head,
        ),
    }
    return {
        "scores": scores,
        "target_class": target_class,
        "contrast_classes": contrast_classes,
        "joint_attention": joint_attention,
        "raw_rollout": raw_rollout,
        "head_relevance": head_relevance,
        "contrastive_head_relevance": contrastive_head,
    }
