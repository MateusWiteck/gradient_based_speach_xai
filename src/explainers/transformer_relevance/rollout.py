import torch


def compute_rollout_attention(all_layer_matrices, start_layer: int = 0):
    """
    Computes attention rollout across Transformer layers.

    Args:
        all_layer_matrices:
            list of attention matrices, one per layer.
            Each tensor should have shape [batch, tokens, tokens].
        start_layer:
            first layer to include in the rollout.

    Returns:
        joint_attention:
            tensor with shape [batch, tokens, tokens].
    """

    if len(all_layer_matrices) == 0:
        raise ValueError("all_layer_matrices is empty.")

    num_tokens = all_layer_matrices[0].shape[-1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device

    eye = torch.eye(num_tokens, device=device)
    eye = eye.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    # Add residual/self-connection
    matrices_aug = [
        layer_attention + eye
        for layer_attention in all_layer_matrices
    ]

    # Normalize rows
    matrices_aug = [
        matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        for matrix in matrices_aug
    ]

    # Multiply attentions across layers
    joint_attention = matrices_aug[start_layer]

    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)

    return joint_attention


def rollout_to_temporal_relevance(joint_attention, strategy: str = "mean"):
    """
    Converts rollout matrix [batch, tokens, tokens] into temporal relevance [batch, tokens].

    HuBERT does not have the same CLS-token structure as ViT, so we need a pooling strategy.

    Strategies:
        mean:
            average relevance received by each token.
        last:
            relevance from the last token to all tokens.
        first:
            relevance from the first token to all tokens.
    """

    if strategy == "mean":
        relevance = joint_attention.mean(dim=1)

    elif strategy == "last":
        relevance = joint_attention[:, -1, :]

    elif strategy == "first":
        relevance = joint_attention[:, 0, :]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize to [0, 1] per sample
    relevance_min = relevance.min(dim=-1, keepdim=True).values
    relevance_max = relevance.max(dim=-1, keepdim=True).values

    relevance = (relevance - relevance_min) / (
        relevance_max - relevance_min
    ).clamp(min=1e-8)

    return relevance


def rollout_from_token_relevance(
    joint_attention: torch.Tensor,
    token_relevance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Propagate relevance seeds through a joint attention-rollout matrix.

    ``joint_attention[i, j]`` describes the rollout path from final token
    ``i`` to token ``j``. Given relevance assigned to the final tokens, this
    function computes ``sum_i token_relevance[i] * joint_attention[i, j]``.
    It is therefore a target-conditioned alternative to averaging every row of
    the rollout matrix.
    """
    if joint_attention.ndim != 3:
        raise ValueError("joint_attention must have shape [batch, tokens, tokens].")
    if token_relevance.ndim != 2:
        raise ValueError("token_relevance must have shape [batch, tokens].")
    if joint_attention.shape[:2] != token_relevance.shape:
        raise ValueError(
            "joint_attention and token_relevance batch/token dimensions must match."
        )
    if not torch.isfinite(joint_attention).all() or not torch.isfinite(token_relevance).all():
        raise ValueError("Conditioned rollout inputs must be finite.")

    seeds = token_relevance.clamp_min(0)
    seed_sums = seeds.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(seeds, 1.0 / seeds.shape[-1])
    seeds = torch.where(seed_sums > eps, seeds / seed_sums.clamp_min(eps), uniform)

    conditioned = torch.bmm(seeds.unsqueeze(1), joint_attention).squeeze(1)
    conditioned = conditioned.clamp_min(0)
    conditioned_sums = conditioned.sum(dim=-1, keepdim=True)
    return conditioned / conditioned_sums.clamp_min(eps)
