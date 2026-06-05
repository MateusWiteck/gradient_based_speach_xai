from __future__ import annotations

import torch


def retain_attention_gradients(attentions: tuple[torch.Tensor, ...]) -> None:
    """Ask PyTorch to keep gradients for returned attention tensors."""
    for attention in attentions:
        if attention.requires_grad:
            attention.retain_grad()


def aggregate_attention_gradient_relevance(
    attentions: tuple[torch.Tensor, ...],
    use_destination_tokens: bool = True,
) -> torch.Tensor:
    """Compute LeGrad-inspired token relevance from attention maps and gradients.

    For each layer: mean_heads(ReLU(grad) * attention), then mean layers.
    The token-token matrix is reduced to token relevance by summing over one axis.
    """
    layer_relevances = []
    for attention in attentions:
        if attention.grad is None:
            continue

        attention_map = attention.detach()
        gradient_map = attention.grad.detach()
        weighted_attention = torch.relu(gradient_map) * attention_map
        # Shape: [batch, tokens, tokens]
        layer_relevance = weighted_attention.mean(dim=1)
        layer_relevances.append(layer_relevance)

    if not layer_relevances:
        raise ValueError("No attention gradients were available. Did you call retain_grad() before backward()?") 

    relevance_matrix = torch.stack(layer_relevances, dim=0).mean(dim=0)
    reduction_dim = 1 if use_destination_tokens else 2
    token_relevance = relevance_matrix.sum(dim=reduction_dim)
    return token_relevance.squeeze(0)


def normalize_torch(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float()
    min_value = torch.min(values)
    max_value = torch.max(values)
    if torch.isclose(max_value, min_value):
        return torch.zeros_like(values)
    return (values - min_value) / (max_value - min_value)

