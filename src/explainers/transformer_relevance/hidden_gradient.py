"""Gradient × hidden-state relevance at the HuBERT classifier input."""

from __future__ import annotations

import torch


def extract_hidden_state_gradients(
    model,
    processor,
    waveform,
    sampling_rate: int = 16000,
    device: str = "cuda",
    target_class: int | None = None,
) -> dict:
    """Compute gradients of a target logit with respect to final time tokens.

    The captured tensor is the input to ``model.projector``. For the SUPERB
    HuBERT checkpoint this is the weighted HuBERT representation that is later
    projected, mean-pooled, and classified. This intentionally stops at the
    final token representation and does not implement Transformer-layer LRP.
    """
    model.eval()
    model.zero_grad()

    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    captured = {}
    projector = getattr(model, "projector", None)
    hook = None
    if isinstance(projector, torch.nn.Module):

        def capture_projector_input(_module, module_inputs):
            hidden_states = module_inputs[0]
            hidden_states.retain_grad()
            captured["head_input_hidden_states"] = hidden_states

        hook = projector.register_forward_pre_hook(capture_projector_input)

    try:
        outputs = model(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        if hook is not None:
            hook.remove()

    logits = outputs.logits
    predicted_class = int(torch.argmax(logits, dim=-1).item())
    if target_class is None:
        target_class = predicted_class
    if not 0 <= target_class < logits.shape[-1]:
        raise ValueError(
            f"target_class must be in [0, {logits.shape[-1] - 1}], got {target_class}."
        )

    head_input = captured.get("head_input_hidden_states")
    if head_input is None:
        if outputs.hidden_states is None:
            raise RuntimeError(
                "Could not capture classifier input hidden states and model did "
                "not return hidden_states."
            )
        head_input = outputs.hidden_states[-1]
        head_input.retain_grad()

    target_logit = logits[:, target_class].sum()
    model.zero_grad()
    target_logit.backward()

    hidden_gradients = head_input.grad
    if hidden_gradients is None:
        raise RuntimeError("Gradient with respect to head input hidden states is None.")

    feature_attention_mask = None
    if inputs.get("attention_mask") is not None:
        get_feature_mask = getattr(model, "_get_feature_vector_attention_mask", None)
        if get_feature_mask is not None:
            feature_attention_mask = get_feature_mask(
                head_input.shape[1],
                inputs["attention_mask"],
            )

    return {
        "logits": logits.detach(),
        "predicted_class": predicted_class,
        "target_class": target_class,
        "head_input_hidden_states": head_input.detach(),
        "head_input_gradients": hidden_gradients.detach(),
        "feature_attention_mask": feature_attention_mask,
    }


def compute_gradient_hidden_relevance(
    hidden_states: torch.Tensor,
    hidden_gradients: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``sum_d |H[t, d] * d logit / d H[t, d]|`` per time token.

    Args:
        hidden_states: Tensor shaped ``[batch, time, hidden_dim]``.
        hidden_gradients: Same shape as ``hidden_states``.
        token_mask: Optional valid-token mask shaped ``[batch, time]``.

    Returns:
        Non-negative tensor shaped ``[batch, time]`` whose rows sum to one.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, time, hidden_dim], got "
            f"{tuple(hidden_states.shape)}."
        )
    if hidden_gradients.shape != hidden_states.shape:
        raise ValueError(
            "hidden_gradients must have the same shape as hidden_states, got "
            f"{tuple(hidden_gradients.shape)} and {tuple(hidden_states.shape)}."
        )
    if not torch.isfinite(hidden_states).all() or not torch.isfinite(hidden_gradients).all():
        raise ValueError("Hidden states and gradients must be finite.")

    token_scores = (hidden_states * hidden_gradients).abs().sum(dim=-1)
    if token_mask is not None:
        if token_mask.shape != token_scores.shape:
            raise ValueError(
                "token_mask must have shape [batch, time], got "
                f"{tuple(token_mask.shape)}."
            )
        token_scores = token_scores * token_mask.to(
            device=token_scores.device,
            dtype=token_scores.dtype,
        )

    score_sums = token_scores.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(token_scores, 1.0 / token_scores.shape[-1])
    if token_mask is not None:
        mask = token_mask.to(device=token_scores.device, dtype=token_scores.dtype)
        valid_tokens = mask.sum(dim=-1, keepdim=True)
        masked_uniform = mask / valid_tokens.clamp_min(1)
        uniform = torch.where(valid_tokens > 0, masked_uniform, uniform)

    return torch.where(score_sums > eps, token_scores / score_sums.clamp_min(eps), uniform)
