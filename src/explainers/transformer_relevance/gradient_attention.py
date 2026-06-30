import torch


def extract_gradient_weighted_attentions(
    model,
    processor,
    waveform,
    sampling_rate: int = 16000,
    device: str = "cuda",
    target_class: int | None = None,
):
    """
    Computes gradient × attention matrices for HuBERT.

    This is the audio adaptation of the Transformer's
    gradient-weighted attention relevance idea.

    Returns:
        dict with:
            logits: [batch, num_classes]
            predicted_class: int
            target_class: int
            attentions: tuple of [batch, heads, tokens, tokens]
            attention_grads: list of [batch, heads, tokens, tokens]
            grad_attentions: list of [batch, tokens, tokens]
    """

    model.eval()
    model.zero_grad()

    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Important: no torch.no_grad() here.
    # We need gradients.
    outputs = model(
        **inputs,
        output_attentions=True,
        return_dict=True,
    )

    logits = outputs.logits
    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError(
            "The model did not return attentions. "
            "Load it with attn_implementation='eager'."
        )

    if len(attentions) == 0 or attentions[0].numel() == 0:
        raise RuntimeError(
            "The returned attentions are empty. "
            "Load the model with attn_implementation='eager'."
        )

    predicted_class = int(torch.argmax(logits, dim=-1).item())

    if target_class is None:
        target_class = predicted_class

    # Retain gradients for non-leaf tensors
    for attn in attentions:
        attn.retain_grad()

    # Select target class logit
    target_logit = logits[:, target_class].sum()

    model.zero_grad()
    target_logit.backward(retain_graph=True)

    attention_grads = []
    grad_attentions = []

    for attn in attentions:
        grad = attn.grad

        if grad is None:
            raise RuntimeError(
                "Attention gradient is None. "
                "Check whether the model implementation exposes differentiable attentions."
            )

        attention_grads.append(grad)

        # grad × attention
        grad_attn = grad * attn

        # keep only positive contribution, as in the paper
        grad_attn = grad_attn.clamp(min=0)

        # average over heads
        # [batch, heads, tokens, tokens] -> [batch, tokens, tokens]
        grad_attn = grad_attn.mean(dim=1)

        grad_attentions.append(grad_attn.detach())

    return {
        "logits": logits.detach(),
        "predicted_class": predicted_class,
        "target_class": target_class,
        "attentions": tuple(attn.detach() for attn in attentions),
        "attention_grads": [grad.detach() for grad in attention_grads],
        "grad_attentions": grad_attentions,
    }