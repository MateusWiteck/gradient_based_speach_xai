import torch


def extract_hubert_attentions(
    model,
    processor,
    waveform,
    sampling_rate: int = 16000,
    device: str = "cuda",
):
    """Run HuBERT once and expose attentions plus the classifier's token input."""
    model.eval()

    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # For this checkpoint, the input to ``projector`` is the exact weighted
    # HuBERT representation that is later mean-pooled by the classifier. A
    # hook avoids assuming that ``outputs.hidden_states[-1]`` is the head input
    # (the checkpoint enables a learned weighted sum across layers).
    captured = {}
    projector = getattr(model, "projector", None)
    hook = None
    if isinstance(projector, torch.nn.Module):
        def capture_projector_input(_module, module_inputs):
            captured["head_input_hidden_states"] = module_inputs[0].detach()

        hook = projector.register_forward_pre_hook(capture_projector_input)

    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        if hook is not None:
            hook.remove()

    logits = outputs.logits
    attentions = outputs.attentions

    if attentions is None:
        attentions = []
        avg_attentions = []
    else:
        clean_attentions = [
            attention if attention is not None else torch.empty(0, device=device)
            for attention in attentions
        ]
        avg_attentions = [
            attention.mean(dim=1) if attention.numel() else attention
            for attention in clean_attentions
        ]
        attentions = clean_attentions

    last_hidden_states = None
    if outputs.hidden_states is not None:
        last_hidden_states = outputs.hidden_states[-1]

    head_input_hidden_states = captured.get("head_input_hidden_states", last_hidden_states)
    feature_attention_mask = None
    if inputs.get("attention_mask") is not None and head_input_hidden_states is not None:
        get_feature_mask = getattr(model, "_get_feature_vector_attention_mask", None)
        if get_feature_mask is not None:
            feature_attention_mask = get_feature_mask(
                head_input_hidden_states.shape[1],
                inputs["attention_mask"],
            )

    return {
        "logits": logits,
        "attentions": attentions,
        "avg_attentions": avg_attentions,
        "predicted_class": int(torch.argmax(logits, dim=-1).item()),
        "last_hidden_states": last_hidden_states,
        "head_input_hidden_states": head_input_hidden_states,
        "feature_attention_mask": feature_attention_mask,
    }
