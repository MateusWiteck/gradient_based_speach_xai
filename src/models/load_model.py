from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def _cached_snapshot_directory(model_name: str) -> str | None:
    """Return a local snapshot directory when the Hugging Face cache has one.

    Some older model caches contain ``preprocessor_config.json`` but not the
    newer optional ``processor_config.json``. Loading the snapshot directory
    directly lets Transformers use the available preprocessor file offline.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        config_path = try_to_load_from_cache(model_name, "config.json")
    except Exception:
        return None
    return str(Path(config_path).parent) if isinstance(config_path, str) else None


def load_hubert_emotion_model(
    model_name: str = "superb/hubert-base-superb-er",
    device: str | None = None,
    local_files_only: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_source = model_name
    if local_files_only:
        load_source = _cached_snapshot_directory(model_name) or model_name

    processor = AutoFeatureExtractor.from_pretrained(
        load_source,
        local_files_only=local_files_only,
    )
    # Some HF model repos declare custom architecture classes (e.g. "HubertForSequenceClassification").
    # Allow loading remote model code when required by the checkpoint. This executes
    # model code from the model repo, so be aware of the security implications.
    model = AutoModelForAudioClassification.from_pretrained(
        load_source,
        local_files_only=local_files_only,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    model.to(device)
    model.eval()

    return model, processor, device
