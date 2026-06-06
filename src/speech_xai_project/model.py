from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Prediction:
    predicted_label: str
    predicted_index: int
    predicted_confidence: float
    probabilities: torch.Tensor


@dataclass
class AudioClassifier:
    network: torch.nn.Module
    feature_extractor: object
    sample_rate: int
    device: torch.device


def load_classifier(source: str, cache_dir: str | Path | None = None) -> AudioClassifier:
    """Load the Hugging Face emotion classifier used by Pastor et al. (2024)."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    resolved_cache_dir = str(cache_dir) if cache_dir is not None else None
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        source,
        cache_dir=resolved_cache_dir,
    )
    network = Wav2Vec2ForSequenceClassification.from_pretrained(
        source,
        cache_dir=resolved_cache_dir,
        attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.eval()
    return AudioClassifier(
        network=network,
        feature_extractor=feature_extractor,
        sample_rate=int(feature_extractor.sampling_rate),
        device=device,
    )


def classify_waveform(classifier: AudioClassifier, waveform: torch.Tensor) -> Prediction:
    """Classify one waveform and return probabilities in the model's label order."""
    waveform_batch = waveform.detach().cpu().float()
    if waveform_batch.ndim == 1:
        waveform_batch = waveform_batch.unsqueeze(0)
    if waveform_batch.ndim != 2 or waveform_batch.shape[0] != 1:
        raise ValueError("classify_waveform expects one waveform shaped [samples] or [1, samples].")

    inputs = classifier.feature_extractor(
        waveform_batch.squeeze(0).numpy(),
        sampling_rate=classifier.sample_rate,
        return_tensors="pt",
    )
    model_inputs = {name: value.to(classifier.device) for name, value in inputs.items()}

    with torch.no_grad():
        logits = classifier.network(**model_inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    predicted_index = int(probabilities.argmax())
    predicted_confidence = float(probabilities[predicted_index])
    predicted_label = str(classifier.network.config.id2label[predicted_index])
    return Prediction(
        predicted_label=predicted_label,
        predicted_index=predicted_index,
        predicted_confidence=predicted_confidence,
        probabilities=probabilities,
    )


def prepare_model_inputs(
    classifier: AudioClassifier,
    waveform: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Prepare one waveform for direct forward passes through the classifier."""
    waveform_batch = waveform.detach().cpu().float()
    if waveform_batch.ndim == 2 and waveform_batch.shape[0] == 1:
        waveform_batch = waveform_batch.squeeze(0)
    if waveform_batch.ndim != 1:
        raise ValueError("prepare_model_inputs expects one mono waveform.")

    inputs = classifier.feature_extractor(
        waveform_batch.numpy(),
        sampling_rate=classifier.sample_rate,
        return_tensors="pt",
    )
    return {name: value.to(classifier.device) for name, value in inputs.items()}


def print_module_tree(network: torch.nn.Module, max_depth: int = 3) -> None:
    """Print a compact module tree for locating wav2vec 2.0 internals."""
    for name, module in network.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth <= max_depth:
            indent = "  " * depth
            label = name or "<root>"
            print(f"{indent}{label}: {module.__class__.__name__}")
