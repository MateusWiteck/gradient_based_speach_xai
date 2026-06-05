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


def load_speechbrain_classifier(source: str, savedir: str | Path):
    """Load SpeechBrain's pretrained emotion classifier.

    Importing SpeechBrain lazily keeps utility imports usable before dependencies are installed.
    """
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(source=source, savedir=str(savedir))


def classify_waveform(classifier, waveform: torch.Tensor) -> Prediction:
    """Run a SpeechBrain classifier and return a normalized prediction object."""
    with torch.no_grad():
        output_probabilities, score, index, text_label = classifier.classify_batch(waveform)

    probabilities = output_probabilities.detach().cpu().squeeze(0)
    predicted_index = int(index.detach().cpu().reshape(-1)[0])
    predicted_confidence = float(score.detach().cpu().reshape(-1)[0])
    predicted_label = str(text_label[0] if isinstance(text_label, list) else text_label)
    return Prediction(
        predicted_label=predicted_label,
        predicted_index=predicted_index,
        predicted_confidence=predicted_confidence,
        probabilities=probabilities,
    )


def print_module_tree(model, max_depth: int = 3) -> None:
    """Print a compact module tree for finding wav2vec2 internals."""
    for name, module in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth <= max_depth:
            indent = "  " * depth
            label = name or "<root>"
            print(f"{indent}{label}: {module.__class__.__name__}")

