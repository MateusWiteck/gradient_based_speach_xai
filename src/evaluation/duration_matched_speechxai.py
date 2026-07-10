"""Duration-matched SpeechXAI/Pastor vs HuBERT temporal relevance evaluation.

The protocol is intentionally centered on the classifier's original prediction:

1. classify the unmodified waveform;
2. use that predicted class as the target;
3. compute SpeechXAI/Pastor leave-one-out word scores;
4. let SpeechXAI top-k words define a real masked duration X;
5. silence the same duration X for each HuBERT temporal explanation mode and
   for the random deletion by silence masking baseline.

SpeechXAI is used here as the word-level explanation protocol: WhisperX word
alignment plus leave-one-out word masking.  The target probabilities are always
computed with the current HuBERT emotion classifier so every method is compared
under the same model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.evaluation.common import MODEL_CLASS_NAMES
from src.explainers.transformer_relevance.attention_extractor import (
    extract_hubert_attentions,
)
from src.explainers.transformer_relevance.gradient_attention import (
    extract_gradient_weighted_attentions,
)
from src.explainers.transformer_relevance.legrad_hubert import (
    LEGRAD_HUBERT_EXPLANATION_MODES,
    extract_legrad_hubert_relevance,
)
from src.explainers.transformer_relevance.score_pipeline import (
    EXPLANATION_MODES,
    compute_relevance_scores,
)
from src.speech_xai_project.intervals import (
    random_intervals_by_duration,
    select_intervals_by_duration,
    token_relevance_to_intervals,
)
from src.speech_xai_project.masking import silence_intervals, total_interval_duration


SPEECHXAI_PASTOR_METHOD = "speechxai_pastor_loo_words"
RANDOM_DURATION_METHOD = "random_duration_matched_bins"
DEFAULT_TOP_K_VALUES = (1, 2, 3, 5)
DEFAULT_SPEECHXAI_PADDING_BEFORE_SECONDS = 0.10
DEFAULT_SPEECHXAI_PADDING_AFTER_SECONDS = 0.04


def disable_huggingface_symlink_cache_on_windows() -> None:
    """Avoid Hugging Face cache symlink failures on Windows without admin rights."""
    if os.name != "nt":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as file_download

        file_download._are_symlinks_supported_in_dir.clear()
        file_download.are_symlinks_supported = lambda cache_dir=None: False
    except Exception:
        # This is a compatibility best-effort. If huggingface_hub is not
        # imported yet or changes internals, the normal download path can try.
        return


@dataclass(frozen=True)
class AudioEvaluationExample:
    """Dataset provenance for one audio file."""

    dataset: str
    audio_path: str
    audio_id: str
    relative_path: str | None = None
    true_class: int | None = None
    true_label: str | None = None
    actor_id: int | None = None
    session_id: int | None = None
    dialogue_id: str | None = None
    utterance_id: str | None = None
    ravdess_emotion: str | None = None
    iemocap_emotion: str | None = None
    annotation_path: str | None = None

    def to_record(self) -> dict:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class PredictionResult:
    """Classifier output for an original or masked waveform."""

    predicted_class: int
    predicted_label: str
    target_class: int
    target_label: str
    target_logit: float
    target_confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class DurationMatchedAudioResult:
    """All tables produced for one evaluated audio."""

    deletion_records: pd.DataFrame
    speechxai_word_scores: pd.DataFrame
    selected_intervals: pd.DataFrame
    relevance_tables: dict[str, pd.DataFrame]
    original_prediction: PredictionResult
    transcript: str


def stable_seed(*items: object) -> int:
    """Return a deterministic uint32 seed from arbitrary identifying values."""
    payload = "|".join(str(item) for item in items).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**32)


def classify_waveform(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    *,
    device: str,
    target_class: int | None = None,
) -> PredictionResult:
    """Return prediction metadata and target-class probability for one waveform."""
    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
    predicted_class = int(torch.argmax(probabilities).item())
    if target_class is None:
        target_class = predicted_class
    if not 0 <= target_class < len(MODEL_CLASS_NAMES):
        raise ValueError(
            f"target_class must be in [0, {len(MODEL_CLASS_NAMES) - 1}], got {target_class}."
        )
    return PredictionResult(
        predicted_class=predicted_class,
        predicted_label=MODEL_CLASS_NAMES[predicted_class],
        target_class=target_class,
        target_label=MODEL_CLASS_NAMES[target_class],
        target_logit=float(logits[0, target_class].detach().cpu().item()),
        target_confidence=float(probabilities[target_class].item()),
        probabilities=tuple(float(value) for value in probabilities.tolist()),
    )


@contextmanager
def _trusted_legacy_torch_load():
    """Temporarily allow old pyannote/WhisperX checkpoint serialization."""
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _prepare_speechxai_namespace(speechxai_root: str | Path) -> None:
    """Expose the local SpeechXAI checkout without importing its top-level API."""
    speechxai_root = Path(speechxai_root).resolve()
    if not speechxai_root.is_dir():
        raise FileNotFoundError(f"SpeechXAI repository not found: {speechxai_root}")

    ffmpeg_dir = speechxai_root.parent / "ffmpeg"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            pass

        torchaudio.AudioMetaData = AudioMetaData

    compatibility = {
        "list_audio_backends": lambda: ["soundfile"],
        "get_audio_backend": lambda: "soundfile",
        "set_audio_backend": lambda backend: None,
    }
    for name, implementation in compatibility.items():
        if not hasattr(torchaudio, name):
            setattr(torchaudio, name, implementation)

    package_path = str(speechxai_root / "speechxai")
    package = sys.modules.get("speechxai")
    if package is None or not hasattr(package, "__path__"):
        package = types.ModuleType("speechxai")
        package.__path__ = [package_path]
        sys.modules["speechxai"] = package
    elif package_path not in package.__path__:
        package.__path__.append(package_path)


def _clean_word_segments(words: Iterable[dict]) -> list[dict]:
    """Keep aligned words that can define a finite positive time interval."""
    cleaned = []
    for word_index, word in enumerate(words):
        if "start" not in word or "end" not in word:
            continue
        start = float(word["start"])
        end = float(word["end"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        raw_score = word.get("score")
        transcription_score = None
        if raw_score is not None:
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                raw_score = None
            if raw_score is not None and np.isfinite(raw_score):
                transcription_score = raw_score
        cleaned.append(
            {
                "word": str(word.get("word", f"word_{word_index}")),
                "start": start,
                "end": end,
                "score": transcription_score,
            }
        )
    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    return cleaned


class SpeechXAITranscriber:
    """WhisperX word aligner using the local SpeechXAI checkout."""

    def __init__(
        self,
        *,
        speechxai_root: str | Path = "third_party/SpeechXAI",
        device: str = "cpu",
        batch_size: int = 2,
        compute_type: str = "int8",
        language: str = "en",
        whisper_model: str = "large-v2",
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        self.speechxai_root = Path(speechxai_root)
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.language = language
        self.whisper_model = whisper_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = use_cache
        self._model_whisperx = None
        self._transcribe_audio_given_model = None

    def _cache_path(self, audio_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.whisper_model).strip("_")
        language_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.language).strip("_")
        return self.cache_dir / f"{audio_id}__{language_slug}__{model_slug}.json"

    def _ensure_imports(self) -> None:
        if self._transcribe_audio_given_model is not None:
            return
        _prepare_speechxai_namespace(self.speechxai_root)
        from speechxai.explainers.utils_removal import transcribe_audio_given_model

        self._transcribe_audio_given_model = transcribe_audio_given_model

    def _ensure_model(self):
        self._ensure_imports()
        if self._model_whisperx is not None:
            return
        disable_huggingface_symlink_cache_on_windows()
        import whisperx

        with _trusted_legacy_torch_load():
            self._model_whisperx = whisperx.load_model(
                self.whisper_model,
                self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

    def transcribe(self, audio_path: str | Path, audio_id: str) -> tuple[str, list[dict]]:
        """Return transcript text and aligned word dictionaries for one audio."""
        cache_path = self._cache_path(audio_id)
        if self.use_cache and cache_path is not None and cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as input_file:
                cached = json.load(input_file)
            return str(cached.get("text", "")), _clean_word_segments(cached.get("words", []))

        self._ensure_model()
        with _trusted_legacy_torch_load():
            text, words = self._transcribe_audio_given_model(
                self._model_whisperx,
                str(audio_path),
                batch_size=self.batch_size,
                device=self.device,
            )
        words = _clean_word_segments(words)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as output_file:
                json.dump(
                    {
                        "audio_id": audio_id,
                        "audio_path": str(audio_path),
                        "text": text,
                        "words": words,
                        "speechxai_root": str(self.speechxai_root),
                        "whisper_model": self.whisper_model,
                        "language": self.language,
                    },
                    output_file,
                    indent=2,
                    sort_keys=True,
                )
                output_file.write("\n")
        return text, words


def words_to_timing_table(
    audio_id: str,
    words: Iterable[dict],
    *,
    audio_duration: float,
    padding_before_seconds: float = DEFAULT_SPEECHXAI_PADDING_BEFORE_SECONDS,
    padding_after_seconds: float = DEFAULT_SPEECHXAI_PADDING_AFTER_SECONDS,
) -> pd.DataFrame:
    """Convert aligned word dictionaries into effective SpeechXAI mask intervals."""
    rows = []
    for word_index, word in enumerate(_clean_word_segments(words)):
        start = max(0.0, float(word["start"]))
        end = min(audio_duration, float(word["end"]))
        if end <= start:
            continue
        rows.append(
            {
                "audio_id": audio_id,
                "word_index": word_index,
                "word": str(word["word"]),
                "start": start,
                "end": end,
                "masked_start": max(0.0, start - padding_before_seconds),
                "masked_end": min(audio_duration, end + padding_after_seconds),
                "transcription_score": word.get("score", np.nan),
            }
        )
    return pd.DataFrame(rows)


def compute_speechxai_loo_word_scores(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    *,
    audio_id: str,
    words: Iterable[dict],
    target_class: int,
    original_confidence: float,
    device: str,
    padding_before_seconds: float = DEFAULT_SPEECHXAI_PADDING_BEFORE_SECONDS,
    padding_after_seconds: float = DEFAULT_SPEECHXAI_PADDING_AFTER_SECONDS,
) -> pd.DataFrame:
    """Score each aligned word by target-confidence drop after silencing it."""
    audio_duration = waveform.shape[-1] / sampling_rate
    word_table = words_to_timing_table(
        audio_id,
        words,
        audio_duration=audio_duration,
        padding_before_seconds=padding_before_seconds,
        padding_after_seconds=padding_after_seconds,
    )
    if word_table.empty:
        return word_table.assign(
            masked_confidence=pd.Series(dtype=float),
            confidence_drop=pd.Series(dtype=float),
            relative_confidence_drop=pd.Series(dtype=float),
            score=pd.Series(dtype=float),
        )

    rows = []
    for row in word_table.itertuples(index=False):
        mask_intervals = pd.DataFrame(
            {
                "audio_id": [audio_id],
                "start": [float(row.masked_start)],
                "end": [float(row.masked_end)],
                "score": [0.0],
            }
        )
        masked_waveform = silence_intervals(waveform, sampling_rate, mask_intervals)
        masked_prediction = classify_waveform(
            model,
            processor,
            masked_waveform,
            sampling_rate,
            device=device,
            target_class=target_class,
        )
        confidence_drop = original_confidence - masked_prediction.target_confidence
        rows.append(
            {
                **row._asdict(),
                "target_class": target_class,
                "target_label": MODEL_CLASS_NAMES[target_class],
                "masked_duration": total_interval_duration(mask_intervals),
                "original_confidence": original_confidence,
                "masked_confidence": masked_prediction.target_confidence,
                "confidence_drop": confidence_drop,
                "relative_confidence_drop": (
                    confidence_drop / original_confidence
                    if original_confidence > 0
                    else np.nan
                ),
                "masked_predicted_class": masked_prediction.predicted_class,
                "masked_predicted_label": masked_prediction.predicted_label,
                "score": confidence_drop,
            }
        )
    return pd.DataFrame(rows)


def compute_hubert_relevance_scores(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    *,
    device: str,
    target_class: int,
    modes: Iterable[str] = EXPLANATION_MODES,
) -> dict:
    """Compute all requested HuBERT temporal explanation scores for one audio."""
    modes = tuple(modes)
    include_legrad = any(mode in LEGRAD_HUBERT_EXPLANATION_MODES for mode in modes)
    base_result = extract_hubert_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
    )
    grad_result = extract_gradient_weighted_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
        target_class=target_class,
    )
    legrad_result = None
    if include_legrad:
        legrad_result = extract_legrad_hubert_relevance(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            target_class=target_class,
        )
    relevance = compute_relevance_scores(
        model,
        base_result,
        grad_result,
        legrad_result=legrad_result,
    )
    return relevance


def relevance_scores_to_interval_tables(
    audio_id: str,
    score_by_mode: dict[str, torch.Tensor],
    *,
    audio_duration: float,
    modes: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Convert temporal score tensors into uniformly spaced audio intervals."""
    tables = {}
    for mode in modes:
        scores = score_by_mode[mode].squeeze().detach().cpu().numpy()
        table = token_relevance_to_intervals(audio_id, scores, audio_duration)
        table.insert(1, "token_index", np.arange(len(table), dtype=int))
        table.insert(1, "method", mode)
        tables[mode] = table
    return tables


def speechxai_top_k_mask_intervals(word_scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return effective padded SpeechXAI mask intervals for the top-k words."""
    if word_scores.empty:
        return pd.DataFrame(columns=["audio_id", "start", "end", "score"])
    selected = word_scores.sort_values("score", ascending=False).head(k).copy()
    return pd.DataFrame(
        {
            "audio_id": selected["audio_id"],
            "start": selected["masked_start"],
            "end": selected["masked_end"],
            "score": selected["score"],
            "word_index": selected["word_index"],
            "word": selected["word"],
        }
    ).reset_index(drop=True)


def _evaluate_masked_intervals(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    intervals_table: pd.DataFrame,
    *,
    device: str,
    target_class: int,
    original_prediction: PredictionResult,
) -> tuple[PredictionResult, float, float, float]:
    masked_waveform = silence_intervals(waveform, sampling_rate, intervals_table)
    masked_prediction = classify_waveform(
        model,
        processor,
        masked_waveform,
        sampling_rate,
        device=device,
        target_class=target_class,
    )
    masked_duration = total_interval_duration(intervals_table)
    confidence_drop = (
        original_prediction.target_confidence - masked_prediction.target_confidence
    )
    relative_drop = (
        confidence_drop / original_prediction.target_confidence
        if original_prediction.target_confidence > 0
        else np.nan
    )
    return masked_prediction, masked_duration, confidence_drop, relative_drop


def _deletion_record(
    *,
    example: AudioEvaluationExample,
    k: int,
    method: str,
    method_family: str,
    selection: str,
    requested_duration: float,
    masked_duration: float,
    original_prediction: PredictionResult,
    masked_prediction: PredictionResult,
    confidence_drop: float,
    relative_confidence_drop: float,
    random_trial: int | None,
    token_count: int | None = None,
    word_count: int | None = None,
) -> dict:
    return {
        **example.to_record(),
        "k": k,
        "method": method,
        "method_family": method_family,
        "selection": selection,
        "requested_duration": requested_duration,
        "masked_duration": masked_duration,
        "duration_error": masked_duration - requested_duration,
        "original_class": original_prediction.predicted_class,
        "original_label": original_prediction.predicted_label,
        "target_class": original_prediction.target_class,
        "target_label": original_prediction.target_label,
        "original_confidence": original_prediction.target_confidence,
        "masked_confidence": masked_prediction.target_confidence,
        "confidence_drop": confidence_drop,
        "relative_confidence_drop": relative_confidence_drop,
        "masked_predicted_class": masked_prediction.predicted_class,
        "masked_predicted_label": masked_prediction.predicted_label,
        "prediction_flipped": masked_prediction.predicted_class
        != original_prediction.predicted_class,
        "random_trial": random_trial,
        "token_count": token_count,
        "word_count": word_count,
    }


def _selected_interval_records(
    *,
    example: AudioEvaluationExample,
    intervals_table: pd.DataFrame,
    k: int,
    method: str,
    method_family: str,
    selection: str,
    requested_duration: float,
    random_trial: int | None,
) -> list[dict]:
    records = []
    for interval_rank, row in enumerate(intervals_table.itertuples(index=False), start=1):
        row_dict = row._asdict()
        records.append(
            {
                **example.to_record(),
                "k": k,
                "method": method,
                "method_family": method_family,
                "selection": selection,
                "random_trial": random_trial,
                "requested_duration": requested_duration,
                "interval_rank": interval_rank,
                "start": float(row_dict["start"]),
                "end": float(row_dict["end"]),
                "score": float(row_dict.get("score", np.nan)),
                "token_index": row_dict.get("token_index"),
                "word_index": row_dict.get("word_index"),
                "word": row_dict.get("word"),
            }
        )
    return records


def evaluate_duration_matched_audio(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    *,
    example: AudioEvaluationExample,
    words: Iterable[dict],
    transcript: str = "",
    device: str,
    top_k_values: Iterable[int] = DEFAULT_TOP_K_VALUES,
    random_trials: int = 20,
    random_bin_seconds: float = 0.05,
    modes: Iterable[str] = EXPLANATION_MODES,
    include_bottom: bool = True,
    seed: int = 13,
    padding_before_seconds: float = DEFAULT_SPEECHXAI_PADDING_BEFORE_SECONDS,
    padding_after_seconds: float = DEFAULT_SPEECHXAI_PADDING_AFTER_SECONDS,
    original_prediction: PredictionResult | None = None,
) -> DurationMatchedAudioResult:
    """Evaluate SpeechXAI, every HuBERT mode, and random deletion by silence masking."""
    modes = tuple(modes)
    unsupported_modes = sorted(set(modes).difference(EXPLANATION_MODES))
    if unsupported_modes:
        raise ValueError(f"Unsupported explanation modes: {unsupported_modes}")

    if original_prediction is None:
        original_prediction = classify_waveform(
            model,
            processor,
            waveform,
            sampling_rate,
            device=device,
        )
    target_class = original_prediction.target_class
    audio_duration = waveform.shape[-1] / sampling_rate

    word_scores = compute_speechxai_loo_word_scores(
        model,
        processor,
        waveform,
        sampling_rate,
        audio_id=example.audio_id,
        words=words,
        target_class=target_class,
        original_confidence=original_prediction.target_confidence,
        device=device,
        padding_before_seconds=padding_before_seconds,
        padding_after_seconds=padding_after_seconds,
    )
    if word_scores.empty:
        raise RuntimeError(f"No valid SpeechXAI word intervals for {example.audio_id}.")

    relevance = compute_hubert_relevance_scores(
        model,
        processor,
        waveform,
        sampling_rate,
        device=device,
        target_class=target_class,
        modes=modes,
    )
    relevance_tables = relevance_scores_to_interval_tables(
        example.audio_id,
        relevance["scores"],
        audio_duration=audio_duration,
        modes=modes,
    )
    token_count = int(next(iter(relevance_tables.values())).shape[0])
    word_count = int(word_scores.shape[0])

    deletion_records = []
    selected_interval_records = []

    for k in top_k_values:
        speechxai_intervals = speechxai_top_k_mask_intervals(word_scores, int(k))
        requested_duration = total_interval_duration(speechxai_intervals)
        if requested_duration <= 0:
            continue

        masked_prediction, masked_duration, confidence_drop, relative_drop = (
            _evaluate_masked_intervals(
                model,
                processor,
                waveform,
                sampling_rate,
                speechxai_intervals,
                device=device,
                target_class=target_class,
                original_prediction=original_prediction,
            )
        )
        deletion_records.append(
            _deletion_record(
                example=example,
                k=int(k),
                method=SPEECHXAI_PASTOR_METHOD,
                method_family="speechxai_pastor",
                selection="top_words",
                requested_duration=requested_duration,
                masked_duration=masked_duration,
                original_prediction=original_prediction,
                masked_prediction=masked_prediction,
                confidence_drop=confidence_drop,
                relative_confidence_drop=relative_drop,
                random_trial=None,
                token_count=token_count,
                word_count=word_count,
            )
        )
        selected_interval_records.extend(
            _selected_interval_records(
                example=example,
                intervals_table=speechxai_intervals,
                k=int(k),
                method=SPEECHXAI_PASTOR_METHOD,
                method_family="speechxai_pastor",
                selection="top_words",
                requested_duration=requested_duration,
                random_trial=None,
            )
        )

        for mode in modes:
            for selection in ("top", "bottom") if include_bottom else ("top",):
                selected_intervals = select_intervals_by_duration(
                    relevance_tables[mode],
                    requested_duration,
                    mode=selection,
                )
                masked_prediction, masked_duration, confidence_drop, relative_drop = (
                    _evaluate_masked_intervals(
                        model,
                        processor,
                        waveform,
                        sampling_rate,
                        selected_intervals,
                        device=device,
                        target_class=target_class,
                        original_prediction=original_prediction,
                    )
                )
                deletion_records.append(
                    _deletion_record(
                        example=example,
                        k=int(k),
                        method=mode,
                        method_family="hubert_temporal_relevance",
                        selection=selection,
                        requested_duration=requested_duration,
                        masked_duration=masked_duration,
                        original_prediction=original_prediction,
                        masked_prediction=masked_prediction,
                        confidence_drop=confidence_drop,
                        relative_confidence_drop=relative_drop,
                        random_trial=None,
                        token_count=token_count,
                        word_count=word_count,
                    )
                )
                selected_interval_records.extend(
                    _selected_interval_records(
                        example=example,
                        intervals_table=selected_intervals,
                        k=int(k),
                        method=mode,
                        method_family="hubert_temporal_relevance",
                        selection=selection,
                        requested_duration=requested_duration,
                        random_trial=None,
                    )
                )

        for random_trial in range(random_trials):
            random_intervals = random_intervals_by_duration(
                example.audio_id,
                audio_duration,
                requested_duration,
                bin_seconds=random_bin_seconds,
                seed=stable_seed(seed, example.dataset, example.audio_id, k, random_trial),
            )
            masked_prediction, masked_duration, confidence_drop, relative_drop = (
                _evaluate_masked_intervals(
                    model,
                    processor,
                    waveform,
                    sampling_rate,
                    random_intervals,
                    device=device,
                    target_class=target_class,
                    original_prediction=original_prediction,
                )
            )
            deletion_records.append(
                _deletion_record(
                    example=example,
                    k=int(k),
                    method=RANDOM_DURATION_METHOD,
                    method_family="random_baseline",
                    selection="random",
                    requested_duration=requested_duration,
                    masked_duration=masked_duration,
                    original_prediction=original_prediction,
                    masked_prediction=masked_prediction,
                    confidence_drop=confidence_drop,
                    relative_confidence_drop=relative_drop,
                    random_trial=random_trial,
                    token_count=token_count,
                    word_count=word_count,
                )
            )
            selected_interval_records.extend(
                _selected_interval_records(
                    example=example,
                    intervals_table=random_intervals,
                    k=int(k),
                    method=RANDOM_DURATION_METHOD,
                    method_family="random_baseline",
                    selection="random",
                    requested_duration=requested_duration,
                    random_trial=random_trial,
                )
            )

    word_scores = word_scores.assign(
        dataset=example.dataset,
        audio_path=example.audio_path,
        transcript=transcript,
    )

    return DurationMatchedAudioResult(
        deletion_records=pd.DataFrame(deletion_records),
        speechxai_word_scores=word_scores,
        selected_intervals=pd.DataFrame(selected_interval_records),
        relevance_tables=relevance_tables,
        original_prediction=original_prediction,
        transcript=transcript,
    )


def summarize_duration_matched_records(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duration-matched deletion records by dataset/method/k."""
    if records.empty:
        return pd.DataFrame()
    return (
        records.groupby(
            ["dataset", "method", "method_family", "selection", "k"],
            as_index=False,
        )
        .agg(
            audios=("audio_id", "nunique"),
            rows=("audio_id", "size"),
            mean_requested_duration=("requested_duration", "mean"),
            mean_masked_duration=("masked_duration", "mean"),
            mean_duration_error=("duration_error", "mean"),
            mean_confidence_drop=("confidence_drop", "mean"),
            std_confidence_drop=("confidence_drop", "std"),
            mean_relative_confidence_drop=("relative_confidence_drop", "mean"),
            prediction_flip_rate=("prediction_flipped", "mean"),
        )
        .fillna(0)
        .sort_values(["dataset", "k", "method_family", "method", "selection"])
        .reset_index(drop=True)
    )
