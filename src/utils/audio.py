import torch


def load_audio_mono_16k(audio_path: str, target_sr: int = 16000):
    try:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)

        # mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        # resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        return waveform, sr
    except Exception:
        # Fallback to librosa if torchaudio is not usable (e.g., missing torchcodec)
        import librosa

        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        waveform = torch.from_numpy(y).to(torch.float32)
        return waveform, sr