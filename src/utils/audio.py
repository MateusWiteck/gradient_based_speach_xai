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
        try:
            import soundfile as sf

            audio, sr = sf.read(audio_path, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio.T)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            if sr != target_sr:
                try:
                    import torchaudio

                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                    sr = target_sr
                except Exception:
                    from scipy.signal import resample_poly
                    import math

                    divisor = math.gcd(sr, target_sr)
                    waveform_np = resample_poly(
                        waveform.numpy(),
                        target_sr // divisor,
                        sr // divisor,
                    )
                    waveform = torch.from_numpy(waveform_np).to(torch.float32)
                    sr = target_sr
            return waveform.to(torch.float32), sr
        except Exception:
            # Fallback to librosa if it is available in the environment.
            import librosa

            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(y).to(torch.float32)
            return waveform, sr
