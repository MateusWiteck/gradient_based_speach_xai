import numpy as np
import soundfile as sf


def generate_sine(path: str = "data/sample.wav", duration: float = 1.0, sr: int = 16000, freq: float = 440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * freq * t)
    sf.write(path, y, sr)


if __name__ == "__main__":
    generate_sine()
