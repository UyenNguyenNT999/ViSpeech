"""Paired original/background-noise views; the original dataset is unchanged."""
from pathlib import Path
import random

import librosa
import numpy as np
import torch

from dataset import LoadDataAndLabel


def mix_at_snr(original, noise, snr_db):
    """Whole-clip RMS SNR, including silence. Apply shared peak scaling if needed."""
    original = np.asarray(original, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    speech_power = float(np.mean(original.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if speech_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("Cannot mix a silent speech/noise segment at a defined SNR.")
    gain = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    added = noise * gain
    mixed = original + added
    scale = max(1.0, float(np.max(np.abs(mixed))), float(np.max(np.abs(original))))
    # Scaling both components equally preserves SNR and avoids clipping distortion.
    measured = 10 * np.log10(speech_power / float(np.mean(added.astype(np.float64) ** 2)))
    return original / scale, mixed / scale, float(measured), float(scale)


class PairedDataset(LoadDataAndLabel):
    def __init__(self, noise_path, snr_db_min, snr_db_max, noise_seed, **kwargs):
        super().__init__(enable_augmentation=False, **kwargs)
        self.noise_files = sorted(p.resolve() for p in Path(noise_path).rglob("*")
                                  if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"})
        if not self.noise_files:
            raise ValueError(f"No noise files found: {noise_path}")
        self.snr_min, self.snr_max = float(snr_db_min), float(snr_db_max)
        if not np.isfinite([self.snr_min, self.snr_max]).all() or self.snr_min > self.snr_max:
            raise ValueError("Invalid SNR range.")
        # Separate RNG: adding a loss must not change the audio draws in A/B.
        self.rng = random.Random(noise_seed)
        self.noise_cache = {}

    def __getitem__(self, idx):
        original = self._load_audio(self.file_list[idx])
        noise_path = self.rng.choice(self.noise_files)
        if noise_path not in self.noise_cache:
            waveform, _ = librosa.load(str(noise_path), sr=self.sampling_rate)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid noise: {noise_path}")
            self.noise_cache[noise_path] = waveform
        noise = self.noise_cache[noise_path]
        if len(noise) < len(original):
            noise = np.pad(noise, (0, len(original) - len(noise)), mode="wrap")
        offset = self.rng.randint(0, len(noise) - len(original))
        snr = self.rng.uniform(self.snr_min, self.snr_max)
        original, mixed, measured, scale = mix_at_snr(
            original, noise[offset:offset + len(original)], snr)
        metadata = {"speech_file": str(self.file_list[idx]), "noise_file": str(noise_path),
                    "noise_offset_samples": offset, "snr_db": snr,
                    "measured_snr_db": measured, "shared_peak_scale": scale}
        return (torch.from_numpy(original.astype(np.float32)),
                torch.from_numpy(mixed.astype(np.float32)),
                self.get_label(self.label_list[idx]), metadata)
