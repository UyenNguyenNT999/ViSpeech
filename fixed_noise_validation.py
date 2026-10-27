"""Replay fixed validation mixtures using the original evaluation preprocessing."""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

from evaluate_noise_robustness import read_inputs, prepare_speech, load_audio, mix_noise
from prepare_fixed_noise_validation import ROOT, resolve, verify_bundle


def metric_values(truth, predicted, total_loss):
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
            "loss": total_loss / len(truth),
            **dict(zip(("recall_north", "recall_central", "recall_south"),
                       map(float, recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0))))}


def choose_checkpoints(clean, noisy_f1, threshold, best_clean, best_robust):
    """Strict improvements; ties keep earlier checkpoint. Threshold is fixed at step 0."""
    eligible = clean["accuracy"] >= threshold
    return (best_clean is None or clean["accuracy"] > best_clean,
            eligible and (best_robust is None or noisy_f1 > best_robust), eligible)


class FixedNoiseValidation:
    def __init__(self, directory):
        self.directory = resolve(directory)
        self.manifest = verify_bundle(self.directory)
        self.config = yaml.safe_load((self.directory / "evaluation_config.yaml").read_text(encoding="utf-8"))
        self.samples, self.noises, _, _ = read_inputs(self.config, ROOT)
        a = self.config["audio"]
        self.sr = a["sampling_rate"]
        self.count = int(a["length_seconds"] * 100) * a["hop_length"] + a["window_length"] - a["hop_length"]
        self.groups = {}
        with (self.directory / "mixture_plan.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["noise_id"], float(row["snr_db"]), int(row["repeat_id"]))
                self.groups.setdefault(key, []).append(row)
        for rows in self.groups.values():
            rows.sort(key=lambda r: int(r["audio_id"]))
            if [int(r["audio_id"]) for r in rows] != list(range(len(self.samples))):
                raise ValueError("Validation condition must include each original exactly once")
        self.speech = None
        self.noise = None

    def load_waveforms(self):
        if self.speech is None:
            self.speech = [prepare_speech(s["audio_path"], self.sr, self.count)
                           for s in tqdm(self.samples, desc="Cache validation originals")]
            self.noise = {n["noise_id"]: load_audio(n["path"], self.sr) for n in self.noises}

    def mixed(self, row):
        speech = self.speech[int(row["audio_id"])]
        start = int(row["noise_start_sample"])
        if int(row["noise_samples"]) != self.count:
            raise ValueError("Sample count differs from frozen plan")
        noise = self.noise[row["noise_id"]][start:start + self.count]
        if not len(noise):
            raise ValueError("Empty noise segment in frozen plan")
        if len(noise) < self.count:
            noise = np.pad(noise, (0, self.count - len(noise)), mode="wrap")
        waveform, scale = mix_noise(speech, noise, float(row["snr_db"]))
        if not np.isclose(scale, float(row["noise_scale"]), rtol=1e-6, atol=1e-10):
            raise ValueError("Noise scale differs from historical plan: audio/preprocessing changed")
        return waveform

    def evaluate(self, model, device, clean_batch_size, noisy_batch_size):
        # Evaluation must not alter training RNG streams (sampler, dropout, augmentation).
        self.load_waveforms()
        model.eval()
        truth = [s["true_label"] for s in self.samples]

        def score(rows, batch_size):
            predicted, total_loss = [], 0.0
            for start in range(0, len(truth), batch_size):
                waves = self.speech[start:start + batch_size] if rows is None else [
                    self.mixed(row) for row in rows[start:start + batch_size]]
                batch = torch.from_numpy(np.stack(waves)).to(device)
                labels = torch.tensor(truth[start:start + batch_size], device=device)
                outputs = model(batch)
                if not torch.isfinite(outputs).all():
                    raise ValueError("Non-finite validation output")
                total_loss += F.nll_loss(outputs, labels, reduction="sum").item()
                predicted.extend(outputs.argmax(1).cpu().tolist())
            return metric_values(truth, predicted, total_loss)

        with torch.no_grad():
            clean = score(None, clean_batch_size)
            details = []
            for (noise_id, snr, repeat), rows in tqdm(self.groups.items(), desc="Fixed noisy validation"):
                details.append(dict(noise_id=noise_id, snr_db=snr, repeat_id=repeat,
                                    **score(rows, noisy_batch_size)))
        means = {snr: float(np.mean([r["macro_f1"] for r in details if r["snr_db"] == snr]))
                 for snr in sorted({r["snr_db"] for r in details})}
        return clean, float(np.mean([r["macro_f1"] for r in details])), means, details
