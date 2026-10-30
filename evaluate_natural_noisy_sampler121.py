"""Evaluate the original sampler 1:2:1 checkpoint on natural noisy test audio.

Same evaluation implementation as evaluate_natural_noisy_testset.py.
Run: uv run --locked python evaluate_natural_noisy_sampler121.py
Reads model/audio settings from config_noise_eval.yaml; no added noise.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm import tqdm

from evaluate_noise_robustness import (
    DIALECT_NAMES, Hub_Model, draw_matrix, file_hash, prepare_speech, write_csv,
)

ROOT = Path(__file__).resolve().parent


def read_samples():
    listing = ROOT / "dataset" / "noisy_testset.txt"
    samples, seen = [], set()
    with listing.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            label = int(row[1])
            if label not in range(6):
                raise ValueError("Invalid label: " + str(row))
            path = (ROOT / row[0]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in seen:
                raise ValueError("Duplicate audio: " + str(path))
            seen.add(path)
            samples.append({"audio_id": len(samples), "audio_path": str(path),
                            "true_label": label % 3})
    if {s["true_label"] for s in samples} != {0, 1, 2}:
        raise ValueError("Expected a nonempty test set with all three dialects.")
    return listing, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sampler121",), default="sampler121")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    # Reuse ONLY model/audio settings. Ignore the old dataset/noise/output fields.
    config = yaml.safe_load((ROOT / "config_noise_eval.yaml")
                            .read_text(encoding="utf-8-sig"))
    checkpoint = (ROOT / config["model"]["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    listing, samples = read_samples()
    audio = config["audio"]
    sample_count = (int(audio["length_seconds"] * 100) * audio["hop_length"]
                    + audio["window_length"] - audio["hop_length"])
    print("Model:", args.model, "| Checkpoint:", checkpoint, flush=True)
    print("Dataset:", listing, "| Samples:", len(samples), flush=True)
    print("Counts:", dict(Counter(DIALECT_NAMES[s["true_label"]] for s in samples)))
    print("Existing background noise only. No added noise; SNR is unknown.", flush=True)
    if args.check_only:
        print("Input check passed. No inference performed.")
        return

    output = ROOT / "logs" / ("natural_noisy_test_" + args.model) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print("Results:", output, flush=True)
    info = {"status": "running", "model": args.model, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "dataset": str(listing),
            "dataset_sha256": file_hash(listing), "n_samples": len(samples),
            "audio": audio, "sample_count": sample_count, "model_settings": config["model"],
            "batch_size": args.batch_size, "seed": 42, "added_noise": False,
            "snr_db": None, "labels": dict(enumerate(DIALECT_NAMES)),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    info_path = output / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if args.device == "auto" else args.device)
        info["device"] = str(device)
        model = Hub_Model(n_classes=3, n_mels=audio["n_mels"],
                          dropout_rate=config["model"]["dropout"],
                          sampling_rate=audio["sampling_rate"], n_fft=audio["n_fft"],
                          window=audio["window_length"], step=audio["hop_length"],
                          model_type="resnet34", version=config["model"]["version"])
        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
        model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(samples), args.batch_size), desc="Natural noisy test"):
                batch = samples[start:start + args.batch_size]
                waves = np.stack([prepare_speech(s["audio_path"], audio["sampling_rate"],
                                                sample_count) for s in batch])
                log_probs = model(torch.from_numpy(waves).to(device)).cpu().numpy()
                if not np.isfinite(log_probs).all():
                    raise ValueError("Non-finite model outputs.")
                for sample, values in zip(batch, log_probs):
                    predicted = int(np.argmax(values))
                    rows.append({**sample, "predicted_label": predicted,
                                 "correct": int(predicted == sample["true_label"]),
                                 "nll": float(-values[sample["true_label"]]),
                                 **{"prob_" + name: float(np.exp(values[i]))
                                    for i, name in enumerate(DIALECT_NAMES)}})
        truth = [r["true_label"] for r in rows]
        predicted = [r["predicted_label"] for r in rows]
        recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
        summary = {"model": args.model, "condition": "natural_noisy_test",
                   "n_samples": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
                   "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2],
                                               average="macro", zero_division=0)),
                   "nll": float(np.mean([r["nll"] for r in rows])),
                   **{"recall_" + name: float(recalls[i]) for i, name in enumerate(DIALECT_NAMES)}}
        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
        write_csv(output / "predictions.csv", rows)
        write_csv(output / "summary.csv", [summary])
        write_csv(output / "confusion_matrix.csv", [
            {"true_dialect": name, **{other: int(matrix[i, j])
                                      for j, other in enumerate(DIALECT_NAMES)}}
            for i, name in enumerate(DIALECT_NAMES)])
        draw_matrix(matrix, output / "confusion_matrix.svg", args.model + " - natural noisy test")
        info["status"] = "completed"
        print("Accuracy: {:.2%} | Macro-F1: {:.2%} | NLL: {:.4f}".format(
            summary["accuracy"], summary["macro_f1"], summary["nll"]))
        for name in DIALECT_NAMES:
            print("Recall {}: {:.2%}".format(name, summary["recall_" + name]))
        print("Completed:", output)
    except Exception as exc:
        info.update(status="failed", error=repr(exc))
        raise
    finally:
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
