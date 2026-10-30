"""Evaluate original ViMD recordings using all non-overlapping audio windows.

Prepare once: uv run --script prepare_vimd_test.py
Evaluate: uv run --locked python evaluate_vimd_full_recording.py --model A
Each recording gets ONE prediction: duration-weighted mean window probabilities.
No added noise, silence removal, training, or changes to existing scripts.
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
    DIALECT_NAMES, Hub_Model, draw_matrix, file_hash, load_audio, write_csv,
)

ROOT = Path(__file__).resolve().parent
CONFIGS = {"sampler121": "config_noise_eval.yaml", "M0": "config_noise_test_M0.yaml",
           "A": "config_noise_test_A.yaml", "A121": "config_noise_test_A121.yaml"}


def split_recording(waveform, window_samples):
    """Cover each real sample once. Repeat only the short final window's content."""
    if window_samples <= 0 or waveform.ndim != 1 or len(waveform) == 0:
        raise ValueError("Expected nonempty mono audio and a positive window size.")
    if not np.isfinite(waveform).all():
        raise ValueError("Non-finite audio.")
    for start in range(0, len(waveform), window_samples):
        segment = waveform[start:start + window_samples]
        real_samples = len(segment)
        if real_samples < window_samples:
            segment = np.pad(segment, (0, window_samples - real_samples), mode="wrap")
        yield start, real_samples, segment


def aggregate_probabilities(probabilities, real_samples):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(real_samples, dtype=np.float64)
    if (probabilities.shape != (len(weights), 3) or len(weights) == 0
            or not np.isfinite(probabilities).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Invalid probabilities or duration weights.")
    return np.average(probabilities, axis=0, weights=weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("Batch size must be positive.")
    listing = ROOT / "dataset/vimd_test.txt"
    if not listing.is_file():
        parser.error("ViMD audio not prepared. Run: uv run --script prepare_vimd_test.py")
    config = yaml.safe_load((ROOT / CONFIGS[args.model]).read_text(encoding="utf-8-sig"))
    checkpoint = (ROOT / config["model"]["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    samples, seen = [], set()
    with listing.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            path, label = (ROOT / row[0]).resolve(), int(row[1])
            if label not in (0, 1, 2) or path in seen:
                raise ValueError("Invalid label or duplicate path: " + str(row))
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(path)
            samples.append((path, label))
    if {label for _, label in samples} != {0, 1, 2}:
        raise ValueError("Expected all three dialects.")
    audio, settings = config["audio"], config["model"]
    sr = audio["sampling_rate"]
    window = int(audio["length_seconds"] * 100) * audio["hop_length"]
    window += audio["window_length"] - audio["hop_length"]
    print("Model:", args.model, "| Checkpoint:", checkpoint, flush=True)
    print("Recordings:", len(samples), "| Labels:", dict(Counter(label for _, label in samples)))
    print("Window: {} samples ({:.6f}s). Original audio only; one prediction per recording.".format(
        window, window / sr), flush=True)
    if args.check_only:
        print("Input paths and configuration checked; no inference performed.")
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    output = ROOT / "logs" / ("vimd_full_recording_" + args.model) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print("Results:", output, flush=True)
    info = {"status": "running", "model": args.model, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "dataset": str(listing),
            "dataset_sha256": file_hash(listing), "recordings": len(samples),
            "audio": audio, "model_settings": settings, "window_samples": window,
            "aggregation": "mean probabilities weighted by real sample count, before padding",
            "padding": "wrap final segment", "overlap": False, "added_noise": False,
            "silence_removal": False, "seed": 42, "batch_size": args.batch_size,
            "device": str(device), "torch_version": str(torch.__version__),
            "labels": dict(enumerate(DIALECT_NAMES))}
    data_root = ROOT.parents[1] / "Data/ViMD"
    for filename in ("download_source.json", "download_manifest_test.json"):
        if (data_root / filename).is_file():
            info[filename] = json.loads((data_root / filename).read_text(encoding="utf-8"))
    info_path = output / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = Hub_Model(n_classes=3, n_mels=audio["n_mels"],
                          dropout_rate=settings["dropout"], sampling_rate=sr,
                          n_fft=audio["n_fft"], window=audio["window_length"],
                          step=audio["hop_length"], model_type="resnet34", version=settings["version"])
        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
        model.to(device).eval()
        records, segments = [], []
        with torch.inference_mode():
            for audio_id, (path, label) in enumerate(tqdm(samples, desc="ViMD recordings")):
                waveform = load_audio(path, sr)
                windows = list(split_recording(waveform, window))
                probabilities = []
                for start in range(0, len(windows), args.batch_size):
                    batch = np.stack([x[2] for x in windows[start:start + args.batch_size]])
                    probabilities.extend(model(torch.from_numpy(batch).to(device)).exp().cpu().numpy())
                weights = [x[1] for x in windows]
                averaged = aggregate_probabilities(probabilities, weights)
                predicted = int(np.argmax(averaged))
                records.append({"audio_id": audio_id, "audio_path": str(path),
                                "audio_sha256": file_hash(path), "true_label": label,
                                "predicted_label": predicted, "correct": int(predicted == label),
                                "duration_seconds": len(waveform) / sr, "segment_count": len(windows),
                                "nll": float(-np.log(max(averaged[label], 1e-300))),
                                **{"prob_" + name: float(averaged[j]) for j, name in enumerate(DIALECT_NAMES)}})
                for segment_id, ((start, length, _), probs) in enumerate(zip(windows, probabilities)):
                    segments.append({"audio_id": audio_id, "segment_id": segment_id,
                                     "start_sample": start, "real_samples": length,
                                     "padded_samples": window - length, "weight": length / len(waveform),
                                     "true_label": label, "predicted_label": int(np.argmax(probs)),
                                     **{"prob_" + name: float(probs[j]) for j, name in enumerate(DIALECT_NAMES)}})
        truth = [r["true_label"] for r in records]
        predicted = [r["predicted_label"] for r in records]
        recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
        summary = {"model": args.model, "condition": "vimd_original_full_recording",
                   "n_recordings": len(records), "n_segments": len(segments),
                   "accuracy": float(accuracy_score(truth, predicted)),
                   "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
                   "nll": float(np.mean([r["nll"] for r in records])),
                   **{"recall_" + name: float(recalls[j]) for j, name in enumerate(DIALECT_NAMES)}}
        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
        write_csv(output / "summary.csv", [summary])
        write_csv(output / "predictions.csv", records)
        write_csv(output / "segment_predictions.csv", segments)
        write_csv(output / "confusion_matrix.csv", [
            {"true_dialect": name, **{other: int(matrix[i, j]) for j, other in enumerate(DIALECT_NAMES)}}
            for i, name in enumerate(DIALECT_NAMES)])
        draw_matrix(matrix, output / "confusion_matrix.svg", args.model + " - ViMD whole recordings")
        print("Accuracy: {:.2%} | Macro-F1: {:.2%}".format(summary["accuracy"], summary["macro_f1"]))
        print("Completed:", output)
        info["status"] = "completed"
    except Exception as exc:
        info.update(status="failed", error=repr(exc))
        raise
    finally:
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
