"""Evaluate ViMD recordings by feeding each WHOLE recording to the model once.

Differs from evaluate_vimd_full_recording.py ONLY in how a recording is fed:
- full_recording: split into 110,332-sample windows, average window probabilities.
- whole_input (this script): ONE forward pass on the entire waveform, no cutting,
  no averaging. Recordings shorter than the training length are wrap-padded to it
  (same rule as dataset.py / prepare_speech); longer recordings are NOT cut.
Batch size is fixed to 1 because recordings have different lengths.

Evaluate: uv run --locked python evaluate_vimd_whole_input.py --model A
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
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
    # Training length; used ONLY as a minimum (short files are padded, long ones kept whole).
    min_samples = int(audio["length_seconds"] * 100) * audio["hop_length"]
    min_samples += audio["window_length"] - audio["hop_length"]
    print("Model:", args.model, "| Checkpoint:", checkpoint, flush=True)
    print("Recordings:", len(samples), "| Labels:", dict(Counter(label for _, label in samples)))
    print("Whole recording per forward pass; minimum length {} samples ({:.6f}s).".format(
        min_samples, min_samples / sr), flush=True)
    if args.check_only:
        print("Input paths and configuration checked; no inference performed.")
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    output = ROOT / "logs" / ("vimd_whole_input_" + args.model) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print("Results:", output, flush=True)
    info = {"status": "running", "model": args.model, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "dataset": str(listing),
            "dataset_sha256": file_hash(listing), "recordings": len(samples),
            "audio": audio, "model_settings": settings, "min_samples": min_samples,
            "input": "whole recording, one forward pass, no windowing or averaging",
            "padding": "wrap only if shorter than min_samples", "truncation": False,
            "added_noise": False, "silence_removal": False, "seed": 42, "batch_size": 1,
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
        records = []
        with torch.inference_mode():
            for audio_id, (path, label) in enumerate(tqdm(samples, desc="ViMD whole input")):
                waveform = load_audio(path, sr)
                real_samples = len(waveform)
                if real_samples < min_samples:
                    waveform = np.pad(waveform, (0, min_samples - real_samples), mode="wrap")
                x = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)[None, :]).to(device)
                probs = model(x).exp().cpu().numpy()[0].astype(np.float64)
                if probs.shape != (3,) or not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-5):
                    raise ValueError("Invalid probabilities for " + str(path))
                predicted = int(np.argmax(probs))
                records.append({"audio_id": audio_id, "audio_path": str(path),
                                "audio_sha256": file_hash(path), "true_label": label,
                                "predicted_label": predicted, "correct": int(predicted == label),
                                "duration_seconds": real_samples / sr, "input_samples": len(waveform),
                                "padded_samples": len(waveform) - real_samples,
                                "nll": float(-np.log(max(probs[label], 1e-300))),
                                **{"prob_" + name: float(probs[j]) for j, name in enumerate(DIALECT_NAMES)}})
        truth = [r["true_label"] for r in records]
        predicted = [r["predicted_label"] for r in records]
        recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
        summary = {"model": args.model, "condition": "vimd_whole_input",
                   "n_recordings": len(records),
                   "accuracy": float(accuracy_score(truth, predicted)),
                   "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
                   "nll": float(np.mean([r["nll"] for r in records])),
                   **{"recall_" + name: float(recalls[j]) for j, name in enumerate(DIALECT_NAMES)}}
        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
        write_csv(output / "summary.csv", [summary])
        write_csv(output / "predictions.csv", records)
        write_csv(output / "confusion_matrix.csv", [
            {"true_dialect": name, **{other: int(matrix[i, j]) for j, other in enumerate(DIALECT_NAMES)}}
            for i, name in enumerate(DIALECT_NAMES)])
        draw_matrix(matrix, output / "confusion_matrix.svg", args.model + " - ViMD whole input")
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
