"""Đánh giá checkpoint wav2vec2 đã fine-tune trên clean test, noisy test (nhiễu tự nhiên) và ViMD test.

Chạy từ Code/ViSpeech:
    uv run --locked python evaluate_wav2vec2.py --checkpoint checkpoints/dialect/wav2vec2_base_vi_<...>/best --set all
    uv run --locked python evaluate_wav2vec2.py --checkpoint ... --set noisy --check-only

Giữ đúng giao thức của các script đánh giá ResNet34 hiện có:
  - clean/noisy: một đoạn ~5 giây, cắt/lặp 'wrap' bằng prepare_speech (như dataset.py);
  - ViMD: chia toàn bộ bản ghi thành các cửa sổ không chồng lấn, trung bình xác suất có trọng số
    theo số mẫu thật (split_recording, aggregate_probabilities của evaluate_vimd_full_recording.py).
Chỉ khác sample rate: 16 kHz (yêu cầu của wav2vec2). Không thêm nhiễu, không train.
Kết quả ghi vào logs/<điều kiện>_<model-label>/<thời điểm>/ với cùng cột summary như các file cũ.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_noise_robustness import (  # noqa: E402
    DIALECT_NAMES, draw_matrix, file_hash, load_audio, prepare_speech, write_csv,
)
from evaluate_vimd_full_recording import aggregate_probabilities, split_recording  # noqa: E402

SAMPLING_RATE = 16000
SAMPLE_COUNT = 5 * 100 * 160 + 400 - 160  # 80.240 mẫu, cùng công thức với train_wav2vec2.py
SETS = {
    "clean": ("dataset/clean_testset.txt", "clean_test"),
    "noisy": ("dataset/noisy_testset.txt", "natural_noisy_test"),
    "vimd": ("dataset/vimd_test.txt", "vimd_original_full_recording"),
}


def read_listing(listing, labels_mod3):
    samples, seen = [], set()
    with listing.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            label = int(row[1])
            if label not in (range(6) if labels_mod3 else range(3)):
                raise ValueError("Nhãn không hợp lệ: " + str(row))
            path = (ROOT / row[0]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path in seen:
                raise ValueError("Trùng file: " + str(path))
            seen.add(path)
            samples.append({"audio_id": len(samples), "audio_path": str(path),
                            "true_label": label % 3})
    if {s["true_label"] for s in samples} != {0, 1, 2}:
        raise ValueError("Cần đủ ba phương ngữ trong " + str(listing))
    return samples


def normalize(waves, enabled):
    if not enabled:
        return waves
    mean = waves.mean(axis=1, keepdims=True)
    std = np.sqrt(waves.var(axis=1, keepdims=True) + 1e-7)
    return (waves - mean) / std


def predict(model, waves, device, do_normalize):
    """Trả về xác suất (softmax) dạng numpy cho một batch waveform 16 kHz."""
    batch = torch.from_numpy(normalize(np.stack(waves).astype(np.float32), do_normalize)).to(device)
    probs = torch.softmax(model(input_values=batch).logits.float(), dim=-1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Đầu ra model không hữu hạn.")
    return probs


def summarize(rows, model_label, condition, count_key):
    truth = [r["true_label"] for r in rows]
    predicted = [r["predicted_label"] for r in rows]
    recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    summary = {"model": model_label, "condition": condition, count_key: len(rows),
               "accuracy": float(accuracy_score(truth, predicted)),
               "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2],
                                           average="macro", zero_division=0)),
               "nll": float(np.mean([r["nll"] for r in rows])),
               **{"recall_" + name: float(recalls[i]) for i, name in enumerate(DIALECT_NAMES)}}
    return summary, confusion_matrix(truth, predicted, labels=[0, 1, 2])


def evaluate_set(name, model, args, device, do_normalize, checkpoint):
    listing_rel, condition = SETS[name]
    listing = ROOT / listing_rel
    if not listing.is_file():
        raise FileNotFoundError(listing)
    samples = read_listing(listing, labels_mod3=(name != "vimd"))
    print("\n[{}] {} mẫu | {}".format(name, len(samples),
                                       dict(Counter(DIALECT_NAMES[s["true_label"]] for s in samples))),
          flush=True)
    if args.check_only:
        return None
    output = ROOT / "logs" / "{}_{}".format(condition, args.model_label) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    info = {"status": "running", "model": args.model_label, "checkpoint": str(checkpoint),
            "dataset": str(listing), "dataset_sha256": file_hash(listing), "condition": condition,
            "n_samples": len(samples), "sampling_rate": SAMPLING_RATE, "sample_count": SAMPLE_COUNT,
            "do_normalize": do_normalize, "added_noise": False, "seed": 42,
            "batch_size": args.batch_size, "device": str(device),
            "labels": dict(enumerate(DIALECT_NAMES)), "torch_version": str(torch.__version__)}
    if name == "vimd":
        info.update(aggregation="mean probabilities weighted by real sample count, before padding",
                    padding="wrap final segment", overlap=False)
    info_path = output / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    rows, segments = [], []
    try:
        with torch.inference_mode():
            if name != "vimd":
                for start in tqdm(range(0, len(samples), args.batch_size), desc=name):
                    batch = samples[start:start + args.batch_size]
                    waves = [prepare_speech(s["audio_path"], SAMPLING_RATE, SAMPLE_COUNT) for s in batch]
                    for sample, probs in zip(batch, predict(model, waves, device, do_normalize)):
                        predicted = int(np.argmax(probs))
                        rows.append({**sample, "predicted_label": predicted,
                                     "correct": int(predicted == sample["true_label"]),
                                     "nll": float(-np.log(max(probs[sample["true_label"]], 1e-300))),
                                     **{"prob_" + n: float(probs[i]) for i, n in enumerate(DIALECT_NAMES)}})
                count_key = "n_samples"
            else:
                for sample in tqdm(samples, desc=name):
                    waveform = load_audio(sample["audio_path"], SAMPLING_RATE)
                    windows = list(split_recording(waveform, SAMPLE_COUNT))
                    probabilities = []
                    for start in range(0, len(windows), args.batch_size):
                        chunk = windows[start:start + args.batch_size]
                        probabilities.extend(predict(model, [w[2] for w in chunk], device, do_normalize))
                    averaged = aggregate_probabilities(probabilities, [w[1] for w in windows])
                    predicted = int(np.argmax(averaged))
                    label = sample["true_label"]
                    rows.append({**sample, "predicted_label": predicted, "correct": int(predicted == label),
                                 "duration_seconds": len(waveform) / SAMPLING_RATE,
                                 "segment_count": len(windows),
                                 "nll": float(-np.log(max(averaged[label], 1e-300))),
                                 **{"prob_" + n: float(averaged[i]) for i, n in enumerate(DIALECT_NAMES)}})
                    for segment_id, ((start, length, _), probs) in enumerate(zip(windows, probabilities)):
                        segments.append({"audio_id": sample["audio_id"], "segment_id": segment_id,
                                         "start_sample": start, "real_samples": length,
                                         "true_label": label, "predicted_label": int(np.argmax(probs)),
                                         **{"prob_" + n: float(probs[i]) for i, n in enumerate(DIALECT_NAMES)}})
                count_key = "n_recordings"
        summary, matrix = summarize(rows, args.model_label, condition, count_key)
        if segments:
            summary["n_segments"] = len(segments)
            write_csv(output / "segment_predictions.csv", segments)
        summary["results_directory"] = str(output)
        write_csv(output / "predictions.csv", rows)
        write_csv(output / "summary.csv", [summary])
        write_csv(output / "confusion_matrix.csv", [
            {"true_dialect": n, **{o: int(matrix[i, j]) for j, o in enumerate(DIALECT_NAMES)}}
            for i, n in enumerate(DIALECT_NAMES)])
        draw_matrix(matrix, output / "confusion_matrix.svg", "{} - {}".format(args.model_label, condition))
        info["status"] = "completed"
        print("Accuracy: {:.2%} | Macro-F1: {:.2%} | NLL: {:.4f} | Recall N/C/S: {}".format(
            summary["accuracy"], summary["macro_f1"], summary["nll"],
            " / ".join("{:.2%}".format(summary["recall_" + n]) for n in DIALECT_NAMES)))
        print("Kết quả:", output)
        return summary
    except Exception as exc:
        info.update(status="failed", error=repr(exc))
        raise
    finally:
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Thư mục best/ hoặc last/ của train_wav2vec2.py")
    parser.add_argument("--set", choices=("clean", "noisy", "vimd", "all"), default="all")
    parser.add_argument("--model-label", default="W2V")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")
    checkpoint = (ROOT / args.checkpoint).resolve()
    if not (checkpoint / "config.json").is_file():
        parser.error("Không thấy config.json trong " + str(checkpoint))

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(checkpoint))
    if feature_extractor.sampling_rate != SAMPLING_RATE:
        raise ValueError("Checkpoint không phải 16 kHz.")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(checkpoint))
    if model.config.num_labels != 3 or [model.config.id2label[i] for i in range(3)] != DIALECT_NAMES:
        raise ValueError("Nhãn của checkpoint không khớp north/central/south: " + str(model.config.id2label))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.manual_seed(42)
    model.to(device).eval()
    print("Checkpoint:", checkpoint, "| Device:", device, flush=True)

    names = ("clean", "noisy", "vimd") if args.set == "all" else (args.set,)
    summaries = [s for s in (evaluate_set(n, model, args, device,
                                          bool(feature_extractor.do_normalize), checkpoint)
                             for n in names) if s]
    if args.check_only:
        print("\nKiểm tra xong. Chưa suy luận.")
        return
    combined = ROOT / "logs" / "comparison_{}_{}.csv".format(
        args.model_label, datetime.now().strftime("%Y%m%d_%H%M%S"))
    keys = []
    for summary in summaries:
        keys.extend(k for k in summary if k not in keys)
    with combined.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print("\nBảng tổng hợp:", combined)


if __name__ == "__main__":
    main()
