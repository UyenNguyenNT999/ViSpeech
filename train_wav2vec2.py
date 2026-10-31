"""Fine-tune wav2vec2 (mặc định nguyenvulebinh/wav2vec2-base-vi) để phân loại phương ngữ Bắc/Trung/Nam.

Chạy từ Code/ViSpeech:
    uv run --locked python train_wav2vec2.py --check-only        # chỉ kiểm tra dữ liệu, RIR, model
    uv run --locked python train_wav2vec2.py --max-steps 20      # chạy thử nhanh (smoke test)
    uv run --locked python train_wav2vec2.py                     # train đầy đủ

Giữ nguyên các thành phần của đối chứng M0 (train_sampler111.py) để chỉ khác biểu diễn/mô hình:
  - cùng dataset/train.txt và dataset/validation.txt, nhãn dialect = label % 3;
  - cùng LoadDataAndLabel: cắt/lặp kiểu 'wrap' ~5 giây, cùng 4 kiểu augmentation
    (RIR gốc 8 kHz được resample sang 16 kHz giống cách đã làm cho RIR_22050);
  - cùng BalancedSampler111 (1:1:1);
  - chọn checkpoint theo validation (mặc định accuracy giống M0, có thể đổi sang macro_f1).
Khác bắt buộc: wav2vec2 nhận waveform 16 kHz thay vì Log-Mel 22,05 kHz.
Siêu tham số mặc định theo ViMD (Dinh et al., 2024, Bảng 11): AdamW, lr 3e-5, 15 epoch,
warmup 0.1, weight decay 0. Không ghi đè checkpoint/cấu hình/kết quả cũ.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # đường dẫn trong dataset/*.txt là tương đối so với Code/ViSpeech
sys.path.insert(0, str(ROOT))

from dataset import LoadDataAndLabel  # noqa: E402
from sampler111 import BalancedSampler111  # noqa: E402

DIALECT_NAMES = ["north", "central", "south"]
SAMPLING_RATE = 16000
# Cấu hình 16 kHz sẵn có trong config.yaml: [n_fft, window_len, step_size] = [512, 400, 160].
# LoadDataAndLabel chỉ dùng window_len/step_size để tính độ dài: 500*160 + 240 = 80.240 mẫu (~5,015 s).
WINDOW_LEN, STEP_SIZE = 400, 160
AUDIO_LENGTH_SECONDS = 5


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="nguyenvulebinh/wav2vec2-base-vi",
                        help="Repo Hugging Face hoặc thư mục local.")
    parser.add_argument("--trainset", default="dataset/train.txt")
    parser.add_argument("--valset", default="dataset/validation.txt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Giới hạn số bước optimizer (dùng để chạy thử).")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch mỗi lần forward.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Batch hiệu dụng = batch-size x grad-accum (mặc định 32, bằng M0).")
    parser.add_argument("--select-metric", choices=("accuracy", "macro_f1"), default="accuracy")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--rir-path", default="augmentation/RIR_16000")
    parser.add_argument("--noise-path", default="augmentation/background_noise")
    parser.add_argument("--fp16", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="checkpoints/dialect")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for name in ("epochs", "lr", "batch_size", "grad_accum"):
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " phải lớn hơn 0")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps phải lớn hơn 0")
    return args


def check_inputs(args):
    """Kiểm tra file audio, nhãn và RIR trước khi tốn thời gian tải model."""
    summary = {}
    for split in (args.trainset, args.valset):
        counts = [0, 0, 0]
        with open(split, encoding="utf-8-sig") as handle:
            rows = [row for row in csv.reader(handle) if row]
        missing = [row[0] for row in rows if not Path(row[0]).is_file()]
        if missing:
            raise FileNotFoundError("{} file không tồn tại trong {}, ví dụ: {}".format(
                len(missing), split, missing[:3]))
        for row in rows:
            label = int(row[1])
            if label not in range(6):
                raise ValueError("Nhãn không hợp lệ: " + str(row))
            counts[label % 3] += 1
        if not all(counts):
            raise ValueError("Thiếu một phương ngữ trong " + split)
        summary[split] = dict(zip(DIALECT_NAMES, counts))
        print("{}: {} mẫu | {}".format(split, len(rows), summary[split]), flush=True)
    train_files = {row.split(",")[0] for row in open(args.trainset, encoding="utf-8-sig") if row.strip()}
    val_files = {row.split(",")[0] for row in open(args.valset, encoding="utf-8-sig") if row.strip()}
    if train_files & val_files:
        raise ValueError("{} file trùng giữa train và validation.".format(len(train_files & val_files)))
    if not args.no_augmentation:
        rir_files = sorted(Path(args.rir_path).glob("*"))
        noise_files = sorted(Path(args.noise_path).glob("*"))
        if not rir_files:
            raise FileNotFoundError(
                "Chưa có RIR 16 kHz trong {}. RIR gốc là 8 kHz; tạo bản 16 kHz bằng resample_rir.py:\n"
                "  uv run --locked python -c \"import os; from resample_rir import resample_rir; "
                "os.makedirs('augmentation/RIR_16000', exist_ok=True); "
                "resample_rir('augmentation/RIR/rir.wav', 'augmentation/RIR_16000/rir_16000.wav', 16000)\""
                .format(args.rir_path))
        if not noise_files:
            raise FileNotFoundError("Thiếu background noise trong " + args.noise_path)
        rir_sr = sf.info(str(rir_files[0])).samplerate
        if rir_sr != SAMPLING_RATE:
            raise ValueError("RIR {} có sample rate {} Hz, cần {} Hz (augment.py không resample RIR)."
                             .format(rir_files[0], rir_sr, SAMPLING_RATE))
        print("RIR: {} ({} Hz) | Background noise: {} file".format(
            rir_files[0], rir_sr, len(noise_files)), flush=True)
    return summary


def normalize(waveforms, enabled):
    """Giống Wav2Vec2FeatureExtractor(do_normalize=True): zero-mean, unit-variance từng đoạn."""
    if not enabled:
        return waveforms
    mean = waveforms.mean(dim=1, keepdim=True)
    var = waveforms.var(dim=1, keepdim=True, unbiased=False)
    return (waveforms - mean) / torch.sqrt(var + 1e-7)


def evaluate(model, loader, device, do_normalize, use_amp):
    model.eval()
    truth, predicted, losses = [], [], []
    with torch.no_grad():
        for waves, labels in loader:
            waves = normalize(waves.to(device), do_normalize)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_values=waves).logits
            losses.append(torch.nn.functional.cross_entropy(logits.float(), labels).item())
            truth.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    return {"val_loss": float(np.mean(losses)),
            "val_accuracy": float(accuracy_score(truth, predicted)),
            "val_macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2],
                                           average="macro", zero_division=0)),
            **{"val_recall_" + name: float(recalls[i]) for i, name in enumerate(DIALECT_NAMES)}}


def main():
    args = parse_args()
    set_seed(args.seed)
    split_counts = check_inputs(args)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    use_amp = device.type == "cuda" and args.fp16 != "off"
    if args.fp16 == "on" and device.type != "cuda":
        raise RuntimeError("--fp16 on cần CUDA.")
    if device.type != "cuda":
        print("CẢNH BÁO: đang chạy CPU, chỉ phù hợp để chạy thử.", flush=True)

    from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                              get_linear_schedule_with_warmup)
    import transformers

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    if feature_extractor.sampling_rate != SAMPLING_RATE:
        raise ValueError("Model cần {} Hz, script dùng {} Hz.".format(
            feature_extractor.sampling_rate, SAMPLING_RATE))
    model, loading_info = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3,
        id2label=dict(enumerate(DIALECT_NAMES)),
        label2id={name: i for i, name in enumerate(DIALECT_NAMES)},
        output_loading_info=True)
    model.freeze_feature_encoder()  # chuẩn khi fine-tune wav2vec2: giữ nguyên CNN trích đặc trưng
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model: {} | {:.1f}M tham số, {:.1f}M trainable".format(
        args.model_name, n_params / 1e6, n_trainable / 1e6), flush=True)
    print("Khởi tạo mới:", sorted(loading_info["missing_keys"]), flush=True)
    print("Bỏ qua (chỉ dùng khi pretrain):", sorted(loading_info["unexpected_keys"]), flush=True)
    if args.check_only:
        print("Kiểm tra xong. Chưa train.")
        return

    dataset_kwargs = dict(audio_length=AUDIO_LENGTH_SECONDS, sampling_rate=SAMPLING_RATE,
                          window_len=WINDOW_LEN, step_size=STEP_SIZE, task_type="dialect")
    train_dataset = LoadDataAndLabel(dataset=args.trainset,
                                     enable_augmentation=not args.no_augmentation,
                                     rir_path=args.rir_path, noise_path=args.noise_path,
                                     **dataset_kwargs)
    val_dataset = LoadDataAndLabel(dataset=args.valset, enable_augmentation=False, **dataset_kwargs)
    loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    micro_batches_per_epoch = len(BalancedSampler111(train_dataset, args.batch_size))
    steps_per_epoch = (micro_batches_per_epoch + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps),
                                                total_steps)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / args.output_root / "wav2vec2_base_vi_{}".format(timestamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path("dataset.py"), Path("sampler111.py"),
                   Path(args.trainset), Path(args.valset)):
        shutil.copy2(source, run_dir / source.name)
    run_info = {"status": "running", "args": vars(args), "device": str(device), "fp16": use_amp,
                "sampling_rate": SAMPLING_RATE,
                "sample_count": AUDIO_LENGTH_SECONDS * 100 * STEP_SIZE + WINDOW_LEN - STEP_SIZE,
                "split_counts": split_counts, "do_normalize": bool(feature_extractor.do_normalize),
                "n_params": n_params, "n_trainable": n_trainable,
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                "model_config": {k: getattr(model.config, k, None) for k in
                                 ("mask_time_prob", "layerdrop", "hidden_dropout",
                                  "classifier_proj_size", "use_weighted_layer_sum")},
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    info_path = run_dir / "run_info.json"
    info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Thư mục kết quả:", run_dir, flush=True)
    print("Mỗi epoch {} bước optimizer, tổng {} bước, fp16={}".format(
        steps_per_epoch, total_steps, use_amp), flush=True)

    columns = ["epoch", "step", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
               "val_macro_f1"] + ["val_recall_" + n for n in DIALECT_NAMES] + ["skipped_steps"]
    best_score, step, skipped = -1.0, 0, 0
    try:
        with (run_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            progress = tqdm(total=total_steps, desc="Fine-tune wav2vec2")
            for epoch in range(1, args.epochs + 1):
                loader_train = DataLoader(
                    train_dataset, batch_sampler=BalancedSampler111(train_dataset, args.batch_size),
                    num_workers=args.num_workers, pin_memory=device.type == "cuda")
                model.train()
                losses, correct, seen = [], 0, 0
                optimizer.zero_grad(set_to_none=True)
                for index, (waves, labels) in enumerate(loader_train, 1):
                    waves = normalize(waves.to(device, non_blocking=True), feature_extractor.do_normalize)
                    labels = labels.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        logits = model(input_values=waves).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                    scaler.scale(loss / args.grad_accum).backward()
                    losses.append(loss.item())
                    correct += int((logits.argmax(dim=1) == labels).sum())
                    seen += len(labels)
                    if index % args.grad_accum == 0 or index == micro_batches_per_epoch:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0)
                        if not torch.isfinite(grad_norm):
                            skipped += 1  # fp16 tràn số: GradScaler tự bỏ qua bước này
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        step += 1
                        progress.update(1)
                        if step >= total_steps:
                            break
                metrics = evaluate(model, loader_val, device, feature_extractor.do_normalize, use_amp)
                row = {"epoch": epoch, "step": step, "lr": scheduler.get_last_lr()[0],
                       "train_loss": float(np.mean(losses)), "train_accuracy": correct / max(seen, 1),
                       **metrics, "skipped_steps": skipped}
                writer.writerow(row)
                csv_file.flush()
                tqdm.write("EPOCH {} | STEP {} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | "
                           "Val Acc {:.4f} | Val Macro-F1 {:.4f}".format(
                               epoch, step, row["train_loss"], row["train_accuracy"],
                               row["val_loss"], row["val_accuracy"], row["val_macro_f1"]))
                score = metrics["val_" + args.select_metric]
                if score > best_score:  # mỗi checkpoint ~380 MB nên chỉ giữ best và last
                    best_score = score
                    model.save_pretrained(run_dir / "best")
                    feature_extractor.save_pretrained(run_dir / "best")
                    (run_dir / "best" / "best_info.json").write_text(
                        json.dumps({"select_metric": args.select_metric, **row}, indent=2),
                        encoding="utf-8")
                    tqdm.write("  -> lưu best ({} = {:.4f})".format(args.select_metric, score))
                if step >= total_steps:
                    break
            progress.close()
        model.save_pretrained(run_dir / "last")
        feature_extractor.save_pretrained(run_dir / "last")
        run_info.update(status="completed", best_score=best_score, finished_steps=step,
                        skipped_steps=skipped)
        print("Train xong. Checkpoint tốt nhất:", run_dir / "best")
    except Exception as exc:
        run_info.update(status="failed", error=repr(exc), finished_steps=step)
        raise
    finally:
        info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
