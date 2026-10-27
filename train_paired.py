"""Controlled paired fine-tuning. Run --check-only before starting a full run."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import LoadDataAndLabel
from dataset_paired import PairedDataset
from model.torch_hub_models import Hub_Model
from sampler111 import BalancedSampler111


ROOT = Path(__file__).resolve().parent


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def consistency_js(log_original, log_noisy):
    """Symmetric Jensen-Shannon divergence at temperature 1; both views learn."""
    log_mean = torch.logaddexp(log_original, log_noisy) - np.log(2.0)
    return 0.5 * ((log_original.exp() * (log_original - log_mean)).sum(1).mean()
                  + (log_noisy.exp() * (log_noisy - log_mean)).sum(1).mean())


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            total_loss += F.nll_loss(output, labels, reduction="sum").item()
            correct += (output.argmax(1) == labels).sum().item()
            count += labels.numel()
    if count == 0:
        raise ValueError("Empty validation set")
    return total_loss / count, correct / count


def prepare(config):
    audio, training = config["audio"], config["training"]
    for key in ("batch_size", "max_steps", "validation_every_steps", "patience_checks"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "consistency_weight", "min_delta"):
        if not np.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"Invalid {key}")
    if training["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    checkpoint = resolve(config["initial_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    n_fft, window, step = audio["window_configs"][audio["sampling_rate"]]
    common = dict(audio_length=audio["length"], sampling_rate=audio["sampling_rate"],
                  window_len=window, step_size=step, task_type="dialect")
    noise = config["noise"]
    train = PairedDataset(dataset=resolve(config["datasets"]["trainset"]),
                          noise_path=resolve(noise["path"]), snr_db_min=noise["snr_db_min"],
                          snr_db_max=noise["snr_db_max"], noise_seed=training["seed"] + 1,
                          **common)
    val = LoadDataAndLabel(dataset=resolve(config["datasets"]["valset"]),
                           enable_augmentation=False, **common)
    for dataset in (train, val):
        dataset.file_list = [str(resolve(path)) for path in dataset.file_list]
        if not dataset.file_list:
            raise ValueError("Empty dataset list")
        missing = [path for path in dataset.file_list if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing speech file: {missing[0]}")
    if set(train.file_list) & set(val.file_list):
        raise ValueError("Train and validation contain overlapping audio paths")
    for dataset in (train, val):
        if set(dataset.get_label(label) for label in dataset.label_list) != {0, 1, 2}:
            raise ValueError("Each split must contain all three dialect labels")
    return train, val, checkpoint, (n_fft, window, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="Check files, split, one pair and checkpoint; no model training")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    train, val, checkpoint, window = prepare(config)
    t = config["training"]
    seed_all(t["seed"])
    sampler = BalancedSampler111(train, t["batch_size"])
    print(f"Train: {len(train)} | Val originals: {len(val)} | Noise files: {len(train.noise_files)}")
    print(f"Source checkpoint: {checkpoint}")
    print(f"Pairs/batch: {t['batch_size']} | Steps/epoch: {len(sampler)} | lambda: {t['consistency_weight']}")
    if args.check_only:
        original, noisy, label, metadata = train[0]
        assert original.shape == noisy.shape and torch.isfinite(noisy).all()
        assert abs(metadata["snr_db"] - metadata["measured_snr_db"]) < 1e-3
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "classifier.0.weight" not in state:
            raise ValueError("Expected a Hub_Model state_dict checkpoint")
        if tuple(state["classifier.0.weight"].shape) != (3, 1000):
            raise ValueError("Expected a three-class dialect checkpoint")
        print(json.dumps(metadata, indent=2))
        print("CHECK PASSED. No optimizer step or training output was created.")
        print(f"CUDA available: {torch.cuda.is_available()}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; --check-only also works on CPU")
    device = torch.device("cuda")
    a, m = config["audio"], config["model"]
    model = Hub_Model(n_classes=3, n_mels=a["n_mels"], dropout_rate=m["dropout"],
                      sampling_rate=a["sampling_rate"], n_fft=window[0], window=window[1],
                      step=window[2], model_type=m["type"], version=m["version"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    # New optimizer for a new fine-tuning experiment, not a resumed old optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    # Dedicated DataLoader generator prevents validation from changing dropout's RNG.
    loader_rng = torch.Generator().manual_seed(t["seed"] + 2)
    val_loader = DataLoader(val, batch_size=t["batch_size"], shuffle=False,
                            num_workers=0, generator=loader_rng)
    output = resolve(config["output_root"]) / (config["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    for path in [config_path, Path(__file__), ROOT / "dataset_paired.py", ROOT / "dataset.py",
                 ROOT / "sampler111.py", ROOT / "model/torch_hub_models.py",
                 resolve(config["datasets"]["trainset"]), resolve(config["datasets"]["valset"])]:
        shutil.copy2(path, output / path.name)
    info = {"config": config, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "train_list_sha256": file_hash(resolve(config["datasets"]["trainset"])),
            "val_list_sha256": file_hash(resolve(config["datasets"]["valset"])),
            "noise_files": [{"path": str(p), "sha256": file_hash(p)} for p in train.noise_files],
            "torch_version": torch.__version__, "status": "running",
            "selection": "original validation accuracy; ties keep earlier checkpoint",
            "snr_definition": "whole-clip RMS including silence; shared peak scaling",
            "optimizer": "fresh Adam", "consistency": "symmetric JS, temperature=1, both branches trainable"}
    (output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    step, epoch, best_step, bad_checks = 0, 0, 0, 0
    val_loss, best_acc = evaluate(model, val_loader, device)
    stopping_reference = best_acc
    shutil.copy2(checkpoint, output / "best_1.pth")
    def save_best_info():
        (output / "best_1_info.json").write_text(json.dumps(
            {"accuracy": best_acc, "step": best_step, "epoch": epoch,
             "source_is_initial_checkpoint": best_step == 0}, indent=2), encoding="utf-8")
    save_best_info()
    columns = ["step", "epoch", "train_classification_loss", "train_consistency_js", "train_total_loss",
               "train_original_accuracy", "train_noisy_accuracy", "val_original_loss", "val_original_accuracy",
               "best_step", "bad_checks", "pairs_seen", "views_seen"]
    mix_columns = ["step", "epoch", "label", "speech_file", "noise_file", "noise_offset_samples",
                   "snr_db", "measured_snr_db", "shared_peak_scale"]
    totals = np.zeros(6, dtype=np.float64)
    pairs_seen, stopped = 0, False
    try:
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as history, \
                (output / "augmentation_log.csv").open("w", newline="", encoding="utf-8") as mixtures:
            writer = csv.DictWriter(history, fieldnames=columns)
            mix_writer = csv.DictWriter(mixtures, fieldnames=mix_columns)
            writer.writeheader()
            mix_writer.writeheader()
            writer.writerow(dict(step=0, epoch=0, val_original_loss=val_loss, val_original_accuracy=best_acc,
                                 best_step=0, bad_checks=0, pairs_seen=0, views_seen=0))
            history.flush()
            print(f"STEP 0 | Val Loss (original): {val_loss:.4f} | Val Acc: {best_acc:.6f}", flush=True)
            while step < t["max_steps"] and not stopped:
                epoch += 1
                loader = DataLoader(train, batch_sampler=BalancedSampler111(train, t["batch_size"]),
                                    num_workers=0, pin_memory=True, generator=loader_rng)
                model.train()
                for original, noisy, labels, metadata in loader:
                    labels_device = labels.to(device)
                    n = len(labels)
                    outputs = model(torch.cat([original, noisy]).to(device))
                    log_original, log_noisy = outputs[:n], outputs[n:]
                    classification = 0.5 * (F.nll_loss(log_original, labels_device)
                                             + F.nll_loss(log_noisy, labels_device))
                    js = consistency_js(log_original, log_noisy)
                    loss = classification + t["consistency_weight"] * js
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    pairs_seen += n
                    totals += [classification.item() * n, js.item() * n, loss.item() * n,
                               (log_original.argmax(1) == labels_device).sum().item(),
                               (log_noisy.argmax(1) == labels_device).sum().item(), n]
                    for i in range(n):
                        row = {key: value[i].item() if torch.is_tensor(value) else value[i]
                               for key, value in metadata.items()}
                        mix_writer.writerow(dict(step=step, epoch=epoch, label=int(labels[i]), **row))
                    if step % 50 == 0:
                        print(f"STEP {step}/{t['max_steps']} | loss {loss.item():.4f}", flush=True)
                    if step % t["validation_every_steps"] == 0 or step == t["max_steps"]:
                        val_loss, val_acc = evaluate(model, val_loader, device)
                        if val_acc > best_acc:
                            best_acc, best_step = val_acc, step
                            torch.save(model.state_dict(), output / "best_1.pth")
                            save_best_info()
                        if val_acc > stopping_reference + t["min_delta"]:
                            stopping_reference, bad_checks = val_acc, 0
                        else:
                            bad_checks += 1
                        averages = totals[:5] / totals[5]
                        writer.writerow(dict(zip(columns, [step, epoch, *averages, val_loss, val_acc,
                                                            best_step, bad_checks, pairs_seen, pairs_seen * 2])))
                        history.flush()
                        mixtures.flush()
                        torch.save(model.state_dict(), output / "last.pth")
                        print(f"STEP {step} | Train Loss (classification, original+noise): {averages[0]:.4f} | "
                              f"Train Loss (total): {averages[2]:.4f} | "
                              f"Val Loss (original): {val_loss:.4f} | "
                              f"Train Acc (original/noise): {averages[3]:.4f}/{averages[4]:.4f} | "
                              f"Val Acc: {val_acc:.6f} | best step: {best_step} | "
                              f"no improvement: {bad_checks}/{t['patience_checks']}", flush=True)
                        totals[:] = 0
                        model.train()
                        stopped = bad_checks >= t["patience_checks"]
                    if stopped or step >= t["max_steps"]:
                        break
        info["status"] = "completed"
        info["stop_reason"] = "early_stopping" if stopped else "max_steps"
    except BaseException as exc:
        info["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        info["error"] = str(exc)
        raise
    finally:
        info.update(final_step=step, best_step=best_step, best_accuracy=best_acc,
                    pairs_seen=pairs_seen, views_seen=2 * pairs_seen)
        (output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output / 'best_1.pth'}", flush=True)
    if best_step == 0:
        print("No fine-tuned checkpoint beat the initial original-validation accuracy; best remains step 0.")


if __name__ == "__main__":
    main()
