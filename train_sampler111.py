"""Bản train 1:1:1 đơn giản, dùng lại mô hình và augmentation của train.py.

Chạy từ Code/ViSpeech: python train_sampler111.py
"""
import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoadDataAndLabel
from model.torch_hub_models import Hub_Model
from sampler111 import BalancedSampler111


def set_seed(seed):
    # Giống train.py cũ.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    true_labels, predicted_labels, losses = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = torch.nn.functional.nll_loss(outputs, labels.to(device))
            losses.append(loss.item())
            true_labels.extend(labels.tolist())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)), float(accuracy_score(true_labels, predicted_labels))


def run_training():
    config_path = Path("config_train_sampler111.yaml")
    with config_path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    audio = config["audio"]
    model_config = config["model"]
    training = config["training"]
    if config["task"]["type"] != "dialect" or model_config["type"] != "resnet34":
        raise ValueError("File này dành cho đối chứng ResNet34 phân loại phương ngữ.")
    if training["max_steps"] <= 0 or training["validation_every_steps"] <= 0:
        raise ValueError("Số bước train và chu kỳ validation phải lớn hơn 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("Chưa có CUDA. Hãy dùng môi trường GPU của lần train cũ.")
    device = torch.device("cuda")
    set_seed(training["seed"])

    # Giữ nguyên mô hình, xử lý âm thanh và optimizer của baseline.
    n_fft, window_len, step_size = audio["window_configs"][audio["sampling_rate"]]
    model = Hub_Model(
        n_classes=3, n_mels=audio["n_mels"], dropout_rate=model_config["dropout"],
        sampling_rate=audio["sampling_rate"], n_fft=n_fft,
        window=window_len, step=step_size, model_type=model_config["type"],
        version=model_config["versions"][model_config["type"]]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    train_dataset = LoadDataAndLabel(
        dataset=config["datasets"]["trainset"], audio_length=audio["length"],
        sampling_rate=audio["sampling_rate"], window_len=window_len,
        step_size=step_size, task_type="dialect",
        enable_augmentation=training["enable_augmentation"],
        rir_path=config["augmentation_resources"]["rir_path"],
        noise_path=config["augmentation_resources"]["noise_path"]
    )
    val_dataset = LoadDataAndLabel(
        dataset=config["datasets"]["valset"], audio_length=audio["length"],
        sampling_rate=audio["sampling_rate"], window_len=window_len,
        step_size=step_size, task_type="dialect", enable_augmentation=False
    )
    loader_val = DataLoader(val_dataset, batch_size=training["batch_size"],
                            num_workers=0, shuffle=False)

    # Thư mục riêng; không ghi đè checkpoint và cấu hình cũ.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checkpoint_dir = Path("checkpoints/dialect") / f"resnet34_sampler111_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for source in (config_path, Path(__file__), Path("sampler111.py"),
                   Path(config["datasets"]["trainset"]), Path(config["datasets"]["valset"])):
        shutil.copy2(source, checkpoint_dir / source.name)
    print(f"Checkpoint folder: {checkpoint_dir}", flush=True)

    step, epoch = 0, 0
    best_checkpoints = []
    losses, correct, samples = [], 0, 0
    class_draws = [0, 0, 0]
    columns = ["step", "epoch", "train_loss", "train_accuracy", "val_loss",
               "val_accuracy", "draws_north", "draws_central", "draws_south"]
    with (checkpoint_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as csv_file, \
            (checkpoint_dir / "training_log.txt").open("w", encoding="utf-8") as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        with tqdm(total=training["max_steps"], desc="Training sampler 1:1:1") as progress:
            while step < training["max_steps"]:
                epoch += 1
                loader_train = DataLoader(
                    train_dataset,
                    batch_sampler=BalancedSampler111(train_dataset, training["batch_size"]),
                    num_workers=0, pin_memory=True
                )
                model.train()
                for inputs, labels in loader_train:
                    outputs = model(inputs.to(device))
                    loss = torch.nn.functional.nll_loss(outputs, labels.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    losses.append(loss.item())
                    correct += int((outputs.argmax(dim=1).cpu() == labels).sum())
                    samples += len(labels)
                    for label in range(3):
                        class_draws[label] += int((labels == label).sum())
                    progress.update(1)

                    # Cùng lịch validation với lần cũ: mỗi 306 bước.
                    if step % training["validation_every_steps"] == 0 or step == training["max_steps"]:
                        val_loss, val_acc = evaluate(model, loader_val, device)
                        row = dict(zip(columns, [step, epoch, float(np.mean(losses)), correct / samples,
                                                 val_loss, val_acc, *class_draws]))
                        writer.writerow(row)
                        csv_file.flush()
                        message = (f"STEP {step} | EPOCH {epoch} | Train Loss: {row['train_loss']:.4f} | "
                                   f"Train Acc: {row['train_accuracy']:.4f} | "
                                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                        tqdm.write(message)
                        log_file.write(message + "\n")
                        log_file.flush()
                        torch.save(model.state_dict(), checkpoint_dir / "last.pth")

                        # Chọn top 3 theo validation accuracy giống baseline.
                        if len(best_checkpoints) < 3 or val_acc > best_checkpoints[-1]["accuracy"]:
                            state = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
                            best_checkpoints.append({"accuracy": val_acc, "step": step,
                                                     "epoch": epoch, "state": state})
                            best_checkpoints = sorted(best_checkpoints, key=lambda item: item["accuracy"],
                                                      reverse=True)[:3]
                            for rank, checkpoint in enumerate(best_checkpoints, 1):
                                torch.save(checkpoint["state"], checkpoint_dir / f"best_{rank}.pth")
                                info = {key: value for key, value in checkpoint.items() if key != "state"}
                                (checkpoint_dir / f"best_{rank}_info.json").write_text(
                                    json.dumps(info, indent=2), encoding="utf-8")
                        losses, correct, samples = [], 0, 0
                        class_draws = [0, 0, 0]
                        model.train()
                    if step >= training["max_steps"]:
                        break
    print(f"Train xong. Model tốt nhất: {checkpoint_dir / 'best_1.pth'}")


if __name__ == "__main__":
    run_training()
