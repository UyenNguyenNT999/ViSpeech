"""Independent paired A/B runs, each with two checkpoint selectors and a fixed budget."""
import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset_paired_dual_validation import PairedDataset
from fixed_noise_validation import FixedNoiseValidation, choose_checkpoints
from prepare_fixed_noise_validation import ROOT, resolve
from evaluate_noise_robustness import file_hash
from model.torch_hub_models import Hub_Model
from sampler111 import BalancedSampler111


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def paired_losses(clean_log, noisy_log, labels, weight):
    """Same supervised and symmetric JS objectives as the original paired A/B."""
    classification = (F.nll_loss(clean_log, labels) + F.nll_loss(noisy_log, labels)) / 2
    log_mean = torch.logaddexp(clean_log, noisy_log) - np.log(2.0)
    js = 0.5 * ((clean_log.exp() * (clean_log - log_mean)).sum(1).mean()
                + (noisy_log.exp() * (noisy_log - log_mean)).sum(1).mean())
    return classification, js, classification + weight * js


def prepare(config):
    a, t, v = config["audio"], config["training"], config["validation"]
    for key in ("batch_size", "max_steps", "validation_every_steps"):
        if not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("clean_batch_size", "noisy_batch_size"):
        if not isinstance(v[key], int) or v[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= v["max_clean_accuracy_drop_pp"] <= 100:
        raise ValueError("Clean accuracy drop must be percentage points in [0,100]")
    if not np.isfinite([t["learning_rate"], t["weight_decay"]]).all() or t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    weight = t.get("consistency_weight", 0.0)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    checkpoint = resolve(config["initial_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    validation = FixedNoiseValidation(v["fixed_directory"])
    ev = validation.config["audio"]
    window = a["window_configs"][a["sampling_rate"]]
    if [a["length"], a["sampling_rate"], a["n_mels"], *window] != [
            ev["length_seconds"], ev["sampling_rate"], ev["n_mels"], ev["n_fft"], ev["window_length"], ev["hop_length"]]:
        raise ValueError("Training and validation audio preprocessing must agree")
    if resolve(config["datasets"]["valset"]) != resolve(validation.config["dataset"]["speech_list_file"]):
        raise ValueError("Validation speech list differs from fixed bundle")
    n = config["noise"]
    train = PairedDataset(dataset=resolve(config["datasets"]["trainset"]),
                          audio_length=a["length"], sampling_rate=a["sampling_rate"],
                          window_len=window[1], step_size=window[2], task_type="dialect",
                          noise_path=resolve(n["path"]), snr_db_min=n["snr_db_min"],
                          snr_db_max=n["snr_db_max"], noise_seed=t["seed"] + 1)
    train.file_list = [str(resolve(p)) for p in train.file_list]
    for path in train.file_list:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if set(train.file_list) & {s["audio_path"] for s in validation.samples}:
        raise ValueError("Train/validation audio paths overlap")
    BalancedSampler111(train, t["batch_size"])
    # Exact-file duplicate detection; this cannot prove distinct recording provenance.
    noise_hashes = {file_hash(p) for p in train.noise_files}
    if noise_hashes & {file_hash(Path(n["path"])) for n in validation.noises}:
        raise ValueError("An identical noise file is used in training and validation")
    return train, validation, checkpoint, window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_train_paired_dual_validation.yaml")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    train, validation, checkpoint, window = prepare(config)
    t, v, a, m = config["training"], config["validation"], config["audio"], config["model"]
    weight = t.get("consistency_weight", 0.0)
    print(f"Experiment: {config['name']} | consistency weight: {weight}", flush=True)
    print(f"Train originals: {len(train)} | Val originals: {len(validation.samples)} | "
          f"Noisy conditions: {len(validation.groups)}", flush=True)
    if args.check_only:
        clean, noisy, _, meta = train[0]
        if clean.shape != noisy.shape or not torch.isfinite(noisy).all():
            raise ValueError("Invalid training pair")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if tuple(state["classifier.0.weight"].shape) != (3, 1000):
            raise ValueError("Expected three-class checkpoint")
        print(f"CHECK PASSED | CUDA: {torch.cuda.is_available()} | no training performed.")
        print(f"Pair SNR target/measured: {meta['snr_db']:.6f}/{meta['measured_snr_db']:.6f}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full training")
    seed_all(t["seed"])
    device = torch.device("cuda")
    model = Hub_Model(n_classes=3, n_mels=a["n_mels"], dropout_rate=m["dropout"],
                      sampling_rate=a["sampling_rate"], n_fft=window[0], window=window[1], step=window[2],
                      model_type=m["type"], version=m["version"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    loader_rng = torch.Generator().manual_seed(t["seed"] + 2)
    output = resolve(config["output_root"]) / (config["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    sources = [config_path, Path(__file__), ROOT / "dataset_paired_dual_validation.py",
               ROOT / "fixed_noise_validation.py", ROOT / "prepare_fixed_noise_validation.py",
               ROOT / "evaluate_noise_robustness.py", ROOT / "dataset.py", ROOT / "sampler111.py",
               ROOT / "model/torch_hub_models.py", resolve(config["datasets"]["trainset"]),
               resolve(config["datasets"]["valset"])]
    for source in sources:
        shutil.copy2(source, output / source.name)
    shutil.copytree(validation.directory, output / "fixed_validation")
    info = {"status": "running", "config": config, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "torch_version": torch.__version__,
            "consistency_weight": weight,
            "consistency": "symmetric JS, temperature=1, both branches trainable; same as original A/B",
            "noise_train": [{"path": str(p), "sha256": file_hash(p)} for p in train.noise_files],
            "early_stopping": False, "selection": "best_clean: clean accuracy; best_robust: mean noisy macro-F1 with fixed clean floor; ties keep earlier",
            "noise_provenance": "Exact file duplicates checked; different source recordings are not proven by hashes alone."}
    step, epoch, pairs_seen = 0, 0, 0
    best_clean, best_robust, threshold = None, None, None
    best_clean_step, best_robust_step = 0, 0
    totals = np.zeros(6, dtype=np.float64)
    start_time = time.perf_counter()
    columns = ["step", "epoch", "train_loss", "train_original_accuracy", "train_noisy_accuracy",
               "val_original_loss", "val_original_accuracy", "val_original_macro_f1", "val_noisy_macro_f1",
               "val_noisy_f1_20db", "val_noisy_f1_10db", "val_noisy_f1_0db", "clean_accuracy_floor",
               "robust_eligible", "best_clean_step", "best_robust_step", "pairs_seen", "views_seen",
               "validation_seconds", "elapsed_seconds", "train_classification_loss", "train_consistency_js"]
    mix_columns = ["step", "epoch", "label", "speech_file", "noise_file", "noise_offset_samples",
                   "snr_db", "measured_snr_db", "shared_peak_scale"]

    def write_info():
        info.update(final_step=step, best_clean_step=best_clean_step, best_robust_step=best_robust_step,
                    clean_accuracy_floor=threshold, pairs_seen=pairs_seen, elapsed_seconds=time.perf_counter()-start_time)
        (output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def validate(writer, history):
        nonlocal best_clean, best_robust, threshold, best_clean_step, best_robust_step
        began = time.perf_counter()
        # Save/restore even if a future evaluation implementation draws random values.
        py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all()
        try:
            clean, noisy_f1, snrs, details = validation.evaluate(
                model, device, v["clean_batch_size"], v["noisy_batch_size"])
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(gpu_rng)
        if threshold is None:
            threshold = clean["accuracy"] - v["max_clean_accuracy_drop_pp"] / 100
            info["initial_clean_accuracy"] = clean["accuracy"]
        take_clean, take_robust, eligible = choose_checkpoints(clean, noisy_f1, threshold, best_clean, best_robust)
        for name, take in (("clean", take_clean), ("robust", take_robust)):
            if take:
                torch.save(model.state_dict(), output / f"best_{name}.pth")
                (output / f"best_{name}_info.json").write_text(json.dumps(
                    {"step": step, "epoch": epoch, "clean": clean, "noisy_macro_f1": noisy_f1,
                     "clean_accuracy_floor": threshold, "noisy_macro_f1_by_snr": snrs}, indent=2), encoding="utf-8")
        if take_clean:
            best_clean, best_clean_step = clean["accuracy"], step
        if take_robust:
            best_robust, best_robust_step = noisy_f1, step
        with (output / f"validation_conditions_step_{step:06d}.csv").open("w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=list(details[0]))
            dw.writeheader()
            dw.writerows(details)
        train_means = list(totals[:3] / totals[3]) if totals[3] else ["", "", ""]
        writer.writerow(dict(zip(columns, [step, epoch, *train_means, clean["loss"], clean["accuracy"],
                                           clean["macro_f1"], noisy_f1, snrs[20.0], snrs[10.0], snrs[0.0],
                                           threshold, eligible, best_clean_step, best_robust_step,
                                           pairs_seen, 2*pairs_seen, time.perf_counter()-began, time.perf_counter()-start_time,
                                           totals[4]/totals[3] if totals[3] else "",
                                           totals[5]/totals[3] if totals[3] else ""])))
        history.flush()
        if totals[3]:
            print(f"STEP {step} | Train Loss total: {train_means[0]:.4f} | "
                  f"classification: {totals[4]/totals[3]:.4f} | JS: {totals[5]/totals[3]:.4f} | "
                  f"Train Acc original/noisy: "
                  f"{train_means[1]:.4f}/{train_means[2]:.4f}", flush=True)
        print(f"STEP {step} | Val Loss: {clean['loss']:.4f} | Val Acc: {clean['accuracy']:.6f} | "
              f"Val F1: {clean['macro_f1']:.4f} | Noisy F1 mean: {noisy_f1:.4f}", flush=True)
        print(f"Noisy F1 20/10/0 dB: {snrs[20.0]:.4f}/{snrs[10.0]:.4f}/{snrs[0.0]:.4f} | "
              f"clean floor: {threshold:.6f} | eligible: {eligible} | "
              f"best clean/robust steps: {best_clean_step}/{best_robust_step}", flush=True)
        torch.save(model.state_dict(), output / "last.pth")
        totals[:] = 0
        write_info()
        model.train()

    print(f"Output: {output}\nFixed budget, no early stopping. Starting STEP 0 validation...", flush=True)
    write_info()
    try:
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as history, \
                (output / "augmentation_log.csv").open("w", newline="", encoding="utf-8") as augmentation_log_file:
            writer = csv.DictWriter(history, fieldnames=columns)
            augmentation_csv_writer = csv.DictWriter(augmentation_log_file, fieldnames=mix_columns)
            writer.writeheader()
            augmentation_csv_writer.writeheader()
            validate(writer, history)
            while step < t["max_steps"]:
                epoch += 1
                loader = DataLoader(train, batch_sampler=BalancedSampler111(train, t["batch_size"]),
                                    num_workers=0, pin_memory=True, generator=loader_rng)
                for original, noisy, labels, metadata in loader:
                    labels_device = labels.to(device)
                    n = len(labels)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(torch.cat([original, noisy]).to(device))
                    clean_log, noisy_log = outputs[:n], outputs[n:]
                    classification, js, loss = paired_losses(clean_log, noisy_log, labels_device, weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                    step += 1
                    pairs_seen += n
                    totals += [loss.item()*n, (clean_log.argmax(1)==labels_device).sum().item(),
                               (noisy_log.argmax(1)==labels_device).sum().item(), n,
                               classification.item()*n, js.item()*n]
                    for i in range(n):
                        row = {key: value[i].item() if torch.is_tensor(value) else value[i]
                               for key, value in metadata.items()}
                        augmentation_csv_writer.writerow(dict(step=step, epoch=epoch, label=int(labels[i]), **row))
                    # Release training graph before evaluating the much larger validation suite.
                    del outputs, clean_log, noisy_log, loss, classification, js
                    if step % 50 == 0:
                        print(f"STEP {step}/{t['max_steps']} | Train Loss mean: {totals[0]/totals[3]:.4f}", flush=True)
                    if step % t["validation_every_steps"] == 0 or step == t["max_steps"]:
                        optimizer.zero_grad(set_to_none=True)
                        augmentation_log_file.flush()
                        validate(writer, history)
                    if step >= t["max_steps"]:
                        break
        info.update(status="completed", stop_reason="max_steps")
    except BaseException as exc:
        info.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        write_info()
    print(f"Completed. Compare best_clean.pth and best_robust.pth in {output}", flush=True)


if __name__ == "__main__":
    main()
