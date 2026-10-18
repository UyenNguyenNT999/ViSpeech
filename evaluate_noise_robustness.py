"""Đánh giá ResNet34 đã train trên bản gốc và bản thêm nhiễu.

Không huấn luyện. Chạy: python evaluate_noise_robustness.py
Hoặc: python evaluate_noise_robustness.py --config config_noise_eval.yaml
Mỗi lần chạy tạo một thư mục kết quả riêng, không ghi đè lần trước.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import torch
import yaml
import soundfile as sf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm import tqdm

from model.torch_hub_models import Hub_Model


SCRIPT_DIRECTORY = Path(os.path.abspath(__file__)).parent
DIALECT_NAMES = ["north", "central", "south"]


def full_path(base_directory, relative_path):
    """Đổi đường dẫn tương đối thành đường dẫn đầy đủ."""
    return Path(os.path.abspath(str(base_directory / relative_path)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs(config, root):
    """Đọc dữ liệu và kiểm tra đường dẫn trước khi nạp mô hình."""
    speech_list_path = full_path(root, config["dataset"]["speech_list_file"])
    speech_paths_directory = full_path(root, config["dataset"]["speech_paths_directory"])
    samples = []
    with speech_list_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            label = int(row[1])
            if label not in range(6):
                raise ValueError(f"Nhãn không hợp lệ: {row}")
            path = full_path(speech_paths_directory, row[0])
            if not path.is_file():
                raise FileNotFoundError(path)
            samples.append({"audio_id": len(samples), "audio_path": str(path),
                            "true_label": label % 3})
    maximum_samples = config["evaluation"]["max_samples"]
    if maximum_samples is not None:
        if not isinstance(maximum_samples, int) or maximum_samples <= 0:
            raise ValueError("max_samples phải là số nguyên dương hoặc null.")
        samples = samples[:maximum_samples]

    noise_metadata_path = full_path(root, config["noise"]["noise_metadata_file"])
    noises = []
    used_noise_ids = set()
    with noise_metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"noise_id", "path"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV nhiễu cần có noise_id,path.")
        for row in reader:
            noise_id = row["noise_id"].strip()
            if not noise_id or noise_id in used_noise_ids or not row["path"].strip():
                raise ValueError(f"Mã nhiễu trống/trùng hoặc đường dẫn trống: {row}")
            used_noise_ids.add(noise_id)
            path = full_path(noise_metadata_path.parent, row["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            noises.append({"noise_id": noise_id, "path": str(path)})
    if not samples or not noises:
        raise ValueError("Danh sách tiếng nói và nhiễu đều phải có dữ liệu.")
    return samples, noises, speech_list_path, noise_metadata_path


def load_audio(path, sample_rate):
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Âm thanh rỗng hoặc không hữu hạn: {path}")
    return audio


def prepare_speech(path, sample_rate, sample_count):
    # Giữ đúng công thức cắt/lặp của dataset.py trong baseline.
    audio = load_audio(path, sample_rate)
    if len(audio) <= sample_count:
        padding_samples = sample_count - len(audio) + 1
        audio = np.pad(audio, (0, padding_samples), mode="wrap")
    return audio[:sample_count]


def mix_noise(speech, noise, snr_db):
    """SNR tính bằng công suất trung bình toàn đoạn, kể cả khoảng lặng.

    Không clip hay chuẩn hóa lại sau khi trộn. Giá trị vượt [-1,1] vẫn
    được giữ dạng float khi đưa trực tiếp vào mô hình, không ghi ra WAV.
    """
    speech_power = np.mean(speech.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    if speech_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("Tiếng nói hoặc đoạn nhiễu có công suất gần bằng 0.")
    noise_volume_scale = math.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    noisy_speech = (speech + noise_volume_scale * noise).astype(np.float32)
    if not np.isfinite(noisy_speech).all():
        raise ValueError("Âm thanh sau trộn chứa giá trị không hữu hạn.")
    return noisy_speech, noise_volume_scale


def predict(model, waveforms, device, batch_size):
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = np.stack(waveforms[start:start + batch_size])
            outputs = model(torch.from_numpy(batch).to(device))
            probabilities.extend(outputs.exp().cpu().numpy())
    return probabilities


def measure_snr(speech, added_noise):
    """Đo công suất toàn đoạn; trả None nếu không xác định được SNR."""
    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(added_noise.astype(np.float64) ** 2))
    if speech_power <= 1e-12 or noise_power <= 1e-12:
        return None
    return 10 * math.log10(speech_power / noise_power)


def select_audio_examples(samples, count):
    """Lấy luân phiên các miền, không chỉ lấy ba tệp đầu cùng một người nói."""
    selected_ids = set()
    groups = [[s for s in samples if s["true_label"] == label] for label in range(3)]
    while len(selected_ids) < min(count, len(samples)):
        for group in groups:
            if group and len(selected_ids) < count:
                selected_ids.add(group.pop(0)["audio_id"])
    return selected_ids


def save_listening_group(folder, sample, noise, speech, mixtures, snrs,
                         start, sample_rate):
    """Một hệ số giảm âm lượng chung cho cả bản gốc và mọi mức SNR.

    Chỉ bản nghe được giảm âm lượng. Đầu vào mô hình không thay đổi.
    WAV float giữ số thực; không tự clip tín hiệu.
    """
    folder.mkdir(exist_ok=True)
    group_peak = max(float(np.max(np.abs(wave))) for wave in [speech] + mixtures)
    listening_gain = min(1.0, 0.95 / max(group_peak, 1e-12))
    group_name = f'audio_{sample["audio_id"]:05d}_{hashlib.sha256(noise["noise_id"].encode()).hexdigest()[:10]}'
    records = []
    versions = [("original", None, speech)]
    versions.extend(("noisy", snr, wave) for snr, wave in zip(snrs, mixtures))
    for index, (condition, snr, wave) in enumerate(versions):
        filename = f"{group_name}_{index:02d}_{condition}.wav"
        sf.write(str(folder / filename), wave * listening_gain, sample_rate, subtype="FLOAT")
        records.append({"audio_id": sample["audio_id"], "true_label": sample["true_label"],
                        "speech_path": sample["audio_path"], "noise_id": noise["noise_id"],
                        "noise_path": noise["path"], "noise_start_sample": start,
                        "condition": condition, "snr_db": "" if snr is None else snr,
                        "repeat_id": 0, "listening_gain": listening_gain,
                        "path": f"audio_examples/{filename}"})
    return records


def prediction_row(sample, condition, noise_id, snr_db, repeat, probabilities):
    predicted = int(np.argmax(probabilities))
    return {**sample, "condition": condition, "noise_id": noise_id,
            "snr_db": snr_db, "repeat_id": repeat,
            "predicted_label": predicted, "correct": int(predicted == sample["true_label"]),
            **{f"prob_{name}": float(probabilities[i]) for i, name in enumerate(DIALECT_NAMES)}}


def summarize(rows, original_predictions, figures_directory):
    groups = {}
    for row in rows:
        key = (row["condition"], row["noise_id"], row["snr_db"], row["repeat_id"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (condition, noise_id, snr, repeat), group in groups.items():
        truth = np.array([r["true_label"] for r in group])
        predicted = np.array([r["predicted_label"] for r in group])
        original = np.array([original_predictions[r["audio_id"]] for r in group])
        original_correct = original == truth
        correct_to_wrong = int(np.sum(original_correct & (predicted != truth)))
        recalls = recall_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0)
        summaries.append({"condition": condition, "noise_id": noise_id,
                          "snr_db": snr, "repeat_id": repeat, "n_samples": len(group),
                          "accuracy": accuracy_score(truth, predicted),
                          "macro_f1": f1_score(truth, predicted, labels=[0, 1, 2],
                                               average="macro", zero_division=0),
                          **{f"recall_{name}": float(recalls[i]) for i, name in enumerate(DIALECT_NAMES)},
                          "prediction_change_rate": float(np.mean(predicted != original)),
                          "original_correct_count": int(original_correct.sum()),
                          "correct_to_wrong_count": correct_to_wrong,
                          "correct_to_wrong_rate": (correct_to_wrong / int(original_correct.sum())
                                                    if original_correct.any() else ""),
                          "wrong_to_correct_count": int(np.sum(~original_correct & (predicted == truth)))})
        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
        # Tên hình dùng số thứ tự, không dùng noise_id như một đường dẫn.
        draw_matrix(matrix, figures_directory / f"confusion_{len(summaries):03d}.svg",
                    f"{condition} {noise_id} SNR={snr} repeat={repeat}")
    return summaries


def draw_matrix(matrix, path, title):
    # SVG dùng thư viện chuẩn, không cần cài matplotlib.
    svg_elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="360">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="25" font-size="14">{escape(title)}</text>',
             '<text x="160" y="50">Columns: predicted; rows: true</text>']
    for i, name in enumerate(DIALECT_NAMES):
        svg_elements.append(f'<text x="{180 + i * 110}" y="80">{name}</text>')
        svg_elements.append(f'<text x="15" y="{130 + i * 85}">{name}</text>')
        for j in range(3):
            svg_elements.append(f'<rect x="{140 + j * 110}" y="{95 + i * 85}" '
                         'width="110" height="85" fill="#e0eef8" stroke="white"/>')
            svg_elements.append(f'<text x="{185 + j * 110}" y="{140 + i * 85}">{matrix[i,j]}</text>')
    path.write_text("\n".join(svg_elements) + "</svg>", encoding="utf-8")


def draw_f1(summaries, noises, path):
    svg_elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="60" y="25">Macro-F1 vs added noise SNR (higher SNR = less noise)</text>']
    snr_levels = sorted({float(row["snr_db"]) for row in summaries if row["condition"] == "noisy"})
    low, high = min(snr_levels), max(snr_levels)
    def x(value):
        return 70 + 500 * (value - low) / (high - low or 1)
    for tick in range(6):
        y = 470 - tick * 80
        svg_elements.append(f'<path d="M70 {y} H570" stroke="#ddd"/>')
        svg_elements.append(f'<text x="20" y="{y}">{tick / 5:.1f}</text>')
    for snr in snr_levels:
        svg_elements.append(f'<text x="{x(snr)}" y="495">{snr:g} dB</text>')
    baseline = summaries[0]["macro_f1"]
    svg_elements.append(f'<path d="M70 {470 - baseline * 400} H570" stroke="black" stroke-dasharray="6 4"/>')
    svg_elements.append('<text x="610" y="55">Dashed: original audio</text>')
    for index, noise in enumerate(noises):
        color = f"hsl({index * 137 % 360},65%,40%)"
        points = []
        for snr in snr_levels:
            values = [r["macro_f1"] for r in summaries
                      if r["noise_id"] == noise["noise_id"] and r["snr_db"] == snr]
            points.append(f"{x(snr)},{470 - float(np.mean(values)) * 400}")
        svg_elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg_elements.append(f'<text x="610" y="{85 + index * 24}" fill="{color}">{escape(noise["noise_id"])}</text>')
    svg_elements.append('<text x="70" y="540">Repeats are averaged per noise file; scores range from 0 to 1.</text>')
    path.write_text("\n".join(svg_elements) + "</svg>", encoding="utf-8")


def run(config, root, results_directory):
    samples, noises, speech_list_path, noise_metadata_path = read_inputs(config, root)
    evaluation_settings = config["evaluation"]
    snr_levels = [float(value) for value in config["noise"]["snr_db"]]
    repeats = config["noise"]["repeats"]
    example_count = config["output"].get("audio_example_count", 3)
    if not isinstance(example_count, int) or example_count < 0:
        raise ValueError("audio_example_count phải là số nguyên không âm.")
    example_ids = select_audio_examples(samples, example_count)
    example_rows = []
    signal_checks = []
    if not snr_levels or not all(math.isfinite(s) and -100 <= s <= 100 for s in snr_levels):
        raise ValueError("SNR phải hữu hạn, nằm trong [-100,100] dB.")
    if len(snr_levels) != len(set(snr_levels)) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SNR không được trùng; repeats phải là số nguyên dương.")
    if not isinstance(evaluation_settings["batch_size"], int) or evaluation_settings["batch_size"] <= 0:
        raise ValueError("batch_size phải là số nguyên dương.")
    seed = evaluation_settings["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested_device = evaluation_settings["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if requested_device == "auto" else requested_device)
    checkpoint = full_path(root, config["model"]["checkpoint"])
    audio = config["audio"]
    sample_rate = audio["sampling_rate"]
    sample_count = int(audio["length_seconds"] * 100) * audio["hop_length"]
    sample_count += audio["window_length"] - audio["hop_length"]
    run_information = {"status": "running", "device": str(device), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "librosa": librosa.__version__,
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "dataset_sha256": file_hash(speech_list_path), "noise_csv_sha256": file_hash(noise_metadata_path),
            "sample_count": sample_count, "labels": dict(enumerate(DIALECT_NAMES)),
            "snr_definition": "whole-segment mean-square power; no clipping or rescaling"}
    (results_directory / "run_info.json").write_text(json.dumps(run_information, indent=2), encoding="utf-8")
    write_csv(results_directory / "noise_metadata.csv", noises)
    # Hub_Model giữ nguyên mã baseline; torch.hub có thể tải tài nguyên nếu chưa có cache.
    model = Hub_Model(n_classes=3, n_mels=audio["n_mels"],
                      dropout_rate=config["model"]["dropout"], sampling_rate=sample_rate,
                      n_fft=audio["n_fft"], window=audio["window_length"],
                      step=audio["hop_length"], model_type="resnet34",
                      version=config["model"]["version"])
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.to(device).eval()
    speech_waveforms = [prepare_speech(sample["audio_path"], sample_rate, sample_count)
                       for sample in tqdm(samples, desc="Read validation")]
    original_probabilities = predict(model, speech_waveforms, device, evaluation_settings["batch_size"])
    rows = [prediction_row(sample, "original", "", "", "", probabilities)
            for sample, probabilities in zip(samples, original_probabilities)]
    original_predictions = {row["audio_id"]: row["predicted_label"] for row in rows}
    mixture_plan_path = results_directory / "mixture_plan.csv"
    with mixture_plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_id", "noise_id", "noise_start_sample",
                                                    "noise_samples", "snr_db", "repeat_id",
                                                    "noise_scale", "noise_sha256"])
        writer.writeheader()
        for noise in tqdm(noises, desc="Evaluate noise files"):
            noise_audio = load_audio(noise["path"], sample_rate)
            noise_file_sha256 = file_hash(Path(noise["path"]))
            # Nhiễu có RNG riêng: thêm tệp khác không làm đổi kế hoạch của tệp cũ.
            noise_random_seed = int(hashlib.sha256(f'{seed}:{noise["noise_id"]}'.encode()).hexdigest()[:16], 16)
            noise_random_generator = np.random.default_rng(noise_random_seed)
            for repeat in range(repeats):
                for sample, speech in zip(samples, speech_waveforms):
                    start = int(noise_random_generator.integers(0, max(1, len(noise_audio) - sample_count + 1)))
                    noise_segment = noise_audio[start:start + sample_count]
                    if len(noise_segment) < sample_count:
                        noise_segment = np.pad(noise_segment, (0, sample_count - len(noise_segment)), mode="wrap")
                    noisy_waveforms = []
                    for snr in snr_levels:
                        noisy_speech, noise_volume_scale = mix_noise(speech, noise_segment, snr)
                        noisy_waveforms.append(noisy_speech)
                        measured_snr = measure_snr(speech, noisy_speech.astype(np.float64) - speech)
                        snr_error = None if measured_snr is None else abs(measured_snr - snr)
                        signal_checks.append({"audio_id": sample["audio_id"], "noise_id": noise["noise_id"],
                                              "snr_db": snr, "repeat_id": repeat,
                                              "measured_snr_db": measured_snr,
                                              "snr_error_db": snr_error,
                                              "snr_matches": snr_error is not None and snr_error < 0.05,
                                              "finite": bool(np.isfinite(noisy_speech).all()),
                                              "peak_before_listening_gain": float(np.max(np.abs(noisy_speech))),
                                              "fraction_above_one": float(np.mean(np.abs(noisy_speech) > 1)),
                                              "noise_repeated": len(noise_audio) < sample_count,
                                              "available_noise_start_positions": max(1, len(noise_audio) - sample_count + 1)})
                        writer.writerow({"audio_id": sample["audio_id"], "noise_id": noise["noise_id"],
                                         "noise_start_sample": start, "noise_samples": sample_count,
                                         "snr_db": snr, "repeat_id": repeat, "noise_scale": noise_volume_scale,
                                         "noise_sha256": noise_file_sha256})
                    if (config["output"].get("save_audio_examples", False)
                            and repeat == 0 and sample["audio_id"] in example_ids):
                        example_rows.extend(save_listening_group(
                            results_directory / "audio_examples", sample, noise, speech,
                            noisy_waveforms, snr_levels, start, sample_rate))
                    probabilities = predict(model, noisy_waveforms, device, evaluation_settings["batch_size"])
                    rows.extend(prediction_row(sample, "noisy", noise["noise_id"], snr, repeat, class_probabilities)
                                for snr, class_probabilities in zip(snr_levels, probabilities))
    write_csv(results_directory / "predictions.csv", rows)
    write_csv(results_directory / "signal_checks.csv", signal_checks)
    if example_rows:
        write_csv(results_directory / "audio_examples.csv", example_rows)
    summaries = summarize(rows, original_predictions, results_directory / "figures")
    write_csv(results_directory / "summary.csv", summaries)
    draw_f1(summaries, noises, results_directory / "figures" / "macro_f1_by_snr.svg")
    run_information["status"] = "completed"
    (results_directory / "run_info.json").write_text(json.dumps(run_information, indent=2), encoding="utf-8")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--config", type=Path, default=SCRIPT_DIRECTORY / "config_noise_eval.yaml")
    command_line_options = argument_parser.parse_args()
    config_path = Path(os.path.abspath(str(command_line_options.config)))
    with config_path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    results_directory = full_path(root, config["output"]["root"]) / run_id
    (results_directory / "figures").mkdir(parents=True, exist_ok=False)
    with (results_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"Results: {results_directory}", flush=True)
    try:
        run(config, root, results_directory)
    except Exception as error:
        (results_directory / "FAILED.txt").write_text(repr(error), encoding="utf-8")
        raise
    print("Completed. See summary.csv and figures/.")


if __name__ == "__main__":
    main()
