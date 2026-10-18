"""Khảo sát SNR do cách thêm nhiễu lúc train tạo ra. Không train mô hình.

Đọc config_noise_eval.yaml. Mô phỏng đúng cách chọn mức Gaussian và
đoạn tiếng nền trong augmentation/augment.py hiện tại, nhưng đo mọi mức
Gaussian và mọi tệp tiếng nền trên cùng nhóm train để dễ so sánh.
Đây không phải khôi phục lịch sử hoặc tần suất augmentation của lần train cũ.
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from tqdm import tqdm

from evaluate_noise_robustness import (
    SCRIPT_DIRECTORY, full_path, load_audio, prepare_speech,
    measure_snr, write_csv, file_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIRECTORY / "config_noise_eval.yaml")
    args = parser.parse_args()
    config_path = Path(os.path.abspath(str(args.config)))
    with config_path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent
    settings = config["training_augmentation_check"]
    for key in ["sample_count", "repeats"]:
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} phải là số nguyên dương.")
    python_random = random.Random(settings["seed"])
    gaussian_random = np.random.default_rng(settings["seed"])
    speech_list = full_path(root, settings["speech_list_file"])
    speech_root = full_path(root, config["dataset"]["speech_paths_directory"])
    samples_by_dialect = {label: [] for label in range(3)}
    with speech_list.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                label = int(row[1])
                if label not in range(6):
                    raise ValueError(f"Nhãn không hợp lệ: {row}")
                samples_by_dialect[label % 3].append((full_path(speech_root, row[0]), label % 3))
    for group in samples_by_dialect.values():
        python_random.shuffle(group)
    selected = []
    while len(selected) < settings["sample_count"] and any(samples_by_dialect.values()):
        for group in samples_by_dialect.values():
            if group and len(selected) < settings["sample_count"]:
                selected.append(group.pop())
    if not selected:
        raise ValueError("Danh sách train không có mẫu.")
    noise_folder = full_path(root, settings["noise_directory"])
    noise_paths = sorted(p for p in noise_folder.glob("*") if p.is_file())
    if not noise_paths:
        raise ValueError("Thư mục tiếng nền train không có tệp.")
    audio = config["audio"]
    rate = audio["sampling_rate"]
    count = int(audio["length_seconds"] * 100) * audio["hop_length"]
    count += audio["window_length"] - audio["hop_length"]
    loaded_noises = [(path, load_audio(path, rate)) for path in noise_paths]
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    output = full_path(root, settings["output_directory"]) / run_name
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    rows = []
    try:
        for speech_path, label in tqdm(selected, desc="Check training augmentation"):
            speech = prepare_speech(speech_path, rate, count)
            for repeat in range(settings["repeats"]):
                # Các mức này lấy từ augment.py; đo riêng từng mức, không lấy tần suất train.
                for deviation in [0.0008, 0.001, 0.002, 0.0025, 0.0015]:
                    noise = gaussian_random.normal(0, deviation, speech.shape)
                    before_clip = speech + noise
                    mixed = np.clip(before_clip, -1.0, 1.0)
                    rows.append({"audio_path": str(speech_path), "true_label": label,
                                 "augmentation": "gaussian", "noise_file": "",
                                 "repeat_id": repeat, "gaussian_std": deviation,
                                 "noise_start_sample": "", "noise_repeated": False,
                                 "added_noise_snr_db": measure_snr(speech, noise),
                                 "effective_snr_db": measure_snr(speech, mixed - speech),
                                 "peak_output": float(np.max(np.abs(mixed))),
                                 "clipped_fraction": float(np.mean(np.abs(before_clip) > 1))})
                for noise_path, noise_audio in loaded_noises:
                    # Baseline lấy offset theo bước 10000 mẫu, không chọn mọi vị trí.
                    start = 0
                    if len(speech) < len(noise_audio):
                        start = python_random.randint(0, (len(noise_audio) - len(speech)) // 10000) * 10000
                    segment = noise_audio[start:start + count]
                    if len(segment) < count:
                        segment = np.pad(segment, (0, count - len(segment)), mode="wrap")
                    added = 0.3 * segment
                    mixed = speech + added
                    rows.append({"audio_path": str(speech_path), "true_label": label,
                                 "augmentation": "background", "noise_file": str(noise_path),
                                 "repeat_id": repeat, "gaussian_std": "",
                                 "noise_start_sample": start, "noise_repeated": len(noise_audio) < count,
                                 "added_noise_snr_db": measure_snr(speech, added),
                                 "effective_snr_db": measure_snr(speech, mixed.astype(np.float64) - speech),
                                 "peak_output": float(np.max(np.abs(mixed))), "clipped_fraction": 0.0})
        write_csv(output / "augmentation_snr.csv", rows)
        information = {"status": "completed", "speech_samples": len(selected),
                       "training_list_sha256": file_hash(speech_list),
                       "augmentation_code_sha256": file_hash(SCRIPT_DIRECTORY / "augmentation" / "augment.py"),
                       "noise_files": [{"path": str(p), "sha256": file_hash(p)} for p in noise_paths],
                       "note": "Mô phỏng công thức hiện tại; không khôi phục lịch sử train. Không đo RIR/tốc độ bằng SNR cộng nhiễu."}
        (output / "run_info.json").write_text(json.dumps(information, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as error:
        (output / "FAILED.txt").write_text(repr(error), encoding="utf-8")
        raise
    print(f"Kết quả: {output}")


if __name__ == "__main__":
    main()
