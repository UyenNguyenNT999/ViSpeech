"""Cắt nhiễu thành các đoạn bằng nhau, lưu riêng, không sửa âm thanh gốc.

Chạy từ Code/ViSpeech:
    uv run --locked python augmentation/split_noise_audio.py

Mặc định đọc evaluation_noise/audio. Độ dài đoạn được tính từ
config_noise_eval.yaml, giống độ dài đầu vào thực tế của mô hình.
Không chồng lấn, không thêm khoảng lặng, không lặp phần cuối thiếu.
Mặc định chỉ lưu 1 đoạn ngẫu nhiên hợp lệ từ mỗi tệp gốc.
Dùng --segments-per-file để chọn nhiều hơn, --seed để đổi lần chọn.
Không tự sửa cấu hình đánh giá. Không ghi đè thư mục kết quả đã có.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml


BASE_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = BASE_DIR.parent
AUDIO_EXTENSIONS = {".wav", ".mp3"}


def absolute_path(path):
    return Path(os.path.abspath(str(path)))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def split_audio(input_dir, output_dir, config_path, segments_per_file=1, seed=42):
    if segments_per_file <= 0:
        raise ValueError("Số đoạn mỗi tệp phải lớn hơn 0.")
    input_dir = absolute_path(input_dir)
    output_dir = absolute_path(output_dir)
    config_path = absolute_path(config_path)
    if not input_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục đầu vào: {input_dir}")
    # Tránh quét cả kết quả của chính script trong những lần sử dụng sau.
    common_path = os.path.normcase(os.path.commonpath([str(input_dir), str(output_dir)]))
    if common_path in {os.path.normcase(str(input_dir)), os.path.normcase(str(output_dir))}:
        raise ValueError("Thư mục đầu vào và đầu ra phải riêng biệt, không nằm trong nhau.")
    if output_dir.exists():
        raise ValueError(f"Thư mục đầu ra đã tồn tại, không ghi đè: {output_dir}. Dùng --output-dir khác.")

    with config_path.open(encoding="utf-8-sig") as config_file:
        config = yaml.safe_load(config_file)
    audio_settings = config["audio"]
    sample_rate = audio_settings["sampling_rate"]
    segment_samples = int(audio_settings["length_seconds"] * 100) * audio_settings["hop_length"]
    segment_samples += audio_settings["window_length"] - audio_settings["hop_length"]
    if sample_rate <= 0 or segment_samples <= 0:
        raise ValueError("Tần số lấy mẫu và độ dài đoạn phải lớn hơn 0.")

    audio_files = sorted(path for path in input_dir.rglob("*")
                         if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    if not audio_files:
        raise ValueError("Không tìm thấy tệp âm thanh trong thư mục đầu vào.")

    audio_output = output_dir / "audio"
    audio_output.mkdir(parents=True, exist_ok=False)
    metadata_rows = []
    segment_rows = []
    source_rows = []
    run_information = {
        "status": "running", "input_directory": str(input_dir),
        "config_path": str(config_path), "sampling_rate": sample_rate,
        "segment_samples": segment_samples,
        "segment_seconds": segment_samples / sample_rate,
        "segments_per_file": segments_per_file,
        "seed": seed,
        "rules": "mono; resample; randomly select full non-overlapping segments without replacement; discard short remainder; no volume normalization",
        "note": "Các đoạn cùng source_recording_id không phải nguồn nhiễu độc lập.",
    }

    def save_reports():
        write_csv(output_dir / "noise_metadata.csv", ["noise_id", "path"], metadata_rows)
        write_csv(output_dir / "segment_sources.csv", [
            "noise_id", "path", "source_recording_id", "source_path", "segment_number",
            "start_sample", "end_sample", "sampling_rate", "duration_seconds", "peak",
        ], segment_rows)
        write_csv(output_dir / "split_report.csv", [
            "source_path", "source_recording_id", "duration_seconds", "saved_segments",
            "silent_segments_skipped", "discarded_tail_seconds", "status", "error",
        ], source_rows)
        (output_dir / "run_info.json").write_text(
            json.dumps(run_information, ensure_ascii=False, indent=2), encoding="utf-8")

    save_reports()
    try:
        for source_path in audio_files:
            relative_source = source_path.relative_to(input_dir).as_posix()
            source_id = "source_" + hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:12]
            print(f"Đang cắt: {relative_source}", flush=True)
            try:
                audio, _ = librosa.load(str(source_path), sr=sample_rate, mono=True)
                if len(audio) == 0 or not np.isfinite(audio).all():
                    raise ValueError("Âm thanh rỗng hoặc chứa giá trị không hợp lệ.")
            except Exception as error:
                source_rows.append({"source_path": str(source_path), "source_recording_id": source_id,
                                    "duration_seconds": "", "saved_segments": 0,
                                    "silent_segments_skipped": 0, "discarded_tail_seconds": "",
                                    "status": "read_error", "error": str(error)})
                continue

            full_segments = len(audio) // segment_samples
            saved_count = 0
            silent_count = 0
            # Xáo trộn các đoạn đủ độ dài; chọn không lặp trong mỗi tệp.
            # Mỗi nguồn có seed riêng: thêm tệp khác không đổi lựa chọn của nguồn cũ.
            random_seed = int(hashlib.sha256(
                f"{seed}:{relative_source}".encode("utf-8")
            ).hexdigest()[:16], 16)
            random_generator = np.random.default_rng(random_seed)
            segment_order = random_generator.permutation(full_segments)
            for segment_index in segment_order:
                segment_index = int(segment_index)
                start = segment_index * segment_samples
                end = start + segment_samples  # end không bao gồm mẫu tại vị trí này.
                segment = audio[start:end]
                # Script đánh giá không trộn được với đoạn có công suất gần 0.
                if float(np.mean(segment.astype(np.float64) ** 2)) <= 1e-12:
                    silent_count += 1
                    continue
                filename = f"{source_path.stem}_{source_id}_part_{segment_index + 1:04d}.wav"
                destination = audio_output / filename
                # WAV float tránh mất chất lượng do mã hóa MP3 lại hoặc cắt ngọn.
                sf.write(str(destination), segment, sample_rate, subtype="FLOAT")
                noise_id = f"{source_id}_part_{segment_index + 1:04d}"
                relative_output = f"audio/{filename}"
                metadata_rows.append({"noise_id": noise_id, "path": relative_output})
                segment_rows.append({
                    "noise_id": noise_id, "path": relative_output,
                    "source_recording_id": source_id, "source_path": str(source_path),
                    "segment_number": segment_index + 1, "start_sample": start, "end_sample": end,
                    "sampling_rate": sample_rate, "duration_seconds": segment_samples / sample_rate,
                    "peak": float(np.max(np.abs(segment))),
                })
                saved_count += 1
                if saved_count >= segments_per_file:
                    break
            source_rows.append({
                "source_path": str(source_path), "source_recording_id": source_id,
                "duration_seconds": len(audio) / sample_rate, "saved_segments": saved_count,
                "silent_segments_skipped": silent_count,
                "discarded_tail_seconds": (len(audio) % segment_samples) / sample_rate,
                "status": ("too_short" if full_segments == 0 else
                           "no_valid_segment" if saved_count == 0 else "processed"),
                "error": "",
            })
            save_reports()
        if not metadata_rows:
            raise ValueError("Không tạo được đoạn nhiễu hợp lệ. Xem split_report.csv.")
        run_information["status"] = "completed_with_read_errors" if any(
            row["status"] == "read_error" for row in source_rows) else "completed"
        run_information["saved_segments"] = len(metadata_rows)
        save_reports()
    except Exception as error:
        run_information["status"] = "failed"
        run_information["error"] = str(error)
        save_reports()
        raise

    print(f"Đã lưu {len(metadata_rows)} đoạn, mỗi đoạn {segment_samples / sample_rate:.6f} giây.")
    print(f"Chọn tối đa {segments_per_file} đoạn ngẫu nhiên mỗi tệp gốc; seed={seed}.")
    print(f"Thư mục mới: {output_dir}")
    print("Xem split_report.csv để biết tệp ngắn, phần cuối bị bỏ và lỗi đọc.")
    print("Muốn dùng khi đánh giá: đổi noise.noise_metadata_file trong config_noise_eval.yaml")
    print(f"thành đường dẫn tới: {output_dir / 'noise_metadata.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "evaluation_noise" / "audio")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "evaluation_noise_segments")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config_noise_eval.yaml")
    parser.add_argument("--segments-per-file", type=int, default=1,
                        help="Số đoạn ngẫu nhiên tối đa mỗi tệp gốc; mặc định 1")
    parser.add_argument("--seed", type=int, default=42,
                        help="Giữ cùng seed và tệp nguồn để chọn lại cùng đoạn; mặc định 42")
    args = parser.parse_args()
    try:
        split_audio(args.input_dir, args.output_dir, args.config,
                    args.segments_per_file, args.seed)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
