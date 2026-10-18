"""Tạo danh mục noise_metadata.csv cho các tệp âm thanh nhiễu.

Mặc định đọc augmentation/evaluation_noise/audio, kể cả thư mục con.
CSV chỉ chứa noise_id và path để tìm tệp nhiễu và chạy đánh giá.
Tên tệp trong path được dùng để nhận diện nhiễu khi báo cáo kết quả.
Chạy lại sẽ giữ mã nhiễu cũ, đồng thời thêm các tệp mới.
Các cột khác trong CSV cũ sẽ bị bỏ khỏi CSV được tạo lại.

Có thể đổi đường dẫn bằng --audio-dir và --output nếu cần.
"""

import argparse
import csv
import os
from pathlib import Path


# Các cột trong CSV, theo đúng thứ tự xuất ra.
FIELDS = [
    "noise_id",
    "path",
]
AUDIO_EXTENSIONS = {".wav", ".mp3"}
BASE_DIR = Path(os.path.abspath(__file__)).parent
DEFAULT_AUDIO_DIR = BASE_DIR / "evaluation_noise" / "audio"


def find_audio_files(audio_dir):
    """Tìm các tệp có đuôi âm thanh và sắp xếp theo đường dẫn."""
    if not audio_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục âm thanh: {audio_dir}")

    audio_files = []
    for audio_path in audio_dir.rglob("*"):
        if (audio_path.is_file()
                and audio_path.suffix.lower() in AUDIO_EXTENSIONS):
            audio_files.append(audio_path)

    return sorted(audio_files)


def read_existing_metadata(csv_path):
    """Đọc hai cột cần dùng; chấp nhận cả CSV ba hoặc sáu cột cũ."""
    rows_by_path = {}
    used_ids = set()
    if not csv_path.exists():
        return rows_by_path, used_ids

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("CSV cũ không đúng các cột yêu cầu. Không ghi đè.")

        for row in reader:
            relative_path = row["path"].replace("\\", "/")
            noise_id = row["noise_id"]

            if relative_path in rows_by_path:
                raise ValueError(f"Đường dẫn bị trùng trong CSV: {relative_path}")
            if noise_id and noise_id in used_ids:
                raise ValueError(f"Mã nhiễu bị trùng trong CSV: {noise_id}")

            # Chỉ giữ cột cần thiết, không mang các cột khác sang CSV mới.
            rows_by_path[relative_path] = {
                "noise_id": noise_id,
                "path": relative_path,
            }
            if noise_id:
                used_ids.add(noise_id)

    return rows_by_path, used_ids


def create_noise_id(used_ids):
    """Chọn mã chưa được sử dụng: noise_0001, noise_0002, ..."""
    number = 1
    noise_id = f"noise_{number:04d}"
    while noise_id in used_ids:
        number += 1
        noise_id = f"noise_{number:04d}"

    used_ids.add(noise_id)
    return noise_id


def write_metadata(csv_path, rows):
    """Ghi tệp tạm trước; chỉ thay CSV chính sau khi ghi xong."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(csv_path.name + ".tmp")

    # utf-8-sig giúp Excel đọc được tiếng Việt.
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(str(temporary_path), str(csv_path))


def create_metadata(audio_dir, output):
    # Bước 1: Xác định đường dẫn đầy đủ và tìm âm thanh.
    audio_dir = Path(os.path.abspath(str(audio_dir)))
    output = Path(os.path.abspath(str(output)))
    audio_files = find_audio_files(audio_dir)

    # Bước 2: Đọc thông tin cũ (nếu đã có CSV).
    rows_by_path, used_ids = read_existing_metadata(output)
    result_rows = []
    found_paths = set()

    # Bước 3: Giữ hàng cũ, hoặc tạo một hàng cho tệp mới.
    for audio_path in audio_files:
        relative_path = os.path.relpath(
            str(audio_path), str(output.parent)
        )
        relative_path = Path(relative_path).as_posix()
        found_paths.add(relative_path)

        if relative_path in rows_by_path:
            result_rows.append(rows_by_path[relative_path])
            continue

        new_row = {
            "noise_id": create_noise_id(used_ids),
            "path": relative_path,
        }
        result_rows.append(new_row)

    # Bước 4: Giữ cả thông tin của tệp cũ không còn thấy trong lần quét này.
    retained_count = 0
    for relative_path, old_row in rows_by_path.items():
        if relative_path not in found_paths:
            result_rows.append(old_row)
            retained_count += 1

    # Bước 5: Ghi kết quả.
    write_metadata(output, result_rows)
    print(f"Đã ghi CSV: {output}")
    print(f"Số tệp âm thanh tìm thấy: {len(audio_files)}")
    print(f"Số hàng cũ không thấy trong lần quét nhưng vẫn giữ: {retained_count}")
    if not audio_files:
        print("Không tìm thấy âm thanh. Chỉ ghi tiêu đề cột và các hàng cũ nếu có.")
    print("CSV gồm: noise_id, path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
        help="Thư mục chứa âm thanh nhiễu",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Đường dẫn CSV; mặc định là noise_metadata.csv cạnh thư mục audio",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.audio_dir.parent / "noise_metadata.csv"

    try:
        create_metadata(args.audio_dir, output)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Lỗi: {error}\n")


if __name__ == "__main__":
    main()
