"""Update audio paths in the dataset lists and check that every audio file exists."""

import csv
import os
from pathlib import Path


if __name__ == "__main__":
    # Thu muc chua ma nguon: LuanVan01/Code/ViSpeech.
    code_folder = Path(os.path.abspath(__file__)).parent

    dataset_file_names = [
        "train.txt",
        "validation.txt",
        "clean_testset.txt",
        "noisy_testset.txt",
    ]

    # Kiem tra tat ca duong dan truoc khi ghi thay doi.
    updated_dataset_rows = {}

    for dataset_file_name in dataset_file_names:
        dataset_file_path = code_folder / "dataset" / dataset_file_name

        with dataset_file_path.open("r", encoding="utf-8", newline="") as dataset_file:
            dataset_rows = list(csv.reader(dataset_file))

        for line_number, sample_row in enumerate(dataset_rows, start=1):
            original_audio_path = sample_row[0]
            normalized_audio_path = original_audio_path.replace("\\", "/")

            # Chi thay tien to dataset/; giu nguyen ten tep va nhan.
            if normalized_audio_path.startswith("dataset/"):
                audio_subpath = normalized_audio_path[len("dataset/"):]
                sample_row[0] = "../../Data/ViSpeech/" + audio_subpath

            audio_file_path = code_folder / sample_row[0]

            if not audio_file_path.is_file():
                raise FileNotFoundError(
                    f"{dataset_file_name}, dong {line_number}: {audio_file_path}"
                )

            # In mau dau tien cua moi danh sach de de kiem tra.
            if line_number == 1:
                absolute_audio_path = os.path.abspath(audio_file_path)
                print("Danh sach:", dataset_file_name)
                print("Duong dan ban dau:", original_audio_path)
                print("Duong dan trong danh sach:", sample_row[0])
                print("Duong dan thuc te:", absolute_audio_path)
                print("Tep ton tai:", True)
                print()

        updated_dataset_rows[dataset_file_name] = dataset_rows

    # Ghi danh sach da sua.
    for dataset_file_name in dataset_file_names:
        dataset_file_path = code_folder / "dataset" / dataset_file_name
        dataset_rows = updated_dataset_rows[dataset_file_name]

        with dataset_file_path.open("w", encoding="utf-8", newline="") as dataset_file:
            csv_writer = csv.writer(dataset_file)
            csv_writer.writerows(dataset_rows)

        print("Da cap nhat:", dataset_file_name, "-", len(dataset_rows), "mau")
