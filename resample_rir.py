"""Resample a RIR audio file and save it to a separate file."""

from pathlib import Path

import torchaudio


def resample_rir(input_file_path, output_file_path, target_sample_rate):
    """Read a RIR, change its sample rate, and save the result."""
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)

    # Kiem tra tan so lay mau moi.
    if target_sample_rate <= 0:
        raise ValueError("Tan so lay mau phai lon hon 0.")

    # Giu nguyen tep RIR goc.
    if input_file_path.absolute() == output_file_path.absolute():
        raise ValueError("Tep dau ra phai khac tep RIR goc.")

    # Doc am thanh va tan so lay mau ban dau.
    rir_audio, original_sample_rate = torchaudio.load(str(input_file_path))

    # Resample am thanh neu tan so lay mau khac nhau.
    if original_sample_rate != target_sample_rate:
        rir_audio = torchaudio.functional.resample(
            rir_audio,
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
        )

    # Luu am thanh vao tep moi voi tan so lay mau moi.
    torchaudio.save(
        str(output_file_path),
        rir_audio,
        target_sample_rate,
    )

    print("Tep RIR goc:", input_file_path)
    print("Tan so ban dau:", original_sample_rate, "Hz")
    print("Tep RIR moi:", output_file_path)
    print("Tan so moi:", target_sample_rate, "Hz")

    return output_file_path


if __name__ == "__main__":
    # Chay tu thu muc Code/ViSpeech.
    input_file_path = "augmentation/RIR/rir.wav"

    # Luu rieng vi augmentation hien tai doc tep dau tien trong RIR/.
    output_file_path = "augmentation/rir_22050.wav"

    # Khop voi audio.sampling_rate trong config.yaml hien tai.
    target_sample_rate = 22050

    resample_rir(input_file_path, output_file_path, target_sample_rate)
