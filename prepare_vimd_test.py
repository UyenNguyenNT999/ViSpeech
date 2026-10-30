# /// script
# requires-python = ">=3.9"
# dependencies = ["pyarrow==17.0.0"]
# ///
"""Extract the downloaded ViMD test split for the existing evaluator.

uv run --script prepare_vimd_test.py --check-only
uv run --script prepare_vimd_test.py
Audio bytes are preserved. Resampling/cropping happens in the existing evaluator.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "Data" / "ViMD"
LABELS = {"North": 0, "Central": 1, "South": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    import pyarrow.parquet as pq

    manifest = json.loads((DATA / "download_manifest_test.json").read_text(encoding="utf-8"))
    shards = sorted(entry["path"] for entry in manifest["files"]
                    if entry["path"].startswith("data/test-") and entry["path"].endswith(".parquet"))
    if not shards:
        raise ValueError("No test shards in download manifest.")
    for entry in manifest["files"]:
        if entry["path"] in shards:
            path = DATA / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError("Missing/incomplete shard: " + str(path))
    counts = Counter()
    speakers = set()
    for shard in shards:
        table = pq.read_table(DATA / shard, columns=["region", "speakerID"])
        for row in table.to_pylist():
            if row["region"] not in LABELS:
                raise ValueError("Unknown region: " + str(row["region"]))
            counts[row["region"]] += 1
            speakers.add(row["speakerID"])
    if set(counts) != set(LABELS):
        raise ValueError("Test split must contain all three regions.")
    print("Test samples:", sum(counts.values()), "| Regions:", dict(counts), flush=True)
    print("Speaker IDs:", len(speakers), "| Label mapping:", LABELS, flush=True)
    # Validate one embedded audio payload even in check-only mode.
    first = next(pq.ParquetFile(DATA / shards[0]).iter_batches(batch_size=1, columns=["audio"]))
    payload = first.to_pylist()[0]["audio"]["bytes"]
    if not payload or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("Expected embedded WAV bytes; inspect audio encoding before extraction.")
    if args.check_only:
        print("Input check passed. No audio was extracted.")
        return

    audio_dir = DATA / "audio" / "test"
    audio_dir.mkdir(parents=True, exist_ok=True)
    list_path = ROOT / "dataset" / "vimd_test.txt"
    metadata_path = DATA / "test_metadata.csv"
    list_tmp = list_path.with_suffix(".tmp")
    metadata_tmp = metadata_path.with_suffix(".tmp")
    fields = ["audio_id", "audio_path", "label", "region", "province_code", "province_name",
              "speakerID", "gender", "filename", "shard", "row_index", "sha256"]
    index = 0
    with list_tmp.open("w", encoding="utf-8", newline="") as listing, \
            metadata_tmp.open("w", encoding="utf-8-sig", newline="") as metadata:
        writer = csv.writer(listing)
        meta_writer = csv.DictWriter(metadata, fieldnames=fields)
        meta_writer.writeheader()
        for shard in shards:
            print("Extracting:", shard, flush=True)
            row_index = 0
            for batch in pq.ParquetFile(DATA / shard).iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    audio = row["audio"]["bytes"]
                    if not audio or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                        raise ValueError("Unexpected audio encoding at {}:{}".format(shard, row_index))
                    path = audio_dir / ("{:06d}.wav".format(index))
                    digest = hashlib.sha256(audio).hexdigest()
                    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        temp = path.with_suffix(".tmp")
                        temp.write_bytes(audio)
                        temp.replace(path)
                    label = LABELS[row["region"]]
                    writer.writerow([path.as_posix(), label])
                    record = {field: row.get(field, "") for field in fields}
                    record.update(audio_id=index, audio_path=path.as_posix(), label=label,
                                  shard=shard, row_index=row_index, sha256=digest)
                    meta_writer.writerow(record)
                    index += 1
                    row_index += 1
    if index != sum(counts.values()):
        raise ValueError("Extracted count does not match metadata count.")
    metadata_tmp.replace(metadata_path)
    list_tmp.replace(list_path)
    print("Ready:", list_path, "| samples:", index)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
