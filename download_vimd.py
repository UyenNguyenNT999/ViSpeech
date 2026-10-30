# /// script
# requires-python = ">=3.9"
# dependencies = ["huggingface-hub==0.36.0"]
# ///
"""Download the official ViMD files without changing the training environment.

Run: uv run --script download_vimd.py
Test only: uv run --script download_vimd.py --split test
Preview files and size: uv run --script download_vimd.py --split test --dry-run

Audio and labels remain in the publisher's Parquet files; this script does not
extract WAV files, resample audio, change labels, or start training/evaluation.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ID = "nguyendv02/ViMD_Dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "Data" / "ViMD"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"),
                        default="all", help="Default: all splits.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=None,
                        help="Optional Hub commit. Otherwise reuse the locally recorded commit.")
    parser.add_argument("--dry-run", action="store_true", help="List sizes without downloading.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        parser.exit(1, "Run with: uv run --script download_vimd.py\n")

    output = args.output_dir.expanduser().resolve()
    source_file = output / "download_source.json"
    saved = json.loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else {}
    revision = args.revision or saved.get("revision") or "main"
    if saved and saved.get("repo_id") != REPO_ID:
        parser.error("Output directory belongs to a different dataset; choose another directory.")
    if saved.get("revision") and args.revision and args.revision != saved["revision"]:
        parser.error("Use a new output directory for another revision to avoid mixing versions.")

    info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
    if not info.sha:
        raise RuntimeError("Hub did not return a dataset commit.")
    # The published validation prefix can be 'validation', 'valid', or 'val'.
    prefixes = {"train": ("train-",), "test": ("test-",),
                "validation": ("validation-", "valid-", "val-")}
    selected = []
    for entry in info.siblings:
        name = entry.rfilename
        is_data = name.startswith("data/") and name.endswith(".parquet")
        matches = args.split == "all" or Path(name).name.startswith(prefixes[args.split])
        if is_data and matches:
            selected.append(entry)
    if not selected:
        raise RuntimeError("No matching Parquet files found; check the publisher's repository.")
    if any(entry.size is None for entry in selected):
        raise RuntimeError("Missing remote file sizes; cannot verify download completeness.")
    docs = [entry for entry in info.siblings
            if entry.rfilename in ("README.md", ".gitattributes", "LICENSE", "LICENSE.md")]
    files = selected + docs
    total = sum(entry.size for entry in selected)
    print("Source: https://huggingface.co/datasets/" + REPO_ID)
    print("Revision:", info.sha)
    print("Destination:", output)
    print("Split: {} | Parquet files: {} | Size: {:.2f} GB ({:.2f} GiB)".format(
        args.split, len(selected), total / 1e9, total / 1024**3))
    if args.dry_run:
        for entry in selected:
            print("  {} ({:.1f} MB)".format(entry.rfilename, entry.size / 1e6))
        return

    output.mkdir(parents=True, exist_ok=True)
    remaining = sum(entry.size for entry in selected
                    if not (output / entry.rfilename).is_file()
                    or (output / entry.rfilename).stat().st_size != entry.size)
    # Conservative: incomplete downloads may already occupy some of this space.
    if shutil.disk_usage(output).free < remaining + 512 * 1024**2:
        raise RuntimeError("Not enough free space. Need about {:.2f} GiB including margin. "
                           "Try --split test or --output-dir on another drive.".format(
                               (remaining + 512 * 1024**2) / 1024**3))
    source_file.write_text(json.dumps({"repo_id": REPO_ID, "revision": info.sha,
                                     "source_url": "https://huggingface.co/datasets/" + REPO_ID},
                                    indent=2) + "\n", encoding="utf-8")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=info.sha,
                      local_dir=str(output),
                      allow_patterns=[entry.rfilename for entry in files], max_workers=2)
    for entry in selected:
        path = output / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError("Incomplete download: {}. Run the same command again.".format(path))
    manifest = {"repo_id": REPO_ID, "revision": info.sha, "split": args.split,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": [{"path": entry.rfilename, "bytes": entry.size} for entry in files]}
    (output / ("download_manifest_" + args.split + ".json")).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Download complete. Original audio and labels are inside data/*.parquet.")
    print("No WAV extraction, training, or evaluation has been performed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print("Download failed: {}\nRun the same command again to retry.".format(exc),
              file=sys.stderr)
        sys.exit(1)
