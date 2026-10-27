"""Freeze and verify the historical noisy validation plan. No audio is written."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml
from evaluate_noise_robustness import file_hash, read_inputs

ROOT = Path(__file__).resolve().parent


def resolve(value):
    return (ROOT / value).resolve()


def verify_bundle(directory):
    directory = resolve(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = resolve(record["path"])
        if not path.is_file() or file_hash(path) != record["sha256"]:
            raise ValueError(f"File missing or changed: {path}")
    for name, digest in manifest["bundle_hashes"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Frozen bundle changed: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_fixed_noise_validation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8-sig"))
    reference = resolve(config["reference_run"])
    target = resolve(config["output_directory"])
    if target.exists():
        manifest = verify_bundle(target)
        if manifest["reference_run"] != str(reference):
            raise ValueError("Existing bundle uses another reference; choose a new output directory")
        print(f"Existing bundle verified: {target}. No files overwritten.")
        return
    evaluation = yaml.safe_load((reference / "config.yaml").read_text(encoding="utf-8-sig"))
    info = json.loads((reference / "run_info.json").read_text(encoding="utf-8"))
    if info["status"] != "completed":
        raise ValueError("Reference evaluation is not completed")
    samples, noises, speech_list, noise_csv = read_inputs(evaluation, ROOT)
    if file_hash(speech_list) != info["dataset_sha256"] or file_hash(noise_csv) != info["noise_csv_sha256"]:
        raise ValueError("Input lists differ from the historical evaluation")
    with (reference / "mixture_plan.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(s["audio_id"], n["noise_id"], float(snr), repeat)
                for s in samples for n in noises for snr in evaluation["noise"]["snr_db"]
                for repeat in range(evaluation["noise"]["repeats"])}
    actual = {(int(r["audio_id"]), r["noise_id"], float(r["snr_db"]), int(r["repeat_id"])) for r in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError("Incomplete or duplicate historical mixture plan")
    noise_hashes = {n["noise_id"]: file_hash(Path(n["path"])) for n in noises}
    for row in rows:
        if row["noise_sha256"] != noise_hashes[row["noise_id"]]:
            raise ValueError("Actual noise audio differs from reference evaluation")
    print("Hashing original speech and noise files to freeze future validation inputs...", flush=True)
    paths = sorted({Path(s["audio_path"]) for s in samples} | {Path(n["path"]) for n in noises}
                   | {speech_list, noise_csv})
    files = [{"path": str(p), "sha256": file_hash(p)} for p in paths]
    manifest = {"reference_run": str(reference), "purpose": "validation only; not independent test",
                "n_speech": len(samples), "n_mixtures": len(rows), "files": files,
                "snr_definition": "whole-clip RMS including silence; evaluation has no clipping/rescaling",
                "historical_speech_hash_note": "Speech hashes frozen now; historical run did not hash speech bytes."}
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(reference / "mixture_plan.csv", target / "mixture_plan.csv")
    (target / "evaluation_config.yaml").write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    manifest["bundle_hashes"] = {name: file_hash(target / name)
                                  for name in ("mixture_plan.csv", "evaluation_config.yaml")}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_bundle(target)
    print(f"READY: {len(samples)} originals, {len(rows)} fixed mixtures. Plan: {target}")


if __name__ == "__main__":
    main()
