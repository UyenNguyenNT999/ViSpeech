"""Evaluate paired A/B using the exact existing evaluation implementation.

Example:
    python evaluate_noise_robustness_pairse.py --config config_noise_eval_paired_A.yaml

No training. Explicit --config is required to avoid evaluating the old model
by accident. Input preparation, random noise offsets, SNR mixing, metrics and
output formats are all delegated to evaluate_noise_robustness.py unchanged.
"""
import sys
from evaluate_noise_robustness import main


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
            raise SystemExit(
                "Specify --config config_noise_eval_paired_A.yaml "
                "or --config config_noise_eval_paired_B.yaml"
            )
    main()
