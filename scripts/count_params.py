"""
count_params.py — Parameter counting utility
===============================================
Prints detailed parameter counts for model configurations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import small_config, medium_config
from model.transformer import GPT


def count_parameters(model):
    """Count parameters by module type."""
    total = 0
    breakdown = {}
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        # Group by top-level module
        top = name.split(".")[0]
        breakdown[top] = breakdown.get(top, 0) + n
    return total, breakdown


def main():
    for name, config_fn in [("Small", small_config), ("Medium", medium_config)]:
        model_cfg, _ = config_fn()

        print(f"\n{'='*60}")
        print(f" {name} Model Configuration")
        print(f"{'='*60}")
        print(f"  d_model:     {model_cfg.d_model}")
        print(f"  n_heads:     {model_cfg.n_heads}")
        print(f"  d_head:      {model_cfg.d_head}")
        print(f"  n_layers:    {model_cfg.n_layers}")
        print(f"  d_ff:        {model_cfg.d_ff}")
        print(f"  vocab_size:  {model_cfg.vocab_size}")
        print(f"  max_seq_len: {model_cfg.max_seq_len}")

        # Analytical estimate
        est = model_cfg.param_count_estimate()
        print(f"\n  --- Analytical Estimate ---")
        for k, v in est.items():
            if k == "total_M":
                print(f"  {k:>25s}: {v:.1f}M")
            else:
                print(f"  {k:>25s}: {v:>15,}")

        # Actual count from instantiated model
        model = GPT(model_cfg)
        actual, breakdown = count_parameters(model)
        print(f"\n  --- Actual (from model) ---")
        for module_name, count in sorted(breakdown.items()):
            print(f"  {module_name:>25s}: {count:>15,} ({count/actual*100:.1f}%)")
        print(f"  {'TOTAL':>25s}: {actual:>15,} ({actual/1e6:.1f}M)")

        print()


if __name__ == "__main__":
    main()
