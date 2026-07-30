#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

from airs.cli import add_model_args, add_scheduler_args, run_config_from_args, scheduler_config_from_args
from airs.evaluation import evaluate_airs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AIRS online evaluation.")
    add_model_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    cfg = run_config_from_args(args)
    total_batches = len(cfg.corruptions) * math.ceil(cfg.num_ex / cfg.batch_size)
    scheduler_cfg = scheduler_config_from_args(args, total_batches)
    run_id = args.run_id or f"airs_{args.policy}_b{int(args.budget):03d}_s{args.severity}_seed{args.seed}"
    run_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_root) / "single" / run_id
    evaluate_airs(cfg, scheduler_cfg, args.resource_trace, run_dir, run_id)


if __name__ == "__main__":
    main()
