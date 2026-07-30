#!/usr/bin/env python
from __future__ import annotations

import argparse

from airs.cli import add_model_args, run_config_from_args
from airs.experiments import EXPERIMENT_IDS, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all six AIRS experiment groups.")
    add_model_args(parser)
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    cfg = run_config_from_args(args)
    root = run_suite(
        cfg,
        experiments=EXPERIMENT_IDS,
        suite_id=args.suite_id,
        smoke=args.smoke,
        dry_run=args.dry_run,
        max_cases=args.max_cases,
    )
    print(root)


if __name__ == "__main__":
    main()
