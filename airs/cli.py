from __future__ import annotations

import argparse
from pathlib import Path

from .config import SchedulerConfig, TIER_NAMES
from .dependencies import WORKSPACE_ROOT, ensure_rrtta_importable
from .resources import TRACE_NAMES

ensure_rrtta_importable()

from rrtta.runner import add_common_args, add_rrtta_args, apply_rrtta_args, config_from_args  # noqa: E402


DEFAULT_DATA_DIR = str(WORKSPACE_ROOT / "data")
DEFAULT_CKPT_DIR = str(WORKSPACE_ROOT / "checkpoints" / "robustbench")
DEFAULT_OUTPUT_ROOT = str(WORKSPACE_ROOT / "outputs" / "airs")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser, output_name="airs")
    add_rrtta_args(parser)
    parser.set_defaults(
        dataset="cifar10",
        data_dir=DEFAULT_DATA_DIR,
        ckpt_dir=DEFAULT_CKPT_DIR,
        output_root=DEFAULT_OUTPUT_ROOT,
    )


def add_scheduler_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        choices=[
            "airs",
            "forced",
            "static",
            "greedy",
            "periodic",
            "random",
            "value_only",
            "resource_only",
            "no_joint",
        ],
        default="airs",
    )
    parser.add_argument("--budget", type=float, default=60.0, help="Adaptation budget in percent")
    parser.add_argument("--credit-mode", choices=["unlimited", "bank", "fixed", "global"], default="bank")
    parser.add_argument("--credit-capacity", type=float, default=4.0)
    parser.add_argument("--pace-budget", action="store_true")
    parser.add_argument("--resource-trace", choices=TRACE_NAMES, default="steady")
    parser.add_argument("--fixed-tier", choices=TIER_NAMES, default="medium")
    parser.add_argument("--value-light-threshold", type=float, default=0.75)
    parser.add_argument("--value-medium-threshold", type=float, default=0.82)
    parser.add_argument("--value-full-threshold", type=float, default=0.88)
    parser.add_argument("--ignore-value", action="store_true")
    parser.add_argument("--ignore-resource", action="store_true")


def run_config_from_args(args):
    cfg = apply_rrtta_args(config_from_args(args, method="AIRS"), args)
    if cfg.dataset != "cifar10":
        raise ValueError("The AIRS experiment suite is defined for CIFAR-10-C only")
    cfg.rrtta.proto_align_weight = 0.0
    cfg.output_root = str(Path(args.output_root))
    return cfg


def scheduler_config_from_args(args, total_batches: int = 1) -> SchedulerConfig:
    return SchedulerConfig(
        policy=args.policy,
        budget_ratio=args.budget / 100.0,
        credit_mode=args.credit_mode,
        credit_capacity=args.credit_capacity,
        pace_budget=args.pace_budget,
        use_value=not args.ignore_value,
        use_resource=not args.ignore_resource,
        value_light_threshold=args.value_light_threshold,
        value_medium_threshold=args.value_medium_threshold,
        value_full_threshold=args.value_full_threshold,
        fixed_tier=args.fixed_tier,
        seed=args.seed,
        total_batches=total_batches,
    )
