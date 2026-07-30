#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import torch

from rrtta.runner import (
    DEFAULT_ARCH,
    DEFAULT_CKPT_DIR,
    DEFAULT_CORRUPTIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_EX,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEVERITY,
    apply_rrtta_args,
    config_from_args,
    set_seed,
)
from robustbench.data import load_cifar10c
try:
    from robustbench.data import load_cifar100c
except ImportError:
    load_cifar100c = None
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from rrtta.method import setup_rrtta


DEFAULT_DATASET = "cifar10"
DEFAULT_BATCH_SIZES = [32, 64, 128, 200, 256]


class PrintLogger:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def info(self, msg, *args):
        text = msg % args if args else msg
        print(f"[{self.prefix}] {text}")

    def warning(self, msg, *args):
        self.info(msg, *args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure RRTTA sensitivity to the online test batch size."
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default=DEFAULT_DATASET)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--teacher-arch", default="")
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / "batch_size_sensitivity"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--num-ex", type=int, default=DEFAULT_NUM_EX)
    parser.add_argument("--severity", type=int, default=DEFAULT_SEVERITY)
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--teacher-momentum", type=float, default=0.999)
    parser.add_argument("--adapt-batch-size", type=int, default=16)
    parser.add_argument("--refine-label-margin", type=float, default=0.02)
    parser.add_argument("--student-disagreement-weight", type=float, default=0.25)
    parser.add_argument("--prediction-blend", type=float, default=0.0)
    parser.add_argument("--sce-alpha", type=float, default=0.70)
    parser.add_argument("--sce-beta", type=float, default=0.30)
    parser.add_argument("--reliability-high-threshold", type=float, default=0.40)
    parser.add_argument("--reliability-low-threshold", type=float, default=0.10)
    parser.add_argument("--max-update-fraction", type=float, default=1.00)
    parser.add_argument("--augment-times", type=int, default=4)
    parser.add_argument("--flip-p", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.20)
    parser.add_argument("--contrast", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--high-loss-weight", type=float, default=1.0)
    parser.add_argument("--medium-loss-weight", type=float, default=0.75)
    parser.add_argument("--low-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-conf-threshold", type=float, default=0.50)
    parser.add_argument("--proto-momentum", type=float, default=0.95)
    parser.add_argument("--proto-align-weight", type=float, default=0.02)
    parser.add_argument("--proto-warmup-batches", type=int, default=5)
    parser.add_argument("--proto-min-count", type=int, default=2)
    parser.add_argument("--proto-all-routes", action="store_true")
    return parser.parse_args()


def validate_batch_sizes(batch_sizes):
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("all batch sizes must be positive")
    return list(dict.fromkeys(batch_sizes))


def build_cfg(args, batch_size: int):
    cfg_args = argparse.Namespace(
        dataset=args.dataset,
        arch=args.arch,
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        batch_size=batch_size,
        num_ex=args.num_ex,
        severity=args.severity,
        corruptions=args.corruptions,
        seed=args.seed,
        lr=args.lr,
        steps=args.steps,
        episodic=args.episodic,
    )
    return apply_rrtta_args(config_from_args(cfg_args), args)


def load_corruption(dataset: str, num_ex: int, severity: int, data_dir: str, corruption: str):
    if dataset == "cifar10":
        loader = load_cifar10c
    elif load_cifar100c is None:
        raise RuntimeError("CIFAR100-C requires a newer RobustBench data loader.")
    else:
        loader = load_cifar100c
    return loader(num_ex, severity, data_dir, False, [corruption])


def evaluate_tensor(model, x_test, y_test, batch_size: int):
    correct = 0
    total = 0
    for start in range(0, x_test.shape[0], batch_size):
        end = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:end]
        yb = y_test[start:end]
        with torch.no_grad():
            logits = model(xb)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    return correct, total


def write_csv(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, summary_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    path = Path(path)
    rows = sorted(summary_rows, key=lambda row: int(row["batch_size"]))
    batch_sizes = [int(row["batch_size"]) for row in rows]
    errors = [float(row["mean_error_percent"]) for row in rows]
    times = [float(row["total_time_seconds"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(
        batch_sizes,
        errors,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color="#3b6f8f",
    )
    axes[0].set_xlabel("Online test batch size")
    axes[0].set_ylabel("Mean error (%)")
    error_upper = max(5, int(math.ceil(max(errors) * 1.06 / 5.0) * 5))
    axes[0].set_ylim(0, error_upper)
    axes[0].set_yticks(range(0, error_upper + 1, 5))
    axes[0].set_xticks(batch_sizes)
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.6)
    for batch_size, error in zip(batch_sizes, errors):
        axes[0].annotate(
            f"{error:.2f}",
            (batch_size, error),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    axes[1].plot(
        batch_sizes,
        times,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color="#b55d3f",
    )
    axes[1].set_xlabel("Online test batch size")
    axes[1].set_ylabel("Total runtime (s)")
    runtime_upper = max(250, int(math.ceil(max(times) * 1.08 / 250.0) * 250))
    axes[1].set_ylim(0, runtime_upper)
    axes[1].set_yticks(range(0, runtime_upper + 1, 250))
    axes[1].set_xticks(batch_sizes)
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.6)
    for batch_size, runtime in zip(batch_sizes, times):
        axes[1].annotate(
            f"{runtime:.0f}",
            (batch_size, runtime),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run_batch_size(args, batch_size: int):
    set_seed(args.seed)
    cfg = build_cfg(args, batch_size)
    logger = PrintLogger(f"batch_size={batch_size}")
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    base_model = load_model(
        cfg.arch,
        cfg.ckpt_dir,
        cfg.dataset,
        ThreatModel.corruptions,
    ).cuda()
    model = setup_rrtta(base_model, cfg, logger)
    model.reset()

    rows = []
    errors = []
    total_start = time.time()
    for index, corruption in enumerate(args.corruptions, start=1):
        logger.info("corruption %s/%s %s start", index, len(args.corruptions), corruption)
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        x_test, y_test = load_corruption(
            args.dataset,
            args.num_ex,
            args.severity,
            args.data_dir,
            corruption,
        )
        x_test, y_test = x_test.cuda(), y_test.cuda()
        correct, total = evaluate_tensor(model, x_test, y_test, batch_size)
        accuracy = correct / max(1, total)
        error = 1.0 - accuracy
        elapsed = time.time() - start_time
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        num_batches = int(math.ceil(total / float(batch_size)))
        errors.append(error)
        rows.append(
            {
                "dataset": args.dataset,
                "method": "RRTTA",
                "batch_size": batch_size,
                "adapt_batch_size": args.adapt_batch_size,
                "severity": args.severity,
                "corruption": corruption,
                "num_examples": total,
                "num_batches": num_batches,
                "correct": correct,
                "accuracy_percent": accuracy * 100.0,
                "error_percent": error * 100.0,
                "time_seconds": elapsed,
                "peak_gpu_mb": peak_gpu_mb,
            }
        )
        logger.info(
            "error %% [%s%s]: %.2f%% | time: %.1fs | peak_gpu: %.0fMB",
            corruption,
            args.severity,
            error * 100.0,
            elapsed,
            peak_gpu_mb,
        )
        del x_test, y_test

    total_time = time.time() - total_start
    mean_error = sum(errors) / len(errors) if errors else 0.0
    summary = {
        "dataset": args.dataset,
        "method": "RRTTA",
        "batch_size": batch_size,
        "adapt_batch_size": args.adapt_batch_size,
        "mean_error_percent": mean_error * 100.0,
        "mean_accuracy_percent": 100.0 - mean_error * 100.0,
        "num_corruptions": len(rows),
        "num_examples": sum(int(row["num_examples"]) for row in rows),
        "num_batches": sum(int(row["num_batches"]) for row in rows),
        "total_time_seconds": total_time,
        "mean_corruption_time_seconds": sum(float(row["time_seconds"]) for row in rows) / max(1, len(rows)),
        "peak_gpu_mb": max((float(row["peak_gpu_mb"]) for row in rows), default=0.0),
        "arch": cfg.arch,
        "seed": args.seed,
        "severity": args.severity,
    }
    del model, base_model
    torch.cuda.empty_cache()
    return rows, summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch-size sensitivity evaluation.")
    args.batch_sizes = validate_batch_sizes(args.batch_sizes)
    run_id = args.run_id or f"{args.dataset}_batch_size_sensitivity_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}")
    print(f"batch_sizes={args.batch_sizes}")

    all_rows = []
    summary_rows = []
    for batch_size in args.batch_sizes:
        rows, summary = run_batch_size(args, batch_size)
        all_rows.extend(rows)
        summary_rows.append(summary)

    per_corruption_path = output_dir / f"batch_size_per_corruption_{run_id}.csv"
    summary_path = output_dir / f"batch_size_summary_{run_id}.csv"
    plot_path = output_dir / f"batch_size_sensitivity_{run_id}.png"
    write_csv(per_corruption_path, all_rows)
    write_csv(summary_path, summary_rows)
    save_plot(plot_path, summary_rows)
    print(f"saved {per_corruption_path}")
    print(f"saved {summary_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
