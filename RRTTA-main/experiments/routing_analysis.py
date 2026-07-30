#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from rrtta.runner import (
    add_common_args,
    add_rrtta_args,
    apply_rrtta_args,
    config_from_args,
    load_corruption,
)
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from rrtta.method import route_by_reliability, setup_rrtta


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze RRTTA high/medium/low routing ratios.")
    add_common_args(parser, output_name="routing_analysis")
    add_rrtta_args(parser)
    return parser.parse_args()


def build_cfg(args):
    return apply_rrtta_args(config_from_args(args), args)


class PrintLogger:
    def info(self, msg, *args):
        print(msg % args if args else msg)

    def warning(self, msg, *args):
        print(msg % args if args else msg)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_stack_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["global_batch"]) for row in rows]
    high = [float(row["high_ratio"]) for row in rows]
    medium = [float(row["medium_ratio"]) for row in rows]
    low = [float(row["low_ratio"]) for row in rows]
    plt.figure(figsize=(11, 4.5))
    plt.stackplot(x, high, medium, low, labels=["High", "Medium", "Low"], alpha=0.85)
    boundaries = []
    last = None
    for row in rows:
        corruption = row["corruption"]
        if last is not None and corruption != last:
            boundaries.append(int(row["global_batch"]))
        last = corruption
    for boundary in boundaries:
        plt.axvline(boundary, color="black", linewidth=0.4, alpha=0.35)
    plt.xlabel("Batch")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.title("routing_analysis route ratio over RRTTA adaptation stream")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_corruption_bar_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["corruption"] for row in rows]
    high = [float(row["mean_high_ratio"]) for row in rows]
    medium = [float(row["mean_medium_ratio"]) for row in rows]
    low = [float(row["mean_low_ratio"]) for row in rows]
    x = list(range(len(names)))
    plt.figure(figsize=(12, 5))
    plt.bar(x, high, label="High", alpha=0.85)
    bottom_medium = high
    plt.bar(x, medium, bottom=bottom_medium, label="Medium", alpha=0.85)
    bottom_low = [h + m for h, m in zip(high, medium)]
    plt.bar(x, low, bottom=bottom_low, label="Low", alpha=0.85)
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean ratio")
    plt.ylim(0, 1)
    plt.title("routing_analysis mean route ratio by corruption")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


@torch.no_grad()
def route_stats(model, xb, cfg):
    logits, _ = model.teacher_extractor(xb)
    route = route_by_reliability(
        logits,
        cfg.num_classes,
        cfg.reliability.reliability_high_threshold,
        cfg.reliability.reliability_low_threshold,
    )
    n = max(1, xb.shape[0])
    return {
        "confidence": float(route["confidence"].mean().item()),
        "margin": float(route["margin"].mean().item()),
        "entropy_score": float(route["entropy_score"].mean().item()),
        "reliability": float(route["reliability"].mean().item()),
        "high_ratio": float(route["high_mask"].float().mean().item()),
        "medium_ratio": float(route["medium_mask"].float().mean().item()),
        "low_ratio": float(route["low_mask"].float().mean().item()),
        "high_count": int(route["high_mask"].sum().item()),
        "medium_count": int(route["medium_mask"].sum().item()),
        "low_count": int(route["low_mask"].sum().item()),
        "batch_size": n,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for routing_analysis route ratio experiment.")
    set_seed(args.seed)
    cfg = build_cfg(args)
    run_id = args.run_id or f"{args.dataset}_routing_analysis_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}")

    student_model = load_model(
        cfg.arch,
        cfg.ckpt_dir,
        cfg.dataset,
        ThreatModel.corruptions,
    ).cuda()
    model = setup_rrtta(student_model, cfg, PrintLogger())
    if hasattr(model, "reset"):
        model.reset()

    history = []
    global_batch = 0
    for corruption_index, corruption in enumerate(args.corruptions, start=1):
        print(f"corruption {corruption_index}/{len(args.corruptions)} {corruption}")
        x_test, _ = load_corruption(args.dataset, args.num_ex, args.severity, args.data_dir, corruption)
        x_test = x_test.cuda()
        local_batch = 0
        for start in range(0, x_test.shape[0], args.batch_size):
            end = min(start + args.batch_size, x_test.shape[0])
            xb = x_test[start:end]
            global_batch += 1
            local_batch += 1
            stats = route_stats(model, xb, cfg)
            row = {
                "global_batch": global_batch,
                "batch": local_batch,
                "severity": args.severity,
                "corruption": corruption,
                **stats,
                "high_threshold": cfg.reliability.reliability_high_threshold,
                "low_threshold": cfg.reliability.reliability_low_threshold,
                "proto_valid_ratio": float((model.proto_count > 0).float().mean().item()),
                "proto_total_count": int(model.proto_count.sum().item()),
            }
            history.append(row)
            with torch.no_grad():
                _ = model(xb)
        del x_test
        torch.cuda.empty_cache()

    by_corruption = []
    grouped = defaultdict(list)
    for row in history:
        grouped[row["corruption"]].append(row)
    for corruption in args.corruptions:
        rows = grouped[corruption]
        if not rows:
            continue
        by_corruption.append(
            {
                "corruption": corruption,
                "num_batches": len(rows),
                "mean_confidence": mean(float(row["confidence"]) for row in rows),
                "mean_reliability": mean(float(row["reliability"]) for row in rows),
                "mean_high_ratio": mean(float(row["high_ratio"]) for row in rows),
                "mean_medium_ratio": mean(float(row["medium_ratio"]) for row in rows),
                "mean_low_ratio": mean(float(row["low_ratio"]) for row in rows),
            }
        )

    summary = [
        {
            "dataset": args.dataset,
            "method": "RRTTA",
            "run_id": run_id,
            "num_batches": len(history),
            "num_corruptions": len(by_corruption),
            "mean_confidence": mean(float(row["confidence"]) for row in history),
            "mean_reliability": mean(float(row["reliability"]) for row in history),
            "mean_high_ratio": mean(float(row["high_ratio"]) for row in history),
            "mean_medium_ratio": mean(float(row["medium_ratio"]) for row in history),
            "mean_low_ratio": mean(float(row["low_ratio"]) for row in history),
            "output_dir": str(output_dir),
        }
    ]
    history_path = output_dir / f"route_history_{run_id}.csv"
    corruption_path = output_dir / f"route_ratio_by_corruption_{run_id}.csv"
    summary_path = output_dir / f"route_ratio_summary_{run_id}.csv"
    stack_path = output_dir / f"route_ratio_stack_{run_id}.png"
    bar_path = output_dir / f"route_ratio_by_corruption_{run_id}.png"
    write_csv(history_path, history)
    write_csv(corruption_path, by_corruption)
    write_csv(summary_path, summary)
    save_stack_plot(stack_path, history)
    save_corruption_bar_plot(bar_path, by_corruption)
    print(f"saved {history_path}")
    print(f"saved {corruption_path}")
    print(f"saved {summary_path}")
    print(f"saved {stack_path}")
    print(f"saved {bar_path}")


if __name__ == "__main__":
    main()
