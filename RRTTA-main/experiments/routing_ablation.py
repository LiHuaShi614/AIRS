#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
import rrtta.method as rrtta_module
from rrtta.method import setup_rrtta


VARIANTS = ["rrtta_full", "rrtta_no_routing"]


def parse_args():
    parser = argparse.ArgumentParser(description="Ablate RRTTA reliability routing.")
    add_common_args(parser, output_name="routing_ablation")
    add_rrtta_args(parser)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    return parser.parse_args()


def build_cfg(args):
    return apply_rrtta_args(config_from_args(args), args)


class PrintLogger:
    def info(self, msg, *args):
        print(msg % args if args else msg)
    def warning(self, msg, *args):
        print(msg % args if args else msg)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def route_no_routing_zero_weight(logits, num_classes, high_threshold, low_threshold):
    rel = rrtta_module.route_by_reliability_original(logits, num_classes, high_threshold, low_threshold)
    labels = rel["labels"]
    high = torch.zeros_like(labels, dtype=torch.bool)
    medium = torch.zeros_like(labels, dtype=torch.bool)
    low = torch.zeros_like(labels, dtype=torch.bool)
    return {
        **rel,
        "high_mask": high,
        "medium_mask": medium,
        "low_mask": low,
        "reliability": torch.zeros_like(rel["reliability"]),
    }


def run_variant(args, cfg, variant, output_dir: Path):
    original_route = rrtta_module.route_by_reliability
    rrtta_module.route_by_reliability_original = original_route
    if variant == "rrtta_no_routing":
        rrtta_module.route_by_reliability = route_no_routing_zero_weight
    try:
        student_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
        model = setup_rrtta(student_model, cfg, PrintLogger())
        model.reset()
        batch_rows = []
        correct = 0
        total = 0
        global_batch = 0
        for corruption in args.corruptions:
            print(f"{variant}: {corruption}")
            x_test, y_test = load_corruption(args.dataset, args.num_ex, args.severity, args.data_dir, corruption)
            x_test, y_test = x_test.cuda(), y_test.cuda()
            for start in range(0, x_test.shape[0], args.batch_size):
                end = min(start + args.batch_size, x_test.shape[0])
                xb, yb = x_test[start:end], y_test[start:end]
                global_batch += 1
                with torch.no_grad():
                    logits, _ = model.teacher_extractor(xb)
                    route = rrtta_module.route_by_reliability(
                        logits,
                        cfg.num_classes,
                        cfg.reliability.reliability_high_threshold,
                        cfg.reliability.reliability_low_threshold,
                    )
                    out = model(xb)
                pred = out.argmax(dim=1)
                batch_correct = int((pred == yb).sum().item())
                batch_total = int(yb.numel())
                correct += batch_correct
                total += batch_total
                batch_rows.append(
                    {
                        "variant": variant,
                        "global_batch": global_batch,
                        "corruption": corruption,
                        "batch_correct": batch_correct,
                        "batch_total": batch_total,
                        "batch_error": 1.0 - batch_correct / max(1, batch_total),
                        "high_ratio": float(route["high_mask"].float().mean().item()),
                        "medium_ratio": float(route["medium_mask"].float().mean().item()),
                        "low_ratio": float(route["low_mask"].float().mean().item()),
                        "low_loss_weight": float(cfg.rrtta.low_loss_weight),
                        "effective_update_weight": 0.0 if variant == "rrtta_no_routing" else float(cfg.rrtta.high_loss_weight),
                    }
                )
            del x_test, y_test
            torch.cuda.empty_cache()
        history_path = output_dir / variant / f"{variant}_history.csv"
        write_csv(history_path, batch_rows)
        return {
            "variant": variant,
            "num_batches": len(batch_rows),
            "num_examples": total,
            "error_percent": 100.0 * (1.0 - correct / max(1, total)),
            "mean_batch_error_percent": 100.0 * mean(float(row["batch_error"]) for row in batch_rows),
            "mean_high_ratio": mean(float(row["high_ratio"]) for row in batch_rows),
            "mean_medium_ratio": mean(float(row["medium_ratio"]) for row in batch_rows),
            "mean_low_ratio": mean(float(row["low_ratio"]) for row in batch_rows),
            "low_loss_weight": cfg.rrtta.low_loss_weight,
            "effective_update_weight": 0.0 if variant == "rrtta_no_routing" else cfg.rrtta.high_loss_weight,
            "history_path": str(history_path),
        }
    finally:
        rrtta_module.route_by_reliability = original_route


def save_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = [row["variant"] for row in rows]
    errors = [float(row["error_percent"]) for row in rows]
    high = [float(row["mean_high_ratio"]) for row in rows]
    medium = [float(row["mean_medium_ratio"]) for row in rows]
    low = [float(row["mean_low_ratio"]) for row in rows]
    x = list(range(len(variants)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, errors)
    axes[0].set_xticks(x, variants, rotation=20, ha="right")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Accuracy effect")
    axes[1].bar(x, high, label="High")
    axes[1].bar(x, medium, bottom=high, label="Medium")
    axes[1].bar(x, low, bottom=[h + m for h, m in zip(high, medium)], label="Low")
    axes[1].set_xticks(x, variants, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean route ratio")
    axes[1].set_title("Routing behavior")
    axes[1].legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for routing_ablation.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = build_cfg(args)
    run_id = args.run_id or f"{args.dataset}_routing_ablation_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(args, cfg, variant, output_dir) for variant in args.variants]
    summary_path = output_dir / f"no_routing_summary_{run_id}.csv"
    plot_path = output_dir / f"no_routing_comparison_{run_id}.png"
    write_csv(summary_path, rows)
    save_plot(plot_path, rows)
    print(f"saved {summary_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
