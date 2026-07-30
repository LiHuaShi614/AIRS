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
from rrtta.method import setup_rrtta


def parse_args():
    parser = argparse.ArgumentParser(description="Measure the purity of RRTTA prototype updates.")
    add_common_args(parser, output_name="prototype_purity")
    add_rrtta_args(parser)
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


def save_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["global_batch"]) for row in rows]
    batch_purity = [float(row["batch_update_purity"]) for row in rows]
    cumulative = [float(row["cumulative_update_purity"]) for row in rows]
    high_ratio = [float(row["high_ratio"]) for row in rows]
    plt.figure(figsize=(11, 5))
    plt.plot(x, cumulative, label="Cumulative Update Accuracy", linewidth=2)
    plt.scatter(x, batch_purity, label="Batch Update Accuracy", s=8, alpha=0.55)
    plt.plot(x, high_ratio, label="High Ratio", linewidth=1.2, alpha=0.8)
    plt.ylim(0, 1)
    plt.xlabel("Batch")
    plt.ylabel("Proportion")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=600)
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".svg"))
    plt.close()


def save_corruption_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["corruption"] for row in rows]
    purity = [float(row["update_purity"]) for row in rows]
    high = [float(row["mean_high_ratio"]) for row in rows]
    x = list(range(len(names)))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], purity, width=width, label="Update Accuracy")
    plt.bar([i + width / 2 for i in x], high, width=width, label="Mean High Ratio")
    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=600)
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".svg"))
    plt.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prototype_purity.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = build_cfg(args)
    run_id = args.run_id or f"{args.dataset}_prototype_purity_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}")

    student_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
    model = setup_rrtta(student_model, cfg, PrintLogger())
    model.reset()

    rows = []
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    total_updates = 0
    total_correct = 0
    global_batch = 0
    for corruption in args.corruptions:
        print(f"corruption {corruption}")
        x_test, y_test = load_corruption(args.dataset, args.num_ex, args.severity, args.data_dir, corruption)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        local_batch = 0
        for start in range(0, x_test.shape[0], args.batch_size):
            end = min(start + args.batch_size, x_test.shape[0])
            xb, yb = x_test[start:end], y_test[start:end]
            global_batch += 1
            local_batch += 1
            labels, _, route, _ = model.make_teacher_targets(xb)
            high = route["high_mask"]
            update_total = int(high.sum().item())
            update_correct = int((labels[high] == yb[high]).sum().item()) if update_total > 0 else 0
            for cls in labels[high].detach().cpu().tolist():
                class_total[int(cls)] += 1
            for cls in labels[high][labels[high] == yb[high]].detach().cpu().tolist():
                class_correct[int(cls)] += 1
            total_updates += update_total
            total_correct += update_correct
            rows.append(
                {
                    "global_batch": global_batch,
                    "batch": local_batch,
                    "severity": args.severity,
                    "corruption": corruption,
                    "high_ratio": float(high.float().mean().item()),
                    "medium_ratio": float(route["medium_mask"].float().mean().item()),
                    "low_ratio": float(route["low_mask"].float().mean().item()),
                    "proto_update_total": update_total,
                    "proto_update_correct": update_correct,
                    "batch_update_purity": update_correct / max(1, update_total),
                    "cumulative_update_total": total_updates,
                    "cumulative_update_correct": total_correct,
                    "cumulative_update_purity": total_correct / max(1, total_updates),
                    "proto_valid_ratio": float((model.proto_count > 0).float().mean().item()),
                }
            )
            with torch.no_grad():
                _ = model(xb)
        del x_test, y_test
        torch.cuda.empty_cache()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["corruption"]].append(row)
    by_corruption = []
    for corruption in args.corruptions:
        part = grouped[corruption]
        total = sum(int(row["proto_update_total"]) for row in part)
        correct = sum(int(row["proto_update_correct"]) for row in part)
        by_corruption.append(
            {
                "corruption": corruption,
                "num_batches": len(part),
                "proto_update_total": total,
                "proto_update_correct": correct,
                "update_purity": correct / max(1, total),
                "mean_high_ratio": mean(float(row["high_ratio"]) for row in part),
            }
        )
    summary = [
        {
            "dataset": args.dataset,
            "method": "RRTTA",
            "run_id": run_id,
            "num_batches": len(rows),
            "prototype_update_total": total_updates,
            "prototype_update_correct": total_correct,
            "overall_update_purity": total_correct / max(1, total_updates),
            "mean_batch_update_purity": mean(float(row["batch_update_purity"]) for row in rows),
            "mean_high_ratio": mean(float(row["high_ratio"]) for row in rows),
            "output_dir": str(output_dir),
        }
    ]
    history_path = output_dir / f"prototype_purity_history_{run_id}.csv"
    corruption_path = output_dir / f"prototype_purity_by_corruption_{run_id}.csv"
    summary_path = output_dir / f"prototype_purity_summary_{run_id}.csv"
    plot_path = output_dir / f"prototype_purity_curve_{run_id}.png"
    corruption_plot_path = output_dir / f"prototype_purity_by_corruption_{run_id}.png"
    write_csv(history_path, rows)
    write_csv(corruption_path, by_corruption)
    write_csv(summary_path, summary)
    save_plot(plot_path, rows)
    save_corruption_plot(corruption_plot_path, by_corruption)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
