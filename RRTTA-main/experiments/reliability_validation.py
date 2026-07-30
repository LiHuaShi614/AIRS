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
from rrtta.method import route_by_reliability, setup_rrtta


def parse_args():
    parser = argparse.ArgumentParser(description="Validate RRTTA reliability against pseudo-label accuracy.")
    add_common_args(parser, output_name="reliability_validation")
    add_rrtta_args(parser)
    parser.add_argument("--max-samples", type=int, default=50000)
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


def group_rows(rows):
    rows = sorted(rows, key=lambda row: float(row["reliability"]), reverse=True)
    n = len(rows)
    if n == 0:
        return []
    groups = [
        ("Top20%", 0, max(1, n // 5)),
        ("Middle20%", max(0, 2 * n // 5), max(1, 3 * n // 5)),
        ("Bottom20%", max(0, 4 * n // 5), n),
    ]
    out = []
    for name, start, end in groups:
        part = rows[start:end]
        out.append(
            {
                "reliability_group": name,
                "num_samples": len(part),
                "mean_reliability": mean(float(row["reliability"]) for row in part),
                "pseudo_label_accuracy": mean(float(row["pseudo_correct"]) for row in part),
                "mean_confidence": mean(float(row["confidence"]) for row in part),
                "mean_margin": mean(float(row["margin"]) for row in part),
                "mean_entropy_score": mean(float(row["entropy_score"]) for row in part),
            }
        )
    return out


def save_plot(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [row["reliability_group"] for row in rows]
    acc = [float(row["pseudo_label_accuracy"]) for row in rows]
    x = list(range(len(names)))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, acc, width=0.55, label="Pseudo-label accuracy")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Pseudo-label accuracy")
    plt.title("reliability_validation pseudo-label accuracy by reliability group")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


def save_scatter(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = sorted(rows, key=lambda row: float(row["reliability"]))
    bins = 20
    chunk = max(1, len(values) // bins)
    xs, ys = [], []
    for i in range(0, len(values), chunk):
        part = values[i:i + chunk]
        if not part:
            continue
        xs.append(mean(float(row["reliability"]) for row in part))
        ys.append(mean(float(row["pseudo_correct"]) for row in part))
    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Reliability")
    plt.ylabel("Pseudo-label accuracy")
    plt.ylim(0, 1)
    plt.title("reliability_validation reliability vs pseudo-label accuracy")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reliability_validation.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = build_cfg(args)
    run_id = args.run_id or f"{args.dataset}_reliability_validation_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}")
    student_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
    model = setup_rrtta(student_model, cfg, PrintLogger())
    model.reset()
    rows = []
    sample_stride = 1
    total_possible = max(1, args.num_ex * len(args.corruptions))
    if args.max_samples > 0 and total_possible > args.max_samples:
        sample_stride = max(1, total_possible // args.max_samples)
    sample_index = 0
    for corruption in args.corruptions:
        print(f"corruption {corruption}")
        x_test, y_test = load_corruption(args.dataset, args.num_ex, args.severity, args.data_dir, corruption)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        for start in range(0, x_test.shape[0], args.batch_size):
            xb, yb = x_test[start:start + args.batch_size], y_test[start:start + args.batch_size]
            with torch.no_grad():
                logits, _ = model.teacher_extractor(xb)
                route = route_by_reliability(
                    logits,
                    cfg.num_classes,
                    cfg.reliability.reliability_high_threshold,
                    cfg.reliability.reliability_low_threshold,
                )
                labels = route["labels"]
                for i in range(xb.shape[0]):
                    sample_index += 1
                    if sample_index % sample_stride != 0:
                        continue
                    rows.append(
                        {
                            "corruption": corruption,
                            "severity": args.severity,
                            "label": int(yb[i].item()),
                            "pseudo_label": int(labels[i].item()),
                            "pseudo_correct": int(labels[i].item() == yb[i].item()),
                            "confidence": float(route["confidence"][i].item()),
                            "margin": float(route["margin"][i].item()),
                            "entropy_score": float(route["entropy_score"][i].item()),
                            "reliability": float(route["reliability"][i].item()),
                        }
                    )
                _ = model(xb)
        del x_test, y_test
        torch.cuda.empty_cache()
    summary = group_rows(rows)
    sample_path = output_dir / f"reliability_samples_{run_id}.csv"
    summary_path = output_dir / f"reliability_group_accuracy_{run_id}.csv"
    bar_path = output_dir / f"reliability_group_accuracy_{run_id}.png"
    curve_path = output_dir / f"reliability_accuracy_curve_{run_id}.png"
    write_csv(sample_path, rows)
    write_csv(summary_path, summary)
    save_plot(bar_path, summary)
    save_scatter(curve_path, rows)
    print(f"saved {sample_path}")
    print(f"saved {summary_path}")
    print(f"saved {bar_path}")
    print(f"saved {curve_path}")


if __name__ == "__main__":
    main()
