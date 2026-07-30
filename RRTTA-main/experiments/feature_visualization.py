#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from rrtta.runner import (
    DEFAULT_ARCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CKPT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEVERITY,
    apply_rrtta_args,
    config_from_args,
    load_corruption,
)
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from rrtta.method import FeatureLogitExtractor, setup_rrtta


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RRTTA features before and after adaptation.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / "feature_visualization"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source-rrtta-run", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-ex", type=int, default=1000)
    parser.add_argument("--severity", type=int, default=DEFAULT_SEVERITY)
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument(
        "--plot-labels",
        default="",
        help="comma-separated class ids to visualize, e.g. 0,1,2,...,9",
    )
    parser.add_argument("--per-class-points", type=int, default=50)
    parser.add_argument("--after-mode", choices=["final", "stream"], default="final")
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--teacher-arch", default="")
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


def latest_rrtta_run(dataset: str) -> Path | None:
    rrtta_root = Path(DEFAULT_OUTPUT_ROOT) / "rrtta" / dataset
    if not rrtta_root.exists():
        return None
    candidates = [
        path for path in rrtta_root.iterdir()
        if path.is_dir() and not path.name.startswith("smoke")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_cfg(args):
    args_for_cfg = argparse.Namespace(
        dataset=args.dataset,
        arch=args.arch,
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        batch_size=args.batch_size,
        num_ex=args.num_ex,
        severity=args.severity,
        corruptions=[args.corruption],
        seed=args.seed,
        lr=args.lr,
        steps=args.steps,
        episodic=args.episodic,
    )
    return apply_rrtta_args(config_from_args(args_for_cfg), args)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_features(model, x: torch.Tensor, batch_size: int):
    extractor = FeatureLogitExtractor(model)
    features = []
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            logits, feat = extractor(xb)
            features.append(feat.detach().cpu())
            preds.append(logits.argmax(dim=1).detach().cpu())
    extractor._hook.remove()
    return torch.cat(features, dim=0), torch.cat(preds, dim=0)


def subsample(features: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, max_points: int):
    if features.shape[0] <= max_points:
        return features, labels, preds
    generator = torch.Generator().manual_seed(1)
    index = torch.randperm(features.shape[0], generator=generator)[:max_points]
    return features[index], labels[index], preds[index]


def parse_plot_labels(value: str) -> set[int] | None:
    if not value.strip():
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def filter_classes(
    features: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    keep_labels: set[int] | None,
):
    if keep_labels is None:
        return features, labels, preds
    keep = torch.tensor(sorted(keep_labels), dtype=labels.dtype)
    mask = torch.isin(labels.cpu(), keep)
    return features[mask], labels[mask], preds[mask]


def class_balanced_subsample(
    features: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    per_class_points: int,
):
    if per_class_points <= 0:
        return features, labels, preds
    generator = torch.Generator().manual_seed(1)
    selected = []
    for cls in sorted(labels.unique().tolist()):
        idx = torch.where(labels == cls)[0]
        if idx.numel() > per_class_points:
            perm = torch.randperm(idx.numel(), generator=generator)[:per_class_points]
            idx = idx[perm]
        selected.append(idx)
    if not selected:
        return features, labels, preds
    index = torch.cat(selected)
    return features[index], labels[index], preds[index]


def embed(features: torch.Tensor):
    values = features.float().numpy()
    try:
        from sklearn.manifold import TSNE
        perplexity = max(5, min(30, (values.shape[0] - 1) // 3))
        points = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=1,
        ).fit_transform(values)
        return points, "tsne"
    except Exception as exc:
        print(f"t-SNE unavailable, falling back to PCA: {exc}")
        centered = torch.tensor(values - values.mean(axis=0, keepdims=True))
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        return (centered @ vt[:2].t()).numpy(), "pca"


def save_embedding_csv(path: Path, points, labels, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "group"])
        for point, label, group in zip(points, labels, groups):
            writer.writerow([float(point[0]), float(point[1]), int(label), group])


def save_plot(path: Path, points, labels, groups, method: str, title: str):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"before": "o", "after": "^", "prototype": "X"}
    sizes = {"before": 12, "after": 12, "prototype": 80}
    plt.figure(figsize=(9, 7))
    for group in ["before", "after", "prototype"]:
        idx = [i for i, value in enumerate(groups) if value == group]
        if not idx:
            continue
        plt.scatter(
            points[idx, 0],
            points[idx, 1],
            c=[labels[i] for i in idx],
            cmap="tab20",
            s=sizes[group],
            marker=markers[group],
            alpha=0.75,
            label=group,
            linewidths=0.2,
            edgecolors="black" if group == "prototype" else "none",
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


class PrintLogger:
    def info(self, msg, *args):
        print(msg % args if args else msg)

    def warning(self, msg, *args):
        print(msg % args if args else msg)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature visualization.")
    set_seed(args.seed)

    source_run = Path(args.source_rrtta_run) if args.source_rrtta_run else latest_rrtta_run(args.dataset)
    run_id = args.run_id or f"{args.dataset}_feature_visualization_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / args.dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source_rrtta_run={source_run if source_run else 'not_found'}")
    print(f"output_dir={output_dir}")

    x_test, y_test = load_corruption(args.dataset, args.num_ex, args.severity, args.data_dir, args.corruption)
    x_test = x_test.cuda()
    y_test = y_test.cuda()

    cfg = build_cfg(args)
    base_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
    base_model.eval()
    before_features, before_preds = collect_features(base_model, x_test, args.batch_size)
    before_labels = y_test.detach().cpu()
    del base_model
    torch.cuda.empty_cache()

    student_model = load_model(
        cfg.arch,
        cfg.ckpt_dir,
        cfg.dataset,
        ThreatModel.corruptions,
    ).cuda()
    rrtta_model = setup_rrtta(student_model, cfg, PrintLogger())
    if hasattr(rrtta_model, "reset"):
        rrtta_model.reset()

    stream_after_features = []
    stream_after_preds = []
    stream_after_labels = []
    for start in range(0, x_test.shape[0], args.batch_size):
        xb = x_test[start:start + args.batch_size]
        yb = y_test[start:start + args.batch_size]
        logits = rrtta_model(xb)
        if args.after_mode == "stream":
            with torch.no_grad():
                _, feat = rrtta_model.student_extractor(xb)
            stream_after_features.append(feat.detach().cpu())
            stream_after_preds.append(logits.argmax(dim=1).detach().cpu())
            stream_after_labels.append(yb.detach().cpu())

    if args.after_mode == "final":
        after_features, after_preds = collect_features(rrtta_model.student, x_test, args.batch_size)
        after_labels = y_test.detach().cpu()
    else:
        after_features = torch.cat(stream_after_features, dim=0)
        after_preds = torch.cat(stream_after_preds, dim=0)
        after_labels = torch.cat(stream_after_labels, dim=0)

    plot_labels = parse_plot_labels(args.plot_labels)
    before_features, before_labels, before_preds = filter_classes(
        before_features, before_labels, before_preds, plot_labels
    )
    after_features, after_labels, after_preds = filter_classes(
        after_features, after_labels, after_preds, plot_labels
    )
    if plot_labels is None:
        half_points = max(1, args.max_points // 2)
        before_features, before_labels, before_preds = subsample(before_features, before_labels, before_preds, half_points)
        after_features, after_labels, after_preds = subsample(after_features, after_labels, after_preds, half_points)
    else:
        before_features, before_labels, before_preds = class_balanced_subsample(
            before_features, before_labels, before_preds, args.per_class_points
        )
        after_features, after_labels, after_preds = class_balanced_subsample(
            after_features, after_labels, after_preds, args.per_class_points
        )

    proto_features = torch.empty(0, before_features.shape[1])
    proto_labels = torch.empty(0, dtype=torch.long)
    proto_preds = torch.empty(0, dtype=torch.long)
    if rrtta_model.proto_bank.numel() > 0:
        valid_proto = (rrtta_model.proto_count > 0).detach().cpu()
        proto_features = rrtta_model.proto_bank.detach().cpu()[valid_proto]
        proto_labels = torch.arange(rrtta_model.num_classes)[valid_proto]
        proto_preds = proto_labels.clone()
    if plot_labels is not None and proto_features.numel() > 0:
        keep = torch.tensor(sorted(plot_labels), dtype=proto_labels.dtype)
        proto_mask = torch.isin(proto_labels.cpu(), keep)
        proto_features = proto_features[proto_mask]
        proto_labels = proto_labels[proto_mask]
        proto_preds = proto_preds[proto_mask]

    all_features = torch.cat([before_features, after_features, proto_features], dim=0)
    if not args.no_normalize_features:
        all_features = F.normalize(all_features, dim=1)
    all_labels = torch.cat([before_labels, after_labels, proto_labels], dim=0)
    groups = (
        ["before"] * before_features.shape[0]
        + ["after"] * after_features.shape[0]
        + ["prototype"] * proto_features.shape[0]
    )

    points, method = embed(all_features)
    csv_path = output_dir / f"{method}_embedding_{run_id}.csv"
    png_path = output_dir / f"{method}_embedding_{run_id}.png"
    save_embedding_csv(csv_path, points, all_labels.tolist(), groups)
    save_plot(
        png_path,
        points,
        all_labels.tolist(),
        groups,
        method,
        f"RRTTA feature visualization ({method})\n{args.dataset} {args.corruption} severity {args.severity}",
    )

    metadata_path = output_dir / f"feature_visualization_{run_id}_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"source_rrtta_run={source_run if source_run else 'not_found'}\n")
        f.write(f"corruption={args.corruption}\n")
        f.write(f"severity={args.severity}\n")
        f.write(f"num_ex={args.num_ex}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"max_points={args.max_points}\n")
        f.write(f"plot_labels={args.plot_labels}\n")
        f.write(f"per_class_points={args.per_class_points}\n")
        f.write(f"after_mode={args.after_mode}\n")
        f.write(f"normalize_features={not args.no_normalize_features}\n")
        f.write(f"num_prototypes={int(proto_features.shape[0])}\n")
        f.write(f"embedding={method}\n")
        f.write(f"csv={csv_path}\n")
        f.write(f"png={png_path}\n")
    print(f"saved {csv_path}")
    print(f"saved {png_path}")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()
