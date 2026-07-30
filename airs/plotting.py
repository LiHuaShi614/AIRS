from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


COLORS = {
    "inference": "#B3B3B3",
    "light": "#009E73",
    "medium": "#E69F00",
    "full": "#D55E00",
    "airs": "#0072B2",
    "baseline": "#6F6F6F",
    "accent": "#CC79A7",
    "resource": "#D55E00",
}

TRACE_COLORS = {
    "steady": "#0072B2",
    "sudden_drop": "#D55E00",
    "recovery": "#009E73",
    "cyclic": "#CC79A7",
}

SCATTER_LABEL_OFFSETS = {
    "AIRS": (5, 5),
    "Greedy": (5, 5),
    "Periodic": (5, 11),
    "Random": (5, -11),
}


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def _save(fig, base: Path) -> List[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    tiff_path = base.with_suffix(".tiff")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return [svg_path, pdf_path, png_path, tiff_path]


def _labels(rows: List[Dict[str, object]]) -> List[str]:
    return [str(row.get("label") or row.get("config_id") or row.get("policy")) for row in rows]


def _values(rows: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)


def _bar(ax, labels, values, colors, ylabel: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)


def _accuracy_points(ax, labels, values, colors, connect: bool = False) -> None:
    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels))
    if connect:
        ax.plot(x, values, color="#8A8A8A", linewidth=0.9, zorder=1)
    ax.scatter(x, values, c=colors, s=34, edgecolors="#333333", linewidths=0.5, zorder=2)
    for xi, value in zip(x, values):
        ax.annotate(
            f"{value:.2f}",
            (xi, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6,
        )
    spread = float(np.ptp(values)) if len(values) > 1 else 0.0
    span = max(1.5, spread * 1.7)
    center = float((values.min() + values.max()) / 2.0)
    lower = max(0.0, center - span / 2.0)
    upper = min(100.0, center + span / 2.0)
    if upper - lower < span:
        lower = max(0.0, upper - span)
    ax.set_ylim(lower, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Online accuracy (%)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)


def _tier_colors(rows: List[Dict[str, object]]) -> List[str]:
    result = []
    for row in rows:
        config = str(row.get("config_id", ""))
        match = next((tier for tier in COLORS if tier in config and tier in {"inference", "light", "medium", "full"}), None)
        result.append(COLORS.get(match or "airs", COLORS["airs"]))
    return result


def _relative_percent(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= 0.0:
        return np.zeros_like(values)
    return values / maximum * 100.0


def _plot_e1_cost_proxy(ax, rows: List[Dict[str, object]], labels, colors) -> None:
    batches = np.maximum(_values(rows, "num_batches"), 1.0)
    raw_cost = (
        _values(rows, "teacher_forward_calls")
        + _values(rows, "backward_calls")
    ) / batches
    relative_cost = _relative_percent(raw_cost)
    _bar(ax, labels, relative_cost, colors, "Relative cost proxy (%)")
    ax.set_ylim(0.0, max(112.0, float(relative_cost.max()) * 1.18))
    for bar, raw, percent in zip(ax.patches, raw_cost, relative_cost):
        ax.annotate(
            f"{raw:.2f}\n({percent:.1f}%)",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=5.5,
        )


def _plot_e1_resource_breakdown(ax, rows: List[Dict[str, object]], labels) -> None:
    metrics = (
        ("teacher_forward_calls", "Teacher calls", "#0072B2"),
        ("backward_calls", "Backward chunks", "#D55E00"),
        ("total_wall_time_seconds", "Runtime", "#009E73"),
        ("total_gpu_time_seconds", "GPU time", "#CC79A7"),
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.19
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    for offset, (key, metric_label, color) in zip(offsets, metrics):
        values = _relative_percent(_values(rows, key))
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            label=metric_label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Relative resource use (%)")
    ax.set_ylim(0.0, 108.0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=5.5, loc="upper left")


def _annotate_scheduler_points(ax, x, y, labels) -> None:
    for xi, yi, label in zip(x, y, labels):
        offset = SCATTER_LABEL_OFFSETS.get(str(label), (5, 5))
        ax.annotate(
            label,
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
            va="center",
        )


def _plot_e1(rows: List[Dict[str, object]], out_dir: Path) -> None:
    labels = _labels(rows)
    colors = _tier_colors(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4))
    _accuracy_points(axes[0, 0], labels, _values(rows, "accuracy_percent"), colors, connect=True)
    _bar(axes[0, 1], labels, _values(rows, "mean_batch_latency_ms"), colors, "Mean latency (ms/batch)")
    _plot_e1_cost_proxy(axes[1, 0], rows, labels, colors)
    _plot_e1_resource_breakdown(axes[1, 1], rows, labels)
    for label, ax in zip("abcd", axes.flat):
        _panel_label(ax, label)
    _save(fig, out_dir / "figures" / "E1_tier_resource_profile")


def _plot_e2(rows: List[Dict[str, object]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["budget_percent"]))
    budget = _values(rows, "budget_percent")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4))
    axes[0, 0].plot(budget, _values(rows, "accuracy_percent"), marker="o", color=COLORS["airs"], linewidth=1.8)
    axes[0, 0].set_ylabel("Online accuracy (%)")
    axes[0, 1].plot(budget, _values(rows, "mean_batch_latency_ms"), marker="o", color=COLORS["accent"], linewidth=1.8)
    axes[0, 1].set_ylabel("Mean latency (ms/batch)")
    axes[1, 0].plot(budget, _values(rows, "budget_utilization_percent"), marker="o", color=COLORS["resource"], linewidth=1.8)
    axes[1, 0].axhline(100.0, color="#B8B8B8", linewidth=0.8, linestyle="--")
    axes[1, 0].set_ylabel("Budget utilization (%)")

    bottom = np.zeros(len(rows))
    for tier in ("inference", "light", "medium", "full"):
        values = _values(rows, f"{tier}_percent")
        axes[1, 1].bar(budget, values, bottom=bottom, width=12, color=COLORS[tier], label=tier.title())
        bottom += values
    axes[1, 1].set_ylabel("Selected batches (%)")
    axes[1, 1].legend(ncol=2, fontsize=6)
    for label, ax in zip("abcd", axes.flat):
        ax.set_xlabel("Adaptation budget (%)")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        _panel_label(ax, label)
    _save(fig, out_dir / "figures" / "E2_budget_sensitivity")


def _read_batch_metrics(case_dir: Path) -> List[Dict[str, str]]:
    path = case_dir / "batch_metrics.csv"
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_e3(rows: List[Dict[str, object]], out_dir: Path) -> None:
    trace_order = {name: index for index, name in enumerate(("steady", "sudden_drop", "recovery", "cyclic"))}
    trace_rows = sorted(
        rows,
        key=lambda row: trace_order.get(str(row.get("resource_trace")), len(trace_order)),
    )
    fig = plt.figure(figsize=(7.1, 5.8))
    grid = fig.add_gridspec(max(1, len(trace_rows)), 2, width_ratios=(2.25, 1.0))
    tier_rank = {"inference": 0, "light": 1, "medium": 2, "full": 3}
    for index, row in enumerate(trace_rows):
        ax = fig.add_subplot(grid[index, 0])
        case_rows = _read_batch_metrics(out_dir / str(row["config_id"]))
        if case_rows:
            x = np.arange(len(case_rows))
            availability = np.asarray(
                [
                    3
                    if float(item["resource_availability"]) >= 1.0
                    else 2
                    if float(item["resource_availability"]) >= 0.5
                    else 1
                    if float(item["resource_availability"]) >= 0.2
                    else 0
                    for item in case_rows
                ]
            )
            selected = np.asarray([tier_rank[item["selected_tier"]] for item in case_rows])
            markevery = max(1, len(case_rows) // 18)
            ax.step(
                x,
                availability,
                where="post",
                color=COLORS["resource"],
                linewidth=1.3,
                linestyle="--",
                label="Maximum allowed tier",
            )
            ax.step(
                x,
                selected,
                where="post",
                color=COLORS["airs"],
                linewidth=1.1,
                marker="o",
                markersize=2.2,
                markevery=markevery,
                label="Selected tier",
            )
        ax.set_ylim(-0.15, 3.15)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["Inf.", "Light", "Medium", "Full"])
        ax.set_ylabel(str(row.get("label", row.get("resource_trace"))))
        if index == len(trace_rows) - 1:
            ax.set_xlabel("Online batch index")
        if index == 0:
            handles, legend_labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    legend_labels,
                    ncol=2,
                    fontsize=6,
                    loc="upper center",
                    bbox_to_anchor=(0.38, 0.995),
                )
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)

    bar_ax = fig.add_subplot(grid[:, 1])
    labels = _labels(trace_rows)
    trace_colors = [TRACE_COLORS.get(str(row.get("resource_trace")), COLORS["baseline"]) for row in trace_rows]
    _accuracy_points(bar_ax, labels, _values(trace_rows, "accuracy_percent"), trace_colors)
    _panel_label(fig.axes[0], "a")
    _panel_label(bar_ax, "b")
    _save(fig, out_dir / "figures" / "E3_dynamic_resource_response")


def _plot_e4(rows: List[Dict[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    labels = _labels(rows)
    x = _values(rows, "total_gpu_time_seconds")
    y = _values(rows, "accuracy_percent")
    colors = [COLORS["airs"] if label == "AIRS" else COLORS["baseline"] for label in labels]
    axes[0].scatter(x, y, c=colors, s=32, edgecolors="#333333", linewidths=0.5)
    _annotate_scheduler_points(axes[0], x, y, labels)
    axes[0].set_xlabel("Total GPU time (s)")
    axes[0].set_ylabel("Online accuracy (%)")
    _bar(axes[1], labels, _values(rows, "p95_batch_latency_ms"), colors, "P95 latency (ms/batch)")
    for label, ax in zip("ab", axes):
        ax.grid(color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        _panel_label(ax, label)
    _save(fig, out_dir / "figures" / "E4_scheduler_comparison")


def _plot_e5(rows: List[Dict[str, object]], out_dir: Path) -> None:
    labels = _labels(rows)
    colors = [COLORS["airs"]] + [COLORS["baseline"]] * max(0, len(rows) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    _accuracy_points(axes[0], labels, _values(rows, "accuracy_percent"), colors)
    _bar(axes[1], labels, _values(rows, "mean_batch_latency_ms"), colors, "Mean latency (ms/batch)")
    _bar(axes[2], labels, _values(rows, "resource_violations"), colors, "Resource violations")
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    _save(fig, out_dir / "figures" / "E5_joint_decision_ablation")


def _plot_e6(rows: List[Dict[str, object]], out_dir: Path) -> None:
    rows = [
        row
        for row in rows
        if str(row.get("credit_mode", "")).lower() != "global"
        and str(row.get("label", "")).lower() != "global budget"
    ]
    labels = _labels(rows)
    colors = [COLORS["airs"], COLORS["accent"]][: len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    _accuracy_points(axes[0], labels, _values(rows, "accuracy_percent"), colors)
    _bar(axes[1], labels, _values(rows, "budget_utilization_percent"), colors, "Budget utilization (%)")
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for tier in ("inference", "light", "medium", "full"):
        values = _values(rows, f"{tier}_percent")
        axes[2].bar(x, values, bottom=bottom, color=COLORS[tier], label=tier.title())
        bottom += values
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_ylabel("Selected batches (%)")
    axes[2].legend(fontsize=5, ncol=2)
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)
    _save(fig, out_dir / "figures" / "E6_credit_bank_ablation")


PLOTTERS = {
    "E1": _plot_e1,
    "E2": _plot_e2,
    "E3": _plot_e3,
    "E4": _plot_e4,
    "E5": _plot_e5,
    "E6": _plot_e6,
}


def plot_experiment(experiment: str, rows: List[Dict[str, object]], out_dir: Path) -> None:
    if not rows:
        return
    PLOTTERS[experiment](rows, out_dir)


def plot_overview(rows: List[Dict[str, object]], suite_root: Path) -> None:
    if not rows:
        return
    e2 = sorted(
        [row for row in rows if row.get("experiment") == "E2"],
        key=lambda row: float(row["budget_percent"]),
    )
    e4 = [row for row in rows if row.get("experiment") == "E4"]
    if not e2 and not e4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    if e2:
        axes[0].plot(
            _values(e2, "budget_percent"),
            _values(e2, "accuracy_percent"),
            marker="o",
            color=COLORS["airs"],
            linewidth=1.8,
        )
    axes[0].set_xlabel("Adaptation budget (%)")
    axes[0].set_ylabel("Online accuracy (%)")
    if e4:
        labels = _labels(e4)
        x = _values(e4, "total_gpu_time_seconds")
        y = _values(e4, "accuracy_percent")
        colors = [COLORS["airs"] if label == "AIRS" else COLORS["baseline"] for label in labels]
        axes[1].scatter(x, y, c=colors, s=32, edgecolors="#333333", linewidths=0.5)
        _annotate_scheduler_points(axes[1], x, y, labels)
    axes[1].set_xlabel("Total GPU time (s)")
    axes[1].set_ylabel("Online accuracy (%)")
    for label, ax in zip("ab", axes):
        ax.grid(color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        _panel_label(ax, label)
    _save(fig, suite_root / "figures" / "E0_AIRS_overview")
