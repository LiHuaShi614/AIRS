from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from .config import SchedulerConfig, TIER_NAMES, TIER_SPECS
from .dependencies import ensure_rrtta_importable
from .method import AIRSModel
from .scheduler import AIRSScheduler

ensure_rrtta_importable()

from rrtta.method import setup_rrtta  # noqa: E402
from rrtta.runner import RunConfig, load_corruption_dataset, set_seed, setup_logger  # noqa: E402


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _safe_percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _aggregate(
    batch_rows: List[Dict[str, object]],
    cfg: RunConfig,
    scheduler_cfg: SchedulerConfig,
    resource_trace: str,
    run_id: str,
    total_time: float,
    peak_gpu_mb: float,
) -> Dict[str, object]:
    total_examples = sum(int(row["num_examples"]) for row in batch_rows)
    total_correct = sum(int(row["correct"]) for row in batch_rows)
    latencies = [float(row["batch_wall_ms"]) for row in batch_rows]
    gpu_times = [float(row["total_gpu_ms"]) for row in batch_rows]
    allocated = scheduler_cfg.budget_ratio * len(batch_rows)
    spent = sum(float(row["planned_cost"]) for row in batch_rows)
    final_balance = float(batch_rows[-1]["credit_after"]) if batch_rows else 0.0
    if not math.isfinite(final_balance):
        final_balance = None
    summary: Dict[str, object] = {
        "run_id": run_id,
        "experiment": getattr(scheduler_cfg, "experiment", ""),
        "config_id": getattr(scheduler_cfg, "config_id", ""),
        "method": "AIRS",
        "policy": scheduler_cfg.policy,
        "dataset": cfg.dataset,
        "arch": cfg.arch,
        "severity": cfg.severity,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "num_examples": total_examples,
        "num_batches": len(batch_rows),
        "num_corruptions": len(cfg.corruptions),
        "resource_trace": resource_trace,
        "budget_percent": scheduler_cfg.budget_ratio * 100.0,
        "credit_mode": scheduler_cfg.credit_mode,
        "pace_budget": scheduler_cfg.pace_budget,
        "accuracy_percent": 100.0 * total_correct / max(1, total_examples),
        "error_percent": 100.0 * (1.0 - total_correct / max(1, total_examples)),
        "mean_batch_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_batch_latency_ms": _safe_percentile(latencies, 95.0),
        "mean_gpu_time_ms": float(np.mean(gpu_times)) if gpu_times else 0.0,
        "total_gpu_time_seconds": sum(gpu_times) / 1000.0,
        "total_wall_time_seconds": total_time,
        "peak_gpu_mb": peak_gpu_mb,
        "planned_budget_units": allocated,
        "used_budget_units": spent,
        "budget_utilization_percent": 100.0 * spent / max(1e-9, allocated),
        "budget_balance_units": final_balance,
        "resource_violations": sum(bool(row["resource_violation"]) for row in batch_rows),
        "teacher_forward_calls": sum(int(row["teacher_forward_calls"]) for row in batch_rows),
        "teacher_examples": sum(int(row["teacher_examples"]) for row in batch_rows),
        "backward_calls": sum(int(row["backward_calls"]) for row in batch_rows),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in batch_rows),
        "selected_samples": sum(int(row["selected_samples"]) for row in batch_rows),
        "adaptation_candidate_samples": sum(
            int(row["adaptation_candidate_samples"]) for row in batch_rows
        ),
        "mean_batch_value": float(np.mean([float(row["batch_value"]) for row in batch_rows])) if batch_rows else 0.0,
        "mean_reliability": float(
            np.mean([float(row["mean_reliability"]) for row in batch_rows])
        ) if batch_rows else 0.0,
    }
    for tier in TIER_NAMES:
        count = sum(row["selected_tier"] == tier for row in batch_rows)
        summary[f"{tier}_batches"] = count
        summary[f"{tier}_percent"] = 100.0 * count / max(1, len(batch_rows))
    return summary


def _per_corruption_rows(batch_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in batch_rows:
        grouped.setdefault(str(row["corruption"]), []).append(row)
    result = []
    for corruption, rows in grouped.items():
        examples = sum(int(row["num_examples"]) for row in rows)
        correct = sum(int(row["correct"]) for row in rows)
        result.append(
            {
                "corruption": corruption,
                "num_examples": examples,
                "accuracy_percent": 100.0 * correct / max(1, examples),
                "mean_batch_latency_ms": float(np.mean([float(row["batch_wall_ms"]) for row in rows])),
                "p95_batch_latency_ms": _safe_percentile([float(row["batch_wall_ms"]) for row in rows], 95.0),
                "used_budget_units": sum(float(row["planned_cost"]) for row in rows),
                "adaptation_candidate_samples": sum(
                    int(row["adaptation_candidate_samples"]) for row in rows
                ),
                "teacher_forward_calls": sum(int(row["teacher_forward_calls"]) for row in rows),
                "backward_calls": sum(int(row["backward_calls"]) for row in rows),
            }
        )
    return result


def evaluate_airs(
    cfg: RunConfig,
    scheduler_cfg: SchedulerConfig,
    resource_trace: str,
    run_dir: Path,
    run_id: str,
) -> Tuple[Path, Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AIRS experiments")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    logger: logging.Logger = setup_logger(run_dir, "AIRS", run_id)

    total_batches = len(cfg.corruptions) * math.ceil(cfg.num_ex / cfg.batch_size)
    scheduler_cfg.total_batches = total_batches
    scheduler_cfg.validate()
    logger.info(
        "AIRS run=%s policy=%s budget=%.0f%% credit=%s pace=%s trace=%s batches=%s",
        run_id,
        scheduler_cfg.policy,
        scheduler_cfg.budget_ratio * 100.0,
        scheduler_cfg.credit_mode,
        scheduler_cfg.pace_budget,
        resource_trace,
        total_batches,
    )
    logger.info("online protocol: predict current batch, then adapt for future batches")
    tier_summary = " ".join(
        f"{name}={TIER_SPECS[name].max_update_fraction:.0%}/{TIER_SPECS[name].augment_times}aug"
        for name in TIER_NAMES
    )
    logger.info("AIRS tiers: %s prototype=disabled", tier_summary)

    torch.cuda.reset_peak_memory_stats()
    base_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
    backend = setup_rrtta(base_model, cfg, logger)
    backend.reset()
    scheduler = AIRSScheduler(scheduler_cfg)
    model = AIRSModel(backend, scheduler, resource_trace, total_batches)

    batch_rows: List[Dict[str, object]] = []
    total_start = time.perf_counter()
    global_batch = 0
    for corruption_index, corruption in enumerate(cfg.corruptions):
        logger.info("corruption %s/%s %s", corruption_index + 1, len(cfg.corruptions), corruption)
        x_test, y_test = load_corruption_dataset(cfg, corruption)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        for local_batch, start in enumerate(range(0, x_test.shape[0], cfg.batch_size)):
            end = min(start + cfg.batch_size, x_test.shape[0])
            x_batch = x_test[start:end]
            y_batch = y_test[start:end]
            logits, record = model.process_batch(x_batch, global_batch)
            correct = int((logits.argmax(dim=1) == y_batch).sum().item())
            record.update(
                {
                    "run_id": run_id,
                    "policy": scheduler_cfg.policy,
                    "budget_percent": scheduler_cfg.budget_ratio * 100.0,
                    "credit_mode": scheduler_cfg.credit_mode,
                    "pace_budget": scheduler_cfg.pace_budget,
                    "corruption": corruption,
                    "corruption_index": corruption_index,
                    "local_batch_index": local_batch,
                    "num_examples": int(y_batch.numel()),
                    "correct": correct,
                    "batch_accuracy_percent": 100.0 * correct / max(1, int(y_batch.numel())),
                }
            )
            batch_rows.append(record)
            global_batch += 1
        del x_test, y_test

    total_time = time.perf_counter() - total_start
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    summary = _aggregate(
        batch_rows,
        cfg,
        scheduler_cfg,
        resource_trace,
        run_id,
        total_time,
        peak_gpu_mb,
    )
    _write_csv(run_dir / "batch_metrics.csv", batch_rows)
    _write_csv(run_dir / "per_corruption.csv", _per_corruption_rows(batch_rows))
    _write_csv(run_dir / "summary.csv", [summary])
    _write_json(run_dir / "completed.json", summary)
    logger.info(
        "complete accuracy=%.3f%% latency=%.2fms p95=%.2fms budget=%.1f/%.1f",
        summary["accuracy_percent"],
        summary["mean_batch_latency_ms"],
        summary["p95_batch_latency_ms"],
        summary["used_budget_units"],
        summary["planned_budget_units"],
    )
    return run_dir, summary
