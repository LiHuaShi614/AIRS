from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from .config import SchedulerConfig, TIER_NAMES, TIER_SPECS
from .evaluation import evaluate_airs
from .plotting import plot_experiment, plot_overview


EXPERIMENT_IDS = ("E1", "E2", "E3", "E4", "E5", "E6")
PROTOCOL_VERSION = 7


@dataclass(frozen=True)
class ExperimentCase:
    experiment: str
    config_id: str
    label: str
    policy: str = "airs"
    budget_ratio: float = 0.6
    credit_mode: str = "bank"
    pace_budget: bool = False
    resource_trace: str = "steady"
    fixed_tier: str = "medium"
    use_value: bool = True
    use_resource: bool = True
    credit_capacity: float = 4.0

    def scheduler_config(self, seed: int) -> SchedulerConfig:
        cfg = SchedulerConfig(
            policy=self.policy,
            budget_ratio=self.budget_ratio,
            credit_mode=self.credit_mode,
            credit_capacity=self.credit_capacity,
            pace_budget=self.pace_budget,
            use_value=self.use_value,
            use_resource=self.use_resource,
            fixed_tier=self.fixed_tier,
            seed=seed,
        )
        cfg.experiment = self.experiment
        cfg.config_id = self.config_id
        return cfg


def experiment_cases(experiment: str) -> List[ExperimentCase]:
    experiment = experiment.upper()
    if experiment == "E1":
        return [
            ExperimentCase("E1", f"tier_{tier}", tier.title(), policy="forced", budget_ratio=1.0, credit_mode="unlimited", fixed_tier=tier, use_resource=False)
            for tier in ("inference", "light", "medium", "full")
        ]
    if experiment == "E2":
        return [
            ExperimentCase("E2", f"budget_{budget:03d}", f"{budget}%", budget_ratio=budget / 100.0)
            for budget in (20, 40, 60, 80, 100)
        ]
    if experiment == "E3":
        labels = {
            "steady": "Steady",
            "sudden_drop": "Sudden drop",
            "recovery": "Recovery",
            "cyclic": "Cyclic",
        }
        return [
            ExperimentCase("E3", f"trace_{trace}", labels[trace], budget_ratio=1.0, resource_trace=trace)
            for trace in labels
        ]
    if experiment == "E4":
        return [
            ExperimentCase(
                "E4",
                "policy_airs",
                "AIRS",
                budget_ratio=0.5,
                credit_mode="bank",
            ),
            ExperimentCase("E4", "policy_greedy", "Greedy", policy="greedy", budget_ratio=0.5, credit_mode="global"),
            ExperimentCase("E4", "policy_periodic", "Periodic", policy="periodic", budget_ratio=0.5, credit_mode="global"),
            ExperimentCase("E4", "policy_random", "Random", policy="random", budget_ratio=0.5, credit_mode="global"),
        ]
    if experiment == "E5":
        return [
            ExperimentCase("E5", "joint_airs", "Value + resource", budget_ratio=0.4, credit_mode="global", pace_budget=True, resource_trace="cyclic"),
            ExperimentCase("E5", "joint_value_only", "Value only", policy="value_only", budget_ratio=0.4, credit_mode="global", pace_budget=True, resource_trace="cyclic", use_resource=False),
            ExperimentCase("E5", "joint_resource_only", "Resource only", policy="resource_only", budget_ratio=0.4, credit_mode="global", pace_budget=True, resource_trace="cyclic", use_value=False),
            ExperimentCase("E5", "joint_none", "No joint decision", policy="no_joint", budget_ratio=0.4, credit_mode="global", pace_budget=True, resource_trace="cyclic", fixed_tier="medium", use_value=False, use_resource=False),
        ]
    if experiment == "E6":
        return [
            ExperimentCase("E6", "credit_bank", "Credit bank", budget_ratio=0.4, credit_mode="bank", resource_trace="cyclic"),
            ExperimentCase("E6", "credit_fixed", "Fixed quota", budget_ratio=0.4, credit_mode="fixed", resource_trace="cyclic"),
        ]
    raise ValueError(f"unsupported experiment: {experiment}")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_suite_id(smoke: bool, severity: int, seed: int) -> str:
    mode = "smoke" if smoke else "full"
    return f"airs_{mode}_v{PROTOCOL_VERSION}_s{severity}_seed{seed}"


def run_suite(
    cfg,
    experiments: Sequence[str] = EXPERIMENT_IDS,
    suite_id: str = "",
    smoke: bool = False,
    dry_run: bool = False,
    max_cases: int = 0,
) -> Path:
    selected = [item.upper() for item in experiments]
    invalid = [item for item in selected if item not in EXPERIMENT_IDS]
    if invalid:
        raise ValueError(f"unsupported experiments: {invalid}")
    if cfg.dataset != "cifar10" or cfg.severity != 5:
        raise ValueError("The AIRS experiment suite is fixed to CIFAR-10-C severity 5")
    if smoke:
        cfg = deepcopy(cfg)
        cfg.num_ex = min(cfg.num_ex, cfg.batch_size)
        cfg.corruptions = ["gaussian_noise"]
    suite_id = suite_id or default_suite_id(smoke, cfg.severity, cfg.seed)
    suite_root = Path(cfg.output_root) / suite_id
    suite_root.mkdir(parents=True, exist_ok=True)

    all_cases = [case for experiment in selected for case in experiment_cases(experiment)]
    if max_cases > 0:
        all_cases = all_cases[:max_cases]
    manifest = {
        "suite_id": suite_id,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "smoke": smoke,
        "dataset": cfg.dataset,
        "severity": cfg.severity,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "num_ex_per_corruption": cfg.num_ex,
        "corruptions": list(cfg.corruptions),
        "tier_specs": [asdict(TIER_SPECS[name]) for name in TIER_NAMES],
        "prototype_enabled": False,
        "experiments": selected,
        "cases": [asdict(case) for case in all_cases],
    }
    with (suite_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if dry_run:
        for case in all_cases:
            print(
                f"{case.experiment} {case.config_id}: policy={case.policy} "
                f"budget={case.budget_ratio:.0%} credit={case.credit_mode} "
                f"pace={case.pace_budget} trace={case.resource_trace}"
            )
        print(f"planned {len(all_cases)} runs under {suite_root}")
        return suite_root

    summaries_by_experiment: Dict[str, List[Dict[str, object]]] = {item: [] for item in selected}
    for index, case in enumerate(all_cases, start=1):
        case_dir = suite_root / case.experiment / case.config_id
        completed = case_dir / "completed.json"
        print(f"[{index}/{len(all_cases)}] {case.experiment} {case.label}")
        if completed.is_file():
            print(f"  skip completed: {case_dir}")
            summary = _load_json(completed)
        else:
            case_cfg = deepcopy(cfg)
            scheduler_cfg = case.scheduler_config(cfg.seed)
            run_id = f"{case.experiment.lower()}_{case.config_id}_s{cfg.severity}_seed{cfg.seed}"
            _, summary = evaluate_airs(
                case_cfg,
                scheduler_cfg,
                case.resource_trace,
                case_dir,
                run_id,
            )
            summary["label"] = case.label
            with completed.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        summary.setdefault("label", case.label)
        summaries_by_experiment[case.experiment].append(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined: List[Dict[str, object]] = []
    for experiment in selected:
        rows = summaries_by_experiment[experiment]
        if not rows:
            continue
        experiment_dir = suite_root / experiment
        _write_csv(experiment_dir / f"{experiment}_summary.csv", rows)
        plot_experiment(experiment, rows, experiment_dir)
        combined.extend(rows)
    _write_csv(suite_root / "E0_all_results.csv", combined)
    plot_overview(combined, suite_root)
    return suite_root
