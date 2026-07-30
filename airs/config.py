from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TierSpec:
    name: str
    rank: int
    cost: float
    max_update_fraction: float
    augment_times: int
    student_disagreement_weight: float


TIER_SPECS: Dict[str, TierSpec] = {
    "inference": TierSpec(
        name="inference",
        rank=0,
        cost=0.0,
        max_update_fraction=0.0,
        augment_times=0,
        student_disagreement_weight=1.0,
    ),
    "light": TierSpec(
        name="light",
        rank=1,
        cost=0.2,
        max_update_fraction=0.25,
        augment_times=0,
        student_disagreement_weight=1.0,
    ),
    "medium": TierSpec(
        name="medium",
        rank=2,
        cost=0.5,
        max_update_fraction=0.50,
        augment_times=1,
        student_disagreement_weight=1.0,
    ),
    "full": TierSpec(
        name="full",
        rank=3,
        cost=1.0,
        max_update_fraction=1.0,
        augment_times=4,
        student_disagreement_weight=-1.0,
    ),
}

TIER_NAMES = tuple(name for name, _ in sorted(TIER_SPECS.items(), key=lambda item: item[1].rank))


@dataclass
class SchedulerConfig:
    policy: str = "airs"
    budget_ratio: float = 0.6
    credit_mode: str = "bank"
    credit_capacity: float = 4.0
    pace_budget: bool = False
    use_value: bool = True
    use_resource: bool = True
    value_light_threshold: float = 0.75
    value_medium_threshold: float = 0.82
    value_full_threshold: float = 0.88
    fixed_tier: str = "medium"
    seed: int = 1
    total_batches: int = 1
    random_update_count: Optional[int] = None

    def validate(self) -> None:
        if self.policy not in {
            "airs",
            "forced",
            "static",
            "greedy",
            "periodic",
            "random",
            "value_only",
            "resource_only",
            "no_joint",
        }:
            raise ValueError(f"unsupported scheduling policy: {self.policy}")
        if self.credit_mode not in {"unlimited", "bank", "fixed", "global"}:
            raise ValueError(f"unsupported credit mode: {self.credit_mode}")
        if self.fixed_tier not in TIER_SPECS:
            raise ValueError(f"unsupported fixed tier: {self.fixed_tier}")
        if not 0.0 <= self.budget_ratio <= 1.0:
            raise ValueError("budget_ratio must be in [0, 1]")
        if self.credit_capacity <= 0.0:
            raise ValueError("credit_capacity must be positive")
        if self.total_batches <= 0:
            raise ValueError("total_batches must be positive")
        thresholds = (
            self.value_light_threshold,
            self.value_medium_threshold,
            self.value_full_threshold,
        )
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError("value thresholds must be non-decreasing")
