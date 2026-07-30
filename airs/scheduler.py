from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Set

from .config import SchedulerConfig, TIER_NAMES, TIER_SPECS


@dataclass(frozen=True)
class ScheduleDecision:
    batch_index: int
    batch_value: float
    resource_availability: float
    desired_tier: str
    selected_tier: str
    planned_cost: float
    credit_before: float
    credit_after: float
    total_spent: float
    resource_violation: bool
    downgrade_reason: str


class AIRSScheduler:
    def __init__(self, cfg: SchedulerConfig):
        cfg.validate()
        self.cfg = cfg
        self.total_spent = 0.0
        self._credit = math.inf if cfg.credit_mode == "unlimited" else 0.0
        if cfg.credit_mode == "global":
            self._credit = cfg.budget_ratio * cfg.total_batches
        self._periodic_positions = self._even_positions(
            cfg.total_batches,
            round(cfg.budget_ratio * cfg.total_batches),
        )
        update_count = cfg.random_update_count
        if update_count is None:
            update_count = round(cfg.budget_ratio * cfg.total_batches)
        rng = random.Random(cfg.seed)
        self._random_positions = set(rng.sample(range(cfg.total_batches), min(cfg.total_batches, update_count)))

    @staticmethod
    def _even_positions(total: int, count: int) -> Set[int]:
        if count <= 0:
            return set()
        if count >= total:
            return set(range(total))
        return {min(total - 1, int((index + 0.5) * total / count)) for index in range(count)}

    def _tier_for_value(self, value: float) -> str:
        if value < self.cfg.value_light_threshold:
            return "inference"
        if value < self.cfg.value_medium_threshold:
            return "light"
        if value < self.cfg.value_full_threshold:
            return "medium"
        return "full"

    @staticmethod
    def _highest_affordable(limit: float) -> str:
        affordable = "inference"
        for name in TIER_NAMES:
            if TIER_SPECS[name].cost <= limit + 1e-9:
                affordable = name
        return affordable

    def _desired_tier(self, batch_index: int, batch_value: float) -> str:
        policy = self.cfg.policy
        if policy in {"forced", "static", "no_joint"}:
            return self.cfg.fixed_tier
        if policy == "periodic":
            return "full" if batch_index in self._periodic_positions else "inference"
        if policy == "random":
            return "full" if batch_index in self._random_positions else "inference"
        if policy in {"greedy", "resource_only"}:
            return "full"
        if policy in {"airs", "value_only"}:
            return self._tier_for_value(batch_value)
        raise ValueError(f"unsupported scheduling policy: {policy}")

    def _prepare_credit(self) -> None:
        if self.cfg.credit_mode == "bank":
            self._credit = min(self.cfg.credit_capacity, self._credit + self.cfg.budget_ratio)
        elif self.cfg.credit_mode == "fixed":
            self._credit = self.cfg.budget_ratio

    def select(self, batch_index: int, batch_value: float, availability: float) -> ScheduleDecision:
        self._prepare_credit()
        credit_before = self._credit
        desired = self._desired_tier(batch_index, batch_value)
        desired_rank = TIER_SPECS[desired].rank

        resource_cap = "full"
        if self.cfg.use_resource and self.cfg.policy != "value_only":
            resource_cap = self._highest_affordable(availability)
        budget_limit = self._credit
        if self.cfg.pace_budget:
            earned_to_date = self.cfg.budget_ratio * (batch_index + 1)
            unspent_earned_budget = max(0.0, earned_to_date - self.total_spent)
            budget_limit = min(budget_limit, unspent_earned_budget)
        budget_cap = self._highest_affordable(budget_limit)
        selected_rank = min(
            desired_rank,
            TIER_SPECS[resource_cap].rank,
            TIER_SPECS[budget_cap].rank,
        )
        selected = TIER_NAMES[selected_rank]
        cost = TIER_SPECS[selected].cost
        if not math.isinf(self._credit):
            self._credit = max(0.0, self._credit - cost)
        self.total_spent += cost

        reasons = []
        if TIER_SPECS[resource_cap].rank < desired_rank:
            reasons.append("resource")
        budget_comparison_rank = min(desired_rank, TIER_SPECS[resource_cap].rank)
        if TIER_SPECS[budget_cap].rank < budget_comparison_rank:
            reasons.append("budget")
        resource_violation = TIER_SPECS[selected].cost > availability + 1e-9
        return ScheduleDecision(
            batch_index=batch_index,
            batch_value=float(batch_value),
            resource_availability=float(availability),
            desired_tier=desired,
            selected_tier=selected,
            planned_cost=cost,
            credit_before=float(credit_before),
            credit_after=float(self._credit),
            total_spent=self.total_spent,
            resource_violation=resource_violation,
            downgrade_reason="+".join(reasons) if reasons else "none",
        )

    def state_dict(self) -> Dict[str, float]:
        return {"credit": float(self._credit), "total_spent": float(self.total_spent)}
