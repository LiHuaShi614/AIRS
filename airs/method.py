from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import time
from typing import Dict, Iterator, Tuple

import torch

from .config import TIER_SPECS
from .dependencies import ensure_rrtta_importable
from .resources import resource_availability
from .scheduler import AIRSScheduler, ScheduleDecision

ensure_rrtta_importable()

from rrtta.method import RRTTA, route_by_reliability  # noqa: E402


@dataclass(frozen=True)
class BatchValueEstimate:
    value: float
    mean_reliability: float
    reliability: torch.Tensor


def batch_value_from_logits(
    logits: torch.Tensor,
    num_classes: int,
    high_threshold: float,
    low_threshold: float,
) -> BatchValueEstimate:
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must be a non-empty [batch, classes] tensor")
    route = route_by_reliability(logits, num_classes, high_threshold, low_threshold)
    reliability = route["reliability"].detach().clamp(0.0, 1.0)
    mean_reliability = reliability.mean()
    return BatchValueEstimate(
        value=float(mean_reliability.item()),
        mean_reliability=float(mean_reliability.item()),
        reliability=reliability,
    )


def _cuda_timed(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


class AIRSModel:
    def __init__(
        self,
        backend: RRTTA,
        scheduler: AIRSScheduler,
        resource_trace: str,
        total_batches: int,
    ):
        self.backend = backend
        self.scheduler = scheduler
        self.resource_trace = resource_trace
        self.total_batches = total_batches
        self.backend.proto_align_weight = 0.0
        self.backend.proto_dim = 0
        self.backend.proto_bank = self.backend.proto_bank.new_empty(0)
        self.backend.proto_count.zero_()
        self._full_settings = {
            "student_disagreement_weight": backend.student_disagreement_weight,
        }

    @contextmanager
    def _tier_settings(self, tier_name: str) -> Iterator[None]:
        spec = TIER_SPECS[tier_name]
        original = {
            "max_update_fraction": self.backend.max_update_fraction,
            "augment_times": self.backend.augment_times,
            "student_disagreement_weight": self.backend.student_disagreement_weight,
        }
        # The teacher and reliability router inspect the complete batch. The
        # tier fraction is applied only when RRTTA caps the final update set.
        self.backend.max_update_fraction = spec.max_update_fraction
        self.backend.augment_times = spec.augment_times
        self.backend.student_disagreement_weight = (
            self._full_settings["student_disagreement_weight"]
            if spec.student_disagreement_weight < 0
            else spec.student_disagreement_weight
        )
        try:
            yield
        finally:
            for name, value in original.items():
                setattr(self.backend, name, value)

    @staticmethod
    def _empty_adaptation_stats(batch_size: int) -> Dict[str, int]:
        return {
            "batch_size": batch_size,
            "adaptation_candidate_samples": 0,
            "high_samples": 0,
            "medium_samples": 0,
            "low_samples": 0,
            "selected_samples": 0,
            "teacher_forward_calls": 0,
            "teacher_examples": 0,
            "student_agreement_forward_calls": 0,
            "student_agreement_examples": 0,
            "student_adapt_forward_calls": 0,
            "student_adapt_examples": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }

    def _adapt(self, x: torch.Tensor) -> None:
        aggregate = self._empty_adaptation_stats(int(x.shape[0]))
        aggregate["batch_size"] = int(x.shape[0])
        for _ in range(self.backend.steps):
            self.backend.adapt_only(x)
            for key, value in self.backend.last_adaptation_stats.items():
                if key == "batch_size":
                    continue
                aggregate[key] = aggregate.get(key, 0) + int(value)
        self.backend.last_adaptation_stats = aggregate

    def process_batch(self, x: torch.Tensor, batch_index: int) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.backend.episodic:
            self.backend.reset()
        batch_seed = self.scheduler.cfg.seed * 1_000_003 + batch_index
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        batch_start = time.perf_counter()
        with torch.no_grad():
            logits, inference_gpu_ms = _cuda_timed(lambda: self.backend.predict(x))

        schedule_start = time.perf_counter()
        value_estimate = batch_value_from_logits(
            logits,
            self.backend.num_classes,
            self.backend.high_threshold,
            self.backend.low_threshold,
        )
        availability = resource_availability(self.resource_trace, batch_index, self.total_batches)
        decision: ScheduleDecision = self.scheduler.select(batch_index, value_estimate.value, availability)
        schedule_wall_ms = (time.perf_counter() - schedule_start) * 1000.0

        adaptation_gpu_ms = 0.0
        selected_augment_times = 0
        stats = self._empty_adaptation_stats(int(x.shape[0]))
        if decision.selected_tier != "inference":
            spec = TIER_SPECS[decision.selected_tier]
            selected_augment_times = spec.augment_times
            with self._tier_settings(decision.selected_tier):
                _, adaptation_gpu_ms = _cuda_timed(lambda: self._adapt(x))
            stats.update(self.backend.last_adaptation_stats)
            stats["batch_size"] = int(x.shape[0])
            stats["adaptation_candidate_samples"] = int(x.shape[0])

        batch_wall_ms = (time.perf_counter() - batch_start) * 1000.0
        record: Dict[str, object] = {
            **asdict(decision),
            "resource_trace": self.resource_trace,
            "mean_reliability": value_estimate.mean_reliability,
            "selected_augment_times": selected_augment_times,
            "inference_gpu_ms": inference_gpu_ms,
            "adaptation_gpu_ms": adaptation_gpu_ms,
            "total_gpu_ms": inference_gpu_ms + adaptation_gpu_ms,
            "schedule_wall_ms": schedule_wall_ms,
            "batch_wall_ms": batch_wall_ms,
            **stats,
        }
        return logits, record
