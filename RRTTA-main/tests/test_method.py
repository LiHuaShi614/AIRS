from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from rrtta.method import RRTTA, collect_params, configure_model, route_by_reliability
from rrtta.optim import setup_optimizer
from rrtta.runner import RunConfig, default_arch


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class ReliabilityRoutingTest(unittest.TestCase):
    def test_routes_form_a_partition(self):
        logits = torch.tensor([[8.0, 0.0, -1.0], [1.0, 0.9, 0.8], [0.0, 0.0, 0.0]])
        route = route_by_reliability(logits, 3, high_threshold=0.4, low_threshold=0.1)
        membership = (
            route["high_mask"].to(torch.int32)
            + route["medium_mask"].to(torch.int32)
            + route["low_mask"].to(torch.int32)
        )
        self.assertTrue(torch.equal(membership, torch.ones_like(membership)))

    def test_reliability_matches_definition(self):
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        route = route_by_reliability(logits, 3, high_threshold=0.4, low_threshold=0.1)
        probs = logits.softmax(dim=1)
        confidence = probs.max(dim=1).values
        margin = probs.topk(2, dim=1).values[:, 0] - probs.topk(2, dim=1).values[:, 1]
        entropy = -(probs * logits.log_softmax(dim=1)).sum(dim=1)
        expected = confidence * margin * (1.0 - entropy / math.log(3))
        self.assertTrue(torch.allclose(route["reliability"], expected))


class RRTTATest(unittest.TestCase):
    def test_forward_updates_high_reliability_prototypes(self):
        cfg = RunConfig(num_classes=3)
        cfg.reliability.reliability_high_threshold = 0.0
        cfg.reliability.reliability_low_threshold = 0.0
        cfg.rrtta.augment_times = 0
        cfg.rrtta.proto_warmup_batches = 0

        teacher = configure_model(TinyClassifier())
        student = configure_model(TinyClassifier())
        params, _ = collect_params(student)
        optimizer = setup_optimizer(params, cfg.optim)
        model = RRTTA(teacher, student, optimizer, cfg)

        output = model(torch.rand(4, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (4, 3))
        self.assertEqual(int(model.proto_count.sum().item()), 4)

    def test_adapt_only_skips_post_update_prediction_and_records_work(self):
        cfg = RunConfig(num_classes=3)
        cfg.reliability.reliability_high_threshold = 0.0
        cfg.reliability.reliability_low_threshold = 0.0
        cfg.rrtta.augment_times = 0
        cfg.rrtta.proto_warmup_batches = 0

        teacher = configure_model(TinyClassifier())
        student = configure_model(TinyClassifier())
        params, _ = collect_params(student)
        optimizer = setup_optimizer(params, cfg.optim)
        model = RRTTA(teacher, student, optimizer, cfg)

        output = model.adapt_only(torch.rand(4, 3, 8, 8))
        self.assertIsNone(output)
        self.assertEqual(model.last_adaptation_stats["optimizer_steps"], 1)
        self.assertGreater(model.last_adaptation_stats["backward_calls"], 0)
        self.assertEqual(model.last_adaptation_stats["teacher_forward_calls"], 1)

    def test_disabled_prototype_does_not_create_or_update_a_bank(self):
        cfg = RunConfig(num_classes=3)
        cfg.reliability.reliability_high_threshold = 0.0
        cfg.reliability.reliability_low_threshold = 0.0
        cfg.rrtta.augment_times = 0
        cfg.rrtta.proto_align_weight = 0.0

        teacher = configure_model(TinyClassifier())
        student = configure_model(TinyClassifier())
        params, _ = collect_params(student)
        optimizer = setup_optimizer(params, cfg.optim)
        model = RRTTA(teacher, student, optimizer, cfg)

        model.adapt_only(torch.rand(4, 3, 8, 8))
        self.assertEqual(model.proto_bank.numel(), 0)
        self.assertEqual(int(model.proto_count.sum().item()), 0)

    def test_dataset_default_models(self):
        self.assertEqual(default_arch("cifar10"), "Standard")
        self.assertEqual(default_arch("cifar100"), "Hendrycks2020AugMix_ResNeXt")


if __name__ == "__main__":
    unittest.main()
