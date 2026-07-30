from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from airs.config import SchedulerConfig
from airs.method import AIRSModel, batch_value_from_logits
from airs.scheduler import AIRSScheduler


class BatchValueTest(unittest.TestCase):
    def test_value_is_mean_reliability(self):
        confident = torch.tensor([[8.0, 0.0, -1.0]] * 10)
        uniform = torch.zeros(10, 3)
        mixed = torch.cat((confident[:5], uniform[:5]), dim=0)

        confident_value = batch_value_from_logits(confident, 3, 0.4, 0.1)
        uniform_value = batch_value_from_logits(uniform, 3, 0.4, 0.1)
        mixed_value = batch_value_from_logits(mixed, 3, 0.4, 0.1)

        self.assertGreater(confident_value.value, mixed_value.value)
        self.assertGreater(mixed_value.value, uniform_value.value)
        self.assertEqual(uniform_value.value, 0.0)
        self.assertAlmostEqual(confident_value.value, confident_value.mean_reliability)
        self.assertAlmostEqual(mixed_value.value, mixed_value.mean_reliability)

    def test_value_rejects_empty_batches(self):
        with self.assertRaises(ValueError):
            batch_value_from_logits(torch.empty(0, 3), 3, 0.4, 0.1)


class TierSettingsTest(unittest.TestCase):
    def test_old_tier_limits_are_applied_inside_rrtta_and_prototypes_stay_disabled(self):
        backend = SimpleNamespace(
            max_update_fraction=0.9,
            augment_times=9,
            student_disagreement_weight=0.25,
            proto_align_weight=0.02,
            proto_dim=128,
            proto_bank=torch.ones(2, 2),
            proto_count=torch.ones(2),
        )
        model = AIRSModel(backend, scheduler=object(), resource_trace="steady", total_batches=1)

        self.assertEqual(backend.proto_align_weight, 0.0)
        self.assertEqual(backend.proto_dim, 0)
        self.assertEqual(backend.proto_bank.numel(), 0)
        self.assertTrue(torch.equal(backend.proto_count, torch.zeros(2)))

        with model._tier_settings("light"):
            self.assertEqual(backend.max_update_fraction, 0.25)
            self.assertEqual(backend.augment_times, 0)
        with model._tier_settings("medium"):
            self.assertEqual(backend.max_update_fraction, 0.5)
            self.assertEqual(backend.augment_times, 1)
        with model._tier_settings("full"):
            self.assertEqual(backend.max_update_fraction, 1.0)
            self.assertEqual(backend.augment_times, 4)

        self.assertEqual(backend.max_update_fraction, 0.9)
        self.assertEqual(backend.augment_times, 9)

    def test_light_teacher_receives_the_complete_batch_before_update_capping(self):
        observed = {}
        backend = SimpleNamespace(
            max_update_fraction=0.9,
            augment_times=9,
            student_disagreement_weight=0.25,
            proto_align_weight=0.02,
            proto_dim=128,
            proto_bank=torch.ones(2, 2),
            proto_count=torch.ones(2),
            episodic=False,
            steps=1,
            num_classes=2,
            high_threshold=0.4,
            low_threshold=0.1,
            last_adaptation_stats={},
        )
        backend.predict = lambda x: torch.tensor([[5.0, 0.0]] * x.shape[0])

        def adapt_only(x):
            observed["batch_size"] = int(x.shape[0])
            observed["fraction"] = backend.max_update_fraction
            observed["augment_times"] = backend.augment_times
            backend.last_adaptation_stats = {"batch_size": int(x.shape[0])}

        backend.adapt_only = adapt_only
        scheduler = AIRSScheduler(
            SchedulerConfig(
                policy="forced",
                fixed_tier="light",
                budget_ratio=1.0,
                credit_mode="unlimited",
                use_resource=False,
                total_batches=1,
            )
        )
        model = AIRSModel(backend, scheduler=scheduler, resource_trace="steady", total_batches=1)
        x = torch.zeros(8, 3, 4, 4)

        with patch("airs.method._cuda_timed", side_effect=lambda fn: (fn(), 0.0)):
            _, record = model.process_batch(x, batch_index=0)

        self.assertEqual(observed["batch_size"], 8)
        self.assertEqual(observed["fraction"], 0.25)
        self.assertEqual(observed["augment_times"], 0)
        self.assertEqual(record["adaptation_candidate_samples"], 8)


if __name__ == "__main__":
    unittest.main()
