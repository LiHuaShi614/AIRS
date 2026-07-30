from __future__ import annotations

import unittest

from airs.config import SchedulerConfig
from airs.resources import resource_availability
from airs.scheduler import AIRSScheduler


class AIRSSchedulerTest(unittest.TestCase):
    def test_resource_caps_a_high_value_batch(self):
        cfg = SchedulerConfig(
            policy="airs",
            credit_mode="unlimited",
            budget_ratio=1.0,
            total_batches=1,
        )
        decision = AIRSScheduler(cfg).select(0, batch_value=0.9, availability=0.2)
        self.assertEqual(decision.desired_tier, "full")
        self.assertEqual(decision.selected_tier, "light")
        self.assertFalse(decision.resource_violation)

    def test_credit_bank_carries_unused_budget(self):
        cfg = SchedulerConfig(
            policy="airs",
            credit_mode="bank",
            budget_ratio=0.2,
            credit_capacity=2.0,
            total_batches=5,
        )
        scheduler = AIRSScheduler(cfg)
        for index in range(4):
            decision = scheduler.select(index, batch_value=0.0, availability=1.0)
            self.assertEqual(decision.selected_tier, "inference")
        decision = scheduler.select(4, batch_value=0.9, availability=1.0)
        self.assertEqual(decision.selected_tier, "full")
        self.assertAlmostEqual(decision.total_spent, 1.0)

    def test_fixed_quota_expires_each_batch(self):
        cfg = SchedulerConfig(
            policy="airs",
            credit_mode="fixed",
            budget_ratio=0.4,
            total_batches=3,
        )
        scheduler = AIRSScheduler(cfg)
        decisions = [scheduler.select(index, batch_value=0.9, availability=1.0) for index in range(3)]
        self.assertEqual([item.selected_tier for item in decisions], ["light"] * 3)

    def test_value_only_records_resource_violation(self):
        cfg = SchedulerConfig(
            policy="value_only",
            credit_mode="unlimited",
            budget_ratio=1.0,
            use_resource=False,
            total_batches=1,
        )
        decision = AIRSScheduler(cfg).select(0, batch_value=0.9, availability=0.2)
        self.assertEqual(decision.selected_tier, "full")
        self.assertTrue(decision.resource_violation)

    def test_random_policy_uses_exact_update_count(self):
        cfg = SchedulerConfig(
            policy="random",
            credit_mode="global",
            budget_ratio=0.5,
            total_batches=10,
            seed=7,
        )
        scheduler = AIRSScheduler(cfg)
        decisions = [scheduler.select(index, batch_value=0.0, availability=1.0) for index in range(10)]
        self.assertEqual(sum(item.selected_tier == "full" for item in decisions), 5)
        self.assertAlmostEqual(decisions[-1].total_spent, 5.0)

    def test_paced_global_budget_cannot_spend_future_credit(self):
        cfg = SchedulerConfig(
            policy="airs",
            credit_mode="global",
            pace_budget=True,
            budget_ratio=0.5,
            total_batches=4,
        )
        scheduler = AIRSScheduler(cfg)
        first = scheduler.select(0, batch_value=0.9, availability=1.0)
        self.assertEqual(first.selected_tier, "medium")
        self.assertAlmostEqual(first.total_spent, 0.5)
        second = scheduler.select(1, batch_value=0.0, availability=1.0)
        self.assertEqual(second.selected_tier, "inference")
        third = scheduler.select(2, batch_value=0.9, availability=1.0)
        self.assertEqual(third.selected_tier, "full")
        self.assertLessEqual(third.total_spent, 0.5 * 3 + 1e-9)

    def test_resource_traces_have_expected_endpoints(self):
        self.assertEqual(resource_availability("sudden_drop", 0, 10), 1.0)
        self.assertEqual(resource_availability("sudden_drop", 9, 10), 0.2)
        self.assertEqual(resource_availability("recovery", 0, 10), 0.2)
        self.assertEqual(resource_availability("recovery", 9, 10), 1.0)


if __name__ == "__main__":
    unittest.main()
