from __future__ import annotations

import unittest

from airs.config import TIER_NAMES, TIER_SPECS
from airs.experiments import EXPERIMENT_IDS, experiment_cases


class ExperimentMatrixTest(unittest.TestCase):
    def test_all_six_experiments_have_cases(self):
        counts = {experiment: len(experiment_cases(experiment)) for experiment in EXPERIMENT_IDS}
        self.assertEqual(counts, {"E1": 4, "E2": 5, "E3": 4, "E4": 4, "E5": 4, "E6": 2})
        self.assertEqual(sum(counts.values()), 23)

    def test_case_ids_are_unique(self):
        ids = [
            (experiment, case.config_id)
            for experiment in EXPERIMENT_IDS
            for case in experiment_cases(experiment)
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_tier_strengths_match_the_registered_protocol(self):
        costs = [TIER_SPECS[name].cost for name in TIER_NAMES]
        update_fractions = [TIER_SPECS[name].max_update_fraction for name in TIER_NAMES]
        augment_times = [TIER_SPECS[name].augment_times for name in TIER_NAMES]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(update_fractions, [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(augment_times, [0, 0, 1, 4])
        self.assertTrue(all(not hasattr(TIER_SPECS[name], "proto_align_scale") for name in TIER_NAMES))

    def test_scheduler_comparison_uses_matched_budget(self):
        cases = experiment_cases("E4")
        self.assertEqual({case.budget_ratio for case in cases}, {0.5})
        self.assertEqual(cases[0].credit_mode, "bank")
        self.assertFalse(cases[0].pace_budget)
        self.assertEqual({case.credit_mode for case in cases[1:]}, {"global"})
        self.assertEqual({case.pace_budget for case in cases[1:]}, {False})
        self.assertEqual([case.label for case in cases], ["AIRS", "Greedy", "Periodic", "Random"])

    def test_joint_ablation_uses_one_budget_protocol(self):
        cases = experiment_cases("E5")
        self.assertEqual({case.budget_ratio for case in cases}, {0.4})
        self.assertEqual({case.credit_mode for case in cases}, {"global"})
        self.assertEqual({case.pace_budget for case in cases}, {True})

    def test_credit_ablation_excludes_global_budget(self):
        cases = experiment_cases("E6")
        self.assertEqual([case.credit_mode for case in cases], ["bank", "fixed"])


if __name__ == "__main__":
    unittest.main()
