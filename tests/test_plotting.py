from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from airs.plotting import plot_experiment, plot_overview


def summary(experiment: str, config_id: str, label: str, index: int):
    row = {
        "experiment": experiment,
        "config_id": config_id,
        "label": label,
        "policy": label.lower(),
        "credit_mode": "bank",
        "resource_trace": "steady",
        "budget_percent": 20.0 + index * 20.0,
        "accuracy_percent": 70.0 + index,
        "mean_batch_latency_ms": 10.0 + index,
        "p95_batch_latency_ms": 15.0 + index,
        "total_gpu_time_seconds": 20.0 + index,
        "total_wall_time_seconds": 30.0 + index,
        "num_batches": 10,
        "budget_utilization_percent": 80.0 + index,
        "resource_violations": index,
        "teacher_forward_calls": index,
        "backward_calls": index * 2,
    }
    for tier in ("inference", "light", "medium", "full"):
        row[f"{tier}_percent"] = 25.0
    return row


class PlottingSmokeTest(unittest.TestCase):
    def test_every_experiment_plotter_builds_a_figure(self):
        rows = {
            "E1": [summary("E1", f"tier_{tier}", tier.title(), index) for index, tier in enumerate(("inference", "light", "medium", "full"))],
            "E2": [summary("E2", f"budget_{budget}", f"{budget}%", index) for index, budget in enumerate((20, 40, 60, 80, 100))],
            "E3": [summary("E3", f"trace_{trace}", trace, index) for index, trace in enumerate(("steady", "sudden_drop", "recovery", "cyclic"))],
            "E4": [summary("E4", f"policy_{name}", name, index) for index, name in enumerate(("AIRS", "Greedy", "Periodic", "Random"))],
            "E5": [summary("E5", f"joint_{index}", label, index) for index, label in enumerate(("Value + resource", "Value only", "Resource only", "No joint decision"))],
            "E6": [summary("E6", f"credit_{index}", label, index) for index, label in enumerate(("Credit bank", "Fixed quota"))],
        }

        captured = {}

        def close_only(fig, base):
            figure_name = Path(base).name
            captured[figure_name] = {
                "axes": len(fig.axes),
                "accuracy_ylim": fig.axes[0].get_ylim(),
                "accuracy_labels": [tick.get_text() for tick in fig.axes[0].get_xticklabels()],
                "figure_legends": len(fig.legends),
                "axis0_has_legend": fig.axes[0].get_legend() is not None,
            }
            if figure_name == "E1_tier_resource_profile":
                captured[figure_name].update(
                    {
                        "panel_c_ylabel": fig.axes[2].get_ylabel(),
                        "panel_c_heights": [bar.get_height() for bar in fig.axes[2].patches],
                        "panel_d_bar_count": len(fig.axes[3].patches),
                        "panel_d_legend": [
                            text.get_text() for text in fig.axes[3].get_legend().get_texts()
                        ] if fig.axes[3].get_legend() else [],
                    }
                )
            plt.close(fig)
            return [Path(base)]

        batch_rows = [
            {"resource_availability": "1.0", "selected_tier": "full"},
            {"resource_availability": "0.5", "selected_tier": "medium"},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch("airs.plotting._save", side_effect=close_only) as save:
            with patch("airs.plotting._read_batch_metrics", return_value=batch_rows):
                root = Path(tmp)
                for experiment, experiment_rows in rows.items():
                    plot_experiment(experiment, experiment_rows, root / experiment)
                plot_overview(rows["E2"] + rows["E4"], root)
            self.assertEqual(save.call_count, 7)
            self.assertEqual(captured["E3_dynamic_resource_response"]["axes"], 5)
            self.assertEqual(captured["E3_dynamic_resource_response"]["figure_legends"], 1)
            self.assertFalse(captured["E3_dynamic_resource_response"]["axis0_has_legend"])
            self.assertGreater(captured["E1_tier_resource_profile"]["accuracy_ylim"][0], 0.0)
            self.assertEqual(captured["E1_tier_resource_profile"]["panel_c_ylabel"], "Relative cost proxy (%)")
            self.assertEqual(captured["E1_tier_resource_profile"]["panel_c_heights"][0], 0.0)
            self.assertAlmostEqual(captured["E1_tier_resource_profile"]["panel_c_heights"][-1], 100.0)
            self.assertEqual(captured["E1_tier_resource_profile"]["panel_d_bar_count"], 16)
            self.assertEqual(
                captured["E1_tier_resource_profile"]["panel_d_legend"],
                ["Teacher calls", "Backward chunks", "Runtime", "GPU time"],
            )
            self.assertNotIn("Global budget", captured["E6_credit_bank_ablation"]["accuracy_labels"])


if __name__ == "__main__":
    unittest.main()
