from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_training_runs import build_summary, load_runs


class TrainingRunSummaryTests(unittest.TestCase):
    def test_summary_picks_best_final_eval_loss_and_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            first = runs_dir / "first"
            second = runs_dir / "second"
            first.mkdir()
            second.mkdir()

            (first / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "train_examples": 4,
                        "eval_examples": 1,
                        "initial_eval_loss": 5.0,
                        "final_eval_loss": 4.0,
                        "config": {"epochs": 1},
                    }
                ),
                encoding="utf-8",
            )
            (second / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "train_examples": 4,
                        "eval_examples": 1,
                        "initial_eval_loss": 3.0,
                        "final_eval_loss": 3.2,
                        "config": {"epochs": 2},
                    }
                ),
                encoding="utf-8",
            )

            records = load_runs(runs_dir)
            summary = build_summary(records)

            self.assertEqual(2, summary["run_count"])
            self.assertEqual(1, summary["improved_run_count"])
            self.assertTrue(summary["best_run"]["run_dir"].endswith("second"))
            self.assertIn(summary["latest_run"]["run_dir"], {str(first), str(second)})


if __name__ == "__main__":
    unittest.main()
