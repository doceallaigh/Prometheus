from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prometheus.basin_analysis import _auc, analyze_suppression_basins


class BasinAnalysisTests(unittest.TestCase):
    def test_auc_handles_order_and_ties(self) -> None:
        self.assertEqual(_auc([0.0, 1.0], [2.0, 3.0]), 1.0)
        self.assertEqual(_auc([1.0], [1.0]), 0.5)

    def test_classifies_rescue_and_failure(self) -> None:
        records = [
            self._record(0, latent="#### 7", root="#### 5", other="#### 5"),
            self._record(1, latent="#### 7", root="#### 9", other="#### 5"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dump = Path(temp_dir) / "fork.jsonl"
            dump.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            summary = analyze_suppression_basins(dump, Path(temp_dir) / "out", bootstrap_samples=20)

            self.assertEqual(summary["outcomes"]["rescue"], 1)
            self.assertEqual(summary["outcomes"]["suppression_failure"], 1)
            self.assertTrue((Path(temp_dir) / "out" / "report.md").exists())

    @staticmethod
    def _record(index: int, latent: str, root: str, other: str) -> dict:
        def branch(text: str, is_root: bool) -> dict:
            return {
                "text": text,
                "root": is_root,
                "mean_logprob": -0.2,
                "mean_delta": 2.0,
                "max_delta": 4.0,
                "intrusion_rate": 0.1,
                "steps": 20,
            }

        return {
            "index": index,
            "gold": "5",
            "latent": latent,
            "branches": [branch(root, True), branch(other, False)],
        }


if __name__ == "__main__":
    unittest.main()