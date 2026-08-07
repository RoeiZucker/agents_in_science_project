from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_condition_dataset_cycle import delete_command, download_command, evaluate_command, reported_failures, selected_datasets


class RunConditionDatasetCycleTests(unittest.TestCase):
    def test_selected_datasets_are_unique_after_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditions.csv"
            write_conditions(path)
            args = SimpleNamespace(
                conditions_csv=path,
                condition=["A"],
                dataset=[],
                limit_pairs=0,
                limit_datasets=0,
            )

            self.assertEqual(selected_datasets(args), ["owner/one", "owner/two"])

    def test_download_command_targets_one_dataset(self) -> None:
        args = base_args()

        command = download_command("owner/one", args, Path("/repo"))

        self.assertIn("download_condition_datasets.py", command[1])
        self.assertIn("--dataset", command)
        self.assertIn("owner/one", command)
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--include-models", command)
        self.assertIn("--limit-pairs", command)

    def test_evaluate_command_keeps_cache_for_delete_step(self) -> None:
        args = base_args()

        command = evaluate_command("owner/one", args, Path("/repo"))

        self.assertIn("run_evaluation_conditions.py", command[1])
        self.assertIn("--cleanup-cache", command)
        self.assertIn("none", command)
        self.assertIn("--dataset", command)
        self.assertIn("owner/one", command)

    def test_delete_command_targets_one_dataset(self) -> None:
        args = base_args()

        command = delete_command("owner/one", args, Path("/repo"))

        self.assertIn("delete_condition_datasets.py", command[1])
        self.assertIn("--dataset", command)
        self.assertIn("owner/one", command)


    def test_reported_failures_reads_eval_summary(self) -> None:
        text = '{"datasets": 1, "failures": 4}'

        self.assertEqual(reported_failures(text), 4)


def base_args() -> SimpleNamespace:
    return SimpleNamespace(
        python="python",
        conditions_csv=Path("conditions.csv"),
        project_root=Path("runtime"),
        output_root=None,
        split="auto",
        download_split=[],
        trust_remote_code=True,
        stage="smoke",
        runner="codex",
        codex_bin="codex",
        codex_approval="never",
        smoke_limit=3,
        sample_size=20,
        timeout=0,
        condition=[],
        limit_pairs=2,
        continue_on_error=True,
        skip_refinement=False,
        fresh_run_dir=True,
        skip_model_download=False,
    )


def write_conditions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "query_dataset", "rank", "model_name"])
        writer.writerow(["A", "owner/one", "1", "model/a"])
        writer.writerow(["A", "owner/two", "1", "model/b"])
        writer.writerow(["B", "owner/one", "1", "model/c"])


if __name__ == "__main__":
    unittest.main()
