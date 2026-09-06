from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_condition_dataset_cycle import (
    delete_command,
    download_command,
    evaluate_command,
    reported_failures,
    require_full_subset,
    selected_datasets,
    summary_json,
    validate_runner_context,
)


class RunConditionDatasetCycleTests(unittest.TestCase):
    def test_full_run_requires_dataset_subset(self) -> None:
        with self.assertRaises(ValueError):
            require_full_subset(
                SimpleNamespace(stage="full", dataset_subset_file=None)
            )
        require_full_subset(
            SimpleNamespace(stage="smoke", dataset_subset_file=None)
        )

    def test_selected_datasets_are_unique_after_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditions.csv"
            write_conditions(path)
            args = SimpleNamespace(
                conditions_csv=path,
                condition=["A"],
                dataset=[],
                limit_pairs=0,
                limit_pairs_per_dataset=0,
                limit_datasets=0,
                dataset_subset_file=None,
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

    def test_manual_context_is_passed_and_requires_one_dataset(self) -> None:
        args = base_args()
        args.runner = "manual"
        args.context_file = Path("context.json")
        command = evaluate_command("owner/one", args, Path("/repo"))
        self.assertIn("--context-file", command)
        self.assertIn("context.json", command)

        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp) / "context.json"
            context.write_text("{}", encoding="utf-8")
            validation_args = SimpleNamespace(runner="manual", context_file=context)
            validate_runner_context(validation_args, ["owner/one"])
            with self.assertRaises(ValueError):
                validate_runner_context(validation_args, ["owner/one", "owner/two"])

    def test_delete_command_targets_one_dataset(self) -> None:
        args = base_args()

        command = delete_command("owner/one", args, Path("/repo"))

        self.assertIn("delete_condition_datasets.py", command[1])
        self.assertIn("--dataset", command)
        self.assertIn("owner/one", command)


    def test_reported_failures_reads_eval_summary(self) -> None:
        text = '{"datasets": 1, "failures": 4}'

        self.assertEqual(reported_failures(text), 4)

    def test_summary_json_uses_final_object_after_progress_output(self) -> None:
        text = '[batch] start\n{"temporary": 1}\n[batch] done\n{"datasets": 2, "failures": 3}\n'

        self.assertEqual(summary_json(text), {"datasets": 2, "failures": 3})

    def test_dataset_subset_applies_to_smoke_pipeline_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conditions = root / "conditions.csv"
            subset = root / "datasets.txt"
            write_conditions(conditions)
            subset.write_text("owner/two\n", encoding="utf-8")
            args = SimpleNamespace(
                conditions_csv=conditions,
                condition=[],
                dataset=[],
                limit_pairs=0,
                limit_pairs_per_dataset=0,
                limit_datasets=0,
                dataset_subset_file=subset,
            )

            self.assertEqual(selected_datasets(args), ["owner/two"])


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
        context_file=None,
        codex_bin="codex",
        codex_approval="never",
        smoke_limit=3,
        sample_size=20,
        timeout=0,
        codex_timeout=120,
        codex_bypass_sandbox=False,
        condition=[],
        limit_pairs=2,
        limit_pairs_per_dataset=0,
        dataset_subset_file=None,
        continue_on_error=True,
        skip_refinement=False,
        fresh_run_dir=True,
        skip_model_download=False,
        keep_model_cache=False,
        keep_dataset_cache=False,
        allow_partial_download=True,
        skip_version_incompatible_datasets=True,
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
