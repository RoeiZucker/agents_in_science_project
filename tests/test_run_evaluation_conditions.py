from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_evaluation_conditions import (
    build_codex_prompt,
    cleanup_cache_dirs,
    codex_command,
    command_without_prompt,
    group_by_dataset,
    eval_command,
    internal_eval_failures,
    join_results,
    missing_or_bad_result_failures,
    select_rows,
    unique_models,
    unique_models_from_candidates,
    write_candidates,
)


class RunEvaluationConditionsTests(unittest.TestCase):
    def test_selects_rows_by_condition_dataset_and_limits(self) -> None:
        rows = condition_rows()
        args = SimpleNamespace(condition=["A"], dataset=["dataset/one"], limit_pairs=0, limit_datasets=1)

        selected = select_rows(rows, args)

        self.assertEqual(len(selected), 2)
        self.assertEqual({row["query_dataset"] for row in selected}, {"dataset/one"})
        self.assertEqual({row["condition"] for row in selected}, {"A"})

    def test_groups_rows_and_keeps_unique_model_order(self) -> None:
        rows = condition_rows()

        grouped = group_by_dataset(rows)
        models = unique_models(grouped["dataset/one"])

        self.assertEqual(set(grouped), {"dataset/one", "dataset/two"})
        self.assertEqual(models, ["model/a", "model/b"])

    def test_writes_candidate_json_from_partner_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_candidates(Path(tmp), "dataset/one", condition_rows()[:1])

            value = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(value["retrieval_agent"], "partner_conditions_csv")
            self.assertEqual(value["candidates"][0]["model"], "model/a")
            self.assertEqual(value["candidates"][0]["rank"], 1)
            self.assertEqual(value["candidates"][0]["score"], 0.9)

    def test_joins_partner_rows_with_eval_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_csv = Path(tmp) / "results.csv"
            write_results(results_csv)

            joined = join_results(condition_rows()[:2], results_csv)

            self.assertEqual(joined[0]["eval_score"], "0.75")
            self.assertEqual(joined[1]["eval_status"], "missing")

    def test_eval_command_passes_trust_remote_code(self) -> None:
        args = SimpleNamespace(
            python="python",
            split="auto",
            stage="smoke",
            smoke_limit=3,
            sample_size=20,
            timeout=0,
            continue_on_error=True,
            trust_remote_code=True,
        )

        command = eval_command(args, Path("/repo"), Path("/project"), Path("/out"), "dataset/one", ["model/a"], Path("/hf"))

        self.assertIn("--trust-remote-code", command)

    def test_builds_codex_command_and_hides_prompt_in_logs(self) -> None:
        args = SimpleNamespace(codex_bin="codex", codex_approval="never")

        command = codex_command(args, Path("/repo"), "hello")
        logged = command_without_prompt(command)

        self.assertEqual(command[:5], ["codex", "-C", "/repo", "-a", "never"])
        self.assertEqual(command[-2:], ["exec", "hello"])
        self.assertIn("<prompt omitted", logged[-1])

    def test_builds_codex_prompt_with_paths_and_refinement(self) -> None:
        args = SimpleNamespace(
            python="python",
            split="auto",
            stage="smoke",
            smoke_limit=3,
            sample_size=20,
            trust_remote_code=True,
            skip_refinement=False,
            cleanup_cache="after-dataset",
        )

        prompt = build_codex_prompt(
            "dataset/one",
            condition_rows()[:2],
            Path("/out/run"),
            Path("/out/run/candidates.json"),
            args,
            Path("/project"),
            Path("/out"),
            Path("/hf"),
            Path("/datasets"),
        )

        self.assertIn("--trust-remote-code", prompt)
        self.assertIn("Run analyze_eval_errors.py", prompt)
        self.assertIn("RUN_DIR=/out/run", prompt)
        self.assertIn("- model/a", prompt)

    def test_reads_unique_models_from_candidate_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_candidates(Path(tmp), "dataset/one", condition_rows()[:2])

            models = unique_models_from_candidates(path)

            self.assertEqual(models, ["model/a", "model/b"])

    def test_reads_internal_eval_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            value = [{"model": "model/a", "stage": "smoke", "returncode": 1}]
            (run_dir / "failures.json").write_text(json.dumps(value), encoding="utf-8")

            failures = internal_eval_failures(run_dir)

            self.assertEqual(failures[0]["model"], "model/a")
            self.assertEqual(failures[0]["stage"], "smoke")

    def test_flags_missing_or_bad_result_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_csv = Path(tmp) / "results.csv"
            write_bad_results(results_csv)

            failures = missing_or_bad_result_failures(results_csv)

            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["model"], "model/b")
            self.assertEqual(failures[0]["status"], "missing")

    def test_cleanup_only_removes_hf_cache_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_home = root / ".hf_cache"
            datasets_cache = root / ".hf_datasets_cache"
            output = root / "eval_results"
            for path in [hf_home / "hub", hf_home / "transformers", datasets_cache, output]:
                path.mkdir(parents=True)

            cleanup_cache_dirs(hf_home, datasets_cache)

            self.assertFalse((hf_home / "hub").exists())
            self.assertFalse((hf_home / "transformers").exists())
            self.assertFalse(datasets_cache.exists())
            self.assertTrue(output.exists())


def condition_rows() -> list[dict[str, str]]:
    return [
        row("A", "dataset/one", "1", "model/a", "0.9"),
        row("A", "dataset/one", "2", "model/b", "0.8"),
        row("B", "dataset/two", "1", "model/a", "0.7"),
    ]


def row(condition: str, dataset: str, rank: str, model: str, score: str) -> dict[str, str]:
    return {
        "condition": condition,
        "query_dataset": dataset,
        "rank": rank,
        "model_name": model,
        "recommendation_score": score,
        "source_method": "mock",
        "reasoning": "because",
    }


def write_results(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "model", "score", "total", "labeled_total", "status", "output_dir", "notes"])
        writer.writerow(["dataset/one", "validation", "model/a", "0.75", "10", "10", "ok", "/out", ""])


def write_bad_results(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "model", "score", "total", "labeled_total", "status", "output_dir", "notes"])
        writer.writerow(["dataset/one", "validation", "model/a", "0.75", "10", "10", "ok", "/out-a", ""])
        writer.writerow(["dataset/one", "validation", "model/b", "", "", "", "missing", "/out-b", "summary.json is missing."])


if __name__ == "__main__":
    unittest.main()
