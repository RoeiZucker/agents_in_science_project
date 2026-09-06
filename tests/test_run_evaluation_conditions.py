from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from run_evaluation_conditions import (
    build_codex_prompt,
    cleanup_cache_dirs,
    codex_command,
    command_without_prompt,
    context_failures,
    group_by_dataset,
    eval_command,
    internal_eval_failures,
    join_results,
    missing_or_bad_result_failures,
    prepare_manual_context,
    prepare_run_dir,
    recover_context_from_stdout,
    retry_codex_prompt,
    run_codex_context_then_script,
    run_codex_dataset_group,
    select_rows,
    tail,
    unique_models,
    unique_models_from_candidates,
    validate_runner_context,
    write_candidates,
)


class RunEvaluationConditionsTests(unittest.TestCase):
    def test_tail_handles_empty_and_long_stderr(self) -> None:
        self.assertEqual(tail(""), "")
        self.assertEqual(tail("abcdef", 3), "def")

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
            self.assertEqual(value["owner"], "partner_retrieval")
            self.assertEqual(value["handoff_version"], 1)
            self.assertEqual(value["candidates"][0]["candidate_id"], "A:1:1")
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
        args = SimpleNamespace(codex_bin="codex", codex_approval="never", codex_bypass_sandbox=False)

        command = codex_command(args, Path("/repo"), "hello")
        logged = command_without_prompt(command)

        self.assertEqual(command[:5], ["codex", "-C", "/repo", "-a", "never"])
        self.assertIn("workspace-write", command)
        self.assertEqual(command[-3:], ["exec", "--skip-git-repo-check", "hello"])
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

        self.assertIn("Do not run run_eval_agent.py", prompt)
        self.assertIn("Do not load model weights", prompt)
        self.assertIn("RUN_DIR=/out/run", prompt)
        self.assertIn("CONTEXT_OUTPUT=/out/run/codex_context.json", prompt)
        self.assertIn("- model/a", prompt)

    def test_codex_context_retries_after_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            calls = []
            result = SimpleNamespace(returncode=0, stdout="", stderr="")

            def fake_run(*_args):
                calls.append(_args)
                if len(calls) == 2:
                    prompt = _args[0][-1]
                    output_line = next(
                        line for line in prompt.splitlines()
                        if line.startswith("- CONTEXT_OUTPUT=")
                    )
                    staged_path = Path(output_line.split("=", 1)[1])
                    staged_path.write_text(
                        json.dumps(valid_context()), encoding="utf-8"
                    )
                return result

            args = SimpleNamespace(
                python="python",
                codex_bin="codex",
                codex_approval="never",
                codex_bypass_sandbox=False,
                codex_timeout=1,
                codex_context_attempts=2,
            )
            with patch(
                "run_evaluation_conditions.run_command", side_effect=fake_run
            ):
                failures = run_codex_dataset_group(
                    "dataset/one",
                    condition_rows()[:2],
                    run_dir,
                    run_dir / "candidates.json",
                    args,
                    Path("/repo"),
                    Path("/project"),
                    Path("/output"),
                    Path("/hf"),
                    Path("/datasets"),
                )

            self.assertEqual(failures, [])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], Path(calls[0][0][2]))
            self.assertNotEqual(calls[0][1], Path("/repo"))
            self.assertTrue((run_dir / "batch_logs/codex_retry_2.stdout.txt").exists())

    def test_recovers_valid_context_printed_to_codex_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codex_context.json"
            stdout = f"Metadata complete.\n{json.dumps(valid_context())}\nDone."

            recovered = recover_context_from_stdout(
                path, stdout, "dataset/one", ["model/a", "model/b"]
            )

            self.assertTrue(recovered)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), valid_context())

    def test_does_not_recover_context_with_wrong_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codex_context.json"
            stdout = json.dumps({**valid_context(), "dataset": "dataset/two"})

            recovered = recover_context_from_stdout(
                path, stdout, "dataset/one", ["model/a", "model/b"]
            )

            self.assertFalse(recovered)
            self.assertFalse(path.exists())

    def test_retry_prompt_reports_validation_error(self) -> None:
        prompt = retry_codex_prompt(
            "base prompt", [{"error": "Missing codex_context.json"}]
        )

        self.assertIn("Missing codex_context.json", prompt)
        self.assertIn("Overwrite CONTEXT_OUTPUT", prompt)

    def test_accepts_valid_full_codex_context_schema(self) -> None:
        self.assertEqual(context_errors(valid_context()), [])

    def test_accepts_codex_selected_subset_split_choices_and_method(self) -> None:
        context = with_dataset_context(
            valid_context(),
            subset="simplified",
            split="validation",
            choices_columns=["answer0", "answer1"],
            task="qa",
            evaluation_method="qa_f1",
        )

        self.assertEqual(context_errors(context), [])

    def test_rejects_unknown_evaluation_method(self) -> None:
        context = with_dataset_context(
            valid_context(), evaluation_method="model_judged_similarity"
        )

        self.assertIn("evaluation_method", context_errors(context)[0]["error"])

    def test_rejects_evaluation_method_incompatible_with_task(self) -> None:
        context = with_dataset_context(
            valid_context(), task="qa", evaluation_method="rouge_l"
        )

        error = context_errors(context)[0]["error"]
        self.assertIn("incompatible", error)

    def test_rejects_missing_and_malformed_codex_context_fields(self) -> None:
        cases = [
            ("missing", without_field(valid_context(), "confidence"), "missing fields"),
            ("models type", {**valid_context(), "models": "model/a"}, "list of strings"),
            (
                "confidence type",
                {**valid_context(), "confidence": []},
                "confidence must be a string",
            ),
            (
                "unknown field",
                {**valid_context(), "extra": "not allowed"},
                "unknown fields",
            ),
            (
                "column type",
                with_dataset_context(valid_context(), question_column=1),
                "question_column must be a string",
            ),
            (
                "model notes type",
                {**valid_context(), "model_context": {"notes": []}},
                "model_context.notes must be a string",
            ),
        ]
        for name, value, message in cases:
            with self.subTest(name=name):
                self.assertIn(message, context_errors(value)[0]["error"])

    def test_rejects_invalid_codex_context_task_and_confidence(self) -> None:
        bad_task = with_dataset_context(valid_context(), task="question_answering")
        bad_confidence = {**valid_context(), "confidence": "certain"}

        self.assertIn("dataset_context.task", context_errors(bad_task)[0]["error"])
        self.assertIn("confidence", context_errors(bad_confidence)[0]["error"])

    def test_rejects_codex_context_dataset_and_model_mismatch(self) -> None:
        bad_dataset = {**valid_context(), "dataset": "dataset/two"}
        bad_models = {**valid_context(), "models": ["model/b", "model/a"]}

        self.assertIn("dataset mismatch", context_errors(bad_dataset)[0]["error"])
        self.assertIn("models mismatch", context_errors(bad_models)[0]["error"])

    def test_rejects_invalid_codex_context_label_map(self) -> None:
        not_object = with_dataset_context(valid_context(), label_map=[])
        bad_value = with_dataset_context(valid_context(), label_map={"0": 0})

        self.assertIn("label_map must be an object", context_errors(not_object)[0]["error"])
        self.assertIn("keys and values must be strings", context_errors(bad_value)[0]["error"])

    def test_invalid_codex_context_skips_script_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = without_field(valid_context(), "confidence")
            (root / "codex_context.json").write_text(
                json.dumps(invalid), encoding="utf-8"
            )
            failures = context_failures(
                "dataset/one", ["model/a", "model/b"], root / "codex_context.json"
            )
            with patch(
                "run_evaluation_conditions.run_codex_dataset_group",
                return_value=failures,
            ), patch(
                "run_evaluation_conditions.run_script_dataset_group"
            ) as script_runner:
                result = run_codex_context_then_script(
                    "dataset/one", condition_rows()[:2], root,
                    root / "candidates.json", SimpleNamespace(), root, root,
                    root, root, root,
                )

            self.assertEqual(result, failures)
            script_runner.assert_not_called()

    def test_manual_context_requires_one_dataset_and_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.json"
            path.write_text(json.dumps(valid_context()), encoding="utf-8")
            args = SimpleNamespace(runner="manual", context_file=path)

            validate_runner_context(args, group_by_dataset(condition_rows()[:2]))

            with self.assertRaises(ValueError):
                validate_runner_context(args, group_by_dataset(condition_rows()))
            with self.assertRaises(ValueError):
                validate_runner_context(SimpleNamespace(runner="manual", context_file=None), {})

    def test_reads_unique_models_from_candidate_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_candidates(Path(tmp), "dataset/one", condition_rows()[:2])

            models = unique_models_from_candidates(path)

            self.assertEqual(models, ["model/a", "model/b"])

    def test_manual_context_is_preserved_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            run_dir = root / "run"
            run_dir.mkdir()
            source.write_text(json.dumps(valid_context()), encoding="utf-8")

            destination, failures = prepare_manual_context("dataset/one", condition_rows()[:2], run_dir, source)

            self.assertEqual(failures, [])
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), valid_context())
            provenance = json.loads((run_dir / "manual_context_source.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["source"], str(source.resolve()))

            bad = {**valid_context(), "models": ["model/a"]}
            source.write_text(json.dumps(bad), encoding="utf-8")
            _, failures = prepare_manual_context("dataset/one", condition_rows()[:2], run_dir, source)
            self.assertEqual(failures[0]["stage"], "manual_context")

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

    def test_prepare_run_dir_can_start_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            stale = run_dir / "results.csv"
            stale.parent.mkdir()
            stale.write_text("old", encoding="utf-8")

            prepare_run_dir(run_dir, fresh=True)

            self.assertTrue(run_dir.exists())
            self.assertFalse(stale.exists())

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


def valid_context() -> dict:
    return {
        "dataset": "dataset/one",
        "models": ["model/a", "model/b"],
        "dataset_context": {
            "question_column": "question",
            "answer_column": "label",
            "choices_column": "",
            "image_column": "",
            "task": "generation",
            "label_map": {"0": "no", "1": "yes"},
            "prompt_template": "{question}",
            "notes": "metadata only",
        },
        "model_context": {"notes": "compatible text models"},
        "confidence": "high",
    }


def context_errors(value: dict) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "codex_context.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return context_failures(
            "dataset/one", ["model/a", "model/b"], path
        )


def without_field(value: dict, field: str) -> dict:
    result = dict(value)
    result.pop(field)
    return result


def with_dataset_context(value: dict, **changes) -> dict:
    return {
        **value,
        "dataset_context": {**value["dataset_context"], **changes},
    }


if __name__ == "__main__":
    unittest.main()
