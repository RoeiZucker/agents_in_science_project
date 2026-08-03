from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from refinement_agent_core import analyze_run, feedback_prompt, make_retrieval_feedback


class RefinementAgentCoreTests(unittest.TestCase):
    def test_analyzes_model_errors_and_candidate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = build_fake_run(Path(tmp))
            candidates_path = write_candidates(Path(tmp))

            analysis = analyze_run(run_dir, candidates_path, max_failed_examples=5)

            self.assertEqual(analysis["dataset"], "google/boolq")
            self.assertEqual(analysis["task"], "boolean_qa")
            self.assertEqual(analysis["comparisons"]["best_model"]["model"], "google/flan-t5-base")
            self.assertAlmostEqual(analysis["comparisons"]["score_gap"], 0.25)
            self.assertEqual(len(analysis["comparisons"]["all_models_failed_examples"]), 1)
            self.assertEqual(analysis["models"][0]["retrieval_candidate"]["rank"], 1)
            self.assertEqual(analysis["models"][0]["per_target_accuracy"]["true"]["accuracy"], 0.0)
            self.assertIn("prediction_collapse", analysis["failure_modes"])

    def test_builds_retrieval_feedback_for_boolean_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analysis = analyze_run(build_fake_run(Path(tmp)), None)

            feedback = make_retrieval_feedback(analysis, next_round=3)
            prompt = feedback_prompt(feedback)

            self.assertEqual(feedback["next_round"], 3)
            self.assertIn("reading-comprehension", feedback["retrieval_constraints"]["prefer_tasks"])
            self.assertIn("google/flan-t5-small", feedback["retrieval_constraints"]["candidate_exclusions"])
            self.assertIn("candidate_models.json", prompt)


def build_fake_run(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    write_dataset_inspection(run_dir)
    small_dir = run_dir / "small"
    base_dir = run_dir / "base"
    small_dir.mkdir()
    base_dir.mkdir()
    write_result_rows(run_dir, small_dir, base_dir)
    write_model_output(small_dir, "google/flan-t5-small", ["false", "false", "false", "false"])
    write_model_output(base_dir, "google/flan-t5-base", ["true", "false", "false", "true"])
    return run_dir


def write_dataset_inspection(run_dir: Path) -> None:
    value = {
        "dataset": "google/boolq",
        "selected_split": "validation",
        "columns": ["question", "passage", "answer"],
        "task": "generation",
        "label_map": {"False": "false", "True": "true"},
    }
    (run_dir / "dataset_inspection.json").write_text(json.dumps(value), encoding="utf-8")


def write_result_rows(run_dir: Path, small_dir: Path, base_dir: Path) -> None:
    rows = [
        ["google/boolq", "validation", "google/flan-t5-small", "0.25", "4", "4", "ok", str(small_dir), ""],
        ["google/boolq", "validation", "google/flan-t5-base", "0.5", "4", "4", "ok", str(base_dir), ""],
    ]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "model", "score", "total", "labeled_total", "status", "output_dir", "notes"])
        writer.writerows(rows)


def write_model_output(output_dir: Path, model: str, predictions: list[str]) -> None:
    answers = ["true", "true", "false", "false"]
    correct = [prediction == answer for prediction, answer in zip(predictions, answers)]
    summary = {
        "model": model,
        "dataset": "google/boolq",
        "split": "validation",
        "task": "generation",
        "total": 4,
        "labeled_total": 4,
        "correct": sum(correct),
        "accuracy": sum(correct) / len(correct),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for idx, prediction in enumerate(predictions):
            row = {
                "index": idx,
                "id": idx,
                "question": f"Question {idx}?",
                "answer": answers[idx],
                "target_answer": answers[idx],
                "prediction": prediction,
                "correct": correct[idx],
            }
            handle.write(json.dumps(row) + "\n")


def write_candidates(root: Path) -> Path:
    path = root / "candidates.json"
    value = {
        "candidates": [
            {"model": "google/flan-t5-small", "rank": 1, "score": 0.7, "source": "mock", "reason": "small"},
            {"model": "google/flan-t5-base", "rank": 2, "score": 0.6, "source": "mock", "reason": "base"},
        ]
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
