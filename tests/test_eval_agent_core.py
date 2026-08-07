from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval_agent_core import (
    audit_eval_dir,
    build_eval_command,
    choose_first,
    has_null_use_cache,
    infer_model_type_from_metadata,
    sanitize_model_dir,
)


class EvalAgentCoreTests(unittest.TestCase):
    def test_recognizes_claim_as_an_input_text_column(self) -> None:
        self.assertEqual(
            choose_first(["claim_id", "claim", "label"], ("question", "text", "claim"), "question"),
            "claim",
        )

    def test_infers_known_model_families(self) -> None:
        self.assertEqual(
            infer_model_type_from_metadata("TinyLlama/TinyLlama-1.1B", "llama", ["LlamaForCausalLM"]),
            "causal_lm",
        )
        self.assertEqual(
            infer_model_type_from_metadata("google/flan-t5-xl", "t5", ["T5ForConditionalGeneration"]),
            "seq2seq_lm",
        )
        self.assertEqual(
            infer_model_type_from_metadata("yaxili96/FactCG-DeBERTa-v3-Large", "deberta-v2", ["DebertaV2ForSequenceClassification"]),
            "sequence_classifier",
        )
        self.assertEqual(
            infer_model_type_from_metadata("lytang/MiniCheck-Flan-T5-Large", None, []),
            "seq2seq_lm",
        )
        self.assertEqual(
            infer_model_type_from_metadata("OpenGVLab/InternVL3_5-2B", "internvl_chat", ["InternVLChatModel"]),
            "vlm_chat",
        )
        self.assertEqual(
            infer_model_type_from_metadata("ZTE-AIM/3B-Curr-ReFT", "qwen2_5_vl", ["Qwen2_5_VLForConditionalGeneration"]),
            "vlm_processor",
        )

    def test_sanitizes_bad_qwen_config_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"model_type": "qwen2_5_vl", "use_cache": None, "nested": {"use_cache": None}}),
                encoding="utf-8",
            )
            (source / "preprocessor_config.json").write_text(
                json.dumps({"image_processor_type": "Qwen2_5_VLImageProcessor"}),
                encoding="utf-8",
            )
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")

            args = SimpleNamespace(model="ZTE-AIM/3B-Curr-ReFT", output_dir=str(root), sanitized_model_dir=str(target))
            fixed = sanitize_model_dir(source, args)
            self.assertEqual(fixed, target)

            original = json.loads((source / "config.json").read_text(encoding="utf-8"))
            config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            preprocessor = json.loads((target / "preprocessor_config.json").read_text(encoding="utf-8"))
            self.assertIsNone(original["use_cache"])
            self.assertTrue(config["use_cache"])
            self.assertTrue(config["nested"]["use_cache"])
            self.assertEqual(preprocessor["image_processor_type"], "Qwen2VLImageProcessor")
            self.assertTrue((target / "tokenizer.json").is_symlink())

    def test_detects_null_use_cache_recursively(self) -> None:
        self.assertTrue(has_null_use_cache({"text_config": {"use_cache": None}}))
        self.assertFalse(has_null_use_cache({"text_config": {"use_cache": True}}))

    def test_audit_flags_unlabeled_and_degenerate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "summary.json").write_text(
                json.dumps({"accuracy": None, "labeled_total": 0, "total": 12}),
                encoding="utf-8",
            )
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for idx in range(12):
                    handle.write(json.dumps({"index": idx, "prediction": "D", "correct": None}) + "\n")

            audit = audit_eval_dir(out)
            self.assertEqual(audit["status"], "warn")
            self.assertIn("labeled_total is 0.", audit["warnings"])
            self.assertGreaterEqual(audit["top_prediction_fraction"], 0.95)

    def test_build_eval_command_omits_false_boolean_flags(self) -> None:
        command = build_eval_command(
            Path("/py"),
            Path("/out"),
            {"dataset": "d", "model": "m", "normalize_by_length": False, "task": "generation"},
            limit=5,
        )
        self.assertNotIn("--normalize-by-length", command)
        self.assertIn("--limit", command)
        self.assertEqual(command[-2:], ["--output-dir", "/out"])

    def test_build_eval_command_includes_true_trust_remote_code(self) -> None:
        command = build_eval_command(
            Path("/py"),
            Path("/out"),
            {"dataset": "d", "model": "m", "trust_remote_code": True},
            limit=0,
        )
        self.assertIn("--trust-remote-code", command)


if __name__ == "__main__":
    unittest.main()
