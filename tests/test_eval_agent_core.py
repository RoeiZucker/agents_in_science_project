from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval_agent_core import (
    DatasetInspection,
    ModelInspection,
    apply_known_dataset_contract,
    audit_eval_dir,
    build_eval_command,
    build_plan,
    choose_dataset_subset,
    choose_evaluation_protocol,
    choose_first,
    dataset_value,
    has_null_use_cache,
    infer_label_map,
    infer_model_type_from_metadata,
    inspect_dataset,
    known_string_label_map,
    load_dataset_compatible,
    needs_label_semantics_lookup,
    requested_dataset_column,
    sanitize_model_dir,
    select_split,
)


class FakeDataset:
    def __init__(self, rows, features):
        self.rows = rows
        self.features = features
        self.column_names = list(features)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def unique(self, column):
        return list(dict.fromkeys(row[column] for row in self.rows))


class FakeLabelFeature:
    def __init__(self, names):
        self.names = names


class EvalAgentCoreTests(unittest.TestCase):

    def test_inspection_flattens_dialogue_before_gold_detection(self) -> None:
        cases = [
            (
                "google-research-datasets/taskmaster1",
                [{"instruction_id": "book", "utterances": [
                    {"speaker": "USER", "text": "Find a film"},
                    {"speaker": "ASSISTANT", "text": "Which city?"},
                ]}],
            ),
            (
                "ParlAI/blended_skill_talk",
                [{"personas": [], "context": "", "additional_context": "",
                  "previous_utterance": ["Hello"], "free_messages": ["Hi"]}],
            ),
            (
                "facebook/empathetic_dialogues",
                [
                    {"conv_id": "c1", "context": "joy", "prompt": "Won",
                     "utterance": "I won!"},
                    {"conv_id": "c1", "context": "joy", "prompt": "Won",
                     "utterance": "Congratulations!"},
                ],
            ),
        ]
        for dataset_name, rows in cases:
            raw = FakeDataset(rows, {key: object() for key in rows[0]})
            with self.subTest(dataset=dataset_name), patch(
                "eval_agent_core.get_dataset_split_names", return_value=["test"]
            ), patch("eval_agent_core.load_dataset_compatible", return_value=raw):
                inspected = inspect_dataset(
                    dataset_name,
                    split="test",
                    question_column="_dialogue_prompt",
                    answer_column="_dialogue_response",
                )

            self.assertTrue(inspected.has_gold)
            self.assertGreater(inspected.labeled_total_sample, 0)
            self.assertEqual(inspected.question_column, "_dialogue_prompt")
            self.assertEqual(inspected.answer_column, "_dialogue_response")

    def test_sequence_classifier_head_overrides_sentence_transformer_metadata(self) -> None:
        model_type = infer_model_type_from_metadata(
            "cross-encoder/nli-deberta-v3-large",
            "deberta-v2",
            ["DebertaV2ForSequenceClassification"],
            pipeline_tag="sentence-similarity",
            library_name="sentence-transformers",
        )

        self.assertEqual(model_type, "sequence_classifier")

    def test_dataset_subset_preserves_declared_default_config(self) -> None:
        with patch(
            "eval_agent_core.get_dataset_config_names",
            return_value=["default", "labels"],
        ):
            selected = choose_dataset_subset("example/dataset", None, False, [])

        self.assertEqual(selected, "default")

    def test_missing_requested_split_falls_back_to_labeled_train(self) -> None:
        ds = FakeDataset(
            [{"text": "example", "label": 1}],
            {"text": object(), "label": object()},
        )
        with patch("eval_agent_core.load_dataset_compatible", return_value=ds):
            selected = select_split(
                "example/dataset",
                None,
                "test",
                ["train"],
                20,
                "label",
            )

        self.assertEqual(selected, "train")

    def test_unlabeled_requested_split_falls_back_to_labeled_validation(self) -> None:
        unlabeled = FakeDataset(
            [{"text": "example", "target": ""}],
            {"text": object(), "target": object()},
        )
        labeled = FakeDataset(
            [{"text": "example", "target": "summary"}],
            {"text": object(), "target": object()},
        )
        with patch(
            "eval_agent_core.load_dataset_compatible",
            side_effect=[unlabeled, labeled],
        ):
            selected = select_split(
                "example/dataset",
                None,
                "test",
                ["train", "test", "validation"],
                20,
                "target",
            )

        self.assertEqual(selected, "validation")

    def test_full_plan_uses_reproducible_random_sample_cap(self) -> None:
        plan = build_plan(
            dataset_inspection({"0": "no", "1": "yes"}),
            model_inspection("causal_lm", []),
            Path("/project"),
            Path("/python"),
            smoke_limit=3,
            output_root=Path("/output"),
            full_limit=1000,
            seed=7,
        )

        self.assertIn("--limit", plan.full_command)
        self.assertEqual(
            plan.full_command[plan.full_command.index("--limit") + 1],
            "1000",
        )
        self.assertEqual(
            plan.full_command[plan.full_command.index("--seed") + 1],
            "7",
        )
        self.assertTrue(any("randomly selected" in note for note in plan.notes))

    def test_qwen3_tagged_plan_disables_thinking(self) -> None:
        model = model_inspection("causal_lm", [])
        model.model = "Qwen/Qwen3-14B"
        model.config_model_type = "qwen3"

        plan = build_plan(
            dataset_inspection({"0": "false", "1": "true"}),
            model,
            Path("/project"),
            Path("/python"),
            smoke_limit=3,
        )

        self.assertTrue(plan.args["disable_thinking"])
        self.assertIn("--disable-thinking", plan.smoke_command)
        self.assertTrue(
            any("Disabled model thinking" in note for note in plan.notes)
        )

    def test_infers_label_map_from_classlabel_names(self) -> None:
        ds = FakeDataset(
            [{"text": "claim", "label": 1}],
            {"text": object(), "label": FakeLabelFeature(["false", "true", "unproven"])},
        )
        self.assertEqual(
            infer_label_map(ds, ds.column_names, "label", 5),
            {"0": "false", "1": "true", "2": "unproven"},
        )

    def test_detects_opaque_numeric_labels_without_metadata(self) -> None:
        ds = FakeDataset(
            [{"text": "claim", "answer": 3}],
            {"text": object(), "answer": object()},
        )
        self.assertTrue(needs_label_semantics_lookup(ds, "answer", 5))

    def test_recognizes_claim_as_an_input_text_column(self) -> None:
        self.assertEqual(
            choose_first(["claim_id", "claim", "label"], ("question", "text", "claim"), "question"),
            "claim",
        )

    def test_recognizes_cfpb_columns(self) -> None:
        columns = ["Date received", "Product", "Consumer complaint narrative"]
        self.assertEqual(
            choose_first(columns, ("text", "Consumer complaint narrative")),
            "Consumer complaint narrative",
        )
        self.assertEqual(
            requested_dataset_column("Complaint Text", columns), ""
        )

    def test_reads_nested_values_through_lists(self) -> None:
        row = {
            "utterances": [
                {"segments": [{"text": "book"}, {"text": "flight"}]},
                {"segments": [{"text": "tomorrow"}]},
            ]
        }

        self.assertEqual(
            dataset_value(row, "utterances.segments.text"),
            ["book", "flight", "tomorrow"],
        )

    def test_cfpb_uses_raw_csv_fallback(self) -> None:
        expected = object()
        with patch(
            "eval_agent_core.load_dataset",
            side_effect=[RuntimeError("stale dataset script"), expected],
        ) as loader:
            actual = load_dataset_compatible(
                "CFPB/consumer-finance-complaints",
                split="train",
                trust_remote_code=True,
            )
        self.assertIs(actual, expected)
        fallback = loader.call_args_list[1]
        self.assertEqual(fallback.kwargs["path"], "csv")
        self.assertIn(
            "complaints.csv.zip", fallback.kwargs["data_files"]["train"]
        )
        self.assertNotIn("trust_remote_code", fallback.kwargs)

    def test_cfpb_string_products_become_finite_labels(self) -> None:
        ds = FakeDataset(
            [{"Product": "Mortgage"}, {"Product": "Credit card"}],
            {"Product": object()},
        )
        self.assertEqual(
            known_string_label_map(
                "CFPB/consumer-finance-complaints", ds, "Product"
            ),
            {"Credit card": "Credit card", "Mortgage": "Mortgage"},
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
        self.assertEqual(
            infer_model_type_from_metadata(
                "sentence-transformers/sentence-t5-base",
                "t5",
                ["T5EncoderModel"],
                pipeline_tag="sentence-similarity",
                library_name="sentence-transformers",
            ),
            "sentence_encoder",
        )
        self.assertEqual(
            infer_model_type_from_metadata("bartowski/Qwen2.5-7B-Instruct-GGUF", None, []),
            "gguf",
        )
        self.assertEqual(
            infer_model_type_from_metadata(
                "michiyasunaga/BioLinkBERT-large",
                "bert",
                ["BertModel"],
                pipeline_tag="text-classification",
            ),
            "sentence_encoder",
        )



    def test_infers_new_adapter_model_families(self) -> None:
        self.assertEqual(
            infer_model_type_from_metadata(
                "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                "roberta",
                ["RobertaForMaskedLM"],
                pipeline_tag="fill-mask",
            ),
            "masked_lm",
        )
        self.assertEqual(
            infer_model_type_from_metadata(
                "OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M",
                "bert",
                ["BertForTokenClassification"],
                pipeline_tag="token-classification",
            ),
            "token_classifier",
        )

    def test_question_answering_tag_keeps_causal_generator(self) -> None:
        model_type = infer_model_type_from_metadata(
            "HPAI-BSC/Llama3-Aloe-8B-Alpha",
            "llama",
            ["LlamaForCausalLM"],
            pipeline_tag="question-answering",
        )
        self.assertEqual(model_type, "causal_lm")


    def test_question_answering_head_uses_extractive_adapter(self) -> None:
        model_type = infer_model_type_from_metadata(
            "deepset/roberta-base-squad2",
            "roberta",
            ["RobertaForQuestionAnswering"],
            pipeline_tag="question-answering",
        )
        self.assertEqual(model_type, "extractive_qa")


    def test_standard_llava_next_uses_processor_adapter(self) -> None:
        model_type = infer_model_type_from_metadata(
            "ibm-granite/granite-vision-3.3-2b",
            "llava_next",
            ["LlavaNextForConditionalGeneration"],
        )
        self.assertEqual(model_type, "vlm_processor")


    def test_ocrflux_qwen_vl_uses_processor_adapter(self) -> None:
        model_type = infer_model_type_from_metadata(
            "ChatDOC/OCRFlux-3B",
            "qwen2_5_vl",
            ["Qwen2_5_VLForConditionalGeneration"],
        )
        self.assertEqual(model_type, "vlm_processor")


    def test_sentence_encoder_uses_label_similarity(self) -> None:
        dataset = dataset_inspection(
            {"0": "business", "1": "sport"}, task="classification"
        )
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("sentence_encoder", []), allow_label_scores=True
        )
        self.assertEqual(protocol, "embedding_label_similarity")
        self.assertEqual(metric, "accuracy")

    def test_sentence_encoder_rejects_free_text(self) -> None:
        dataset = dataset_inspection({}, task="qa")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("sentence_encoder", [])
        )
        self.assertEqual(protocol, "unsupported")
        self.assertIsNone(metric)

    def test_masked_lm_uses_label_likelihood(self) -> None:
        dataset = dataset_inspection(
            {"0": "not related", "1": "related"}, task="classification"
        )
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("masked_lm", []), allow_label_scores=True
        )
        self.assertEqual(protocol, "masked_label_likelihood")
        self.assertEqual(metric, "accuracy")

    def test_masked_lm_scores_multiple_choices(self) -> None:
        dataset = dataset_inspection({}, task="multiple_choice")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset,
            model_inspection("masked_lm", []),
            allow_label_scores=True,
        )
        self.assertEqual(protocol, "masked_choice_likelihood")
        self.assertEqual(metric, "accuracy")

    def test_token_classifier_uses_entity_set_f1(self) -> None:
        dataset = dataset_inspection({}, task="token_classification")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("token_classifier", []), allow_label_scores=True
        )
        self.assertEqual(protocol, "token_entity_set_f1")
        self.assertEqual(metric, "set_f1")

    def test_token_classifier_rejects_document_classification(self) -> None:
        dataset = dataset_inspection(
            {"0": "negative", "1": "positive"}, task="classification"
        )
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("token_classifier", [])
        )
        self.assertEqual(protocol, "unsupported")
        self.assertIsNone(metric)


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
            self.assertEqual(audit["status"], "unmeasured")
            self.assertIn("No labeled score was produced.", audit["issues"])

    def test_audit_reports_tag_parse_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "summary.json").write_text(
                json.dumps({
                    "accuracy": 0.5,
                    "labeled_total": 2,
                    "total": 2,
                    "tag_parse_failures": 1,
                }),
                encoding="utf-8",
            )
            (out / "predictions.jsonl").write_text(
                '{"prediction":"true","correct":true}\n'
                '{"prediction":"","correct":false}\n',
                encoding="utf-8",
            )

            audit = audit_eval_dir(out)

        self.assertEqual(audit["status"], "ok")
        self.assertIn(
            "Tagged-answer parsing failed for 1 examples.",
            audit["warnings"],
        )

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

    def test_protocol_rejects_generic_classifier_labels(self) -> None:
        dataset = dataset_inspection(label_map={"0": "negative", "1": "positive"})
        model = model_inspection("sequence_classifier", ["LABEL_0", "LABEL_1"])

        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model, allow_label_scores=True
        )

        self.assertEqual(protocol, "unsupported")
        self.assertIsNone(metric)

    def test_protocol_automatically_uses_label_scores_for_classifier(self) -> None:
        dataset = dataset_inspection(label_map={"0": "negative", "1": "positive"})
        model = model_inspection("sequence_classifier", ["contradiction", "neutral", "entailment"])

        protocol, metric, _ = choose_evaluation_protocol(dataset, model)

        self.assertEqual(protocol, "zero_shot_nli_accuracy")
        self.assertEqual(metric, "accuracy")

    def test_protocol_accepts_nli_classifier_without_index_remapping(self) -> None:
        dataset = dataset_inspection(label_map={"0": "negative", "1": "positive"})
        model = model_inspection("sequence_classifier", ["contradiction", "neutral", "entailment"])

        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model, allow_label_scores=True
        )

        self.assertEqual(protocol, "zero_shot_nli_accuracy")
        self.assertEqual(metric, "accuracy")

    def test_protocol_uses_tagged_answers_for_generative_classification(self) -> None:
        dataset = dataset_inspection(label_map={"0": "negative", "1": "positive"})
        model = model_inspection("causal_lm", [])

        protocol, metric, _ = choose_evaluation_protocol(dataset, model)

        self.assertEqual(protocol, "tagged_label_generation_accuracy")
        self.assertEqual(metric, "accuracy")

    def test_protocol_uses_tags_for_multiple_choice_by_default(self) -> None:
        dataset = dataset_inspection(label_map={}, task="multiple_choice")
        model = model_inspection("causal_lm", [])

        protocol, metric, _ = choose_evaluation_protocol(dataset, model)

        self.assertEqual(protocol, "tagged_multiple_choice_accuracy")
        self.assertEqual(metric, "accuracy")

    def test_multiple_choice_scores_require_opt_in(self) -> None:
        dataset = dataset_inspection(label_map={}, task="multiple_choice")
        model = model_inspection("causal_lm", [])

        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model, allow_label_scores=True
        )

        self.assertEqual(protocol, "multiple_choice_accuracy")
        self.assertEqual(metric, "accuracy")

    def test_protocol_uses_tagged_answers_for_free_text(self) -> None:
        dataset = dataset_inspection(label_map={})
        model = model_inspection("causal_lm", [])

        protocol, metric, _ = choose_evaluation_protocol(dataset, model)

        self.assertEqual(protocol, "tagged_answer_exact_match")
        self.assertEqual(metric, "exact_match")

    def test_protocol_uses_qa_f1_for_qa(self) -> None:
        dataset = dataset_inspection(label_map={}, task="qa")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("causal_lm", [])
        )

        self.assertEqual(protocol, "tagged_answer_exact_match")
        self.assertEqual(metric, "qa_f1")

    def test_protocol_uses_extractive_qa_when_context_is_available(self) -> None:
        dataset = dataset_inspection(label_map={}, task="qa", context=True)
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("extractive_qa", [])
        )
        self.assertEqual(protocol, "extractive_qa")
        self.assertEqual(metric, "qa_f1")


    def test_protocol_rejects_extractive_qa_without_context(self) -> None:
        dataset = dataset_inspection(label_map={}, task="qa")
        protocol, metric, reason = choose_evaluation_protocol(
            dataset, model_inspection("extractive_qa", [])
        )
        self.assertEqual(protocol, "unsupported")
        self.assertIsNone(metric)
        self.assertIn("context", reason)


    def test_protocol_uses_numeric_match_for_numeric_qa(self) -> None:
        dataset = dataset_inspection(label_map={}, task="numeric_qa")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("causal_lm", [])
        )

        self.assertEqual(protocol, "tagged_answer_exact_match")
        self.assertEqual(metric, "numeric_match")

        plan = build_plan(
            dataset, model_inspection("causal_lm", []),
            Path("/project"), Path("/python"), smoke_limit=3,
        )
        self.assertEqual(plan.args["max_new_tokens"], 512)

    def test_protocol_uses_rouge_l_for_summarization(self) -> None:
        dataset = dataset_inspection(label_map={}, task="summarization")
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("seq2seq_lm", [])
        )

        self.assertEqual(protocol, "tagged_answer_exact_match")
        self.assertEqual(metric, "rouge_l")

    def test_protocol_uses_set_f1_for_multilabel_generation(self) -> None:
        dataset = dataset_inspection(
            label_map={"0": "joy", "1": "anger"},
            task="multilabel_classification",
        )
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("causal_lm", [])
        )

        self.assertEqual(protocol, "tagged_set_generation")
        self.assertEqual(metric, "set_f1")

    def test_protocol_requires_declared_multilabel_classifier(self) -> None:
        dataset = dataset_inspection(
            label_map={"0": "joy", "1": "anger"},
            task="multilabel_classification",
        )
        protocol, _, _ = choose_evaluation_protocol(
            dataset, model_inspection("sequence_classifier", ["joy", "anger"]),
            allow_label_scores=True,
        )

        self.assertEqual(protocol, "unsupported")

    def test_protocol_accepts_nli_for_multilabel_dataset(self) -> None:
        dataset = dataset_inspection(
            label_map={"0": "joy", "1": "anger"},
            task="multilabel_classification",
        )
        model = model_inspection(
            "sequence_classifier", ["contradiction", "neutral", "entailment"]
        )
        protocol, metric, _ = choose_evaluation_protocol(dataset, model)
        self.assertEqual(protocol, "zero_shot_nli_multilabel")
        self.assertEqual(metric, "set_f1")


    def test_multiwoz_contract_recovers_nested_active_intents(self) -> None:
        dataset = dataset_inspection(label_map={})
        dataset.dataset = "tuetschek/multi_woz_v22"
        dataset.has_gold = False
        recovered = apply_known_dataset_contract(dataset)
        self.assertTrue(recovered.has_gold)
        self.assertEqual(recovered.task, "multilabel_classification")
        self.assertEqual(
            recovered.answer_column, "turns.frames.state.active_intent"
        )
        self.assertIn("book_hotel", recovered.label_map)
        self.assertIn("active user intent", recovered.prompt_template)
        self.assertNotIn("Return the active intent labels", recovered.prompt_template)

        plan = build_plan(
            recovered, model_inspection("causal_lm", []),
            Path("/project"), Path("/python"), smoke_limit=3,
        )
        self.assertEqual(plan.args["max_new_tokens"], 96)


    def test_protocol_accepts_declared_multilabel_classifier(self) -> None:
        dataset = dataset_inspection(
            label_map={"0": "joy", "1": "anger"},
            task="multilabel_classification",
        )
        model = model_inspection(
            "sequence_classifier",
            ["joy", "anger"],
            problem_type="multi_label_classification",
        )

        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model, allow_label_scores=True
        )

        self.assertEqual(protocol, "multilabel_classifier_set_f1")
        self.assertEqual(metric, "set_f1")

    def test_protocol_rejects_method_that_does_not_match_task(self) -> None:
        dataset = dataset_inspection(
            label_map={}, task="qa", evaluation_method="rouge_l"
        )
        protocol, metric, reason = choose_evaluation_protocol(
            dataset, model_inspection("causal_lm", [])
        )

        self.assertEqual(protocol, "unsupported")
        self.assertIsNone(metric)
        self.assertIn("incompatible", reason)

    def test_protocol_rejects_text_model_for_image_dataset(self) -> None:
        dataset = dataset_inspection(label_map={"0": "forest"}, image=True)
        model = model_inspection("seq2seq_lm", [])

        protocol, _, _ = choose_evaluation_protocol(dataset, model)

        self.assertEqual(protocol, "unsupported")

    def test_protocol_accepts_zero_shot_image_model(self) -> None:
        dataset = dataset_inspection(
            label_map={"0": "forest", "1": "river"},
            image=True,
            task="classification",
        )
        protocol, metric, _ = choose_evaluation_protocol(
            dataset, model_inspection("zero_shot_image", [])
        )

        self.assertEqual(protocol, "zero_shot_image_accuracy")
        self.assertEqual(metric, "accuracy")


def dataset_inspection(
    label_map: dict[str, str],
    image: bool = False,
    task: str = "generation",
    evaluation_method: str = "auto",
    context: bool = False,
) -> DatasetInspection:
    columns = ["text", "label"] + (["context"] if context else []) + (["image"] if image else [])
    return DatasetInspection(
        dataset="dataset/one",
        subset=None,
        splits=["test"],
        selected_split="test",
        total=2,
        columns=columns,
        features={},
        question_column="text",
        context_column="context" if context else "",
        answer_column="label",
        choices_column="",
        image_column="image" if image else "",
        task=task,
        labeled_total_sample=2,
        inspected_sample=2,
        has_gold=True,
        label_map=label_map,
        evaluation_method=evaluation_method,
    )


def model_inspection(
    model_type: str, labels: list[str], problem_type: str | None = None
) -> ModelInspection:
    return ModelInspection(
        model="model/a",
        model_type=model_type,
        config_model_type=None,
        architectures=[],
        pipeline_tag=None,
        library_name=None,
        needs_eager_attention=False,
        needs_sanitized_config=False,
        classifier_labels=labels,
        classifier_problem_type=problem_type,
    )


if __name__ == "__main__":
    unittest.main()
