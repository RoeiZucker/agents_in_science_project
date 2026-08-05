from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from delete_condition_datasets import cache_names, deletion_targets
from download_condition_datasets import selected_datasets, target_splits


class DatasetCacheScriptTests(unittest.TestCase):
    def test_selects_unique_datasets_from_conditions_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditions.csv"
            write_conditions(path)
            args = SimpleNamespace(
                conditions_csv=path,
                condition=["A"],
                dataset=[],
                limit_datasets=0,
            )

            self.assertEqual(selected_datasets(args), ["owner/one", "owner/two"])

    def test_explicit_dataset_filter_applies_after_csv_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditions.csv"
            write_conditions(path)
            args = SimpleNamespace(
                conditions_csv=path,
                condition=[],
                dataset=["owner/two"],
                limit_datasets=0,
            )

            self.assertEqual(selected_datasets(args), ["owner/two"])

    def test_explicit_splits_do_not_need_hub_lookup(self) -> None:
        args = SimpleNamespace(split=["validation", "test"], subset="", trust_remote_code=False)

        self.assertEqual(target_splits("owner/one", args), ["validation", "test"])

    def test_cache_names_cover_huggingface_dataset_cache_forms(self) -> None:
        names = cache_names("ImperialCollegeLondon/health_fact")

        self.assertIn("ImperialCollegeLondon___health_fact", names)
        self.assertIn("ImperialCollegeLondon_health_fact", names)
        self.assertIn("health_fact", names)

    def test_all_dataset_cache_targets_known_cache_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_cache = root / ".hf_datasets_cache"
            modules_cache = root / ".hf_cache" / "modules" / "datasets_modules" / "datasets"
            datasets_cache.mkdir(parents=True)
            modules_cache.mkdir(parents=True)
            args = SimpleNamespace(all_datasets_cache=True)

            targets = deletion_targets(root, args)

            self.assertEqual(targets, [datasets_cache, modules_cache])


def write_conditions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "query_dataset", "rank", "model_name"])
        writer.writerow(["A", "owner/one", "1", "model/a"])
        writer.writerow(["A", "owner/two", "1", "model/b"])
        writer.writerow(["B", "owner/one", "1", "model/c"])


if __name__ == "__main__":
    unittest.main()
