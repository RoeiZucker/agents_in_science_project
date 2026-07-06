#!/usr/bin/env python3
"""Inspect a Hugging Face model for evaluation planning."""
from __future__ import annotations

import argparse
import json

from eval_agent_core import inspect_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_model(args.model)
    print(json.dumps(inspection.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
