#!/usr/bin/env python3
"""Audit evaluator outputs for missing labels, blank outputs, and degenerate predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_agent_core import audit_eval_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(audit_eval_dir(args.output_dir, args.max_rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
