"""Shared result-status rules for evaluation, reporting, and refinement."""
from __future__ import annotations

import math
from typing import Any


SUCCESS_STATUS = "ok"
PLANNED_STATUS = "planned"


def measured_success(status: Any, score: Any, labeled_total: Any) -> bool:
    """Return true only for a real, labeled measurement."""
    return (
        str(status).lower() == SUCCESS_STATUS
        and number(score) is not None
        and integer(labeled_total) > 0
    )


def result_outcome(status: Any, score: Any, labeled_total: Any) -> str:
    if str(status).lower() == PLANNED_STATUS:
        return PLANNED_STATUS
    return "success" if measured_success(status, score, labeled_total) else "failure"


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def integer(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
