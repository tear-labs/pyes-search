"""Shared toy tokenizer and task for the tests."""

from __future__ import annotations

from pyes_search.task import FillTask


def encode(text: str) -> list[int]:
    # Deterministic toy tokenizer: two characters per token, ids in [3, 61).
    return [3 + (sum(map(ord, text[i : i + 2])) % 58) for i in range(0, len(text), 2)]


def make_task(task_id="t", **overrides) -> FillTask:
    base = dict(
        task_id=task_id,
        source="unit",
        context="Ann has 3 apples and buys 2 more.",
        template='{"total": <BLANK:total>, "ok": <BLANK:ok>, "again": <BLANK:total>}',
        domains={"total": ("5", "6", "15"), "ok": ("yes", "no")},
        gold={"total": "5", "ok": "yes"},
    )
    base.update(overrides)
    return FillTask(**base).validate()
