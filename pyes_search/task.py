"""The fill-in-the-blank task record that every data source converts to.

A task is one context (text, optionally with images), one template containing
``<BLANK:name>`` markers, a finite list of options per blank, and, for labelled
tasks, the correct option per blank.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

BLANK = re.compile(r"<BLANK:([^>]+)>")


class InvalidTask(ValueError):
    """A task violates the fill contract and must not be evaluated."""


@dataclass(frozen=True)
class FillTask:
    task_id: str
    source: str
    context: str
    template: str
    domains: Mapping[str, tuple[str, ...]]
    gold: Mapping[str, str] | None = None
    images: tuple[str, ...] = ()
    instructions: str = ""
    split: str = "train"
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.domains)

    def validate(self) -> FillTask:
        names = tuple(BLANK.findall(self.template))
        if not names:
            raise InvalidTask(f"{self.task_id}: template has no blanks")
        if set(names) != set(self.domains):
            raise InvalidTask(f"{self.task_id}: template blanks and domains differ")
        for name, values in self.domains.items():
            if len(values) < 2:
                raise InvalidTask(f"{self.task_id}/{name}: fewer than two candidates")
            if len(set(values)) != len(values):
                raise InvalidTask(f"{self.task_id}/{name}: duplicate candidates")
            if any(not isinstance(value, str) or not value for value in values):
                raise InvalidTask(f"{self.task_id}/{name}: candidates must be non-empty strings")
        if self.gold is not None:
            if set(self.gold) != set(self.domains):
                raise InvalidTask(f"{self.task_id}: gold does not cover every blank")
            for name, value in self.gold.items():
                if value not in self.domains[name]:
                    raise InvalidTask(f"{self.task_id}/{name}: gold is not a declared candidate")
        return self

    def gold_block(self) -> tuple[int, ...]:
        if self.gold is None:
            raise InvalidTask(f"{self.task_id}: unlabelled task has no gold block")
        return tuple(self.domains[name].index(self.gold[name]) for name in self.fields)

    def values(self, block: Sequence[int]) -> dict[str, str]:
        if len(block) != len(self.fields):
            raise ValueError("assignment width differs from the task's blank count")
        return {
            name: self.domains[name][int(i)] for name, i in zip(self.fields, block, strict=True)
        }

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "context": self.context,
            "template": self.template,
            "domains": {name: list(values) for name, values in self.domains.items()},
            "gold": None if self.gold is None else dict(self.gold),
            "images": list(self.images),
            "instructions": self.instructions,
            "split": self.split,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, row: Mapping) -> FillTask:
        return cls(
            task_id=str(row["task_id"]),
            source=str(row["source"]),
            context=str(row["context"]),
            template=str(row["template"]),
            domains={str(k): tuple(str(v) for v in vs) for k, vs in row["domains"].items()},
            gold=None
            if row.get("gold") is None
            else {str(k): str(v) for k, v in row["gold"].items()},
            images=tuple(row.get("images") or ()),
            instructions=str(row.get("instructions") or ""),
            split=str(row.get("split") or "train"),
            meta=dict(row.get("meta") or {}),
        ).validate()


def read_jsonl(path) -> list[FillTask]:
    with open(path, encoding="utf-8") as handle:
        return [FillTask.from_json(json.loads(line)) for line in handle if line.strip()]


def write_jsonl(path, tasks: Sequence[FillTask]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")
