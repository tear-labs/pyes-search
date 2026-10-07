"""Convert puzzle jobs into fill-in-the-blank tasks.

A job has a ``state`` (the context), ``questions{name: {type, instructions,
criteria}}`` and ``expected`` answers; :func:`job_to_task` turns it into a
:class:`FillTask`. Every question is a ``choice``: its options are the criteria
keys, and each option's meaning is shown next to it in the form.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pyes_search.task import FillTask, InvalidTask

MAX_FIELDS = 32
MAX_CANDIDATES = 64


def state_text(state: object) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


MEANING_CHARS = 160


def _display(key: str, meaning: object) -> str:
    """Blank fill text: ``key: meaning`` (meaning on one line, capped) or the key."""

    text = " ".join(str(meaning or "").split())
    if not text or text == key:
        return key
    if len(text) > MEANING_CHARS:
        text = text[: MEANING_CHARS - 1].rstrip() + "…"
    if text.startswith((f"{key}:", f"{key} ")):
        return text  # the meaning already leads with its key ("3: high")
    return f"{key}: {text}"


def question_domain(question: Mapping) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    """Return ``(keys, fill texts, description lines)`` for one choice question.

    Fill texts carry the meaning of each option so the model judges content, not
    bare identifiers (``B`` -> ``B: the larger room``).
    """

    kind = question.get("type")
    criteria = question.get("criteria")
    lines: list[str] = []
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise InvalidTask("choice question without criteria")
        keys = tuple(str(key) for key in criteria)
        return keys, tuple(_display(str(k), m) for k, m in criteria.items()), lines
    raise InvalidTask(f"unsupported question type {kind!r}")


def _gold(kind: str, value: object, values: tuple[str, ...]) -> str:
    text = str(value)
    if text not in values:
        raise InvalidTask(f"gold {value!r} is not a declared candidate")
    return text


def job_to_task(
    *,
    task_id: str,
    source: str,
    state: object,
    questions: Mapping[str, Mapping],
    expected: Mapping[str, object] | None,
    images: Sequence[str] = (),
    split: str = "train",
    meta: Mapping[str, object] | None = None,
    max_candidates: int = MAX_CANDIDATES,
    max_fields: int = MAX_FIELDS,
) -> FillTask:
    if not questions:
        raise InvalidTask(f"{task_id}: no questions")
    if len(questions) > max_fields:
        raise InvalidTask(f"{task_id}: {len(questions)} fields exceeds {max_fields}")
    domains: dict[str, tuple[str, ...]] = {}
    display_keys: dict[str, dict[str, str]] = {}
    gold: dict[str, str] | None = None if expected is None else {}
    blocks: list[str] = []
    for name, question in questions.items():
        name = str(name)
        if any(ch in name for ch in "<>"):
            raise InvalidTask(f"{task_id}: unsafe field name {name!r}")
        keys, values, lines = question_domain(question)
        if len(values) > max_candidates:
            raise InvalidTask(f"{task_id}/{name}: {len(values)} candidates exceeds {max_candidates}")
        if len(set(values)) != len(values):
            values = keys  # fill texts collided; fall back to the (unique) keys
        domains[name] = values
        display_keys[name] = dict(zip(values, keys, strict=True))
        if gold is not None and expected is not None:
            if name not in expected:
                raise InvalidTask(f"{task_id}/{name}: no expected value")
            key = _gold(str(question.get("type")), expected[name], keys)
            gold[name] = values[keys.index(key)]
        header = f"- {name} ({question.get('type')}): {question.get('instructions') or ''}".rstrip()
        blocks.append("\n".join([header, *lines]))
    context = f"{state_text(state)}\n\nQuestions:\n" + "\n".join(blocks)
    template = "\n".join(f"{name}: <BLANK:{name}>" for name in domains)
    return FillTask(
        task_id=task_id,
        source=source,
        context=context,
        template=template,
        domains=domains,
        gold=gold,
        images=tuple(images),
        split=split,
        meta={**dict(meta or {}), "display_keys": display_keys},
    ).validate()


