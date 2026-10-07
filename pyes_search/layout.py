"""The verification prompt and its fixed-width token layout.

The prompt is: the context, the declared options for every blank (placed before
the answer so they belong to the shared, cached prefix), the completed form
inside ``<filled>``, then one Yes/No verification question.

* The prompt is tokenized segment by segment (fixed text, then each blank on its
  own), so an option's tokens never depend on its neighbours.
* Every blank occupies ``W`` positions, the longest option's token count; shorter
  options are right-padded with ``pad_id`` inside the blank. All answers to a
  task therefore share one layout and differ only in the blank positions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from pyes_search.readout import YES_NO, Readout
from pyes_search.task import BLANK, FillTask

GLOBAL_VERIFIER = (
    "Verification: Is the entire completed fill correct for every blank, "
    "given the context and assignment instructions?\n"
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def prompt_segments(
    task: FillTask, verifier: str = GLOBAL_VERIFIER, readout: Readout = YES_NO
) -> list[tuple[str, str]]:
    """Return ``[("text", s) | ("blank", name), ...]`` for the verifier prompt.

    ``verifier`` is the question P(Yes) is conditioned on. It sits in the suffix,
    after the filled template, so the shared prefix (and its cache) is unchanged.
    """

    images = "".join(IMAGE_PLACEHOLDER for _ in task.images)
    schema = {name: list(values) for name, values in task.domains.items()}
    # Everything before the first blank (context, images, declared domains) is the
    # shared prefix that every relaxed/hard row of a task reuses (tree packing).
    prefix = (
        # The outer tag is only a delimiter; it is kept as-is because the reported numbers used it.
        f"<jev_joint>\n<context>\n{images}{task.context}\n</context>\n"
        f"<domains>{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}</domains>\n"
        "<filled>\n"
    )
    if task.instructions:
        prefix = f"{task.instructions}\n{prefix}"
    segments: list[tuple[str, str]] = [("text", prefix)]
    cursor = 0
    for match in BLANK.finditer(task.template):
        segments.append(("text", task.template[cursor : match.start()]))
        segments.append(("blank", match.group(1)))
        cursor = match.end()
    suffix = (
        task.template[cursor:]
        + "\n</filled>\n"
        + verifier
        + readout.cue
    )
    segments.append(("text", suffix))
    return [(kind, value) for kind, value in segments if kind == "blank" or value]



@dataclass(frozen=True)
class FieldLayout:
    name: str
    candidate_tokens: tuple[tuple[int, ...], ...]  # (C, W), right-padded with pad_id
    positions: tuple[tuple[int, ...], ...]  # (occurrences, W) absolute prompt positions

    @property
    def candidates(self) -> int:
        return len(self.candidate_tokens)

    @property
    def width(self) -> int:
        return len(self.candidate_tokens[0])


@dataclass(frozen=True)
class TaskLayout:
    task: FillTask
    ids: tuple[int, ...]  # blank positions hold candidate 0 (overwritten at read time)
    fields: tuple[FieldLayout, ...]
    mm_token_type_ids: tuple[int, ...] | None = None
    vision: dict | None = None  # processor tensors for the prefix images

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def prefix_len(self) -> int:
        """Tokens before the first blank: shared by every row of this task."""

        return min(occurrence[0] for field in self.fields for occurrence in field.positions)


class LayoutTooLong(ValueError):
    """The rendered prompt exceeds the configured token budget."""


def build_layout(
    task: FillTask,
    *,
    encode: Callable[[str], list[int]],
    pad_id: int,
    max_tokens: int | None = None,
    verifier: str = GLOBAL_VERIFIER,
    readout: Readout = YES_NO,
    encode_prefix: Callable[[FillTask, str], tuple[list[int], list[int] | None, dict | None]]
    | None = None,
) -> TaskLayout:
    """Tokenize ``task`` into a fixed-width layout.

    ``encode`` maps text to token ids without special tokens. ``encode_prefix``
    (required when the task has images) maps the first text segment to
    ``(ids, mm_token_type_ids, vision_tensors)`` using the multimodal processor.
    """

    segments = prompt_segments(task, verifier, readout)
    tokens: dict[str, tuple[tuple[int, ...], ...]] = {}
    for name, values in task.domains.items():
        encoded = [tuple(encode(value)) for value in values]
        if any(not item for item in encoded):
            raise ValueError(f"{task.task_id}/{name}: a candidate encodes to no tokens")
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"{task.task_id}/{name}: candidates collapse to one token sequence")
        width = max(len(item) for item in encoded)
        tokens[name] = tuple(item + (pad_id,) * (width - len(item)) for item in encoded)

    ids: list[int] = []
    mm: list[int] | None = None
    vision = None
    positions: dict[str, list[tuple[int, ...]]] = {name: [] for name in task.domains}
    for index, (kind, value) in enumerate(segments):
        if kind == "text":
            if index == 0 and task.images:
                if encode_prefix is None:
                    raise ValueError(
                        f"{task.task_id}: image task needs a multimodal prefix encoder"
                    )
                piece, mm_piece, vision = encode_prefix(task, value)
                mm = list(mm_piece) if mm_piece is not None else [0] * len(piece)
            else:
                piece = encode(value)
                if mm is not None:
                    mm.extend([0] * len(piece))
            ids.extend(piece)
        else:
            width = len(tokens[value][0])
            start = len(ids)
            positions[value].append(tuple(range(start, start + width)))
            ids.extend(tokens[value][0])
            if mm is not None:
                mm.extend([0] * width)
    if max_tokens is not None and len(ids) > max_tokens:
        raise LayoutTooLong(f"{task.task_id}: {len(ids)} tokens > {max_tokens}")
    fields = tuple(FieldLayout(name, tokens[name], tuple(positions[name])) for name in task.domains)
    return TaskLayout(
        task=task,
        ids=tuple(ids),
        fields=fields,
        mm_token_type_ids=None if mm is None else tuple(mm),
        vision=vision,
    )
