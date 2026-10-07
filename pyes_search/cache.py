"""Prefix-cached P(Yes) energy: run each task's long prefix once, read many fills.

Every fill of a task shares the prompt up to the first blank (context, declared
options). Only the short suffix (the filled-in template plus the question)
differs. So:

* :func:`make_forest` packs each distinct task's prefix once (varlen, no padding);
* the model's ``prefix`` pass runs those tokens once and returns per-layer state;
* :func:`make_rows` right-pads the suffix rows, and the ``suffix`` pass reads them
  on the cached state.

A fill is given as per-blank weights over the declared options; one-hot weights
are ordinary token sequences.
"""

from __future__ import annotations


from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pyes_search.layout import TaskLayout
from pyes_search.readout import YES_NO, Readout


def task_key(layout: TaskLayout) -> str:
    return layout.task.task_id


@dataclass
class Forest:
    """Distinct task prefixes packed into one sequence."""

    tasks: tuple[TaskLayout, ...]
    index: dict[str, int]
    input_ids: torch.Tensor  # (1, Lp)
    cu: torch.Tensor  # (T + 1,)
    conv_index: torch.Tensor  # (Lp, K) causal-conv source per tap (Lp = zero row)
    vision: dict | None


@dataclass
class Rows:
    """Suffix rows (right-padded) plus, per blank, where its option tokens go."""

    layouts: tuple[TaskLayout, ...]
    forest: Forest
    row_task: torch.Tensor  # (R,)
    suffix_ids: torch.Tensor  # (R, S)
    suffix_len: torch.Tensor  # (R,)
    candidate_tokens: torch.Tensor  # (Slots, C, W)
    candidate_mask: torch.Tensor  # (Slots, C)
    slot_row: torch.Tensor  # (Slots,)
    row_slots: tuple[tuple[int, int], ...]
    src_slot: torch.Tensor  # (P,)
    src_width: torch.Tensor  # (P,)
    dst_row: torch.Tensor  # (P,)
    dst_position: torch.Tensor  # (P,) position inside the suffix

    @property
    def rows(self) -> int:
        return len(self.layouts)

    @property
    def slots(self) -> int:
        return self.candidate_tokens.shape[0]

    def one_hot(self, blocks: Sequence[Sequence[int]]) -> torch.Tensor:
        if len(blocks) != self.rows:
            raise ValueError("one block per row is required")
        weights = torch.zeros(self.candidate_mask.shape, device=self.candidate_mask.device)
        for (start, stop), block in zip(self.row_slots, blocks, strict=True):
            if len(block) != stop - start:
                raise ValueError("block width differs from the row's blank count")
            for slot, choice in zip(range(start, stop), block, strict=True):
                if not bool(self.candidate_mask[slot, int(choice)]):
                    raise ValueError("block selects an undeclared candidate")
                weights[slot, int(choice)] = 1.0
        return weights


def make_forest(layouts: Sequence[TaskLayout], *, device, conv_kernel: int = 4) -> Forest:
    tasks: list[TaskLayout] = []
    index: dict[str, int] = {}
    for layout in layouts:
        if task_key(layout) not in index:
            index[task_key(layout)] = len(tasks)
            tasks.append(layout)
    ids: list[int] = []
    bounds = []
    for layout in tasks:
        start = len(ids)
        ids.extend(layout.ids[: layout.prefix_len])
        bounds.append((start, len(ids)))
    length = len(ids)
    conv = torch.full((length, conv_kernel), length, dtype=torch.long)
    for start, stop in bounds:
        offsets = torch.arange(stop - start)
        for j in range(conv_kernel):
            shift = conv_kernel - 1 - j
            inside = offsets >= shift
            src = torch.full((stop - start,), length, dtype=torch.long)
            src[inside] = start + offsets[inside] - shift
            conv[start:stop, j] = src
    pixels = [x.vision["pixel_values"] for x in tasks if x.vision is not None]
    grids = [x.vision["image_grid_thw"] for x in tasks if x.vision is not None]
    vision = None
    if pixels:
        vision = {
            "pixel_values": torch.cat(pixels).to(device),
            "image_grid_thw": torch.cat(grids).to(device),
        }
    return Forest(
        tasks=tuple(tasks),
        index=index,
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        cu=torch.tensor([0] + [stop for _, stop in bounds], dtype=torch.long, device=device),
        conv_index=conv.to(device),
        vision=vision,
    )


def make_rows(layouts: Sequence[TaskLayout], forest: Forest, *, device, pad_id: int = 0) -> Rows:
    layouts = tuple(layouts)
    lengths = [len(x) - x.prefix_len for x in layouts]
    width = max(lengths)
    suffix = torch.full((len(layouts), width), pad_id, dtype=torch.long)
    for r, layout in enumerate(layouts):
        suffix[r, : lengths[r]] = torch.tensor(layout.ids[layout.prefix_len :], dtype=torch.long)
    slot_fields = [(r, f) for r, layout in enumerate(layouts) for f in layout.fields]
    max_c = max(f.candidates for _, f in slot_fields)
    max_w = max(f.width for _, f in slot_fields)
    candidate_tokens = torch.zeros((len(slot_fields), max_c, max_w), dtype=torch.long)
    candidate_mask = torch.zeros((len(slot_fields), max_c), dtype=torch.bool)
    slot_row, src_slot, src_width, dst_row, dst_position = [], [], [], [], []
    row_slots = []
    slot = 0
    for r, layout in enumerate(layouts):
        start = slot
        for field in layout.fields:
            candidate_tokens[slot, : field.candidates, : field.width] = torch.tensor(
                field.candidate_tokens, dtype=torch.long
            )
            candidate_mask[slot, : field.candidates] = True
            slot_row.append(r)
            for occurrence in field.positions:
                for w, position in enumerate(occurrence):
                    src_slot.append(slot)
                    src_width.append(w)
                    dst_row.append(r)
                    dst_position.append(position - layout.prefix_len)
            slot += 1
        row_slots.append((start, slot))

    def tensor(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    return Rows(
        layouts=layouts,
        forest=forest,
        row_task=tensor([forest.index[task_key(x)] for x in layouts]),
        suffix_ids=suffix.to(device),
        suffix_len=tensor(lengths),
        candidate_tokens=candidate_tokens.to(device),
        candidate_mask=candidate_mask.to(device),
        slot_row=tensor(slot_row),
        row_slots=tuple(row_slots),
        src_slot=tensor(src_slot),
        src_width=tensor(src_width),
        dst_row=tensor(dst_row),
        dst_position=tensor(dst_position),
    )


class CachedVerifier:
    """``E(row) = softplus(logit_No - logit_Yes)`` read after the cached prefix.

    Subclasses implement ``prefix(forest) -> state`` and
    ``suffix_hidden(state, rows, embeds) -> (R, d)`` (hidden at each row's last
    real token, after the final norm) and provide ``embedding`` and
    ``answer_rows()``.
    """

    embedding: torch.nn.Embedding
    #: Which answer tokens the energy is read through (see :mod:`pyes_search.readout`).
    readout: Readout = YES_NO

    def answer_rows(self) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def prefix(self, forest: Forest):  # pragma: no cover - interface
        raise NotImplementedError

    def suffix_hidden(self, state, rows: Rows, embeds: torch.Tensor):
        raise NotImplementedError  # pragma: no cover

    def suffix_embeds(self, rows: Rows, weights: torch.Tensor) -> torch.Tensor:
        # Gather only the rows' unique token ids from the (large-vocabulary) table.
        ids = torch.cat([rows.suffix_ids.reshape(-1), rows.candidate_tokens.reshape(-1)])
        unique, inverse = torch.unique(ids, return_inverse=True)
        table = self.lookup(unique)
        split = rows.suffix_ids.numel()
        embeds = table[inverse[:split].view(rows.suffix_ids.shape)]
        cand = table[inverse[split:].view(rows.candidate_tokens.shape)]  # (Slots, C, W, d)
        w = weights.masked_fill(~rows.candidate_mask, 0.0).to(cand.dtype)
        mixed = torch.einsum("sc,scwd->swd", w, cand)[rows.src_slot, rows.src_width]
        return embeds.index_put((rows.dst_row, rows.dst_position), mixed.to(embeds.dtype))

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """Embedding rows for ``ids``."""

        return self.embedding.weight.index_select(0, ids)

    def answer_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(R, K)`` readout-token logits from final-normed hidden states."""

        dtype = torch.promote_types(hidden.dtype, torch.float32)
        return hidden.to(dtype) @ self.answer_rows().to(dtype).T

    def energies(self, state, rows: Rows, weights: torch.Tensor) -> torch.Tensor:
        """E = -log P(Yes) (or the chosen readout) for each row's fill."""

        if weights.shape != rows.candidate_mask.shape:
            raise ValueError("weights do not match the row slots")
        hidden = self.suffix_hidden(state, rows, self.suffix_embeds(rows, weights))
        return self.readout.energy(self.answer_logits(hidden))

    def bind(self, state):
        """An ``(rows, weights) -> energies`` callable over one cached prefix."""

        def energy(rows: Rows, weights: torch.Tensor) -> torch.Tensor:
            return self.energies(state, rows, weights)

        return energy
