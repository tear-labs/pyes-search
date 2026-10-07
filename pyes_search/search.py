"""Test-time search over a frozen model's P(Yes).

Every method starts from the same one-read answer and only ever scores real,
complete fills, so every step is checked by the model:

* ``pyes``    one read per option (other blanks left uniform), then the best of a
              small beam of complete fills, read exactly;
* ``random``  rounds of random single-blank switches of the current best fill,
              each read exactly; move when the energy falls;
* ``bp``      belief propagation (BP) steps from exact single and pair reads (see ``bp_step``);
* ``local``   random single and two-blank changes (see :mod:`pyes_search.local`);
* ``flips``   coupled flips: flip a blank, read how the others react, follow up;
* ``bp+local`` one BP step, then ``local``.

Energy: E = -log P(Yes | filled template). Lower is better.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pyes_search.bp import HyperFactor, exact_factor_map, loopy_min_sum
from pyes_search.bp.solver import ordered_statistics_candidates
from pyes_search.cache import CachedVerifier, make_forest, make_rows
from pyes_search.layout import TaskLayout
from pyes_search.local import LocalSearch, run_local


@dataclass(frozen=True)
class Decision:
    task_id: str
    block: tuple[int, ...]  # chosen option index per blank
    energy: float  # -log P(Yes) of that fill
    shortlist: tuple[tuple[tuple[int, ...], float], ...]  # every fill read, with its energy


# ------------------------------------------------------------------------- reads
@torch.no_grad()
def hard_energies(energy, forest, pairs, *, batch_rows: int, device, pad_id: int) -> list[float]:
    """Energies of complete fills: (layout, block) pairs read on a shared prefix cache."""

    out: list[float] = []
    for start in range(0, len(pairs), batch_rows):
        chunk = pairs[start : start + batch_rows]
        real = len(chunk)
        # Fixed row count (repeat the last pair) so compiled kernels see one shape.
        chunk = chunk + [chunk[-1]] * (batch_rows - real)
        hard = make_rows([x for x, _ in chunk], forest, device=device, pad_id=pad_id)
        values = energy(hard, hard.one_hot([b for _, b in chunk]))
        out.extend(float(e) for e in values[:real].float().cpu())
    return out


@torch.no_grad()
def one_read_scores(energy, rows, *, batch_rows: int) -> torch.Tensor:
    """z[s, c] = -E(blank s = option c, every other blank a uniform mix of its options)."""

    mask = rows.candidate_mask
    uniform = mask.float() / mask.float().sum(-1, keepdim=True)
    z = torch.zeros(mask.shape, device=mask.device)
    jobs = [(s, c) for s in range(rows.slots) for c in range(int(mask[s].sum()))]
    slot_row = rows.slot_row.tolist()
    for start in range(0, len(jobs), batch_rows):
        chunk = jobs[start : start + batch_rows]
        sub = make_rows([rows.layouts[slot_row[s]] for s, _ in chunk], rows.forest,
                        device=mask.device)
        weights = []
        for s, c in chunk:
            a, b = rows.row_slots[slot_row[s]]
            w = uniform[a:b].clone()
            w[s - a] = 0.0
            w[s - a, c] = 1.0
            weights.append(w)
        stacked = torch.cat(weights)
        width = sub.candidate_mask.shape[1]
        if stacked.shape[1] >= width:
            stacked = stacked[:, :width]
        else:
            stacked = torch.nn.functional.pad(stacked, (0, width - stacked.shape[1]))
        for (s, c), value in zip(chunk, energy(sub, stacked).tolist(), strict=True):
            z[s, c] = -value
    return z.masked_fill(~mask, 0.0)


def shortlist(marginals: Sequence[Sequence[float]], *, top_k: int, cap: int) -> list[tuple[int, ...]]:
    """Beam over blanks by summed log belief; the per-blank argmax comes first."""

    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for row in marginals:
        order = sorted(range(len(row)), key=lambda c: -row[c])[:top_k]
        expanded = [(score + math.log(max(row[c], 1e-30)), block + (c,))
                    for score, block in beams for c in order]
        expanded.sort(key=lambda item: -item[0])
        beams = expanded[:cap]
    return [block for _, block in beams]


# ------------------------------------------------------------------------ search
def search(
    verifier: CachedVerifier,
    layouts: Sequence[TaskLayout],
    *,
    method: str = "pyes",
    rounds: int = 1,
    moves: int = 4,
    bp_k: int = 2,
    bp_temperature: float = 0.0,
    pad_id: int,
    device,
    top_k: int = 3,
    cap: int = 16,
    batch_rows: int = 64,
) -> list[Decision]:
    """Decide every task in ``layouts`` with one prefix pass per task.

    ``method``: ``"pyes"`` (one read), ``"random"`` (``rounds`` rounds of ``moves``
    random single-blank switches), ``"bp"`` (``rounds`` BP steps, pair reads
    over each blank's ``bp_k`` best options, BP at ``bp_temperature``), ``"local"``
    (``rounds`` rounds of ``moves`` random single- and two-blank changes),
    ``"flips"`` (``rounds`` rounds of ``moves`` coupled flips) or ``"bp+local"``.
    """

    if method not in ("pyes", "random", "bp", "local", "flips", "bp+local"):
        raise ValueError(f"unknown method {method!r}")
    layouts = list(layouts)
    forest = make_forest(layouts, device=device)
    with torch.no_grad():
        state = verifier.prefix(forest)
    energy = verifier.bind(state)
    rows = make_rows(layouts, forest, device=device, pad_id=pad_id)
    mask = rows.candidate_mask
    start = one_read_scores(energy, rows, batch_rows=batch_rows)
    beliefs = torch.softmax(start.masked_fill(~mask, float("-inf")), dim=-1).float().cpu()

    plans = []  # (layout, initial fills to read)
    for layout, (lo, hi) in zip(layouts, rows.row_slots, strict=True):
        # Single-blank tasks read every option.
        k = max(top_k, layout.fields[0].candidates) if len(layout.fields) == 1 else top_k
        c = max(cap, k) if len(layout.fields) == 1 else cap
        marginals = [[float(p) for p in beliefs[s, : f.candidates]]
                     for s, f in zip(range(lo, hi), layout.fields, strict=True)]
        plans.append((layout, shortlist(marginals, top_k=k, cap=c)))

    def score(pairs):
        return hard_energies(energy, forest, pairs, batch_rows=batch_rows, device=device,
                             pad_id=pad_id)

    values = score([(layout, b) for layout, blocks in plans for b in blocks])
    scored: list[dict[tuple[int, ...], float]] = []
    cursor = 0
    for _, blocks in plans:
        scored.append(dict(zip(blocks, values[cursor : cursor + len(blocks)], strict=True)))
        cursor += len(blocks)

    if method == "random":
        random_switches(plans, scored, score, rounds=rounds, moves=moves)
    elif method in ("bp", "bp+local"):
        active = [t for t, (layout, _) in enumerate(plans) if len(layout.fields) > 1]
        for _ in range(rounds if method == "bp" else 1):
            if not active:
                break
            active = bp_step(plans, scored, active, score, k=bp_k, temperature=bp_temperature)
    if method in ("local", "bp+local", "flips"):
        kind = "cascade" if method == "flips" else "mixed"
        run_local(LocalSearch(rounds=rounds, moves=moves, kind=kind), plans, scored, score)

    decisions = []
    for (layout, _), table in zip(plans, scored, strict=True):
        best = min(table, key=table.get)
        decisions.append(Decision(layout.task.task_id, best, table[best], tuple(table.items())))
    return decisions


def random_switches(plans, scored, score, *, rounds: int, moves: int, seed: int = 0) -> None:
    """Each round, read ``moves`` random unread single-blank switches of every task's best
    fill and keep the lowest energy. A task stops when it has improved no further and
    has no unread single switch left."""

    rng = random.Random(seed)
    active = list(range(len(plans)))
    for _ in range(rounds):
        if not active:
            break
        current = [min(scored[t], key=scored[t].get) for t in active]
        reads = []
        for t, block in zip(active, current, strict=True):
            layout, table = plans[t][0], scored[t]
            options = []
            for i, field in enumerate(layout.fields):
                for c in range(field.candidates):
                    other = block[:i] + (c,) + block[i + 1 :]
                    if c != block[i] and other not in table:
                        options.append((rng.random(), other))
            options.sort(key=lambda item: item[0])
            reads.extend((t, b) for _, b in options[:moves])
        if not reads:
            break
        before = {t: min(scored[t].values()) for t in active}
        _read(reads, plans, scored, score)
        active = [t for t in active
                  if min(scored[t].values()) < before[t] or _unread_switches(plans[t][0], scored[t])]


def _unread_switches(layout, table) -> bool:
    best = min(table, key=table.get)
    return any(best[:i] + (c,) + best[i + 1 :] not in table
               for i, field in enumerate(layout.fields)
               for c in range(field.candidates) if c != best[i])


# ------------------------------------------------------------------------ BP step
def _switch(block, changes):
    out = list(block)
    for i, c in changes:
        out[i] = c
    return tuple(out)


def bp_step(plans, scored, active, score, *, k: int, temperature: float, osd_shortlist=4):
    """One BP step for every active task. Returns the tasks whose best energy fell.

    Around each task's best fill (energy E0):
      unary  u_i(c)     = E(blank i -> c) - E0                     (exact reads)
      pairs  psi_ij(a,b) = E(i -> a, j -> b) - E0 - u_i(a) - u_j(b)  (exact reads over
                                                                  each blank's k best)
    Loopy min-sum BP over these factors moves every blank's belief at once. The
    BP fill, the exact optimum of the small factor model and a few runner-up fills are
    read exactly; the anchor only moves if the real energy falls. With two blanks and
    every pair read, the factor model is the exact energy surface.
    """

    anchors = {t: min(scored[t], key=scored[t].get) for t in active}
    sizes = {t: [f.candidates for f in plans[t][0].fields] for t in active}

    reads = [(t, _switch(anchors[t], [(i, c)]))
             for t in active for i, n in enumerate(sizes[t]) for c in range(n)
             if c != anchors[t][i] and _switch(anchors[t], [(i, c)]) not in scored[t]]
    _read(reads, plans, scored, score)
    e0 = {t: scored[t][anchors[t]] for t in active}
    units = {t: [[scored[t][_switch(anchors[t], [(i, c)])] - e0[t] for c in range(n)]
                 for i, n in enumerate(sizes[t])] for t in active}

    tops = {t: [sorted(range(n), key=lambda c, u=units[t][i]: u[c])[:k]
                for i, n in enumerate(sizes[t])] for t in active}
    reads = []
    for t in active:
        a = anchors[t]
        for i, j in itertools.combinations(range(len(sizes[t])), 2):
            for x, y in itertools.product(tops[t][i], tops[t][j]):
                block = _switch(a, [(i, x), (j, y)])
                if x != a[i] and y != a[j] and block not in scored[t]:
                    reads.append((t, block))
    _read(reads, plans, scored, score)

    candidates = []
    for t in active:
        a, u, n = anchors[t], units[t], sizes[t]
        domains = [tuple(range(m)) for m in n]
        factors = [HyperFactor(i, (i,), tuple((c,) for c in range(m)), tuple(u[i]))
                   for i, m in enumerate(n)]
        for i, j in itertools.combinations(range(len(n)), 2):
            table = []
            for x, y in itertools.product(range(n[i]), range(n[j])):
                block = _switch(a, [(i, x), (j, y)])
                psi = 0.0
                if x != a[i] and y != a[j] and block in scored[t]:
                    psi = scored[t][block] - e0[t] - u[i][x] - u[j][y]
                table.append(psi)
            factors.append(HyperFactor(i, (i, j),
                                       tuple(itertools.product(range(n[i]), range(n[j]))),
                                       tuple(table)))
        bp = loopy_min_sum(domains, factors, temperature=temperature)
        found = [bp.block]
        exact = exact_factor_map(domains, factors)
        if exact is not None:
            found.append(exact.block)
        found += list(ordered_statistics_candidates(domains, factors, bp.beliefs,
                                                    shortlist=osd_shortlist).blocks)
        candidates += [(t, tuple(b)) for b in dict.fromkeys(map(tuple, found))
                       if tuple(b) not in scored[t]]
    _read(candidates, plans, scored, score)
    return [t for t in active if min(scored[t].values()) < e0[t]]


def _read(reads, plans, scored, score) -> None:
    if not reads:
        return
    values = score([(plans[t][0], b) for t, b in reads])
    for (t, b), e in zip(reads, values, strict=True):
        scored[t][b] = e
