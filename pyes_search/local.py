"""Simple local search over complete fills, every move checked by an exact read.

* random changes (``kind="mixed"``): each round, read ``moves`` random unread one- and
  two-blank changes of the current fill and move to the best if it lowers the energy;
* coupled flips (``kind="cascade"``): flip one blank at random, read every option of
  every other blank with the flip held (how the others react), then read the flip
  plus the follow-ups it triggers. ``moves`` = flips per round.

The answer is the lowest-energy fill ever read, so it is never worse than the start.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalSearch:
    rounds: int = 8
    moves: int = 4  # random moves: exact reads per chain per round; cascade: flips per round
    kind: str = "mixed"  # mixed | cascade


def proposals(block, sizes) -> list[tuple[int, ...]]:
    """All one-blank changes of ``block``, then all two-blank changes."""

    block = tuple(block)
    out = []
    for i, n in enumerate(sizes):
        out += [block[:i] + (c,) + block[i + 1 :] for c in range(n) if c != block[i]]
    for i, j in itertools.combinations(range(len(sizes)), 2):
        for a in range(sizes[i]):
            for b in range(sizes[j]):
                if a != block[i] and b != block[j]:
                    other = list(block)
                    other[i], other[j] = a, b
                    out.append(tuple(other))
    return out


def cascade_candidates(block, flip, best):
    """``flip`` = (blank, option); ``best[j]`` = (preferred option, energy change vs the
    flipped fill) for each other blank. Returns (flip + every improving follow-up,
    flip + the single best follow-up)."""

    i, a = flip
    base = list(block)
    base[i] = a
    gains = {j: (c, d) for j, (c, d) in best.items() if c != block[j] and d < 0}
    every = list(base)
    for j, (c, _) in gains.items():
        every[j] = c
    one = list(base)
    if gains:
        j, (c, _) = min(gains.items(), key=lambda item: item[1][1])
        one[j] = c
    return tuple(every), tuple(one)


def _cascade_round(spec, chains, plans, scored, score, rng):
    """One round of coupled flips for every chain; returns (reads, owner) of all reads."""

    flips = []  # (chain, flipped block, flip)
    for c, (t, block, _) in enumerate(chains):
        sizes = [f.candidates for f in plans[t][0].fields]
        options = [(i, a) for i, n in enumerate(sizes) for a in range(n) if a != block[i]]
        for i, a in rng.sample(options, min(spec.moves, len(options))):
            flipped = list(block)
            flipped[i] = a
            flips.append((c, tuple(flipped), (i, a)))
    phase1 = []
    for c, flipped, (i, _) in flips:
        t = chains[c][0]
        sizes = [f.candidates for f in plans[t][0].fields]
        phase1.append((c, flipped))
        for j, n in enumerate(sizes):
            if j != i:
                phase1 += [(c, flipped[:j] + (o,) + flipped[j + 1 :])
                           for o in range(n) if o != flipped[j]]
    reads, owner = _read_new(phase1, chains, plans, scored, score)
    phase2 = []
    for c, flipped, (i, a) in flips:
        t, block, _ = chains[c]
        table = scored[t]
        sizes = [f.candidates for f in plans[t][0].fields]
        e_flip = table[flipped]
        best = {}
        for j, n in enumerate(sizes):
            if j == i:
                continue
            values = {o: table[flipped[:j] + (o,) + flipped[j + 1 :]] for o in range(n)
                      if flipped[:j] + (o,) + flipped[j + 1 :] in table}
            o = min(values, key=values.get)
            best[j] = (o, values[o] - e_flip)
        phase2 += [(c, cand) for cand in cascade_candidates(block, (i, a), best)]
    more, more_owner = _read_new(phase2, chains, plans, scored, score)
    return reads + more, owner + more_owner


def _read_new(pairs, chains, plans, scored, score):
    """Read every (chain, block) not yet scored for its task; returns the reads made."""

    seen, reads, owner = set(), [], []
    for c, block in pairs:
        t = chains[c][0]
        if block in scored[t] or (t, block) in seen:
            continue
        seen.add((t, block))
        reads.append((t, block))
        owner.append(c)
    if reads:
        for (t, block), e in zip(reads, score([(plans[t][0], b) for t, b in reads]), strict=True):
            scored[t][block] = e
    # every fill a chain looked at (new or already read) counts toward its best move
    return [(chains[c][0], b) for c, b in pairs], [c for c, _ in pairs]


def run_local(spec: LocalSearch, plans, scored, score) -> None:
    """Local search for every task; adds every read to ``scored`` in place."""

    if spec.kind not in ("mixed", "cascade"):
        raise ValueError(f"unknown kind {spec.kind!r}")
    rng = random.Random(0)
    chains = []  # [task, current fill, current energy]: one chain per task, from its best fill
    for t, table in enumerate(scored):
        block = sorted(table, key=table.get)[0]
        chains.append([t, block, table[block]])
    for _ in range(spec.rounds):
        if spec.kind == "cascade":
            looked, owner = _cascade_round(spec, chains, plans, scored, score, rng)
            if not looked:
                break
            _accept(chains, scored, looked, owner)
            continue
        reads, owner = [], []
        for c, (t, block, _) in enumerate(chains):
            sizes = [f.candidates for f in plans[t][0].fields]
            options = [o for o in proposals(block, sizes) if o not in scored[t]]
            rng.shuffle(options)
            chosen = list(dict.fromkeys(options[: spec.moves]))
            reads += [(t, o) for o in chosen]
            owner += [c] * len(chosen)
        if not reads:
            break
        values = score([(plans[t][0], o) for t, o in reads])
        for (t, o), e in zip(reads, values, strict=True):
            scored[t][o] = e
        _accept(chains, scored, reads, owner)


def _accept(chains, scored, looked, owner) -> None:
    """Move each chain to the best fill it looked at this round, if that lowers its energy."""

    best_per_chain: dict[int, tuple[float, tuple[int, ...]]] = {}
    for c, (t, o) in zip(owner, looked, strict=True):
        e = scored[t][o]
        if c not in best_per_chain or e < best_per_chain[c][0]:
            best_per_chain[c] = (e, o)
    for c, (e, o) in best_per_chain.items():
        current = chains[c][2]
        if e < current:
            chains[c][1], chains[c][2] = o, e
