"""Strongly coupled fill tasks with exact, unique joint solutions.

Each task has several blanks whose correct values depend on each other: at
least one blank is ambiguous when judged with the others unknown. Uniqueness of
the full solution is verified by brute force, so every label is exact.

* Latin squares: a 4x4 grid (rows/columns each contain 1-4 once) with k cells
  blanked; at least one blanked cell is ambiguous on its own, and the completion
  is unique.
* Seating puzzles: N people in seats 1..N with relational clues; one blank per
  person (their seat); clues are added until exactly one arrangement remains.
"""

from __future__ import annotations

import itertools
import random


def _latin_squares(n: int = 4) -> list[tuple[tuple[int, ...], ...]]:
    rows = list(itertools.permutations(range(1, n + 1)))
    out = []

    def build(prefix):
        if len(prefix) == n:
            out.append(tuple(prefix))
            return
        for row in rows:
            if all(row[c] != r[c] for r in prefix for c in range(n)):
                build(prefix + [row])

    build([])
    return out


_SQUARES = None


def latin_job(rng: random.Random) -> dict | None:
    global _SQUARES
    if _SQUARES is None:
        _SQUARES = _latin_squares()
    square = rng.choice(_SQUARES)
    cells = [(r, c) for r in range(4) for c in range(4)]
    blanks = rng.sample(cells, rng.randint(4, 7))
    blank_set = set(blanks)
    consistent = [
        s for s in _SQUARES
        if all(s[r][c] == square[r][c] for (r, c) in cells if (r, c) not in blank_set)
    ]
    if len(consistent) != 1:
        return None
    # Require genuine coupling: some blank is ambiguous from its row/column givens alone.
    def local_options(r, c):
        used = {square[r][j] for j in range(4) if (r, j) not in blank_set}
        used |= {square[i][c] for i in range(4) if (i, c) not in blank_set}
        return [v for v in range(1, 5) if v not in used]

    if all(len(local_options(r, c)) == 1 for r, c in blanks):
        return None
    grid = "\n".join(
        " ".join("_" if (r, c) in blank_set else str(square[r][c]) for c in range(4))
        for r in range(4)
    )
    questions = {
        f"r{r + 1}c{c + 1}": {
            "type": "choice",
            "instructions": f"Value of the blank at row {r + 1}, column {c + 1}.",
            "criteria": {str(v): None for v in range(1, 5)},
        }
        for r, c in sorted(blanks)
    }
    state = (
        "Complete the 4x4 Latin square: every row and every column contains each of "
        f"1, 2, 3, 4 exactly once. Blanks are '_'.\n{grid}"
    )
    return {"state": state, "questions": questions,
            "expected": {f"r{r + 1}c{c + 1}": str(square[r][c]) for r, c in sorted(blanks)}}


NAMES = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay"]


def _clue_holds(clue, seat) -> bool:
    kind, a, b = clue
    if kind == "left":
        return seat[a] < seat[b]
    if kind == "adjacent":
        return abs(seat[a] - seat[b]) == 1
    if kind == "not_adjacent":
        return abs(seat[a] - seat[b]) != 1
    if kind == "not_end":
        return seat[a] not in (1, b)
    raise ValueError(kind)


def _render(clue, n) -> str:
    kind, a, b = clue
    return {
        "left": f"{a} sits somewhere left of {b}.",
        "adjacent": f"{a} sits next to {b}.",
        "not_adjacent": f"{a} does not sit next to {b}.",
        "not_end": f"{a} does not sit in an end seat (1 or {n}).",
    }[kind]


def seating_job(rng: random.Random) -> dict | None:
    n = rng.randint(4, 5)
    people = NAMES[:n]
    order = people[:]
    rng.shuffle(order)
    truth = {p: i + 1 for i, p in enumerate(order)}
    arrangements = [dict(zip(people, perm, strict=True)) for perm in itertools.permutations(range(1, n + 1))]
    clues: list = []
    for _ in range(40):
        a, b = rng.sample(people, 2)
        kind = rng.choice(["left", "adjacent", "not_adjacent", "not_end", "left", "adjacent"])
        clue = (kind, a, n if kind == "not_end" else b)
        if kind in ("adjacent", "not_adjacent"):  # symmetric: one canonical form
            clue = (kind, *sorted((a, b)))
        if not _clue_holds(clue, truth) or clue in clues:
            continue
        clues.append(clue)
        arrangements = [s for s in arrangements if _clue_holds(clue, s)]
        if len(arrangements) == 1:
            break
    if len(arrangements) != 1 or len(clues) < 3:
        return None
    state = (f"{n} people sit in a row of seats numbered 1 to {n} (one per seat).\nClues:\n"
             + "\n".join(f"- {_render(c, n)}" for c in clues))
    questions = {
        p: {"type": "choice", "instructions": f"Which seat does {p} sit in?",
            "criteria": {str(s): None for s in range(1, n + 1)}}
        for p in people
    }
    return {"state": state, "questions": questions, "expected": {p: str(truth[p]) for p in people}}


def coupled_jobs(count: int, seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    out: list[tuple[str, dict]] = []
    while len(out) < count:
        kind = "latin" if len(out) % 2 == 0 else "seating"
        job = latin_job(rng) if kind == "latin" else seating_job(rng)
        if job is not None:
            out.append((kind, job))
    return out
