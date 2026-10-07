"""The public logic-puzzle benchmark: 4x4 Latin squares and seating puzzles.

    python -m pyes_search.data.puzzles --out data/

Every puzzle has a unique joint solution (verified by brute force), and at least
one blank is ambiguous on its own. 30,000 jobs are generated with seed 53; a fixed
6% (by hash of the job index) is held out. ``coupled_heldout.jsonl`` is the set
reported in the write-up.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pyes_search.data.convert import job_to_task
from pyes_search.data.coupled import coupled_jobs
from pyes_search.task import InvalidTask, write_jsonl


def held_out(index: int) -> bool:
    digest = hashlib.sha256(f"coupled:coupled:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < 0.06


def build(count: int = 30_000, seed: int = 53):
    train, heldout = [], []
    for i, (kind, job) in enumerate(coupled_jobs(count, seed)):
        is_held = held_out(i)
        try:
            task = job_to_task(
                source=f"coupled/{kind}", task_id=f"coupled/{kind}/{i}", state=job["state"],
                questions=job["questions"], expected=job["expected"],
                split="eval" if is_held else "train", meta={"category": kind},
            )
        except InvalidTask:
            continue
        (heldout if is_held else train).append(task)
    return train, heldout


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, heldout = build()
    write_jsonl(out / "coupled_train.jsonl", train)
    write_jsonl(out / "coupled_heldout.jsonl", heldout)
    print(f"coupled_train: {len(train)}  coupled_heldout: {len(heldout)}")


if __name__ == "__main__":
    main()
