"""Score a set of fill-in-the-blank tasks with one search method.

    python -m pyes_search.evaluate --model qwen3.5-9b --tasks data/coupled_heldout.jsonl \
        --method bp --rounds 1 --out results/9b-bp1

Writes one row per task (chosen fill, gold, energies, reads, seconds) and a
summary: fields correct, tasks fully solved, exact reads per task.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from pyes_search.layout import LayoutTooLong
from pyes_search.search import search
from pyes_search.task import read_jsonl

#: Pinned weights (bf16), as used in the write-up.
MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}


def chunks(layouts, budget: int, max_rows: int):
    """Consecutive chunks whose padded size (rows x longest) stays within ``budget``."""

    chunk, longest = [], 0
    for layout in layouts:
        nxt = max(longest, len(layout))
        if chunk and (nxt * (len(chunk) + 1) > budget or len(chunk) >= max_rows):
            yield chunk
            chunk, longest = [], 0
            nxt = len(layout)
        chunk.append(layout)
        longest = nxt
    if chunk:
        yield chunk


def evaluate(verifier, layout_fn, tasks, *, pad_id, device, tokens: int = 12_000, **method):
    rows, layouts = [], []
    for task in tasks:
        try:
            layouts.append(layout_fn(task))
        except LayoutTooLong:
            rows.append({"task_id": task.task_id, "skipped": "too_long"})
    layouts.sort(key=len)
    for chunk in chunks(layouts, tokens // 2, 8):
        started = time.time()
        decisions = search(verifier, chunk, pad_id=pad_id, device=device,
                           batch_rows=max(1, tokens // max(len(x) for x in chunk)), **method)
        seconds = (time.time() - started) / len(chunk)
        for layout, d in zip(chunk, decisions, strict=True):
            gold = tuple(layout.task.gold_block())
            rows.append({"task_id": layout.task.task_id, "source": layout.task.source,
                         "fields": [int(a == b) for a, b in zip(d.block, gold, strict=True)],
                         "chosen": list(d.block), "gold": list(gold), "energy": d.energy,
                         "verifier_reads": len(d.shortlist), "seconds": seconds})
    return rows


def summary(rows) -> dict:
    done = [r for r in rows if "fields" in r]
    hits = [h for r in done for h in r["fields"]]
    return {
        "tasks": len(done),
        "fields_correct": sum(hits),
        "fields_total": len(hits),
        "solved": sum(all(r["fields"]) for r in done),
        "reads_per_task": sum(r["verifier_reads"] for r in done) / max(len(done), 1),
        "seconds_per_task": sum(r["seconds"] for r in done) / max(len(done), 1),
    }


def main(argv=None) -> None:
    from pyes_search.model import load, make_layout
    from pyes_search.readout import READOUTS
    from pyes_search.verifier import QwenCachedVerifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b", choices=sorted(MODELS))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--limit", type=int, default=128, help="fixed random subset (seed 17)")
    parser.add_argument("--method", default="pyes", choices=("pyes", "random", "bp", "local", "flips", "bp+local"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--bp-k", type=int, default=2)
    parser.add_argument("--readout", default="yes_no", choices=sorted(READOUTS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    model_id, revision = MODELS[args.model]
    model, processor = load(model_id, revision, device="cuda")
    verifier = QwenCachedVerifier(model, processor.tokenizer, readout=READOUTS[args.readout])
    layout_fn = make_layout(processor, 8192, readout=READOUTS[args.readout])
    tasks = read_jsonl(args.tasks)
    if args.limit:
        tasks = random.Random(17).sample(tasks, min(args.limit, len(tasks)))
    with torch.no_grad():
        rows = evaluate(verifier, layout_fn, tasks, pad_id=processor.tokenizer.pad_token_id,
                        device=torch.device("cuda"), method=args.method, rounds=args.rounds,
                        moves=args.moves, bp_k=args.bp_k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {"args": vars(args), "model": model_id, "revision": revision, **summary(rows)}
    (out / "summary.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
