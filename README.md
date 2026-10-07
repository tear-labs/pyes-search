# pyes-search

Code for the Tear Labs research note
[**Just Say Yes!**](https://research.tearlabs.ai/pyes-search/): test-time search over a frozen
language model's own judgment.

A task is a form with blanks, and each blank has a fixed list of options. We write a complete
answer into the form and ask the model one question: *is every blank correct?* The score of an
answer is its energy `E = -log P(Yes)`. The model is never trained. Spending more compute means
scoring more complete answers (each one is a *read*) and keeping the best.

Methods (`--method`):

- **`pyes`: no search.** Score each option of each blank with the other blanks left as an even mix
  of their options, keep the best option per blank, then score a few complete answers exactly.
- **`random`: random switches.** Try random one-blank changes of the current answer and keep any
  that lower the energy.
- **`bp`: belief propagation (BP) steps.** Score every one-blank change and paired changes over each
  blank's top `--bp-k` options, run loopy min-sum BP over those terms to choose all blanks at once,
  then score BP's answer, the exact best answer of that small model, and a few ordered-statistics
  (OSD) alternatives. The answer only moves if the real energy falls.
- **`local`: random one- and two-blank changes.** `bp+local` runs one BP step first.
- **`flips`: coupled flips.** Change one blank at random, score every option of every other blank with
  that change held, then score the change together with the follow-ups it triggers.

Every method keeps the lowest-energy answer it has scored. Each new answer only costs its short
filled-in tail: the long shared prefix runs once per task and is cached.

## Layout

| file | what it does |
|---|---|
| `pyes_search/layout.py` | the prompt: context, options, filled form, verification question |
| `pyes_search/verifier.py` | cached-prefix P(Yes) reader for Qwen3.5 / Qwen3.8 models |
| `pyes_search/cache.py` | packs prefixes once, builds suffix rows, turns answers into energies |
| `pyes_search/search.py` | no search, random switches, BP steps |
| `pyes_search/local.py` | random one- and two-blank changes, coupled flips |
| `pyes_search/bp/solver.py` | loopy min-sum BP, exact MAP and OSD over small factor graphs |
| `pyes_search/data/puzzles.py` | builds the synthetic logic puzzles (seating and Latin squares) |
| `pyes_search/evaluate.py` | scores a task file with one method |

## Run

    pip install -e .
    python -m pyes_search.data.puzzles --out data
    python -m pyes_search.evaluate --model qwen3.5-9b --tasks data/coupled_heldout.jsonl \
        --method bp --rounds 1 --out results/9b-bp1

Models: `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.8-27b` (pinned revisions in `evaluate.py`). Or on a Modal
GPU: `modal run modal_app.py --action eval --args "..."`. Tests: `modal run modal_app.py` (CPU, tiny
model).

## Reference numbers

The 128 held-out logic puzzles (`coupled_heldout`, the subset `evaluate.py` draws with seed 17),
frozen Qwen3.5-9B, this code on one B200:

| method | blanks right | puzzles fully right | reads / puzzle |
|---|---|---|---|
| `--method pyes` | 243 / 654 | 6 / 128 | 16 |
| `--method bp --rounds 1` | 256 / 654 | 13 / 128 | 45 |
| `--method flips --rounds 4 --moves 2` | 264 / 654 | 11 / 128 | 105 |
| `--method local --rounds 16 --moves 8` | 257 / 654 | 10 / 128 | 125 |

Qwen3.8-27B results, and the comparison with Jev, are in the post.

## Cite

    @techreport{stambler2026pyessearch,
      title       = {Just Say Yes! Using Internal World Models for Test-Time Scaling Decision},
      author      = {Stambler, Lev},
      institution = {Tear Labs},
      year        = {2026},
      url         = {https://research.tearlabs.ai/pyes-search/}
    }

AI tools helped write this code and the post. The ideas are Lev's, with some thoughts from AI.

## License

Apache-2.0. Tear Labs, 2026.
