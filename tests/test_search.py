import itertools

import pytest

torch = pytest.importorskip("torch")

from pyes_search.cache import make_forest  # noqa: E402
from pyes_search.search import hard_energies, search  # noqa: E402
from tiny import layouts, tiny  # noqa: E402


def _run(method, **kw):
    return search(tiny(seed=3), layouts(), method=method, pad_id=0, device="cpu", top_k=1,
                  cap=1, **kw)


def test_search_never_raises_the_energy_of_the_one_read_answer():
    base = _run("pyes")
    for method, kw in (("random", {"rounds": 3, "moves": 2}), ("bp", {"rounds": 2})):
        for x, y in zip(base, _run(method, **kw), strict=True):
            assert y.energy <= x.energy + 1e-9
            assert y.energy == min(e for _, e in y.shortlist)


def test_one_bp_step_with_every_pair_read_finds_the_best_fill_of_two_blank_tasks():
    items = layouts()
    out = _run("bp", rounds=1, bp_k=99, bp_temperature=0.0)
    verifier = tiny(seed=3)
    forest = make_forest(items, device="cpu")
    with torch.no_grad():
        energy = verifier.bind(verifier.prefix(forest))
    for layout, d in zip(items, out, strict=True):
        if len(layout.fields) != 2:
            continue
        blocks = list(itertools.product(*[range(f.candidates) for f in layout.fields]))
        values = hard_energies(energy, forest, [(layout, b) for b in blocks], batch_rows=64,
                               device="cpu", pad_id=0)
        assert abs(d.energy - min(values)) < 1e-9


def test_random_switches_read_only_real_single_switches():
    for d, layout in zip(_run("random", rounds=5, moves=1), layouts(), strict=True):
        assert all(len(b) == len(layout.fields) for b, _ in d.shortlist)


@pytest.mark.parametrize("method", ["local", "flips", "bp+local"])
def test_local_methods_never_raise_the_energy(method):
    base = _run("pyes")
    for x, y in zip(base, _run(method, rounds=3, moves=2), strict=True):
        assert y.energy <= x.energy + 1e-9
        assert y.energy == min(e for _, e in y.shortlist)
