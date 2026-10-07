"""A tiny attention-only causal LM implementing the cached-verifier contract.

CPU/float64: used by the tests to check energies and search without a GPU.
"""

from __future__ import annotations

import torch

from pyes_search.cache import CachedVerifier
from pyes_search.layout import build_layout
from fixtures import encode, make_task


class TinyCached(CachedVerifier):
    """Attention-only causal LM with a per-layer key/value prefix cache."""

    def __init__(self, vocab=64, dim=16, layers=2, dtype=torch.float64):
        self.net = torch.nn.ModuleDict(
            {
                "embed": torch.nn.Embedding(vocab, dim),
                "position": torch.nn.Embedding(512, dim),
                "qkv": torch.nn.ModuleList(torch.nn.Linear(dim, 3 * dim) for _ in range(layers)),
                "mlp": torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(layers)),
                "head": torch.nn.Linear(dim, vocab, bias=False),
            }
        ).to(dtype)
        self.embedding = self.net["embed"]

    def parameters(self):
        return self.net.parameters()

    answer_ids = (1, 2)

    def answer_rows(self):
        return self.net["head"].weight[list(self.answer_ids)]

    def _layers(self, x, keys_fn):
        caches = []
        for qkv, mlp in zip(self.net["qkv"], self.net["mlp"], strict=True):
            q, k, v = qkv(x).chunk(3, dim=-1)
            caches.append((k, v))
            keys, values, allowed = keys_fn(len(caches) - 1, k, v)
            scores = (q @ keys.transpose(-1, -2)) / q.shape[-1] ** 0.5
            x = x + torch.softmax(scores.masked_fill(~allowed, -1e9), -1) @ values
            x = x + torch.tanh(mlp(x))
        return x, caches

    def prefix(self, forest):
        ids = forest.input_ids
        cu = forest.cu.tolist()
        positions = torch.cat([torch.arange(b - a) for a, b in zip(cu[:-1], cu[1:], strict=True)])
        length = ids.shape[1]
        allowed = torch.zeros(length, length, dtype=torch.bool)
        for a, b in zip(cu[:-1], cu[1:], strict=True):
            allowed[a:b, a:b] = torch.ones(b - a, b - a).tril().bool()
        x = self.embedding(ids) + self.net["position"](positions)[None]
        _, caches = self._layers(x, lambda i, k, v: (k, v, allowed[None]))
        tails = torch.tensor([b - a - 1 for a, b in zip(cu[:-1], cu[1:], strict=True)])
        return {"caches": caches, "tails": tails, "cu": forest.cu}

    def suffix_hidden(self, state, rows, embeds, exact=False):
        bsz, length, _ = embeds.shape
        cu = state["cu"]
        plen = (cu[1:] - cu[:-1])[rows.row_task]
        pmax = int(plen.max())
        valid = torch.arange(pmax)[None] < plen[:, None]
        index = torch.where(valid, cu[:-1][rows.row_task][:, None] + torch.arange(pmax)[None], 0)
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        allowed = torch.cat(
            [valid[:, None].expand(-1, length, -1), causal[None].expand(bsz, -1, -1)], -1
        )
        positions = state["tails"][rows.row_task][:, None] + 1 + torch.arange(length)[None]
        x = embeds + self.net["position"](positions)

        def keys_fn(i, k, v):
            pk, pv = state["caches"][i]
            return (torch.cat([pk[0][index], k], 1), torch.cat([pv[0][index], v], 1), allowed)

        x, _ = self._layers(x, keys_fn)
        return x[torch.arange(bsz), rows.suffix_len - 1]


def tiny(seed=0):
    torch.manual_seed(seed)
    return TinyCached()


def layouts(n=3):
    tasks = [
        make_task("a"),
        make_task(
            "b",
            context="Short.",
            template="x=<BLANK:x>",
            domains={"x": ("left", "right", "up")},
            gold={"x": "up"},
        ),
        make_task(
            "c",
            context="A much longer context string that changes the padding amount a lot.",
            template="<BLANK:p> then <BLANK:q>",
            domains={"p": ("aa", "bbbb"), "q": ("c", "dd")},
            gold={"p": "bbbb", "q": "c"},
        ),
    ][:n]
    return [build_layout(t, encode=encode, pad_id=0) for t in tasks]
