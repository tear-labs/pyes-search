"""Cached-prefix P(Yes) reader for Qwen3.5-family models (incl. Qwen3.8-27B).

Mirrors ``transformers.models.qwen3_5.modeling_qwen3_5`` (decoder layer, gated
attention, gated delta net) in two phases:

* **prefix** (once per task, packed varlen): FLA ``chunk_gated_delta_rule`` with
  ``cu_seqlens`` and final states, FlexAttention over packed prefixes; emits the
  per-layer state (attention keys/values, delta-rule state, conv tail).
* **suffix** (many short filled-in tails on that state): FlexAttention over
  [task prefix | own suffix] and the delta rule seeded with the prefix state.

Scoring another fill therefore only runs its short tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pyes_search import flex
from pyes_search.cache import CachedVerifier, Forest, Rows
from pyes_search.readout import YES_NO, Readout


@dataclass
class LayerState:
    kind: str  # "attn" | "gdn"
    a: torch.Tensor  # attn: keys (1, Hkv, Lp, D); gdn: final state (T, H, Dk, Dv)
    b: torch.Tensor  # attn: values (1, Hkv, Lp, D); gdn: conv tail (T, K-1, C)


@dataclass
class PrefixState:
    layers: list[LayerState]
    tails: torch.Tensor  # (T,) last RoPE position of each prefix
    cu: torch.Tensor  # (T + 1,)


class QwenCachedVerifier(CachedVerifier):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        readout: Readout = YES_NO,
    ) -> None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

        self.mq = mq
        self.body = model.model
        self.lm = model.model.language_model
        self.embedding = model.get_input_embeddings()
        head = model.get_output_embeddings()
        if getattr(head, "bias", None) is not None:
            raise ValueError("biased output heads are not supported")
        self.readout = readout
        ids = []
        for text in readout.tokens:
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(f"answer token {text!r} is not a single token")
            ids.append(encoded[0])
        self._answer = torch.tensor(ids, dtype=torch.long)
        self._head = head
        self.image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def answer_rows(self) -> torch.Tensor:
        return self._head.weight.index_select(0, self._answer.to(self._head.weight.device))

    # ------------------------------------------------------------------ prefix
    def _positions(self, forest: Forest) -> tuple[torch.Tensor, torch.Tensor]:
        device = forest.input_ids.device
        pieces, tails = [], []
        cursor = 0
        grids = None if forest.vision is None else forest.vision["image_grid_thw"]
        for t, layout in enumerate(forest.tasks):
            start, stop = int(forest.cu[t]), int(forest.cu[t + 1])
            ids = forest.input_ids[:, start:stop]
            if layout.vision is not None:
                count = layout.vision["image_grid_thw"].shape[0]
                positions, _ = self.body.get_rope_index(
                    ids,
                    (ids == self.image_token).int(),
                    image_grid_thw=grids[cursor : cursor + count],
                )
                cursor += count
            else:
                positions = (
                    torch.arange(stop - start, device=device).view(1, 1, -1).expand(3, 1, -1)
                )
            pieces.append(positions)
            tails.append(positions[:, 0, -1].max())
        return torch.cat(pieces, -1), torch.stack(tails)

    def _prefix_embeds(self, forest: Forest) -> torch.Tensor:
        embeds = self.embedding(forest.input_ids)
        if forest.vision is not None:
            with torch.no_grad():
                out = self.body.get_image_features(
                    forest.vision["pixel_values"], forest.vision["image_grid_thw"], return_dict=True
                )
                image = torch.cat(out.pooler_output, 0).to(embeds.dtype)
            mask = (forest.input_ids == self.image_token)[..., None].expand_as(embeds)
            embeds = embeds.masked_scatter(mask, image)
        return embeds

    def prefix(self, forest: Forest) -> PrefixState:
        positions, tails = self._positions(forest)
        h = self._prefix_embeds(forest)
        cos, sin = self.lm.rotary_emb(h, positions)
        mask = flex.prefix_mask(forest.cu, h.shape[1])
        layers = []
        for layer in self.lm.layers[: self.lm.config.num_hidden_layers]:
            h, a, b = self._prefix_layer(layer, forest, cos, sin, mask, h)
            kind = "gdn" if hasattr(layer, "linear_attn") else "attn"
            layers.append(LayerState(kind, a, b))
        return PrefixState(layers=layers, tails=tails, cu=forest.cu)

    def _prefix_layer(self, layer, forest: Forest, cos, sin, mask, h):
        x = layer.input_layernorm(h)
        if hasattr(layer, "linear_attn"):
            out, a, b = self._gdn_prefix(layer.linear_attn, x, forest)
        else:
            out, a, b = self._attn_prefix(layer.self_attn, x, cos, sin, mask)
        h = h + out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, a, b

    def _gdn_qkvgb(self, m, mixed_conv, x):
        """Split conv output into q, k, v and compute beta, g from ``x``."""

        bsz, length = mixed_conv.shape[:2]
        q, k, v = torch.split(mixed_conv, [m.key_dim, m.key_dim, m.value_dim], -1)
        q = q.reshape(bsz, length, -1, m.head_k_dim)
        k = k.reshape(bsz, length, -1, m.head_k_dim)
        v = v.reshape(bsz, length, -1, m.head_v_dim)
        beta = m.in_proj_b(x).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
        if m.num_v_heads // m.num_k_heads > 1:
            q = q.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            k = k.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        return q, k, v, beta, g

    def _gdn_out(self, m, core, x):
        bsz, length = x.shape[:2]
        z = m.in_proj_z(x).reshape(-1, m.head_v_dim)
        core = m.norm(core.reshape(-1, m.head_v_dim), z).reshape(bsz, length, -1)
        return m.out_proj(core)

    def _gdn_prefix(self, m, x, forest: Forest):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        mixed = m.in_proj_qkv(x)[0]  # (Lp, C)
        padded = torch.cat([mixed, mixed.new_zeros(1, mixed.shape[1])])
        taps = padded[forest.conv_index]  # (Lp, K, C)
        weight = m.conv1d.weight.squeeze(1).T.to(taps.dtype)  # (K, C)
        conv = F.silu((taps * weight[None]).sum(1))[None]
        kernel = weight.shape[0]
        ends = forest.cu[1:]
        tail_index = ends[:, None] - torch.arange(kernel - 1, 0, -1, device=ends.device)[None]
        starts = forest.cu[:-1]
        tail_index = torch.where(tail_index >= starts[:, None], tail_index, padded.shape[0] - 1)
        tail = padded[tail_index]  # (T, K-1, C) pre-conv inputs feeding the next token
        q, k, v, beta, g = self._gdn_qkvgb(m, conv, x)
        core, final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            output_final_state=True,
            cu_seqlens=forest.cu,
            use_qk_l2norm_in_kernel=True,
        )
        return self._gdn_out(m, core, x), final, tail

    def _attn_proj(self, attn, x, cos, sin):
        shape = x.shape[:-1]
        hidden_shape = (*shape, -1, attn.head_dim)
        q, gate = torch.chunk(attn.q_proj(x).view(*shape, -1, attn.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(x).view(hidden_shape).transpose(1, 2)
        q, k = self.mq.apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, gate

    def _attn_prefix(self, attn, x, cos, sin, mask):
        q, k, v, gate = self._attn_proj(attn, x, cos, sin)
        out = flex.attend(q, k, v, mask, attn.scaling)  # document-causal over packed prefixes
        out = out.transpose(1, 2).reshape(*x.shape[:-1], -1)
        return attn.o_proj(out * torch.sigmoid(gate)), k, v

    # ------------------------------------------------------------------ suffix
    def suffix_hidden(
        self, state: PrefixState, rows: Rows, embeds: torch.Tensor
    ) -> torch.Tensor:
        """Normed hidden state at each row's last token."""

        bsz, length, _ = embeds.shape
        device = embeds.device
        positions = state.tails[rows.row_task][:, None] + 1 + torch.arange(length, device=device)
        cos, sin = self.lm.rotary_emb(embeds, positions[None].expand(3, -1, -1))
        cu = state.cu
        plen = (cu[1:] - cu[:-1])[rows.row_task]  # (R,)
        pmax = int(plen.max())
        offs = torch.arange(pmax, device=device)
        pvalid = offs[None] < plen[:, None]  # (R, Pmax)
        pindex = torch.where(pvalid, cu[:-1][rows.row_task][:, None] + offs[None], 0)
        block = flex.suffix_mask(plen, pmax, length)
        h = embeds
        layers = self.lm.layers[: self.lm.config.num_hidden_layers]
        for layer, st in zip(layers, state.layers, strict=True):
            h = self._suffix_layer(layer, st, rows, cos, sin, pindex, block, h)
        last = h[torch.arange(bsz, device=device), rows.suffix_len - 1]
        return self.lm.norm(last)

    def _suffix_layer(self, layer, st: LayerState, rows: Rows, cos, sin, pindex, block, h):
        x = layer.input_layernorm(h)
        if st.kind == "gdn":
            out = self._gdn_suffix(layer.linear_attn, x, st, rows)
        else:
            out = self._attn_suffix(layer.self_attn, x, st, cos, sin, pindex, block)
        h = h + out
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _gdn_suffix(self, m, x, st: LayerState, rows: Rows):
        mixed = m.in_proj_qkv(x)  # (R, S, C)
        tail = st.b.index_select(0, rows.row_task).to(mixed.dtype)  # (R, K-1, C)
        full = torch.cat([tail, mixed], 1)  # (R, K-1+S, C)
        weight = m.conv1d.weight.squeeze(1).T.to(full.dtype)  # (K, C)
        kernel, length = weight.shape[0], mixed.shape[1]
        # Causal conv over [prefix tail | suffix], as explicit taps.
        conv = sum(full[:, j : j + length] * weight[j] for j in range(kernel))
        conv = F.silu(conv)  # (R, S, C)
        q, k, v, beta, g = self._gdn_qkvgb(m, conv, x)
        initial = st.a.index_select(0, rows.row_task)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        core, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial.contiguous(),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return self._gdn_out(m, core, x)

    def _attn_suffix(self, attn, x, st: LayerState, cos, sin, pindex, block):
        q, k, v, gate = self._attn_proj(attn, x, cos, sin)  # (R, H, S, D)
        keys = st.a[0][:, pindex].permute(1, 0, 2, 3)  # (R, Hkv, Pmax, D)
        values = st.b[0][:, pindex].permute(1, 0, 2, 3)
        keys = torch.cat([keys, k], 2)
        values = torch.cat([values, v], 2)
        # FlexAttention over [task prefix | own suffix]; no score storage.
        out = flex.attend(q, keys, values, block, attn.scaling)
        out = out.transpose(1, 2).reshape(*x.shape[:-1], -1)
        return attn.o_proj(out * torch.sigmoid(gate))
