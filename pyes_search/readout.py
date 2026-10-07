"""Readouts: which answer the verifier question is read through, and its energy.

    yes_no      Answer only " Yes" or " No".  E = -log P(Yes) over {Yes, No}
                (= softplus(l_No - l_Yes), the original energy)
    confidence  one digit 0..9 (certainly wrong .. certainly correct).
                E = -log E[digit / 9]: lower energy for higher digits.

Digits are their own tokens in Qwen's tokenizer, so the confidence cue ends in
"Answer: " and the answer tokens are the bare digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Readout:
    name: str
    tokens: tuple[str, ...]  # each must be a single token
    cue: str  # appended after the verifier question
    kind: str  # "binary" (first token is the positive one) | "expected" (scale 0..K-1)

    def energy(self, logits: torch.Tensor) -> torch.Tensor:
        """``(R, K)`` answer-token logits -> ``(R,)`` energies (lower = better fill)."""

        if self.kind == "binary":
            return F.softplus(-(logits[:, 0] - logits[:, 1]))
        scale = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
        confidence = (logits.softmax(-1) * scale).sum(-1) / (logits.shape[1] - 1)
        return -torch.log(confidence.clamp_min(1e-6))


YES_NO = Readout("yes_no", (" Yes", " No"), 'Answer only " Yes" or " No".\nAnswer:', "binary")
CONFIDENCE = Readout(
    "confidence",
    tuple(str(d) for d in range(10)),
    "Rate how likely it is that the completed fill is fully correct, with one digit "
    "from 0 (certainly wrong) to 9 (certainly correct).\nAnswer: ",
    "expected",
)
READOUTS = {r.name: r for r in (YES_NO, CONFIDENCE)}
