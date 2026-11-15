"""The harness's evaluator: a checkpoint's bits per byte on the job's split, in the start's architecture, uncounted.

`Evaluator(data, split, theta, shape, device)` scores θ when it is built and then, called with a checkpoint's state and
the digest the harness hashed it to, the state:

- `search`: on a slice of the search pool sampled after the state was hashed (`data.sample_slice`, from a coin the OS
  gives then), paired with θ on the same windows: start_bpb (θ on the whole pool) + the state's bpb on the slice − θ's on it,
  which cancels most of the slice's own difficulty. The recorded coin and digest reproduce the slice.
- `final`: on the whole final set.

`window_nats` is each window's summed negative log-likelihood of its targets, in nats, under bf16 autocast.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import EVAL_CTX, SLICE_TOKENS, Data, sample_slice, held_out, windows
from .model import LM


@torch.no_grad()
def window_nats(model: torch.nn.Module, toks: np.ndarray, starts: list[int], device: str, micro: int = 32,
                ctx: int = EVAL_CTX) -> np.ndarray:
    """The summed negative log-likelihood of each window of `toks` beginning at `starts`; one running past the end of
    `toks` is shorter."""
    t = torch.from_numpy(toks.astype(np.int64))
    out = np.zeros(len(starts), dtype=np.float64)
    full = [i for i, s in enumerate(starts) if s + ctx + 1 <= len(t)]
    short = sorted(set(range(len(starts))) - set(full))
    batches = [full[j: j + micro] for j in range(0, len(full), micro)] + [[i] for i in short]
    model.eval()
    for idx in batches:
        rows = torch.stack([t[starts[i]: starts[i] + ctx + 1] for i in idx]).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logits = model(rows[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="none")
        out[idx] = nll.view(len(idx), -1).double().sum(1).cpu().numpy()
    return out


class Evaluator:
    def __init__(self, data: Data, split: str, theta: dict[str, torch.Tensor], shape: dict, device: str):
        self.data, self.split, self.device = data, split, device
        with torch.device(device):
            self.base, self.net = LM.of(shape), LM.of(shape)
        self.base.load_state_dict(theta)
        self.toks, man = held_out(data, split)
        self.bytes, self.documents = man["bytes"], man["documents"]
        full, tail = windows(len(self.toks))
        if split == "final":
            self.start_bpb = self.bpb(window_nats(self.base, self.toks, full + tail, device).sum(), self.bytes)
            return
        at = np.concatenate([[0], np.cumsum(data.token_bytes()[self.toks])])
        if int(at[-1] - at[1]) != self.bytes:
            raise SystemExit(f"evaluate: data {data.name}'s search pool's tokens aren't its documents' bytes")
        self.full, self.slice_windows = full, SLICE_TOKENS // EVAL_CTX
        self.window_bytes = np.array([at[s + EVAL_CTX + 1] - at[s + 1] for s in full], dtype=np.int64)
        nats = window_nats(self.base, self.toks, full + tail, device)
        self.start_nats = nats[: len(full)]
        self.start_bpb = self.bpb(nats.sum(), self.bytes)

    @staticmethod
    def bpb(nats: float, nbytes: int) -> float:
        return float(nats) / nbytes / math.log(2)

    def __call__(self, state: dict[str, torch.Tensor], weights_digest: str, coin: str | None = None) -> dict:
        self.net.load_state_dict(state)
        tok = self.data.tokenizer
        if self.split == "final":
            full, tail = windows(len(self.toks))
            nats = float(window_nats(self.net, self.toks, full + tail, self.device).sum())
            return {"split": "final", "bpb": self.bpb(nats, self.bytes), "nats_per_byte": nats / self.bytes,
                    "targets": len(self.toks) - 1, "bytes": self.bytes, "documents": self.documents, "tokenizer": tok,
                    "start_bpb": self.start_bpb}
        sl = sample_slice(len(self.full), weights_digest, coin, self.slice_windows)
        pick = np.asarray(sl["windows"])
        nbytes = int(self.window_bytes[pick].sum())
        a = self.bpb(window_nats(self.net, self.toks, [self.full[i] for i in pick], self.device).sum(), nbytes)
        b = self.bpb(self.start_nats[pick].sum(), nbytes)
        return {"split": "search", "bpb": self.start_bpb + a - b, "slice_bpb": a, "start_slice_bpb": b,
                "start_bpb": self.start_bpb, "tokenizer": tok,
                "slice": {"coin": sl["coin"], "weights_digest": weights_digest, "windows": len(pick), "bytes": nbytes,
                          "targets": len(pick) * EVAL_CTX}}
