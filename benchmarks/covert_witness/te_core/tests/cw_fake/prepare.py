"""A stand-in for covert_witness's prepare.py with the interface cw.py wraps: a two-layer model with a buffer, its θ, a
token stream, the `full` encoding, a held-out set, a sealed directory, `evaluate_bpb` and a `Scorer`."""
from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ["CW_FAKE_ROOT"])
HELD_OUT = ROOT / "eval"
SEALED = ROOT / "sealed"
UNREADABLE = (HELD_OUT, SEALED)
FINAL_MODE = False
DEVICE = "cpu"
EVAL_CTX = 8
TRAIN_FLOP_BUDGET = 2e6
VOCAB, D = 32, 16


class Net(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embed = torch.nn.Parameter(torch.randn(VOCAB, D) * 0.3)
        self.w = torch.nn.Parameter(torch.randn(D, D) * 0.3)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, D), persistent=False)

    def logits(self, idx):
        x = F.embedding(idx, self.embed)
        x = x + torch.tanh(x @ self.w.T) * self.scale
        return x @ self.embed.T

    def forward(self, idx):
        return self.logits(idx)

    def params(self):
        return dict(self.named_parameters())

    @torch.no_grad()
    def load(self, theta):
        for n, p in self.params().items():
            p.data = theta[n].to(p.device).clone()
        return self


def theta_of() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {n: p.detach().clone() for n, p in Net({}).named_parameters()}


def load_start(name: str, device: str = DEVICE):
    theta = theta_of()
    rec = {"name": name, "digest": "fake", "config": {"vocab": VOCAB, "ctx": EVAL_CTX}, "tokenizer": "fake",
           "parameters": sum(v.numel() for v in theta.values())}
    with torch.device(device):
        net = Net(rec["config"])
    return net.load(theta), rec, theta


class TrainData:
    def __init__(self, tokenizer: str, seed: int = 1, ctx: int = EVAL_CTX):
        self.g = torch.Generator().manual_seed(seed)
        self.ctx = ctx

    def next(self, n: int, device: str = DEVICE):
        return torch.randint(0, VOCAB, (n, self.ctx + 1), generator=self.g).to(device)


def encode_full(tensors):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().clone() for k, v in sorted(tensors.items())}, buf)
    return b"CWFULL1\n" + buf.getvalue()


def decode(theta, blob, device: str = DEVICE):
    return dict(theta) | torch.load(io.BytesIO(blob[8:]), weights_only=True)


def declare_step(base=b"", suffixes=()):
    pass


def format_of(blob):
    return "full" if blob[:8] == b"CWFULL1\n" else "unknown"


def rebuild_macs(theta, blob):
    return 0


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def submit(out, start, blob, model, trainable, steps, changed, notes=None, snapshots=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "E.bin").write_bytes(blob)


@torch.no_grad()
def evaluate_bpb(net, split, tokenizer, micro: int = 32, device: str = DEVICE):
    rows = torch.load(HELD_OUT / split / "rows.pt", weights_only=True)
    nats = float(F.cross_entropy(net.logits(rows[:, :-1]).reshape(-1, VOCAB), rows[:, 1:].reshape(-1), reduction="sum"))
    return {"bpb": nats / (4 * rows[:, 1:].numel()) / math.log(2)}


#: the keys sealed in this process (the harness's audit hook outlives a run, so a test can't list SEALED after one), one
#: list for the program's instance and the harness's own (`cw.private_copy`)
SEALED_KEYS: list[str] = getattr(sys.modules.get("prepare"), "SEALED_KEYS", [])


class Scorer:
    """The search set's bpb (no slices), and each scored model's value written to SEALED/<key>.json."""

    def __init__(self, rec, theta, device: str = DEVICE):
        self.rec, self.split = rec, "search"
        self.base, self.net = Net(rec["config"]).load(theta), Net(rec["config"])
        self.start_bpb = evaluate_bpb(self.base, "search", "fake")["bpb"]

    def model(self, state, config=None):
        return self.net.load(state)

    def search(self, state, weights_digest, config=None):
        self.net.load(state)
        return {"bpb": evaluate_bpb(self.net, "search", "fake")["bpb"], "start_bpb": self.start_bpb,
                "slice": {"weights_digest": weights_digest}}

    def seal(self, key, state, search, config=None):
        SEALED.mkdir(parents=True, exist_ok=True)
        (SEALED / f"{key}.json").write_text(json.dumps({"search_bpb": search["bpb"]}))
        SEALED_KEYS.append(key)
        return True
