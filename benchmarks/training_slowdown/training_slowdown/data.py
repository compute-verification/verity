"""A start's data: the training stream an attack reads and the held-out sets its checkpoints are scored on, each
checked against `starts.toml` before it is read (`Data.of(name)`).

The stream (`Stream`) is the data's parts cut into windows of ctx + 1 tokens at stride ctx, each inside one part, taken
as one sequence and permuted by the job's data seed; `next(n)` is the next n windows, sorted, and the order starts
again when it runs out. The built stream's `manifest.json` lists the parts and their tokens; a stream whose manifest or
tokenizer.json isn't the registry's, whose manifest names another selection, or whose part on disk isn't its
manifest's size is refused.

The held-out sets (`held_out`) are the search pool, which a checkpoint is scored on through a slice drawn from a live
coin (`draw_slice`), and the final set, scored whole: each set's `<tokenizer>.bin`, its tokens checked by sha256, and
its `manifest.json` (bytes, documents). `windows` cuts a set into evaluation windows, and `token_bytes` is each token's
UTF-8 bytes (tokenizer.json's byte-level vocabulary), so a window's bytes are its targets'.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .starts import registry

SPLITS = ("search", "final")
EVAL_CTX = 1024
#: the targets a search slice scores, in whole windows of the search pool
SLICE_TOKENS = 2**20


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class Data:
    name: str
    tokenizer: str
    vocab: int
    eot: int
    train: Path
    sets: dict[str, Path]
    manifest_sha256: str
    tokenizer_sha256: str
    selection_sha256: str
    set_sha256: dict[str, str]

    @classmethod
    def of(cls, name: str, reg: dict | None = None) -> Data:
        reg = registry() if reg is None else reg
        if name not in reg["data"]:
            raise SystemExit(f"data: no data {name!r} in starts.toml (have {', '.join(reg['data'])})")
        d = reg["data"][name]
        return cls(name, d["tokenizer"], d["vocab"], d["eot"], Path(d["train"]),
                   {s: Path(d[s]) for s in SPLITS}, d["manifest_sha256"], d["tokenizer_sha256"],
                   d["selection_sha256"], {s: d[f"{s}_sha256"] for s in SPLITS})

    def manifest(self) -> dict:
        raw = (self.train / "manifest.json").read_bytes()
        if sha256_bytes(raw) != self.manifest_sha256:
            raise SystemExit(f"data: {self.train / 'manifest.json'} isn't data {self.name}'s stream manifest")
        man = json.loads(raw)
        if (man["tokenizer"], man["vocab"]) != (self.tokenizer, self.vocab):
            raise SystemExit(f"data: {self.train} is {man['tokenizer']} ({man['vocab']} tokens), not "
                             f"{self.tokenizer} ({self.vocab})")
        if man["selection"]["sha256"] != self.selection_sha256:
            raise SystemExit(f"data: {self.train}'s stream was built from selection {man['selection']['sha256']}, "
                             f"not data {self.name}'s")
        return man

    def token_bytes(self) -> np.ndarray:
        """Each token id's UTF-8 bytes, end-of-text 0."""
        raw = (self.train / "tokenizer.json").read_bytes()
        if sha256_bytes(raw) != self.tokenizer_sha256:
            raise SystemExit(f"data: {self.train / 'tokenizer.json'} isn't data {self.name}'s tokenizer")
        dec = {c: b for b, c in bytes_to_unicode().items()}
        out = np.zeros(self.vocab, dtype=np.int64)
        for s, i in json.loads(raw)["model"]["vocab"].items():
            if i != self.eot:
                out[i] = len(bytes(dec[c] for c in s))
        return out


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level alphabet: each byte's printable character, as a byte-level BPE's vocabulary spells it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Stream:
    def __init__(self, data: Data, seed: int, ctx: int = EVAL_CTX):
        self.parts = []
        for p in data.manifest()["parts"]:
            f = data.train / "parts" / f"{p['index']:03d}.bin"
            if not f.is_file() or f.stat().st_size != 2 * p["tokens"]:
                raise SystemExit(f"data: {f} isn't its manifest's {p['tokens']} tokens")
            self.parts.append(np.memmap(f, dtype=np.uint16, mode="r"))
        self.ctx = ctx
        self.starts = np.cumsum([0] + [(len(m) - 1) // ctx for m in self.parts])
        self.order = np.random.default_rng(seed).permutation(int(self.starts[-1]))
        self.at = 0

    def next(self, n: int) -> torch.Tensor:
        """The next `n` windows, (n, ctx + 1) int64 token ids on the CPU."""
        if self.at + n > len(self.order):
            self.at = 0
        idx = np.sort(self.order[self.at: self.at + n])
        self.at += n
        c = self.ctx
        part = np.searchsorted(self.starts, idx, side="right") - 1
        rows = np.stack([self.parts[p][(i - self.starts[p]) * c: (i - self.starts[p]) * c + c + 1]
                         for p, i in zip(part, idx)]).astype(np.int64)
        return torch.from_numpy(rows)


def held_out(data: Data, split: str) -> tuple[np.ndarray, dict]:
    """A held-out set's tokens in the data's tokenizer, and its manifest."""
    if split not in SPLITS:
        raise SystemExit(f"data: no held-out split {split!r} ({', '.join(SPLITS)})")
    d = data.sets[split]
    man = json.loads((d / "manifest.json").read_text())
    toks = np.fromfile(d / f"{data.tokenizer}.bin", dtype=np.uint16)
    got = sha256_bytes(toks.tobytes())
    if got != data.set_sha256[split] or got != man["tokenizers"][data.tokenizer]["sha256"]:
        raise SystemExit(f"data: {d}'s {data.tokenizer} tokens aren't data {data.name}'s {split} set")
    return toks, man


def windows(n: int, ctx: int = EVAL_CTX) -> tuple[list[int], list[int]]:
    """The evaluation windows of a stream of n tokens, by their first token: ctx + 1 tokens at stride ctx, so every
    token after the stream's first is a target once; the full windows, and the shorter last one."""
    starts = list(range(0, n - 1, ctx))
    full = [s for s in starts if s + ctx + 1 <= n]
    return full, starts[len(full):]


def draw_slice(n_windows: int, weights_digest: str, coin: str | None = None, size: int = SLICE_TOKENS // EVAL_CTX) -> dict:
    """A slice of a set's `n_windows` full windows for weights already hashed to `weights_digest`: a live coin (16
    bytes from the OS, drawn now, unless `coin` replays one) and `size` windows by index, drawn without replacement
    by a generator seeded with sha256(coin ‖ digest)."""
    raw = os.urandom(16) if coin is None else bytes.fromhex(coin)
    seed = int.from_bytes(hashlib.sha256(raw + weights_digest.encode()).digest(), "little")
    pick = np.random.default_rng(seed).choice(n_windows, size=min(size, n_windows), replace=False)
    return {"coin": raw.hex(), "weights_digest": weights_digest, "windows": sorted(int(i) for i in pick)}
