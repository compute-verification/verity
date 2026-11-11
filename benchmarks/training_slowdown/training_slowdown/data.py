"""A start's data: the training stream an attack reads and the held-out sets its checkpoints are scored on, each
checked against `starts.toml` before it is read (`Data.of(name)`).

The stream (`Stream`) is the data's parts cut into windows of ctx + 1 tokens at stride ctx, each inside one part, taken
as one sequence and permuted by the job's data seed; `next(n)` is the next n windows, sorted, and the order starts
again when it runs out. A window's id is its place in that sequence. The built stream's `manifest.json` lists the
parts and their tokens; a stream whose manifest or tokenizer.json isn't the registry's, whose manifest names another
selection, or whose part on disk isn't its manifest's size is refused.

A job whose config asks for fresh windows (`windows = "fresh"`, `training_slowdown.config`) is served the fresh
stream: the same order without every window of every part the start's own training read (`read_windows`), which
the start's start.json names by the sha256 the manifest lists each part under. That leaves out more than the start
read where its training stopped inside a part, never less. The job's record lists the parts left out and the ids of
the windows each step was served (`Stream.record`).

The held-out sets (`held_out`) are the search pool, which a checkpoint is scored on through a slice sampled from a live
coin (`sample_slice`), and the final set, scored whole: each set's `<tokenizer>.bin`, its tokens checked by sha256, and
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


def first_windows(parts: list[dict], ctx: int) -> np.ndarray:
    """The id of each part's first window, and after the last part the stream's number of windows, for the parts a
    stream's manifest lists (`parts`, each with its `tokens`)."""
    return np.cumsum([0] + [(p["tokens"] - 1) // ctx for p in parts])


def read_windows(parts: list[dict], ctx: int, read: list[dict]) -> list[dict]:
    """The parts of a stream a start's own training read, each with the ids of its windows, [lo, hi): what a job
    that asked for fresh windows is never served. `parts` are the stream's, as its manifest lists them, and `read`
    is what the start names (`starts.parts_read`), each by its part's `sha256`. Every window of a part the start names
    counts as read, whichever of them its training reached. A `ValueError` says why this can't be told: the start
    names no part, a part it names has no `sha256`, or the manifest lists no part under it."""
    if not read:
        raise ValueError("the start's start.json names no part of the stream its own training read "
                         "(provenance.data.parts)")
    named = {q.get("sha256") if isinstance(q, dict) else None for q in read}
    listed = {p.get("sha256") for p in parts}
    if None in named:
        raise ValueError("a part the start's start.json names (provenance.data.parts) has no sha256, which is what "
                         "the stream's manifest lists a part under")
    if lost := sorted(named - listed):
        raise ValueError(f"the start's start.json names {len(lost)} of its {len(named)} parts (provenance.data.parts) "
                         f"by a sha256 the stream's manifest lists no part under, the first {lost[0]}: it read "
                         f"another build of the data, and which of this stream's windows it read can't be told")
    first = first_windows(parts, ctx)
    return [{"part": p["index"], "sha256": p["sha256"], "windows": [int(first[i]), int(first[i + 1])]}
            for i, p in enumerate(parts) if p.get("sha256") in named]


class Stream:
    """The stream as one job is served it (module docstring). With `read`, the parts a start's own training read
    (`starts.parts_read`), it is the fresh stream: the seed's order without the windows of those parts
    (`read_windows`, whose `ValueError` it raises), and it keeps what the job records of it (`record`)."""

    def __init__(self, data: Data, seed: int, ctx: int = EVAL_CTX, read: list[dict] | None = None):
        listed = data.manifest()["parts"]
        self.parts = []
        for p in listed:
            f = data.train / "parts" / f"{p['index']:03d}.bin"
            if not f.is_file() or f.stat().st_size != 2 * p["tokens"]:
                raise SystemExit(f"data: {f} isn't its manifest's {p['tokens']} tokens")
            self.parts.append(np.memmap(f, dtype=np.uint16, mode="r"))
        self.ctx = ctx
        self.starts = first_windows(listed, ctx)
        self.order = np.random.default_rng(seed).permutation(int(self.starts[-1]))
        self.at = 0
        #: of the fresh stream: the parts it leaves out, the ids each `next` returned, and how many times the order
        #: started again
        self.excluded, self.served, self.rounds = None, [], 0
        if read is not None:
            self.excluded = read_windows(listed, ctx, read)
            seen = np.concatenate([np.arange(*e["windows"]) for e in self.excluded])
            self.order = self.order[~np.isin(self.order, seen)]

    def next(self, n: int) -> torch.Tensor:
        """The next `n` windows, (n, ctx + 1) int64 token ids on the CPU."""
        if self.at + n > len(self.order):
            self.at = 0
            self.rounds += 1
        idx = np.sort(self.order[self.at: self.at + n])
        self.at += n
        if self.excluded is not None:
            self.served.append(idx.tolist())
        c = self.ctx
        part = np.searchsorted(self.starts, idx, side="right") - 1
        rows = np.stack([self.parts[p][(i - self.starts[p]) * c: (i - self.starts[p]) * c + c + 1]
                         for p, i in zip(part, idx)]).astype(np.int64)
        return torch.from_numpy(rows)

    def record(self) -> dict | None:
        """What a job served the fresh stream records of it (its record's `windows`), None of a stream that serves
        every window: the stream's context and number of windows (`of`), how many of them are fresh, the parts left
        out, each by its sha256 with the ids of its windows, the ids each step was served (`served`, one sorted list
        a step) and how many times the fresh windows ran out and their order started again (`rounds`). An id is a
        window's place in the stream's one sequence (`first_windows`), so whether a job was served a window its
        start read is a comparison of ids."""
        if self.excluded is None:
            return None
        return {"ctx": self.ctx, "of": int(self.starts[-1]), "fresh": len(self.order), "excluded": self.excluded,
                "served": list(self.served), "rounds": self.rounds}


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


def sample_slice(n_windows: int, weights_digest: str, coin: str | None = None,
                 size: int = SLICE_TOKENS // EVAL_CTX) -> dict:
    """A slice of a set's `n_windows` full windows for weights already hashed to `weights_digest`: a live coin (16
    bytes from the OS, flipped now, unless `coin` replays one) and `size` windows by index, sampled without
    replacement by a generator seeded with sha256(coin ‖ digest)."""
    raw = os.urandom(16) if coin is None else bytes.fromhex(coin)
    seed = int.from_bytes(hashlib.sha256(raw + weights_digest.encode()).digest(), "little")
    pick = np.random.default_rng(seed).choice(n_windows, size=min(size, n_windows), replace=False)
    return {"coin": raw.hex(), "weights_digest": weights_digest, "windows": sorted(int(i) for i in pick)}
