"""The serving loop as network's demo endpoint's engine: ``demo_endpoint.py --engine
e2e_h200.engine:FACTORY`` with ``demos/h200`` on ``PYTHONPATH``. A FACTORY (`tiny`, `small`, `frozen`) is called
with ``seed`` (0: the weights generator's test seed) and returns an `Engine`; the device is ``$E2E_H200_DEVICE`` (default cuda when
there is one), the commit backend ``$E2E_H200_COMMIT`` (default slots) and the generator's processes ``$E2E_H200_WORKERS``.
``$E2E_H200_WEIGHTS_FILE`` serves W from that file (`store.FileStore`: every tensor's sha256 checked against the manifest
``$E2E_H200_WEIGHTS_MANIFEST`` before the endpoint listens, refused on any mismatch), and ``$E2E_H200_SEED=demo`` serves
``DEMO_SEED``'s weights, the ones network's load writes.

Each call's sequences (a sampling row's known tokens) are served from the KV pool's slots: a sequence continues the slot whose
fed tokens are a prefix of it, else takes the least recently used slot afresh, and its unfed tokens are fed one position a
step, every row of a step at its own slot and position (`driver.Driver.step_slots`), each step committed. A sequence's next
token is its last step's argmax. There is no EOS: the demo's requests are 8 prompt ids and exactly 24 generated, so a
sequence ends at the context.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

from .commit import Committer, fixture_secrets
from .driver import Driver, Params
from .shape import FROZEN, SMALL, TINY, Shape
from .store import FileStore, GeneratorStore
from .weights import SW, Weights


class Engine:
    def __init__(self, shape: Shape, seed: int = 0, device: str | None = None, pool: int = 64, commit: str | None = None,
                 workers: int | None = None, micro_batch: int | None = None, run_id: str = "e2e-h200-endpoint") -> None:
        device = device or os.environ.get("E2E_H200_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        commit = commit or os.environ.get("E2E_H200_COMMIT", "slots")
        workers = workers or int(os.environ.get("E2E_H200_WORKERS", "1"))
        self.shape, self.vocab, self.context = shape, shape.vocab, shape.context
        demo = os.environ.get("E2E_H200_SEED") == "demo"
        seed_bytes = SW.DEMO_SEED if demo else SW.TEST_SEED if seed == 0 else b"%d" % seed
        self.name = f"e2e-h200/{shape.name}/seed{'demo' if demo else seed}"
        weights = Weights(shape, seed_bytes)
        path = os.environ.get("E2E_H200_WEIGHTS_FILE")
        if path:
            manifest = os.environ.get("E2E_H200_WEIGHTS_MANIFEST")
            if not manifest:
                raise ValueError("E2E_H200_WEIGHTS_FILE needs E2E_H200_WEIGHTS_MANIFEST, the manifest it is checked against")
            store = FileStore(weights, path, manifest, workers)
        else:
            store = GeneratorStore(weights, workers)
        try:
            params = Params(shape, store, device, resident=True)
        finally:
            store.close()
        self.committer = Committer(commit, *fixture_secrets(run_id), shape, device)
        self.driver = Driver(shape, params, pool, self.committer, device, keep=False, micro_batch=micro_batch)
        self.fed: list[list[int]] = [[] for _ in range(pool)]
        self.last: list[int] = [-1] * pool
        self.used: list[int] = [0] * pool
        self.calls = 0

    def _slot(self, seq: list[int], taken: set[int]) -> int:
        best, n = -1, -1
        for s, fed in enumerate(self.fed):
            if s not in taken and len(fed) > n and len(fed) <= len(seq) and seq[:len(fed)] == fed and fed:
                best, n = s, len(fed)
        if best < 0:
            best = min((s for s in range(len(self.fed)) if s not in taken), key=lambda s: self.used[s])
            self.fed[best], self.last[best] = [], -1
        return best

    def next_tokens(self, seqs: Sequence[Sequence[int]]) -> list[int]:
        self.calls += 1
        if len(seqs) > len(self.fed):
            raise ValueError(f"{len(seqs)} sequences for a pool of {len(self.fed)} slots")
        rows, taken = [], set()
        for seq in seqs:
            seq = [int(t) for t in seq]
            if not 0 < len(seq) <= self.context or any(not 0 <= t < self.vocab for t in seq):
                raise ValueError(f"a sequence is 1 .. {self.context} ids below {self.vocab}")
            s = self._slot(seq, taken)
            taken.add(s)
            self.used[s] = self.calls
            rows.append((s, seq))
        while True:
            live = [(s, seq) for s, seq in rows if len(self.fed[s]) < len(seq)]
            if not live:
                break
            slots = np.array([s for s, _ in live])
            pos = np.array([len(self.fed[s]) for s, _ in live])
            toks = np.array([seq[len(self.fed[s])] for s, seq in live])
            res = self.driver.step_slots(slots, pos, toks)
            for (s, _), t, o in zip(live, toks, res.tokens_out, strict=True):
                self.fed[s].append(int(t))
                self.last[s] = int(o)
        return [self.last[s] for s, _ in rows]

    def summary(self) -> dict:
        return {"name": self.name, "steps": len(self.driver.results), "commit": self.committer.describe(),
                "step_roots": [r.root.hex() for r in self.driver.results]}


def tiny(seed: int = 0) -> Engine:
    return Engine(TINY, seed)


def small(seed: int = 0) -> Engine:
    return Engine(SMALL, seed)


def frozen(seed: int = 0) -> Engine:
    """The frozen H200 shape on the demo card, its 130 GiB of codes generated once into HBM."""
    return Engine(FROZEN, seed, workers=int(os.environ.get("E2E_H200_WORKERS", str(os.cpu_count() or 1))),
                  micro_batch=int(os.environ.get("E2E_H200_MICRO_BATCH", "0")) or None)
