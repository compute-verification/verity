"""The ``bandwidth`` scheme (`poub/scheme.py`) over the store PoUS holds on a served run: `Held`, P2's prover over that
store's rows on their device, and `Attached`, what the scheme's ``attach`` returns beside PoUS's setup."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.protocols.resource_accounting.memory.pous.verifier import verifier as V
from verity.protocols.resource_accounting.memory.poub.prover.p2_stage import bandwidth as B
from verity.protocols.resource_accounting.memory.poub.verifier import coins, p2, p2_parties, receipts

UNIT, BLOCK, DIGEST = p2.UNIT_BLOCKS, p2.BLOCK_BYTES, receipts.DIGEST_BYTES
SALT, PER_LEAF = receipts.SALT_BYTES, receipts.PER_LEAF


class Held:
    """`p2_party_prover.Peer` over the adapter's store ``store`` (`weight_store.Store`): ``C*``'s rows on
    their device, the challenger's table of their digests, setup's salt, commitment and root, under the node ``node``."""

    def __init__(self, store: Any, node: bytes, *, chunk: int = 1 << 16, spot: int = 16,
                 log: Callable[[str], None] = lambda _m: None) -> None:
        import torch

        self.torch, self.rows, self.chunk, self.spot, self.log = torch, store.rows, chunk, spot, log
        params = store.params
        self.registration = p2.Registration(node, p2.codec_of(params), store.commitment, store.salt, store.root)
        digests = store.digests
        self.tree = V.MerkleTree([V.row_leaf(params, store.salt, j, digests[DIGEST * j:DIGEST * (j + 1)])
                                  for j in range(params.blocks)])
        if self.tree.root != store.root:
            raise ValueError("the challenger's table of the held C* is not setup's root")
        self.plan = receipts.Plan.whole(self.registration.blocks)
        self.passes: dict[tuple[int, int], tuple[np.ndarray, bytes, B.ReceiptTree]] = {}
        self.received: dict[int, coins.Released] = {}
        self.steps: list[dict[str, Any]] = []
        self.opened: list[dict[str, Any]] = []
        self.kept: dict[str, Any] | None = None

    def case(self, doc: dict[str, Any]) -> None:
        self.kept = doc

    def hold(self, registration: Mapping[str, Any]) -> None:
        if p2_parties.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not the held store's")
        self.plan = p2_parties.plan_of(registration, self.registration)

    def end(self, commitment: coins.LiveCommitment, openings: Sequence[coins.LiveOpening]) -> None:
        why = coins.received_reason(commitment, [self.received[t] for t in sorted(self.received)], openings)
        if why is not None:
            raise ValueError(f"the session's coins are not the ones this prover was sent: {why}")

    def unit(self, g: int) -> bytes:
        return self.rows[UNIT * g:UNIT * (g + 1)].cpu().numpy().tobytes()

    def sigma(self, seed: bytes) -> Any:
        """σ of every unit up to the plan's last, from the held rows on their device."""
        torch, units = self.torch, max(f + c for f, c in self.plan.runs)
        torch.backends.cuda.matmul.allow_tf32 = False
        sk = B.Sketch(torch, seed, self.registration.rows, units, self.rows.device)
        for j0 in range(0, UNIT * units, self.chunk):
            sk.add(self.rows[j0:min(j0 + self.chunk, UNIT * units)], j0)
        if sk.covered(self.plan.runs) != self.plan.length:
            raise RuntimeError(f"the pass sketched {sk.covered(self.plan.runs)} of the plan's {self.plan.length} units")
        return sk.sigma

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray) -> int:
        """The plan's first and last positions and `spot` sampled from the seed: σ held to `p2.Sketcher` on the unit's
        bytes and y to hashlib."""
        plan, sk = self.plan, p2.Sketcher(seed, self.registration.rows)
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        positions = sorted({0, plan.length - 1, *rng.integers(0, plan.length, self.spot).tolist()})
        for p in positions:
            g = plan.block(p)
            got = bytes(sigma[g].cpu().numpy().astype("<i4").tobytes())
            if got != sk.sketch(self.unit(g)) or bytes(ys[p]) != sk.receipt(g, got):
                raise RuntimeError(f"unit {g}'s σ or y is not the reference's")
        return len(positions)

    def run(self, challenge: coins.Released, t: int, i: int, previous: bytes) -> bytes:
        if challenge.label != coins.CHALLENGE or self.received.setdefault(challenge.index, challenge) != challenge:
            raise ValueError(f"not window {challenge.index}'s one challenge")
        seed = receipts.seed(coins.coin(challenge), i, previous)
        t0 = time.perf_counter()
        sigma = self.sigma(seed)
        a = time.perf_counter()
        parts = [B.receipts_of(self.torch, seed, sigma[f:f + c], first=f) for f, c in self.plan.runs]
        ys = parts[0] if len(parts) == 1 else np.concatenate(parts)
        b = time.perf_counter()
        spotted = self._spot(seed, sigma, ys)
        del sigma
        salts = os.urandom(SALT * receipts.leaves(self.plan.length))
        tree = B.ReceiptTree(seed, self.plan, ys, salts)
        self.passes[t, i] = (ys, salts, tree)
        rec = {"window": t, "pass": i, "sketch_s": round(a - t0, 3), "receipts_s": round(b - a, 3),
               "spot_units": spotted, "pass_s": round(time.perf_counter() - t0, 3)}
        self.steps.append(rec)
        self.log(json.dumps({"pass": rec}))
        return tree.root

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        t0 = time.perf_counter()
        out = []
        for i, p in draws:
            ys, salts, tree = self.passes[t, i]
            c, g = p // PER_LEAF, self.plan.block(p)
            unit = self.unit(g)
            out.append(p2.Opening(i, p, tuple(unit[BLOCK * b:BLOCK * (b + 1)] for b in range(UNIT)),
                                  tuple(self.tree.path(UNIT * g + b) for b in range(UNIT)),
                                  ys[PER_LEAF * c:min(PER_LEAF * (c + 1), self.plan.length)].tobytes(),
                                  salts[SALT * c:SALT * (c + 1)], tree.path(c)))
        for key in [k for k in self.passes if k[0] <= t]:
            del self.passes[key]
        rec = {"window": t, "openings": len(out), "open_s": round(time.perf_counter() - t0, 3)}
        self.opened.append(rec)
        self.log(json.dumps({"open": rec}))
        return out


@dataclass(eq=False)
class Attached:
    """``bandwidth`` on a served run beside ``storage``, PoUS's setup on the same exchange
    (`pous.prover.served.Served`), which hands it the store once it binds one (`Served.on_bound`)."""

    storage: Any
    ctx: Mapping[str, Any]
    store: Any = None

    def __post_init__(self) -> None:
        self.storage.on_bound(self.bound)

    def bound(self, store: Any) -> None:
        self.store = store

    def peer(self, node: bytes, **kw: Any) -> Held:
        """P2's prover over the store PoUS holds, under ``node``, with `Held`'s keywords; refused before PoUS binds
        one."""
        if self.store is None:
            raise RuntimeError(f"bandwidth: {self.storage.entry} holds no store yet, and the audit reads the one it "
                               "holds")
        return Held(self.store, node, **kw)


__all__ = ["Attached", "Held"]
