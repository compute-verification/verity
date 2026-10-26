"""Onsite's real components' cases for the composition's difftest, each with the answer its Python verifier gives it
(`reference.Recorded`): a small period of each on node `n0`, made by its hook. Space's (`pous.period.run`, signed by a
key at S1's deployed parameters, `mc_cases.deployed_signer`) honest and under a prover that answers every other round
with zeros; bandwidth's (`remote_bandwidth.period.run`, one window) honest, under a prover whose store dropped every
other block, and stopped before its window is verified; and of each, its honest case with params or evidence that don't
decode."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import reference as R
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period as pous
from verity.core.protocols.remote_bandwidth import audit, codec
from verity.core.protocols.remote_bandwidth import period as bandwidth

ROOT = Path(__file__).resolve().parents[5]
STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(2 * 4032 - 17)
"""Two blocks of the bandwidth codec's 4,032 payload bytes."""
POUS_STORE = STORE[:500]
"""Four blocks of P2's 126 payload bytes at the small point's width, challenged `POUS_K` times: the composition reads
only the component's answer, so its case is kept small (the hook's own point is `pous_check`'s difftest's)."""
POUS_K = 8
COINS = hashlib.sha512(b"the period's live coins").digest()


def _mc_cases():
    spec = importlib.util.spec_from_file_location("mc_cases", ROOT / "verity/core/devices/memory_challenger/lean/tests/mc_cases.py")
    module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    """The bandwidth verifier's clock: each reading a microsecond after the last, and waiting moves it on."""

    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1_000
        return self.now

    def sleep_until(self, clock, t: int) -> None:
        self.now = max(self.now, t)


def _undecodable(cid: str, case: dict, digest: str) -> dict:
    return {"broken-params": ({**case, "params": {}}, P.outcome(cid, R.REJECT, "params: not its params")),
            "broken-evidence": ({**case, "evidence": "00"},
                                P.outcome(cid, R.REJECT, "evidence: not JSON", bytes.fromhex(digest)))}


def build(tmp: Path) -> None:
    """Fill `reference.POUS` and `reference.BANDWIDTH` with their cases, the memory challenger's session logs in `tmp`."""
    M = _mc_cases()
    signer = M.deployed_signer()

    def space(name: str, **kw) -> pous.Period:
        mc.SessionLog.create(tmp / name)
        return pous.run("n0", POUS_STORE, COINS, signer=signer, log=mc.SessionLog(tmp / name), clock=M.Clock(10 ** 9),
                        point=pous.point_for(len(POUS_STORE), k=POUS_K), **kw)

    def reads(**kw) -> bandwidth.Period:
        c = Clock()
        return bandwidth.run("n0", STORE, COINS, windows=1, clock=c, sleep_until=c.sleep_until,
                             entropy=_entropy(b"salts"), **kw)

    honest = space("honest")
    zeros = space("zeros", prover=lambda blocks: lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i])
    R.POUS.cases.update({R.ACCEPT: (honest.case, honest.answer), R.REJECT: (zeros.case, zeros.answer)}
                        | _undecodable(R.POUS.id, honest.case, honest.answer["commitment"]))
    honest, dropping, unverified = reads(), reads(cheat=_dropping), reads(verify=0)
    R.BANDWIDTH.cases.update({R.ACCEPT: (honest.case, honest.answer), R.REJECT: (dropping.case, dropping.answer),
                              R.UNDECIDED: (unverified.case, unverified.answer)}
                             | _undecodable(R.BANDWIDTH.id, honest.case, honest.answer["commitment"]))
    for C in (R.POUS, R.BANDWIDTH):
        got = {k: a["verdict"] for k, (_, a) in C.cases.items()}
        assert all(v == (k if k in (R.ACCEPT, R.REJECT, R.UNDECIDED) else R.REJECT) for k, v in got.items()), (C.id, got)


def _entropy(label: bytes):
    """SHA-512 in counter mode under `label`, in place of the operating system's generator."""
    count = 0

    def draw(n: int) -> bytes:
        nonlocal count
        out = b""
        while len(out) < n:
            out += hashlib.sha512(label + count.to_bytes(8, "big")).digest()
            count += 1
        return out[:n]
    return draw


def _dropping(store: audit.Store) -> audit.Store:
    """The store with every other encoded block zeroed: its prover reads blocks it no longer holds."""
    out = copy.copy(store)
    out.blocks = [bytes(codec.BLOCK_BYTES) if j % 2 else b for j, b in enumerate(store.blocks)]
    return out
