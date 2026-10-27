"""HIGHUTIL's period on the CPU: the pick draw, ε, the operand openings, and whole periods at small shapes through v31.py's
reference prover (`period.RefDriver`) with an in-process coin source: an honest period is accepted, and a skipping
prover's, a tampered root's and a swapped operand's are refused."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import period as P


def test_eps_skip_at_delta_one_percent():
    assert math.isclose(P.eps_skip(2048), math.log(100) / (P.ETA * 2048))
    assert P.eps_skip(462) < 0.01 < P.eps_skip(460)


def test_picks_are_the_coin_s_and_weigh_each_mac_equally():
    P.configure({"layers": [["qkv", 6144, 4096], ["o", 4096, 4096], ["up", 28672, 4096], ["down", 4096, 14336]], "m": 16384,
                 "nb": 32})
    coin, receipt = b"\x01" * 64, b"\x02" * 64
    a = P.picks(coin, "p", 3, receipt, 300, 400, 32768)
    assert a == P.picks(coin, "p", 3, receipt, 300, 400, 32768)
    assert a != P.picks(coin, "p", 3, b"\x03" * 64, 300, 400, 32768)
    w = np.array([n * k for _, n, k in P.LAYERS], dtype=float)
    counts = np.bincount([p.layer for p in a], minlength=len(w))
    assert np.all(np.abs(counts - 400 * w / w.sum()) < 5 * np.sqrt(400 * w / w.sum()) + 1)
    for p in a:
        _, n, k = P.LAYERS[p.layer]
        sh = P.shape(n, k, 32768)
        assert 0 <= p.step < 300 and 0 <= p.i < 16384 and 0 <= p.j < n and 1 <= p.s <= sh.S


def test_seeded_activations_look_quantized_and_rarely_reach_full_range():
    a = P.seeded_activations(64, 4096, 5).astype(np.int64)
    assert a.min() >= -127 and a.max() <= 127
    assert 0 < (np.abs(a) > 32).mean() < 2 / 1024
    assert 7 < a[np.abs(a) <= 32].std() < 11


@dataclass
class _Opening:
    label: str
    index: int
    after: bytes
    coin: bytes


class _Client:
    def __init__(self, secret: bytes) -> None:
        self.secret, self.opened = secret, set()

    def open(self, label: str, index: int, after: bytes) -> _Opening:
        assert (label, index) not in self.opened, "a coin opens once"
        self.opened.add((label, index))
        return _Opening(label, index, after, hashlib.sha512(self.secret + label.encode() + P.u64(index) + after).digest())

    def open_many(self, requests):
        return [self.open(*r) for r in requests]

    def close(self) -> None:
        pass


class _Coins:
    """An in-process coin source: each coin opens once, after what it was asked after."""

    class _Session:
        session = "in-process"

        def open(self, root, layout, minutes, scheme):
            return "in-process", root[:64], None

    def session(self):
        return self._Session()

    def connect(self, address, secret):
        return _Client(bytes.fromhex(secret))

    def evidence(self, session, root, openings, dest):
        return {"trusted": True, "bad_openings": [], "session": session.session}


class _Serial:
    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("cfg")
    assert P.main(["prep", str(cfg), "--small"]) == 0
    prep = json.loads((cfg / "prep.json").read_text())
    P.configure(prep)
    secret = b"\x07" * 64
    trees = P.operand_trees(cfg, secret, _Serial())
    drv = P.RefDriver(cfg, prep["row"])
    work = tmp_path_factory.mktemp("work")
    return P.Period(drv, _Coins(), trees, cfg, prep["row"], prep["source"], secret, work, _Serial())


def test_every_operand_row_opens_against_its_root_and_no_other(small):
    P.configure(json.loads((small.cfg / "prep.json").read_text()))
    for (kind, li, t), tree in small.trees.items():
        name, n, k = P.LAYERS[li]
        rows = P.M if kind == "A" else n
        X = P.operand(small.cfg, kind, li, t)
        j = (7 * li + 3 * t + 1) % rows
        op = tree.opening(X, j)
        assert P.operand_reason(kind, name, t, rows, k, tree.root, j, op) is None
        other = small.trees[(kind, li, 1 - t)]
        assert P.operand_reason(kind, name, t, rows, k, other.root, j, op)
        bad = (bytes([op[0][0] ^ 1]) + op[0][1:],) + op[1:]
        assert P.operand_reason(kind, name, t, rows, k, tree.root, j, bad)


def test_an_honest_period_is_accepted(small):
    """The whole period at small shapes, over both blocks of the model: coins, seeds, commits, paths, answers and the
    verifier's checks."""
    r = small.proto("t-honest", P.Plan(2, 2, 12))
    assert r["accepted"], r["verdict"]
    assert r["credited_macs"] == r["macs"] == 4 * P.step_macs() and r["q"] == 12
    assert r["verdict"].startswith("ACCEPT")
    assert len(r["registration"]["operand_roots"]["A"][0]) == P.NB == 2


def test_a_skipping_prover_is_refused(small):
    r = small.proto("t-skip", P.Plan(2, 2, 12), "skip:1@1")
    assert not r["accepted"] and r["credited_macs"] == 0 and r["failed"] > 0, r["verdict"]
    assert r["verdict"].startswith("REFUSE")


def test_a_tampered_root_is_refused(small, monkeypatch):
    """A root the GPU didn't commit (here the first call's, altered after the block) fails every pick of that call."""
    real = small.drv.block

    def block(b, g0, steps, arm, cheat, seeds, roots):
        got = real(b, g0, steps, arm, cheat, seeds, roots)
        raw = bytearray(Path(roots).read_bytes())
        raw[0] ^= 1
        Path(roots).write_bytes(bytes(raw))
        return got
    monkeypatch.setattr(small.drv, "block", block)
    r = small.proto("t-root", P.Plan(1, 1, 24))
    first = [p for p in r["failures"] if "layer=0" in p]
    assert not r["accepted"] and first


class _FirstBlockOnly(P.RefDriver):
    """A prover that runs every step on block 0's operands, never the registered block 1's."""

    def block_of(self, step: int) -> int:
        return 0


def test_computing_on_another_block_s_operands_is_refused(small, monkeypatch):
    """The block-1 step's roots commit values its registered operands don't produce, so its picks fail (all 24 picks
    missing that step: 2^-24)."""
    monkeypatch.setattr(small, "drv", _FirstBlockOnly(small.cfg, small.row))
    r = small.proto("t-swap", P.Plan(1, 2, 24))
    assert not r["accepted"] and 0 < r["failed"] < 24, r["verdict"]
