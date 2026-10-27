"""HIGHUTIL's period on the CPU: the activations' twin of hu.cu's generator, the pick draw, ε, and whole periods at small
shapes through v31.py's reference prover (`period.RefDriver`) with an in-process coin source: an honest period is
accepted, a skipping prover's is refused."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import period as P

MASK = (1 << 64) - 1


def act_value(key: int, call: int, e: int) -> int:
    """hu.cu's act_value, in Python integers."""
    def fmix(z: int) -> int:
        z ^= z >> 30
        z = z * 0xBF58476D1CE4E5B9 & MASK
        z ^= z >> 27
        z = z * 0x94D049BB133111EB & MASK
        return z ^ (z >> 31)
    h = fmix(key ^ (((call << 32 | e) * 0x9E3779B97F4A7C15) & MASK))
    s = sum((h >> (8 * b)) & 255 for b in range(4)) - 510
    if (h >> 32) & 1023 == 0:
        o = (h >> 40) & 255
        o = o - 256 if o > 127 else o
        return max(o, -127)
    return s >> 4


def test_activations_are_hu_cu_s_generator_entry_by_entry():
    key, call, k = 0x0123456789ABCDEF, 77, 4096
    a = P.act_rows(key, call, range(5, 9), k)
    for r in range(4):
        for c in (0, 1, 17, 4095):
            assert a[r, c] == act_value(key, call, (5 + r) * k + c)
    big = P.act_rows(key, call, range(0, 64), k).astype(np.int64)
    outliers = np.abs(big) > 32
    assert 0 < outliers.mean() < 2 / 1024
    assert big.min() >= -127 and big.max() <= 127
    # every outlier is one of the generator's (bits 32..41 zero), so the scalar twin agrees on them too
    r, c = np.argwhere(outliers)[0]
    assert big[r, c] == act_value(key, call, int(r) * k + int(c))


def test_eps_skip_at_delta_one_percent():
    assert math.isclose(P.eps_skip(2048), math.log(100) / (P.ETA * 2048))
    assert P.eps_skip(462) < 0.01 < P.eps_skip(460)


def test_picks_are_the_coin_s_and_weigh_each_mac_equally():
    P.configure({"layers": [list(x) for x in P.LAYERS], "m": 16384})
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
    trees = [P.WeightTree.build(name, np.fromfile(cfg / f"B-{name}.bin", dtype=np.int8).reshape(n, k), secret[:32] + name.encode())
             for name, n, k in P.LAYERS]
    actkey = 0x5EED
    drv = P.RefDriver(cfg, prep["row"], actkey)
    work = tmp_path_factory.mktemp("work")
    return P.Period(drv, _Coins(), trees, cfg, prep["row"], actkey, b"\x09" * 64, secret, work, _Serial())


def test_an_honest_period_is_accepted(small):
    """The whole period at small shapes: coins, seeds, commits, paths, answers and the verifier's checks."""
    r = small.proto("t-honest", P.Plan(2, 2, 12))
    assert r["accepted"], r["verdict"]
    assert r["credited_macs"] == r["macs"] == 4 * P.step_macs() and r["q"] == 12
    assert r["verdict"].startswith("ACCEPT")


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
