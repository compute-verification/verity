"""n2_zk's instances and rows on certifier-serve's runs on an injected clock (`n2_zk.virtual`, a stub endpoint): every
honest head check passes its reference; window 0's egress head one count off, committed by the prover, fails on its
counts and its signed digest; the extra run's egress heads fail on their counts alone. The adapter's rows carry the head
check's arguments in its signature's order: around an identity of the head check's signature it returns them. The
verifier's own ``pub`` row is the prover's. Nothing here lowers or proves (`n2_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.core.protocols.certifier import head_check as HC

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_zk", HERE / "n2_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n2 = _load()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("n2zk")
    plan = n2.virtual(out, requests=8, tokens=4)
    return {"plan": plan, "honest": n2.link_windows(out / "honest", stream=False),
            "tampered": n2.link_windows(out / "honest", edit=n2.one_count, stream=False),
            "extra": n2.link_windows(out / "extra", stream=False)}


def _failed(inst) -> list[str]:
    return [k for k, ok in HC.reference_parts(inst).items() if not ok]


def test_honest_heads_pass_and_a_head_one_count_off_fails_on_its_counts(runs):
    honest = runs["honest"]
    assert len(honest.windows) == runs["plan"]["windows"] >= 2 and not honest.issuer_refused
    assert honest.lw.head_key(0)[2] == 1 and honest.lw.head_key(1)[2] == 0
    assert all(c is not None and HC.reference(c) for x in honest.windows for c in x.checks)
    for name in ("tampered", "extra"):
        rd = runs[name]
        assert n2.expected(name, [_failed(x.checks[0]) for x in rd.windows]), name
        assert all(HC.reference(x.checks[1]) for x in rd.windows), name
    e0, t0 = honest.windows[0].opening.records[0].head, runs["tampered"].windows[0].opening.records[0].head
    assert sum(a != b for a, b in zip(e0.counts, t0.counts)) == 1 and sum(t0.counts) == sum(e0.counts) + 1
    assert runs["tampered"].windows[0].opening.records[0].certificate == honest.windows[0].opening.records[0].certificate


@cache
def _identity(L: int, T: int, counts: int):
    """A Definition of the head check's signature that returns its arguments' bits, in order."""
    params, _ = HC.unit(L, T, bool(counts)).signature()
    ret = Array(sum(t.leaves for _, t in params), BL.BIT)
    return bind(CompositeDefinition("N2ZkIdentity", 1, (), lambda S: (tuple(params), ret),
                                    lambda B, S, *args: Coll(ret, concat([a.refs for a in args])), register=False))


def _bits(words: list[int]) -> list[int]:
    return [(w >> j) & 1 for w in words for j in range(16)]


def test_the_rows_hold_the_head_checks_arguments_and_the_verifier_derives_the_public_row(runs):
    rows = n2.adapter("N2ZkIdentityRows", _identity)
    for name in ("honest", "tampered"):
        rd = runs[name]
        for x in rd.windows:
            for d, inst in enumerate(x.checks):
                got = n2.rows_of(inst)
                width = n2.row_bits(inst.unit)
                assert {r: 16 * len(w) for r, w in got.items()} == width and all(v % 16 == 0 for v in width.values())
                fn = bind(rows, L=inst.key[0], T=inst.key[1], COUNTS=int(inst.key[2]))
                assert [t.leaves for _, t in fn.signature()[0]] == [width["pub"], width["wit"]]
                args = [np.asarray([_bits(got[r])], dtype=np.uint8) for r in n2.ROWS]
                out = np.asarray(evaluate_bits(fn, *args)).reshape(-1).tolist()
                assert out == [b for a in HC.arguments(inst) for b in a], (name, x.window, d)
                mine = n2.public_words(rd.lw, x.window, d, bytes.fromhex(x.root), x.opening.records[d].certificate.digest)
                assert mine == got["pub"], (name, x.window, d)
    assert HC.Shape(*runs["honest"].lw.head_key(0)[:2]).HL % 2 == 1
