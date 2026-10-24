"""n2_advised_zk's link and rows on the advised tests' traffic (`advised_traffic`). At the driver's T = 512 the session
is varied, every window receives, its honest windows pass the unit's reference and N2's host verdict of record, and
each negative (a moved release bucket, an output length one short, extra egress) fails in some window on the advised
counts alone, its head checks passing and every view taken; the verifier's own ``pub`` row is the prover's. At the
tests' T = 8 the adapter, evaluated on the rows the prover stages, is the unit's reference (both head checks and the
advised counts on the counts the heads hold) in every window of every member and on a head the prover changed. Each
builder `n2_advised_zk.builders` attaches, called as C-Flock's lowering calls it, traces to its part's own gates.
Nothing here lowers or proves (`n2_advised_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import advice as ADV, head_check as HC

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_advised_zk", HERE / "n2_advised_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n2 = _load()
AT = n2.AT


def _clock() -> float:
    return 0.0


@cache
def _link(T: int):
    shape = n2.SHAPE if T == n2.T else AT.SHAPE
    sim = n2.simulated(n2.SEED, shape)
    return sim, {name: n2.link_windows(sim.lw, link, pairs, clock=_clock) for name, (link, pairs, _) in sim.members.items()}


def test_the_fixture_is_varied_and_every_window_receives():
    sim, _ = _link(n2.T)
    s, a = sim.session, sim.session.link.advice
    rows = [r for st in a.steps for r in st.rows]
    assert {AT.FRAMING[q.prompt_len - 1] for q in a.requests} == {1, 2, 3} and len({q.output_len for q in a.requests}) > 3
    assert any(r.kind == "prefill" and r.start + r.n < a.requests[r.request].prompt_len for r in rows)
    assert any(r.cache_hit and not r.preempted for r in rows) and any(r.preempted and r.cache_hit for r in rows)
    assert any([r.request for r in st.rows] != sorted(r.request for r in st.rows) for st in a.steps)
    assert any(b // n2.T > h // n2.T for h, b in zip(s.hand, a.release))
    assert max(len({b // n2.T for o, b in zip(ADV.outputs(a), a.release) if o.request == j})
               for j in range(len(a.requests))) > 1
    assert all(s.link.received) and len(s.link.received) == n2.WINDOWS
    assert sim.lw.unit.id == "NetworkAdvisedLinkWindow_v1{EGRESS=(9,512,0),INGRESS=(9,512,0),ADVISED=(512,4,1)}"
    assert n2.key_of(sim.lw) == (9, 512, 4, 1)


def test_honest_passes_and_each_negative_fails_on_the_advised_counts_alone():
    _, rounds = _link(n2.T)
    assert list(rounds) == list(n2.MEMBERS)
    for name, rd in rounds.items():
        fails = [n2.failed(x) for x in rd.windows]
        assert n2.expected(name, fails), (name, fails)
        assert [x.host["accepted"] for x in rd.windows] == [bool(x.ref) for x in rd.windows], name
        assert rd.network["accepted"] is (name == "honest")
    assert not n2.expected("moved", [[]] * n2.WINDOWS)
    assert not n2.expected("extra", [["egress/signed", "advised"]] + [[]] * (n2.WINDOWS - 1))


def test_the_verifier_derives_the_public_row():
    sim, rounds = _link(n2.T)
    key = n2.key_of(sim.lw)
    width = n2.row_bits(SimpleNamespace(**dict(zip(n2.STATICS, key))))
    for name, rd in rounds.items():
        for x in rd.windows:
            got = n2.rows_of(key, x.checks, x.view)
            assert {r: 16 * len(w) for r, w in got.items()} == width
            signed = tuple(r.certificate.digest for r in x.opening.records)
            assert n2.public_words(sim.lw, x.window, bytes.fromhex(x.root), signed, x.view) == got["pub"], (name, x.window)


def _bits(words: list[int]) -> list[int]:
    return [(w >> j) & 1 for w in words for j in range(16)]


def _evaluate(key, windows) -> list[int]:
    rows = [n2.rows_of(key, x.checks, x.view) for x in windows]
    args = [np.asarray([_bits(r[name]) for r in rows], dtype=np.uint8) for name in n2.ROWS]
    return [int(b) for b in np.asarray(evaluate_bits(n2.unit(key), *args)).reshape(-1).tolist()]


@pytest.mark.slow
def test_the_adapter_on_the_staged_rows_is_the_units_reference():
    """At T = 8: every member's windows, and the honest records with window 0's egress head one count off, committed by
    the prover (its certificate the original's), which the egress head check refuses. About 85 s, nearly all of it
    building the head check's evaluation; it is the one test that evaluates the circuit the node proves, so a row
    sliced off by a bit, a head's counts cut at the wrong offset or the ends swapped into the advised counts fail here
    and pass every other test."""
    sim, rounds = _link(AT.T)
    key = n2.key_of(sim.lw)
    link, pairs, _ = sim.members["honest"]
    tampered = n2.link_windows(sim.lw, link, [n2.Z.one_count(sim.lw, w, p) for w, p in enumerate(pairs)], stream=False)
    assert "egress/signed" in n2.failed(tampered.windows[0])
    windows = [x for rd in (*rounds.values(), tampered) for x in rd.windows]
    assert _evaluate(key, windows) == [x.ref for x in windows]
    assert [x.ref for x in rounds["honest"].windows] == [1] * AT.WINDOWS
    assert all(0 in [x.ref for x in rd.windows] for name, rd in rounds.items() if name != "honest")


@pytest.mark.slow
def test_each_advised_builder_traces_to_its_parts_gates_called_as_c_flock_calls_it():
    parts = n2.builders()
    assert {"NetworkAdvisedOwn", "NetworkAdvisedBucket", "NetworkAdvisedEnd"} <= {d.name for d in parts}
    for d in parts:
        if not d.name.startswith("NetworkAdvised"):
            continue
        key = (n2.T, 4)
        S = SimpleNamespace(**dict(zip(d.statics, key)))
        params, ret = d.signature(S)
        got, want = TR.trace(lambda C, *e, d=d, S=S: d.build(C, S, *e), [t for _, t in params], ret), d.traced(*key)
        assert (got.gates, got.outputs) == (want.gates, want.outputs), d.name


def test_the_scan_names_both_ends_pieces():
    _, rounds = _link(n2.T)
    blobs, held = n2.blobs_of(rounds["honest"])
    for end in n2.ENDS:
        assert {f"{end} head {w}" for w in range(n2.WINDOWS)} <= held
        assert f"{end} certifier salt 0" in blobs and f"{end} sibling 1 0" in blobs
    assert held == {k for k in blobs if " head " in k and " salt" not in k}
    assert HC.Shape(9, n2.T).HL == len(blobs["egress head 0"])
