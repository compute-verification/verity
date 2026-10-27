"""The loop's PoUW window (`verity_torch_loop.pouw`): its linears as the pinned H1 tile carry the loop's own output, a
tampering loop's tiles are not the tile's function, every tile's ``a`` and ``b`` are the committed operand rows its
ports' reads name, and PoUW's window check (`core_check`) takes the written registration, its operands among it, under
the verifier's parameters and accepts a proof service's answer for it; a live window's salt session, opened on its
weights' and operands' roots, is accepted, and one opened on other operands is refused."""
from __future__ import annotations

import json
import sys
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from verity.core.protocols.pouw import core_check as CC, operands as OP, window as Wd
from verity.ml.protocols.pouw.circuit import h1tile as Ht
from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity_torch_loop import loop as L, pouw as PW, pouw_audit as PA

SALT = bytes(range(64))


class Recorder:
    """A window's `record`, kept."""

    def __init__(self) -> None:
        self.units: list[tuple[list[np.ndarray], list[np.ndarray], dict]] = []

    def record(self, template, inputs, outputs, label=None):
        assert template.id in CC.TILES
        self.units.append((inputs, outputs, label))


def recorded(tamper: bool = False, requests=(3, 7, 1, 12), steps: int = 2) -> tuple[PW.H1Loop, Recorder]:
    h1 = PW.H1Loop(L.TorchLoop(L.Model(), list(requests), steps, tamper))
    rec = Recorder()
    h1.run(rec, SALT)
    return h1, rec


def reference(inputs) -> tuple[np.ndarray, np.ndarray]:
    a, b, _salt, pos, roots = (np.asarray(x[0]) for x in inputs)
    u = (int(pos[0]) << 32) | int(pos[1])
    rs = tuple(b"".join(int(w).to_bytes(4, "big") for w in r) for r in roots)
    z, rc = Ht.reference(a.tolist(), b.tolist(), SALT, u, rs)
    return np.asarray(z, np.int64).astype(np.uint32), np.asarray(rc, np.int64).astype(np.uint32)


def test_the_pinned_tile_carries_the_loops_linears():
    h1, rec = recorded()
    assert PW.tile().id in CC.TILES and CC.TILES[PW.tile().id].tile_work == PW.TILE_WORK
    assert h1.real == 2 * 2 * 1 * (L.H // PW.TILE_C) * (L.H // PW.TILE_K) == 128
    assert h1.work == Wd.bucket(128) and len(rec.units) == h1.work.n == 144
    assert [u for u, (_i, _o, label) in enumerate(rec.units) if "filler" in label] == list(range(128, 144))
    for inputs, outputs, _ in rec.units[::17]:
        z, rc = reference(inputs)
        assert np.array_equal(outputs[0][0], z) and np.array_equal(outputs[1][0], rc)


def test_every_tile_reads_its_operands_from_the_committed_rows():
    """Each unit's ``a`` and ``b`` are the rows of the committed operand its port's read names, at the read's position: what
    the members commit under the operand's salts and Lean's registered reads open against its root."""
    h1, rec = recorded(requests=(3, 7, 1, 12, 5), steps=2)
    words = dict(PW.IF.layout(PW.tile()).rows)
    reads = h1.circuit().reads
    assert sorted((r.port, r.param) for r in reads) == [("a", "a"), ("a", "a"), ("b", "b"), ("b", "b")]
    assert h1.roots == (h1.values["a"].root, h1.values["b"].root)
    seen = {"a": set(), "b": set()}
    for u, (inputs, _outputs, _label) in enumerate(rec.units):
        port = {"a": inputs[0][0], "b": inputs[1][0]}
        for r in (r for r in reads if r.at <= u < r.at + r.reader_units):
            pos = int(r.source(np.asarray([u - r.at]))[0]) // words[r.port]
            assert PW._words(port[r.port]) == list(h1.values[r.param].rows[pos]), (u, r.port)
            seen[r.port].add(pos)
    assert seen == {"a": set(range(len(h1.a_rows))), "b": set(range(len(h1.b_rows)))}


def test_the_operands_are_fixed_before_the_salt():
    """The operands and their root X come from the served loop alone; the salt changes the tiles' receipts, not them."""
    loop = L.TorchLoop(L.Model(), [3, 7, 1, 12], 1, run_id="x")
    h1 = PW.H1Loop(loop)
    again = PW.H1Loop(loop, salts={v: r.salts for v, r in h1.values.items()})
    x = h1.X
    one, two = Recorder(), Recorder()
    h1.run(one, SALT)
    h1.run(two, bytes(64))
    assert h1.X == x == OP.root(h1.operands) and one.units[0][1][1].tolist() != two.units[0][1][1].tolist()
    assert again.X == x
    other = PW.H1Loop(L.TorchLoop(L.Model(), [3, 7, 1, 12], 1, run_id="y"))
    assert other.operands["values"]["a"]["domain"] != h1.operands["values"]["a"]["domain"]


def test_a_tampering_loops_tiles_are_not_the_tiles():
    _, rec = recorded(tamper=True)
    for inputs, outputs, label in rec.units[::9]:
        z, _ = reference(inputs)
        assert np.array_equal(outputs[0][0] ^ z, np.full_like(z, 0 if "filler" in label else 1))


def test_a_request_count_off_the_tile_pads_with_zero_rows():
    h1, rec = recorded(requests=(3, 7, 1), steps=1)
    assert h1.real == 2 * 1 * 16 * 2 and not rec.units[0][0][0][0][3].any()


@pytest.mark.slow
def test_core_registers_the_written_window_and_accepts_a_service_answer_for_it(tmp_path):
    """Builds and writes the loop's H1 window (about a minute: the population's port check and the commitment)."""
    h1 = PW.H1Loop(L.TorchLoop(L.Model(), [3, 7, 1, 12], 1, run_id="test"))
    rec = h1.write(h1.window(SALT), tmp_path, {"loop": "torch-loop/pouw-h1"}, salt_key=bytes(32))
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert sorted(rec["window"]) == ["activations", "context", "kind", "public"] and rec["window"]["context"] == prov["run"]
    assert OP.held(rec["window"]) == h1.operands["values"] and OP.root(rec["window"]["activations"]) == h1.X
    assert (tmp_path / PW.OPERANDS_PRIVATE).stat().st_mode & 0o777 == 0o600
    mine, _doc, _sets = PA.verifier_window(tmp_path, rec["law"], None)
    assert mine.partition_digest == rec["partition"] and mine.population == h1.work.n
    p = h1.params()
    c = CC.commitment(CC.line(rec).encode())
    assert CC.registered(CC.params(p), c) is None
    assert CC.TILES[PW.tile().id].query == rec["query"]
    units = list(range(h1.k))
    draw = {"k": h1.k, "law": "subset", "population": h1.work.n, "units": units}
    profile = A.profile(rec, draw, float(PW.DELTA)).to_json() | {"delta": str(PW.DELTA)}
    ans = {"commitment": R.digest(rec), "profile": profile, "protocol": R.PROTOCOL, "public": {"draw": draw},
           "reason": None, "verdict": "accept"}
    got = CC.answer(CC.line(p).encode(), CC.line(rec).encode(), CC.line(ans).encode())
    assert got["verdict"] == "accept", got["reason"]
    assert got["public"]["work"]["credited"] == Wd.bucket(h1.real).credited * PW.TILE_WORK
    other = {**ans, "profile": {**profile, "delta": str(Fraction(1, 2))}}
    assert CC.answer(CC.line(p).encode(), CC.line(rec).encode(), CC.line(other).encode())["verdict"] == "reject"
    eps = {**p, "eps": "1/4"}
    assert CC.answer(CC.line(eps).encode(), CC.line(rec).encode(), CC.line(ans).encode())["verdict"] == "reject"


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """A local gate under `tmp_path` (`window_salt.local`), its challenger under live_challenger's stub of the Lean
    verifier's coins, so the test builds no `sampled-proofs-verifier`."""
    WS = PW.window_salt()
    AC = WS.a5_coins()
    auditor = AC._module("a5_auditor", WS.TREE / "benchmarks" / "live_challenger" / "auditor.py")
    real, stub = auditor.serve_argv, WS.TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"
    monkeypatch.setattr(auditor, "serve_argv", lambda *a, **k: [sys.executable, str(stub), *real(*a, **k)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = [AC.on_main([])]
    return WS, WS.local(tmp_path / "host", main), main


@pytest.mark.slow
@pytest.mark.parametrize("late", [False, True], ids=["salt-after-operands", "operands-after-salt"])
def test_a_live_windows_salt_is_released_against_its_operands(tmp_path, gate, late):
    """Builds and writes a live H1 window (about a minute): its registration names the pinned weights and the operands
    and no W root, its salt session opens on R over their roots, and PoUW accepts its answer; a session opened on other
    operands (the prover took the salt, then committed the rows its tiles read) is refused at the salt."""
    WS, g, main = gate
    loop = L.TorchLoop(L.Model(), [3, 7], 1, run_id="live")
    weights = PW.weights_group(loop.model, L.commit_weights(loop.model))
    h1 = PW.H1Loop(loop, weights=weights, live=True)
    anchors = h1.anchors()
    if late:
        early = OP.block(h1.program, OP.commit(h1.program, loop.run_root,
                                               {v: [[0] * e["words"]] for v, e in h1.operands["values"].items()}))
        anchors = {**anchors, "window": {**anchors["window"], OP.KEY: early}}
    issued = WS.issue(g, anchors, 0, main, tmp_path / "salt-session")
    rec = h1.write(h1.window(issued.salt), tmp_path / "pouw", {"loop": "torch-loop/pouw-h1"}, salt_key=bytes(32),
                   issued=issued)
    assert json.loads((tmp_path / "pouw" / PW.SALT).read_text()) == {"answer": issued.answer, "live": issued.live}
    assert rec["window"]["registered"] == weights and not any(r["port"] == "W" for r in rec["roots"])
    assert CC.window_anchors(rec) == (CC.values_root(weights), h1.X)
    p = h1.params(issued.live)
    assert p["check"][Wd.COINS] == "live" and CC.registered(CC.params(p), CC.commitment(CC.line(rec).encode())) is None
    draw = {"k": h1.k, "law": "subset", "population": h1.work.n, "units": list(range(h1.k))}
    profile = A.profile(rec, draw, float(PW.DELTA)).to_json() | {"delta": str(PW.DELTA)}
    ans = {"commitment": R.digest(rec), "profile": profile, "protocol": R.PROTOCOL,
           "public": {"draw": draw, "salt": issued.answer}, "reason": None, "verdict": "accept"}
    got = CC.answer(CC.line(p).encode(), CC.line(rec).encode(), CC.line(ans).encode())
    if late:
        assert got["verdict"] == "reject" and got["reason"].startswith("salt: the challenger's session is not trusted"), got
    else:
        assert got["verdict"] == "accept", got["reason"]
        unsalted = {**ans, "public": {"draw": draw}}
        assert CC.answer(CC.line(p).encode(), CC.line(rec).encode(), CC.line(unsalted).encode())["verdict"] == "reject"
