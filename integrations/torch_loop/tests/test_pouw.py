"""The loop's PoUW window (`verity_torch_loop.pouw`): its linears as the pinned H1 tile carry the loop's own output, a
tampering loop's tiles are not the tile's function, and PoUW's window check (`core_check`) takes the written
registration under the verifier's parameters and accepts a proof service's answer for it. Its audit (`pouw_audit`) proves
in scratch and leaves only the records under ``--out``."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.core.protocols.pouw.circuit import h1tile as Ht
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
    h1 = PW.H1Loop(L.TorchLoop(L.Model(), list(requests), steps, tamper), SALT)
    rec = Recorder()
    h1.run(rec)
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
    h1 = PW.H1Loop(L.TorchLoop(L.Model(), [3, 7, 1, 12], 1, run_id="test"), SALT)
    rec = h1.write(h1.window(), tmp_path, {"loop": "torch-loop/pouw-h1"}, salt_key=bytes(32))
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert sorted(rec["window"]) == ["context", "kind", "public"] and rec["window"]["context"] == prov["run"]
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


#: an audit's tree as `records_audit.py` leaves it (r20261026-004315-feca's), each file with whether `pouw_audit` keeps it
TREE = {"summary.json": True, "window/audit.json": True, "window/registration.json": True, "window/receipt.json": True,
        "window/program.json": True, "window/case.json": True, "window/m3/pub-8.bin": True,
        "window/m3/circuit.txt": False, "window/m3/inst-8.bin": False, "window/m3/prover/m3-pub-8.bin": False,
        "window/m3/prover/m3-inst-8.bin": False, "window/m3/rows-private.bin": False,
        "window/session-m3/archive/sha512/ab12": False, "window/session-m3/prove.log": True,
        "window/session-m3/sessions/l0000-1/session.json": True, "window/session-m3/sessions/l0000-1/circuit.rep0.bin": True,
        "window/session-m3/sessions/l0000-1/verifier.json": True}


def tree(root):
    for rel in TREE:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}" if rel.endswith(".json") else rel)


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_the_audit_keeps_its_records_and_no_circuit_archive_or_prover_file(tmp_path):
    tree(tmp_path / "work")
    PA.keep(tmp_path / "work", tmp_path / "kept")
    assert files(tmp_path / "kept") == sorted(r for r, kept in TREE.items() if kept)


def test_the_audit_proves_in_scratch_and_answers_from_what_it_kept(tmp_path, monkeypatch):
    rec = {"law": {"law": "subset", "k": 1}, "window": {"context": "ab", "kind": "pouw", "public": {
        "length": {"max_tokens": [2]}, "sampling": {"eps": "1/2", "delta": "1/1000000", "law": {"k": 1}},
        "work": {"credited": 0, "credited_tiles": 0, "n": 1}}}}
    win = tmp_path / "win"
    win.mkdir()
    (win / "registration.json").write_text(json.dumps(rec))
    (win / "provenance.json").write_text(json.dumps({"run": "ab"}))
    seen = []

    def records_audit(argv):
        work = Path(argv[argv.index("--out") + 1])
        seen.append(work)
        tree(work)
        (work / "summary.json").write_text(json.dumps({"refused": {"records": "a test"}}))
        return 1

    monkeypatch.setattr(PA.RA, "main", records_audit)
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    scratch.mkdir()
    PA.main(["--window", str(win), "--circuit", "00", "--run", "ab", "--m0-pythonpath", "", "--bin", "b", "--lean", "l",
             "--firewall", "f", "--out", str(out), "--scratch", str(scratch)])
    assert seen[0].parent == scratch and not seen[0].exists()
    assert files(out) == sorted(["pouw.json", *(f"audit/{r}" for r, kept in TREE.items() if kept)])
    got = json.loads((out / "pouw.json").read_text())
    assert got["verdict"] == "reject" and got["audit_rc"] == 1
    assert json.loads((win / "answer.json").read_text())["reason"].startswith("refused:")
