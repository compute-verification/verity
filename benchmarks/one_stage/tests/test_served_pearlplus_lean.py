"""The served Pearl+ flow's Lean step end to end (`served_pearlplus.sh` step 4b) with the real Lean verifier of record,
PouwCheck.PearlPlus's ``pouw-pearlplus`` (PR #7221's package, built here as its difftest builds it), and the row PoUW's
compose then makes of the piece.

A toy serve stands in for the GPU's: a warm-up pass and two timed passes of one epoch (the pouw suite's `proved_pass`, the
H100 record's -h3 scheme at k = 128 in 16 x 16 tiles), the commands after it the flow's own: ``served_passes.py retain
--window-record W`` (the warm-up dropped, W over the two timed rounds), ``verify --tie W --test-instance`` on the node's
draw, its coins fixed, and ``lean``, which writes the Lean's answer to the piece's ``pearlplus-lean.json`` only where it
is the Python's.

The compose's earlier conditions (the live salt session, the lane's period and reserve, the live draw and its published
coin, the window's registration) are `test_served_verdicts._pearlplus_piece`'s stand-ins, since this toy's coins are an
in-process challenger's and its draw the node's (a live run is `pearlplus_local_e2e.py`'s); Pearl+'s answer, W and the
Lean's answer are this run's. The Lean is given W (``--tie W``) and holds the passes to it, the warm-up's round among
the coin record's dropped. The row then reaches the Lean condition and reads the Lean's answer: without one it names
none; with it, every condition up to the credit holds (the Lean's verdict, run, source and W) and the row stops at the
last, the Lean's credit, since a test instance's passes credit nothing.

What it catches that the pouw suite's stand-in tests can't: #7221's command line and printout as they are, so a head that
changes either fails here."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "verity" / "protocols" / "pouw" / "verifier" / "lean" / "pearlplus"
H100 = "pearl-c-h100-v1-h3"
SOURCE = b"verifier" * 4
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"),
              pytest.mark.skipif(not (PACKAGE / "lakefile.toml").exists(),
                                 reason="no PouwCheck.PearlPlus package in this tree (PR #7221)")]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


TV = _module("test_served_verdicts", HERE / "test_served_verdicts.py")
S = TV.S
T = _module("test_pearl_c_vllm", ROOT / "benchmarks" / "pouw" / "tests" / "test_pearl_c_vllm.py")
SP = T.load("pearl_c_vllm_served_passes", T.KERNEL.parent / "pearl_c_vllm" / "served_passes.py")


def test_the_lean_step_writes_the_real_lean_answer_on_the_served_run_and_the_compose_reads_it(tmp_path, monkeypatch):
    import verity_difftest as D
    from verity.protocols.pouw.verifier import verifier as PV
    from verity.protocols.spot.verifier.one_stage import registration as REG

    monkeypatch.setattr(PV, "coins", lambda n: SOURCE[:n])
    for k in ("PEARLC_CHALLENGER_GATE", "PEARLC_CHALLENGER_PEER"):
        monkeypatch.delenv(k, raising=False)
    warm, a, b = (T.proved_pass(tmp_path / name, H100, ix)[0] for name, ix in (("w", (0, 1)), ("a", (2, 3)), ("b", (4, 5))))
    d = T._serve_dir(tmp_path, [warm, a, b])
    served = json.loads((d / "served.json").read_text())
    served["passes"] = [{**served["passes"][0], "timed": False}, *({**p, "timed": True} for p in served["passes"][1:])]
    (d / "served.json").write_text(json.dumps(served))
    wfile, node = tmp_path / "window-record.json", tmp_path / "node"
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(wfile)]) == 0
    w = REG.digest(json.loads(wfile.read_text()))
    assert SP.main(["verify", "--passes", str(d), "--out", str(node / "verify.json"), "--answer",
                    str(node / S.PEARLPLUS_ANSWER), "--jobs", "1", "--draws", "16", "--tie", str(wfile),
                    "--test-instance"]) == 0
    answer = json.loads((node / S.PEARLPLUS_ANSWER).read_text())
    assert answer["public"]["tie"]["record"] == w and answer["public"]["rounds"] == [1, 2]

    exe = D.build(PACKAGE, "pouw-pearlplus")
    rc = SP.main(["lean", "--passes", str(d), "--verify", str(node / "verify.json"), "--exe", str(exe), "--out",
                  str(node / S.PEARLPLUS_LEAN)])
    check = json.loads((node / SP.LEAN_CHECK).read_text())
    assert rc == 0 and check["differ"] == [] and check["reason"] is None, check
    assert check["argv"][-4:] == ["--tie", str(wfile), str(a), str(b)] and check["lean"]["drawn"] in (None, 16)
    lean = json.loads((node / S.PEARLPLUS_LEAN).read_text())
    assert (lean["verdict"], lean["commitment"], lean["source"]) == ("accept", answer["commitment"], SOURCE.hex())
    assert lean["tie"] == answer["public"]["tie"] and lean["tie"]["record"] == w
    assert lean["work"] == answer["public"]["work"] and len(lean["verdicts"]) > 16

    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: TV._accepted())
    monkeypatch.setattr(S, "pearlplus_coins", lambda served, out, gate, key, main: {
        "session": TV.SID, "closed": served["coins"]["closed"], "passes": 4, "reason": None})
    (toy := tmp_path / "compose").mkdir()
    live = ([], {"gate": TV.GATE, "key": "ff" * 64, "main": toy / "main.txt"})
    pieces, audit, pin, lane = TV._pearlplus_piece(toy, monkeypatch, live, "no Lean")
    piece = pieces / "pouw"
    (piece / S.PEARLPLUS_ANSWER).write_text(json.dumps(answer))
    (audit / S.WINDOW_REGISTRATION).write_text(json.dumps({"format": REG.FORMAT_V2, "previous": w, "registered": []}))
    (audit / S.AUDIT_SUMMARY).write_text(json.dumps({"tie": {"record": w}}))

    def row() -> dict:
        return S.piece_row("work", pieces, sampled_proofs=audit, key_sha512=pin, lane=lane)

    before = row()
    assert before["verdict"] == "missing" and f"no Lean PouwCheck.PearlPlus verdict ({S.PEARLPLUS_LEAN}" in before["why"]
    shutil.copyfile(node / S.PEARLPLUS_LEAN, piece / S.PEARLPLUS_LEAN)
    got = row()
    assert lean["work"]["credited"] == 0 and lean["work"]["credited_tiles"] > 0
    assert got["verdict"] == "missing" and got["why"] == (f"{S.DEV_VERDICT}: the Lean verdict credits 0 MACs over "
                                                           f"{lean['work']['credited_tiles']} tiles: nothing"), got
