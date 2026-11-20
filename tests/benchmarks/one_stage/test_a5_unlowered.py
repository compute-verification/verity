"""a5 on a window whose members the door refuses to lower (`door_circuit` raising `one_circuit.Refused`): each refusal is
recorded by name, and each member's public body, the commit's bytes, is staged all the same, so the registration's leaf
layer covers every member and R4 passes; a draw of such a member is a failed proof by name (``failed_drawn``)."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.commitments import randomness as RND
from verity.commitments.periods import interface as I, rows as CS
from verity.protocols.proofs.spot.prover import docket as DK
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))
import a5 as A5

pytestmark = pytest.mark.slow

G8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
WHY = f"unit/{descriptor_id(G8)}: no row has a compression before its last block"


def _window(d: Path):
    """Four units of ``Gemm_v2{K=16, N=8}``, two a member, written to ``d``: -> the verifier's window."""
    rng = np.random.default_rng(0)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    x, w = bf16(rng.standard_normal((4, 16))), bf16(rng.standard_normal((4, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(G8, x, w, kernel="reference"), dtype=np.uint16).reshape(4, 8)
    win = I.Window(I.Circuit("unlowered", ((G8, 4),)), "ab" * 64, RND.parse_design("stratified:2"), 2)
    win.record(G8, [x, w], [y])
    c = win.close(salts_of={p: hashlib.shake_256(p.encode()).digest(k * CS.SALT_BYTES)
                            for p, k in CS.full_ports(win.window).items()})
    win.write(c, d, {"label": "test"}, registrar=DK)
    return win.window


class Stop(Exception):
    """Past the checks before the draw."""


def _refuse(*_a, **_k):
    raise OC.Refused("E_LOWER", WHY)


def _audit(tmp: Path, monkeypatch, units: list[int], stop: str) -> tuple[dict, bool, dict]:
    mine = _window(tmp / "w")
    seen = {}
    check = A5.V.check

    def recorded(*a, **k):
        seen["registration"] = v = check(*a, **k)
        return v

    monkeypatch.setattr(A5, "door_circuit", _refuse)
    monkeypatch.setattr(A5.V, "check", recorded)

    def halt(*_a, **_k):
        seen["summary"] = dict(inspect.currentframe().f_back.f_locals["summary"])
        raise Stop

    # "checks": past the registration, before the reads; "draw": past `failed_drawn`, before any proof server
    monkeypatch.setattr(*{"checks": (A5.CS, "own_reads"), "draw": (A5, "proof_server")}[stop], halt)
    a = argparse.Namespace(door_lower=Path("door"), delta=0.5, zk_server=None, rule="stratified-rate:1/2", session_timeout=1,
                           window_root=mine.run, coins=None)
    try:
        summary, ok = A5.audit(a, {descriptor_id(G8): (None, G8)}, tmp / "out", mine, tmp / "w", lambda w: {"ok": True}, 0, {},
                               drawn=({"units": units}, {}, {}))
    except Stop:
        summary, ok = {}, None
    return summary, ok, seen


def test_a_member_the_door_refuses_is_registered_and_r4_covers_it(tmp_path, monkeypatch, capsys):
    summary, ok, seen = _audit(tmp_path, monkeypatch, [], stop="checks")
    said = capsys.readouterr().out
    assert ok is None, summary
    reg = seen["registration"].to_json()
    assert reg["ok"] and "R4-leaf-layer" not in reg["codes"], reg
    assert said.count("UNLOWERED ") == 2 and said.count(WHY) == 2, said


def test_a_draw_of_a_member_the_door_refused_is_a_failed_proof_by_name(tmp_path, monkeypatch):
    _summary, _ok, seen = _audit(tmp_path, monkeypatch, [0], stop="draw")
    summary = seen["summary"]
    assert summary["failed_drawn"] == {descriptor_id(G8): 1}, summary
    assert set(summary["unlowered"]) == {"0", "1"} and WHY in summary["unlowered"]["0"]["why"], summary
