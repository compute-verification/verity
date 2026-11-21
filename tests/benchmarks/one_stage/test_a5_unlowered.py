"""a5 on a window whose members the door refuses to lower (`door_circuit` raising `one_circuit.Refused`): the window is
refused before any coin at the first such member, with the door's refusal, the member's template and its block."""
from __future__ import annotations

import argparse
import hashlib
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
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC

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


def _refuse(*_a, **_k):
    raise OC.Refused("E_LOWER", WHY)


def test_the_first_member_the_door_refuses_refuses_the_window_before_any_coin(tmp_path, monkeypatch, capsys):
    mine = _window(tmp_path / "w")
    monkeypatch.setattr(A5, "door_circuit", _refuse)
    a = argparse.Namespace(door_lower=Path("door"), delta=0.5, zk_server=None, rule="stratified-rate:1/2", session_timeout=1,
                           window_root=mine.run, coins=None)
    summary, ok = A5.audit(a, {descriptor_id(G8): (None, G8)}, tmp_path / "out", mine, tmp_path / "w", lambda w: {"ok": True},
                           0, {}, drawn=({"units": [0]}, {}, {}))
    assert ok is False and summary["checks_before_draw"] == {"door": False}, summary
    door = summary["checks"]["door"]["detail"]
    assert door["why"] == f"{A5.DOOR_REFUSES} 0 of {descriptor_id(G8)}: Refused: E_LOWER: {WHY}", door
    assert (door["member"], door["block"], door["units"]) == (0, 0, [0, 2]), door
    assert "refused_before_draw" in capsys.readouterr().out
    assert not (tmp_path / "out" / A5.SAMPLE_FILE).exists()
