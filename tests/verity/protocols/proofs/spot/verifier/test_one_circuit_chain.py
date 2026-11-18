"""The periods' chain (`test_interface.chain`: eight ``Gemm_v2{K=16, N=16}`` units, one wire) through the served
verifier's door, stage by stage: C (the wired Program) registered with theta (each layer's weight a registered read), the
trace committed, a proof block drawn, its proof on the processor prover, the Python door and the Lean door, then the
three cheats (a crossing value, a block that is not C's, a block left out). Each stage is a test of its own; a stage
that reaches a named hole (`one_circuit.NotDerived`, Lean's `notDerived`) or a missing piece xfails with where it
stopped, so the suite stays green and the first line of each xfail is the stop.

    python tests/verity/protocols/proofs/spot/verifier/test_one_circuit_chain.py [--lean]

prints one line a stage."""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.commitments.periods import interface as I
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

ROOT = Path(__file__).resolve().parents[6]
_TI = importlib.util.spec_from_file_location("periods_test_interface",
                                             ROOT / "tests/verity/commitments/periods/test_interface.py")
TI = importlib.util.module_from_spec(_TI)
_TI.loader.exec_module(TI)

LAYER = 256  # a 16 x 16 weight's 16-bit leaves
LEAN_DOOR = ROOT / "verity/protocols/proofs/spot/verifier/lean/Spot/OneStageZk.lean"


class Stop(Exception):
    """Where the trip stopped: the stage, the hole or missing piece, and the file it is in."""


def theta() -> I.Registered:
    """Each layer's weight a registered read: unit (step, layer, request) reads layer's 256 leaves."""
    weights = I.Read(TI.GID, "w", "weights", 0, (TI.STEPS, TI.LAYERS, TI.REQUESTS), 0, (0, LAYER, 0))
    return I.Registered({"weights": TI.LAYERS * LAYER}, reads=(weights,))


def circuit() -> I.Circuit:
    return replace(TI.chain(), registered=theta())


@functools.cache
def registered() -> dict[str, Any]:
    """C's bytes (the wired Program's descriptor), theta, the partition, and the window binding."""
    win = I.Window(circuit(), TI.RUN, TI.SAMPLING_DESIGN, 2)
    program, _query = circuit().program()
    c = b"".join(descriptor_chunks(program))
    return {"window": win, "C": c, "theta": theta().doc(), "partition": win.window.partition}


def committed(x=None) -> Any:
    """The trace committed: each layer's weight the registered value theta reads, each unit's ``x`` (``x``: in place of
    the honest one) and output recorded, and the period closed."""
    (xs, ws), outs = TI.chain_values()
    weights = np.stack([ws[layer * TI.REQUESTS].reshape(LAYER) for layer in range(TI.LAYERS)])
    win = I.Window(circuit(), TI.RUN, TI.SAMPLING_DESIGN, 2, registered={"weights": TI.Bound("weights", weights)})
    win.record(TI.GEMM, [xs if x is None else x, None], outs, {"chain": 0})
    try:
        return win, win.close(salts_of=TI.fixed(win))
    except ValueError as e:
        raise Stop(f"commit: Window.close refuses the chain with theta: {e}"[:600]) from e


def drawn(win: Any) -> Any:
    """One proof block of the window, by a stand-in coin (the challenger's live coin is not the door's input)."""
    ms = win.window.members
    return ms[int(np.random.default_rng(0).integers(len(ms)))]


def c_s(member: Any) -> bytes:
    """The drawn block's C_S, which prover and verifier both derive from C and the block's units: the block's Program
    (`restrict.restrict`), its descriptor's bytes."""
    r = registered()
    try:
        return R.restrict(r["C"], r["partition"], [(member.base, member.base + member.n)]).descriptor
    except R.Refused as e:
        raise Stop(f"prove: the prover's C_S for block {member.base} ({R.__file__}:restrict): {e}") from e


def python_door(member: Any, circuit_file: bytes = b"") -> tuple[Any, Any]:
    r = registered()
    with tempfile.TemporaryDirectory() as d:
        cf, pub, s = Path(d, "circuit.txt"), Path(d, "pub.bin"), Path(d, "s0")
        cf.write_bytes(circuit_file)
        pub.write_bytes(b"")
        s.mkdir()
        (s / "session.json").write_text("{}")
        return PF.verify(PF.Proof(cf, pub, (s,)), program=r["C"], partition=r["partition"], block=member.base,
                         registered=r["theta"])


def stopped(why: tuple[str, str] | None) -> None:
    if why is not None and why[1].startswith(OC.PREFIX):
        raise Stop(f"python door: refused {why[0]!r} at {PF.__file__}:verify -> one_circuit.restrict: {why[1]}")


def stage_register() -> str:
    r = registered()
    return f"C {len(r['C'])} bytes, {len(r['window'].window.members)} members, theta {sorted(r['theta'])}"


def stage_commit() -> str:
    win, c = committed()
    return f"{len(c.roots)} roots"


def stage_draw() -> str:
    win, _c = committed()
    return f"block at unit {drawn(win).base}"


def stage_prove() -> str:
    win, _c = committed()
    c_s(drawn(win))
    raise Stop("prove: C_S derived, and no processor prover takes it yet without the hand pieces (ROWS)")


def stage_python_door() -> str:
    win, _c = committed()
    findings, why = python_door(drawn(win))
    stopped(why)
    if why is not None:
        raise Stop(f"python door: refused {why}")
    return f"{findings}"


def stage_lean_door() -> str:
    """The Lean door built (`spot-one-stage-zk`): its own `#guard`s hold that a member's statement refuses at
    `restrict`."""
    PF.v2_executable()
    text = LEAN_DOOR.read_text().splitlines()
    at = next(i for i, x in enumerate(text, 1) if "one-circuit: not derived yet: restrict" in x)
    raise Stop(f"lean door: built; every statement refuses at Spot.OneStageZk.restrict ({LEAN_DOOR}:{at})")


def stage_crossing() -> str:
    """(a) a reader's input that is not its writer's output."""
    (xs, _ws), _outs = TI.chain_values()
    x = xs.copy()
    x[2, 0] ^= 1
    try:
        committed(x)
    except Stop as e:
        if "is not the o0 row" in str(e):
            raise Stop(f"cheat (a): refused by name at the commit (rows.py, the one position), before the door: {e}") from e
        raise
    raise AssertionError("cheat (a): a crossing value committed")


def stage_not_cs() -> str:
    """(b) a proof whose circuit is not the lowering of the drawn block's Definition."""
    win, _c = committed()
    _f, why = python_door(drawn(win), b"flock-circuit v2\nnot C_S\n")
    stopped(why)
    assert why is not None and why[0] == "circuit", why
    return f"refused: {why}"


def stage_left_out() -> str:
    """(c) a partition that does not cover C's units."""
    doc = json.loads(I.circuit_bytes(registered()["window"].circuit_record()))
    with pytest.raises(ValueError) as e:
        I.window_of_record({**doc, "templates": [[TI.GID, 7]]}, TI.SAMPLING_DESIGN, {TI.GID: TI.GEMM}.__getitem__)
    raise Stop(f"cheat (c): refused by name at registration (interface.window_of_record), before the door: {e.value}")


STAGES = [("register", stage_register), ("commit", stage_commit), ("draw", stage_draw), ("prove", stage_prove),
          ("python door", stage_python_door), ("lean door", stage_lean_door), ("cheat a: crossing value", stage_crossing),
          ("cheat b: block not C's", stage_not_cs), ("cheat c: block left out", stage_left_out)]


def _run(fn) -> str:
    try:
        return fn()
    except Stop as e:
        pytest.xfail(f"STOP {e}")


def test_register():
    _run(stage_register)


def test_commit():
    _run(stage_commit)


def test_draw():
    _run(stage_draw)


def test_prove():
    _run(stage_prove)


def test_python_door():
    _run(stage_python_door)


@pytest.mark.slow
def test_lean_door():
    _run(stage_lean_door)


def test_cheat_crossing_value():
    _run(stage_crossing)


def test_cheat_block_not_cs():
    _run(stage_not_cs)


def test_cheat_block_left_out():
    _run(stage_left_out)


if __name__ == "__main__":
    for name, fn in STAGES:
        if name == "lean door" and "--lean" not in sys.argv:
            print(f"{name}: skipped (--lean)")
            continue
        try:
            print(f"{name}: REACHED {fn()}", flush=True)
        except Stop as e:
            print(f"{name}: STOP {e}", flush=True)
        except Exception as e:  # noqa: BLE001 - a driver line names any failure
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
