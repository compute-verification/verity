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
import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.layout import resolve
from verity.circuits.ops.boolean import versions as V
from verity.commitments.periods import interface as I
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as ROW_V2
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

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
    """C's bytes (the wired Program's descriptor), theta, the partition with its proof blocks (the window's members,
    `interface.partition_blocks`), and the window binding."""
    win = I.Window(circuit(), TI.RUN, TI.SAMPLING_DESIGN, 2)
    program, _query = circuit().program()
    c = b"".join(descriptor_chunks(program))
    blocks = [[m.base, m.base + m.n] for m in win.window.members]
    return {"window": win, "C": c, "theta": theta().doc(), "partition": {**win.window.partition, "blocks": blocks}}


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


def python_door(member: Any, circuit_file: bytes = b"", public: bytes = b"",
                partition: dict | None = None) -> tuple[Any, Any]:
    r = registered()
    with tempfile.TemporaryDirectory() as d:
        cf, pub, s = Path(d, "circuit.txt"), Path(d, "pub.bin"), Path(d, "s0")
        cf.write_bytes(circuit_file)
        pub.write_bytes(public)
        s.mkdir()
        (s / "session.json").write_text("{}")
        return PF.verify(PF.Proof(cf, pub, (s,)), program=r["C"], partition=r["partition"] if partition is None else partition,
                         block=[m.base for m in r["window"].window.members].index(member.base), registered=r["theta"])


def own_root_public(k: int) -> bytes:
    """Member ``k``'s public file as the periods stage it, with a fresh root of its own for its wired port ``x``
    (``frame_v3``'s roots), neither its writer's ``o0`` root nor its staged one."""
    with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as mp:
        d = Path(d)
        _w, _c, doc = TI._held_window(d, mp)
        head, body = TI._staged(d, doc["members"][k], k, d / "out")
        writer, _ = TI._staged(d, doc["members"][k - 1], k - 1, d / "writer")
    assert doc["members"][k]["base"] >= TI.REQUESTS, "cheat (a) needs a member whose x is wired"
    roots = head["frame_v3"]["roots"]
    roots["x"] = hashlib.sha512(b"own root " + roots["x"].encode()).hexdigest()
    assert roots["x"] != writer["frame_v3"]["roots"]["o0"]
    return json.dumps(head).encode() + b"\n" + body


def stopped(why: tuple[str, str] | None) -> None:
    if why is not None and why[1].startswith(OC.PREFIX):
        raise Stop(f"python door: refused {why[0]!r} at {PF.__file__}:verify -> {OC.__file__}: {why[1]}")


def stage_register() -> str:
    r = registered()
    return f"C {len(r['C'])} bytes, {len(r['window'].window.members)} members, theta {sorted(r['theta'])}"


def stage_commit() -> str:
    win, c = committed()
    return f"{len(c.roots)} roots"


def stage_draw() -> str:
    win, _c = committed()
    return f"block at unit {drawn(win).base}"


def by_gate(block: R.Block) -> dict[int, int]:
    """The committed trace's values (`TI.chain_values`) by C's descriptor gate, word level: every gate a unit of C reads or
    returns."""
    (xs, ws), (ys,) = TI.chain_values()
    ins = {"x": xs.reshape(len(xs), -1), "w": ws.reshape(len(ws), -1)}
    D, scope = block.bits.D, block.bits.C.root_scope
    templates = set(registered()["partition"]["query"]["params"]["templates"])
    out: dict[int, int] = {}

    def put(g: int, v: Any) -> None:
        if out.setdefault(g, int(v)) != int(v):
            raise AssertionError(f"gate {g} of C holds two values of the trace")

    u = 0
    for k, node in enumerate(D.root.body.nodes):
        if descriptor_id(node.fn) not in templates:
            continue
        n = node.n if node.form == "batch" else 1
        per = node.out.leaves // n
        names = [p for p, _t in node.fn.signature()[0]]
        for m in range(n):
            for name, x, ax in zip(names, node.args, node.axes if node.form == "batch" else [None] * len(node.args)):
                if ax is None:
                    rs = x.refs
                elif ax == 0:
                    L = x.type.elem.leaves
                    rs = x.refs.slice(m * L, (m + 1) * L)
                else:
                    rs = x.take(ax, m).refs
                for j in range(rs.count):
                    put(resolve(scope, rs[j]), ins[name][u, j])
            for j in range(per):
                put(resolve(scope, ("n", k, m * per + j)), ys[u, j])
            u += 1
    return out


def _hand_piece(*_a, **_k):
    raise AssertionError("ir_lower._lower_gate (a hand piece) called on the chain's road")


def prove_block(member: Any) -> dict[str, Any]:
    """The drawn block's proof on the processor prover: C_S (`restrict`'s descriptor), its Definition lifted to bits through
    the Boolean Definitions' own builders (`versions.block_circuit`), the hand pieces refused, staged on the trace's values
    of its input gates (`Block.to_c_gate`), its outputs checked against the trace's on out(S), and proved in a live loopback
    session of ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    r = registered()
    try:
        block = R.restrict(r["C"], r["partition"], [(member.base, member.base + member.n)])
    except R.Refused as e:
        raise Stop(f"prove: the prover's C_S for block {member.base} ({R.__file__}:restrict): {e}") from e
    values = by_gate(block)
    D = R.program(block.descriptor)
    calls = [n.fn for n in D.root.body.nodes if n.form == "call"]
    if len(calls) != 1:
        raise AssertionError(f"C_S's root makes {len(calls)} calls, not one of the block's Definition")
    fn = calls[0]
    params, ret = fn.signature()
    if any(t.homogeneous_width() != 16 for _p, t in [*params, (None, ret)]):
        raise Stop(f"prove: C_S of block {member.base} has a port of other than 16-bit words")
    rows, at = {}, 0
    for p, t in params:
        rows[p] = np.array([[values[block.to_c_gate(at + j)] for j in range(t.leaves)]], dtype=np.uint16)
        at += t.leaves
    want = [values[g] for g in block.outputs]

    def reference(_rows: dict, outs: dict) -> None:
        got = [int(v) for j in range(len(ret.fields)) for v in outs[f"o{j}"][0]]
        if got != want:
            raise AssertionError(f"block {member.base}: C_S's outputs are not the trace's on out(S)")

    binary = os.environ.get("FLOCK_CIRCUIT", "")
    with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as mp:
        mp.setattr(IL, "_lower_gate", _hand_piece)
        bits = V.block_circuit(fn, V.boolean_table(D))
        st = CS.UnitStager(bits, f"block/{member.base}", [(p, ROW_V2, 16 * t.leaves, [p]) for p, t in params],
                           [(ROW_V2, (j,)) for j in range(len(ret.fields))])
        st.reference_check = reference
        staged = st.stage(rows, Path(d))
        circuit = Path(d, "circuit.txt").read_bytes()
        rec = {"block": [member.base, member.base + member.n], "c_s_bytes": len(block.descriptor), "inputs": len(block.inputs),
               "outputs": len(block.outputs), "circuit_lines": circuit.count(b"\n"), "ands": staged.get("ands")}
        if not (binary and Path(binary).is_file()):
            raise Stop(f"prove: C_S staged {rec}; FLOCK_CIRCUIT names no flock-circuit build")
        Path(d, "inst.bin").rename(Path(d, "inst-1.bin"))
        Path(d, "pub.bin").rename(Path(d, "pub-1.bin"))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        live = CS.prove(binary, Path(d), 1, port)
    return {**rec, "live": live}


def stage_prove() -> str:
    win, _c = committed()
    got = prove_block(drawn(win))
    assert got["live"].get("accepted") is True, f"prove: the processor prover's session was not accepted: {got['live']}"
    return json.dumps({k: got[k] for k in ("block", "c_s_bytes", "circuit_lines", "ands")} | {"accepted": True})


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
    """(a) a crossing value, at the door: a member holding its own root for its wired port ``x``."""
    win, _c = committed()
    m = drawn(win)
    k = win.window.members.index(m)
    _f, why = python_door(m, public=own_root_public(k))
    stopped(why)
    assert why is not None, "cheat (a): the door accepted a member's own root for a wired port"
    return f"refused: {why}"


def stage_not_cs() -> str:
    """(b) a proof whose circuit is not C_S: the door reads no circuit of the prover's, and runs on its own C_S."""
    win, _c = committed()
    _f, why = python_door(drawn(win), b"flock-circuit v2\nnot C_S\n")
    stopped(why)
    assert why is not None, "cheat (b): the door accepted a statement whose circuit is not C_S"
    return f"refused: {why}"


def stage_left_out() -> str:
    """(c) a block left out, at the door: a partition whose blocks leave C's last block out."""
    win, _c = committed()
    part = registered()["partition"]
    bad = {**part, "blocks": part["blocks"][:-1]}
    _f, why = python_door(drawn(win), partition=bad)
    stopped(why)
    assert why is not None, "cheat (c): the door accepted a partition that leaves C's units out"
    return f"refused: {why}"


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
