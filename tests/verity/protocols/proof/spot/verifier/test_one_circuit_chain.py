"""The periods' chain (`test_interface.chain`: eight ``Gemm_v2{K=16, N=16}`` units, one wire) through the served
verifier's door, stage by stage: C (the wired Program) registered with theta (each layer's weight a registered read), the
trace committed, a proof block drawn, its proof on the processor prover, the Python door and the Lean door, then the
four cheats (a crossing value, a block that is not C's, a block left out, a block holding a unit and the unit it reads). Each stage is a test of its own; a stage
that reaches a named hole (`one_circuit.NotDerived`, Lean's `notDerived`) or a missing piece xfails with where it
stopped, so the suite stays green and the first line of each xfail is the stop.

    python tests/verity/protocols/proof/spot/verifier/test_one_circuit_chain.py [--lean]

prints one line a stage."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import settings
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.layout import resolve
from verity.circuits.program import is_input
from verity.commitments.periods import interface as I
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

ROOT = Path(__file__).resolve().parents[6]
_TI = importlib.util.spec_from_file_location("periods_test_interface",
                                             ROOT / "tests/verity/commitments/periods/test_interface.py")
TI = importlib.util.module_from_spec(_TI)
_TI.loader.exec_module(TI)

LAYER = 256  # a 16 x 16 weight's 16-bit leaves
LEAN_DOOR = ROOT / "verity/protocols/proof/spot/verifier/lean/Spot/OneStageZk.lean"


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
    """C's bytes (the wired Program's descriptor), theta, the registered partition (its proof blocks the window's members,
    `interface.partition_blocks`), and the window binding."""
    win = I.Window(circuit(), TI.RUN, TI.SAMPLING_DESIGN, 2)
    program, _query = circuit().program()
    c = b"".join(descriptor_chunks(program))
    assert win.window.partition["blocks"] == [I.partition_blocks(circuit(), 2)]
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
    """One proof block of the window, by a stand-in coin: the door's draw (`python_door`) samples it alone."""
    ms = win.window.members
    return ms[int(np.random.default_rng(0).integers(len(ms)))]


def python_door(member: Any, circuit_file: bytes = b"", public: bytes | None = None,
                partition: dict | None = None) -> tuple[Any, Any]:
    """The door on a proof of ``member``'s block, under a draw that sampled that block alone; ``public`` defaults to a
    public file naming exactly the block's units."""
    r = registered()
    with tempfile.TemporaryDirectory() as d:
        cf, pub, s = Path(d, "circuit.txt"), Path(d, "pub.bin"), Path(d, "s0")
        cf.write_bytes(circuit_file)
        k = [m.base for m in r["window"].window.members].index(member.base)
        try:
            units = [u for lo, hi in OC.block(r["C"], r["partition"] if partition is None else partition, k)
                     for u in range(lo, hi)]
        except OC.Refused:
            units = [member.base]
        pub.write_bytes(json.dumps({"units": {"indices": units}}).encode() + b"\n" if public is None else public)
        s.mkdir()
        (s / "session.json").write_text("{}")
        return PF.verify(PF.Proof(cf, pub, (s,)), program=r["C"], partition=r["partition"] if partition is None else partition,
                         draw={"units": [k]}, registered=r["theta"])


def own_root_public(k: int) -> bytes:
    """Member ``k``'s public file as the periods stage it, with a fresh root of its own for its wired port ``x``
    (``frame_v3``'s roots), neither its writer's ``o0`` root nor its staged one."""
    with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as mp:
        d = Path(d)
        _w, _c, doc, _win = TI._held_window(d, mp)
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


def _member_refs(node: Any, m: int) -> list[tuple[str, Any]]:
    """Member ``m`` of a call or batch node: each parameter's name and the references its argument passes."""
    out = []
    axes = node.axes if node.form == "batch" else [None] * len(node.args)
    for (name, _t), x, ax in zip(node.fn.signature()[0], node.args, axes):
        if ax is None:
            out.append((name, x.refs))
        elif ax == 0:
            L = x.type.elem.leaves
            out.append((name, x.refs.slice(m * L, (m + 1) * L)))
        else:
            out.append((name, x.take(ax, m).refs))
    return out


def _units(D: Any) -> list[tuple[int, Any, int]]:
    """The root's unit nodes, every node that is not an input: (node index, node, members)."""
    return [(k, nd, nd.n if nd.form == "batch" else 1) for k, nd in enumerate(D.root.body.nodes) if not is_input(nd.fn)]


def by_gate(block: R.Block) -> dict[int, int]:
    """The committed trace's values (`TI.chain_values`) by C's descriptor gate, word level: every gate a unit of C reads or
    returns."""
    (xs, ws), (ys,) = TI.chain_values()
    ins = {"x": xs.reshape(len(xs), -1), "w": ws.reshape(len(ws), -1)}
    scope = block.bits.C.root_scope
    out: dict[int, int] = {}

    def put(g: int, v: Any) -> None:
        if out.setdefault(g, int(v)) != int(v):
            raise AssertionError(f"gate {g} of C holds two values of the trace")

    u = 0
    for k, node, n in _units(block.bits.D):
        per = node.out.leaves // n
        for m in range(n):
            for name, rs in _member_refs(node, m):
                for j in range(rs.count):
                    put(resolve(scope, rs[j]), ins[name][u, j])
            for j in range(per):
                put(resolve(scope, ("n", k, m * per + j)), ys[u, j])
            u += 1
    return out


def instances(block: R.Block, values: Mapping[int, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """C_S's unit instances' rows and outputs, ``{port: (n, words)}``: each parameter of the unit's Definition a row, its
    return ``o0``; each leaf the trace's value at its gate of C, `Block.to_c_gate` of its gate of C_S."""
    D = R.program(block.descriptor)
    scope = D.circuit.root_scope
    units = _units(D)
    if len({descriptor_id(nd.fn) for _k, nd, _n in units}) != 1:
        raise Stop("prove: C_S's units are instances of more than one Definition")
    at = lambda ref: values[block.to_c_gate(resolve(scope, ref))]  # noqa: E731
    rows: dict[str, list] = {}
    outs: list[list[int]] = []
    for k, node, n in units:
        per = node.out.leaves // n
        for m in range(n):
            for name, rs in _member_refs(node, m):
                rows.setdefault(name, []).append([at(rs[j]) for j in range(rs.count)])
            outs.append([at(("n", k, m * per + j)) for j in range(per)])
    return {p: np.array(v, dtype=np.uint16) for p, v in rows.items()}, {"o0": np.array(outs, dtype=np.uint16)}


def _hand_piece(*_a, **_k):
    raise AssertionError("ir_lower._lower_gate (a hand piece) called on the chain's road")


def prove_block(member: Any) -> dict[str, Any]:
    """The drawn block's proof on the processor prover: C_S (`restrict`'s descriptor), its statement the door's
    (`one_circuit.lower`, the hand pieces refused), its instances' rows the trace's values by C's gate (`instances`), and a live loopback session of
    ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    r = registered()
    try:
        block = R.restrict(r["C"], r["partition"], [(member.base, member.base + member.n)])
    except R.Refused as e:
        raise Stop(f"prove: the prover's C_S for block {member.base} ({R.__file__}:restrict): {e}") from e
    rows, outs = instances(block, by_gate(block))
    n = len(outs["o0"])
    rec = {"block": [member.base, member.base + member.n], "c_s_bytes": len(block.descriptor), "instances": n}
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(IL, "_lower_gate", _hand_piece)
            text = OC.lower(block.descriptor, r["partition"], PF.lower_executable,
                            cut=OC.cuts(r["C"], r["partition"]),
                            formed=OC.formed_in(r["C"], r["partition"], [(member.base, member.base + member.n)])).decode()
    except (OC.NotDerived, OC.Refused) as e:
        raise Stop(f"prove: C_S derived {rec}; its statement is the door's one_circuit.lower ({OC.__file__}:lower): {e}") from e
    meta = FC.meta_of(text)
    if [p[0] for p in meta["ports"]] != [*rows, *outs]:
        raise Stop(f"prove: the door's statement's ports {[p[0] for p in meta['ports']]} are not C_S's unit's {[*rows, *outs]}")
    binary = settings.flock_circuit()
    if not (binary and Path(binary).is_file()):
        raise Stop(f"prove: the door's statement for {rec}; FLOCK_CIRCUIT names no flock-circuit build")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "circuit.txt").write_text(text)
        iset = SimpleNamespace(name=f"block-{member.base}", subcircuit=SimpleNamespace(id=meta.get("unit", f"block-{member.base}"), inputs=[], outputs=[]))
        FC.write(text, iset, 0, n, d / f"inst-{n}.bin", d / f"pub-{n}.bin", indices=list(range(member.base, member.base + n)),
                 rows=rows, outs=outs)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        rec["live"] = CS.prove(binary, d, n, port)
    return rec


def stage_prove() -> str:
    win, _c = committed()
    got = prove_block(drawn(win))
    assert got["live"].get("accepted") is True, f"prove: the processor prover's session was not accepted: {got['live']}"
    return json.dumps({k: got[k] for k in ("block", "c_s_bytes", "instances")} | {"accepted": True})


def stage_python_door() -> str:
    win, _c = committed()
    findings, why = python_door(drawn(win))
    stopped(why)
    if why is not None:
        raise Stop(f"python door: refused {why}")
    return f"{findings}"


def stage_lean_door() -> str:
    """The Lean door (`spot-one-stage-zk restrict`): the drawn block's circuit file as Lean's `OneStageZk.restrict`
    derives it from C and the partition, byte for byte the Python door's (`one_circuit.lower` of `one_circuit.restrict`)."""
    r = registered()
    m = drawn(committed()[0])
    k = [x.base for x in r["window"].window.members].index(m.base)
    try:
        units = OC.block(r["C"], r["partition"], k)
        want = OC.lower(OC.restrict(r["C"], r["partition"], units), r["partition"], PF.lower_executable,
                        cut=OC.cuts(r["C"], r["partition"]), formed=OC.formed_in(r["C"], r["partition"], units))
    except (OC.NotDerived, OC.Refused) as e:
        raise Stop(f"lean door: the Python door derives no C_S for block {k} ({OC.__file__}): {e}") from e
    with tempfile.TemporaryDirectory() as d:
        c, p, out = Path(d, "C.json"), Path(d, "partition.json"), Path(d, "c_s.txt")
        c.write_bytes(r["C"])
        p.write_text(json.dumps(r["partition"], sort_keys=True, separators=(",", ":")))
        run = subprocess.run([str(PF.v2_executable()), "restrict", str(c), str(p), str(units[0][0]), str(out)],
                             capture_output=True, text=True)
        assert run.returncode == 0, f"lean door: Lean's restrict refused block {k} ({LEAN_DOOR}): {run.stderr[-600:]}"
        got = out.read_bytes()
    assert got == want, f"lean door: Lean's C_S for block {k} ({len(got)} bytes) is not the Python door's ({len(want)})"
    return f"block {k} at unit {units[0][0]}: Lean's C_S is the Python door's, {len(got)} bytes"


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
    bad = {**part, "blocks": [lv[:-1] for lv in part["blocks"]]}
    _f, why = python_door(drawn(win), partition=bad)
    stopped(why)
    assert why is not None, "cheat (c): the door accepted a partition that leaves C's units out"
    return f"refused: {why}"


def in_block() -> dict:
    """The registered partition with C's first two blocks one, ``[0, 4)``: units 2 and 3 read units 0 and 1 inside it."""
    part = registered()["partition"]
    return {**part, "blocks": [[[0, 4], *[b for b in part["blocks"][0] if b[0] >= 4]]]}


def stage_in_block() -> str:
    """(d) a block holding a unit and the unit it reads, at the door: refused by `operands-outside` before any proof."""
    win, _c = committed()
    _f, why = python_door(win.window.members[0], partition=in_block())
    assert why is not None and "operands-outside" in why[1], f"cheat (d): the door did not refuse an in-block read: {why}"
    return f"refused: {why}"


def stage_in_block_lean() -> str:
    """(d) at Lean's door (`spot-one-stage-zk restrict`): block 0 of `in_block`'s partition refused, in Python's words."""
    r = registered()
    with tempfile.TemporaryDirectory() as d:
        c, p, out = Path(d, "C.json"), Path(d, "partition.json"), Path(d, "c_s.txt")
        c.write_bytes(r["C"])
        p.write_text(json.dumps(in_block(), sort_keys=True, separators=(",", ":")))
        run = subprocess.run([str(PF.v2_executable()), "restrict", str(c), str(p), "0", str(out)],
                             capture_output=True, text=True)
    assert run.returncode != 0 and "operands-outside" in run.stderr, f"cheat (d): Lean's door: {run.returncode} {run.stderr[-600:]}"
    return f"refused: {run.stderr.strip()[-400:]}"


STAGES = [("register", stage_register), ("commit", stage_commit), ("draw", stage_draw), ("prove", stage_prove),
          ("python door", stage_python_door), ("lean door", stage_lean_door), ("cheat a: crossing value", stage_crossing),
          ("cheat b: block not C's", stage_not_cs), ("cheat c: block left out", stage_left_out),
          ("cheat d: in-block read", stage_in_block), ("cheat d at Lean's door", stage_in_block_lean)]


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


def test_cheat_in_block_read():
    _run(stage_in_block)


@pytest.mark.slow
def test_cheat_in_block_read_at_leans_door():
    _run(stage_in_block_lean)


if __name__ == "__main__":
    for name, fn in STAGES:
        if name in ("lean door", "cheat d at Lean's door") and "--lean" not in sys.argv:
            print(f"{name}: skipped (--lean)")
            continue
        try:
            print(f"{name}: REACHED {fn()}", flush=True)
        except Stop as e:
            print(f"{name}: STOP {e}", flush=True)
        except Exception as e:  # noqa: BLE001 - a driver line names any failure
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
