"""The sm_120 FP8 arm (pearl_c_sm120/pearlc_arm.py) on the CPU stand-in for libcuda, driven as GPU 6's harness drives it, for
test_pearl_c_sm120.py.  argv: STUB_DIR ARM_CLASS MODE [CHECK_JSON]; FAKE_LOG is the stand-in's log, and this script appends
{"op": "mark"} lines to it around each launch and weight side, which go to stream STREAM.

  dry:     at a prefill-like and a decode shape, both flag sets on two operand sets of gen_f32 rows, the weight side, the
           gate and `known_bad`; one JSON line per gate.
  accept:  at the fixture's shape on its inputs and seeds, the fixture's bytes written where the device would have put them
           (the call's buffers and the gate's second launch), the gate; then one C~ word flipped, the gate again.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [sys.argv[1], str(HERE.parent / "pearl_c_sm120")]

import numpy as np  # noqa: E402
import pearlc_arm as A  # noqa: E402
from arm import Buffer, OperandSet, Shape  # noqa: E402

STREAM = 0x1234
LOG = os.environ["FAKE_LOG"]


def mark(what: str, **kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"op": "mark", "what": what, **kw}) + "\n")


class HarnessDev:
    """The harness's Device, as far as the arm touches it."""

    stream = STREAM

    def xor_byte(self, ptr: int, off: int, mask: int = 0x01, stream: int | None = None) -> None:
        mark("xor_byte", ptr=ptr, off=off, stream=stream)


def write(rd, ptr: int, data: bytes) -> None:
    rd.ck("cuMemcpyHtoD_v2", A.R.u64(ptr), A.R.C.c_char_p(data), len(data))


def operands(rd, shape: Shape, seeds: list[int]) -> list[OperandSet]:
    out = []
    for i, seed in enumerate(seeds):
        a = A.W.gen_f32(np.arange(shape.m), shape.k, seed).astype("<u4").tobytes()
        b = A.W.gen_f32(np.arange(shape.n), shape.k, seed + 1).astype("<u4").tobytes()
        out.append(OperandSet(i, i, seed, Buffer(rd.put(a), len(a)), Buffer(rd.put(b), len(b))))
    return out


def dry(arm) -> None:
    hdev = HarnessDev()
    for shape in (Shape("pre", 256, 1024, 128), Shape("dec", 32, 1024, 128)):
        ops = operands(arm.rd, shape, [11, 22])
        calls = []
        for role, flags in arm.flag_sets.items():
            mine = [arm.prepare(hdev, shape, flags, o) for o in ops]
            calls += mine
            for i, c in enumerate(mine):
                mark("launch", shape=shape.label, role=role, set=i, keys=c.keys)
                c.launch(STREAM)
                mark("end")
            mark("weight", shape=shape.label, role=role)
            mine[1].weight_side(STREAM)
            mark("end")
            g = arm.gate(hdev, mine[0], shape, flags)
            print(json.dumps({"shape": shape.label, "role": role, "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
        kb = arm.known_bad(hdev, shape, ops[0])
        mark("launch", shape=shape.label, role="known_bad", set=0, keys=kb.keys)
        kb.launch(STREAM)
        mark("end")
        kb.free()
        for c in calls:
            c.free()
        for o in ops:
            arm.rd.free(o.a.ptr)
            arm.rd.free(o.b.ptr)
    print(json.dumps({"live": sorted(arm.rd.live), "work": {s: vars(arm.work(Shape(s, *d))) for s, d in
                                                              (("pre", (256, 1024, 128)), ("dec", (32, 1024, 128)))}}))


def accept(arm, check: Path) -> None:
    man = json.loads(check.read_text())
    m, n, k = man["shape"]
    base = check.parent
    variant = man["variants"][arm.variant]
    load = lambda name: ((base / arm.variant / variant["files"][name]["file"]) if name in variant["files"]  # noqa: E731
                         else (base / man["common"][name]["file"])).read_bytes()
    arm.seeds = lambda shape, seed: {kk: bytes.fromhex(v) for kk, v in man["seeds"].items()}
    shape = Shape("fixture", m, k, n)
    rd = arm.rd
    xa, xb = load("xa"), load("xb")
    op = OperandSet(0, 0, 0, Buffer(rd.put(xa), len(xa)), Buffer(rd.put(xb), len(xb)))
    call = arm.prepare(HarnessDev(), shape, arm.flag_sets["hash"], op)
    call.launch(STREAM)
    s, w = call.scratch.b, call.own.b
    for name in ("at", "alpha_a", "c", "u", "y", "leaves", "tickets", "rows"):
        write(rd, s[name], load(name))
    for name in ("bt", "alpha_b"):
        write(rd, w[name], load(name))
    for name in A.REF:
        write(rd, s["ref_" + name], load("c" if name == "c2" else name))
    g = arm.gate(None, call, shape, arm.flag_sets["hash"])
    print(json.dumps({"flipped": None, "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    r0, c0 = A.W.samples(A.W.Shape("fixture", m, n, k))[-1]
    word = (r0 + 5) * n + c0 + 7
    write(rd, s["c"] + 4 * word, (int.from_bytes(load("c")[4 * word:4 * word + 4], "little") ^ 1).to_bytes(4, "little"))
    g = arm.gate(None, call, shape, arm.flag_sets["hash"])
    print(json.dumps({"flipped": [r0 + 5, c0 + 7], "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    call.free()


if __name__ == "__main__":
    arm = getattr(A, sys.argv[2])()
    arm.build({})
    dry(arm) if sys.argv[3] == "dry" else accept(arm, Path(sys.argv[4]))
