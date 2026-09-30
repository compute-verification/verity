"""The sm_120 FP8 arm (pearl_c_sm120/pearlc_arm.py) on the CPU stand-in for libcuda, driven as GPU 6's harness drives it, for
test_pearl_c_sm120.py.  argv: STUB_DIR ARM_CLASS MODE [CHECK_JSON]; FAKE_LOG is the stand-in's log, and this script appends
{"op": "mark"} lines to it around each launch and weight side, which go to stream STREAM (the stand-in's, created as the
harness creates its own: the stand-in checks every stream handle).

  dry:     at a prefill-like and a decode shape, both flag sets on two operand sets of gen_f32 rows (each call launched, then
           run again as its three phases), the weight side, the gate and `known_bad`; one JSON line per gate.
  accept:  at the fixture's shape on its inputs and seeds, the fixture's bytes written where the device would have put them
           (the call's buffers and the gate's second launch), the gate; then one C~ word flipped (fused: one message digest's
           byte), the gate again.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [sys.argv[1], str(HERE.parent / "pearl_c_sm120")]

import numpy as np  # noqa: E402
import pearlc_arm as A  # noqa: E402
from arm import Buffer, OperandSet, Shape  # noqa: E402

STREAM = 0
LOG = os.environ["FAKE_LOG"]


def harness_stream() -> int:
    lib, s = ctypes.CDLL(os.environ["PEARLC_LIBCUDA"]), ctypes.c_void_p()
    assert lib.cuStreamCreate(ctypes.byref(s), 1) == 0
    return s.value


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
                mark("phases", shape=shape.label, role=role, set=i)
                for name in ("before", "gemm", "after"):
                    mark("phase", name=name, stream=STREAM)
                    c.phases[name](STREAM)
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
    base, vb, fmt = check.parent, A.R.hashing_of(arm.variant)[0], arm.hashing
    variant = man["variants"][vb]
    load = lambda name: ((base / vb / variant["files"][name]["file"]) if name in variant["files"]  # noqa: E731
                         else (base / man["common"][name]["file"])).read_bytes()
    ctx = A.H.context(m, n, k)
    sd = {**{kk: bytes.fromhex(v) for kk, v in man["seeds"].items()}, "salt": ctx.salt, "root_b": ctx.root_b}
    arm.seeds = lambda shape, seed: sd
    arm.domains = lambda shape, sd: (A.H.a_domain(ctx), A.H.t_domain(ctx, A.tiles(shape)))
    shape = Shape("fixture", m, k, n)
    assert arm.keys(shape, sd) == man["keys"]
    rd = arm.rd
    xa, xb = load("xa"), load("xb")
    op = OperandSet(0, 0, 0, Buffer(rd.put(xa), len(xa)), Buffer(rd.put(xb), len(xb)))
    call = arm.prepare(HarnessDev(), shape, arm.flag_sets["hash"], op)
    call.launch(STREAM)
    s, w = call.scratch.b, call.own.b
    for name in ("at", "alpha_a", "c", "u", "y", "bt", "alpha_b"):
        write(rd, w[name], load(name))
    for out, bufs in A.HASHED[fmt].items():
        write(rd, w[out], load(bufs[0]))
    for name in arm.ref:
        if name not in A.LEVEL_KEYS[fmt]:
            write(rd, s["ref_" + name], load("c" if name == "c2" else name))
    g = arm.gate(None, call, shape, arm.flag_sets["hash"])
    print(json.dumps({"flipped": None, "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    r0, c0 = A.W.samples(A.W.Shape("fixture", m, n, k))[-1]
    name, off = (("digests", ((r0 + 5) * (n // 32) + (c0 + 7) // 32) * 32) if call.fused else
                 ("c", 4 * ((r0 + 5) * n + c0 + 7)))
    write(rd, w[name] + off, bytes([load(A.HASHED[fmt].get(name, (name,))[0])[off] ^ 1]))
    g = arm.gate(None, call, shape, arm.flag_sets["hash"])
    print(json.dumps({"flipped": [name, r0 + 5, c0 + 7], "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    call.free()


if __name__ == "__main__":
    arm = getattr(A, sys.argv[2])()
    arm.build({})
    STREAM = HarnessDev.stream = harness_stream()
    dry(arm) if sys.argv[3] == "dry" else accept(arm, Path(sys.argv[4]))
