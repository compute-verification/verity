"""R1's arm (pearl_c_sm120/r1_arm.py) on the CPU stand-in for libcuda, driven as the harness drives it, for
test_pearl_c_sm120.py.  argv: ARM_DIR OUT_DIR; FAKE_LOG is the stand-in's log, and this script appends {"op": "mark"} lines to
it around each launch, phase split and free, on a stream it creates through the stand-in (the harness's non-blocking stream).

At a fused (prefill-like) and an unfused (decode) shape, both flag sets on two operand sets of E4M3 bytes: each call launched,
then run again as its three phases, then the gate; then `known_bad`, launched and freed; then every call freed.  One JSON line
per gate, and a last line with the live allocations and the stream.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [sys.argv[1], str(HERE.parent / "pearl_c_sm120")]

import numpy as np  # noqa: E402
import r1_arm as A  # noqa: E402
from arm import Buffer, OperandSet, Shape  # noqa: E402

STREAM = 0
LOG = os.environ["FAKE_LOG"]
SHAPES = (Shape("pre", 256, 256, 256), Shape("dec", 32, 256, 256))


def mark(what: str, **kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"op": "mark", "what": what, **kw}) + "\n")


class HarnessDev:
    """The harness's Device, as far as the arm touches it."""

    def __init__(self):
        self.stream = STREAM

    def xor_byte(self, ptr: int, off: int, mask: int = 0x01, stream: int | None = None) -> None:
        mark("xor_byte", ptr=ptr, off=off, stream=stream)


def e4m3(rng: np.random.Generator, n: int) -> bytes:
    """n finite E4M3 bytes (no 0x7f or 0xff, the NaNs)."""
    return (rng.integers(0, 0x7f, n, dtype=np.uint8) | (rng.integers(0, 2, n, dtype=np.uint8) << 7)).tobytes()


def operands(rd, shape: Shape, seeds: list[int]) -> list[OperandSet]:
    out = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        a, b = e4m3(rng, shape.m * shape.k), e4m3(rng, shape.n * shape.k)
        out.append(OperandSet(i, i, seed, Buffer(rd.put(a), len(a)), Buffer(rd.put(b), len(b))))
    return out


def dry(arm) -> None:
    hdev = HarnessDev()
    for shape in SHAPES:
        ops = operands(arm.rd, shape, [11, 22])
        calls = []
        for role, flags in arm.flag_sets.items():
            mine = [arm.prepare(hdev, shape, flags, o) for o in ops]
            calls += mine
            for i, c in enumerate(mine):
                mark("launch", shape=shape.label, role=role, set=i)
                c.launch(STREAM)
                mark("end")
                mark("phases", shape=shape.label, role=role, set=i)
                for name in ("before", "gemm", "after"):
                    mark("phase", name=name, stream=STREAM)
                    c.phases[name](STREAM)
                mark("end")
            g = arm.gate(hdev, mine[0], shape, flags)
            print(json.dumps({"shape": shape.label, "role": role, "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
        kb = arm.known_bad(hdev, shape, ops[0])
        mark("launch", shape=shape.label, role="known_bad", set=0)
        kb.launch(STREAM)
        mark("end")
        kb.free()
        for c in calls:
            c.free()
        for o in ops:
            arm.rd.free(o.a.ptr)
            arm.rd.free(o.b.ptr)
    print(json.dumps({"stream": STREAM, "live": sorted(arm.rd.live),
                      "work": {s.label: vars(arm.work(s)) for s in SHAPES}, "domain": {s.label: arm.domain(s) for s in SHAPES}}))


if __name__ == "__main__":
    arm = A.R1ChainSm120()
    arm.build({"out_dir": sys.argv[2]})
    s = A.R.C.c_void_p()
    arm.rd.ck("cuStreamCreate", A.R.C.byref(s), 1)
    STREAM = s.value
    dry(arm)
