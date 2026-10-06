"""The sm_120 FP8 arm (pearl_c_sm120/pearlc_arm.py) on the CPU stand-in for libcuda, driven as GPU 6's harness drives it, for
test_pearl_c_sm120.py.  argv: STUB_DIR ARM_CLASS MODE OUT_DIR [CHECK_JSON]; FAKE_LOG is the stand-in's log, and this script
appends {"op": "mark"} lines to it around each launch and weight side, which go to a stream it creates through the stand-in
(the harness's non-blocking stream).  OUT_DIR is the run's
directory the arm writes its transcripts under (`build`'s out_dir); PEARLC_ROWS holds their rows.

  dry:     at a prefill-like and a decode shape, both flag sets on two operand sets of gen_f32 rows (each call launched, then
           run again as its three phases), the weight side, the gate and `known_bad`; one JSON line per gate.  The hash calls
           of set 0 are then taken as gated, so freeing them writes their transcripts (of the stand-in's bytes).
  accept:  at the fixture's shape on its inputs and epoch (its test challenger, matmul index and weight, B's root), the
           fixture's bytes written where the device would have put them (the call's buffers and the gate's second launch), the
           gate; then one C~ word flipped (a message digest's where the call stores no C~, the fused call), the gate again;
           then the word
           restored, the gate, and the call freed, which writes its transcript and its negative control.  Each later launch
           of the whole call writes the fixture's bytes again, as the device would; the negative control's launches don't.
  stale:   accept, but no launch writes the fixture's bytes again: the transcript's call, like the negative control's,
           leaves the poisoned buffers as they are, as a kernel that skipped its writes would.
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

STREAM = 0
LOG = os.environ["FAKE_LOG"]


def mark(what: str, **kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"op": "mark", "what": what, **kw}) + "\n")


class HarnessDev:
    """The harness's Device, as far as the arm touches it."""

    def __init__(self):
        self.stream = STREAM

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
            if role == "hash":
                mine[0].gated = True
        kb = arm.known_bad(hdev, shape, ops[0])
        kb.gated = True
        mark("launch", shape=shape.label, role="known_bad", set=0, keys=kb.keys)
        kb.launch(STREAM)
        mark("end")
        kb.free()
        mark("transcripts", shape=shape.label)
        for c in calls:
            c.free()
        mark("end")
        for o in ops:
            arm.rd.free(o.a.ptr)
            arm.rd.free(o.b.ptr)
    print(json.dumps({"stream": STREAM, "live": sorted(arm.rd.live), "transcribed": arm.transcribed,
                      "work": {s: vars(arm.work(Shape(s, *d))) for s, d in (("pre", (256, 1024, 128)), ("dec", (32, 1024, 128)))}}))


def accept(arm, check: Path, rewrite: bool = True) -> None:
    man = json.loads(check.read_text())
    m, n, k = man["shape"]
    base = check.parent
    variant = man["variants"][arm.variant]
    load = lambda name: ((base / arm.variant / variant["files"][name]["file"]) if name in variant["files"]  # noqa: E731
                         else (base / man["common"][name]["file"])).read_bytes()
    arm.index, arm.challenger = A.F.INDEX, lambda shape, ops: A.F.test_challenger()
    A._Call.root_b = lambda self: bytes.fromhex(man["seeds"]["root_b"])
    shape = Shape("fixture", m, k, n)
    rd = arm.rd
    xa, xb = load("xa"), load("xb")
    op = OperandSet(0, A.F.WEIGHT, 0, Buffer(rd.put(xa), len(xa)), Buffer(rd.put(xb), len(xb)))
    call = arm.prepare(HarnessDev(), shape, arm.flag_sets["hash"], op)
    call.launch(STREAM)
    s, w = call.scratch.b, call.own.b

    def written() -> None:
        for name in ("at", "alpha_a", "c", "u", "y", "digests", "leaves", "tree_a", "tree_b", "tree_t", "seed_a", "line_a", "bt",
                     "alpha_b"):
            write(rd, w[name], load(name))
    written()
    if rewrite:
        launch = call.launch
        call.launch = lambda stream: (launch(stream), written())
    for name in A.REF:
        write(rd, s["ref_" + name], load("c" if name == "c2" else name))
    flags = arm.flag_sets["hash"]
    g = arm.gate(None, call, shape, flags)
    print(json.dumps({"flipped": None, "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    r0, c0 = A.W.samples(A.W.Shape("fixture", m, n, k))[-1]
    what = "c" if "c" in call.outputs else "digests"      # the fused call stores no C~: its row's message digest instead
    off = 4 * ((r0 + 5) * n + c0 + 7) if what == "c" else 32 * ((r0 + 5) * (n // 32) + (c0 + 7) // 32)
    word = load(what)[off:off + 4]
    write(rd, w[what] + off, bytes([word[0] ^ 1]))
    g = arm.gate(None, call, shape, flags)
    print(json.dumps({"flipped": [what, r0 + 5, c0 + 7], "ok": g.ok, "checked": g.checked, "detail": g.detail}), flush=True)
    write(rd, w[what] + off, word)
    g = arm.gate(None, call, shape, flags)
    call.free()
    print(json.dumps({"restored": True, "ok": g.ok, "transcribed": arm.transcribed}), flush=True)


if __name__ == "__main__":
    arm = getattr(A, sys.argv[2])()
    arm.build({"out_dir": sys.argv[4]})
    s = A.R.C.c_void_p()
    arm.rd.ck("cuStreamCreate", A.R.C.byref(s), 1)
    STREAM = s.value
    dry(arm) if sys.argv[3] == "dry" else accept(arm, Path(sys.argv[5]), rewrite=sys.argv[3] == "accept")
