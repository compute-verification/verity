#!/usr/bin/env python3
"""The GPU cost on sm_120 of committing a served call's rows in the ZK statement's format, hm96-sha512 row leaves of a
frame-v3-sha512 tree (hm96_rows.cu), beside -h2's BLAKE3 frame-b3s rows on GPU 1's pipeline (pearl_c_sm120/run.py), at
Llama-3.1-8B's four linears with m = 32 (decode_served.py's shapes): a microbenchmark whose step figures are projections.

Gates before any timing, per shape, from outputs poisoned to 0xA5: hm96_rows.cu's x (sha512/row-seg/v1 and sha512/row/v2),
b ‖ c, the hm96 leaf and the frame-v3-sha512 root equal Python's (verity.ml.commitments: rowleaf, hm96.SHA512,
merkle) under salts read from the OS on the host, as C-Flock's are (zk_hooks::LeafSalts), and handed to the device: 192 bytes a
row, read and copied once per commitment, outside the timed graphs; each no-write control (the launch left out) leaves them
unequal.

Each variant's launches, CALLS times over in one stream, are captured as a CUDA graph and replayed REPS rounds in rotated
order; the figure is the median replay over CALLS, in µs a call.  Variants, per shape:
  a.h2s / a.h2w   -h2's A commitment as served (rows_form s: hash_rows_b3s_stats, A's leaves and forming's stats in one pass)
                  and in form w (hash_rows_b3s), each with tree_a
  k.stats_a       forming's stats_a, which form s folds into A's commitment and the hm96 variants run on their own
  a.hm96          the statement's A commitment: A's 32 FP32 rows (4k bytes) as hm96-sha512/row-seg/v1 leaves and their tree
  a.sha512        the same rows' unsalted sha512/row-seg/v1 digests alone (a ZK server salting them later)
  p.hm96          A's fp8-p rows (A' ‖ P_A, k + 128 bytes) the same way: the tile's input rows, which a prover may commit after
                  the pass instead
  t.h2            -h2's tile hashing (hash_msg, hash_leaf, the tile tree)
  t.hm96          the tile rows as hm96-sha512/row/v2 leaves (32 bytes each) and their tree, over -h2's tile leaves as
                  stand-in digests: the statement's tile digest (TurboSHAKE128, leaves.tile_digest) is not measured
  call.h2         the call as served (commit, forming, clean-up, hashing; rows_form s)
  call.nohash     the call without A's commitment and the tile hashing (form w, so stats_a stays)
  call.hm96       call.nohash with a.hm96 in A's commitment's place and the tile rows as hash_msg, hash_leaf, t.hm96
  call.both       call.h2 with a.hm96 and t.hm96 added (both formats)
A decode step's four linears LAYERS deep, each a graph replayed REPS rounds, in ms:
  step.nohash, step.h2, step.hm96   every call as call.nohash, call.h2, call.hm96
  step.both.side    every call as call.h2, each call's a.hm96 and t.hm96 on a side stream forked after it, joined at the step's
                    end (-h2's root seeding forming, the statement's roots off the critical path)
  step.h2.batch     step.h2, then batch.hm96
  step.tdig.batch   every call as call.nohash with -h2's tile digests (hash_msg, hash_leaf: the stand-in for the statement's),
                    then batch.hm96: the statement's format instead of -h2's, committed once after the step
  batch.hm96        the step's rows in one commitment a format: its FP32 A rows of each k as hm96-sha512 row-seg leaves and
                    the tile rows as row/v2 leaves, each in one tree (grid-wide launches)
  batch.sha512      the step's A rows' unsalted sha512/row-seg/v1 digests alone
  batch.p.hm96      the step's fp8-p rows (k + 128 bytes) of each k as hm96-sha512 row-seg leaves, each k in one tree: what a
                    prover commits after the pass, before the sample
Throughput, outside graphs: hash_rows_b3s against a.hm96's kernels over PREFILL rows of each k, in GB/s of row bytes.

  python3 rows_bench.py --cubin pearl_c_sm120.cubin --rows-cubin hm96_rows.cubin --out OUT.json
  (env: PEARLC_EXPECT_UUID; CALLS=32, REPS=30, PREFILL=8192)
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

from verity.ml.commitments import hm96
from verity.ml.commitments import merkle as MK
from verity.ml.commitments import rowleaf as RL
from verity.ml.commitments.indexed import RangeIndexedDomain

HERE = Path(__file__).resolve().parent
M_REAL, LAYERS = 32, 32
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}   # (n, k)
POISON, SLOT, SEG_TPB = 0xA5, 256, 256
INST, KEY = hm96.SHA512, hm96.SHA512.default_key
SEG, V2 = RL.SCHEMA_HM96_SHA512_ROW_SEG, RL.SCHEMA_HM96_SHA512_ROW_V2
HASH_A = ("hash_rows_a", "tree_a")
HASH_T = ("hash_msg", "hash_leaf", "tree_leaves_t", "tree_t")
CALL_PHASES = ("commit", "forming", "cleanup", "hashing")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def slots(heads: list[bytes]) -> bytes:
    out = b""
    for h in heads:
        if len(h) > SLOT - 4:
            raise ValueError(f"a message head of {len(h)} bytes does not fit a slot")
        out += (len(h).to_bytes(4, "little") + h).ljust(SLOT, b"\0")
    return out


def node_head(dom: MK.CommitmentDomain, level: int, index: int) -> bytes:
    """The bytes of a frame-v3-sha512 node's message before its left child: the node is SHA-512(head ‖ left ‖ u64be 64 ‖
    right) (merkle's `_message`)."""
    parts = (dom.domain_id, MK._uint(level), MK._uint(index))
    return (MK._FRAME + len(MK._NODE).to_bytes(4, "big") + MK._NODE + b"".join(len(p).to_bytes(8, "big") + p for p in parts)
            + (64).to_bytes(8, "big"))


def constants(h) -> dict:
    """hm96_rows.cu's constants on the device: M_key's byte tables and the midstates of hm96's salt and leaf prefixes."""
    tables = b"".join(v.to_bytes(64, "little") for row in INST.byte_tables(KEY) for v in row)
    pre = h.put(INST.salt_prefix + INST.leaf_prefix(KEY))
    consts = {"tables": h.put(tables), "hm_mid": h.alloc(128)}
    h.launch("sz_mid", 1, 32, p=(pre, consts["hm_mid"]), i=(2,))
    return consts


class Commit:
    """One call's rows (`rows` of `nbytes` each at device address `data`, stride `stride`) as hm96-sha512 leaves of a
    frame-v3-sha512 tree under schema `schema` (SEG: row-seg/v1 rows, any whole number of 128-byte chunks a segment; V2: rows of
    at most 111 bytes), with its constants on the device, and its rows' salts read from the OS (`fresh_salts` reads new ones)."""

    def __init__(self, dev, h, consts: dict, tag: str, data: int, rows: int, nbytes: int, stride: int, schema: str,
                 prefix: str = "served-zk/rows-bench"):
        self.dev, self.h, self.data, self.rows, self.nbytes, self.stride, self.schema = dev, h, data, rows, nbytes, stride, schema
        self.c = consts
        self.salts = os.urandom(192 * rows)
        self.d_salts = h.put(self.salts)
        n_bits = 8 * nbytes
        if schema == SEG:
            segs = RL.sha512_row_seg_segments(n_bits)
            if any(m % 1024 for m in segs):
                raise ValueError(f"hm96_rows.cu takes segments of whole 128-byte chunks, not {segs}")
            self.s, self.last = len(segs), segs[-1] // 8
            blocks = [RL.sha512_row_seg_segment_prefix(j, self.s, m) for j, m in enumerate(segs)] + [RL.sha512_row_seg_prefix(n_bits)]
        else:
            if nbytes > 111:
                raise ValueError("a v2 row of hm96_rows.cu is at most 111 bytes")
            self.s, self.last = 0, 0
            blocks = [RL.sha512_row_v2_prefix(n_bits)]
        mids = h.alloc(64 * len(blocks))
        h.launch("sz_mid", 1, 32 * ((len(blocks) + 31) // 32), p=(h.put(b"".join(blocks)), mids), i=(len(blocks),))
        self.seg_mids, self.q_mid = mids, mids + 64 * (len(blocks) - 1)
        self.dom = MK.CommitmentDomain(hashlib.sha256(f"{prefix}/{tag}".encode()).digest(), -1,
                                       RangeIndexedDomain(0, rows), hash="sha512")
        self.width = 1 << max(rows - 1, 0).bit_length()
        self.leaf_heads = h.put(slots([self.dom.leaf_prefix(r, r, schema, 64) for r in range(rows)]))
        heads, w, lvl = [], self.width, 0
        while w > 1:
            heads += [node_head(self.dom, lvl, j) for j in range(w // 2)]
            w, lvl = w // 2, lvl + 1
        self.node_heads = h.put(slots(heads) if heads else bytes(SLOT))
        self.digests = h.alloc(64 * rows * max(self.s, 1))
        self.x, self.bc = h.alloc(64 * rows), h.alloc(128 * rows)
        self.tree_bytes = 64 * (2 * self.width - 1)
        self.tree = h.alloc(self.tree_bytes)
        self.pads = b"".join(self.dom.pad(r) for r in range(rows, self.width))
        self.write_pads()

    def fresh_salts(self) -> None:
        """New salts from the OS for the next commitment, copied on the device's stream (after every launch already on it)."""
        self.salts = os.urandom(192 * self.rows)
        self.h.ck("cuMemcpyHtoDAsync_v2", C.c_uint64(self.d_salts), C.c_char_p(self.salts), len(self.salts), self.h.stream)

    def write_pads(self) -> None:
        if self.pads:
            self.dev.ck("cuMemcpyHtoD_v2", C.c_uint64(self.tree + 64 * self.rows), C.c_char_p(self.pads), len(self.pads))

    def segments(self) -> None:
        if self.s:
            total = self.rows * self.s
            blocks = -(-total // SEG_TPB)
            self.h.launch("sz_segments", blocks, SEG_TPB, (SEG_TPB // 32) * 32 * 33 * 4, p=(self.data, self.seg_mids, self.digests),
                          i=(self.rows, self.stride, self.s, self.last))

    def finish(self, mode: int, one_cta: bool = True) -> None:
        c = self.c
        block = min(256, 32 * (-(-self.rows // 32))) if one_cta else 128
        grid = 1 if one_cta else -(-self.rows // block)
        self.h.launch("sz_finish", grid, block, p=(self.digests if self.s else self.data, self.q_mid, c["tables"], self.d_salts,
                                                    c["hm_mid"], self.leaf_heads, self.node_heads, self.x, self.bc, self.tree),
                      i=(self.rows, self.s, self.stride, mode, 0, 0, self.width, self.nbytes))
        if mode == 2 and not one_cta:
            w, off, heads = self.width, 0, self.node_heads
            while w > 1:
                self.h.launch("sz_level", -(-(w // 2) // 128), 128, p=(self.tree, heads), i=(off, w // 2))
                off, heads, w = off + w, heads + SLOT * (w // 2), w // 2

    def run(self, mode: int, one_cta: bool = True) -> None:
        self.segments()
        self.finish(mode, one_cta)

    def reference(self, rows: list[bytes]) -> dict:
        """Python's x, b ‖ c, hm96 leaves and root for these row bytes."""
        n_bits = 8 * self.nbytes
        xs = [RL.sha512_row_seg_digest(r, n_bits) if self.schema == SEG else RL.sha512_row_v2_digest(r, n_bits) for r in rows]
        bcs = [INST.commit_string(KEY, x, self.salts[192 * i:192 * (i + 1)]) for i, x in enumerate(xs)]
        hls = [INST.tree_leaf(KEY, bc) for bc in bcs]
        levels = MK.tree_levels(self.dom, [RL.row_leaf(self.dom, i, hl, self.schema) for i, hl in enumerate(hls)])
        return {"x": b"".join(xs), "bc": b"".join(bcs), "hl": b"".join(hls), "root": levels[-1][0]}

    def read_rows(self) -> list[bytes]:
        return [self.dev.read(self.data + r * self.stride, self.nbytes) for r in range(self.rows)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cubin", required=True, help="pearl_c_sm120.cu built as build.sh builds it")
    ap.add_argument("--rows-cubin", required=True, help="hm96_rows.cu built for sm_120a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = load("gpu1_run", HERE.parent / "pearl_c_sm120" / "run.py")
    vp, u64, P = C.c_void_p, C.c_uint64, C.POINTER
    image, rimage = Path(args.cubin).read_bytes(), Path(args.rows_cubin).read_bytes()
    dev = R.Dev(image)
    refused = dev.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    calls, reps, prefill = int(os.environ.get("CALLS", "32")), int(os.environ.get("REPS", "30")), int(os.environ.get("PREFILL", "8192"))
    res: dict = {"schema": "verity-pouw/served-zk-rows-bench/v0",
                 "run": os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local",
                 "device": dev.name, "uuid": dev.uuid, "sms": dev.sms, "cubin_sha256": hashlib.sha256(image).hexdigest(),
                 "rows_cubin_sha256": hashlib.sha256(rimage).hexdigest(), "refused": refused, "m_real": M_REAL,
                 "shapes": {s: list(v) for s, v in SHAPES.items()}, "layers": LAYERS, "calls": calls, "reps": reps,
                 "prefill_rows": prefill, "projection": True, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save = lambda: Path(args.out).write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    if refused:
        save()
        return 3
    dev.open()
    h = R.Dev(rimage)
    h.open()
    R.bind(dev.cu, {"cuStreamCreate": [P(vp), C.c_uint], "cuStreamBeginCapture_v2": [vp, C.c_int], "cuStreamEndCapture": [vp, P(vp)],
                    "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp],
                    "cuStreamWaitEvent": [vp, vp, C.c_uint]})
    nv = R.Nvml(dev.uuid)
    res["clk_before"] = nv.read()
    s = vp()
    dev.ck("cuStreamCreate", C.byref(s), 1)
    keys = {nm: dev.put(hashlib.shake_256(f"pearl-c-decode-served/{nm}".encode()).digest(sz)) for nm, sz in R.TIMING_KEYS.items()}
    keys.update(t_row=dev.put(R.b3s_template(R.ROW_SCHEMA)), t_tile=dev.put(R.b3s_template(R.TILE_SCHEMA)))
    consts = constants(h)
    m = R.cdiv(M_REAL, R.BM_D) * R.BM_D
    ok, graphs, gates, per = True, {}, {}, {}

    def capture(body, times: int | None = None) -> vp:
        dev.stream = h.stream = s
        dev.ck("cuStreamBeginCapture_v2", s, 0)
        for _ in range(calls if times is None else times):
            body()
        g, ge = vp(), vp()
        dev.ck("cuStreamEndCapture", s, C.byref(g))
        dev.stream = h.stream = None
        dev.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

    def poison(ptr: int, n: int) -> None:
        dev.ck("cuMemsetD8_v2", u64(ptr), POISON, n)

    def gate(cm: Commit, mode: int, tag: str, g: dict, one_cta: bool = True) -> None:
        """cm's outputs from poisoned buffers against Python's (equal), and the no-write control's (unequal)."""
        want = cm.reference(cm.read_rows())
        for run, name, same in ((lambda: cm.run(mode, one_cta), tag, True), (lambda: None, f"{tag}.control", False)):
            for p_, n_ in ((cm.x, 64 * cm.rows), (cm.bc, 128 * cm.rows), (cm.tree, cm.tree_bytes)):
                poison(p_, n_)
            cm.write_pads()
            run()
            dev.sync()
            x, bc = dev.read(cm.x, 64 * cm.rows), dev.read(cm.bc, 128 * cm.rows)
            root = dev.read(cm.tree + cm.tree_bytes - 64, 64)
            got = x == want["x"] if mode == 0 else x == want["hl"] and bc == want["bc"] and root == want["root"]
            g[name] = got == same

    for shape, (n, k) in SHAPES.items():
        rb = 4 * k
        xa, xb = dev.alloc(m * rb), dev.alloc(n * rb)
        dev.launch("gen_f32", R.cdiv(M_REAL * k, 256), 256, p=(xa,), i=(M_REAL, k, 11))
        dev.launch("gen_f32", R.cdiv(n * k, 256), 256, p=(xb,), i=(n, k, 12))
        pipes = {f: R.Pipeline(dev, m, n, k, "sm120", keys, xa, xb, m_real=M_REAL, hashing="h2", rows_form=f) for f in ("s", "w")}
        steps, phases = {}, {}
        for f, pipe in pipes.items():
            pipe.run_all()
            steps[f] = {nm: fn for nm, _, fn in pipe.steps()}
            phases[f] = [(nm, ph) for nm, ph, _ in pipe.steps()]
        dev.sync()
        pb = k + 128
        xp = dev.alloc(M_REAL * pb)
        dev.launch("gen_f32", R.cdiv(M_REAL * pb // 4, 256), 256, p=(xp,), i=(M_REAL, pb // 4, 13))
        tiles = pipes["s"].tiles
        cm_a = Commit(dev, h, consts, f"{shape}/a", xa, M_REAL, rb, rb, SEG)
        cm_p = Commit(dev, h, consts, f"{shape}/p", xp, M_REAL, pb, pb, SEG)
        cm_t = Commit(dev, h, consts, f"{shape}/t", pipes["w"].b["leaves"], tiles, 32, 32, V2)
        dev.sync()
        g: dict = {}
        gate(cm_a, 2, "a.hm96", g)
        gate(cm_a, 0, "a.sha512", g)
        gate(cm_p, 2, "p.hm96", g)
        gate(cm_t, 2, "t.hm96", g)
        gates[shape] = g
        print(f"gates {shape}: {g}", flush=True)
        ok = ok and all(g.values())
        for f in ("s", "w"):
            for nm in (*HASH_A, "seed_a", *HASH_T):
                steps[f][nm]()
        dev.sync()

        def seq(f: str, names) -> callable:
            fs = [steps[f][x] for x in names if x in steps[f]]
            return lambda: [fn() for fn in fs]

        call = {f: [nm for nm, ph in phases[f] if ph in CALL_PHASES] for f in ("s", "w")}
        nohash = [nm for nm in call["w"] if nm not in HASH_A + HASH_T]
        a_hm96, t_hm96 = (lambda cm=cm_a: cm.run(2)), (lambda cm=cm_t: cm.run(2))

        def hm96_call(a_hm96, t_hm96, fs) -> callable:
            def run() -> None:
                a_hm96()
                for fn in fs:
                    fn()
                t_hm96()
            return run

        call_hm96 = hm96_call(a_hm96, t_hm96, [steps["w"][nm] for nm in (*nohash, "hash_msg", "hash_leaf") if nm in steps["w"]])
        call_s = seq("s", call["s"])
        v = {"a.h2s": seq("s", HASH_A), "a.h2w": seq("w", HASH_A), "k.stats_a": seq("w", ("stats_a",)),
             "a.hm96": a_hm96, "a.sha512": lambda cm=cm_a: cm.run(0), "p.hm96": lambda cm=cm_p: cm.run(2),
             "t.h2": seq("s", HASH_T), "t.hm96": t_hm96,
             "call.h2": call_s, "call.nohash": seq("w", nohash), "call.hm96": call_hm96,
             "call.both": lambda c=call_s, a=a_hm96, t=t_hm96: (c(), a(), t()),
             "k.sz_segments.a": cm_a.segments, "k.sz_finish.a": lambda cm=cm_a: cm.finish(2), "k.hash_rows_a.s": seq("s", ("hash_rows_a",))}
        graphs[shape] = {nm: capture(fn) for nm, fn in v.items()}
        tdig = [steps["w"][nm] for nm in (*nohash, "hash_msg", "hash_leaf") if nm in steps["w"]]
        per[shape] = {"h2": call_s, "nohash": v["call.nohash"], "hm96": call_hm96, "tdig": hm96_call(lambda: None, lambda: None, tdig),
                      "a": [Commit(dev, h, consts, f"{shape}/a/{lyr}", xa, M_REAL, rb, rb, SEG) for lyr in range(LAYERS)],
                      "t": [Commit(dev, h, consts, f"{shape}/t/{lyr}", pipes["s"].b["leaves"], tiles, 32, 32, V2)
                            for lyr in range(LAYERS)]}
        dev.sync()
    # a decode step's four linears, LAYERS deep, and one batched commitment of all of its rows
    s2, evf, evj = vp(), vp(), vp()
    dev.ck("cuStreamCreate", C.byref(s2), 1)
    for e in (evf, evj):
        dev.ck("cuEventCreate", C.byref(e), 2)
    step_rows = {4096: 3 * LAYERS * M_REAL, 14336: LAYERS * M_REAL}
    step_tiles = LAYERS * sum(R.cdiv(M_REAL, R.TILE) * (n // R.TILE) for n, _ in SHAPES.values())
    batch = []
    for k, rows in step_rows.items():
        x = dev.alloc(rows * 4 * k)
        dev.launch("gen_f32", R.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, 31))
        batch.append(Commit(dev, h, consts, f"step/a/{k}", x, rows, 4 * k, 4 * k, SEG))
    tl = dev.alloc(step_tiles * 32)
    dev.launch("gen_f32", R.cdiv(step_tiles * 8, 256), 256, p=(tl,), i=(step_tiles, 8, 32))
    batch.append(Commit(dev, h, consts, "step/t", tl, step_tiles, 32, 32, V2))
    batch_p = []
    for k, rows in step_rows.items():
        x = dev.alloc(rows * (k + 128))
        dev.launch("gen_f32", R.cdiv(rows * (k + 128) // 4, 256), 256, p=(x,), i=(rows, (k + 128) // 4, 33))
        batch_p.append(Commit(dev, h, consts, f"step/p/{k}", x, rows, k + 128, k + 128, SEG))
    dev.sync()
    g = {}
    for cm, tag in zip(batch + batch_p, ("batch.a4096", "batch.a14336", "batch.t", "batch.p4096", "batch.p14336")):
        gate(cm, 2, tag, g, one_cta=False)
    gate(batch[0], 0, "batch.a4096.sha512", g, one_cta=False)
    gates["step"] = g
    print(f"gates step: {g}", flush=True)
    ok = ok and all(g.values())

    def forward(per_call) -> callable:
        return lambda: [per_call(lyr, sh) for lyr in range(LAYERS) for sh in SHAPES]

    def side(lyr: int, sh: str) -> None:
        per[sh]["h2"]()
        dev.ck("cuEventRecord", evf, s)
        dev.ck("cuStreamWaitEvent", s2, evf, 0)
        dev.stream = h.stream = s2
        per[sh]["a"][lyr].run(2)
        per[sh]["t"][lyr].run(2)
        dev.stream = h.stream = s

    def both_side() -> None:
        forward(side)()
        dev.ck("cuEventRecord", evj, s2)
        dev.ck("cuStreamWaitEvent", s, evj, 0)

    run_batch = lambda: [cm.run(2, one_cta=False) for cm in batch]  # noqa: E731
    step_v = {"step.nohash": forward(lambda lyr, sh: per[sh]["nohash"]()), "step.h2": forward(lambda lyr, sh: per[sh]["h2"]()),
              "step.hm96": forward(lambda lyr, sh: per[sh]["hm96"]()), "step.both.side": both_side,
              "step.h2.batch": lambda: (forward(lambda lyr, sh: per[sh]["h2"]())(), run_batch()),
              "step.tdig.batch": lambda: (forward(lambda lyr, sh: per[sh]["tdig"]())(), run_batch()),
              "batch.hm96": run_batch, "batch.sha512": lambda: [cm.run(0, one_cta=False) for cm in batch[:2]],
              "batch.p.hm96": lambda: [cm.run(2, one_cta=False) for cm in batch_p]}
    step_graphs = {nm: capture(fn, 1) for nm, fn in step_v.items()}
    dev.sync()
    res["gates"] = gates
    res["step"] = {"rows": {str(k): v for k, v in step_rows.items()}, "tiles": step_tiles}
    save()
    print(f"gates: {gates}", flush=True)
    if not ok:
        return 4

    ev = [vp(), vp()]
    for e in ev:
        dev.ck("cuEventCreate", C.byref(e), 0)

    def replay(ge) -> float:
        ms = C.c_float()
        dev.ck("cuEventRecord", ev[0], s)
        dev.ck("cuGraphLaunch", ge, s)
        dev.ck("cuEventRecord", ev[1], s)
        dev.ck("cuEventSynchronize", ev[1])
        dev.ck("cuEventElapsedTime", C.byref(ms), ev[0], ev[1])
        return ms.value

    order = [(sh, nm) for sh, gs in graphs.items() for nm in gs]
    for key in order:
        replay(graphs[key[0]][key[1]])
    t_: dict = {key: [] for key in order}
    for r in range(reps):
        for j in range(len(order)):
            key = order[(j + r) % len(order)]
            t_[key].append(replay(graphs[key[0]][key[1]]) * 1e3 / calls)
    us = {sh: {} for sh in graphs}
    for (sh, nm), xs in t_.items():
        q = statistics.quantiles(xs, n=4)
        us[sh][nm] = {"median": round(statistics.median(xs), 2), "q1": round(q[0], 2), "q3": round(q[2], 2)}
    res["us_per_call"] = us
    med = lambda nm: sum(us[sh][nm]["median"] for sh in us if nm in us[sh])  # noqa: E731
    res["per_forward_ms"] = {nm: round(LAYERS * med(nm) / 1e3, 3) for nm in next(iter(us.values()))}
    for ge in step_graphs.values():
        replay(ge)
    st_: dict = {nm: [] for nm in step_graphs}
    names = list(step_graphs)
    for r in range(reps):
        for j in range(len(names)):
            nm = names[(j + r) % len(names)]
            st_[nm].append(replay(step_graphs[nm]))
    res["step_ms"] = {nm: {"median": round(statistics.median(xs), 3), "q1": round(statistics.quantiles(xs, n=4)[0], 3),
                           "q3": round(statistics.quantiles(xs, n=4)[2], 3)} for nm, xs in st_.items()}
    print("step (ms):", {nm: v["median"] for nm, v in res["step_ms"].items()}, flush=True)
    save()

    # throughput: prefill's rows of each k, hash_rows_b3s (frame-b3s leaves) against hm96-sha512 row-seg leaves and tree
    thr = {}
    for k in sorted({k for _, k in SHAPES.values()}):
        rb = 4 * k
        x = dev.alloc(prefill * rb)
        dev.launch("gen_f32", R.cdiv(prefill * k, 256), 256, p=(x,), i=(prefill, k, 21))
        tree, sk = dev.alloc(32 * prefill), dev.alloc(32 * 16)
        b3s = lambda: dev.launch("hash_rows_b3s", prefill, 32 * R.cdiv(R.cdiv(rb, R.B3S_SEG), 32),  # noqa: E731
                                 p=(x, tree, keys["dom_a"], sk, keys["t_row"]), i=(prefill, rb, 0, R.b3s_template_len(R.ROW_SCHEMA)))
        cm = Commit(dev, h, consts, f"prefill/{k}", x, prefill, rb, rb, SEG)
        dev.sync()
        row = {}
        for nm, fn in (("b3s", b3s), ("hm96", lambda: cm.run(2, one_cta=False)), ("sha512", lambda: cm.run(0, one_cta=False)),
                       ("sz_segments", cm.segments)):
            ms = statistics.median(dev.timed(fn, 10))
            row[nm] = {"ms": round(ms, 3), "gb_s": round(prefill * rb / ms / 1e6, 1)}
        thr[f"k{k}"] = row
        dev.free(x)
        print(f"throughput k={k}: {row}", flush=True)
    res["prefill_throughput"] = thr
    res["clk_after"] = nv.read()
    save()
    for sh in us:
        print(sh, {nm: us[sh][nm]["median"] for nm in us[sh]}, flush=True)
    print("per forward (ms, 32 layers):", res["per_forward_ms"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
