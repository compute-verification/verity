"""The sketch-then-hash benchmark on one GPU: `sketch.cu`'s libraries (`--lib NAME=PATH`, one per payload packing)
gated against `sketch.py`, then timed against the native FP8 GEMM. Writes `sketch-tag.json`, `codec-frontier-points.json`
and `bench.log` under `--out`, and in a `research run` its `result.json`.

Nothing is timed before its gate passed. At each of Llama-3.1-8B's linears, for the plain store and the encoded one
(`--kr` rounds of the word codec), and for every k (`--ks`) and n (`--ns`, rounded to whole tiles: n = 32 m T):
- the fused kernel's partial sums equal the plain FP8 kernel's bit for bit (both runs: a run a tile, and the shorter runs
  added atomically), and the plain kernel's are within FP32 rounding of torch's;
- the device sketch equals `sketch.sketch` (exact, numpy) over the store's vectors, every vector;
- the device tags equal `sketch.tags_of` (ro-sketch-blake3/v0 through the bandwidth audit's `Blake3Tag`), every tag at the
  smallest linear and sampled at the others, and the leaves and tagger.cu's root equal `sketch.leaves` / `sketch.root`.
Then interleaved reps, each arm queued behind a spin kernel, every call on cold weights (a ring of copies past the L2):
cuBLASLt FP8 (`torch._scaled_mm`), the word codec's plain warp-specialized kernel, this kernel without the sketch
(`base`), with it (`sk`), the hash and the leaves alone, and the three back to back (`stack`); on the encoded store also
the decode without the sketch and the word codec's full-byte ro-blake3/v1 in the same kernel (`dt`, `dtc`).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "word_codec"))
sys.path.insert(2, str(HERE.parents[1]))

import codec as C  # noqa: E402
import sketch as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("wc_bench", HERE.parent / "word_codec" / "bench.py")
WB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WB)

SEED = WB.SEED
SKETCH_SEED = hashlib.sha256(b"verity/sketch-tag/bench/seed").digest()
SHAPES = WB.SHAPES
LAYERS = WB.LAYERS


class Lib(WB.Lib):
    def __init__(self, name: str, path: str) -> None:
        super().__init__(name, path)
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        so = self.so
        so.sk_gemm.argtypes = [i, i, v, v, v, i, i, i, q, i, i, i, i, v, i, i, v, v, i, ctypes.c_char_p, i, v, q, q, v, i, v]
        so.sk_hash.argtypes = [i, v, q, ctypes.c_char_p, q, v, i, v]
        so.sk_leaves.argtypes = [v, q, ctypes.c_char_p, v]
        cs = (ctypes.c_int * 8)()
        so.sk_constants(cs)
        self.sk_constants = list(cs)
        self.steps = {0: cs[3], 1: cs[4]}
        if cs[3] != S.steps(S.PLAIN) or cs[4] != S.steps(self.bits):
            raise RuntimeError(f"{name}: steps {cs[3]}, {cs[4]} are not sketch.py's")
        self.skmax = cs[5]


def divisors(t: int) -> list[int]:
    return [d for d in range(1, t + 1) if t % d == 0]


def tile_rows(n_target: int, bits: int) -> int:
    return max(1, round(n_target / (32 * S.steps(bits))))


class Sketch:
    """One (layout, k, n) on one linear: the packed signs, the sketch buffer, the commit workspace."""

    def __init__(self, lib: Lib, torch, n: int, kd: int, bits: int, ks: int, n_target: int, warps: int) -> None:
        self.bits, self.ks, self.t = bits, ks, tile_rows(n_target, bits)
        self.m = S.steps(bits)
        self.n = 32 * self.m * self.t
        self.n_target = n_target
        self.sign_bits = S.sign_bits(SKETCH_SEED, ks, self.n)
        self.frag = torch.from_numpy(S.sign_fragments(self.sign_bits, self.m, self.t)).cuda()
        self.tiles = S.tiles(n, kd, self.t)
        self.nvec = 16 * self.tiles
        self.sk = torch.zeros(self.nvec * ks, dtype=torch.int32, device="cuda")
        self.commit = WB.Commit(lib, torch, self.nvec)
        self.sk2 = torch.zeros_like(self.sk)
        self.commit2 = WB.Commit(lib, torch, self.nvec)
        nbs, nch = n // 16, kd // C.TASK_COLS
        groups = -(-nbs // self.t)
        # the longest run that still gives every consumer warp of the device two runs
        self.run = max([d for d in divisors(self.t) if groups * (self.t // d) * nch >= 2 * warps] or [1])

    def name(self) -> str:
        return f"{self.ks}_{self.n_target}"

    def side_buffers(self, lib: Lib, torch, count: int) -> list:
        """count (sketch buffer, commit workspace) pairs for the side-stream commit, the first two the gated ones."""
        if not hasattr(self, "_side"):
            self._side = [(self.sk, self.commit), (self.sk2, self.commit2)]
        while len(self._side) < count:
            self._side.append((torch.zeros_like(self.sk), WB.Commit(lib, torch, self.nvec)))
        return self._side[:count]


def gemm(lib: Lib, torch, kr: int, ks: int, wbuf, x, y, n: int, kd: int, partials: bool, cfg, t: int, run: int,
         frag=None, sk=None, decode: int = 1, groups: int = 0, launched=None, commit=None, committers: int = 0) -> None:
    """One sk_gemm call; commit = (a previous call's sketch tensor, its vectors, a Commit, clear) for `committers` warps
    a CTA to hash."""
    csk, cnvec, cws = (WB.ptr(commit[0]), commit[1], commit[2].ptr) if commit else (None, 0, None)
    lib.call("sk_gemm", kr, ks, WB.ptr(wbuf), WB.ptr(x.view(torch.uint8)), WB.ptr(y), x.shape[0], n, kd, 0,
             int(partials), cfg[0], cfg[1], groups, launched, t, run, WB.ptr(frag) if frag is not None else None,
             WB.ptr(sk) if sk is not None else None, decode, SKETCH_SEED, committers if commit else 0, csk, cnvec, 0,
             cws, int(commit[3]) if commit else 0, WB.stream(torch))


def digest(tags: list[bytes], leaves: list[bytes]) -> bytes:
    return hashlib.sha256(b"".join(tags) + b"".join(leaves)).digest()


def gate_sketch(lib: Lib, torch, sko: Sketch, store_np: np.ndarray, n: int, kd: int, kr: int, wbuf, x, pp, cfg,
                full_tags: bool, committers_of, extra, log) -> dict:
    out = {"n": sko.n, "T": sko.t, "run": sko.run, "vectors": sko.nvec}
    ref = S.sketch_f64(S.vectors(store_np, n, kd, sko.bits, sko.t), S.signs(SKETCH_SEED, sko.ks, sko.n))
    out["max_abs"] = int(np.abs(ref).max())
    for run in sorted({sko.t, sko.run}):
        sko.sk.zero_()
        y = torch.zeros_like(pp)
        gemm(lib, torch, kr, sko.ks, wbuf, x, y, n, kd, True, cfg, sko.t, run, sko.frag, sko.sk)
        torch.cuda.synchronize()
        dev = sko.sk.cpu().numpy().reshape(sko.nvec, sko.ks).astype(np.int64)
        out[f"run{run}"] = {"y_equal_plain": bool(torch.equal(y, pp)),
                            "sketch_equal": bool(np.array_equal(dev, ref)),
                            "bad_vectors": int((dev != ref).any(axis=1).sum())}
    lib.call("sk_hash", sko.ks, WB.ptr(sko.sk), sko.nvec, SKETCH_SEED, 0, sko.commit.tags, 1, WB.stream(torch))
    lib.call("sk_leaves", sko.commit.ptr, sko.nvec, SKETCH_SEED, WB.stream(torch))
    torch.cuda.synchronize()
    out["cleared"] = not bool(sko.sk.any())
    root = (ctypes.c_uint8 * 64)()
    lib.call("wc_commit_tree", sko.commit.ptr, sko.nvec, root, WB.stream(torch))
    torch.cuda.synchronize()
    ys = sko.commit.tag_list()
    if full_tags:
        want = S.tags_of(SKETCH_SEED, ref)
        out["tags_checked"] = len(want)
        out["bad_tags"] = sum(a != b for a, b in zip(ys, want))
        leaves = S.leaves(SKETCH_SEED, want)
        out["leaves_ok"] = sko.commit.leaf_list() == leaves
        out["root_ok"] = bytes(root) == WB.rfc6962_root(leaves)
    else:
        rng = random.Random(sko.nvec * 7 + sko.ks)
        js = sorted({0, 1, 15, 16, sko.nvec - 1, *(rng.randrange(sko.nvec) for _ in range(48))})
        keyed, words = S.TAG.keyed(SKETCH_SEED), S.sketch_words(ref)
        out["tags_checked"] = len(js)
        out["bad_tags"] = sum(ys[j] != keyed(j, words[j].tobytes()) for j in js)
        leaves = S.leaves(SKETCH_SEED, ys)
        out["leaves_ok"] = sko.commit.leaf_list() == leaves
        out["root_ok"] = bytes(root) == WB.rfc6962_root(leaves)
    sko.digest = digest(ys, leaves)
    # the committer warps, at each configuration: call 1 sketches into sk, call 2 into sk2 while hashing sk into commit2
    clear = sko.run < sko.t
    for c in [cfg, *extra]:
        sko.sk.zero_()
        sko.sk2.zero_()
        y = torch.zeros_like(pp)
        gemm(lib, torch, kr, sko.ks, wbuf, x, y, n, kd, True, c, sko.t, sko.run, sko.frag, sko.sk)
        y.zero_()
        gemm(lib, torch, kr, sko.ks, wbuf, x, y, n, kd, True, c, sko.t, sko.run, sko.frag, sko.sk2,
             commit=(sko.sk, sko.nvec, sko.commit2, clear), committers=committers_of(c))
        torch.cuda.synchronize()
        dev2 = sko.sk2.cpu().numpy().reshape(sko.nvec, sko.ks).astype(np.int64)
        out[f"committers_{cfg_name(c)}"] = {
            "y_equal_plain": bool(torch.equal(y, pp)), "sketch_equal": bool(np.array_equal(dev2, ref)),
            "tags_equal": sko.commit2.tag_list() == ys, "leaves_equal": sko.commit2.leaf_list() == sko.commit.leaf_list(),
            "cleared": (not bool(sko.sk.any())) if clear else True}
    out["ok"] = (all(out[f"run{r}"]["y_equal_plain"] and out[f"run{r}"]["sketch_equal"] for r in {sko.t, sko.run})
                 and out["bad_tags"] == 0 and out["leaves_ok"] and out["root_ok"] and out["cleared"]
                 and all(all(v.values()) for k, v in out.items() if k.startswith("committers_")))
    return out


def cfg_name(cfg) -> str:
    return f"c{cfg[0]}s{cfg[1]}"


def case(lib: Lib, torch, name: str, n: int, kd: int, a, warps_of, log) -> dict:
    cfg, extra = a.cfgs[0], a.cfgs[1:]

    def committers_of(c):
        return min(a.committers, WB.WS_MAX_WARPS - 1 - c[0])

    focus_of = {lay: (lay, 16 if 16 in a.ks else a.ks[0], 4096 if 4096 in a.ns else a.ns[0]) for lay in ("plain", "enc")}
    w = WB.fp8_weights(torch, n, kd, 100 + n + kd)
    wb = w.view(torch.uint8)
    plain = WB.plain_layout(wb)
    layouts = {"plain": (0, plain)}
    if a.kr:
        payloads = WB.fragment_payloads(torch, wb, lib.bits)
        WB.set_key(lib, torch, C.key(SEED, a.kr))
        enc = torch.empty_like(payloads)
        lib.call("wc_encode", WB.ptr(payloads), WB.ptr(enc), payloads.numel(), 0, a.kr, 188 * 6, WB.stream(torch))
        WB.set_key(lib, torch, C.key(SEED, WB.KMAX))
        torch.cuda.synchronize()
        del payloads
        layouts["enc"] = (a.kr, enc)
    nch = kd // C.TASK_COLS
    xs = {}
    gx = torch.Generator(device="cuda").manual_seed(5)
    for mb in a.batches:
        xs[mb] = torch.randn(mb, kd, device="cuda", generator=gx).to(torch.float8_e4m3fn)
    warps = warps_of(n, kd)
    res = {"N": n, "K": kd, "weight_bytes": n * kd, "gate": {}, "times": {}, "warps": warps}
    sks = {}
    for lay, (kr, wbuf) in layouts.items():
        bits = S.PLAIN if kr == 0 else lib.bits
        res[f"{lay}_store_bytes"] = wbuf.numel() * wbuf.element_size()
        for ks in a.ks:
            for nt in a.ns:
                sks[(lay, ks, nt)] = Sketch(lib, torch, n, kd, bits, ks, nt, warps)

    # gates
    store_np = {lay: wbuf.cpu().numpy().view(np.uint8).reshape(-1) for lay, (kr, wbuf) in layouts.items()} \
        if a.mode == "bench" else {}
    for mb in a.batches if a.mode == "bench" else ():
        x = xs[mb]
        ref = x.float() @ w.float().t()
        pp = torch.zeros(nch * mb * n, dtype=torch.float32, device="cuda")
        lib.call("wc_gemm", 0, WB.ptr(plain), WB.ptr(x.view(torch.uint8)), WB.ptr(pp), mb, n, kd, 0, 1,
                 WB.stream(torch))
        scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-30)
        err = float(((pp.view(nch, mb, n).sum(0) - ref).abs() / scale).max())
        entry = {"plain_vs_torch_rel": err, "base_equal_plain": {}}
        for lay, (kr, wbuf) in layouts.items():
            y = torch.zeros_like(pp)
            sk0 = sks[(lay, a.ks[0], a.ns[0])]
            gemm(lib, torch, kr, 0, wbuf, x, y, n, kd, True, cfg, sk0.t, sk0.run)
            entry["base_equal_plain"][lay] = bool(torch.equal(y, pp))
        if mb == a.batches[0]:
            for (lay, ks, nt), sko in sks.items():
                kr, wbuf = layouts[lay]
                g = gate_sketch(lib, torch, sko, store_np[lay], n, kd, kr, wbuf, x, pp, cfg,
                                full_tags=name == a.full_tags, committers_of=committers_of,
                                extra=extra if (lay, ks, nt) == focus_of[lay] else [], log=log)
                entry.setdefault("sketch", {})[f"{lay}_{sko.name()}"] = g
                log(f"sketch gate {name} {lay} k={ks} n={sko.n} (T={sko.t}, run {sko.run}): "
                    f"{'ok' if g['ok'] else 'FAIL'} ({g['vectors']} vectors, {g['tags_checked']} tags, "
                    f"max |sketch| {g['max_abs']})")
        else:
            for (lay, ks, nt), sko in sks.items():
                if nt != a.ns[0] or ks != a.ks[-1]:
                    continue
                kr, wbuf = layouts[lay]
                y = torch.zeros_like(pp)
                gemm(lib, torch, kr, ks, wbuf, x, y, n, kd, True, cfg, sko.t, sko.run, sko.frag, sko.sk)
                entry.setdefault("sketch_y_equal_plain", {})[f"{lay}_{sko.name()}"] = bool(torch.equal(y, pp))
        entry["ok"] = (err < 1e-5 and all(entry["base_equal_plain"].values())
                       and all(g["ok"] for g in entry.get("sketch", {}).values())
                       and all(entry.get("sketch_y_equal_plain", {}).values()))
        res["gate"][mb] = entry
        log(f"gemm gate {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (plain rel {err:.1e})")
        if not entry["ok"]:
            res["ok"] = False
            return res
    res["ok"] = True
    del store_np

    # timing
    ring = {}
    for lay, (kr, wbuf) in layouts.items():
        copies = max(2, -(-WB.L2_DEFEAT_BYTES // (wbuf.numel() * wbuf.element_size()))) if a.mode == "bench" else 1
        ring[lay] = [wbuf] + [wbuf.clone() for _ in range(copies - 1)]
    wring = [w] + [w.clone() for _ in range(len(ring["plain"]) - 1)]
    res["copies"] = {lay: len(r) for lay, r in ring.items()}
    spin_s = statistics.median(WB.timed(torch, lambda: lib.call("wc_spin", WB.SPIN_CYCLES, WB.stream(torch)), 3))
    so, st = lib.so, WB.stream(torch)
    tagcm = [WB.Commit(lib, torch, layouts["enc"][1].numel() // C.lane_words(lib.bits)) for _ in range(2)] \
        if "enc" in layouts else []
    for mb in a.batches:
        x = xs[mb]
        xp = WB.ptr(x.view(torch.uint8))
        y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        yp = WB.ptr(y)
        one = torch.ones((), device="cuda")
        try:
            torch._scaled_mm(x, w.t(), one, one, out_dtype=torch.bfloat16)
            xc = x
        except Exception:  # noqa: BLE001  cuBLASLt's FP8 path needs M a multiple of 16 here
            xc = torch.zeros(16 * -(-mb // 16), kd, device="cuda").to(torch.float8_e4m3fn)
            xc[:mb] = x
        wts = [wc_.t() for wc_ in wring]
        launched = {}

        def arm_cublas():
            for wt in wts:
                torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)

        def arm_wc_ws(k, lay):
            ps = [WB.ptr(b) for b in ring[lay]]
            lc = ctypes.c_int(0)
            lib.call("wc_gemm_ws", k, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(lc), st)
            groups = lc.value // nch

            def f():
                rc = 0
                for p in ps:
                    rc |= so.wc_gemm_ws(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, st)
                if rc:
                    raise RuntimeError(f"wc_gemm_ws {k}: error {rc}")
            return f

        def arm_wc_tag(k, commit):
            ps = [WB.ptr(b) for b in ring["enc"]]
            blocks = tagcm[0].blocks
            c = 1 if commit else 0
            lc = ctypes.c_int(0)
            lib.call("wc_gemm_ws_tag", k, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(lc), 1,
                     WB.TAG_SEED, 0, tagcm[0].tags, c, tagcm[1].tags if c else None, blocks if c else 0,
                     tagcm[1].leaves if c else None, st)
            groups = lc.value // nch

            def f():
                rc = 0
                for i, p in enumerate(ps):
                    u, v = tagcm[i % 2], tagcm[1 - i % 2]
                    rc |= so.wc_gemm_ws_tag(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, 1,
                                            WB.TAG_SEED, 0, u.tags, c, v.tags if c else None, blocks if c else 0,
                                            v.leaves if c else None, st)
                if rc:
                    raise RuntimeError(f"wc_gemm_ws_tag {k}: error {rc}")
            return f

        def arm_sk(lay, sko, ks, decode=1, stack=False, cfg=cfg):
            kr = layouts[lay][0]
            ps = [WB.ptr(b) for b in ring[lay]]
            lc = ctypes.c_int(0)
            fp, skp = (WB.ptr(sko.frag), WB.ptr(sko.sk)) if ks else (None, None)
            lib.call("sk_gemm", kr, ks, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(lc), sko.t,
                     sko.run, fp, skp, decode, SKETCH_SEED, 0, None, 0, 0, None, 0, st)
            groups = lc.value // nch
            launched[f"{lay}_{ks}_{sko.n_target}_{cfg_name(cfg)}"] = lc.value
            nvec, cptr, tags = sko.nvec, sko.commit.ptr, sko.commit.tags
            clear = int(sko.run < sko.t)

            def f():
                rc = 0
                for p in ps:
                    rc |= so.sk_gemm(kr, ks, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, sko.t,
                                     sko.run, fp, skp, decode, SKETCH_SEED, 0, None, 0, 0, None, 0, st)
                    if stack:
                        rc |= so.sk_hash(ks, skp, nvec, SKETCH_SEED, 0, tags, clear, st)
                        rc |= so.sk_leaves(cptr, nvec, SKETCH_SEED, st)
                if rc:
                    raise RuntimeError(f"sk_gemm {lay} {ks}: error {rc}")
            return f

        def arm_fused(lay, sko, ks, cfg=cfg):
            """Call i sketches into buffer i mod 2 while its committer warps hash buffer (i + 1) mod 2, the previous
            call's, into that buffer's commit (tags and leaves)."""
            kr = layouts[lay][0]
            ps = [WB.ptr(b) for b in ring[lay]]
            fp = WB.ptr(sko.frag)
            bufs = [(WB.ptr(sko.sk), sko.commit2.ptr), (WB.ptr(sko.sk2), sko.commit.ptr)]
            clear, cn = int(sko.run < sko.t), committers_of(cfg)
            lc = ctypes.c_int(0)
            lib.call("sk_gemm", kr, ks, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(lc), sko.t,
                     sko.run, fp, bufs[0][0], 1, SKETCH_SEED, cn, bufs[1][0], sko.nvec, 0, bufs[1][1], clear, st)
            groups = lc.value // nch

            def f():
                rc = 0
                for i, p in enumerate(ps):
                    cur, prev = bufs[i % 2], bufs[1 - i % 2]
                    rc |= so.sk_gemm(kr, ks, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, sko.t, sko.run,
                                     fp, cur[0], 1, SKETCH_SEED, cn, prev[0], sko.nvec, 0, prev[1], clear, st)
                if rc:
                    raise RuntimeError(f"sk_gemm fused {lay} {ks}: error {rc}")
            return f

        def arm_side(lay, sko, ks, nbuf=4):
            """The GEMMs on a high-priority stream, call i sketching into buffer i mod nbuf; each call's hash and leaves
            on a low-priority side stream behind it, so they overlap the next calls (call i waits for buffer i's
            previous commit); the arm ends when both streams have."""
            kr = layouts[lay][0]
            ps = [WB.ptr(b) for b in ring[lay]]
            fp = WB.ptr(sko.frag)
            bufs = [(WB.ptr(b), c) for b, c in sko.side_buffers(lib, torch, nbuf)]
            clear, nvec = int(sko.run < sko.t), sko.nvec
            lc = ctypes.c_int(0)
            lib.call("sk_gemm", kr, ks, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(lc), sko.t,
                     sko.run, fp, bufs[0][0], 1, SKETCH_SEED, 0, None, 0, 0, None, 0, st)
            groups = lc.value // nch
            lo, hi = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=-1)
            lo_h, hi_h, cur = lo.cuda_stream, hi.cuda_stream, torch.cuda.current_stream()
            start, made, done = torch.cuda.Event(), [torch.cuda.Event() for _ in bufs], [torch.cuda.Event() for _ in bufs]

            def f():
                rc = 0
                start.record(cur)
                hi.wait_event(start)
                lo.wait_event(start)
                for i, p in enumerate(ps):
                    b = i % nbuf
                    skp, cmt = bufs[b]
                    if i >= nbuf:
                        hi.wait_event(done[b])
                    rc |= so.sk_gemm(kr, ks, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, sko.t, sko.run,
                                     fp, skp, 1, SKETCH_SEED, 0, None, 0, 0, None, 0, hi_h)
                    made[b].record(hi)
                    lo.wait_event(made[b])
                    rc |= so.sk_hash(ks, skp, nvec, SKETCH_SEED, 0, cmt.tags, clear, lo_h)
                    rc |= so.sk_leaves(cmt.ptr, nvec, SKETCH_SEED, lo_h)
                    done[b].record(lo)
                for b in range(min(nbuf, len(ps))):
                    cur.wait_event(done[b])
                if rc:
                    raise RuntimeError(f"sk_gemm side {lay} {ks}: error {rc}")
            if mb == a.batches[0] and a.mode == "bench" and (lay, ks, sko.n_target) == focus_of[lay]:
                # after a pass over the ring every buffer holds the gated tags and leaves
                for b, _ in sko.side_buffers(lib, torch, nbuf):
                    b.zero_()
                f()
                torch.cuda.synchronize()
                res.setdefault("side_equal", {})[lay] = all(
                    digest(c.tag_list(), c.leaf_list()) == sko.digest for _, c in sko.side_buffers(lib, torch, nbuf))
            return f

        def arm_hash(lay, sko, leaves_only=False):
            copies = len(ring[lay])
            skp, nvec, cptr, tags = WB.ptr(sko.sk), sko.nvec, sko.commit.ptr, sko.commit.tags
            clear = int(sko.run < sko.t)

            def f():
                rc = 0
                for _ in range(copies):
                    if leaves_only:
                        rc |= so.sk_leaves(cptr, nvec, SKETCH_SEED, st)
                    else:
                        rc |= so.sk_hash(sko.ks, skp, nvec, SKETCH_SEED, 0, tags, clear, st)
                if rc:
                    raise RuntimeError(f"sk_hash: error {rc}")
            return f

        def arm_empty():
            rc = 0
            for _ in range(len(ring["plain"])):
                rc |= so.wc_empty((n // 16 // lib.constants[5]) * nch, st)
            if rc:
                raise RuntimeError(f"wc_empty: error {rc}")

        arms = {"cublas": arm_cublas, "plain_ws": arm_wc_ws(0, "plain"), "empty": arm_empty}
        for lay in layouts:
            first = sks[(lay, a.ks[0], a.ns[a.ns.index(4096) if 4096 in a.ns else 0])]
            arms[f"base_{lay}"] = arm_sk(lay, first, 0, decode=0)
            if lay == "enc":
                arms["dec_enc"] = arm_sk(lay, first, 0, decode=1)
                arms["wcws_enc"] = arm_wc_ws(a.kr, "enc")
                arms["dt_enc"] = arm_wc_tag(a.kr, False)
                arms["dtc_enc"] = arm_wc_tag(a.kr, True)
            for (l2, ks, nt), sko in sks.items():
                if l2 != lay:
                    continue
                arms[f"sk_{lay}_{sko.name()}"] = arm_sk(lay, sko, ks)
                arms[f"stack_{lay}_{sko.name()}"] = arm_sk(lay, sko, ks, stack=True)
                arms[f"fused_{lay}_{sko.name()}"] = arm_fused(lay, sko, ks)
                arms[f"side_{lay}_{sko.name()}"] = arm_side(lay, sko, ks)
                if (l2, ks, nt) == focus_of[lay]:
                    for c in extra:
                        arms[f"base_{lay}@{cfg_name(c)}"] = arm_sk(lay, first, 0, decode=0, cfg=c)
                        if lay == "enc":
                            arms[f"dec_enc@{cfg_name(c)}"] = arm_sk(lay, first, 0, decode=1, cfg=c)
                        arms[f"sk_{lay}_{sko.name()}@{cfg_name(c)}"] = arm_sk(lay, sko, ks, cfg=c)
                        arms[f"fused_{lay}_{sko.name()}@{cfg_name(c)}"] = arm_fused(lay, sko, ks, cfg=c)
                if mb == a.batches[0]:
                    arms[f"hash_{lay}_{sko.name()}"] = arm_hash(lay, sko)
                    if ks == a.ks[0]:
                        arms[f"leaves_{lay}_{nt}"] = arm_hash(lay, sko, leaves_only=True)
        for f in arms.values():
            f()
        torch.cuda.synchronize()
        if a.mode == "ncu":
            continue
        times = {k: [] for k in arms}
        names = list(arms)
        host_max = 0.0
        for rep in range(a.reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for k in order:
                t, h = WB.queued(lib, torch, arms[k])
                lay = "enc" if "enc" in k else "plain"
                times[k].append(t / len(ring[lay]))
                host_max = max(host_max, h)
        med = {k: statistics.median(v) for k, v in times.items()}
        native = min(med["cublas"], med["plain_ws"])
        # each arm at its fastest configuration
        best_s, best_cfg = {}, {}
        for k, v in med.items():
            if "@" in k:
                continue
            alts = [(v, cfg_name(cfg))] + [(med[f"{k}@{cfg_name(c)}"], cfg_name(c)) for c in extra
                                           if f"{k}@{cfg_name(c)}" in med]
            best_s[k], best_cfg[k] = min(alts)
        res["times"][mb] = {"median_s": med, "best_s": best_s, "best_cfg": best_cfg, "times_s": times,
                            "native_s": native, "launched": launched,
                            "host_enqueue_max_s": host_max, "host_bound": host_max > 0.8 * spin_s,
                            "ratio_vs_native": {k: v / native for k, v in med.items()}}
        best = {lay: min((min(med[f"{m}_{lay}_{s.name()}"] for m in ("stack", "fused", "side")) / native, s.name())
                         for (l2, _, _), s in sks.items() if l2 == lay) for lay in layouts}
        log(f"gemm {name} M={mb}: cublas {med['cublas'] * 1e6:.1f} us, plain_ws {med['plain_ws'] * 1e6:.1f} us, "
            f"native {native * 1e6:.1f} us; base_plain {med['base_plain'] / native:.3f}x; "
            + "; ".join(f"sk_{lay}_{s.name()} {med[f'sk_{lay}_{s.name()}'] / native:.3f}x"
                        f" stack {med[f'stack_{lay}_{s.name()}'] / native:.3f}x"
                        f" fused {med[f'fused_{lay}_{s.name()}'] / native:.3f}x"
                        f" side {med[f'side_{lay}_{s.name()}'] / native:.3f}x"
                        for (lay, ks, nt), s in sks.items() if s.name() == "16_4096")
            + (f"; dec_enc {med['dec_enc'] / native:.3f}x, dt_enc {med['dt_enc'] / native:.3f}x, "
               f"dtc_enc {med['dtc_enc'] / native:.3f}x" if "enc" in layouts else "")
            + f"; best stack {best}" + (" HOST-BOUND" if host_max > 0.8 * spin_s else ""))
    res["sketches"] = {f"{lay}_{s.name()}": {"n": s.n, "T": s.t, "run": s.run, "vectors": s.nvec, "tiles": s.tiles,
                                             "sign_bytes": s.frag.numel(), "sketch_bytes": s.sk.numel() * 4}
                       for (lay, ks, nt), s in sks.items()}
    res["ok"] = all(res.get("side_equal", {}).values())
    del ring, wring, layouts, plain, w, sks, tagcm
    torch.cuda.empty_cache()
    return res


def frontier(res: dict, lib: Lib, a) -> list[dict]:
    """Layer-weighted (the four linears' sum) per layout, k, n and batch: y_total = stack / native - 1, split as base,
    decode, sketch, hash, commit (leaves; the tree above them is a window's, under 0.1% of it)."""
    cases = res["gemm"]
    points = []
    for mb in a.batches:
        for lay in ("plain", "enc"):
            for ks in a.ks:
                for nt in a.ns:
                    tot = {"native": 0.0, "base": 0.0, "dec": 0.0, "sk": 0.0, "hash": 0.0, "leaves": 0.0, "stack": 0.0,
                           "fused": 0.0, "side": 0.0, "dt": 0.0, "dtc": 0.0}
                    ok = True
                    for name, g in cases.items():
                        tm = g.get("times", {}).get(mb)
                        if not tm or f"sk_{lay}_{ks}_{nt}" not in tm["median_s"]:
                            ok = False
                            break
                        med, first = tm["best_s"], g["times"][a.batches[0]]["median_s"]
                        tot["native"] += tm["native_s"]
                        tot["base"] += med[f"base_{lay}"]
                        tot["dec"] += med.get("dec_enc", med[f"base_{lay}"]) if lay == "enc" else med[f"base_{lay}"]
                        tot["sk"] += med[f"sk_{lay}_{ks}_{nt}"]
                        tot["hash"] += first[f"hash_{lay}_{ks}_{nt}"]
                        tot["leaves"] += first[f"leaves_{lay}_{nt}"]
                        tot["stack"] += med[f"stack_{lay}_{ks}_{nt}"]
                        tot["fused"] += med[f"fused_{lay}_{ks}_{nt}"]
                        tot["side"] += med[f"side_{lay}_{ks}_{nt}"]
                        if lay == "enc":
                            tot["dt"] += med["dt_enc"]
                            tot["dtc"] += med["dtc_enc"]
                    if not ok:
                        continue
                    nat = tot["native"]
                    sk0 = next(iter(cases.values()))["sketches"][f"{lay}_{ks}_{nt}"]
                    pt = {"schema": "verity/codec-frontier-point/v0", "lane": "ra-sketch-tag",
                          "b": 61 if lay == "enc" else 8, "k": a.kr if lay == "enc" else 0,
                          "arithmetic": f"cuda-int8-mma/sketch-k{ks}-n{sk0['n']}/{lib.name}/pb{lib.bits}"
                                        if lay == "enc" else f"cuda-int8-mma/sketch-k{ks}-n{sk0['n']}/plain",
                          "batch": int(mb), "y_total": min(tot["stack"], tot["fused"], tot["side"]) / nat - 1,
                          "commit_mode": min(("separate", "fused", "side"),
                                             key=lambda m: tot["stack" if m == "separate" else m]),
                          "y_total_separate": tot["stack"] / nat - 1, "y_total_fused": tot["fused"] / nat - 1,
                          "y_total_side": tot["side"] / nat - 1,
                          "y_base": tot["base"] / nat - 1, "y_decode": (tot["dec"] - tot["base"]) / nat,
                          "y_tag": (tot["sk"] - tot["dec"]) / nat + tot["hash"] / nat,
                          "y_sketch": (tot["sk"] - tot["dec"]) / nat, "y_hash": tot["hash"] / nat,
                          "y_commit": tot["leaves"] / nat, "y_sum": (tot["sk"] + tot["hash"] + tot["leaves"]) / nat - 1,
                          "x": None, "delta": None, "tau": tot["hash"] / nat, "measured": True,
                          "label": f"sketch-then-hash k={ks} n={sk0['n']} ({'encoded' if lay == 'enc' else 'plain'})",
                          "sketch_k": ks, "sketch_n": sk0["n"], "hashed_fraction": 4 * ks / sk0["n"],
                          "native_step_s": nat, "stack_step_s": tot["stack"], "fused_step_s": tot["fused"],
                          "side_step_s": tot["side"],
                          "run": os.environ.get("RESEARCH_RUN_ID"), "art": None}
                    if lay == "enc":
                        pt["y_fullbyte_tag"] = (tot["dt"] - tot["dec"]) / nat
                        pt["y_fullbyte_total"] = tot["dtc"] / nat - 1
                    points.append(pt)
    return points


def write_result(ok: bool, points: list[dict], libs: list[Lib], out: Path) -> None:
    """`research/result/v0.1` in $RESEARCH_RUN_DIR (`tools/research/src/research/result.py`), when the run has one."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = []
    for p in points:
        lay = "enc" if p["b"] == 61 else "plain"
        stem = f"{lay}_k{p['sketch_k']}_n{p['sketch_n']}_m{p['batch']}"
        ms += [{"name": f"y_{f}_{stem}", "value": p[f"y_{f}"], "unit": "1"}
               for f in ("total", "total_fused", "total_side", "total_separate", "sketch", "hash", "commit") if p.get(f"y_{f}") is not None]
    rel = os.path.relpath(out, run_dir)
    files = [f"{rel}/sketch-tag.json", f"{rel}/codec-frontier-points.json", f"{rel}/bench.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": SKETCH_SEED.hex()},
           "validation": {"status": "passed" if ok else "failed", "checker": "sketch-tag gates",
                          "detail": "y = plain FP8 kernel's, sketch = sketch.py's int64 over every vector, "
                                    "tags/leaves/root = ro-sketch-blake3/v0's", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH, a build of sketch.cu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--kr", type=int, default=2, help="the word codec's rounds on the encoded store (0: plain only)")
    ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--ns", default="1024,4096,16384")
    ap.add_argument("--cfgs", default="4x2,8x1,6x1",
                    help="consumer warps x stages each: every arm at the first, base, decode, sketch and fused also at "
                         "the rest for k=16, n=4096 (each arm's fastest is the frontier's)")
    ap.add_argument("--committers", type=int, default=2, help="committer warps a CTA in the fused arm")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--full-tags", default="o", help="the linear whose every tag is checked against the reference")
    ap.add_argument("--mode", default="bench", choices=["bench", "ncu"])
    a = ap.parse_args()
    a.batches = [int(b) for b in a.batches.split(",")]
    a.ks = [int(k) for k in a.ks.split(",")]
    a.ns = [int(v) for v in a.ns.split(",")]
    a.cfgs = [tuple(int(v) for v in c.split("x")) for c in a.cfgs.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "bench.log", "a")
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    libs = [Lib(*spec.split("=", 1)) for spec in a.lib]
    res = {"schema": "verity/sketch-tag/bench/v0", "device": props.name, "sms": sms, "torch": torch.__version__,
           "args": vars(a), "run_id": os.environ.get("RESEARCH_RUN_ID"), "libs": {}}
    ok = True
    for lib in libs:
        if a.kr > lib.skmax:
            raise SystemExit(f"{lib.name}: built for kr <= {lib.skmax}")
        warp_cache = {}

        def warps_of(n, kd, lib=lib):
            if (n, kd) not in warp_cache:
                x = torch.zeros(1, kd, device="cuda").to(torch.float8_e4m3fn)
                y = torch.zeros(kd // C.TASK_COLS, n, device="cuda")
                w = torch.zeros(n * kd, dtype=torch.uint8, device="cuda")
                lc = ctypes.c_int(0)
                gemm(lib, torch, 0, 0, w, x, y, n, kd, False, a.cfgs[0], 1, 1,
                     launched=ctypes.byref(lc))
                torch.cuda.synchronize()
                warp_cache[(n, kd)] = lc.value * a.cfgs[0][0]
            return warp_cache[(n, kd)]

        lr = {"path": lib.path, "bits": lib.bits, "gemm": {}}
        res["libs"][lib.name] = lr
        for name in a.shapes.split(","):
            n, kd = SHAPES[name]
            log(f"{lib.name}: {name} {n}x{kd}")
            g = case(lib, torch, name, n, kd, a, warps_of, log)
            lr["gemm"][name] = g
            ok &= g.get("ok", False)
            if not g.get("ok", False):
                break
        if ok and a.mode == "bench":
            lr["points"] = frontier(lr, lib, a)
    res["ok"] = ok
    res["elapsed_s"] = time.time() - t0
    (out / "sketch-tag.json").write_text(json.dumps(res, indent=1))
    points = [p for lr in res["libs"].values() for p in lr.get("points", [])]
    (out / "codec-frontier-points.json").write_text(json.dumps(
        {"schema": "verity/sketch-tag/frontier-points/v0", "run_id": res["run_id"], "layers": LAYERS,
         "shapes": {k: list(v) for k, v in SHAPES.items()}, "points": points}, indent=1))
    if a.mode == "bench":
        write_result(ok, points, libs, out)
    log(f"done: ok={ok}, {len(points)} points")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
