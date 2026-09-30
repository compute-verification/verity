"""Decode's -h2 and -h3 hashing on GPU 1's real call at m = 32, n = k = 8,192 (sm_120, node 2), as a dependent chain, with
the hashing in series or beside the call's other kernels, against cuBLASLt's FP8 GEMM in the same run.

  python3 decode_chain.py --gpu1 GPU1_SHIP --cubin hash_sm120a.cubin --standin h1_standin.cubin --vectors h1_vectors.json \
      --out decode_chain.json --transcripts DIR [--reps R] [--calls 32]

Standard library only, beside prefill_fused.py (its helpers) and GPU 1's run.py and cubin (its branch as pushed, unedited).
At m = 32 GPU 1's call is its 64 x 64 pipeline (m padded to 64 rows, 32 of them real and committed): E_A's forming over the
64 rows, the clean-up's factors, and gemm_pearlc_d, which stores C~, U and y; the message digests are hash_msg_v's pass
over the 32 real rows, then hash_leaf_p (32-row tiles), the frame-b3s tile tree over the 128 tiles.

0. The GPU the lease named (GPU 1's identity check), then GPU 1's check (its fixture, both variants): any buffer that isn't
   bit-exact exits 4 with nothing timed.
1. Per variant, CALLS dependent calls in one CUDA graph. Call i's A is derived from call i - 1's y (the first call's from
   y_in); four weight sets (four GPU 1 pipelines) are cycled, so the 64 MB of weights never sit in L2, and call i's
   buffers are pipeline i % 4's. The arms:
   - nohash: the chain's usual producer (derive_rows, A = y ^ 0x3C00), forming, the factors and gemm_pearlc_d;
   - h3s, h2s: the coalesced producer with -h3's or -h2's segment leaves and trees fused (derive_rows_b3s_coal; -h3 adds
     the row seeds), A's tree and seed_line_a, forming, the factors, the GEMM, then the tile hashing, in series;
   - h3o: A's tree and seed on a side stream beside forming (E_A's lines from a fixed stand-in key, not seed_line_a's),
     and each call's tile hashing (hash_msg_v, leaves, tile tree) on a low-priority side stream once its GEMM is done;
   - h3g: h3o with each call's tile hashing launched with the next call's GEMM (a single-wave GEMM: 128 CTAs of 188 SMs);
   - h2o, h2g: the same tile hashing placements; -h2's forming needs A's root, so A's tree stays in series.
   Call i + 4's GEMM (the same pipeline's C~, U and tile buffers) waits for call i's tile hashing; the graph ends when the
   side streams have joined, so the last call's tile hashing counts. Graphs honour their nodes' priorities.
2. The baseline: cuBLASLt's FP8 E4M3 GEMM (TN, BF16 out, FP32 compute) at m = 32 on the same weights, CALLS calls cycling the
   four sets, autotuned over its heuristic's algorithms in the run, the fastest kept.
3. REPS rounds, each graph once a round in rotated order, a read-only L2 flush before each and the SM clock after it. Per
   call is a graph over CALLS; an arm's hashing is its per call less nohash's, its hashing-only slowdown 1 + hashing over
   cuBLASLt's per call.
4. Transcripts (verify_h1.py's format, with A's 32 rows): each arm once more, from its own y_in, after every output its
   last call's transcript reads is poisoned (0xA5); the negative control (nc) is nohash dumped as -h3's.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_fused import ATTR_PRIORITY, POISON, ROW, TILE, USE_NODE_PRIORITY, Kernels, b3s_template, load, y_rows  # noqa: E402

M_REAL, M, N, K = 32, 64, 8192, 8192
RB, WSETS = 4 * K, 4
#: (format, A's tree and seed beside forming, the tile hashing's placement: None in series, "next" after its own GEMM,
#: "tail" launched with the next call's GEMM)
ARMS = {"nohash": (None, False, None), "h3s": ("h3", False, None), "h3o": ("h3", True, "next"), "h3g": ("h3", True, "tail"),
        "h2s": ("h2", False, None), "h2o": ("h2", False, "next"), "h2g": ("h2", False, "tail")}
#: cuBLASLt's enum values: compute and scale types, matmul-descriptor and preference attributes, data types
COMPUTE_32F, R_32F, R_8F_E4M3, R_16BF = 68, 0, 28, 14
DESC_TRANSA, DESC_TRANSB, PREF_MAX_WORKSPACE, OP_N, OP_T = 3, 4, 1, 0, 1


class HeuristicResult(C.Structure):
    _fields_ = [("algo", C.c_uint64 * 8), ("workspace", C.c_size_t), ("state", C.c_int), ("waves", C.c_float),
                ("reserved", C.c_int * 4)]


class CublasLt:
    """cuBLASLt's FP8 GEMM y (m x n BF16, row-major) = A (m x k E4M3) W^T (W n x k E4M3), as column-major D (n x m) =
    op_T(W) op_N(A^T): the TN form FP8 needs."""

    def __init__(self, m: int, n: int, k: int, workspace: int, ws_bytes: int):
        for path in (os.environ.get("PEARLC_LIBCUBLASLT"), "libcublasLt.so.13", "/usr/local/cuda/lib64/libcublasLt.so.13"):
            if not path:
                continue
            try:
                self.lib = C.CDLL(path)
                self.path = path
                break
            except OSError:
                continue
        else:
            raise RuntimeError("no libcublasLt")
        vp, lib = C.c_void_p, self.lib
        for f in ("cublasLtCreate", "cublasLtMatmulDescCreate", "cublasLtMatmulDescSetAttribute", "cublasLtMatrixLayoutCreate",
                  "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceSetAttribute", "cublasLtMatmulAlgoGetHeuristic",
                  "cublasLtMatmul", "cublasLtGetVersion"):
            getattr(lib, f).restype = C.c_int if f != "cublasLtGetVersion" else C.c_size_t
        self.version = lib.cublasLtGetVersion()
        self.h, self.desc, self.pref = vp(), vp(), vp()
        self.ck(lib.cublasLtCreate(C.byref(self.h)))
        self.ck(lib.cublasLtMatmulDescCreate(C.byref(self.desc), COMPUTE_32F, R_32F))
        for attr, op in ((DESC_TRANSA, OP_T), (DESC_TRANSB, OP_N)):
            v = C.c_int(op)
            self.ck(lib.cublasLtMatmulDescSetAttribute(self.desc, attr, C.byref(v), C.c_size_t(4)))
        self.lw, self.la, self.ld = vp(), vp(), vp()
        lib.cublasLtMatrixLayoutCreate.argtypes = [C.POINTER(vp), C.c_int, C.c_uint64, C.c_uint64, C.c_int64]
        self.ck(lib.cublasLtMatrixLayoutCreate(C.byref(self.lw), R_8F_E4M3, k, n, k))
        self.ck(lib.cublasLtMatrixLayoutCreate(C.byref(self.la), R_8F_E4M3, k, m, k))
        self.ck(lib.cublasLtMatrixLayoutCreate(C.byref(self.ld), R_16BF, n, m, n))
        self.ck(lib.cublasLtMatmulPreferenceCreate(C.byref(self.pref)))
        w = C.c_uint64(ws_bytes)
        self.ck(lib.cublasLtMatmulPreferenceSetAttribute(self.pref, PREF_MAX_WORKSPACE, C.byref(w), C.c_size_t(8)))
        self.results, got = (HeuristicResult * 16)(), C.c_int()
        self.ck(lib.cublasLtMatmulAlgoGetHeuristic(self.h, self.desc, self.lw, self.la, self.ld, self.ld, self.pref, 16, self.results,
                                                   C.byref(got)))
        self.algos = [j for j in range(got.value) if self.results[j].state == 0 and self.results[j].workspace <= ws_bytes]
        self.ws, self.ws_bytes, self.algo = workspace, ws_bytes, (self.algos[0] if self.algos else None)
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        lib.cublasLtMatmul.argtypes = [vp, vp, vp, C.c_uint64, vp, C.c_uint64, vp, vp, C.c_uint64, vp, C.c_uint64, vp, vp,
                                       C.c_uint64, C.c_size_t, vp]

    @staticmethod
    def ck(r: int) -> None:
        if r:
            raise RuntimeError(f"cuBLASLt returned status {r}")

    def gemm(self, w: int, a: int, y: int, stream, algo: int | None = None) -> None:
        j = self.algo if algo is None else algo
        self.ck(self.lib.cublasLtMatmul(self.h, self.desc, C.addressof(self.alpha), w, self.lw, a, self.la, C.addressof(self.beta), y,
                                        self.ld, y, self.ld, C.addressof(self.results[j]), self.ws, self.ws_bytes, stream))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in ("--gpu1", "--cubin", "--standin", "--vectors", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--calls", type=int, default=32)
    args = ap.parse_args()
    calls = args.calls
    R = load(os.path.join(args.gpu1, "run.py"), "gpu1_run")
    P, u64, vp = C.POINTER, C.c_uint64, C.c_void_p
    image = open(os.path.join(args.gpu1, "pearl_c_sm120.cubin"), "rb").read()
    dev = R.Dev(image)
    expect = os.environ.get("GPU_LEASE_UUID")
    any_gpu, ungated = bool(os.environ.get("PEARLC_ANY_GPU")), bool(os.environ.get("PEARLC_UNGATED"))
    refused = [] if any_gpu else dev.identity(expect)
    if ungated and not any_gpu:
        refused.append("PEARLC_UNGATED is for dry runs on a stand-in driver (with PEARLC_ANY_GPU)")
    run = os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local"
    res: dict = {"run": run, "device": dev.name, "uuid": dev.uuid, "lease_uuid": expect, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "sms": dev.sms, "gpu1_cubin_sha256": hashlib.sha256(image).hexdigest(),
                 "gpu1_source": open(os.path.join(args.gpu1, "source.txt")).read().split()[0],
                 "cubin_sha256": hashlib.sha256(open(args.cubin, "rb").read()).hexdigest(),
                 "standin_sha256": hashlib.sha256(open(args.standin, "rb").read()).hexdigest(), "refused": refused, "ungated": ungated,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shape": [M_REAL, N, K], "calls": calls}
    save = lambda: open(args.out, "w").write(json.dumps(res, indent=1))  # noqa: E731
    if refused:
        save()
        sys.exit(3)
    dev.open()
    R.bind(dev.cu, {"cuStreamBeginCapture_v2": [vp, C.c_int], "cuStreamEndCapture": [vp, P(vp)],
                    "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp],
                    "cuStreamWaitEvent": [vp, vp, C.c_uint], "cuCtxGetStreamPriorityRange": [P(C.c_int), P(C.c_int)],
                    "cuStreamCreateWithPriority": [P(vp), C.c_uint, C.c_int], "cuGraphGetNodes": [vp, vp, P(C.c_size_t)],
                    "cuGraphNodeGetType": [vp, P(C.c_int)], "cuGraphKernelNodeGetAttribute": [vp, C.c_int, vp]})
    nv = R.Nvml(dev.uuid)
    res["clocks_idle"] = nv.read()
    man_path = os.path.join(args.gpu1, "check", "check.json")
    chk = R.check(dev, json.load(open(man_path)), os.path.dirname(man_path), list(R.VARIANT_G))
    res["check"] = {v: {"buffers": r["buffers"], "failed": r["failed"]} for v, r in chk.items()}
    save()
    print(f"GPU 1's check: {res['check']}", flush=True)
    if not all(r["pearl_c_ok"] and r["plain_ok"] for r in chk.values()) and not ungated:
        sys.exit(4)

    Kn = Kernels(R, dev, [open(args.cubin, "rb").read(), open(args.standin, "rb").read()])
    V = json.load(open(args.vectors))
    pa, pt = V["decode"]["a3"], V["decode"]["t2"]
    seed = pa["seed"]
    salt, root_b, index = bytes.fromhex(seed["salt"]), bytes.fromhex(seed["root_b"]), seed["index"]
    seed_key = bytes.fromhex(V["h3"]["seed_key"])
    keys = {"line_b": dev.put(hashlib.shake_256(b"pearl-c-timing/line_b").digest(32)), "dom_a": dev.put(bytes.fromhex(pa["domain"])),
            "dom_b": dev.put(bytes(32)), "dom_t": dev.put(bytes.fromhex(pt["domain"])),
            "k_msg": dev.put(bytes.fromhex(V["h1_keys"]["message"])), "k_leaf": dev.put(bytes.fromhex(V["h1_keys"]["leaf"])),
            "unit": dev.put(salt + root_b + index.to_bytes(8, "little") + seed_key)}
    rt, tt = b3s_template(ROW), b3s_template(TILE)
    rtmpl, ttmpl = dev.put(rt[0]), dev.put(tt[0])
    segkeys, seedk, saltp, rootbp = dev.alloc(32 * 16), dev.put(seed_key), dev.put(salt), dev.put(root_b)
    seeds, line_h3 = dev.alloc(32 * M_REAL), dev.put(hashlib.shake_256(b"decode -h3 E_A line key stand-in").digest(32))
    Kn.go("b3s_level_keys", 1, 32, (segkeys, keys["dom_a"]), (9,))
    yb = 2 * M * N                                           # y: 64 rows of n BF16
    y_in, xa, xb = dev.alloc(yb), dev.alloc(M * RB), dev.alloc(N * RB)
    fl, sink = dev.alloc(512 << 20), dev.alloc(16)
    ws_bytes = 32 << 20
    ws = dev.alloc(ws_bytes)
    dev.launch("gen_f32", -(-N * K // 256), 256, p=(xb,), i=(N, K, 12))
    put_y = lambda tag: dev.ck("cuMemcpyHtoD_v2", u64(y_in), C.c_char_p(y_rows(tag, M, 2 * N)), yb)  # noqa: E731
    put_y("decode y timing")
    dev.sync()
    tiles = N // 64

    pipes = {v: [R.Pipeline(dev, M, N, K, v, keys, xa, xb, m_real=M_REAL) for _ in range(WSETS)] for v in R.VARIANT_G}
    for v, ps in pipes.items():
        for pipe in ps:
            steps = {nm: (ph, f) for nm, ph, f in pipe.steps()}
            for nm, (ph, f) in steps.items():                # the unit's level keys and the weight side
                if ph in ("unit", "job"):
                    f()
            b = pipe.b
            dev.launch("plainq", N, 256, p=(xb, b["qb"], b["sb"]), i=(N, K))
            Kn.go("derive_rows", -(-M_REAL * RB // 2 // 256), 256, (y_in, xa), (M_REAL * RB // 2,))
            dev.launch("plainq", M_REAL, 256, p=(xa, b["qa"], b["sa"]), i=(M_REAL, K))
            for nm in ("lines_ea", "stats_a", "eprime_a", "form_a", "af", "pa", "seed_a"):   # GPU 1's shared-memory probes too
                steps[nm][1]()
            ex = pipe.extra()
            ex["gemm_pearlc_cu"]()
            ex["hash_msg"]()
            steps["hash_leaf"][1]()
            dev.sync()
    res["pipelines"] = {"weight_sets": WSETS, "m": M, "m_real": M_REAL, "bm": pipes["sm120"][0].bm, "fused": pipes["sm120"][0].fused}

    least, greatest = C.c_int(), C.c_int()
    dev.ck("cuCtxGetStreamPriorityRange", C.byref(least), C.byref(greatest))

    def stream(priority: int) -> vp:
        x = vp()
        dev.ck("cuStreamCreateWithPriority", C.byref(x), 1, priority)
        return x

    s, side_a, side_lo = stream(greatest.value), stream(greatest.value), stream(least.value)
    evs = {k: vp() for k in ("p", "g", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3")}
    for e in evs.values():
        dev.ck("cuEventCreate", C.byref(e), 2)

    def on(stream_, fns) -> None:
        old = dev.stream
        dev.stream = stream_
        for f in fns:
            f()
        dev.stream = old

    def levels(base: int, lk: int, count: int, threads: int) -> None:
        width, level = 1 << max(count - 1, 0).bit_length(), 0
        if width > 1:
            dev.launch("hash_tree_small", 1, min(threads, width // 2), p=(base, lk + 32 * level), i=(width,))

    def node_priorities(g) -> dict:
        n = C.c_size_t()
        dev.ck("cuGraphGetNodes", g, None, C.byref(n))
        nodes = (vp * n.value)()
        dev.ck("cuGraphGetNodes", g, nodes, C.byref(n))
        seen: dict = {}
        for node in nodes:
            kind, val = C.c_int(), C.create_string_buffer(64)
            dev.ck("cuGraphNodeGetType", node, C.byref(kind))
            if kind.value == 0:
                dev.ck("cuGraphKernelNodeGetAttribute", node, ATTR_PRIORITY, val)
                p_ = str(int.from_bytes(val.raw[:4], "little", signed=True))
                seen[p_] = seen.get(p_, 0) + 1
        return seen

    def capture(body, prio: bool = True) -> tuple:
        dev.stream = s
        dev.ck("cuStreamBeginCapture_v2", s, 0)
        body()
        g, ge = vp(), vp()
        dev.ck("cuStreamEndCapture", s, C.byref(g))
        dev.stream = None
        pr = node_priorities(g)
        dev.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, USE_NODE_PRIORITY if prio else 0)
        return ge, pr

    def chain(v: str, arm: str) -> None:
        fmt, a_side, t_place = ARMS[arm]
        ps = pipes[v]
        a_rec, t_rec, pending = [False] * WSETS, [False] * WSETS, None

        def side_hashing(fns, w) -> None:
            dev.ck("cuEventRecord", evs["g"], s)
            dev.ck("cuStreamWaitEvent", side_lo, evs["g"], 0)
            on(side_lo, fns)
            dev.ck("cuEventRecord", evs[f"t{w}"], side_lo)
            t_rec[w] = True

        for i in range(calls):
            w = i % WSETS
            pipe = ps[w]
            b = pipe.b
            st = {nm: f for nm, _ph, f in pipe.steps()}
            ex = pipe.extra()
            y_prev = y_in if i == 0 else ps[(i - 1) % WSETS].b["y"]
            if a_rec[w]:
                dev.ck("cuStreamWaitEvent", s, evs[f"a{w}"], 0)
            if fmt is None:
                Kn.go("derive_rows", -(-M_REAL * RB // 2 // 256), 256, (y_prev, xa), (M_REAL * RB // 2,))
            else:
                Kn.go("derive_rows_b3s_coal", M_REAL, 1024, (y_prev, xa, b["tree_a"], keys["dom_a"], rtmpl, segkeys, seedk, saltp, rootbp,
                                                             seeds), (M_REAL, RB, N, rt[1], 2 if fmt == "h3" else 1, index))
                root_a = b["tree_a"] + pipe.size["tree_a"] - 32
                commit = [lambda b=b: levels(b["tree_a"], b["keys_a"], M_REAL, 1024),
                          lambda b=b, root_a=root_a: dev.launch("seed_line_a", 1, 32, p=(root_a, keys["unit"], b["line_a"], b["seed_a"]))]
                if a_side:
                    dev.ck("cuEventRecord", evs["p"], s)
                    dev.ck("cuStreamWaitEvent", side_a, evs["p"], 0)
                    on(side_a, commit)
                    dev.ck("cuEventRecord", evs[f"a{w}"], side_a)
                    a_rec[w] = True
                else:
                    on(s, commit)
            lines = (lambda b=b: dev.launch("lines", -(-M // 128), 128, p=(line_h3, b["ea_codes"], 0, 0), i=(0, 0, M, R.NORM_256))) \
                if a_side else st["lines_ea"]
            on(s, [lines, st["stats_a"], st["eprime_a"], st["form_a"], st["af"], st["pa"]])
            if t_rec[w]:
                dev.ck("cuStreamWaitEvent", s, evs[f"t{w}"], 0)
            if t_place == "tail" and pending:
                side_hashing(*pending)
            ex["gemm_pearlc_cu"]()
            if fmt is None:
                continue
            hashing = [ex["hash_msg"], st["hash_leaf"],
                       lambda b=b: Kn.go("hash_leaves_b3s", -(-tiles // 128), 128, (b["leaves"], b["tree_t"], keys["dom_t"], ttmpl),
                                         (tiles, 32, 0, tt[1])),
                       lambda b=b: levels(b["tree_t"], b["keys_t"], tiles, 1024)]
            if t_place == "next":
                side_hashing(hashing, w)
            elif t_place == "tail":
                pending = (hashing, w)
            else:
                on(s, hashing)
        if pending:
            side_hashing(*pending)
        for w in range(WSETS):
            if a_rec[w]:
                dev.ck("cuStreamWaitEvent", s, evs[f"a{w}"], 0)
            if t_rec[w]:
                dev.ck("cuStreamWaitEvent", s, evs[f"t{w}"], 0)

    graphs, prios = {}, {}
    for v in R.VARIANT_G:
        for arm in ARMS:
            graphs[f"{v} {arm}"], prios[f"{v} {arm}"] = capture(lambda v=v, arm=arm: chain(v, arm))
    res["node_priorities"], res["stream_priorities"] = prios, {"least": least.value, "greatest": greatest.value}

    # cuBLASLt at m = 32 on the same four weight sets (GPU 1's plainq codes), autotuned: each heuristic algorithm's CALLS-call
    # graph timed, the fastest kept
    base = pipes["sm120"]
    try:
        lt = CublasLt(M_REAL, N, K, ws, ws_bytes)
    except (RuntimeError, OSError) as e:
        if not ungated:
            raise
        print(f"dry run without cuBLASLt ({e}): the baseline is a stand-in", flush=True)
        lt = None
    if lt:
        res["cublaslt"] = {"library": lt.path, "version": lt.version, "algos": len(lt.algos)}
    else:
        res["cublaslt"] = {"stand_in": True, "algos": 0}
    ev = [vp(), vp()]
    for e in ev:
        dev.ck("cuEventCreate", C.byref(e), 0)

    def replay(ge, flush: bool = True) -> float:
        if flush:
            Kn.go("l2_flush", 4 * dev.sms, 256, (fl, sink), ((512 << 20) // 4,), stream=s)
        ms = C.c_float()
        dev.ck("cuEventRecord", ev[0], s)
        dev.ck("cuGraphLaunch", ge, s)
        dev.ck("cuEventRecord", ev[1], s)
        dev.ck("cuEventSynchronize", ev[1])
        dev.ck("cuEventElapsedTime", C.byref(ms), ev[0], ev[1])
        return ms.value

    def lt_chain(j: int) -> None:
        for i in range(calls):
            b = base[i % WSETS].b
            lt.gemm(b["qb"], b["qa"], b["y_plain"], s, j)

    tune = {}
    if lt is None:                                           # a dry run's stand-in: GPU 1's GEMM chain
        tune[0] = {"ms": 1.0, "graph": capture(lambda: [base[i % WSETS].extra()["gemm_pearlc_cu"]() for i in range(calls)])[0]}
    for j in (lt.algos if lt else []):
        try:
            ge, _ = capture(lambda j=j: lt_chain(j), prio=False)
        except RuntimeError as e:
            tune[j] = {"error": str(e)}
            dev.stream = None
            continue
        replay(ge)
        tune[j] = {"ms": statistics.median(replay(ge) for _ in range(5)), "graph": ge}
    ok = {j: t for j, t in tune.items() if "ms" in t}
    if not ok:
        res["cublaslt"]["tune"] = {str(j): t for j, t in tune.items()}
        save()
        sys.exit(6)
    best = min(ok, key=lambda j: ok[j]["ms"])
    res["cublaslt"]["tune"] = {str(j): {"per_call_ms": round(t["ms"] / calls, 5)} if "ms" in t else t for j, t in tune.items()}
    res["cublaslt"]["best"] = best
    graphs["plain cublaslt"] = ok[best]["graph"]
    save()

    for ge in graphs.values():
        replay(ge)
    raw = {nm: [] for nm in graphs}
    clk = {nm: [] for nm in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for nm in order[rep % len(order):] + order[:rep % len(order)]:
            raw[nm].append(replay(graphs[nm]))
            clk[nm].append(nv.read().get("sm_mhz"))
    med = {nm: statistics.median(x) / calls for nm, x in raw.items()}
    res["timing"] = {nm: {"per_call_ms": round(med[nm], 6), "raw_ms": [round(x, 5) for x in raw[nm]], "sm_mhz": clk[nm],
                          "sm_mhz_median": statistics.median(c) if (c := [x for x in clk[nm] if x is not None]) else None}
                     for nm in graphs}
    plain = med["plain cublaslt"]
    res["baseline"] = {"graph": "plain cublaslt", "per_call_ms": round(plain, 6)}
    res["rows"] = {}
    for v in R.VARIANT_G:
        for arm in list(ARMS)[1:]:
            h = med[f"{v} {arm}"] - med[f"{v} nohash"]
            res["rows"][f"{v} {arm}"] = {"hashing_ms": round(h, 6), "slowdown_hashing_only": round(1 + h / plain, 4),
                                         "call_ms": round(med[f"{v} {arm}"], 6), "nohash_ms": round(med[f"{v} nohash"], 6)}
            print(f"{v} {arm}: hashing {h * 1e3:.1f} us a call, {1 + h / plain:.4f}x over cuBLASLt {plain * 1e3:.1f} us", flush=True)
    save()
    if not args.transcripts:
        return

    os.makedirs(args.transcripts, exist_ok=True)
    res["transcripts"] = {}
    for v, ps in pipes.items():
        last = ps[(calls - 1) % WSETS]
        b, sz = last.b, last.size
        poison = [(xa, M_REAL * RB), (seeds, 32 * M_REAL)] + [(p.b[nm], p.size[nm]) for p in ps for nm in
                                                              ("tree_a", "seed_a", "line_a", "c", "u", "y", "digests", "leaves", "tree_t")]
        for arm in [*list(ARMS)[1:], "nc"]:
            fmt = "h3" if arm == "nc" else ARMS[arm][0]
            put_y(f"decode y {v} {arm}")
            for p_, n_ in poison:
                dev.ck("cuMemsetD8_v2", u64(p_), POISON, n_)
            dev.sync()
            replay(graphs[f"{v} {'nohash' if arm == 'nc' else arm}"], flush=False)
            dev.sync()
            parts = [("a_rows", dev.read(xa, M_REAL * RB)), ("a_tree", dev.read(b["tree_a"], sz["tree_a"]))]
            if fmt == "h3":
                parts.append(("row_seeds", dev.read(seeds, 32 * M_REAL)))
            parts += [("unit_seed", dev.read(b["seed_a"], 32)), ("c", dev.read(b["c"], 4 * M_REAL * N)), ("u", dev.read(b["u"], 4 * M_REAL * N)),
                      ("digests", dev.read(b["digests"], sz["digests"])), ("leaves", dev.read(b["leaves"], 32 * tiles)),
                      ("tile_tree", dev.read(b["tree_t"], sz["tree_t"]))]
            head = {"run": run, "arm": f"decode {v} {arm}, call {calls - 1} of a {calls}-call dependent chain", "variant": v,
                    "m": M_REAL, "rb": RB, "n": N, "hashing": fmt, "a": pa, "t": pt, "sections": [[k, len(x)] for k, x in parts]}
            blob = json.dumps(head).encode() + b"\n" + b"".join(x for _k, x in parts)
            path = os.path.join(args.transcripts, f"decode-{'v1' if v == 'sm120' else 'v2'}-{arm}.bin")
            open(path, "wb").write(blob)
            res["transcripts"][f"{v} {arm}"] = {"path": path, "sha256": hashlib.sha256(blob).hexdigest()}
            print(f"transcript {v} {arm}: {path} sha256:{res['transcripts'][f'{v} {arm}']['sha256']}", flush=True)
    res["clocks_after"] = nv.read()
    save()


if __name__ == "__main__":
    main()
