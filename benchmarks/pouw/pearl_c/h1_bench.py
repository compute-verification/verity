"""-h1's hashing on sm_120 (node 2): device gates, cost per byte, and decode's dependent chain at m = 32, n = k = 8,192.

  python3 h1_bench.py --cubin hash_sm120a.cubin --standin h1_standin.cubin --vectors h1_vectors.json --out h1_bench.json

Standard library only; the driver API through `run.py`'s `Dev`. The GPU is the one the lease named (`GPU_LEASE_UUID`, checked
against `cuDeviceGetUuid`); NVML reads that UUID's clocks.

**Gates, all before any timing** (a difference exits 4 with nothing timed). Each kernel form against `h1_vectors.json`, which
`h1_vectors.py` writes from `verity.commitments`' frame-b3 and Pearl-C's h1 digests:
- the row leaves of `hash_rows` (as pushed) and `hash_rows_w` (tuned), at every chunk count 1 … 257: rows of 1,024c − 37
  bytes (exactly c chunks) and 1,024c bytes (37 bytes past a chunk boundary), rows 0 and 1;
- A's tree at m = 32 (FP32 rows at k = 8,192, as `-h1` commits them, and BF16), `hash_nodes` level by level and
  `hash_tree_small` in one launch;
- the tile hashing at m = 32, n = 8,192: as pushed, over the tile padded to 64 rows (`hash_msg`, `hash_leaf`: it has no
  partial tile), and tuned over the 32 rows (`hash_msg_v`, `hash_leaf_p`), with the tile tree both ways.

**Cost per byte**: each form at prefill size (m = 8,192), a read-only L2 flush before each rep, bytes hashed over the median
time, per SM per SM clock too.

**Decode, dependent chain**: per call, A = f(y of the call before) (`derive_rows`), A's commitment (row leaves and tree), the
stand-in GEMM (`h1_standin.cu`: the weights' DRAM traffic, not a GEMM; four weight sets cycled, past twice the 128 MB L2), and
the post-GEMM hashing (message digests, tile leaves, tile tree). CALLS calls are captured into one CUDA graph and replayed;
per call is a replay over CALLS. The arms: `plain` (derive and the stand-in alone), `h1` (the kernels as pushed), `h1-tuned`
(serial), `h1-tuned-side` (the post-GEMM hashing on a second stream, two buffer slots, joined before a slot is reused).
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
import run as R  # noqa: E402

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
EXTRA = {
    "cuStreamCreate": [P(vp), u], "cuStreamWaitEvent": [vp, vp, u], "cuStreamSynchronize": [vp],
    "cuStreamBeginCapture_v2": [vp, i32], "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong],
    "cuGraphLaunch": [vp, vp], "cuGraphExecDestroy": [vp], "cuGraphDestroy": [vp], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
}
M, N, K = 32, 8192, 8192
CALLS, WSETS = 32, 4
MAX_DYN, SMEM_OPTIN = 8, 97
ROW_F32, ROW_BF16 = 4 * K, 2 * K


def shake(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


class Sensors(R.Nvml):
    """run.py's NVML reader on the leased GPU's handle (by UUID; NVML's index 0 is another GPU on a shared node)."""

    def __init__(self, uuid: str | None):
        super().__init__()
        if self.lib and uuid:
            self.lib.nvmlDeviceGetHandleByUUID.argtypes = [C.c_char_p, P(vp)]
            h = vp()
            if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(h)) == 0:
                self.h = h
            else:
                self.lib = None


class Bench(R.Dev):
    def __init__(self, cubin: bytes, standin: bytes):
        super().__init__(cubin)
        R.bind(self.cu, EXTRA)
        self.mod2 = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod2), standin)
        self.fns2: dict[str, vp] = {}
        self.s1, self.s2 = vp(), vp()
        self.ck("cuStreamCreate", C.byref(self.s1), 1)
        self.ck("cuStreamCreate", C.byref(self.s2), 1)
        self.args: list = []
        self.dyn: dict[str, int] = {}

    def uuid(self) -> str:
        b = C.create_string_buffer(16)
        self.ck("cuDeviceGetUuid_v2", b, self.dev)
        h = b.raw.hex()
        return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    def fn_any(self, name: str) -> vp:
        try:
            return self.fn(name)
        except RuntimeError:
            if name not in self.fns2:
                f = vp()
                self.ck("cuModuleGetFunction", C.byref(f), self.mod2, name.encode())
                self.fns2[name] = f
            return self.fns2[name]

    def go(self, stream, name: str, grid: int, block: int, p=(), i=(), smem: int = 0) -> None:
        a = R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        self.args.append(a)                                  # alive until the graph that captured it is gone
        params = (vp * 1)(C.addressof(a))
        self.args.append(params)
        f = self.fn_any(name)
        if smem > 48 * 1024 and self.dyn.get(name, 0) < smem:  # set once, before any capture reaches it
            self.ck("cuFuncSetAttribute", f, MAX_DYN, smem)
            self.dyn[name] = smem
        self.ck("cuLaunchKernel", f, max(1, grid), 1, 1, block, 1, 1, smem, stream, params, None)

    def event(self, stream) -> vp:
        e = vp()
        self.ck("cuEventCreate", C.byref(e), 2)              # CU_EVENT_DISABLE_TIMING
        self.ck("cuEventRecord", e, stream)
        return e

    def wait(self, stream, e) -> None:
        self.ck("cuStreamWaitEvent", stream, e, 0)

    def graph(self, body) -> vp:
        self.ck("cuStreamBeginCapture_v2", self.s1, 0)       # CU_STREAM_CAPTURE_MODE_GLOBAL
        body()
        g, ge = vp(), vp()
        self.ck("cuStreamEndCapture", self.s1, C.byref(g))
        self.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

    def replay_ms(self, ge, reps: int) -> list[float]:
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, self.s1)
            self.ck("cuGraphLaunch", ge, self.s1)
            self.ck("cuEventRecord", b, self.s1)
            self.ck("cuEventSynchronize", b)
            self.ck("cuEventElapsedTime", C.byref(ms), a, b)
            out.append(ms.value)
        return out


# ---- the kernels' launches, both forms --------------------------------------------------------------------------------------


def rows_w_words(rb: int) -> int:
    return 8 * ((37 + rb + 1023) // 1024)


def launch_rows(d: Bench, s, form: str, rows_p: int, out_p: int, dom_p: int, m: int, rb: int) -> None:
    if form == "pushed":
        d.go(s, "hash_rows", m, 256, (rows_p, out_p, dom_p), (m, rb))
    else:
        warps = 4
        d.go(s, "hash_rows_w", -(-m // warps), 32 * warps, (rows_p, out_p, dom_p), (m, rb), smem=4 * warps * rows_w_words(rb))


def launch_tree(d: Bench, s, form: str, tree_p: int, dom_p: int, width: int) -> None:
    if form == "pushed" or width > 2048:
        off, w, lv = 0, width, 0
        while w > 1:
            d.go(s, "hash_nodes", -(-(w // 2) // 128), 128, (tree_p + off, tree_p + off + 32 * w, dom_p), (lv, w // 2))
            off, w, lv = off + 32 * w, w // 2, lv + 1
    else:
        d.go(s, "hash_tree_small", 1, max(1, min(1024, width // 2)), (tree_p, dom_p), (width,))


def launch_tiles(d: Bench, s, form: str, c_p: int, u_p: int, dg_p: int, lv_p: int, tree_p: int, keys: dict, dom_t: int,
                 rows: int, n: int) -> int:
    """Message digests, tile leaves, the tile tree's leaves and its levels; returns the tile count."""
    per, tn = n // 32, n // 64
    if form == "pushed":
        tiles = rows // 64 * tn
        d.go(s, "hash_msg", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf", -(-tiles // 64), 64, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
    else:
        tiles = -(-rows // 64) * tn
        d.go(s, "hash_msg_v", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf_p", -(-tiles // 32), 128, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
    d.go(s, "hash_tree_leaves", -(-tiles // 128), 128, (lv_p, tree_p, dom_t), (tiles,))
    launch_tree(d, s, form, tree_p, dom_t, tiles)
    return tiles


# ---- gates -------------------------------------------------------------------------------------------------------------------


def gates(d: Bench, V: dict) -> dict:
    out: dict = {"rows": {}, "trees": {}, "tiles": {}}
    bad: list[str] = []
    dom = d.put(bytes.fromhex(V["rows"]["domain"]))
    for form in ("pushed", "tuned"):
        wrong = 0
        for case in V["rows"]["cases"]:
            rb = case["rb"]
            rows = shake(f"gate rows {rb}", 2 * rb)
            rp, op = d.put(rows), d.alloc(64)
            launch_rows(d, None, form, rp, op, dom, 2, rb)
            d.sync()
            got = d.read(op, 64).hex()
            if got != case["leaves"][0] + case["leaves"][1]:
                wrong += 1
                bad.append(f"{form} rows rb={rb}")
            d.free(rp)
            d.free(op)
        out["rows"][form] = {"cases": len(V["rows"]["cases"]), "wrong": wrong}
    for label, t in V["trees"].items():
        rows = shake(t["tag"], t["m"] * t["rb"])
        dp = d.put(bytes.fromhex(t["domain"]))
        for form in ("pushed", "tuned"):
            rp, tp = d.put(rows), d.alloc(32 * 2 * t["m"])
            launch_rows(d, None, form, rp, tp, dp, t["m"], t["rb"])
            launch_tree(d, None, form, tp, dp, t["m"])
            d.sync()
            ok = d.read(tp + 32 * (2 * t["m"] - 2), 32).hex() == t["root"]
            out["trees"][f"{label} {form}"] = ok
            if not ok:
                bad.append(f"{label} {form} root")
            d.free(rp)
            d.free(tp)
    keys = {k: d.put(bytes.fromhex(R.H1_KEYS[k])) for k in ("k_msg", "k_leaf")}
    for form, case in (("pushed", V["tiles"]["padded64"]), ("tuned", V["tiles"]["m32"])):
        rows, n = case["rows"], case["n"]
        c = shake("gate c", 4 * M * n) + bytes(4 * (rows - M) * n)
        uu = shake("gate u", 4 * M * n) + bytes(4 * (rows - M) * n)
        cp, up = d.put(c), d.put(uu)
        dg, lv = d.alloc(32 * rows * (n // 32)), d.alloc(32 * 256)
        tp, dt = d.alloc(32 * 2 * 256), d.put(bytes.fromhex(case["domain"]))
        tiles = launch_tiles(d, None, form, cp, up, dg, lv, tp, keys, dt, rows, n)
        d.sync()
        checks = {"digests": hashlib.sha256(d.read(dg, 32 * rows * (n // 32))).hexdigest() == case["digests_sha256"],
                  "leaves": hashlib.sha256(d.read(lv, 32 * tiles)).hexdigest() == case["leaves_sha256"],
                  "root": d.read(tp + 32 * (2 * tiles - 2), 32).hex() == case["root"]}
        out["tiles"][form] = {"rows": rows, "tiles": tiles, **checks}
        bad += [f"tiles {form} {k}" for k, v in checks.items() if not v]
        for p in (cp, up, dg, lv, tp, dt):
            d.free(p)
    out["failures"] = bad
    return out


# ---- timing ------------------------------------------------------------------------------------------------------------------


def flush(d: Bench, fl: int, sink: int) -> None:
    """A read of 512 MB (4x the L2) that writes nothing: no dirty line left behind."""
    d.go(None, "l2_flush", 4 * d.sms, 256, (fl, sink), ((512 << 20) // 4,))


def rates(d: Bench, nv: R.Nvml, reps: int) -> dict:
    """Bytes hashed per second at prefill size, each form, after an L2 flush (a 256 MB device copy) before each rep."""
    fl, sink = d.alloc(512 << 20), d.alloc(16)
    keys = {k: d.put(bytes.fromhex(R.H1_KEYS[k])) for k in ("k_msg", "k_leaf")}
    dom = d.put(bytes(32))
    m, n = 8192, 8192
    out = {}

    def timed(name: str, nbytes: int, body) -> None:
        body()
        d.sync()
        ms = []
        for _ in range(reps):
            flush(d, fl, sink)
            ms += d.timed(body, 1)
        med = statistics.median(ms)
        clk = nv.read()
        out[name] = {"bytes": nbytes, "ms": med, "raw_ms": ms, "gb_s": nbytes / med / 1e6,
                     "bytes_per_sm_clk": nbytes / (med * 1e-3) / (d.sms * max(clk["sm_mhz"], 1) * 1e6), "clocks": clk}

    for rb in (ROW_F32, ROW_BF16):
        rp, op = d.alloc(m * rb), d.alloc(32 * m)
        for form in ("pushed", "tuned"):
            timed(f"rows {rb} B {form}", m * rb, lambda form=form: launch_rows(d, None, form, rp, op, dom, m, rb))
        d.free(rp)
        d.free(op)
    cp, up, dg, lv = d.alloc(4 * m * n), d.alloc(4 * m * n), d.alloc(32 * m * (n // 32)), d.alloc(32 * (m // 64) * (n // 64))
    for form, kmsg, kleaf, bl in (("pushed", "hash_msg", "hash_leaf", 64), ("tuned", "hash_msg_v", "hash_leaf_p", 128)):
        timed(f"messages {form}", 8 * m * n,
              lambda kmsg=kmsg: d.go(None, kmsg, m * (n // 32) // 128, 128, (cp, up, dg, keys["k_msg"]), (m, n)))
        tiles = (m // 64) * (n // 64)
        grid = tiles // 64 if form == "pushed" else tiles // 32
        timed(f"tile leaves {form}", 32 * m * (n // 32),
              lambda kleaf=kleaf, bl=bl, grid=grid: d.go(None, kleaf, grid, bl, (dg, lv, keys["k_leaf"]), (m, n)))
    for p in (cp, up, dg, lv, fl):
        d.free(p)
    return out


def decode(d: Bench, reps: int) -> dict:
    """Per call of the dependent chain at m = 32, n = k = 8,192, each arm replayed as one graph of CALLS calls."""
    n, k, m = N, K, M
    keys = {kk: d.put(bytes.fromhex(R.H1_KEYS[kk])) for kk in ("k_msg", "k_leaf")}
    dom_a, dom_t = d.put(shake("dom a", 32)), d.put(shake("dom t", 32))
    w = [d.put(shake(f"weights {s}", n * k)) for s in range(WSETS)]
    y = d.alloc(2 * m * n)
    rows = {rb: d.alloc(m * rb) for rb in (ROW_F32, ROW_BF16)}
    a_tree = d.alloc(32 * 2 * m)
    slots = [{"c": d.alloc(4 * 64 * n), "u": d.alloc(4 * 64 * n), "dg": d.alloc(32 * 64 * (n // 32)), "lv": d.alloc(32 * 256),
              "tree": d.alloc(32 * 2 * 256)} for _ in range(2)]
    out: dict = {"calls_per_graph": CALLS, "weight_sets": WSETS, "shape": f"m{m}-n{n}-k{k}"}

    def call(i: int, arm: str, rb: int, stage: str, joins: list) -> None:
        s1, s2 = d.s1, d.s2
        sl = slots[i % 2]
        tiled = 64 if arm == "h1" else m
        d.go(s1, "derive_rows", -(-m * k // 256), 256, (y, rows[ROW_BF16]), (m * k,))
        if rb == ROW_F32:
            d.go(s1, "derive_rows", -(-m * k // 256), 256, (y, rows[ROW_F32]), (m * k,))
        if arm != "plain" and stage in ("all", "before"):
            form = "pushed" if arm == "h1" else "tuned"
            launch_rows(d, s1, form, rows[rb], a_tree, dom_a, m, rb)
            launch_tree(d, s1, form, a_tree, dom_a, m)
        if len(joins) >= 2:                                   # the side stream's slot from two calls back is free again
            d.wait(s1, joins[-2])
        d.go(s1, "standin_gemm", n * 32 // 256, 256, (w[i % WSETS], rows[ROW_BF16], sl["c"], sl["u"], y), (m, n, k))
        if arm == "plain" or stage == "before":
            return
        form = "pushed" if arm == "h1" else "tuned"
        if arm == "h1-tuned-side":
            e = d.event(s1)
            d.wait(s2, e)
            launch_tiles(d, s2, form, sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], keys, dom_t, tiled, n)
            joins.append(d.event(s2))
        else:
            launch_tiles(d, s1, form, sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], keys, dom_t, tiled, n)

    arms = [("plain", ROW_F32, "all"), ("plain", ROW_BF16, "all"), ("h1", ROW_F32, "all"), ("h1-tuned", ROW_F32, "all"), ("h1-tuned-side", ROW_F32, "all"),
            ("h1", ROW_F32, "before"), ("h1-tuned", ROW_F32, "before"), ("h1-tuned", ROW_BF16, "all"),
            ("h1-tuned-side", ROW_BF16, "all")]
    graphs = {}
    for arm, rb, stage in arms:
        def body(arm=arm, rb=rb, stage=stage):
            joins: list = []
            for i in range(CALLS):
                call(i, arm, rb, stage, joins)
            if joins:
                d.wait(d.s1, joins[-1])
        graphs[(arm, rb, stage)] = d.graph(body)
    for key, ge in graphs.items():                           # warm every graph once
        d.replay_ms(ge, 1)
    raw = {key: [] for key in graphs}
    for rep in range(reps):                                  # the arms interleaved, rotated each rep
        order = list(graphs)
        order = order[rep % len(order):] + order[:rep % len(order)]
        for key in order:
            raw[key] += d.replay_ms(graphs[key], 1)
    for (arm, rb, stage), ms in raw.items():
        name = f"{arm} {'fp32' if rb == ROW_F32 else 'bf16'} rows" + ("" if stage == "all" else f" ({stage} the GEMM only)")
        out[name] = {"per_call_us": statistics.median(ms) * 1e3 / CALLS, "raw_ms": ms}
    for name, r in out.items():
        if isinstance(r, dict) and "per_call_us" in r:
            plain = out[f"plain {'fp32' if 'fp32' in name else 'bf16'} rows"]["per_call_us"]
            r["hashing_us"] = r["per_call_us"] - plain
            r["x_plain_standin"] = r["per_call_us"] / plain
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--standin", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    cubin, standin = open(args.cubin, "rb").read(), open(args.standin, "rb").read()
    d = Bench(cubin, standin)
    want = os.environ.get("GPU_LEASE_UUID")
    nv = Sensors(want or d.uuid())
    res = {"device": d.name, "uuid": d.uuid(), "lease_uuid": want, "driver": d.driver, "sms": d.sms, "nvml_driver": nv.driver,
           "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "standin_sha256": hashlib.sha256(standin).hexdigest(),
           "clock_label": "locked-2100", "clocks_idle": nv.read(), "time": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    save = lambda: open(args.out, "w").write(json.dumps(res, indent=1))  # noqa: E731
    if want and want != res["uuid"]:
        res["error"] = "the visible device is not the leased one"
        save()
        sys.exit(3)
    V = json.load(open(args.vectors))
    res["gates"] = gates(d, V)
    save()
    print(json.dumps({"gates": res["gates"]["failures"], "rows": res["gates"]["rows"]}), flush=True)
    if res["gates"]["failures"]:
        sys.exit(4)
    res["rates"] = rates(d, nv, args.reps)
    save()
    res["decode"] = decode(d, args.reps)
    res["clocks_after"] = nv.read()
    save()
    for name, r in res["decode"].items():
        if isinstance(r, dict):
            print(f"decode {name}: {r['per_call_us']:.1f} us per call, hashing {r['hashing_us']:.1f} us", flush=True)
    for name, r in res["rates"].items():
        print(f"rate {name}: {r['gb_s']:.0f} GB/s, {r['bytes_per_sm_clk']:.2f} B/SM/clk", flush=True)


if __name__ == "__main__":
    main()
