#!/usr/bin/env python3
"""The served decode's hashing on GPU 1's sm_120 pipeline (the `run.Pipeline` the vLLM path drives), per call, at
Llama-3.1-8B's four linears with m = 32 (m₁₆ = 32 rows, padded to 64): diagnostic timing for the served-gap attribution and
for the spread and fuse levers, not a panel row.

Each variant's launches, CALLS times over in one stream, are captured as a CUDA graph and replayed REPS rounds, the variants
in rotated order within a round, with L2 warm as a decode step finds A's rows and the digests. The figure is the median
replay over CALLS, in µs a call; a graph node costs about 1 µs of its own, which the vLLM path's launches from Python exceed.

Variants, per shape:
  a.h1 / a.h1.wN   the -h1 A commitment (hash_node_keys, hash_rows_w, hash_tree_small, seed_line_a) as run.py launches it
                   (`run.spread`'s rows a CTA), and with hash_rows_w at N rows a CTA (N = 8, the served 8ca97148's; 4, 2, 1)
  a.h2             -h2's, hash_rows_b3s deriving its segment level keys (GPU 1's 997ac6bd: no b3s_level_keys launch)
  t / t.lN         the tile hashing (hash_msg_v, hash_leaf_p) as run.py launches it, and hash_leaf_p at N tiles a CTA (32, the
                   served 8ca97148's; 8, 2)
  call.h1, call.h2, call.nohash   the whole call as the vLLM path runs it (pouw_pearl_c_device.CALL_STEPS; nohash without
                   HASH_STEPS)
  k.<kernel>       each hashing kernel of the call alone

Gates before any timing, per shape: from outputs poisoned to 0xA5, each spread variant writes the bytes of its default (A's
leaves, the tile leaves), hash_rows_b3s writes the same leaves with p3 null as with it set, and the level keys it stores at p3
are b3s_level_keys'; each no-write control (the kernel left out) leaves them unequal.

  python3 decode_served.py --gpu1 SHIP --out OUT.json   (env: PEARLC_EXPECT_UUID; CALLS=32, REPS=30; PEARLC_ANY_GPU and
                                                         PEARLC_UNGATED for dry runs on a stand-in driver)
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

M_REAL = 32
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}   # (n, k)
LAYERS = 32
CALL_STEPS = ("pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "stats_a", "eprime_a", "form_a", "af", "pa",
              "gemm_pearlc", "hash_msg", "hash_leaf")
HASH_STEPS = frozenset({"pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "hash_msg", "hash_leaf"})
ROWS_W, LEAF_T = (8, 4, 2, 1), (32, 8, 2)
POISON = 0xA5


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu1", required=True, help="GPU 1's ship (build.sh's output): run.py and pearl_c_sm120.cubin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = load("gpu1_run", os.path.join(args.gpu1, "run.py"))
    vp, u64, P = C.c_void_p, C.c_uint64, C.POINTER
    image = open(os.path.join(args.gpu1, "pearl_c_sm120.cubin"), "rb").read()
    dev = R.Dev(image)
    any_gpu, ungated = bool(os.environ.get("PEARLC_ANY_GPU")), bool(os.environ.get("PEARLC_UNGATED"))
    refused = [] if any_gpu else dev.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    calls, reps = int(os.environ.get("CALLS", "32")), int(os.environ.get("REPS", "30"))
    res: dict = {"schema": "verity-pouw/pearl-c-sm120-decode-served/v1", "run": os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local",
                 "device": dev.name, "uuid": dev.uuid, "sms": dev.sms, "cubin_sha256": hashlib.sha256(image).hexdigest(),
                 "gpu1_source": open(os.path.join(args.gpu1, "source.txt")).read().split()[0], "refused": refused, "ungated": ungated,
                 "m_real": M_REAL, "shapes": {s: list(v) for s, v in SHAPES.items()}, "layers": LAYERS, "calls": calls, "reps": reps,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save = lambda: open(args.out, "w").write(json.dumps(res, indent=1) + "\n")  # noqa: E731
    if refused:
        save()
        return 3
    dev.open()
    R.bind(dev.cu, {"cuStreamCreate": [P(vp), C.c_uint], "cuStreamBeginCapture_v2": [vp, C.c_int], "cuStreamEndCapture": [vp, P(vp)],
                    "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp]})
    nv = R.Nvml(dev.uuid)
    res["clk_before"] = nv.read()
    s = vp()
    dev.ck("cuStreamCreate", C.byref(s), 1)
    keys = {nm: dev.put(hashlib.shake_256(f"pearl-c-decode-served/{nm}".encode()).digest(sz)) for nm, sz in R.TIMING_KEYS.items()}
    keys.update(t_row=dev.put(R.b3s_template(R.ROW_SCHEMA)), t_tile=dev.put(R.b3s_template(R.TILE_SCHEMA)))
    m = R.cdiv(M_REAL, R.BM_D) * R.BM_D
    ok, graphs, gates = True, {}, {}

    def capture(body) -> vp:
        dev.stream = s
        dev.ck("cuStreamBeginCapture_v2", s, 0)
        for _ in range(calls):
            body()
        g, ge = vp(), vp()
        dev.ck("cuStreamEndCapture", s, C.byref(g))
        dev.stream = None
        dev.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

    def poison(ptr: int, n: int) -> None:
        dev.ck("cuMemsetD8_v2", u64(ptr), POISON, n)

    for shape, (n, k) in SHAPES.items():
        rb, tiles = 4 * k, R.cdiv(M_REAL, R.TILE) * (n // R.TILE)
        xa, xb = dev.alloc(m * rb), dev.alloc(n * rb)
        dev.launch("gen_f32", R.cdiv(M_REAL * k, 256), 256, p=(xa,), i=(M_REAL, k, 11))
        dev.launch("gen_f32", R.cdiv(n * k, 256), 256, p=(xb,), i=(n, k, 12))
        pipes = {h: R.Pipeline(dev, m, n, k, "sm120", keys, xa, xb, m_real=M_REAL, hashing=h) for h in ("h1", "h2")}
        steps = {}
        for h, pipe in pipes.items():
            pipe.run_all()
            steps[h] = {nm: f for nm, _, f in pipe.steps()}
        b1, b2 = pipes["h1"].b, pipes["h2"].b
        skeys, levels = dev.alloc(32 * 16), (R.cdiv(rb, R.B3S_SEG) - 1).bit_length()
        leaves_a = lambda b: (b["tree_a"], 32 * M_REAL)  # noqa: E731
        dev.sync()

        def rows_w(w: int) -> None:
            dev.launch("hash_rows_w", R.cdiv(M_REAL, w), 32 * w, 4 * w * R.rows_w_words(rb), p=(xa, b1["tree_a"], keys["dom_a"]),
                       i=(M_REAL, rb))

        def b3s_keys() -> None:
            dev.launch("b3s_level_keys", 1, 32, p=(skeys, keys["dom_a"]), i=(levels,))

        def rows_b3s_null() -> None:
            dev.launch("hash_rows_b3s", M_REAL, 32 * R.cdiv(R.cdiv(rb, R.B3S_SEG), 32),
                       p=(xa, b2["tree_a"], keys["dom_a"], 0, keys["t_row"]), i=(M_REAL, rb, 0, R.b3s_template_len(R.ROW_SCHEMA)))

        def leaf_t(t: int) -> None:
            dev.launch("hash_leaf_p", R.cdiv(tiles, t), 4 * t, p=(b1["digests"], b1["leaves"], keys["k_leaf"]), i=(M_REAL, n))

        # the gates: each variant's bytes against its default's, from poisoned outputs; each no-write control unequal
        def written(run, out) -> bytes:
            poison(*out)
            run()
            dev.sync()
            return dev.read(*out)

        g: dict = {}
        want = written(steps["h1"]["hash_rows_a"], leaves_a(b1))
        for w in ROWS_W:
            g[f"a.h1.w{w}"] = written(lambda w=w: rows_w(w), leaves_a(b1)) == want
        g["a.h1.control"] = written(lambda: None, leaves_a(b1)) != want
        want = written(rows_b3s_null, leaves_a(b2))
        g["a.h2"] = written(steps["h2"]["hash_rows_a"], leaves_a(b2)) == want
        g["a.h2.control"] = written(lambda: None, leaves_a(b2)) != want
        sk = (b2["skeys_a"], 32 * levels)
        g["a.h2.stored_keys"] = written(steps["h2"]["hash_rows_a"], sk) == written(b3s_keys, (skeys, 32 * levels))
        g["a.h2.stored_keys.control"] = written(lambda: None, sk) != written(b3s_keys, (skeys, 32 * levels))
        tl = (b1["leaves"], 32 * tiles)
        want = written(steps["h1"]["hash_leaf"], tl)
        for t in LEAF_T:
            g[f"t.l{t}"] = written(lambda t=t: leaf_t(t), tl) == want
        g["t.control"] = written(lambda: None, tl) != want
        for h in ("h1", "h2"):                                 # leave every buffer written for the call graphs
            for nm in ("hash_rows_a", "tree_a", "seed_a", "hash_msg", "hash_leaf"):
                steps[h][nm]()
        dev.sync()
        gates[shape] = g
        ok = ok and all(g.values())

        def seq(h: str, names) -> callable:
            fs = [steps[h][x] for x in names if x in steps[h]]
            return lambda: [f() for f in fs]

        a_names = ("pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a")
        pre, post = steps["h1"].get("node_keys_a", lambda: None), (steps["h1"]["tree_a"], steps["h1"]["seed_a"])
        v = {"a.h1": seq("h1", a_names), "a.h2": seq("h2", a_names),
             "t": seq("h1", ("hash_msg", "hash_leaf")),
             "call.h1": seq("h1", CALL_STEPS), "call.h2": seq("h2", CALL_STEPS),
             "call.nohash": seq("h1", [x for x in CALL_STEPS if x not in HASH_STEPS])}
        for w in ROWS_W:
            v[f"a.h1.w{w}"] = lambda w=w: (pre(), rows_w(w), post[0](), post[1]())
        for t in LEAF_T:
            v[f"t.l{t}"] = lambda t=t: (steps["h1"]["hash_msg"](), leaf_t(t))
        for nm in ("node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "hash_msg", "hash_leaf"):
            if nm in steps["h1"]:
                v[f"k.{nm}"] = steps["h1"][nm]
        v["k.hash_rows_a.h2"] = steps["h2"]["hash_rows_a"]
        v["k.b3s_level_keys"] = b3s_keys
        graphs[shape] = {nm: capture(f) for nm, f in v.items()}
        dev.sync()
    res["gates"] = gates
    save()
    print(f"gates: {gates}", flush=True)
    if not ok and not ungated:
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
    res["clk_after"] = nv.read()
    us = {sh: {} for sh in graphs}
    for (sh, nm), xs in t_.items():
        q = statistics.quantiles(xs, n=4)
        us[sh][nm] = {"median": round(statistics.median(xs), 2), "q1": round(q[0], 2), "q3": round(q[2], 2)}
    res["us_per_call"] = us
    med = lambda nm: sum(us[sh][nm]["median"] for sh in us if nm in us[sh])  # noqa: E731
    res["per_forward_ms"] = {nm: round(LAYERS * med(nm) / 1e3, 3) for nm in next(iter(us.values()))}
    res["launches_per_call"] = {"hashing_h1": sum(1 for x in CALL_STEPS if x in HASH_STEPS and x in steps["h1"]),
                                "hashing_h2": sum(1 for x in CALL_STEPS if x in HASH_STEPS and x in steps["h2"]),
                                "all_h1": sum(1 for x in CALL_STEPS if x in steps["h1"])}
    save()
    for sh in us:
        print(sh, {nm: us[sh][nm]["median"] for nm in us[sh]}, flush=True)
    print("per forward (ms, 32 layers):", res["per_forward_ms"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
