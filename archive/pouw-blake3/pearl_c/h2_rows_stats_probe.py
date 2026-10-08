"""-h2's rows form `s` on the card: hash_rows_b3s_stats (A's row leaves and forming's stats in one pass over A) held bit for
bit to hash_rows_b3s and stats_s5, then graph-replayed µs a call of A's row leaves plus the stats under `w` (hash_rows_b3s,
then stats_s5) and `s`, and the served Llama-3.1-8B prefill pass and decode step they come to.  Diagnostic, not a panel row.

  python3 h2_rows_stats_probe.py SHIP OUT.json   (env: PEARLC_EXPECT_UUID; CALLS, REPS)

SHIP is build.sh's tree with the kernel.  Gates, from outputs poisoned to 0xA5, before any timing:
- per shape, `s` writes `w`'s leaves (m_real rows) and level keys, and stats_s5's alpha, beta, 1/alpha and (s, rho) for all m
  rows, the filler rows' included; a whole call under each (run_all) writes the same bytes in every buffer; and the no-write
  controls leave the poisoned buffers unequal;
- on EDGE rows (zeros, -0, NaN, ±inf, subnormals, a square past FP32's top, and normals), `s`'s stats are stats_s5's.
"""
import ctypes as C
import hashlib
import importlib.util
import json
import os
import statistics
import struct
import sys
import time

SHAPES = [(8192, 4096), (8192, 14336), (8192, 8192), (8192, 28672), (32, 4096), (32, 14336)]
N, POISON, EDGE_K = 1024, 0xA5, 4096
#: Llama-3.1-8B's linears a layer as (k, count): q/k/v, o and gate/up at 4,096, down at 14,336; 32 layers
LLAMA_8B, LAYERS = ((4096, 3), (14336, 1)), 32
STATS = ("alpha_a", "beta_a", "ia_a", "aux_a")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edge_rows(k: int) -> bytes:
    """64 FP32 rows of k words: normals at one binade a row, and rows 0–9 the edge cases."""
    import random
    rnd, f = random.Random(7), lambda x: struct.pack("<f", x)
    nan, inf, sub = struct.pack("<I", 0x7FC00000), f(float("inf")), struct.pack("<I", 0x00000401)
    rows = []
    for r in range(64):
        scale = 2.0 ** rnd.randint(-30, 30)
        w = [f(rnd.gauss(0, 1) * scale) for _ in range(k)]
        if r == 0:
            w = [f(0.0)] * k
        elif r == 1:
            w = [f(-0.0)] * k
        elif r == 2:
            w = [nan] * k
        elif r == 3:
            w[::8] = [nan] * (k // 8)
        elif r == 4:
            w[5], w[16] = inf, f(float("-inf"))
        elif r == 5:
            w = [sub] * k
        elif r == 6:
            w[24] = f(3e17)
        elif r == 7:
            w[32] = f(3e19)
        elif r == 8:
            w[8], w[9] = nan, f(-0.0)
        elif r == 9:
            w = [f(rnd.gauss(0, 1) * 2.0 ** rnd.randint(-40, 40)) for _ in range(k)]
        rows.append(b"".join(w))
    return b"".join(rows)


def main() -> int:
    ship, out = sys.argv[1:3]
    R = load("h2s_run", os.path.join(ship, "run.py"))
    cubin = open(os.path.join(ship, "pearl_c_sm120.cubin"), "rb").read()
    d = R.Dev(cubin)
    refused = d.identity(os.environ.get("PEARLC_EXPECT_UUID"))
    calls, reps = int(os.environ.get("CALLS", "16")), int(os.environ.get("REPS", "25"))
    res = {"schema": "verity-pouw/h2-rows-stats-probe/v1", "device": d.name, "uuid": d.uuid, "refused": refused,
           "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "calls": calls, "reps": reps, "n": N, "gates": {}, "shapes": {}}
    save = lambda: open(out, "w").write(json.dumps(res, indent=1) + "\n")  # noqa: E731
    if refused:
        save()
        return 3
    d.open()
    vp, P = C.c_void_p, C.POINTER
    R.bind(d.cu, {"cuStreamCreate": [P(vp), C.c_uint], "cuStreamBeginCapture_v2": [vp, C.c_int], "cuStreamEndCapture": [vp, P(vp)],
                  "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp]})
    nv = R.Nvml(d.uuid)
    res["clk_before"] = nv.read()
    s = vp()
    d.ck("cuStreamCreate", C.byref(s), 1)
    ev = [vp(), vp()]
    for e in ev:
        d.ck("cuEventCreate", C.byref(e), 0)
    ok = True

    def written(run, ptr, nbytes) -> bytes:
        d.ck("cuMemsetD8_v2", C.c_uint64(ptr), POISON, nbytes)
        run()
        d.sync()
        return d.read(ptr, nbytes)

    def pair(mr, k, xa, xb, keys):
        m = R.cdiv(mr, R.BM_D) * R.BM_D
        w = R.Pipeline(d, m, N, k, "sm120", keys, xa, xb, m_real=mr, hashing="h2")
        f = R.Pipeline(d, m, N, k, "sm120", keys, xa, xb, m_real=mr, hashing="h2", rows_form="s")
        return w, {nm: fn for nm, _, fn in w.steps()}, f, {nm: fn for nm, _, fn in f.steps()}

    def gates(w, ws, f, fs, whole: bool) -> dict:
        g, mr = {}, w.m_real
        want = written(ws["hash_rows_a"], w.b["tree_a"], 32 * mr)
        g["leaves"] = written(fs["hash_rows_a"], f.b["tree_a"], 32 * mr) == want
        g["control.leaves"] = written(lambda: None, f.b["tree_a"], 32 * mr) != want
        g["skeys"] = written(fs["hash_rows_a"], f.b["skeys_a"], f.size["skeys_a"]) == written(ws["hash_rows_a"], w.b["skeys_a"],
                                                                                                    w.size["skeys_a"])
        for buf in STATS:
            ref = written(ws["stats_a"], w.b[buf], w.size[buf])
            g[buf] = written(fs["hash_rows_a"], f.b[buf], f.size[buf]) == ref
        g["control.stats"] = written(lambda: None, f.b["aux_a"], f.size["aux_a"]) != ref
        if whole:
            for p_ in (w, f):
                for nm, size in p_.size.items():
                    if nm in p_.b and nm not in ("xa", "xb", "tma"):         # tma: the GEMMs' tensor maps, encoded once
                        d.ck("cuMemsetD8_v2", C.c_uint64(p_.b[nm]), POISON, size)
                p_.run_all()
            names = [nm for nm in w.size if nm in w.b and nm in f.b and nm not in ("xa", "xb", "tma")]
            differ = [nm for nm in names if d.read(w.b[nm], w.size[nm]) != d.read(f.b[nm], f.size[nm])]
            g["call.buffers"], g["call.differing"] = len(names), differ
            g["call"] = not differ
        return g

    def capture(fs) -> vp:
        d.stream = s
        d.ck("cuStreamBeginCapture_v2", s, 0)
        for _ in range(calls):
            for fn in fs:
                fn()
        g, ge = vp(), vp()
        d.ck("cuStreamEndCapture", s, C.byref(g))
        d.stream = None
        d.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

    def replay(ge) -> float:
        ms = C.c_float()
        d.ck("cuEventRecord", ev[0], s)
        d.ck("cuGraphLaunch", ge, s)
        d.ck("cuEventRecord", ev[1], s)
        d.ck("cuEventSynchronize", ev[1])
        d.ck("cuEventElapsedTime", C.byref(ms), ev[0], ev[1])
        return ms.value * 1e3 / calls

    def stand_in_keys():
        keys = {nm: d.put(hashlib.shake_256(f"probe/{nm}".encode()).digest(sz)) for nm, sz in R.TIMING_KEYS.items()}
        keys.update(t_row=d.put(R.b3s_template(R.ROW_SCHEMA)), t_tile=d.put(R.b3s_template(R.TILE_SCHEMA)))
        return keys

    keys = stand_in_keys()
    xa, xb = d.put(edge_rows(EDGE_K)), d.alloc(N * 4 * EDGE_K)
    d.launch("gen_f32", R.cdiv(N * EDGE_K, 256), 256, p=(xb,), i=(N, EDGE_K, 12))
    w, ws, f, fs = pair(64, EDGE_K, xa, xb, keys)
    res["gates"]["edge"] = g = gates(w, ws, f, fs, whole=False)
    ok = ok and all(v for nm, v in g.items() if not nm.startswith("call."))
    print("edge", g, flush=True)
    for p_ in (w, f):
        p_.free()
    d.free(xa)
    d.free(xb)
    save()

    for mr, k in SHAPES:
        m = R.cdiv(mr, R.BM_D) * R.BM_D
        xa, xb = d.alloc(m * 4 * k), d.alloc(N * 4 * k)
        d.launch("gen_f32", R.cdiv(mr * k, 256), 256, p=(xa,), i=(mr, k, 11))
        d.launch("gen_f32", R.cdiv(N * k, 256), 256, p=(xb,), i=(N, k, 12))
        if m > mr:
            d.launch("gen_f32", R.cdiv((m - mr) * k, 256), 256, p=(xa + mr * 4 * k,), i=(m - mr, k, 13))
        w, ws, f, fs = pair(mr, k, xa, xb, keys)
        g = gates(w, ws, f, fs, whole=True)
        ok = ok and all(v for nm, v in g.items() if nm not in ("call.buffers", "call.differing"))
        for fn in (ws["hash_rows_a"], ws["stats_a"], fs["hash_rows_a"]):
            fn()
        d.sync()
        v = {"w": [ws["hash_rows_a"], ws["stats_a"]], "s": [fs["hash_rows_a"]], "rows.w": [ws["hash_rows_a"]], "stats": [ws["stats_a"]]}
        graphs = {nm: capture(fns) for nm, fns in v.items()}
        for ge in graphs.values():
            replay(ge)
        t = {nm: [] for nm in graphs}
        names = list(graphs)
        for r in range(reps):
            for j in range(len(names)):
                nm = names[(j + r) % len(names)]
                t[nm].append(replay(graphs[nm]))
        us = {nm: round(statistics.median(xs), 2) for nm, xs in t.items()}
        res["shapes"][f"{mr}x{k}"] = {"gates": g, "us": us, "spread_us": {nm: [round(min(xs), 2), round(max(xs), 2)] for nm, xs in t.items()}}
        print(f"{mr}x{k}", g, us, flush=True)
        save()
        for p_ in (w, f):
            p_.free()
        d.free(xa)
        d.free(xb)
    for x in keys.values():
        d.free(x)

    per = {}
    for label, rows in (("prefill_pass", 8192), ("decode_step", 32)):
        sh = {k: res["shapes"].get(f"{rows}x{k}", {}).get("us") for k, _ in LLAMA_8B}
        if all(sh.values()):
            per[label] = {form: round(LAYERS * sum(n * sh[k][form] for k, n in LLAMA_8B) / 1000, 3) for form in ("w", "s", "rows.w", "stats")}
            per[label]["saved_ms"] = round(per[label]["w"] - per[label]["s"], 3)
    res["llama_3_1_8b_ms"] = per
    res["clk_after"] = nv.read()
    res["ok"] = ok
    save()
    print("llama_3_1_8b_ms", per, "ok", ok, flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
