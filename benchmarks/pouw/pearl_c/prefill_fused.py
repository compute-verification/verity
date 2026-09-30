"""Prefill's -h2 and -h3 hashing on GPU 1's GEMM at m = n = k = 8,192 (sm_120, node 2): the message digests fused into
gemm_pearlc_b's epilogue, against the plain FP8 GEMM in the same run.

  python3 prefill_fused.py --gpu1 GPU1_SHIP --cubin hash_sm120a.cubin --standin h1_standin.cubin --vectors h1_vectors.json \
      --out prefill_fused.json --transcripts DIR [--reps R]

Standard library only. GPU 1's `run.py` and cubin (its branch as pushed, built by its own build.sh, unedited) run the GEMMs,
the forming, the clean-up's factors and the kernels -h1 shares with -h2 and -h3: the frame-b3 levels of A's and the tile
tree, seed_A with E_A's line key from A's root (seed_line_a), the tile leaves (hash_leaf_p) and the message digests as
their own pass (hash_msg_v). This bench's cubins add A's producer with -h2's segment leaves and trees fused (derive_rows_b3s;
with -h3's row seeds too), its plain twin (derive_rows_vec), the frame-b3s leaves of the tile tree (hash_leaves_b3s) and a
read-only L2 flush.

0. The GPU the lease named (GPU_LEASE_UUID, through GPU 1's identity check), then GPU 1's check (its fixture, both
   variants): any buffer that isn't bit-exact exits 4 with nothing timed.
1. Per variant (sm120, the v1 lines; sm120-unpromoted, v2), one call, each arm a CUDA graph on one stream:
   - nohash: A's producer (derive_rows_vec: A's row r is y's row r cycled, ^ 0x3C00), E_A's forming, the clean-up's factors
     and gemm_pearlc_s (y alone);
   - h2: the producer with A's -h2 row leaves (derive_rows_b3s), A's tree levels, seed_line_a over A's root, forming, the
     factors, gemm_pearlc_b (y, and the message digests hashed from registers in its epilogue: GPU 1's form, all 8
     consumer warps hashing together after each tile's last MMA, "lockstep"), the tile leaves and the tile tree;
   - h3: h2 with -h3's row seeds in the producer; seed_line_a stays (the unit's seed_A keys the tickets), and its one line
     key stands in for -h3's per-row line keys in E_A's forming (the same keyed hash per row, under another key);
   - h2u, h3u: unfused, gemm_pearlc (C~, U and y stored) and hash_msg_v in place of gemm_pearlc_b;
   and the plain FP8 GEMMs gemm_plain and gemm_fast. REPS rounds, every graph once a round in rotated order, a read-only
   L2 flush before each and the SM clock after it. An arm's hashing is its median less nohash's, and its hashing-only
   slowdown is 1 + hashing / the faster plain GEMM's median. --components adds each hashing part alone to the rounds (the
   producers, A's tree and seed, gemm_pearlc_b, gemm_pearlc and gemm_pearlc_s, hash_msg_v, the tile leaves and tree) and
   writes no transcripts: a diagnostic breakdown.
2. Transcripts (verify_h1.py's format), written to DIR: each arm once more from its own y, so no arm's output fits another's,
   after every buffer its transcript reads is poisoned (0xA5): A (the producer's output), A's tree, the row seeds, seed_A,
   E_A's line key, the digests, the tile leaves and tree, C~, U and y. A fused arm's C~ and U come from an unfused
   gemm_pearlc launch after it; its digests stay the fused kernel's. The negative control (nc) is nohash dumped as an -h3
   transcript, which the verifier must reject. A's rows aren't in a transcript: row r is `a_half(tag, r)` twice.

GPU 1's kernel has no staggered form of the fused digests (two consumer groups half a tile apart, GPU 2's K3), so only the
lockstep form is measured.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
#: an FP32 word's top byte: the sign kept and the exponent in [120, 127], so every word of A is a finite normal in [2^-7, 2)
HI = bytes((b & 0x80) | 0x3C | (b & 3) for b in range(256))
X3C = bytes(b ^ 0x3C for b in range(256))
POISON = 0xA5
ARMS = ("nohash", "h2", "h3", "h2u", "h3u")
DUMPS = ("h2", "h3", "h2u", "h3u", "nc")


def a_half(tag: str, r: int, nbytes: int) -> bytes:
    """A's row r, its first nbytes (the row is them repeated: the producer cycles y's row): SHAKE128 of the tag and r, each
    FP32 word's top byte through HI."""
    s = bytearray(hashlib.shake_128(f"{tag} {r}".encode()).digest(nbytes))
    s[3::4] = s[3::4].translate(HI)
    return bytes(s)


def y_rows(tag: str, m: int, nbytes: int) -> bytes:
    """The producer's input: m rows of nbytes (BF16), A's halves with each BF16's high byte ^ 0x3C, which the producer's
    ^ 0x3C00 undoes."""
    out = bytearray()
    for r in range(m):
        s = bytearray(a_half(tag, r, nbytes))
        s[1::2] = s[1::2].translate(X3C)
        out += s
    return bytes(out)


def b3s_template(schema: str) -> tuple[bytes, int]:
    """Format -h2's segment-key message with the rank, position, L and j left zero (hash_h2.cuh fills them), and its length."""
    name = schema.encode()
    head = b"\x05" + bytes(28) + len(name).to_bytes(2, "little") + name
    return head + bytes(64 - len(head)), len(head)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Kernels:
    """This bench's cubins in GPU 1's context, launched on the device's stream unless told otherwise (so a capture records
    them)."""

    def __init__(self, R, dev, blobs: list[bytes]):
        self.R, self.dev, self.fns, self.mods = R, dev, {}, []
        for blob in blobs:
            mod = C.c_void_p()
            dev.ck("cuModuleLoadData", C.byref(mod), blob)
            self.mods.append(mod)

    def fn(self, name: str):
        if name not in self.fns:
            for mod in self.mods:
                f = C.c_void_p()
                if self.dev.cu.cuModuleGetFunction(C.byref(f), mod, name.encode()) == 0:
                    self.fns[name] = f
                    break
            else:
                raise RuntimeError(f"no kernel {name} in this bench's cubins")
        return self.fns[name]

    def go(self, name: str, grid: int, block: int, p=(), i=(), stream=None) -> None:
        a = self.R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        params = (C.c_void_p * 1)(C.addressof(a))
        self.dev.ck("cuLaunchKernel", self.fn(name), max(1, grid), 1, 1, block, 1, 1, 0, stream or self.dev.stream, params, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu1", required=True, help="GPU 1's ship: pearl_c_sm120.cubin, run.py, check/")
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--standin", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--size", type=int, default=8192, help="m = n = k (a dry run's smaller shape; the domains are 8,192's)")
    ap.add_argument("--components", action="store_true",
                    help="also time each hashing part alone, interleaved with the arms (a diagnostic breakdown; no transcripts)")
    args = ap.parse_args()
    if not (args.transcripts or args.components):
        ap.error("--transcripts is required unless --components")
    M = N = K = args.size
    RB = 4 * K
    R = load(os.path.join(args.gpu1, "run.py"), "gpu1_run")
    P, u64, vp = C.POINTER, C.c_uint64, C.c_void_p
    image = open(os.path.join(args.gpu1, "pearl_c_sm120.cubin"), "rb").read()
    dev = R.Dev(image)
    expect = os.environ.get("GPU_LEASE_UUID")
    # GPU 1's dry-run switches: PEARLC_ANY_GPU skips the identity check, and PEARLC_UNGATED (refused without it) times past a
    # failed check
    any_gpu, ungated = bool(os.environ.get("PEARLC_ANY_GPU")), bool(os.environ.get("PEARLC_UNGATED"))
    refused = [] if any_gpu else dev.identity(expect)
    if ungated and not any_gpu:
        refused.append("PEARLC_UNGATED is for dry runs on a stand-in driver (with PEARLC_ANY_GPU)")
    run = os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local"
    res: dict = {"run": run, "device": dev.name, "uuid": dev.uuid, "lease_uuid": expect, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "sms": dev.sms, "cuda_driver_api": dev.driver, "gpu1_cubin_sha256": hashlib.sha256(image).hexdigest(),
                 "gpu1_source": open(os.path.join(args.gpu1, "source.txt")).read().split()[0],
                 "cubin_sha256": hashlib.sha256(open(args.cubin, "rb").read()).hexdigest(),
                 "standin_sha256": hashlib.sha256(open(args.standin, "rb").read()).hexdigest(), "refused": refused,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shape": [M, N, K], "ungated": ungated,
                 "staggered": "not in GPU 1's kernel: gemm_pearlc_b's 8 consumer warps hash together after each tile's last MMA"}
    save = lambda: open(args.out, "w").write(json.dumps(res, indent=1))  # noqa: E731
    if refused:
        save()
        sys.exit(3)
    dev.open()
    R.bind(dev.cu, {"cuStreamCreate": [P(vp), C.c_uint], "cuStreamBeginCapture_v2": [vp, C.c_int],
                    "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong],
                    "cuGraphLaunch": [vp, vp]})
    nv = R.Nvml(dev.uuid)
    res["clocks_idle"] = nv.read()
    man_path = os.path.join(args.gpu1, "check", "check.json")
    t0 = time.time()
    chk = R.check(dev, json.load(open(man_path)), os.path.dirname(man_path), list(R.VARIANT_G))
    res["check"] = {v: {"buffers": r["buffers"], "failed": r["failed"]} for v, r in chk.items()}
    res["check_seconds"] = round(time.time() - t0, 2)
    save()
    print(f"GPU 1's check: {res['check']}", flush=True)
    if not all(r["pearl_c_ok"] and r["plain_ok"] for r in chk.values()) and not ungated:
        sys.exit(4)

    # Everything but the captures and their timed replays runs on the legacy stream, where GPU 1's memsets and copies are
    # ordered with the launches; the graphs run on s, joined by a synchronize at each switch.
    s = vp()
    dev.ck("cuStreamCreate", C.byref(s), 1)
    Kn = Kernels(R, dev, [open(args.cubin, "rb").read(), open(args.standin, "rb").read()])
    V = json.load(open(args.vectors))
    pa, pt = V["decode"]["prefill_a3"], V["decode"]["prefill_t2"]
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
    seeds = dev.alloc(32 * M)
    Kn.go("b3s_level_keys", 1, 32, (segkeys, keys["dom_a"]), (9,))
    y_in, xa, xb = dev.alloc(2 * M * N), dev.alloc(M * RB), dev.alloc(N * RB)
    fl, sink = dev.alloc(512 << 20), dev.alloc(16)
    dev.launch("gen_f32", -(-N * K // 256), 256, p=(xb,), i=(N, K, 12))
    put_y = lambda tag: dev.ck("cuMemcpyHtoD_v2", u64(y_in), C.c_char_p(y_rows(tag, M, 2 * N)), 2 * M * N)  # noqa: E731
    put_y("prefill y timing")
    dev.sync()

    tiles = (M // 64) * (N // 64)
    pipes, seqs = {}, {}
    for variant in R.VARIANT_G:
        pipe = R.Pipeline(dev, M, N, K, variant, keys, xa, xb)
        pipes[variant] = pipe
        st = {nm: (ph, f) for nm, ph, f in pipe.steps()}
        ex = pipe.extra()
        b = pipe.b
        forming = [st[nm][1] for nm in ("lines_ea", "stats_a", "eprime_a", "form_a")]
        factors = [st["af"][1], st["pa"][1]]
        plain = lambda: Kn.go("derive_rows_vec", -(-M * (RB // 256) // 128), 128, (y_in, xa), (M, RB, N))  # noqa: E731
        fused = lambda mode, b=b: Kn.go("derive_rows_b3s", M, RB // 256,  # noqa: E731
                                        (y_in, xa, b["tree_a"], keys["dom_a"], rtmpl, segkeys, seedk, saltp, rootbp, seeds),
                                        (M, RB, N, rt[1], mode, index))
        commit = [st["tree_a"][1], st["seed_a"][1]]
        tile_tree = [lambda b=b: Kn.go("hash_leaves_b3s", -(-tiles // 128), 128, (b["leaves"], b["tree_t"], keys["dom_t"], ttmpl),
                                       (tiles, 32, 0, tt[1])), st["tree_t"][1]]
        post_b = [ex["gemm_pearlc_b"], st["hash_leaf"][1], *tile_tree]
        post_u = [ex["gemm_pearlc_cu"], ex["hash_msg"], st["hash_leaf"][1], *tile_tree]
        seqs[variant] = {"nohash": [plain, *forming, *factors, ex["gemm_pearlc_s"]],
                         "h2": [lambda f=fused: f(1), *commit, *forming, *factors, *post_b],
                         "h3": [lambda f=fused: f(2), *commit, *forming, *factors, *post_b],
                         "h2u": [lambda f=fused: f(1), *commit, *forming, *factors, *post_u],
                         "h3u": [lambda f=fused: f(2), *commit, *forming, *factors, *post_u]}
        if args.components:
            seqs[variant].update({"part gemm_s": [ex["gemm_pearlc_s"]], "part gemm_b": [ex["gemm_pearlc_b"]],
                                  "part gemm_cu": [ex["gemm_pearlc_cu"]], "part hash_msg_v": [ex["hash_msg"]],
                                  "part a_tree_seed": commit, "part tiles": [st["hash_leaf"][1], *tile_tree],
                                  "part forming_factors": [*forming, *factors]})
            seqs.setdefault("producer", {"plain": [plain], "h2": [lambda f=fused: f(1)], "h3": [lambda f=fused: f(2)]})
        for nm, (ph, f) in st.items():                       # the unit's level keys, the weight side, B's plain codes
            if ph in ("unit", "job", "plain_job"):
                f()
        seqs[variant]["nohash"][0]()
        st["plainq_a"][1]()
        for seq in seqs[variant].values():                   # every kernel once outside a capture (GPU 1's shared-memory probes)
            for f in seq:
                f()
        dev.sync()
    base = pipes["sm120"]
    bst = {nm: f for nm, _ph, f in base.steps()}
    seqs["plain"] = {"gemm_plain": [bst["gemm_plain"]], "gemm_fast": [bst["gemm_fast"]]}
    dev.sync()

    def graph(seq) -> vp:
        dev.stream = s
        dev.ck("cuStreamBeginCapture_v2", s, 0)
        for f in seq:
            f()
        g, ge = vp(), vp()
        dev.ck("cuStreamEndCapture", s, C.byref(g))
        dev.stream = None
        dev.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

    graphs = {f"{v} {arm}": graph(seq) for v, arms in seqs.items() for arm, seq in arms.items()}
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

    for ge in graphs.values():
        replay(ge)
    raw = {nm: [] for nm in graphs}
    clk = {nm: [] for nm in graphs}
    before = nv.read()
    order = list(graphs)
    for rep in range(args.reps):
        for nm in order[rep % len(order):] + order[:rep % len(order)]:
            raw[nm].append(replay(graphs[nm]))
            clk[nm].append(nv.read().get("sm_mhz"))
    res["clocks_timing"] = {"before": before, "after": nv.read()}
    med = {nm: statistics.median(x) for nm, x in raw.items()}
    fastest = min(("plain gemm_plain", "plain gemm_fast"), key=lambda nm: med[nm])
    res["timing"] = {nm: {"ms": round(med[nm], 5), "raw_ms": [round(x, 5) for x in raw[nm]], "sm_mhz": clk[nm],
                          "sm_mhz_median": statistics.median(c) if (c := [x for x in clk[nm] if x is not None]) else None}
                     for nm in graphs}
    res["baseline"] = {"graph": fastest, "ms": round(med[fastest], 5)}
    res["rows"] = {}
    for v in R.VARIANT_G:
        for arm in ARMS[1:]:
            h = med[f"{v} {arm}"] - med[f"{v} nohash"]
            res["rows"][f"{v} {arm}"] = {"hashing_ms": round(h, 5), "slowdown_hashing_only": round(1 + h / med[fastest], 4),
                                         "call_ms": round(med[f"{v} {arm}"], 5), "nohash_ms": round(med[f"{v} nohash"], 5)}
    if args.components:
        res["parts"] = {}
        for v in R.VARIANT_G:
            p = lambda nm: med[f"{v} part {nm}"]  # noqa: E731
            parts = {"a_tree_seed": p("a_tree_seed"), "digests_fused": p("gemm_b") - p("gemm_s"),
                     "c_u_stores": p("gemm_cu") - p("gemm_s"), "hash_msg_v": p("hash_msg_v"), "tiles": p("tiles"),
                     "forming_factors": p("forming_factors")}
            for f in ("h2", "h3"):
                parts[f"producer_{f}"] = med[f"producer {f}"] - med["producer plain"]
                parts[f"sum_{f}"] = parts[f"producer_{f}"] + parts["a_tree_seed"] + parts["digests_fused"] + parts["tiles"]
                parts[f"sum_{f}u"] = parts[f"sum_{f}"] - parts["digests_fused"] + parts["c_u_stores"] + parts["hash_msg_v"]
            res["parts"][v] = {k: round(x, 5) for k, x in parts.items()}
            print(f"{v} parts (ms): {res['parts'][v]}", flush=True)
    save()
    for nm, r in res["rows"].items():
        print(f"{nm}: hashing {r['hashing_ms']:.4f} ms, {r['slowdown_hashing_only']:.4f}x over {fastest} {med[fastest]:.4f} ms",
              flush=True)
    if args.components:
        return

    os.makedirs(args.transcripts, exist_ok=True)
    res["transcripts"], res["consistency"] = {}, {}
    for v, pipe in pipes.items():
        b, sz, ex = pipe.b, pipe.size, pipe.extra()
        poison = [(xa, M * RB), (seeds, 32 * M)] + [(b[nm], sz[nm]) for nm in ("tree_a", "seed_a", "line_a", "digests", "leaves",
                                                                                  "tree_t", "c", "u", "y")]
        for arm in DUMPS:
            tag = f"prefill A {v} {arm}"
            put_y(tag)
            for p_, n_ in poison:
                dev.ck("cuMemsetD8_v2", u64(p_), POISON, n_)
            dev.sync()
            replay(graphs[f"{v} {'nohash' if arm == 'nc' else arm}"], flush=False)
            if arm not in ("h2u", "h3u"):
                ex["gemm_pearlc_cu"]()
            dev.sync()
            fmt = "h2" if arm in ("h2", "h2u") else "h3"
            parts = [("a_tree", dev.read(b["tree_a"], sz["tree_a"]))]
            if fmt == "h3":
                parts.append(("row_seeds", dev.read(seeds, 32 * M)))
            parts += [("unit_seed", dev.read(b["seed_a"], 32)), ("c", dev.read(b["c"], 4 * M * N)), ("u", dev.read(b["u"], 4 * M * N)),
                      ("digests", dev.read(b["digests"], sz["digests"])), ("leaves", dev.read(b["leaves"], 32 * tiles)),
                      ("tile_tree", dev.read(b["tree_t"], sz["tree_t"]))]
            if arm in ("h2", "h3"):                          # the fused digests against hash_msg_v over the unfused C~ and U
                dev.copy(b["digests2"], b["digests"], sz["digests"])
                ex["hash_msg"]()
                dev.sync()
                res["consistency"][f"{v} {arm}"] = {"digests_fused_vs_hash_msg_v_words_differing": pipe.diff("digests", "digests2")}
            head = {"run": run, "arm": f"prefill {v} {arm}", "variant": v, "m": M, "rb": RB, "n": N, "hashing": fmt, "a": pa, "t": pt,
                    "a_rows_y": {"tag": tag, "half": RB // 2}, "sections": [[k, len(x)] for k, x in parts]}
            path = os.path.join(args.transcripts, f"prefill-{'v1' if v == 'sm120' else 'v2'}-{arm}.bin")
            h = hashlib.sha256()
            with open(path, "wb") as f:
                for chunk in [json.dumps(head).encode() + b"\n"] + [x for _k, x in parts]:
                    f.write(chunk)
                    h.update(chunk)
            res["transcripts"][f"{v} {arm}"] = {"path": path, "sha256": h.hexdigest(), "bytes": os.path.getsize(path)}
            save()
            print(f"transcript {v} {arm}: {path} sha256:{h.hexdigest()}", flush=True)
    res["clocks_after"] = nv.read()
    save()
    bad = [nm for nm, c in res["consistency"].items() if any(c.values())]
    if bad:
        print(f"fused digests differ from hash_msg_v's: {bad}", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
