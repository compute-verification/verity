#!/usr/bin/env python3
"""-h2's hashing on one RTX PRO 6000 Blackwell (sm_120), from pearl_c_sm120's cubin through run.py's Pipeline: at one shape,
per variant and hashing format, a bit-exact gate of every buffer the hashing writes against `verity.commitments`' frame-b3s
(-h2) or frame-b3 (-h1) and Pearl-C's digests, then the hashing steps' diagnostic timing (an untimed screen: CUDA events and
CUDA graphs at the node's clocks, not a panel row).

  python3 h2_bench.py [check.json]   (env: PEARLC_EXPECT_UUID, SHAPE="8192" or "MxNxK", VARIANTS="sm120 sm120-unpromoted",
                                      HASHINGS="h2 h1", REPS=9, BUDGET=140 seconds, SEED=0; PYTHONPATH holds `verity`)

0. Identity, as run.py's (one visible device, compute capability 12.0, the RTX PRO 6000's name, the UUID PEARLC_EXPECT_UUID),
   then run.py's check: every kernel against the ship's fixture, byte for byte (-h1, both variants).  A mismatch exits 3 or 4
   before any timing.
1. Per (variant, format): the pipeline at SHAPE (m padded to 64 rows, as run.py's timing), inputs gen_f32 at seeds 11 and 12;
   the domains `CommitmentDomain`s of the format's tree hash over stand-in bindings (A's owner 0, B's -2, the tile tree's 0),
   the digest and tile-leaf keys Pearl-C's -h1 keys (-h2 shares them), E_B's line key and the unit block stand-ins.  Every step
   runs once, then gemm_pearlc_cu so C~ and U are stored.
2. The gate, before any timing, sampled at a fixed seed (SEED, the variant and the format): A's, B's and the tile tree's node
   keys, every level-0 leaf sampled (the first and last two, and random ones; a row read back from the device), its pads,
   every node of each level at most FULL wide and N_NODES of each wider one (its children as the device holds them), the root;
   -h2's segment level keys (`skeys_<a|b>`, hash_rows_b3s's first CTA's); sampled message digests against keyed(k_msg, C~ ‖ U)
   of the device's C~ and U, and all of them against hash_msg's (the fused chain's digests at 128 x 128 tiles); sampled tile
   leaves against keyed(k_leaf, the tile's digests).  A mismatch emits the gate and exits 4, nothing timed.
3. Timing rounds until the (variant, format)'s share of BUDGET: each hashing step (the unit's pads and node keys, B's tree, A's
   commitment, hash_msg, the tile leaves and tile tree), the chain GEMMs that price the fused digests (gemm_pearlc_b less
   gemm_pearlc_s), gemm_ct and the plain GEMMs for scale; REPS CUDA-event reps each after a warm-up (run.py's `timed`, one
   launch between events), and for steps under GRAPH_BELOW_MS a CUDA graph of back-to-back copies (at least ~2 ms a replay,
   L2-warm), REPS replays.  `call_hash_chain` is the call's hashing in order (A's commitment, hash_msg where unfused, the tile
   leaves and tile tree), timed both ways.  NVML clocks before and after each round.

One JSON object per line on stdout; a summary on stderr.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE if os.path.exists(os.path.join(HERE, "run.py")) else os.path.join(HERE, "..", "pearl_c_sm120"))
import run as R  # noqa: E402

from verity.commitments import CommitmentDomain, RangeIndexedDomain  # noqa: E402
from verity.commitments.blake3 import blake3, keyed  # noqa: E402
from verity.commitments.merkle import b3s_level_key, blake3_node_key  # noqa: E402

#: each format's tree hash (verity_pouw.serving.HASH_FORMATS' tree_hash)
TREE_HASH = {"h1": "blake3", "h2": "blake3-s256"}
#: Pearl-C's digest keys under -h1 and -h2 (pearl_c.H1_KEYS)
DIGEST_KEYS = {"k_msg": blake3(b"pearl-c/h1/message"), "k_leaf": blake3(b"pearl-c/h1/leaf")}
N_ROWS, N_PADS, N_NODES, N_TREE_LEAVES, N_DIGESTS, N_TILES, FULL = 6, 16, 256, 256, 512, 96, 2048
GRAPH_BELOW_MS, GRAPH_REPLAY_MS, GRAPH_MAX_COPIES = 2.0, 2.0, 64
TIMED = ("pad_a", "node_keys_a", "pad_t", "node_keys_t", "pad_b", "node_keys_b", "hash_rows_b", "tree_b", "hash_rows_a", "tree_a",
         "seed_a", "hash_msg", "hash_leaf", "tree_leaves_t", "tree_t", "gemm_pearlc_s", "gemm_pearlc_b", "gemm_ct", "gemm_plain",
         "gemm_fast")
COMMIT, TILE_HASH = ("hash_rows_a", "tree_a", "seed_a"), ("hash_leaf", "tree_leaves_t", "tree_t")
LABEL = ("untimed screen (fill, preemptible): diagnostic medians at the node's clocks (the clock record says which), not a "
         "panel row; graphs of L2-warm back-to-back copies; stand-in domain bindings, Pearl-C's -h1 digest keys")

vp, u, i32, P = C.c_void_p, C.c_uint, C.c_int, C.POINTER
GRAPH_API = {"cuStreamCreate": [P(vp), u], "cuStreamSynchronize": [vp], "cuStreamBeginCapture_v2": [vp, i32],
             "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong],
             "cuGraphLaunch": [vp, vp], "cuGraphExecDestroy": [vp], "cuGraphDestroy": [vp]}
CAPTURE_RELAXED, STREAM_NON_BLOCKING = 2, 1


def stand_in(tag: str, n: int = 32) -> bytes:
    return hashlib.shake_256(f"pouw-h2-bench/{tag}".encode()).digest(n)


def domains(hashing: str, m_real: int, n: int, tiles: int) -> dict[str, CommitmentDomain]:
    h = TREE_HASH[hashing]
    return {"a": CommitmentDomain(stand_in("binding-a"), 0, RangeIndexedDomain(m_real), hash=h),
            "b": CommitmentDomain(stand_in("binding-b"), -2, RangeIndexedDomain(n), hash=h),
            "t": CommitmentDomain(stand_in("binding-t"), 0, RangeIndexedDomain(tiles), hash=h)}


def sample(count: int, want: int, rng: random.Random) -> list[int]:
    """Indices into range(count): all of them if at most `want`, else the first and last two and random ones up to `want`."""
    if count <= want:
        return list(range(count))
    out = {0, 1, count - 2, count - 1}
    while len(out) < want:
        out.add(rng.randrange(count))
    return sorted(out)


class Tally(dict):
    def __call__(self, name: str, ok: bool, where) -> None:
        c = self.setdefault(name, {"checked": 0, "failed": 0, "first": []})
        c["checked"] += 1
        if not ok:
            c["failed"] += 1
            if len(c["first"]) < 4:
                c["first"].append(where)

    def failed(self) -> list[str]:
        return sorted(nm for nm, c in self.items() if c["failed"] or not c["checked"])


def check_tree(dev: R.Dev, pipe: R.Pipeline, tag: str, dom: CommitmentDomain, count: int, leaf_of, n_leaves: int,
               rng: random.Random, t: Tally) -> None:
    """tree_<tag> (levels concatenated, level 0 the leaves then the pads, the root last) and keys_<tag> against `dom`."""
    base, width = pipe.b[f"tree_{tag}"], R.pow2(count)
    at = lambda idx, nodes=1: dev.read(base + 32 * idx, 32 * nodes)  # noqa: E731
    dep = R.depth(count)
    if dep:
        t(f"node_keys_{tag}", dev.read(pipe.b[f"keys_{tag}"], 32 * dep) == b"".join(blake3_node_key(dom.domain_id, lv)
                                                                                   for lv in range(dep)), dep)
    for i in sample(count, n_leaves, rng):
        t(f"leaves_{tag}", at(i) == leaf_of(i), i)
    for r in (count + i for i in sample(width - count, N_PADS, rng)):
        t(f"pads_{tag}", at(r) == dom.pad(r), r)
    off, w, level = 0, width, 0
    while w > 1:
        up, half = off + w, w // 2
        if half <= FULL:
            kids, nodes = at(off, w), at(up, half)
            for i in range(half):
                t(f"nodes_{tag}", nodes[32 * i:32 * i + 32] == dom.node(level, i, kids[64 * i:64 * i + 32], kids[64 * i + 32:64 * i + 64]),
                  [level + 1, i])
        else:
            for i in sample(half, N_NODES, rng):
                kid = at(off + 2 * i, 2)
                t(f"nodes_{tag}", at(up + i) == dom.node(level, i, kid[:32], kid[32:]), [level + 1, i])
        off, w, level = up, half, level + 1
    t(f"root_{tag}", dev.read(pipe.root(tag), 32) == at(off) and pipe.root(tag) == base + 32 * off, off)


def gate(dev: R.Dev, pipe: R.Pipeline, doms: dict[str, CommitmentDomain], rng: random.Random) -> Tally:
    """Every hashed buffer of a pipeline whose steps have run, then gemm_pearlc_cu (C~ and U stored), against the references."""
    b, mr, n, rb, t = pipe.b, pipe.m_real, pipe.n, pipe.row_bytes, Tally()
    per, row_tiles = n // 32, n // R.TILE
    rows = lambda x, schema, dom: lambda i: dom.leaf(i, i, schema, dev.read(b[x] + i * rb, rb))  # noqa: E731
    check_tree(dev, pipe, "a", doms["a"], mr, rows("xa", R.ROW_SCHEMA, doms["a"]), N_ROWS, rng, t)
    check_tree(dev, pipe, "b", doms["b"], n, rows("xb", R.ROW_SCHEMA, doms["b"]), N_ROWS, rng, t)
    if pipe.hashing == "h2":
        lv = R.b3s_levels(rb)
        for tag in ("a", "b"):
            t(f"skeys_{tag}", dev.read(b[f"skeys_{tag}"], 32 * lv) == b"".join(b3s_level_key(doms[tag].domain_id, x)
                                                                              for x in range(lv)), lv)
    for d in sample(mr * per, N_DIGESTS, rng):
        i, h = divmod(d, per)
        msg = dev.read(b["c"] + 4 * (i * n + 32 * h), 128) + dev.read(b["u"] + 4 * (i * n + 32 * h), 128)
        t("digests", dev.read(b["digests"] + 32 * d, 32) == keyed(DIGEST_KEYS["k_msg"], msg), [i, h])
    for x in sample(pipe.tiles, N_TILES, rng):
        ti, tj = divmod(x, row_tiles)
        msg = b"".join(dev.read(b["digests"] + 32 * (r * per + 2 * tj), 64) for r in range(R.TILE * ti, min(R.TILE * ti + R.TILE, mr)))
        t("tile_leaves", dev.read(b["leaves"] + 32 * x, 32) == keyed(DIGEST_KEYS["k_leaf"], msg), [ti, tj])
    check_tree(dev, pipe, "t", doms["t"], pipe.tiles, lambda i: doms["t"].leaf(i, i, R.TILE_SCHEMA, dev.read(b["leaves"] + 32 * i, 32)),
               N_TREE_LEAVES, rng, t)
    dev.copy(b["digests2"], b["digests"], pipe.size["digests"])
    pipe.extra()["hash_msg"]()
    dev.sync()
    t("digests_vs_hash_msg", pipe.diff("digests", "digests2") == 0, "words differing")
    return t


class Graphs:
    """CUDA graphs of back-to-back copies of launches, captured on a stream of their own through run.py's Dev.stream."""

    def __init__(self, dev: R.Dev):
        R.bind(dev.cu, GRAPH_API)
        self.dev, self.s = dev, vp()
        dev.ck("cuStreamCreate", C.byref(self.s), STREAM_NON_BLOCKING)

    def time(self, fns, copies: int, reps: int) -> list[float]:
        d, g, ge = self.dev, vp(), vp()
        d.ck("cuStreamBeginCapture_v2", self.s, CAPTURE_RELAXED)
        d.stream = self.s
        try:
            for _ in range(copies):
                for f in fns:
                    f()
        finally:
            d.stream = None
            d.ck("cuStreamEndCapture", self.s, C.byref(g))
        d.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        d.ck("cuGraphLaunch", ge, self.s)
        d.ck("cuStreamSynchronize", self.s)
        out = []
        for _ in range(reps):
            a, e, ms = vp(), vp(), C.c_float()
            d.ck("cuEventCreate", C.byref(a), 0)
            d.ck("cuEventCreate", C.byref(e), 0)
            d.ck("cuEventRecord", a, self.s)
            d.ck("cuGraphLaunch", ge, self.s)
            d.ck("cuEventRecord", e, self.s)
            d.ck("cuEventSynchronize", e)
            d.ck("cuEventElapsedTime", C.byref(ms), a, e)
            d.cu.cuEventDestroy_v2(a)
            d.cu.cuEventDestroy_v2(e)
            out.append(ms.value / copies)
        d.cu.cuGraphExecDestroy(ge)
        d.cu.cuGraphDestroy(g)
        return out


def derived(ms: dict[str, float], fused: bool) -> dict[str, float]:
    """The hashing's parts from step medians: A's commitment, B's tree (once per weight and epoch), the unit's setup, the
    digests (fused: gemm_pearlc_b less gemm_pearlc_s), the tile leaves and tile tree, their sum a call's, and scale."""
    g = lambda *names: sum(ms.get(x, 0.0) for x in names)  # noqa: E731
    out = {"a_commit": g(*COMMIT), "b_tree": g("pad_b", "node_keys_b", "hash_rows_b", "tree_b"),
           "unit": g("pad_a", "node_keys_a", "pad_t", "node_keys_t"), "digests_unfused": g("hash_msg"),
           "tile_leaves": g("hash_leaf"), "tile_tree": g("tree_leaves_t", "tree_t")}
    if "gemm_pearlc_b" in ms and "gemm_pearlc_s" in ms:
        out["digests_fused"] = ms["gemm_pearlc_b"] - ms["gemm_pearlc_s"]
    out["call_hashing"] = out["a_commit"] + out["digests_fused" if fused else "digests_unfused"] + out["tile_leaves"] + out["tile_tree"]
    plain = [ms[x] for x in ("gemm_plain", "gemm_fast") if x in ms]
    if plain:
        out["plain_gemm"] = min(plain)
        out["call_hashing_over_plain_gemm"] = out["call_hashing"] / min(plain)
    return {k: round(v, 5) for k, v in out.items()}


def bench(dev: R.Dev, nv: R.Nvml, graphs: Graphs, shape, variant: str, hashing: str, reps: int, budget: float, seed: str) -> int:
    t0 = time.time()
    m_real, n, k = shape
    m = R.cdiv(m_real, R.BM_D) * R.BM_D
    tiles = R.cdiv(m_real, R.TILE) * (n // R.TILE)
    doms = domains(hashing, m_real, n, tiles)
    host = {"line_b": stand_in("line_b"), "unit": stand_in("unit", 104), **DIGEST_KEYS,
            "t_row": R.b3s_template(R.ROW_SCHEMA), "t_tile": R.b3s_template(R.TILE_SCHEMA),
            **{f"dom_{x}": d.domain_id for x, d in doms.items()}}
    keys = {nm: dev.put(v) for nm, v in host.items()}
    xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
    for x, rows, s in ((xa, m_real, 11), (xb, n, 12)):
        dev.launch("gen_f32", R.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, s))
    pipe = R.Pipeline(dev, m, n, k, variant, keys, xa, xb, m_real=m_real, hashing=hashing)
    head = {"shape": [m_real, n, k], "m_padded": m, "variant": variant, "hashing": hashing, "fused": pipe.fused,
            "tree_hash": TREE_HASH[hashing], "tiles": tiles, "uuid": dev.uuid}
    try:
        pipe.run_all()
        fns = {**pipe.extra(), **{nm: f for nm, _, f in pipe.steps()}}
        fns["gemm_pearlc_cu"]()
        dev.sync()
        g0 = time.time()
        t = gate(dev, pipe, doms, random.Random(f"{seed}/{variant}/{hashing}"))
        failed = t.failed()
        R.emit({"kind": "gate", **head, "ok": not failed, "failed": failed, "checks": t, "seconds": round(time.time() - g0, 2)})
        if failed:
            print(f"GATE FAILED {head}: {failed}", file=sys.stderr)
            return 4
        timed = [x for x in TIMED if x in fns]
        chain = [fns[x] for x in COMMIT + (() if pipe.fused else ("hash_msg",)) + TILE_HASH]
        fns["call_hash_chain"] = lambda: [f() for f in chain]
        rnd = 0
        while rnd == 0 or time.time() - t0 < budget:
            before, ev, gr, copies, errs = nv.read(), {}, {}, {}, {}
            for x in timed + ["call_hash_chain"]:
                ev[x] = statistics.median(dev.timed(fns[x], reps))
            for x in timed + ["call_hash_chain"]:
                if ev[x] < GRAPH_BELOW_MS:
                    copies[x] = max(1, min(GRAPH_MAX_COPIES, int(GRAPH_REPLAY_MS / max(ev[x], 1e-3))))
                    try:
                        gr[x] = statistics.median(graphs.time([fns[x]], copies[x], reps))
                    except RuntimeError as e:
                        errs[x] = str(e)
            after = nv.read()
            both = {**ev, **gr}
            if not {"gemm_pearlc_b", "gemm_pearlc_s"} <= gr.keys():
                both.update({x: ev[x] for x in ("gemm_pearlc_b", "gemm_pearlc_s") if x in ev})
            evd, grd = derived(ev, pipe.fused), derived(both, pipe.fused)
            evd["call_hash_chain"] = round(ev["call_hash_chain"] + (evd.get("digests_fused", 0.0) if pipe.fused else 0.0), 5)
            if "call_hash_chain" in gr:
                grd["call_hash_chain"] = round(gr["call_hash_chain"] + (grd.get("digests_fused", 0.0) if pipe.fused else 0.0), 5)
            R.emit({"kind": "h2bench", "label": LABEL, **head, "round": rnd, "reps": reps,
                    "ms_event": {x: round(v, 5) for x, v in ev.items()}, "ms_graph": {x: round(v, 5) for x, v in gr.items()},
                    "graph_copies": copies, "graph_errors": errs, "derived_event_ms": evd, "derived_graph_ms": grd,
                    "gate": {"ok": True, "checked": sum(c["checked"] for c in t.values())}, "clk_before": before, "clk_after": after,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"{'x'.join(map(str, shape))} {variant} {hashing} round {rnd}: call hashing {grd['call_hashing']:.4f} ms (graph) "
                  f"{evd['call_hashing']:.4f} ms (event); A commit {grd['a_commit']:.4f}, tile {grd['tile_leaves'] + grd['tile_tree']:.4f}",
                  file=sys.stderr)
            rnd += 1
        return 0
    finally:
        pipe.free()
        for p_ in (xa, xb, *keys.values()):
            dev.free(p_)


def main(argv: list[str]) -> int:
    man_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "check", "check.json")
    man = json.load(open(man_path))
    image = open(os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "pearl_c_sm120.cubin"), "rb").read()
    dev = R.Dev(image)
    expect = os.environ.get("PEARLC_EXPECT_UUID")
    refused = dev.identity(expect)
    nv = R.Nvml(dev.uuid)
    R.emit({"kind": "env", "tool": "h2_bench", "device": dev.name, "uuid": dev.uuid, "expect_uuid": expect,
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "sms": dev.sms, "cuda_driver_api": dev.driver,
            "nvml_driver": nv.driver, "cubin_sha256": hashlib.sha256(image).hexdigest(),
            "check_json_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest(), "clk": nv.read(), "refused": refused,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    variants = os.environ.get("VARIANTS", "sm120 sm120-unpromoted").split()
    hashings = os.environ.get("HASHINGS", "h2 h1").split()
    t0 = time.time()
    res = R.check(dev, man, os.path.dirname(man_path), variants)
    R.emit({"kind": "check", "shape": man["shape"], "seconds": round(time.time() - t0, 2), "variants": res})
    if not all(r["pearl_c_ok"] and r["plain_ok"] for r in res.values()):
        print("check failed: " + "; ".join(f"{v} {r['failed']}" for v, r in res.items()), file=sys.stderr)
        return 4
    shape = R.parse_shape(os.environ.get("SHAPE", "8192"))
    reps, budget, seed = int(os.environ.get("REPS", "9")), float(os.environ.get("BUDGET", "140")), os.environ.get("SEED", "0")
    graphs, combos = Graphs(dev), [(v, h) for v in variants for h in hashings]
    for j, (variant, hashing) in enumerate(combos):
        share = (budget - (time.time() - t0)) / (len(combos) - j)
        rc = bench(dev, nv, graphs, shape, variant, hashing, reps, share, f"{seed}/{'x'.join(map(str, shape))}")
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
