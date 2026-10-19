#!/usr/bin/env python3
"""Pearl-C -h3's decode hashing per call on one RTX PRO 6000 (sm_120): Pearl-C's own launches (`run.Pipeline`, the served
call's steps `CALL_STEPS`), and pd.cu's fused ones, at Llama-3.1-8B's linears and decode's m16 = 32 and 128.

Per shape:

1. The reference call, eager: every step once on real-format keys (dom_a, the unit block and the hm96 constants as the
   served path derives them; line_b, dom_b and dom_t stand-ins of their sizes, which no A-side byte reads).
2. With pd.cubin: the check.  pd's launches run into poisoned (0xA5) copies of every buffer they write and each is compared
   byte for byte with the reference launches' (row digests, A's tree, seed_A, line_a, the row seeds and line keys, E_A's
   codes, the tile leaves), and the row digests, A's root, seed_A, line_a, the row seeds and a sample of tile leaves with
   the Python references (`pearl_c.row_digest`, `audit.HM96`'s tree, `pearl_c.unit_seed_a`, `pearl_c.row_seed`,
   `pearl_c.line_key`, `pearl_c.sample_line` for E_A's codes, `pearl_c.leaf`).  A no-write control: pd's launches on a zero-row call leave the poison.  A shape is
   timed only if every comparison passes.
3. Graphs of CHAIN consecutive calls of the shape (a ring of 2 buffer sets, as the served path rotates what a side lane
   touches), one per schedule (`SCHEDULES`), timed interleaved rep by rep, the order rotating, with NVML's SM clock each rep.
4. With a PD_STAMP variant: its serial graph once more, and the last call's pd_commit stamps (each row's and the tail's
   phase ends, global timer and SM clock).
5. With --trace: each graph launched TRACE times for nsys (--cuda-graph-trace=node); `trace.py` reads the kernels back.

One JSON object per line into OUT/bench.jsonl.

  python3 bench.py PEARL_CUBIN [PD_CUBIN] --out DIR [--reps N] [--trace] [--shapes qkv,o,gate_up,down] [--rows 32,128]
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks", "pouw", "pearl_c_sm120"), os.path.join(ROOT, "integrations", "vllm")]

import run as R  # noqa: E402


def _served_steps() -> tuple[tuple[str, ...], frozenset[str]]:
    """CALL_STEPS and HASH_STEPS as the served path declares them (pouw_pearl_c_device.py), read from its source: importing
    it pulls in vLLM's side of the integration, which the node's bare python3 doesn't need."""
    import ast
    path = os.path.join(ROOT, "integrations", "vllm", "verity_vllm", "protocol_options", "pouw_pearl_c_device.py")
    found = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in ("CALL_STEPS", "HASH_STEPS"):
            v = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
            found[node.targets[0].id] = ast.literal_eval(v)
    return tuple(found["CALL_STEPS"]), frozenset(found["HASH_STEPS"])


CALL_STEPS, HASH_STEPS = _served_steps()

LLAMA = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
INDEX, CHAIN, TRACE = 5, 8, 3
u, i32, vp, sz, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.POINTER
GRAPH_API = {"cuStreamCreate": [P(vp), u], "cuStreamBeginCapture_v2": [vp, i32], "cuStreamEndCapture": [vp, P(vp)],
             "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong], "cuGraphLaunch": [vp, vp],
             "cuStreamSynchronize": [vp], "cuStreamWaitEvent": [vp, vp, u], "cuGraphGetNodes": [vp, P(vp), P(sz)],
             "cuGraphDestroy": [vp], "cuGraphExecDestroy": [vp], "cuModuleLoadData": [P(vp), C.c_char_p],
             "cuFuncSetAttribute": [vp, i32, i32], "cuFuncGetAttribute": [P(i32), i32, vp]}
CAPTURE_THREAD_LOCAL = 1
POISON = 0xA5


def emit(out, obj: dict) -> None:
    line = json.dumps(obj)
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


# ---- the device: two modules in one context, streams, graphs ------------------------------------------------------

class Module(R.Dev):
    """A second module in `dev`'s context (pd.cubin): its own functions, launched on `dev`'s current stream."""

    def __init__(self, dev: R.Dev, cubin: bytes):  # noqa: D107 - shares dev's driver and context
        self.__dict__.update({k: v for k, v in dev.__dict__.items() if k not in ("mod", "fns")})
        self.host, self.fns, self.mod = dev, {}, vp()
        self.cubin = cubin
        self.ck("cuModuleLoadData", C.byref(self.mod), cubin)

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        super().launch(name, grid, block, smem, p, i, self.host.stream if stream is None else stream)


class Lanes:
    """The streams a call runs on (main, and the side lanes `tile` and `tree`) and a pool of events for forks and joins,
    captured into a graph as its edges."""

    def __init__(self, dev: R.Dev, mods=()):
        self.dev, self.mods = dev, [dev, *mods]
        self.main, self.tile, self.tree = (self.stream() for _ in range(3))
        self.pool = [self.event() for _ in range(256)]
        self.k = 0

    def stream(self):
        s = vp()
        self.dev.ck("cuStreamCreate", C.byref(s), 1)   # CU_STREAM_NON_BLOCKING
        return s

    def event(self):
        e = vp()
        self.dev.ck("cuEventCreate", C.byref(e), 2)    # CU_EVENT_DISABLE_TIMING
        return e

    def mark(self, stream):
        e = self.pool[self.k % len(self.pool)]
        self.k += 1
        self.dev.ck("cuEventRecord", e, stream)
        return e

    def wait(self, stream, e) -> None:
        if e is not None:
            self.dev.ck("cuStreamWaitEvent", stream, e, 0)

    def on(self, stream, fns) -> None:
        old = self.dev.stream
        self.dev.stream = stream
        try:
            for f in fns:
                f()
        finally:
            self.dev.stream = old

    def capture(self, body) -> tuple[vp, int, list]:
        """`body` captured from the main stream into an instantiated graph: (exec, node count, the kernels in capture order as
        [step, kernel, lane]: the call step that launched it (`Shape.ref`), or for pd's module the kernel, and the stream it
        was launched on, `main` the call's critical path)."""
        d, g, ge, n, seq = self.dev, vp(), vp(), sz(), []
        lane = {self.main.value: "main", self.tile.value: "tile", self.tree.value: "tree"}
        for m in self.mods:
            def logged(name, *a, _m=m, **kw):
                seq.append([getattr(d, "step", name) if _m is d else name, name, lane.get(getattr(d.stream, "value", None), "main")])
                type(_m).launch(_m, name, *a, **kw)
            m.launch = logged
        self.k = 0
        d.ck("cuStreamBeginCapture_v2", self.main, CAPTURE_THREAD_LOCAL)
        try:
            body()
        finally:
            d.ck("cuStreamEndCapture", self.main, C.byref(g))
            for m in self.mods:
                del m.launch
        d.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        d.ck("cuGraphGetNodes", g, None, C.byref(n))
        d.ck("cuGraphDestroy", g)
        return ge, n.value, seq

    def run(self, ge) -> None:
        self.dev.ck("cuGraphLaunch", ge, self.main)
        self.dev.ck("cuStreamSynchronize", self.main)


def timed_interleaved(dev: R.Dev, lanes: Lanes, nv: R.Nvml, graphs: dict[str, vp], reps: int) -> dict:
    """Each graph `reps` times, rep by rep, the order rotating each rep: ms per launch and the SM clock read before each rep."""
    names = list(graphs)
    ev = [[vp(), vp()] for _ in names]
    for pair in ev:
        for j in range(2):
            dev.ck("cuEventCreate", C.byref(pair[j]), 0)
    for name in names:                       # warm
        lanes.run(graphs[name])
    out = {name: [] for name in names}
    clocks = []
    for r in range(reps):
        clocks.append(nv.read().get("sm_mhz"))
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            a, b = ev[names.index(name)]
            ms = C.c_float()
            dev.ck("cuEventRecord", a, lanes.main)
            dev.ck("cuGraphLaunch", graphs[name], lanes.main)
            dev.ck("cuEventRecord", b, lanes.main)
            dev.ck("cuEventSynchronize", b)
            dev.ck("cuEventElapsedTime", C.byref(ms), a, b)
            out[name].append(round(ms.value, 5))
    return {"ms": out, "sm_mhz": clocks}


# ---- keys -----------------------------------------------------------------------------------------------------------

def real_keys(m16: int) -> dict[str, bytes]:
    """The A side's keys as the served path derives them for call INDEX of m16 rows (`pouw_pearl_c_host.call_keys`), the
    prover's salt key and the hm96 constants; line_b, dom_b, dom_t stand-ins (no A-side byte reads them)."""
    from verity.core.protocols.pouw.audit import HM96, activation_binding, hm96_kernel_constants, rows_domain
    shake = lambda tag, n: hashlib.shake_256(f"pouw-decode/{tag}".encode()).digest(n)  # noqa: E731
    salt, root_b = shake("salt", 32), shake("root_b", 64)
    return {"salt": salt, "root_b": root_b, "salt_key_raw": shake("salt-key", 32),
            "dom_a": rows_domain(activation_binding(salt, INDEX), INDEX, m16, HM96).domain_id,
            "unit": salt + root_b + INDEX.to_bytes(8, "little"), "salt_key": shake("salt-key", 32),
            "hm": hm96_kernel_constants(), "line_b": shake("line_b", 32), "dom_b": shake("dom_b", 64),
            "dom_t": shake("dom_t", 64)}


# ---- the shapes -----------------------------------------------------------------------------------------------------

#: each launch's serial SHA-512 compressions (its depth: the longest one-thread chain), by step, at (m16, k): Pearl-C's, then
#: pd's (`pd_commit` mode 0, PD_PAIR's: its tail after the row's leaf, one more without PD_PAIR; mode 1 as `pd_commit/rows`)
def depths(m16: int, k: int) -> dict[str, list[tuple[str, int]]]:
    s, levels, r = k // 256, R.depth(m16), min(64, m16)
    rows = 9 + s // 2 + 1 + 4                      # segments, the fold, then the row's seed (2), line key and E_A line
    leaf = 9 + s // 2 + 1 + 3                      # segments, the fold, then the hm96 leaf (2) and the frame leaf
    fill = int(m16 % R.BM_D != 0)                  # E_A's filler lines, after line_a
    return {"rowdig_a": [("s512_row_segs", 9), ("s512_folds", s // 2 + 1)],
            "hash_rows_a": [("hm_leaves", 4), ("s512_frame_leaves", 2)],
            "tree_a": [("s512_tree_small", 3 * levels)],
            "seed_a": [("s512_seeds", 5)], "seeds_a": [("s512_seeds", 5)], "lines_ea": [("lines", 2)],
            "hash_leaf": [("s512_tile_segs", 9), ("s512_folds", r // 4 + 1)],
            "pd": [("pd_commit", leaf + 2 * levels + 3 + fill), ("pd_commit/rows", rows), ("pd_tree", 2 * levels + 4 + fill),
                   ("pd_tiles", 9 + r // 4 + 1)]}


class Shape:
    """One linear at decode: the reference pipeline (set 0 of the ring, owning every buffer), set 1 (its own copies of
    what a call writes), and pd's launches when pd.cubin is given."""

    RING = ("c", "u", "y", "segdig_t", "leaves", "tree_a", "seed_a", "line_a", "rowdig_a", "seeds_a", "lkeys_a",
            "segdig_a", "ea_codes")

    def __init__(self, dev: R.Dev, pd: Module | None, name: str, m16: int, n: int, k: int):
        self.dev, self.pd, self.name, self.m16, self.n, self.k, self.index = dev, pd, name, m16, n, k, INDEX
        self.mp = R.cdiv(m16, R.BM_D) * R.BM_D
        self.kb = real_keys(m16)
        self.keys = {x: dev.put(self.kb[x]) for x in ("line_b", "dom_a", "dom_b", "dom_t", "unit", "salt_key", "hm")}
        self.xa, self.xb = dev.alloc(self.mp * 4 * k), dev.alloc(n * 4 * k)
        for x, rows, seed in ((self.xa, m16, 11), (self.xb, n, 12)):
            dev.launch("gen_f32", R.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
        self.p0 = R.Pipeline(dev, self.mp, n, k, "sm120", self.keys, self.xa, self.xb, m_real=m16)
        size = self.p0.size
        own = {x: dev.alloc(size[x]) for x in self.RING}
        self.p1 = R.Pipeline(dev, self.mp, n, k, "sm120", self.keys, self.xa, self.xb, bufs={**self.p0.b, **own}, m_real=m16)
        self.sets = [self.p0, self.p1]
        self.steps = [{nm: f for nm, _, f in p.steps()} for p in self.sets]
        for nm, phase, f in self.p0.steps():
            if phase in ("epoch", "unit", "job"):
                f()
        for st in self.steps:
            for nm in self.call_names(hashing=True):
                st[nm]()
        dev.sync()
        self.tiles = self.p0.tiles
        self.pdc = []
        if pd is not None:
            from pd_host import PdCall
            self.pdc = [PdCall(pd, p, self.keys) for p in self.sets]

    def call_names(self, hashing: bool) -> list[str]:
        have = self.steps[0]
        return [x for x in CALL_STEPS if x in have and (hashing or x not in HASH_STEPS)]

    def ref(self, j: int, names) -> list:
        """Set j's steps `names`, each naming itself on the device (`dev.step`) as it launches, for Lanes.capture's log."""
        def labeled(x, f):
            def g():
                self.dev.step = x
                f()
            return g
        return [labeled(x, self.steps[j][x]) for x in names]

    # each schedule is a body for Lanes.capture: CHAIN calls on the ring, the side lanes joined at the end
    def schedules(self, lanes: Lanes) -> dict[str, object]:
        hashing, plain = self.call_names(True), self.call_names(False)
        commit = [x for x in hashing if x in ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a")]
        main_rest = [x for x in hashing if x not in commit and x != "hash_leaf"]

        def chain(call):
            def body():
                done = [None, None]
                for c in range(CHAIN):
                    done[c % 2] = call(c % 2, done[c % 2])
                for e in done:
                    lanes.wait(lanes.main, e)
            return body

        def serial(names, pdfn=None):
            def call(j, _):
                lanes.on(lanes.main, self.ref(j, names) if pdfn is None else pdfn(j))
                return None
            return chain(call)

        def deferred(main_fns, tile_fns):
            def call(j, done):
                lanes.wait(lanes.main, done)            # set j is rewritten once its side work is done
                lanes.on(lanes.main, main_fns(j))
                e = lanes.mark(lanes.main)
                lanes.wait(lanes.tile, e)
                lanes.on(lanes.tile, tile_fns(j))
                return lanes.mark(lanes.tile)
            return chain(call)

        out = {"ref_serial": serial(hashing), "ref_nohash": serial(plain),
               "ref_deferred": deferred(lambda j: self.ref(j, commit + main_rest), lambda j: self.ref(j, ["hash_leaf"]))}
        if self.pd is not None:
            from pd_host import PD_STEPS, SIDE, VARIANTS
            P_ = self.pdc
            pd_rest = [x for x in hashing if x not in PD_STEPS]   # the call's hashing steps pd's launches don't do

            def side_tree(v):
                # A's tree and seed_A on the tree lane beside forming and the GEMM: no byte changes when every row of the
                # call is real (m16 = mp); at m16 < mp the filler rows' lines read line_a, so there this is the
                # filler-keying proposal's schedule, timed only (`lines_ea` reads the ring's earlier line_a)
                def call(j, done):
                    lanes.wait(lanes.main, done)
                    lanes.on(lanes.main, P_[j].rows(v))
                    e = lanes.mark(lanes.main)
                    lanes.wait(lanes.tree, e)
                    lanes.on(lanes.tree, P_[j].tree(v))
                    t = lanes.mark(lanes.tree)
                    lanes.on(lanes.main, self.ref(j, pd_rest))
                    e = lanes.mark(lanes.main)
                    lanes.wait(lanes.tile, e)
                    lanes.on(lanes.tile, P_[j].tiles(v))
                    lanes.wait(lanes.tile, t)
                    return lanes.mark(lanes.tile)
                return chain(call)

            for v in VARIANTS:
                out[f"pd_serial_{v}"] = serial(None, lambda j, v=v: P_[j].commit(v) + self.ref(j, pd_rest) + P_[j].tiles(v))
            for v in SIDE:
                out[f"pd_deferred_{v}"] = deferred(lambda j, v=v: P_[j].commit(v) + self.ref(j, pd_rest),
                                                   lambda j, v=v: P_[j].tiles(v))
                out[f"pd_side_tree_{v}" if self.mp == self.m16 else f"pd_side_tree_{v}_proposal"] = side_tree(v)
        return out


def run_shape(dev, pd, lanes, nv, out, name, m16, n, k, reps, trace):
    t0 = time.time()
    sh = Shape(dev, pd, name, m16, n, k)
    row = {"kind": "shape", "shape": name, "m16": m16, "mp": sh.mp, "n": n, "k": k, "tiles": sh.tiles,
           "depths": depths(m16, k), "setup_s": round(time.time() - t0, 2)}
    ok = True
    if pd is not None:
        from pd_host import check
        res = check(sh, python=not trace)
        ok = res["ok"]
        emit(out, {"kind": "check", "shape": name, "m16": m16, **res})
    graphs, nodes, seqs = {}, {}, {}
    for key, body in sh.schedules(lanes).items():
        if key.startswith("pd") and not ok:
            continue
        graphs[key], nodes[key], seqs[key] = lanes.capture(body)
    row["graph_nodes"] = nodes
    if trace:
        row["graph_kernels"] = seqs
    emit(out, row)
    if pd is not None and ok:
        from pd_host import check_graph
        failed = {key: check_graph(sh, lambda ge=ge: lanes.run(ge), partial=key.endswith("_proposal"))
                  for key, ge in graphs.items() if key != "ref_nohash"}
        emit(out, {"kind": "graph_check", "shape": name, "m16": m16, "ok": not any(failed.values()), "failed": failed})
        for key, bad in failed.items():
            if bad and key.startswith("pd"):
                dev.ck("cuGraphExecDestroy", graphs.pop(key))
    if not trace:
        t = timed_interleaved(dev, lanes, nv, graphs, reps)
        med = {key: statistics.median(v) for key, v in t["ms"].items()}
        per_call = {key: round(1e3 * v / CHAIN, 2) for key, v in med.items()}
        emit(out, {"kind": "timing", "shape": name, "m16": m16, "n": n, "k": k, "chain": CHAIN, "reps": reps,
                   "us_per_call_median": per_call,
                   "hashing_us_per_call": {key: round(v - per_call["ref_nohash"], 2) for key, v in per_call.items()},
                   "sm_mhz": t["sm_mhz"], "ms": t["ms"]})
        from pd_host import PD_STAMP, VARIANTS
        for v in (x for x in VARIANTS if x & PD_STAMP and f"pd_serial_{x}" in graphs):
            # the serial graph once more, its stamps cleared: the last call's pd_commit (ring set (CHAIN - 1) % 2)
            for c in sh.pdc:
                c.clear_stamps()
            dev.sync()
            lanes.run(graphs[f"pd_serial_{v}"])
            emit(out, {"kind": "stamps", "shape": name, "m16": m16, "variant": v, "schedule": f"pd_serial_{v}",
                       "clk": nv.read(), **sh.pdc[(CHAIN - 1) % 2].read_stamps()})
    else:
        order = []
        for _ in range(TRACE):
            for key, ge in graphs.items():
                lanes.run(ge)
                order.append(key)
        emit(out, {"kind": "trace_order", "shape": name, "m16": m16, "launches": order})
    for ge in graphs.values():
        dev.ck("cuGraphExecDestroy", ge)
    for c in sh.pdc:
        c.free()
    for p in sh.sets[1:]:
        for x in Shape.RING:
            dev.free(p.b[x])
    sh.p0.free()
    for p_ in (sh.xa, sh.xb, *sh.keys.values()):
        dev.free(p_)


#: pd_probe's forms (i1): the paired form (two warps) in the PD_PAIR variants alone
PROBE_FORMS = {0: "compress", 1: "rounds", 2: "paired"}


def probe(pd, nv, out) -> None:
    """One warp's cycles per chained compression (pd_probe_V) in each variant and form, at 1, 16 and 256 chained
    compressions, each launched twice: the first launch's cycles (at n = 1 the code's first fetch since the module loaded)
    and the second's; and whether every variant's final state is the same."""
    from pd_host import PD_PAIR, PAIR_SMEM, VARIANTS
    buf = pd.alloc(9 * 8)
    for n in (1, 16, 256):
        for form, fname in PROBE_FORMS.items():
            st = {}
            for v in (x for x in VARIANTS if form != 2 or x & PD_PAIR):
                cyc = []
                for _ in range(2):
                    if form == 2:
                        pd.launch(f"pd_probe_{v}", 1, 64, PAIR_SMEM, p=(buf,), i=(n, form))
                    else:
                        pd.launch(f"pd_probe_{v}", 1, 32, p=(buf,), i=(n, form))
                    pd.sync()
                    raw = pd.read(buf, 9 * 8)
                    cyc.append(int.from_bytes(raw[:8], "little"))
                st[v] = raw[8:72]
                emit(out, {"kind": "probe", "variant": v, "form": fname, "compressions": n, "cycles_first": cyc[0],
                           "cycles": cyc[1], "cycles_per_compression": round(cyc[1] / n, 1), "clk": nv.read()})
            emit(out, {"kind": "probe_states", "form": fname, "compressions": n, "agree": len(set(st.values())) == 1})
    pd.free(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pearl_cubin")
    ap.add_argument("pd_cubin", nargs="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--shapes", default=",".join(LLAMA))
    ap.add_argument("--rows", default="32,128")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = open(os.path.join(a.out, "bench.trace.jsonl" if a.trace else "bench.jsonl"), "a")
    image = open(a.pearl_cubin, "rb").read()
    dev = R.Dev(image)
    R.bind(dev.cu, GRAPH_API)
    nv = R.Nvml(dev.uuid)
    emit(out, {"kind": "env", "device": dev.name, "uuid": dev.uuid, "sms": dev.sms, "cc": list(dev.cc), "driver": dev.driver,
               "nvml_driver": nv.driver, "pearl_cubin_sha256": hashlib.sha256(image).hexdigest(),
               "pd_cubin_sha256": hashlib.sha256(open(a.pd_cubin, "rb").read()).hexdigest() if a.pd_cubin else None,
               "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "clk": nv.read(), "chain": CHAIN, "reps": a.reps,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not dev.name.startswith(R.NAME_PREFIX) or dev.cc != (12, 0) or dev.count != 1:
        print(f"refused: {dev.name} cc {dev.cc}, {dev.count} devices", file=sys.stderr)
        return 3
    dev.open()
    pd = None
    if a.pd_cubin:
        sys.path.insert(0, HERE)
        pd = Module(dev, open(a.pd_cubin, "rb").read())
        from pd_host import setup
        emit(out, {"kind": "pd_setup", "dyn_smem": setup(pd)})
        probe(pd, nv, out)
    lanes = Lanes(dev, [pd] if pd is not None else [])
    dev.stream = None
    for m16 in (int(x) for x in a.rows.split(",")):
        for name in a.shapes.split(","):
            n, k = LLAMA[name]
            run_shape(dev, pd, lanes, nv, out, name, m16, n, k, a.reps, a.trace)
    emit(out, {"kind": "end", "clk": nv.read(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
