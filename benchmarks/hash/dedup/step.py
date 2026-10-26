"""Pearl-C's -h3 step with A's commitment taken from a Commit-style tree, against today's (`pearl-c-sm120-v1-h3`), on one
RTX PRO 6000: the dedup's measured upper bound.  A what-if: nothing here is served and no protocol changes.

  python benchmarks/hash/dedup/step.py --kernel benchmarks/pouw/pearl_c_sm120 --cubin SHIP/pearl_c_sm120.cubin --out OUT

The step is Llama-3.1-8B's 128 linear calls (32 layers of qkv, o, gate_up and down, on synthetic BF16 weights) made through
the served executor (`pouw_pearl_c_device.PearlCDeviceRun`, imported unchanged), each call's input the glue of the one
before (RMS norm, residual adds, silu(gate)·up), so every activation depends on the calls before it.  Decode (m = 32, 128)
is a whole forward captured as one CUDA graph with each call's tile hashing beside the next call (``--whole-step
--whole-defer`` as served); prefill (m = 2560, 8192) runs eagerly.

The arms (`ARMS`), each a call's steps under one `PearlCDeviceRun` subclass (`DedupRun`):
- ``h3``: today's call;
- ``pre``: A's own row digests and tree omitted (`A_STEPS`), and the call's ``tree_a`` and ``rowdig_a`` pointed at a
  Commit-style tree and its row digests over the same activation, made before the step (the upper bound: what is left on
  the critical path is seed_A's read of that root and the row seeds' read of those digests);
- ``inline``: the same, with the Commit-style tree made in the call, before seed_A (a combined serve whose Commit tree of x
  must exist before the seed that reads it);
- ``tree``: only A's tree from Commit, the row seeds still over A's own FP32 row digests;
- ``pre-od``, ``inline-od``: ``pre`` and ``inline`` on o and down alone, the linears whose input Commit commits today;
- ``nohash``: no hashing.

A Commit-style tree (`DedupRun._commit`) is Pearl-C's own A-tree kernels over the call's BF16 rows read as k/2 FP32 words:
``sha512/row-seg/v1`` digests of the BF16 rows, hm96-sha512 leaves under the prover's salt key at nonce `COMMIT_NONCE` + w,
in a frame-v3-sha512 tree under a Commit-style domain (`commit_domain`), one aligned subtree per call.

Before anything is timed (`check`), at each m, eagerly, from poisoned outputs: ``pre`` against ``inline`` (each call's A
words, y, seed_A, row seeds, E_A's lines, A root, row digests and tile leaves, and the pass's transcript and A roots), the
same for the o/down arms, a poisoned-Commit control that must differ, the Python references (`audit`'s hm96 tree,
`pearl_c`'s row digests, seeds and lines) in the h3, inline and tree arms on the first and last layer's calls at the first
decode m and on the first two calls at every other m, y against x·Wᵀ, and at decode
each arm's whole-graph replay against its eager pass, bit for bit.  Then the arms are timed interleaved, rep by rep, with
the SM clock read each rep.  Writes OUT/dedup.json and, under ``research run``, its result.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

#: Llama-3.1-8B's linears, (n, k), in a layer's order
ROLES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
ORDER = tuple(ROLES)
LAYERS = 32
#: the linears whose input Commit's serving rows commit today: attention's output (o) and silu(gate)·up (down)
OVERLAP = ("o", "down")
SCHEME = "pearl-c-sm120-v1-h3"
#: A's own commitment in a call: its pads, row digests, hm96 leaves and tree
A_STEPS = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a"})
#: arm: (hashing, steps a dedup call omits, its mode, the linears it applies to)
ARMS = {
    "nohash": (False, frozenset(), None, ()),
    "h3": (True, frozenset(), None, ()),
    "pre": (True, A_STEPS, "pre", ORDER),
    "inline": (True, A_STEPS, "inline", ORDER),
    "tree": (True, A_STEPS - {"rowdig_a"}, "tree", ORDER),
    "pre-od": (True, A_STEPS, "pre", OVERLAP),
    "inline-od": (True, A_STEPS, "inline", OVERLAP),
}
#: the Commit-style trees' hm96 salt nonce, plus the weight: below 2^62 and above any unit index
COMMIT_NONCE = 1 << 61
DECODE, PREFILL = (32, 128), (2560, 8192)
#: y against the float product: the scheme's tested bound (e2e.ERR)
ERR = 0.08


def commit_domain(w: int, m16: int):
    """The Commit-style domain of weight w's call of m16 rows: hm96-sha512 rows under a binding of the what-if's own."""
    from verity.ml.pouw.audit import HM96, rows_domain
    from verity.ml.pouw.protocol import tagged
    return rows_domain(tagged("verity/hash-dedup/what-if/commit-rows/v1", w), -1, m16, HM96)


def fp(t) -> list[int]:
    """Three wrapping int64 sums over t's bytes (zero-padded to 8): equal bytes give equal sums, for comparing buffers on the
    device without copying them out."""
    import torch
    v = t.contiguous().view(-1).view(torch.uint8)
    if v.numel() % 8:
        v = torch.cat([v, v.new_zeros(8 - v.numel() % 8)])
    x = v.view(torch.int64)
    i = torch.arange(x.numel(), device=x.device, dtype=torch.int64)
    return [int(x.sum()), int((x * (2 * i + 1)).sum()), int((x ^ (i * 0x9E3779B97F4A7C15 % (1 << 62))).sum())]


def run_class():
    from verity_vllm.protocols.pouw import pearl_c as PPD
    from verity_vllm.protocols.pouw.pearl_c.host import _put, _zeros
    from verity_vllm.protocols.pouw.pearl_c.screen import scatter_into

    class DedupRun(PPD.PearlCDeviceRun):
        """The served run with the arms' calls: `use` picks the arm; a dedup call omits `A_STEPS` (but ``rowdig_a`` under
        ``tree``) and its pipeline's ``tree_a`` (and ``rowdig_a``) are its weight's Commit-style buffers at its m16, which
        ``inline`` writes in the call (its steps first) and ``pre`` and ``tree`` read as left.  The o/down arms have buffers
        of their own (their activations differ from the full arms').  The pass's A root of a dedup call is the Commit-style
        root, in the eager record and in the graph's scatter alike.  The arm's mode and buffer set ride in the omitted steps
        (a ``dedup:<mode>:<set>`` name no step has), which key the executor's pipelines and graphs."""

        arm, mode, arm_omit, arm_roles, role, dump, keep = "h3", None, frozenset(), (), {}, None, frozenset()

        def setup(self, role: dict) -> None:
            self.role, self.cm, self.cx = role, {}, {}

        def use(self, arm: str) -> None:
            hashing, omit, mode, roles = ARMS[arm]
            self.arm, self.hashing, self.mode, self.arm_roles = arm, hashing, mode, roles
            self.arm_omit = omit | {f"dedup:{mode}:{'od' if roles == OVERLAP else 'all'}"} if mode else frozenset()

        def _x16(self, m16: int, k: int):
            import torch
            if (m16, k) not in self.cx:
                R = self.kernel.run
                mp = R.cdiv(m16, R.BM_D) * R.BM_D
                self.cx[(m16, k)] = (torch.zeros((mp, k), dtype=torch.bfloat16, device=self.device),
                                     _zeros(64 * (k // 2 // 256) * m16, self.device))
            return self.cx[(m16, k)]

        def _commit(self, w: int, sh, tag: str) -> SimpleNamespace:
            """Weight w's Commit-style buffers at sh's m16 in set `tag`, and the steps that write them from the shape's BF16
            rows."""
            key = (w, sh.m16, tag)
            if key not in self.cm:
                R, dev = self.kernel.run, self.device
                xbf, seg = self._x16(sh.m16, sh.k)
                size = R.Pipeline.sizes(sh.mp, R.BM, sh.k // 2, sh.m16)
                tree, rowdig, keys = _zeros(size["tree_a"], dev), _zeros(size["rowdig_a"], dev), _zeros(64 + 104, dev)
                dom = commit_domain(w, sh.m16)
                _put(keys, dom.domain_id + bytes(96) + (COMMIT_NONCE + w).to_bytes(8, "little"))
                pipe = PPD._pipeline(self.kernel, sh.mp, R.BM, sh.k // 2,
                                     {"tree_a": tree.data_ptr(), "rowdig_a": rowdig.data_ptr(), "segdig_a": seg.data_ptr()}, 0,
                                     {"dom_a": keys.data_ptr(), "unit": keys.data_ptr() + 64, **self.const_keys}, xbf.data_ptr(),
                                     sh.m16, variant=self.variant)
                steps = [f for _, f in pipe.tree_setup("a", sh.m16, "dom_a") + pipe.row_tree("a", "xa", sh.m16, "dom_a")]
                self.cm[key] = SimpleNamespace(tree=tree, rowdig=rowdig, keys=keys, pipe=pipe, steps=steps, xbf=xbf,
                                               size=size["tree_a"], nonce=COMMIT_NONCE + w)
            return self.cm[key]

        def poison_commit(self, m16: int, value: int = 0xA5) -> None:
            """The Commit-style trees and row digests at m16 filled with `value`, under inference mode as `poison` (they are
            made in a call).  Other m16's are left as their last inline pass wrote them, which their ``pre`` passes read."""
            import torch
            with torch.inference_mode():
                for (_, c_m16, _), c in self.cm.items():
                    if c_m16 == m16:
                        c.tree.fill_(value)
                        c.rowdig.fill_(value)

        def _gemm(self, w, x, weight, bias):
            omit, on = self.omit, self.hashing and self.mode is not None and self.role[w] in self.arm_roles
            self.omit = self.arm_omit if on else frozenset()
            try:
                if on and self.mode == "inline":
                    k = x.shape[-1]
                    x2 = x.reshape(-1, k)
                    m = x2.shape[0]
                    m16 = -(-m // PPD.ROW_ALIGN) * PPD.ROW_ALIGN
                    xbf = self._x16(m16, k)[0]
                    xbf[:m].copy_(x2)
                    if m16 > m:
                        xbf[m:m16].zero_()
                return super()._gemm(w, x, weight, bias)
            finally:
                self.omit = omit

        def _pipe(self, w, sh):
            key = (w, sh.m16, self.omit, self.hashing, self.schedule.name, self._graphed(sh.m16))
            fresh = key not in self.pipes
            pipe, main, aside = super()._pipe(w, sh)
            marker = next((x for x in self.omit if x.startswith("dedup:")), None) if self.hashing else None
            if fresh and marker:
                _, mode, tag = marker.split(":")
                c = self._commit(w, sh, tag)
                if c.size != pipe.size["tree_a"]:
                    raise RuntimeError(f"weight {w}: the Commit-style tree is {c.size} bytes and A's {pipe.size['tree_a']}")
                pipe.b["tree_a"] = c.tree.data_ptr()
                if mode != "tree":
                    pipe.b["rowdig_a"] = c.rowdig.data_ptr()
                pipe.dedup = (mode, c)
                if mode == "inline":
                    self.pipes[key] = (pipe, [*c.steps, *main], aside)
            return self.pipes[key]

        def _call_steps(self, pipe):
            d = getattr(pipe, "dedup", None)
            steps = super()._call_steps(pipe)
            return [*d[1].steps, *steps] if d and d[0] == "inline" else steps

        def _call_cuts(self, pipe):
            d = getattr(pipe, "dedup", None)
            a, b = super()._call_cuts(pipe)
            p = len(d[1].steps) if d and d[0] == "inline" else 0
            return a + p, b + p

        def _scatter(self, w, sh, pipe, g, tiles, flag_rows):
            d = getattr(pipe, "dedup", None)
            if d is None:
                return super()._scatter(w, sh, pipe, g, tiles, flag_rows)
            size, offs = pipe.size["tree_a"], self.offs[self.order[w]]
            root, flags, leaves = d[1].tree[size - self.nb:size], self.flag_buf.view(-1), g.leaves[:self.nb * tiles]

            def scatter():
                scatter_into(sh.xa, sh.b["alpha_a"], sh.b["aux_a"], flag_rows, flags, leaves, self.leaves, root, self.roots_buf, offs)
            return scatter

        def _after(self, cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias, screened=False, rows=None, scattered=False):
            out = super()._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias, screened=screened, rows=rows,
                                 scattered=scattered)
            d = getattr(pipe, "dedup", None)
            if d is not None and self.hashing and not scattered:
                size = pipe.size["tree_a"]
                self.roots_buf[self.nb * index:self.nb * (index + 1)].copy_(d[1].tree[size - self.nb:size])
            if self.dump is not None:
                self.dump.append(self._record(w, sh, pipe, index, m, tiles, y, d))
            return out

        def _record(self, w, sh, pipe, index, m, tiles, y, d) -> dict:
            """An eager call's buffers as its steps left them: fingerprints of each, and for the calls in `keep` their bytes."""
            from verity_vllm.protocols.pouw.pearl_c.host import _bytes
            rec = {"w": w, "index": index, "m": m, "m16": sh.m16, "mp": sh.mp, "mode": d[0] if d else None,
                   "fp": {"x": fp(sh.xa[:m]), "y": fp(y[:m])}}
            if not self.hashing:
                return rec
            nb, size = self.nb, pipe.size["tree_a"]
            tree = d[1].tree if d else sh.b["tree_a"]
            rowdig = d[1].rowdig if d and d[0] != "tree" else sh.b["rowdig_a"]
            t1 = self.cur.tiles
            bufs = {"seed_a": sh.b["seed_a"][:32], "seeds_a": sh.b["seeds_a"][:32 * sh.m16], "ea": sh.b["ea_codes"][:32 * sh.mp],
                    "root": tree[size - nb:size], "rowdig": rowdig[:64 * sh.m16], "leaves": self.leaves[nb * (t1 - tiles):nb * t1]}
            rec["fp"].update({name: fp(t) for name, t in bufs.items()})
            if w in self.keep:
                rec["bytes"] = {name: _bytes(t) for name, t in bufs.items() if name != "leaves"}
                rec["bytes"]["tree"] = _bytes(tree[:size])
                rec["bytes"]["f32"] = _bytes(sh.xa[:sh.m16])
                if d and d[0] == "inline":
                    rec["bytes"]["bf16"] = _bytes(d[1].xbf[:sh.m16].view(__import__("torch").uint8))
            return rec

    return DedupRun


def forward_of(run, W: dict, layers: int):
    """One step's 128 calls, each input the glue of the calls before."""
    import torch
    F = torch.nn.functional

    def rms(t):
        t = t.float()
        return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)

    def forward(x0):
        h = x0.float()
        for layer in range(layers):
            w = 4 * layer
            qkv = run.gemm(w, rms(h), W[w], None)
            h = h + run.gemm(w + 1, rms(qkv[:, :4096]), W[w + 1], None).float()
            gu = run.gemm(w + 2, rms(h), W[w + 2], None)
            h = h + run.gemm(w + 3, (F.silu(gu[:, :14336].float()) * gu[:, 14336:].float()).to(torch.bfloat16), W[w + 3], None).float()
        return h
    return forward


def clocks() -> dict:
    uuid = os.environ.get("E2E_GPU_UUID", "0")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv,noheader,nounits",
                            "-i", uuid], capture_output=True, text=True, timeout=10, check=False).stdout.strip().split(", ")
        return {"sm_mhz": int(q[0]), "mem_mhz": int(q[1]), "watts": float(q[2]), "temp_c": int(q[3])}
    except Exception as ex:  # noqa: BLE001 - a missing reading is recorded
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---- the check ------------------------------------------------------------------------------------------------------

def python_ref(run, rec: dict, fixture) -> dict:
    """One kept call against the Python references: its tree (h3: A's hm96 tree over its FP32 rows under A's domain;
    inline: the Commit-style tree over its BF16 rows), its row digests (`pearl_c.row_digest` of the rows its seeds read),
    seed_A over the root it read, each row's seed and E_A line, and the padding rows' lines from seed_A."""
    from verity.ml.pouw.audit import HM96, ROW_SCHEMA, activation_binding, rows_domain
    from verity.ml.pouw.schemes import pearl_c as C
    b, w, i, m16, mp, mode = rec["bytes"], rec["w"], rec["index"], rec["m16"], rec["mp"], rec["mode"]
    salt, root_b = run.salt, run.roots[w]
    k = len(b["f32"]) // (4 * m16)
    f32 = [b["f32"][4 * k * r:4 * k * (r + 1)] for r in range(m16)]
    out: dict = {"w": w, "index": i, "mode": mode}
    if mode == "inline":
        import numpy as np
        bf = [b["bf16"][2 * k * r:2 * k * (r + 1)] for r in range(m16)]
        words = np.frombuffer(b["f32"], "<u4")
        out["bf16_is_f32_high_halves"] = bool(np.array_equal(words >> 16, np.frombuffer(b["bf16"], "<u2")))
        out["f32_low_halves_zero"] = not bool((words & 0xFFFF).any())
        tree = fixture.tree_bytes(commit_domain(w, m16), bf, ROW_SCHEMA, run.salt_key, COMMIT_NONCE + w)
        digests = [C.row_digest(r) for r in bf]
    else:
        digests = [C.row_digest(r) for r in f32]
        tree = (fixture.tree_bytes(rows_domain(activation_binding(salt, i), i, m16, HM96), f32, ROW_SCHEMA, run.salt_key, i)
                if mode is None else None)
    root = tree[-64:] if tree is not None else b["root"]
    if tree is not None:
        out["tree"] = b["tree"] == tree
    out["root_read"] = b["root"] == root
    out["row_digests"] = b["rowdig"] == b"".join(digests)
    seed_a = C.unit_seed_a(salt, root, root_b, i, "h3")
    out["seed_a"] = b["seed_a"] == seed_a
    seeds = [C.row_seed(salt, d, root_b, i, r, "h3") for r, d in enumerate(digests)]
    out["row_seeds"] = b["seeds_a"] == b"".join(seeds)
    ea = b["ea"]
    out["ea_rows"] = all(ea[32 * r:32 * r + 32] == bytes(C.sample_line(s, 0, 0, 0)) for r, s in enumerate(seeds))
    out["ea_padding"] = all(ea[32 * r:32 * r + 32] == bytes(C.sample_line(C.S512Seed(seed_a), 0, 0, r)) for r in range(m16, mp))
    out["ok"] = all(v for key, v in out.items() if key not in ("w", "index", "mode"))
    return out


def eager_pass(run, forward, x0, arm: str, poison: bool = True) -> dict:
    run.use(arm)
    run.graphs = False
    run.begin()
    if poison:
        run.poison()
    run.dump = []
    try:
        h = forward(x0)
        end = run.end()
    finally:
        calls, run.dump = run.dump, None
    return {"h": fp(h), "end": {k: end.get(k) for k in ("transcript_root", "activation_roots", "units", "tiles")}, "calls": calls}


def same_calls(a: dict, b: dict) -> dict:
    bad = [i for i, (x, y) in enumerate(zip(a["calls"], b["calls"], strict=True)) if x["fp"] != y["fp"]]
    first = None
    if bad:
        x, y = a["calls"][bad[0]], b["calls"][bad[0]]
        first = {"call": bad[0], "w": x["w"], "differ": sorted(n for n in x["fp"] if x["fp"][n] != y["fp"].get(n))}
    return {"calls_equal": not bad, "calls_differing": len(bad), "first": first, "end_equal": a["end"] == b["end"],
            "h_equal": a["h"] == b["h"]}


def check(run, forward, x0, W: dict, fixture, m: int, keep: frozenset) -> dict:
    """The check at m, before any timing (module docstring)."""
    import torch
    from verity_vllm.protocols.pouw.pearl_c import ROW_ALIGN
    out: dict = {"m": m}
    m16 = -(-m // ROW_ALIGN) * ROW_ALIGN
    passes = {}
    run.poison_commit(m16)
    for arm in ("inline", "pre", "inline-od", "pre-od", "h3", "tree"):
        run.keep = keep if arm in ("inline", "h3", "tree") else frozenset()
        passes[arm] = eager_pass(run, forward, x0, arm)
        if arm == "pre-od":
            run.keep = frozenset()
            passes["inline-again"] = eager_pass(run, forward, x0, "inline")
    run.keep = frozenset()
    run.poison_commit(m16)
    passes["pre-poisoned"] = eager_pass(run, forward, x0, "pre")
    passes["inline-restore"] = eager_pass(run, forward, x0, "inline")
    passes["inline-od-restore"] = eager_pass(run, forward, x0, "inline-od")
    out["pre_vs_inline"] = same_calls(passes["pre"], passes["inline"])
    out["pre_od_vs_inline_od"] = same_calls(passes["pre-od"], passes["inline-od"])
    out["inline_repeats"] = same_calls(passes["inline-again"], passes["inline"])
    po = passes["pre-poisoned"]
    out["poisoned_commit_differs"] = {"seed_a_differs_at_call0": po["calls"][0]["fp"]["seed_a"] != passes["inline"]["calls"][0]["fp"]["seed_a"],
                                      "transcript_differs": po["end"]["transcript_root"] != passes["inline"]["end"]["transcript_root"]}
    out["dedup_differs_from_h3"] = passes["inline"]["end"]["transcript_root"] != passes["h3"]["end"]["transcript_root"]
    out["python"] = [python_ref(run, rec, fixture) for arm in ("h3", "inline", "tree") for rec in passes[arm]["calls"] if "bytes" in rec]
    errs = {}
    x = x0.float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
    ref = x.float() @ W[0].float().T
    for arm in ("h3", "inline", "pre", "tree"):
        run.use(arm)
        run.graphs = False
        run.begin()
        y = run.gemm(0, x, W[0], None)
        errs[arm] = float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
        run.end()
    out["first_call_rel_rms"] = errs
    out["ok"] = bool(out["pre_vs_inline"]["calls_equal"] and out["pre_vs_inline"]["end_equal"] and out["pre_vs_inline"]["h_equal"]
                     and out["pre_od_vs_inline_od"]["calls_equal"] and out["pre_od_vs_inline_od"]["end_equal"]
                     and out["inline_repeats"]["calls_equal"] and out["inline_repeats"]["end_equal"]
                     and all(out["poisoned_commit_differs"].values()) and out["dedup_differs_from_h3"]
                     and out["python"] and all(p["ok"] for p in out["python"]) and all(e < ERR for e in errs.values()))
    return out


# ---- decode: whole-forward graphs -----------------------------------------------------------------------------------

def capture(run, forward, x0, arm: str):
    """The arm's decode forward captured whole (as served: `whole_capture_begin`, a warm-up forward outside the capture that
    makes each call's graph, then the capture)."""
    import torch
    run.use(arm)
    run.graphs = True
    run.begin()
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=run.device)
    run.whole_capture_begin()
    try:
        forward(x0)
        side.wait_stream(torch.cuda.current_stream(run.device))
        with torch.cuda.graph(graph, stream=side):
            out = forward(x0)
    finally:
        calls = run.whole_capture_end()
    run.end()
    torch.cuda.synchronize()
    return SimpleNamespace(graph=graph, calls=calls, out=out)


def replay(run, g, arm: str, steps: int, poison: bool = False) -> dict:
    import torch
    run.use(arm)
    run.graphs = True
    run.begin()
    if poison:
        run.poison()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    a.record()
    for _ in range(steps):
        run.whole_replay(g.calls)
        g.graph.replay()
        run.whole_replayed(g.calls)
    b.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    end = run.end()
    torch.cuda.synchronize()
    wall_end = time.perf_counter() - t0
    return {"gpu_ms": a.elapsed_time(b) / steps, "wall_ms": 1e3 * wall / steps, "end_ms": 1e3 * (wall_end - wall),
            "end": {k: end.get(k) for k in ("transcript_root", "activation_roots", "units", "tiles")}}


def prefill_rep(run, forward, x0, arm: str) -> dict:
    import torch
    run.use(arm)
    run.graphs = False
    run.begin()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    forward(x0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    run.end()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return {"ms": 1e3 * (t1 - t0), "end_ms": 1e3 * (t2 - t1)}


def rotated(arms: list[str], rep: int) -> list[str]:
    r = rep % len(arms)
    return arms[r:] + arms[:r]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernel", required=True, help="GPU 1's kernel directory (run.py, fixture.py)")
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "dedup.json")
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--decode", default=",".join(map(str, DECODE)))
    ap.add_argument("--prefill", default=",".join(map(str, PREFILL)))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=32, help="decode steps a timed rep replays")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    import torch
    from verity_vllm.protocols.pouw import pearl_c as PPD

    from verity.ml.pouw.audit import weights_root
    from verity.core.protocols.pouw import verifier
    from verity.ml.pouw.coins import Challenger

    t_start = time.perf_counter()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20261018)
    W, role = {}, {}
    for layer in range(args.layers):
        for j, name in enumerate(ORDER):
            n, k = ROLES[name]
            W[4 * layer + j] = (0.02 * torch.randn(n, k, device=dev, generator=gen)).to(torch.bfloat16)
            role[4 * layer + j] = name
    bad = PPD.refusals(SCHEME, {w: tuple(t.shape) for w, t in W.items()})
    if bad:
        raise SystemExit("; ".join(bad))
    kernel = PPD.Kernel.load(args.kernel, args.cubin)
    salt_key = PPD.new_salt_key(SCHEME)
    roots = PPD.weight_roots(kernel, W, SCHEME, salt_key=salt_key)
    challenger = Challenger(0, verifier.coins)
    opening = challenger.open("salt", weights_root(roots))
    DedupRun = run_class()
    run = DedupRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, W, roots, challenger.commitment, opening, graphs=True,
                   whole_defer=True, salt_key=salt_key)
    run.setup(role)
    forward = forward_of(run, W, args.layers)
    last = 4 * (args.layers - 1)
    keep = frozenset({0, 1, 2, 3, last, last + 1, last + 2, last + 3})
    arms = [a for a in args.arms.split(",") if a]
    decode = [int(x) for x in args.decode.split(",") if x]
    prefill = [int(x) for x in args.prefill.split(",") if x]
    doc: dict = {"schema": "verity-hash/dedup-step/v1", "run": os.environ.get("RESEARCH_RUN_ID"),
                 "source": os.environ.get("RESEARCH_SOURCE_SHA"), "scheme": SCHEME, "cubin_sha256": kernel.cubin_sha256,
                 "layers": args.layers, "arms": {a: {"hashing": ARMS[a][0], "omit": sorted(ARMS[a][1]), "mode": ARMS[a][2],
                                                     "roles": list(ARMS[a][3])} for a in arms},
                 "install_s": time.perf_counter() - t_start, "checks": {}, "graph_checks": {}, "timing": {}}

    def save():
        args.out.write_text(json.dumps(doc, indent=1) + "\n")

    xs = {m: torch.randn(m, 4096, device=dev, generator=gen).to(torch.bfloat16) for m in decode + prefill}
    for m in decode + prefill:
        t0 = time.perf_counter()
        kept = keep if m in decode[:1] else frozenset({0, 1})
        doc["checks"][str(m)] = c = check(run, forward, xs[m], W, kernel.fixture, m, kept)
        c["s"] = time.perf_counter() - t0
        print(f"dedup: check m={m}: {'ok' if c['ok'] else 'FAIL'} ({c['s']:.0f} s)", flush=True)
        save()
    graphs = {}
    for m in decode:
        eager = {a: eager_pass(run, forward, xs[m], a) for a in arms}
        gc = doc["graph_checks"][str(m)] = {}
        for a in arms:
            graphs[(m, a)] = g = capture(run, forward, xs[m], a)
            r = replay(run, g, a, 1, poison=True)
            gc[a] = {"replay_equals_eager": r["end"] == eager[a]["end"] and fp(g.out) == eager[a]["h"], "calls": len(g.calls),
                     "end": r["end"]}
        print(f"dedup: graphs m={m}: {sum(v['replay_equals_eager'] for v in gc.values())}/{len(gc)} replay = eager", flush=True)
        save()
    ok = all(c["ok"] for c in doc["checks"].values()) and all(v["replay_equals_eager"] for gc in doc["graph_checks"].values()
                                                            for v in gc.values())
    doc["checks_ok"] = ok
    save()
    if not ok or args.check_only:
        print(f"dedup: checks {'passed' if ok else 'FAILED'}; {'no timing' if not ok else 'check only'}", flush=True)
        return 0 if ok else 1
    for m in decode:
        reps = {a: [] for a in arms}
        for rep in range(args.reps + 1):
            for a in rotated(arms, rep):
                r = replay(run, graphs[(m, a)], a, args.steps)
                r["clock"] = clocks()
                if rep:
                    reps[a].append(r)
        doc["timing"][f"decode_m{m}"] = {a: {"step_ms": statistics.median(r["wall_ms"] for r in rs),
                                              "gpu_step_ms": statistics.median(r["gpu_ms"] for r in rs),
                                              "end_ms": statistics.median(r["end_ms"] for r in rs), "reps": rs} for a, rs in reps.items()}
        line = ", ".join(f"{a} {v['step_ms']:.2f}" for a, v in doc["timing"][f"decode_m{m}"].items())
        print(f"dedup: decode m={m}: {line}", flush=True)
        save()
    for m in prefill:
        reps = {a: [] for a in arms}
        for rep in range(args.reps + 1):
            for a in rotated(arms, rep):
                r = prefill_rep(run, forward, xs[m], a)
                r["clock"] = clocks()
                if rep:
                    reps[a].append(r)
        doc["timing"][f"prefill_m{m}"] = {a: {"pass_ms": statistics.median(r["ms"] for r in rs),
                                               "end_ms": statistics.median(r["end_ms"] for r in rs), "reps": rs} for a, rs in reps.items()}
        line = ", ".join(f"{a} {v['pass_ms']:.1f}" for a, v in doc["timing"][f"prefill_m{m}"].items())
        print(f"dedup: prefill m={m}: {line}", flush=True)
        save()
    doc["total_s"] = time.perf_counter() - t_start
    save()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pouw"))
    import run_result
    ms = [(f"{ph}.{a}", v.get("step_ms", v.get("pass_ms")), "ms") for ph, t in doc["timing"].items() for a, v in t.items()]
    run_result.write(args.out, ok, "dedup step: pre = inline bit for bit, Python references, poisoned control, graph replay = eager",
                     json.dumps({ph: {a: round(v.get("step_ms", v.get("pass_ms")), 3) for a, v in t.items()} for ph, t in doc["timing"].items()}),
                     ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
