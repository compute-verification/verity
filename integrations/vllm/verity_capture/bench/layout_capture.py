#!/usr/bin/env python3
"""R10 lane `layout` (owner 3), POD RUNNER: one serving run of the accepted SmolLM2 B1 C256 workload under CMT-3's complete acquisition
(`NativeHostCommitter`, fast ctypes D2D snapshots, GPU-resident SHA-256 tree) that

  * builds the L1 tree exactly as R9 (256-B chunks, R9's `hidden_gpu_tree` extension, 4-B tensor alignment by default), and OPTIONALLY
    (`--l2`) the L2-row tree (1,152-B leaves, `layout_gpu_tree.py`, same arena, tensor starts aligned to `--align` bytes) IN THE SAME
    STEP, on the same side stream, before the arena block is recycled -- a genuine prechallenge commitment with its own step/run roots
    and its own weights root (1,152-B `pos_leaf` chunks);
  * registers the weights under L1 (256-B pos_leaf, == CMT-3's `register_weights` root) and L2 (1,152-B), retaining subtree levels;
  * derives the challenge AFTER all compared roots are fixed (owner 6's formula; both layouts' roots in the preimage when --l2), samples
    B=1 / B=64 dispersed / B=64 shared-x GEMM coordinates (+ SiLU·mul scalar coordinates), opens them under every layout of the run
    with `layout_open.py`, writes owner-6-format sidecars and counters (measured opening time), and
  * dumps the retained arena bytes / roots / layouts / opened weight tensors so `layout_verify.py` (hashlib only) can recompute every
    root offline.

    PYTHONPATH=. HF_HOME=/workspace/hf CUDA_HOME=/usr/local/cuda HIDDEN_GPU_BUILD=... LAYOUT_GPU_BUILD=... VERITOR_REPO=. \
      python verity_capture/bench/layout_capture.py --out /workspace/cp/r10/layout/run_l1 --engine-arg gpu_memory_utilization=0.5
      python verity_capture/bench/layout_capture.py --out /workspace/cp/r10/layout/run_l1l2 --l2 --align 1152 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "vllm-poc", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity_vllm_adapter import profile as prof  # noqa: E402

prof.apply_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import layout_gpu_tree as LG  # noqa: E402
import layout_open as LO  # noqa: E402
import layout_verify as LV  # noqa: E402
from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.bench.commit_delta import assert_pristine, baseline_state, run_workload  # noqa: E402
from verity_capture.experimental.commit_integ.api import StepCommitment  # noqa: E402
from verity_capture.experimental.commit_integ.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, _dtype_name  # noqa: E402
from verity_capture.experimental.fa2_commit import engine as eng  # noqa: E402

PROGRAM_DIGEST = "5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9"   # pu-cat catalogue.json (active reference Program)
QUERY_ID_PREFIX = "17593554089537b1"                                                  # census query id prefix as pinned (full digest: owner 6)
L2_CHUNK = 1152


class LayoutCommitter(NativeHostCommitter):
    """CMT-3's committer with (a) configurable tensor-start alignment in the arena and (b) an optional L2-row tree per step."""

    def __init__(self, engine, *, run_id: str, align: int = 4, l2: bool = False, program_digest: bytes | None = None, arena_bytes: int = 512 << 20) -> None:
        super().__init__(engine, run_id=run_id, program_digest=program_digest, fast_copy=True, gpu_tree=True, arena_bytes=arena_bytes)
        self.align = int(align)
        self.l2 = bool(l2)
        self.name = f"layout_align{self.align}_" + ("L1L2" if l2 else "L1")
        self._l2_levels: dict[int, list] = {}
        self._l2_done: dict[int, dict] = {}
        self._used: dict[int, int] = {}
        self.gpu_ms = {"l1_leaf": 0.0, "l1_tree": 0.0, "l2_leaf": 0.0, "l2_tree": 0.0, "pad_keep": 0.0}
        self.step_stats: list[dict] = []
        self.run_root_l2: bytes | None = None
        self.weights_detail: dict[int, dict] = {}

    # -- acquisition: identical to the parent's fast path except for the alignment of tensor starts
    def _acquire_fast(self, name, t, job, t0):
        if self.align == 4:
            return super()._acquire_fast(name, t, job, t0)
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            return
        metas = job.metas
        first_leaf = (metas[-1].first_leaf + metas[-1].n_leaves) if metas else 0
        chunk = self.chunk
        dn = self._dtype_names.get(t.dtype)
        if dn is None:
            dn = self._dtype_names[t.dtype] = _dtype_name(t)
        from verity_capture.experimental.commit_integ.native_host import TensorMeta

        meta = TensorMeta(len(metas), name, dn, tuple(t.shape), nbytes, chunk, first_leaf, (nbytes + chunk - 1) // chunk)
        A = self.align
        off = -(-job.cursor // A) * A
        if off + nbytes > job.arena.numel():
            job.overflow.append((job.arena, job.cursor))
            job.arena = self.arena.take(max(nbytes, self.arena.block_bytes))
            job.arena_ptr = job.arena.data_ptr()
            off = 0
        stream = job.stream_handle
        if stream is None:
            stream = job.stream_handle = torch.cuda.current_stream().cuda_stream
        base = job.arena_ptr
        if base is None:
            base = job.arena_ptr = job.arena.data_ptr()
        if off > job.cursor:
            self._memset(base + job.cursor, 0, off - job.cursor, stream)
        rc = self._memcpy(base + off, t.data_ptr(), nbytes, 3, stream)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync rc={rc}")
        meta.dev_off = off
        meta.dev_block = job.arena
        job.cursor = off + nbytes
        metas.append(meta)
        job.acq_s += time.perf_counter() - t0
        job.acq_bytes += nbytes

    @staticmethod
    def layout_rows(metas, chunk: int) -> list:
        return [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes, chunk, m.first_leaf if chunk == GPU_CHUNK_BYTES else 0,
                 m.n_leaves if chunk == GPU_CHUNK_BYTES else -(-m.nbytes // chunk), m.stream_off] for m in metas]

    # -- worker: L1 tree (R9 kernel, unchanged) + optional L2 tree over the same arena bytes, both before the block is recycled
    def _commit_step_gpu(self, job) -> None:
        e = self._gpu_ext_load()
        side = self._side_stream()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(side):
            side.wait_event(job.event)
            blocks = job.overflow + [(job.arena, job.cursor)]
            if len(blocks) > 1:
                total = sum(u for _b, u in blocks)
                cat = self.arena.take(total)
                base = 0
                for blk, used in blocks:
                    if used:
                        cat[base:base + used].copy_(blk[:used])
                        for m in job.metas:
                            if m.dev_block is blk:
                                m.dev_off += base
                        base += used
                for blk, _u in blocks:
                    self.arena.release(blk)
                job.arena, job.cursor, job.overflow = cat, total, []
            used = job.cursor
            padded1 = -(-used // GPU_CHUNK_BYTES) * GPU_CHUNK_BYTES
            padded2 = -(-used // L2_CHUNK) * L2_CHUNK if self.l2 else 0
            padded = max(padded1, padded2)
            if padded > job.arena.numel():
                big = self.arena.take(padded)
                big[:used].copy_(job.arena[:used])
                self.arena.release(job.arena)
                job.arena = big
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record(side)
            if padded > used:
                job.arena[used:padded].zero_()
            for m in job.metas:
                m.stream_off = m.dev_off
                m.dev_block = None
            ev[1].record(side)
            leaves1 = e.chunk_leaves(job.arena[:padded1].view(torch.int32), GPU_CHUNK_BYTES // 4, job.step, self.GPU_TREE_SRC, job.step, padded1 // 4, 0)
            ev[2].record(side)
            levels1 = e.tree_levels(leaves1)
            ev[3].record(side)
            levels2 = None
            if self.l2:
                levels2 = LG.build_tree(job.arena[:padded2].view(torch.int32), L2_CHUNK // 4, job.step)
            ev[4].record(side)
            keep = torch.empty(padded, dtype=torch.uint8, device=job.arena.device)
            keep.copy_(job.arena[:padded])
            mr1_t = levels1[-1].to("cpu", non_blocking=True)
            mr2_t = levels2[-1].to("cpu", non_blocking=True) if levels2 is not None else None
            ev[5].record(side)
            done = torch.cuda.Event()
            done.record(side)
            done.synchronize()
            self.arena.release(job.arena)
        st = {"step": job.step, "used": used, "padded_l1": padded1, "padded_l2": padded2, "n_leaves_l1": int(leaves1.shape[0]),
              "n_leaves_l2": int(levels2[0].shape[0]) if levels2 is not None else None, "tensors": len(job.metas),
              "ms_pad": ev[0].elapsed_time(ev[1]), "ms_l1_leaf": ev[1].elapsed_time(ev[2]), "ms_l1_tree": ev[2].elapsed_time(ev[3]),
              "ms_l2_leaf_tree": ev[3].elapsed_time(ev[4]), "ms_keep": ev[4].elapsed_time(ev[5])}
        self.gpu_ms["l1_leaf"] += st["ms_l1_leaf"]
        self.gpu_ms["l1_tree"] += st["ms_l1_tree"]
        self.gpu_ms["l2_leaf"] += st["ms_l2_leaf_tree"]
        self.gpu_ms["pad_keep"] += st["ms_pad"] + st["ms_keep"]
        self.step_stats.append(st)
        mr1 = mr1_t.numpy().astype("uint32").tobytes()
        t1 = time.perf_counter()
        n1 = int(leaves1.shape[0])
        nbytes = sum(m.nbytes for m in job.metas)
        layout_dig = self.layout_digest(job.metas)
        root1 = eng.bind_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n1, mr1)
        if levels2 is not None:
            mr2 = mr2_t.numpy().astype("uint32").tobytes()
            n2 = int(levels2[0].shape[0])
            lay2 = self.layout_rows(job.metas, L2_CHUNK)
            ld2 = LV.layout_digest(lay2)
            root2 = LV.step_root(LV.ROOT_TAG_STEP_L2, self.program_digest, self._ctx_digest(job.step), self._geo_digest, ld2, n2, mr2)
            self._l2_done[job.step] = {"root": root2, "merkle_root": mr2, "n_leaves": n2, "layout_digest": ld2, "layout": lay2}
            self._l2_levels[job.step] = levels2
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes)
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        self._gpu_levels[job.step] = levels1
        self._gpu_blocks[job.step] = (keep, used)
        self._used[job.step] = used
        self._layouts[job.step] = job.metas
        job.arena = None
        self._done[job.step] = StepCommitment(job.step, root1, mr1, n1, nbytes, len(job.metas), layout_dig, job.t_step_end, t2, job.backlog,
                                              hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    def finalize(self):
        rc = super().finalize()
        t0 = time.perf_counter()
        if self.l2:
            roots2 = [self._l2_done[s]["root"] for s in range(len(rc.steps))]
            self.run_root_l2 = LV.run_root(LV.ROOT_TAG_RUN_L2, self.program_digest, self._geo_digest, roots2)
        rc.extra["l2_run_root"] = self.run_root_l2.hex() if self.run_root_l2 else None
        rc.extra["l2_finalize_s"] = time.perf_counter() - t0
        rc.extra["gpu_ms"] = dict(self.gpu_ms)
        rc.extra["step_stats"] = list(self.step_stats)
        return rc

    def reset(self) -> None:
        super().reset()
        self._l2_levels, self._l2_done, self._used = {}, {}, {}
        self.gpu_ms = {k: 0.0 for k in self.gpu_ms}
        self.step_stats = []
        self.run_root_l2 = None

    # -- weights: CMT-3's rule with retained levels (chunk 256 == register_weights root), and the L2 variant (chunk 1152)
    def register_weights_detail(self, chunk: int, tag: bytes) -> dict:
        t0 = time.perf_counter()
        seen: set[int] = set()
        tensors = []
        subroots: list[bytes] = []
        names = []
        n_bytes = 0
        for name, p in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if not isinstance(p, torch.Tensor) or id(p) in seen or p.numel() == 0:
                continue
            seen.add(id(p))
            src = p.detach().contiguous().reshape(-1)
            flat = src.view(torch.uint8) if src.dtype != torch.uint8 else src
            raw = flat.cpu().numpy()
            mv = memoryview(raw).cast("B")
            n = -(-len(mv) // chunk)
            leaves = [LV.leaf_hash("cmt-posleaf-v0", k, n, bytes(mv[k * chunk:(k + 1) * chunk])) for k in range(n)]
            levels = LV.fold_levels(leaves)
            subroots.append(levels[-1][0])
            n_bytes += len(mv)
            names.append([name, _dtype_name(p), list(p.shape), len(mv)])
            tensors.append(LO.WeightView(name, _dtype_name(p), list(p.shape), len(mv), levels[-1][0], n, raw, levels, len(tensors)))
        names_dig = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).digest()
        root = LV.weights_root(tag, self._geo_digest, names, subroots)
        d = {"root": root, "names": names, "names_digest": names_dig, "tensors": tensors, "outer_levels": LV.fold_levels(list(subroots)),
             "bytes": n_bytes, "wall_s": time.perf_counter() - t0, "chunk": chunk, "n_leaves": sum(t.n_leaves for t in tensors)}
        self.weights_detail[chunk] = d
        return d


def retained_view(com: LayoutCommitter, rc, layout_id: str, ctx: dict) -> LO.RetainedRun:
    chunk = LO.LAYOUTS[layout_id]
    steps = []
    for s in rc.steps:
        keep, used = com._gpu_blocks[s.step]
        if layout_id == "L1-chunk256":
            steps.append(LO.StepView(s.step, s.root, s.merkle_root, s.n_leaves, LayoutCommitter.layout_rows(com._layouts[s.step], 256), s.layout_digest, used, keep, com._gpu_levels[s.step]))
        else:
            d = com._l2_done[s.step]
            steps.append(LO.StepView(s.step, d["root"], d["merkle_root"], d["n_leaves"], d["layout"], d["layout_digest"], used, keep, com._l2_levels[s.step]))
    wd = com.weights_detail[chunk]
    run_root = rc.run_root if layout_id == "L1-chunk256" else com.run_root_l2
    return LO.RetainedRun(layout_id, chunk, dict(ctx), steps, run_root, wd["tensors"], wd["root"], wd["names_digest"], wd["outer_levels"], live=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align", type=int, default=4, help="tensor-start alignment in the arena (4 = CMT-3 as shipped; 1152 = row-aligned for L2)")
    ap.add_argument("--l2", action="store_true", help="also build the L2-row (1152-B leaf) tree per step + 1152-B weights root, same run")
    ap.add_argument("--engine-arg", action="append", default=None)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--rho", required=True, help="challenge-v1 verifier randomness (64 hex chars = 32 raw bytes), obtained AFTER the roots")
    ap.add_argument("--reps", type=int, default=3, help="opening-time repetitions (min reported; the first includes cold D2H)")
    ap.add_argument("--dump-all-weights", action="store_true")
    ap.add_argument("--hidden-so", default=None, help="CMT-2 tapped FA2 extension (.so); attaches CMT-3's HiddenM1Source (FA2 M1 hidden rows into the same arena)")
    ap.add_argument("--hidden-ns", default="verity_fa2_matReq", help="torch.ops namespace of the tapped extension")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "openings").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)
    utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    log = open(out / "capture.log", "a")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(f"[{utc()}] {s}\n")
        log.flush()

    wl = va.load_workload(a.workload)
    engine_args = dict(wl.get("engine_args_required") or {})
    engine_args.update(va.parse_engine_args(a.engine_arg))
    eng_ = va.build_engine(a.checkpoints, max_num_seqs=8, case=va.case_of(wl, a.case), engine_args=engine_args)
    say(f"engine built {eng_.build_s:.1f}s case {eng_.case} {eng_.cp['repo']} env {eng_.env_flags}")
    versions = va.versions_doc(eng_)
    versions["argv"] = sys.argv
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    base = baseline_state(eng_)
    reqs, sampling = wl["requests"], wl["sampling"]
    run_id = Path(a.workload).stem
    r = run_workload(eng_, va.throwaway_requests(eng_, reqs, seed=100), sampling)
    say(f"warm-up control {r['wall_s']:.3f}s")

    com = LayoutCommitter(eng_, run_id=run_id, align=a.align, l2=a.l2, program_digest=bytes.fromhex(PROGRAM_DIGEST))
    rows_label = "native rows only (no FA2 M1 source attached)"
    if a.hidden_so:
        # CMT-3's FA2 M1 hidden-row source (R9 hidden_source.py, unchanged): the tapped kernel's stream buffers are acquired into the SAME
        # step arena (through this committer's _acquire -> aligned _acquire_fast), so they are bound by both the L1 and the L2 step roots.
        from verity_capture.experimental.commit_integ.hidden_source import HiddenM1Source

        hs = HiddenM1Source(com, a.hidden_so, ns=a.hidden_ns)
        com.attach_source(hs)
        rows_label = f"native + FA2 M1 hidden rows ({a.hidden_ns}, tap mode {hs.mode} src {hs.src}, so sha256 {hashlib.sha256(Path(a.hidden_so).read_bytes()).hexdigest()[:16]})"
        say("hidden source attached:", rows_label)
    # R9-vs-this-lane kernel cross-check (CW=64 root equality on a random stream + hashlib)
    com._gpu_ext_load()
    say("layout_gpu_tree vs R9 hidden_gpu_tree (CW=64) and hashlib (CW=64, 288) root equality:", LG.check_against_r9(com._gpu_ext))
    w_ref = com.register_weights()                    # CMT-3's own function (host hashlib, chunk 256) for the root equality check
    w1 = com.register_weights_detail(256, LV.ROOT_TAG_WEIGHTS)
    say(f"weights L1: {w1['bytes']} B, {len(w1['tensors'])} tensors, {w1['n_leaves']} leaves, {w1['wall_s']:.2f}s; == CMT-3 register_weights root: {w1['root'].hex() == w_ref['root']} ({w_ref['wall_s']:.2f}s)")
    if a.l2:
        w2 = com.register_weights_detail(L2_CHUNK, LV.ROOT_TAG_WEIGHTS_L2)
        say(f"weights L2: {w2['n_leaves']} leaves, {w2['wall_s']:.2f}s, root {w2['root'].hex()[:16]}")
    # instrumented warm-up (sizes the arena pool), then THE run
    com.install()
    r = run_workload(eng_, va.throwaway_requests(eng_, reqs, seed=200), sampling)
    rc = com.finalize()
    com.uninstall()
    com.reset()
    assert_pristine(eng_, base)
    say(f"warm-up instrumented {r['wall_s']:.3f}s root-ready {rc.root_ready_delay_s * 1e3:.1f} ms errors {rc.extra.get('errors')}")
    torch.cuda.synchronize()
    com.install()
    t_run0 = time.perf_counter()
    r = run_workload(eng_, reqs, sampling)
    rc = com.finalize()
    t_roots = time.perf_counter()
    roots_utc = utc()
    com.uninstall()
    assert_pristine(eng_, base)
    say(f"RUN wall {r['wall_s'] * 1e3:.1f} ms, steps {len(rc.steps)}, leaves L1 {sum(s.n_leaves for s in rc.steps)}, bytes {sum(s.n_bytes for s in rc.steps)}, "
        f"root-ready {rc.root_ready_delay_s * 1e3:.2f} ms (L1) / {(t_roots - r['t_last_step']) * 1e3:.2f} ms (all roots), gpu_ms {rc.extra['gpu_ms']}, errors {rc.extra.get('errors')}")
    say(f"run_root L1 {rc.run_root.hex()}  L2 {rc.extra['l2_run_root']}  weights L1 {w1['root'].hex()}  L2 {(com.weights_detail.get(L2_CHUNK) or {}).get('root', b'').hex()}")
    # CMT-3's own verifier on 64 challenge openings of the L1 tree (the R9 acceptance check, unchanged)
    from verity_capture.experimental.commit_integ.api import challenge_positions

    smp = challenge_positions(rc.run_root, [s.n_leaves for s in rc.steps], 64)
    ops = com.open(smp)
    okv = sum(com.verify(rc, o) for o in ops)
    say(f"CMT-3 verify(): {okv}/64 L1 openings against the original run root")

    ctx = {"program_digest": PROGRAM_DIGEST, "query_id": QUERY_ID_PREFIX, "run_id": run_id, "geo_digest": com._geo_digest.hex(),
           "model": {"repo": eng_.cp.get("repo"), "revision": eng_.cp.get("revision")}, "vllm": versions.get("vllm"), "roots_written_utc": roots_utc,
           "align_bytes": a.align, "committer": com.name, "rows": rows_label, "gpu": torch.cuda.get_device_name(0)}
    layouts = ["L1-chunk256"] + (["L2-row"] if a.l2 else [])
    runs = {lid: retained_view(com, rc, lid, ctx) for lid in layouts}
    # challenge-v1 AFTER all roots (docs/protocol/challenge-v1.md): the co-committed L2 roots are FURTHER ROLES of the same instance
    # (`weights-l2`, `run-values-l2`, layout digest of the L2 descriptor); rho comes from --rho (verifier randomness, not derived here).
    from verity_protocol import challenge as chal_v1

    rho = bytes.fromhex(a.rho)
    l2_layout = chal_v1.layout_digest({**chal_v1.L1_DESCRIPTOR, "id": "L2-row", "chunk_bytes": L2_CHUNK}) if a.l2 else None
    extra = [("weights-l2", l2_layout, com.weights_detail[L2_CHUNK]["root"]), ("run-values-l2", l2_layout, com.run_root_l2)] if a.l2 else None
    batches = [("dispersed", 1), ("dispersed", a.B), ("shared_x", a.B), ("shared_w", a.B), ("scalar_dispersed", 1), ("scalar_dispersed", a.B), ("dispersed_qkv", a.B)]
    results = []
    for pattern, B in batches:
        base_run = runs["L1-chunk256"]
        pop_modules = ("qkv_proj",) if pattern == "dispersed_qkv" else tuple(LO.CALLS)
        population_size = sum(T * N for (_s, _l, _c, T, N) in LO.population(base_run, pop_modules))
        inst = LO.challenge_instance(base_run, B, population_size, extra)
        seed = chal_v1.challenge_seed(rho, inst)
        if pattern == "dispersed_qkv":
            coords = LO.sample_coords(base_run, seed, B, "dispersed", modules=("qkv_proj",))
        else:
            coords = LO.sample_coords(base_run, seed, B, pattern)
        ch = {"rule": "challenge-v1" + ("" if pattern.startswith("dispersed") else f" (seed) + NON-PROTOCOL pattern {pattern}"), "spec": "docs/protocol/challenge-v1.md",
              "rho_hex": rho.hex(), "seed_hex": seed.hex(), "instance": chal_v1.instance_to_json(inst),
              "indices": [c["rank_in_population"] for c in coords] if pattern.startswith("dispersed") else None,
              "derived_utc": utc(), "B": B, "roots_written_utc": roots_utc,
              "population": ("K576 {qkv_proj,o_proj,gate_up_proj} x steps x tokens x j (lexicographic; owner 6 rule int(SHA-256(c||u32be(k))) mod N)" if pattern.startswith("dispersed") and pattern != "dispersed_qkv"
                             else "K576 qkv_proj only (owner 6's population)" if pattern == "dispersed_qkv" else "same_instance: one (step,layer,call,token), B distinct j" if pattern == "shared_x" else "same_row: one (layer,call,j), B distinct (step,token)" if pattern == "shared_w" else "SiluMul elements (step,layer,token,i)")}
        for lid in layouts:
            kind = "scalar" if pattern == "scalar_dispersed" else "gemm"
            walls = []
            for rep in range(a.reps):
                torch.cuda.synchronize()
                try:
                    doc = LO.open_batch(runs[lid], coords, kind, ch, {"pattern": pattern, "B": B, "source": "live GPU levels, lazy per-leaf D2H (CMT-3 open())", "reps": a.reps})
                except KeyError as ex:
                    say(f"KeyError {ex!r}; step-0 tensor names: {[row[1] for row in runs[lid].steps[0].layout][:80]}")
                    raise
                walls.append(doc["counters"]["batch_wall_s"])
            doc["counters"]["batch_wall_s_reps"] = walls
            doc["counters"]["batch_wall_s_first"] = walls[0]
            doc["counters"]["batch_wall_s"] = min(walls)
            doc["l0_calculated"] = LO.l0_counters(runs[lid], doc)
            name = f"{lid}_{pattern}_B{B}"
            (out / "openings" / f"{name}.json").write_text(json.dumps(doc) + "\n")
            c = doc["counters"]
            row = {"layout": lid, "pattern": pattern, "B": B, **c, "l0": doc["l0_calculated"], "file": f"openings/{name}.json"}
            results.append(row)
            say(f"{name}: leaves {c['distinct_leaves']} nodes(mp) {c['path_nodes_multiproof']} nodes(naive) {c['path_nodes_naive']} "
                f"wbytes(mp) {c['witness_bytes_multiproof']} wbytes(naive) {c['witness_bytes_naive']} sha(mp) {c['sha_compressions_multiproof']} "
                f"open {c['batch_wall_s'] * 1e3:.2f} ms (first {c.get('batch_wall_s_first', c['batch_wall_s']) * 1e3:.2f}) | L0 calc leaves {doc['l0_calculated']['distinct_leaves']} nodes {doc['l0_calculated']['path_nodes_multiproof']}")
    # dumps for the offline verifier: arena bytes (used part), roots/layouts, opened weight tensors
    for lid in layouts:
        rr = runs[lid]
        meta = {"layout_id": lid, "chunk": rr.chunk, "context": ctx, "run_root": rr.run_root.hex(),
                "steps": [{"step": s.step, "root": s.root.hex(), "merkle_root": s.merkle_root.hex(), "n_leaves": s.n_leaves, "layout_digest": s.layout_digest.hex(),
                           "used": s.used, "layout": s.layout} for s in rr.steps],
                "weights": {"root": rr.weights_root.hex(), "names_digest": rr.names_digest.hex(),
                            "tensors": [{"name": w.name, "dtype": w.dtype, "shape": w.shape, "nbytes": w.nbytes, "subtree_root": w.subtree_root.hex(), "n_leaves": w.n_leaves} for w in rr.weights]},
                "weights_root": rr.weights_root.hex()}
        (out / f"roots_{lid}.json").write_text(json.dumps(meta) + "\n")
    for s in rc.steps:
        keep, used = com._gpu_blocks[s.step]
        (out / f"step_{s.step}.bin").write_bytes(keep[:used].cpu().numpy().tobytes())
    opened = set()
    for row in results:
        d = json.loads((out / row["file"]).read_text())
        for t in d["trees"]:
            if t["id"].startswith("weights:"):
                opened.add(t["id"][8:])
    for w in com.weights_detail[256]["tensors"]:
        if a.dump_all_weights or w.name in opened:
            (out / "weights" / (w.name + ".bin")).write_bytes(w.raw.tobytes())
    summary = {"utc": utc(), "committer": com.name, "align": a.align, "l2": a.l2, "gpu": torch.cuda.get_device_name(0), "run_wall_s": r["wall_s"],
               "steps": len(rc.steps), "step_stats": rc.extra["step_stats"], "gpu_ms_per_request": rc.extra["gpu_ms"],
               "root_ready_delay_l1_s": rc.root_ready_delay_s, "hidden_source": (dict(hs.stats, so=a.hidden_so, ns=a.hidden_ns) if a.hidden_so else None), "root_ready_delay_all_s": t_roots - r["t_last_step"],
               "leaves_l1": sum(s.n_leaves for s in rc.steps), "leaves_l2": sum(d["n_leaves"] for d in com._l2_done.values()) if a.l2 else None,
               "bytes_bound": sum(s.n_bytes for s in rc.steps), "arena_used": sum(com._used.values()),
               "weights": {str(k): {"root": v["root"].hex(), "bytes": v["bytes"], "leaves": v["n_leaves"], "wall_s": v["wall_s"], "tensors": len(v["tensors"])} for k, v in com.weights_detail.items()},
               "weights_cmt3_register_weights": w_ref, "cmt3_verify_64": okv, "run_root_l1": rc.run_root.hex(), "run_root_l2": rc.extra["l2_run_root"],
               "spans_total_s": rc.spans["total_s"], "errors": rc.extra.get("errors"), "tokens": r["generated_tokens"], "results": results, "ctx": ctx}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    say("wrote", out)
    com.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
