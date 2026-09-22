"""acq-life (R10 owner 2): SAFE-BATCHING CANDIDATE — packed copies at the legal collection points of lifetimes.md.

DIAGNOSTIC / candidate code (not the headline path).  Subclasses the frozen R9 reference committer (`NativeHostCommitter`, fast path
+ arena + GPU tree) and replaces "one cudaMemcpyAsync per acquired tensor" (365/step) with "one gather-copy launch per collection point":

  * a hook no longer copies; it records (src_ptr, arena_dst_ptr, nbytes) in a pending table and HOLDS the Tensor reference (a held
    reference stops allocator block reuse — it does NOT stop in-place writers, which is why the flush points below are forced);
  * the pending table is flushed (one pinned H2D of the pointer table + ONE kernel launch on the compute stream) at the three forced
    points per layer from lifetimes.md — P0 = qkv_proj hook (RoPE rewrites q,k in place as the very next kernel), P1 = o_proj hook
    (the post-attention fused norm writes into o_proj's buffer next), P2 = down_proj hook (the next layer's input fused norm writes into
    down_proj's buffer next) — and at step end (P3);
  * everything is on the SAME stream as the producers (stream 0), so the flush enqueued before the overwriting kernel is stream-ordered
    ahead of it: no event is needed for the producer side; the consumer side (worker / GPU tree) is unchanged from R9 (step event).

Arena layout, offsets, alignment, TensorMeta and the tree are byte-identical to the reference => the step roots must be IDENTICAL to
the frozen path for the same deterministic run (that is the correctness gate), and 64/64 openings must verify.

`--violate` skips the P0 flush (qkv/normed/residual copied at P1 instead — after RoPE ran): the roots must DIFFER from the reference and
the hook-time value check must fail LOUDLY.  That is the deliberate late-read stress test.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from verity_vllm_adapter import profile as prof  # noqa: F401  (pins the vLLM env exactly like commit_delta)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.bench.commit_delta import assert_pristine, baseline_state, run_workload  # noqa: E402
from verity_capture.experimental.commit_integ.api import challenge_positions  # noqa: E402
from verity_capture.experimental.commit_integ.native_host import NativeHostCommitter, TensorMeta, _dtype_name  # noqa: E402

SCHEMA = "acqlife_packed/v1"

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// one segment per blockIdx.y; grid.x blocks stride over the segment.  Byte-exact copy (memcpy semantics) with 16-B / 4-B fast paths.
__global__ void gather_copy_kernel(const int64_t* __restrict__ tab, int nseg) {
  int seg = blockIdx.y;
  if (seg >= nseg) return;
  const char* src = reinterpret_cast<const char*>(tab[seg * 3 + 0]);
  char* dst = reinterpret_cast<char*>(tab[seg * 3 + 1]);
  int64_t n = tab[seg * 3 + 2];
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  uintptr_t a = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) | (uintptr_t)n;
  if ((a & 15) == 0) {
    const uint4* s = reinterpret_cast<const uint4*>(src); uint4* d = reinterpret_cast<uint4*>(dst);
    for (long long j = i; j < n / 16; j += stride) d[j] = s[j];
  } else if ((a & 3) == 0) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src); uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    for (long long j = i; j < n / 4; j += stride) d[j] = s[j];
  } else {
    for (long long j = i; j < n; j += stride) dst[j] = src[j];
  }
}

void gather_copy(torch::Tensor tab, int64_t nseg, int64_t max_bytes) {
  TORCH_CHECK(tab.is_cuda() && tab.scalar_type() == torch::kInt64, "tab must be int64 cuda");
  int bx = (int)std::min<int64_t>(32, (max_bytes / 16 + 255) / 256 + 1);
  dim3 grid(bx, (unsigned)nseg);
  gather_copy_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(tab.data_ptr<int64_t>(), (int)nseg);
}
"""


def load_ext(build_dir: str):
    from torch.utils.cpp_extension import load_inline

    os.makedirs(build_dir, exist_ok=True)
    return load_inline(name="acqlife_gather_copy", cpp_sources="void gather_copy(torch::Tensor tab, int64_t nseg, int64_t max_bytes);",
                       cuda_sources=CUDA_SRC, functions=["gather_copy"], build_directory=build_dir, verbose=False)


class PackedCommitter(NativeHostCommitter):
    """Reference committer with the per-tensor copies replaced by packed gather copies at the legal collection points."""

    name = "native_host_packed"
    FLUSH_MODULES = ("qkv_proj", "o_proj", "down_proj")      # P0, P1, P2 of lifetimes.md (forced by zero-window in-place overwriters)
    RING_SLOTS, SLOT_ENTRIES = 256, 1024                       # pinned pointer-table ring: >= 2 steps of flushes (91/step); a slot is reused
    #                                                            only after the step that used it has been synchronised by the runner's D2H

    def __init__(self, *a, build_dir: str, violate: bool = False, **kw) -> None:
        super().__init__(*a, **kw)
        self.ext = load_ext(build_dir)
        self.violate = violate
        self._pend: list[tuple[int, int, int]] = []
        self._refs: list[torch.Tensor] = []
        self._tab_host = torch.empty((self.RING_SLOTS, self.SLOT_ENTRIES, 3), dtype=torch.int64, pin_memory=True)
        self._tab_np = self._tab_host.numpy()
        self._tab_dev = torch.empty((self.RING_SLOTS, self.SLOT_ENTRIES, 3), dtype=torch.int64, device="cuda")
        self._slot = 0
        self.stats = {"flushes": 0, "segments": 0, "bytes": 0, "flush_s": 0.0, "acq_s": 0.0, "max_segments": 0, "noncontig_rows": 0, "contig_fallback": 0}

    # ---- acquisition: reference bookkeeping, deferred copy --------------------------------------------------------------------------
    def _acquire(self, name: str, t: torch.Tensor, job) -> None:
        t0 = time.perf_counter()
        if t.device.type != "cuda":
            return
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            return
        metas = job.metas
        first_leaf = (metas[-1].first_leaf + metas[-1].n_leaves) if metas else 0
        chunk = self.chunk
        dn = self._dtype_names.get(t.dtype)
        if dn is None:
            dn = self._dtype_names[t.dtype] = _dtype_name(t)
        meta = TensorMeta(len(metas), name, dn, tuple(t.shape), nbytes, chunk, first_leaf, (nbytes + chunk - 1) // chunk)
        off = (job.cursor + 3) & ~3
        if off + nbytes > job.arena.numel():
            self._flush(job)                                    # the pending segments target the old block: flush before switching
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
            self._memset(base + job.cursor, 0, off - job.cursor, stream)     # same gap rule as the reference (byte-identical arena)
        dst = base + off
        if t.is_contiguous():
            self._pend.append((t.data_ptr(), dst, nbytes))
            self._refs.append(t)
        elif t.dim() == 2 and t.stride(1) == 1:
            # prefill RoPE q/k views (rows contiguous, row stride 960 elements): one segment per row => the arena receives the
            # row-major bytes exactly as the reference's `.contiguous()` copy would produce
            rowb = t.shape[1] * t.element_size()
            sb = t.stride(0) * t.element_size()
            src0 = t.data_ptr()
            self._pend.extend((src0 + r * sb, dst + r * rowb, rowb) for r in range(t.shape[0]))
            self._refs.append(t)
            self.stats["noncontig_rows"] += t.shape[0]
        else:
            c = t.contiguous()                                  # rare fallback (a device copy on the compute stream; counted)
            self._pend.append((c.data_ptr(), dst, nbytes))
            self._refs.append(c)
            self.stats["contig_fallback"] += 1
        meta.dev_off = off
        meta.dev_block = job.arena
        job.cursor = off + nbytes
        metas.append(meta)
        job.acq_bytes += nbytes
        short = name.rsplit("/", 1)[0].rsplit(".", 1)[-1]
        if short in self.FLUSH_MODULES and not (self.violate and short == "qkv_proj"):
            self._flush(job)
        dt = time.perf_counter() - t0
        job.acq_s += dt
        self.stats["acq_s"] += dt

    def _flush(self, job) -> None:
        pend = self._pend
        if not pend:
            return
        t0 = time.perf_counter()
        n = len(pend)
        if n > self.SLOT_ENTRIES:                               # more segments than a slot holds: split (never happens on this workload)
            self._pend = pend[self.SLOT_ENTRIES:]
            pend = pend[:self.SLOT_ENTRIES]
            n = len(pend)
            self._flush_now(job, pend, n, t0)
            return self._flush(job)
        self._flush_now(job, pend, n, t0)

    def _flush_now(self, job, pend, n: int, t0: float) -> None:
        s = self._slot
        self._slot = (s + 1) % self.RING_SLOTS
        arr = np.asarray(pend, dtype=np.int64)
        self._tab_np[s, :n] = arr
        self._tab_dev[s, :n].copy_(self._tab_host[s, :n], non_blocking=True)     # H2D of the pointer table (stream 0)
        self.ext.gather_copy(self._tab_dev[s], n, int(arr[:, 2].max()))         # ONE launch: all pending segments -> arena
        self.stats["flushes"] += 1
        self.stats["segments"] += n
        self.stats["max_segments"] = max(self.stats["max_segments"], n)
        self.stats["bytes"] += int(arr[:, 2].sum())
        self._pend = []
        self._refs.clear()                                      # references released only after the copy is enqueued (stream-ordered)
        self.stats["flush_s"] += time.perf_counter() - t0

    def _end_step(self) -> None:
        job = self._cur
        if job is not None:
            self._flush(job)                                    # P3: everything still pending (final norm, logits, sampled ids)
        super()._end_step()


class ValueSnapshots:
    """Hook-time clones of a few classes at the point the packed committer defers them (truth for the late-read stress test).
    Installed AFTER the committer so its hooks run second; clones are taken on the compute stream => the bits at hook time."""

    def __init__(self, model: torch.nn.Module, committer: PackedCommitter, layers: tuple[int, ...] = (0, 7, 15, 29)) -> None:
        self.com = committer
        self.model = model
        self.layers = layers
        self.h = []
        self.snap: dict[tuple[int, str], torch.Tensor] = {}
        self.step = -1

    def install(self) -> None:
        for name, m in self.model.named_modules():
            parts = name.split(".")
            if "layers" in parts and parts[-1] == "qkv_proj" and int(parts[parts.index("layers") + 1]) in self.layers:
                self.h.append(m.register_forward_hook(self._post(name)))

    def _post(self, name: str):
        def post(module, args, output):
            job = self.com._cur
            if job is None:
                return None
            if isinstance(output, torch.Tensor):
                self.snap[(job.step, name)] = output.detach().clone()
            elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                self.snap[(job.step, f"{name}/0")] = output[0].detach().clone()      # vLLM Linear returns (output, bias): committer names it `<module>/0`
            return None
        return post

    def uninstall(self) -> None:
        for h in self.h:
            h.remove()
        self.h = []

    def compare(self) -> dict:
        """Compare every snapshot with the bytes the committer retained for that tensor (GPU tree: `_gpu_blocks[step]` at meta.stream_off)."""
        ok = bad = 0
        first_bad = None
        for (step, name), t in self.snap.items():
            metas = self.com._layouts.get(step)
            blk = self.com._gpu_blocks.get(step)
            if metas is None or blk is None:
                continue
            m = next((m for m in metas if m.name == name), None)
            if m is None:
                continue
            got = blk[0][m.stream_off:m.stream_off + m.nbytes]
            want = t.reshape(-1).view(torch.uint8)
            if torch.equal(got, want):
                ok += 1
            else:
                bad += 1
                if first_bad is None:
                    first_bad = {"step": step, "tensor": name, "bytes_differ": int((got != want).sum())}
        return {"compared": ok + bad, "equal": ok, "differ": bad, "first_differ": first_bad}


def run(engine: va.Engine, requests: list[dict], sampling: dict) -> dict:
    return run_workload(engine, requests, sampling)      # the R9 serving loop, unchanged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/workspace/cp/acqlife/gather_build")
    ap.add_argument("--hidden-so", default=None)
    ap.add_argument("--reps", type=int, default=5, help="diagnostic timing reps per arm (NOT the headline three-arm measurement)")
    ap.add_argument("--openings", type=int, default=64)
    ap.add_argument("--engine-arg", action="append", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload)
    engine_args = dict(wl.get("engine_args_required") or {})
    engine_args.update(va.parse_engine_args(args.engine_arg))
    eng = va.build_engine(args.checkpoints, max_num_seqs=8, case=va.case_of(wl, args.case), engine_args=engine_args)
    reqs, sampling = wl["requests"], wl["sampling"]
    base = baseline_state(eng)
    for i in range(2):
        run(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling)

    def mk(kind: str):
        kw = dict(run_id="acqlife", chunk=256, hash_threads=2, staging="arena", arena_bytes=512 << 20, fast_copy=True, gpu_tree=True)
        if kind == "reference":
            c = NativeHostCommitter(eng, **kw)
        else:
            c = PackedCommitter(eng, build_dir=args.build_dir, violate=(kind == "violate"), **kw)
        c._gpu_ext_load()
        if args.hidden_so:
            from verity_capture.experimental.commit_integ.hidden_source import HiddenM1Source

            c.attach_source(HiddenM1Source(c, args.hidden_so))
        c.register_weights()
        return c

    results: dict = {"schema": SCHEMA, "label": "diagnostic on cp-acq2 — owner 1 owns the headline three-arm measurement", "arms": {}}
    roots: dict[str, list[str]] = {}
    tokens: dict[str, list] = {}
    for kind in ("reference", "packed", "violate"):
        com = mk(kind)
        snaps = ValueSnapshots(com.model, com) if kind != "reference" else None
        com.install()
        if snaps is not None:
            snaps.install()
        r0 = run(eng, va.throwaway_requests(eng, reqs, seed=200), sampling)      # warm-up with the committer (JIT, arena prealloc)
        rc0 = com.finalize()
        com.reset()
        if snaps is not None:
            snaps.snap.clear()
        r = run(eng, reqs, sampling)
        rc = com.finalize()
        sample = challenge_positions(rc.run_root, [s.n_leaves for s in rc.steps], args.openings, salt=b"acqlife")
        t_open = time.perf_counter()
        ops = com.open(sample)
        t_open = time.perf_counter() - t_open
        ok = [com.verify(rc, o) for o in ops]
        vc = snaps.compare() if snaps is not None else None
        if snaps is not None:
            snaps.uninstall()
        com.uninstall()
        assert_pristine(eng, base)
        roots[kind] = [s.root.hex() for s in rc.steps]
        tokens[kind] = r["tokens"]
        arm = {"run_root": rc.run_root.hex(), "n_steps": len(rc.steps), "leaves": [s.n_leaves for s in rc.steps], "bytes": [s.n_bytes for s in rc.steps],
               "openings": {"n": len(ops), "verified": sum(ok), "generate_s": t_open}, "errors": rc.extra.get("errors"), "value_check": vc,
               "wall_s_first": r["wall_s"], "tokens_equal_reference": r["tokens"] == tokens["reference"]}
        if isinstance(com, PackedCommitter):
            arm["packed_stats"] = dict(com.stats)
        # diagnostic timing reps (same committer, same requests) — mean/min of wall over reps
        walls = []
        for _ in range(args.reps):
            com.reset()
            com.install()
            rr = run(eng, reqs, sampling)
            com.finalize()
            com.uninstall()
            walls.append(rr["wall_s"])
        arm["diag_wall_s"] = {"reps": walls, "mean": statistics.fmean(walls), "min": min(walls)}
        com.close()
        results["arms"][kind] = arm
        print(f"[packed] {kind:9s}: run_root {arm['run_root'][:16]}… openings {arm['openings']['verified']}/{arm['openings']['n']} value_check {vc} "
              f"wall mean {arm['diag_wall_s']['mean'] * 1e3:.1f} ms min {arm['diag_wall_s']['min'] * 1e3:.1f} ms errors {arm['errors']} stats {arm.get('packed_stats')}", flush=True)
    # control timing for the same request set
    walls = []
    for _ in range(args.reps):
        walls.append(run(eng, reqs, sampling)["wall_s"])
    results["control_diag_wall_s"] = {"reps": walls, "mean": statistics.fmean(walls), "min": min(walls)}
    results["roots_identical_packed_vs_reference"] = roots["packed"] == roots["reference"]
    results["roots_identical_violate_vs_reference"] = roots["violate"] == roots["reference"]
    results["steps_differing_violate"] = [i for i, (a, b) in enumerate(zip(roots["violate"], roots["reference"])) if a != b]
    results["tokens_identical"] = {k: v == tokens["reference"] for k, v in tokens.items()}
    results["step_roots"] = roots
    (out / "packed_results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    print(f"[packed] roots packed==reference: {results['roots_identical_packed_vs_reference']}; violate==reference: {results['roots_identical_violate_vs_reference']} "
          f"(differing steps {results['steps_differing_violate']}); control wall mean {results['control_diag_wall_s']['mean'] * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
