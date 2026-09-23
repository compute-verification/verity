"""Adapter: CMT-1's secure engine (`verity_capture.experimental.commit`, lane/r9-cmt-ref) driven by this lane's acquisition hooks,
behind the `api.Committer` interface the benchmark harness (`verity_capture/bench/commit_delta.py`) expects.

Execution path per engine step (the SERVING thread only sees the hooks and one event record):
  serving thread   forward hooks -> uint8 views of every Instance output (no copy)                         [acquisition]
  worker thread    side CUDA stream waits on the step event; builds the CMT-1 `StepLayout` (segment per tensor, manifest class
                   ids, 4-B aligned, 64-B pos-leaf leaves) and the packed step stream on the DEVICE (slice copies)     [movement]
                   `Committer.commit_step(layout, stream)`  backend "torch": torch-vectorised SHA-256 on the device (leaves + fold);
                                                            backend "host" : one hashlib call per 64-B leaf (CMT-1's slow reference)
                                                                                                                      [hashing]
                   D2H of the packed stream into pinned memory (values retained for openings; device stream freed)   [movement]
  finalize         `Committer.finish()` -> request root R = SHA-256(tag || program || ctx || positions digest || S || N || merkle)
                                                                                                                      [root_finalisation]
  openings         `open_value` from the retained host stream + retained levels; verifier = `verify_opening` (production
                   `path_shape`/`fold_path` core, rebinding R)                                                        [opening_generation]

Coverage declaration: the segments' class ids ARE manifest ids (`covers`), so the checker matches exactly.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from verity_capture.experimental import commit as C
from verity_capture.experimental.commit_integ.api import Opening, RunCommitment, StepCommitment, declaration
from verity_capture.experimental.commit_integ.native_host import NativeHostCommitter, TensorMeta, _mro_names, _StepJob

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

TORCH_DTYPE = {"torch.bfloat16": "bf16", "torch.float16": "f16", "torch.float32": "f32", "torch.int32": "i32", "torch.int64": "i64",
               "torch.uint8": "u8"}

# module class -> manifest class id of output k (SmolLM2 / Llama family census ids, CMT-1 interface_manifest.json)
CLASS_ID = {
    ("Linear", 0): "gemm1/out/gemmcoordinate_v1/f2fpbf16",
    ("LogitsProcessor", 0): "gemm1/out/gemmcoordinate_v1/f2fpbf16",
    ("RMSNorm", 0): "rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16",       # fused (norm, residual) tuple: normalised word
    ("RMSNorm", 1): "rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp",               # fused: residual word
    ("RMSNorm", "single"): "rmsnormtriton1/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16",   # no residual (final norm): Triton kernel
    ("SiluAndMul", 0): "silumul1/out/silumulbf16_v1/silumulbf16",
    ("GeluAndMul", 0): "gelumul1/out/gelumulbf16_v1/gelumulbf16",
    ("RotaryEmbedding", 0): "rope1/out/ropeout_v1/ropeout",
    ("RotaryEmbedding", 1): "rope1/out/ropeoutadd_v1/ropeoutadd",
    ("Attention", 0): "attention3/out/f2fpbf16_v1/f2fpbf16",
    ("VocabParallelEmbedding", 0): "embedding1/out/gatherbf16x49152_v1/gatherbf16x49152",   # vocab-specific id (SmolLM2); family fallback below
    ("VocabParallelEmbedding", "in"): "input321/in/Input32",
    ("Sampler", "sampled_token_ids"): "tokenselect1/out/tokenselect_v1/selecti32",
}
FAMILY_OF_CLASS_ID = {"gemm1/": "Gemm_v1", "rmsnormfusedcuda2/": "RMSNormFusedCuda_v2", "rmsnormtriton1/": "RMSNormTriton_v1", "silumul1/": "SiluMul_v1",
                      "gelumul1/": "GeluMul_v1", "rope1/": "RoPE_v1", "attention3/": "Attention_v3", "embedding1/": "Embedding_v1",
                      "input321/": "Input32_v1", "tokenselect1/": "TokenSelect_v1"}


class CmtRefCommitter(NativeHostCommitter):
    """native acquisition + CMT-1 engine.  `backend`: "torch" (device SHA-256) | "host" (hashlib per 64-B leaf)."""

    def __init__(self, engine: Any, *, backend: str = "torch", compiled: bool = False, manifest_ids: dict | None = None, **kw) -> None:
        kw.setdefault("staging", "arena")
        super().__init__(engine, **kw)
        if self.arena is None and engine is not None:
            raise ValueError("CmtRefCommitter needs arena staging")
        self.name = f"cmt_ref_{backend}{'_compiled' if compiled else ''}"
        self.backend, self.compiled = backend, compiled
        self.manifest_ids = manifest_ids
        self._c: C.Committer | None = None
        self._streams_host: dict[int, Any] = {}
        self.gpu_pack_ms: dict[int, float] = {}
        self._ctx = C.Context("cmt-integ", self.run_id, 0, model=str(getattr(getattr(engine, "cfg", None), "model", "")) if engine else "")
        self._family_cache: dict[str, str] = {}
        self._new_committer()

    def _new_committer(self) -> None:
        self._c = C.Committer(self.program_digest, self._ctx, backend=self.backend, compiled=self.compiled, keep_levels=True)

    # -- class ids -------------------------------------------------------------------------------------------------------------
    def _class_key(self, module_name: str) -> str:
        if module_name in self._family_cache:
            return self._family_cache[module_name]
        key = "Other"
        for n, m in self._modules:
            if n == module_name:
                names = _mro_names(m)
                for k in ("LogitsProcessor", "RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding", "Attention",
                          "VocabParallelEmbedding", "Sampler"):
                    if k in names:
                        key = "RMSNorm" if k == "GemmaRMSNorm" else k
                        break
                else:
                    if names & {"LinearBase", "ColumnParallelLinear", "RowParallelLinear", "Linear"}:
                        key = "Linear"
                    else:
                        key = type(m).__name__
                break
        if module_name == "runner.sampler":
            key = "Sampler"
        self._family_cache[module_name] = key
        return key

    def _class_id(self, name: str) -> tuple[str, str]:
        """acquired tensor name ("<module>", "<module>/<k>", "<module>/in", "<module>/sampled_token_ids") -> (class_id, instance)."""
        base, _, suf = name.partition("/")
        key = self._class_key(base)
        k: Any = 0 if suf == "" else (int(suf) if suf.isdigit() else suf)
        if key == "RMSNorm" and suf == "":
            k = "single"
        cid = CLASS_ID.get((key, k)) or f"unmapped/{key}/{k}"
        return cid, base

    # -- acquisition: the base class's arena path (D2D snapshot on the compute stream, 4-B aligned = CMT-1's StepLayout offsets)

    # -- worker ---------------------------------------------------------------------------------------------------------------
    def _commit_step(self, job: _StepJob) -> None:
        side = self._side_stream()
        with torch.inference_mode(), torch.cuda.stream(side):     # the arena blocks were allocated under vLLM's inference_mode
            side.wait_event(job.event)
            t0 = time.perf_counter()
            L = C.StepLayout(job.step, n_tokens=0)
            for m in job.metas:
                dt = TORCH_DTYPE.get(m.dtype, "u8")
                shape = m.shape if dt != "u8" else (m.nbytes,)
                cid, inst = self._class_id(m.name)
                try:
                    seg = L.add(cid, inst, dt, shape)
                except ValueError:  # duplicate (class, instance) e.g. a module called twice in one step: disambiguate by ordinal
                    seg = L.add(cid, f"{inst}#{m.ordinal}", dt, shape)
                if seg.offset != m.dev_off and not job.overflow:
                    self.errors.append(f"layout offset mismatch step {job.step} {m.name}: {seg.offset} != {m.dev_off}")
            L.freeze()
            ev_a = torch.cuda.Event(enable_timing=True)
            ev_b = torch.cuda.Event(enable_timing=True)
            ev_a.record(side)
            if job.overflow:                                   # rare: the step spilled over several blocks -> pack them
                dev = torch.empty(L.padded_bytes, dtype=torch.uint8, device="cuda")
                base = 0
                for blk, used in job.overflow + [(job.arena, job.cursor)]:
                    dev[base:base + used].copy_(blk[:used])
                    base += used
                dev[base:].zero_()
            else:
                dev = job.arena[:L.padded_bytes]
                if L.padded_bytes > job.cursor:
                    dev[job.cursor:].zero_()                   # zero the leaf tail (<= 63 B)
            ev_b.record(side)
            t1 = time.perf_counter()
            sc = self._c.commit_step(L, dev if self.backend == "torch" else dev[:L.stream_bytes].cpu().numpy().tobytes())
            t2 = time.perf_counter()
            host = self.pool.take(L.padded_bytes)
            host.copy_(dev, non_blocking=True)                 # retain the values on the host (openings); frees the device block
            ev_c = torch.cuda.Event(enable_timing=True)
            ev_c.record(side)
            ev_c.synchronize()
            t3 = time.perf_counter()
        for blk, _u in job.overflow + [(job.arena, job.cursor)]:
            self.arena.release(blk)
        for m in job.metas:
            m.dev_block = None
        job.arena, job.overflow = None, []
        del dev
        nbytes = L.stream_bytes
        self._streams_host[job.step] = host
        self._layouts[job.step] = job.metas
        self.spans.add("movement", (t1 - t0) + (t3 - t2), step=job.step, nbytes=nbytes)
        self.spans.add("hashing", t2 - t1, step=job.step, nbytes=nbytes)
        self.gpu_pack_ms[job.step] = float(ev_a.elapsed_time(ev_b))     # device time of the tail zero / spill packing (CUDA events)
        self._done[job.step] = StepCommitment(job.step, sc.root, sc.root, sc.leaf_count, nbytes, len(job.metas),
                                              hashlib.sha256(repr(L.to_json()["segments"]).encode()).digest(), job.t_step_end, t2, job.backlog,
                                              hashing_s=t2 - t1, root_finalisation_s=0.0)

    def reset(self) -> None:
        super().reset()
        self._streams_host = {}
        self.gpu_pack_ms = {}
        self._new_committer()

    def finalize(self) -> RunCommitment:
        t0 = time.perf_counter()
        self._end_step()
        with self.spans.span("wait"):
            while len(self._done) < self._step_index:
                time.sleep(0.0005)
        steps = [self._done[s] for s in range(self._step_index)]
        t1 = time.perf_counter()
        if not self._c.steps:
            raise RuntimeError(f"cmt_ref: no steps committed; worker errors: {self.errors[:3]}")
        root, positions = self._c.finish()
        t2 = time.perf_counter()
        self.spans.add("root_finalisation", t2 - t1)
        self.positions = positions
        last_end = max((s.t_step_end for s in steps), default=t0)
        pb = self._c.persistent_bytes()
        return RunCommitment(run_root=root, steps=steps, spans=self.spans.snapshot(), root_ready_delay_s=t2 - last_end, t_finalize_s=t2 - t0,
                             persistent_bytes=pb["roots_bytes"] + pb["levels_bytes"],
                             temporary_bytes_peak=max((s.n_bytes for s in steps), default=0), weight_registration=self.weights,
                             extra={"values_retained_bytes": sum(int(h.numel()) for h in self._streams_host.values()), "errors": self.errors[:10],
                                    "backend": self.backend, "compiled": self.compiled, "hash_calls": self._c.hash_calls,
                                    "hash_blocks": self._c.hash_blocks, "leaf_bytes": C.LEAF_BYTES, "engine": "verity_capture.experimental.commit",
                                    "positions_bytes": len(repr(positions)), "modules_hooked": len(self._modules),
                                    "gpu_pack_ms_per_step": [self.gpu_pack_ms.get(s.step, 0.0) for s in steps]})

    # -- openings --------------------------------------------------------------------------------------------------------------
    def open(self, sample: list[tuple[int, int]]) -> list[Opening]:
        out = []
        with self.spans.span("opening_generation"):
            for step, index in sample:
                L = self._c.steps[step].layout
                ident = None
                for in_leaf in range(0, C.LEAF_BYTES, 2):
                    ident = L.unrank(index, in_leaf)
                    if ident is not None:
                        break
                if ident is None:                      # a pure-padding leaf: open the last value of the step instead
                    seg = L.segments[-1]
                    ident = (seg.class_id, seg.instance, seg.numel - 1)
                cid, inst, el = ident
                op = C.open_value(self._c, step, cid, inst, el, stream=self._streams_host[step].numpy())
                out.append(Opening(step, op.leaf_index, op.value, op.path, {"cmt_ref": op.to_json()}))
        return out

    def verify(self, run: RunCommitment, opening: Opening) -> bool:
        op = C.Opening.from_json(opening.identity["cmt_ref"])
        if op.leaf_index != opening.index or op.value != opening.value or tuple(op.path) != tuple(opening.path):
            return False
        ok, _why = C.verify_opening(run.run_root, self.positions, op)
        return bool(ok)

    # -- coverage --------------------------------------------------------------------------------------------------------------
    def coverage_declaration(self, run: RunCommitment) -> dict:
        per_class: dict[str, dict] = {}
        for sc in self._c.steps:
            for seg in sc.layout.segments:
                d = per_class.setdefault(seg.class_id, {"bytes": 0, "count": 0, "tensors": 0})
                d["bytes"] += seg.nbytes
                d["count"] += seg.numel
                d["tensors"] += 1
        classes = [{"class": cid, "how": "bound-as-leaf", "covers": [cid], "count": d["count"], "bytes": d["bytes"], "tensors": d["tensors"],
                    "covers_families": [f for pre, f in FAMILY_OF_CLASS_ID.items() if cid.startswith(pre)],
                    "rule": None, "acquisitions": ["naturally-materialized", "registered-input"],
                    "evidence": f"segment(s) of the packed 64-B pos-leaf step stream; step root -> request root R (positions digest bound)"}
                   for cid, d in sorted(per_class.items())]
        if self.weights:
            classes.append({"class": "weights", "how": "registered-input", "covers_families": ["Input16_v1"], "acquisitions": ["registered-input"],
                            "rule": "register_weights(): per-tensor chunked pos_leaf subtree, fold of subtree roots bound with names/dtypes/shapes",
                            "count": None, "bytes": self.weights["bytes"], "tensors": self.weights["tensors"], "evidence": f"root {self.weights['root'][:16]}"})
        if any(c.startswith("rmsnorm") for c in per_class):
            classes.append({"class": "norm_bf16tof32_lift", "how": "derived-from-bound",
                            "covers": ["rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo2",
                                       "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo2"],
                            "rule": "out/gen/r8/fa2-commit/derived_value_rule.md §1 (Bf16ToF32 injective lift of the bound bf16 residual word) — "
                                    "a PROTOCOL CLAUSE, not a hash; reported separately", "count": None, "bytes": None,
                            "evidence": "source = the RMSNorm residual output segment bound above"})
        notes = ["engine = CMT-1 verity_capture.experimental.commit (packed 64-B pos-leaf leaves, positions digest bound in R)",
                 "binds ONLY naturally materialised values; hidden-fa2 / hidden-vllm-kernel classes are NOT bound by this path",
                 f"unmapped module outputs (class id 'unmapped/...') are bound but not matched to a manifest class: "
                 f"{sorted(c for c in per_class if c.startswith('unmapped'))}"]
        return declaration(self.name, f"native-acq+cmt-ref-{self.backend}", classes, notes)
