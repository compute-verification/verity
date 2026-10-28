"""Reference committer for the NATURALLY MATERIALISED interface class (Call outputs in HBM + input ids + sampled ids),
host-hashed.  Value acquisition (the hooks) -> commitment (the position-identified Merkle roots); the
class tables `ACQUIRE_CLASSES` / `FAMILY_OF_MODULE` / `HAZARD_LEAVES` are a hand-written runtime correspondence (module class ->
Definition family) and are pinned by `tests/test_no_by_name_rules.py` until it is recorded as data.

EXPERIMENTAL.  This is the "correct first, then fast" reference for the class the census calls
`HBM-materialized (native)`; it binds NOTHING hidden inside FA2 / the norm kernels (that is the hidden-value path) and its
coverage declaration says so — the checker turns that into a `partial` verdict until the hidden path is wired.

Construction (one-stage, binding, position-identified):

  acquisition   `register_forward_hook` on every module whose output is a Call output of the eager step (Linear family,
                RMSNorm family, SiluAndMul/GeluAndMul, RotaryEmbedding, Attention, VocabParallelEmbedding (+ its input ids),
                LogitsProcessor (logits), Sampler (sampled token ids)); tensors are taken as raw bytes (bf16/fp32/int) — no conversion
  movement      async D2H (`copy_(non_blocking=True)` on the compute stream, so the copy is stream-ordered right after the producer)
                into pinned host blocks; one CUDA event per step marks "all of this step's bytes are on the host"
  hashing       a worker thread waits on the event, then hashes CHUNK-byte position-identified leaves
                leaf_i = SHA-256("verity/pos-leaf/v0" || u64 nbytes || bytes)   (C0 `pos_leaf`; the frame binds the length)
                in tree order (tensor 0 chunks, tensor 1 chunks, …) with a thread pool (hashlib releases the GIL above 2 KB), then
                folds with the production pairing/lift rule (`scheme.fold_levels`, same shape rule as `MerkleTree`)
  root          step root R_s = SHA-256(ROOT_TAG || program_digest || ctx_digest(run, step) || geo_digest(model) || layout_digest(step)
                || u64 N_s || merkle_root)   (`scheme.step_root`); the layout digest pins (name, dtype, shape, nbytes, chunk)
                of every tensor in order, so a rank names exactly one (tensor, byte range); run root = fold(R_0 … R_{S-1}) bound with S
  openings      (step, rank, value, path) with the path read from the retained levels; verified with the production core
                (`verity_vllm.commit.merkle.path_shape` / `fold_path`) against the ORIGINAL R_s and the run root

What is retained after the run (persistent): the step roots, the layouts, the tree levels (32 B per leaf ≈ 0.8 % of the bytes at
4 KB leaves) and — for opening generation in this benchmark — the pinned host copies of the values (a prover must keep or
regenerate the values to answer a challenge; the bytes are reported as `values_retained_bytes`, not hidden).
"""

from __future__ import annotations

import hashlib
import math
import json
import re
import queue
import ctypes
import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:                      # the offline core (tests, CPU cross-checks) needs no torch
    torch = None  # type: ignore[assignment]

from verity_vllm import settings
from verity_vllm.commit import leaf_rule, scheme, step_rows
from verity_vllm.commit.committer.input_classes import TOKEN_IDS_INPUT, weights_class
from verity_vllm.commit.committer.pinned_pool import PINNED_EXACT, PinnedPool
from verity_vllm.commit.committer import weights_rows
from verity_vllm.commit.committer.topp_splits import SPLITS_IDENTITY
from verity_vllm.commit.committer.window_views import WindowViews, _StepJob
from verity_vllm.commit.native_ranges import NativeRangeOpenings, open_levels
from verity_vllm.acquire.committer_api import Opening, RunCommitment, StepCommitment, declaration
from verity_vllm.engine import hooks
from verity_vllm.engine.profiles import layer_classes as LC
from verity_vllm.pipeline.spans import Spans

GPU_CHUNK_BYTES = 256
DEFAULT_CHUNK = 4096

# the construction-time selection: TP1 Commits replace it with the acquisition plan's (apply_plan_to_committer; compiled rows:
# CompiledGraphSource.wrap_model); TP rank committers (pipeline/tp/rank_worker.py) have no plan yet and keep it
# module classes whose forward OUTPUT is an Instance output (census §3 `HBM-materialized (native)`); matched on the MRO class names
ACQUIRE_CLASSES = {
    "ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear", "LinearBase",
    "RMSNorm", "GemmaRMSNorm", "LayerNorm",
    "SiluAndMul", "GeluAndMul", "GeluAndMulSparse", "MulAndSilu", "SiluAndMulWithClamp",
    "RotaryEmbedding",
    "Attention",
    "VocabParallelEmbedding",
    "LogitsProcessor",
    "Sampler",
}
# classes whose INPUT ids are a root input of the Program (prompt ids / fed-back sampled ids)
ACQUIRE_INPUT_CLASSES = {"VocabParallelEmbedding"}
SKIP_CLASSES = {"ParallelLMHead"}   # its forward is not called by vLLM (LogitsProcessor applies the quant method); the logits are acquired there


def _mro_names(m: Any) -> set[str]:
    return {c.__name__ for c in type(m).__mro__}


class SkipRefused(RuntimeError):
    """VERITY_SKIP names an ablation that cannot be applied as named: the run is refused, never labelled with a skip that did not happen."""


def skip_classes_of_env() -> frozenset[str]:
    """VERITY_SKIP: a comma list of module classes whose module-output acquisition a per-row MARGINAL-cost ablation drops (NOT a commitment)."""
    return frozenset(settings.VERITY_SKIP)


def _dtype_name(t: Any) -> str:
    return str(t.dtype).replace("torch.", "")


@dataclass
class TensorMeta:
    ordinal: int
    name: str            # module name (+ "/in" for an acquired input, "/k" for tuple element k)
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunk: int
    first_leaf: int      # rank of the first leaf of this tensor in the step tree
    n_leaves: int
    host: memoryview | None = field(default=None, repr=False)   # pinned bytes (set when copied)
    dev_off: int = 0                                             # arena staging: byte offset of this tensor in its device block
    stream_off: int = -1                                         # packed mode: byte offset of this tensor in the step's packed stream
    dev_block: Any = field(default=None, repr=False)             # arena staging: the device block (released after D2H)


class _DeviceArena:
    """Device (GPU) staging blocks, one per in-flight step.  A hook copies every acquired tensor D2D into the step's block ON THE
    COMPUTE STREAM (stream-ordered => the snapshot is taken before any later kernel overwrites the tensor in place, e.g. the fused
    add+RMSNorm residual), so the worker can move / hash the packed step stream from a SIDE stream while the engine runs on.
    Blocks are allocated once (warm-up) and recycled when the worker releases them."""

    def __init__(self, block_bytes: int, n_blocks: int = 2, extra_blocks: list[int] | None = None) -> None:
        self.block_bytes = block_bytes
        self.free: list[Any] = []
        self.all: list[Any] = []
        self.allocs = 0
        self.alloc_s = 0.0
        self._n_init = n_blocks
        self._extra = list(extra_blocks or [])       # additional preallocated blocks of other sizes (one prefill-sized block)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.waits = 0                                # backpressure: how often / how long the serving thread waited for a block
        self.wait_s = 0.0

    def prealloc(self) -> None:
        for _ in range(self._n_init):
            self.free.append(self._alloc(self.block_bytes))
        for nb in self._extra:
            self.free.append(self._alloc(nb))

    def _alloc(self, nbytes: int) -> Any:
        t0 = time.perf_counter()
        blk = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.all.append(blk)
        self.allocs += 1
        self.alloc_s += time.perf_counter() - t0
        return blk

    # Backpressure: the ring never grows past its preallocation while a fitting block is merely BUSY; the serving thread waits for
    # the worker to release one (the engine stalls instead of the collector OOMing).  Growth only when NO block of the ring can ever fit (a
    # step larger than every block), with VERITY_ARENA_GROW=1, or `wait=False`: the worker, the only releaser, must not wait on itself.
    GROW = bool(settings.VERITY_ARENA_GROW)
    WAIT_TIMEOUT_S = settings.VERITY_ARENA_WAIT_S

    def _best_free(self, min_bytes: int) -> int:
        best = -1
        for i, b in enumerate(self.free):
            if b.numel() >= min_bytes and (best < 0 or b.numel() < self.free[best].numel()):
                best = i
        return best

    def take(self, min_bytes: int = 0, *, wait: bool = True) -> Any:
        # best fit: the SMALLEST free block that holds the step (decode steps must not consume the prefill-sized block)
        with self._cv:
            best = self._best_free(min_bytes)
            if best >= 0:
                return self.free.pop(best)
            if wait and not self.GROW and any(b.numel() >= min_bytes for b in self.all):
                t0 = time.perf_counter()
                while best < 0:
                    if not self._cv.wait(timeout=1.0):
                        if time.perf_counter() - t0 > self.WAIT_TIMEOUT_S:
                            raise RuntimeError(f"native collector arena: waited {self.WAIT_TIMEOUT_S:.0f}s for a free block >= {min_bytes} B "
                                               f"(ring {[int(b.numel()) >> 20 for b in self.all]} MiB, all busy): the worker is not releasing blocks")
                    best = self._best_free(min_bytes)
                self.waits += 1
                self.wait_s += time.perf_counter() - t0
                return self.free.pop(best)
        return self._alloc(max(self.block_bytes, min_bytes))

    def release(self, blk: Any) -> None:
        with self._cv:
            self.free.append(blk)
            self._cv.notify_all()

    @property
    def bytes_total(self) -> int:
        return sum(int(b.numel()) for b in self.all)


# ---------------------------------------------------------------------------------------------------------------------------------
# Collector footprint.  Two things, both OFF by default (roots/layouts/openings are unchanged in every mode):
#
#  * `VERITY_FOOTPRINT=<path.jsonl>` -- a per-step device-memory timeline written from the serving thread (begin/end of step) and the worker
#    (absorb = hashing done, release = arena block back in the pool): bytes staged by value family, arena block sizes, retained copies and
#    tree levels, torch allocated/reserved.  Design input for the streaming absorb/release work; no effect on what is bound.
#  * retention policy `VERITY_RETAIN=device|host|none` (commit_delta `--retain`): WHERE the raw staged bytes live after the step is
#    cryptographically absorbed (leaves + tree computed, root bound).  `device` = a compact device copy per step, kept
#    for lazy openings (~9.2 GB/seq, which OOMs Llama 1024/128 at B=2); `host` = the copy goes to host memory (D2H on the side
#    stream, then the arena block is recycled) and the tree levels move to host too, so the device holds only the arena blocks in flight;
#    `none` = no raw copy at all: only the authentication state (levels) is kept on host, and openings are answered by REPLAY -- a second
#    deterministic (BI) execution of the same committed inputs reproduces the same bytes and the same roots (`open(..., values_from=other)`),
#    verified against the ORIGINAL run's roots.  One-stage binding is untouched: the leaves are hashed from the arena block before any copy.
RETAIN_MODES = ("device", "host", "none", "sample")
# either FlashAttention tap's hidden-value rows (`fa2.m1/L<k>/...`, `fa3.m1/L<k>/...`): one family for the identity check, the trace and the footprint
HIDDEN_M1_PREFIXES = ("fa2.m1/", "fa3.m1/")
TP_PARTIAL_RE = re.compile(r"/(part|shard)_rank\d+$")   # <module path>/part_rank<r>: the rank-local input of that module's collective
CALL_DEPENDENT_RE = re.compile(r"/(all_reduce|x_q|x_s)$")   # bound only on the calls whose token count takes that path; the manifest requires them
#: bound only on the sampling steps whose batch takes vLLM's split top-p path (a row with top_p < 1), so traffic that mixes it with greedy
#: rows binds it on some steps; a derived-S Program's manifest requires it at every sampling event (`sampler_splits`)
STEP_DEPENDENT_NAMES = frozenset({SPLITS_IDENTITY})


def tp_partials_bad(man: dict[str, Any], names: "Counter[str]") -> dict[str, int]:
    """TP rank-local collective inputs: exactly one `tp/<site>/{part,shard}_rank<r>` per expected site, none from elsewhere
    (`NativeHostCommitter.identity_check`'s per-step TP rule over the step's tensor names)."""
    from collections import Counter

    tp_bad: dict[str, int] = {}
    tp_sites = man.get("tp_sites") or []
    tp_counts = Counter(n.rsplit("/", 1)[0] for n in names.elements() if TP_PARTIAL_RE.search(n))
    if tp_sites:
        tp_want = man.get("tp_site_counts") or {}                     # a mid-module site binds its static call count per step
        tp_bad = {s: tp_counts.get(s, 0) for s in tp_sites if tp_counts.get(s, 0) != int(tp_want.get(s, 1))}
        tp_bad.update({s: c for s, c in tp_counts.items() if s not in tp_sites})
        rk = man.get("tp_rank")
        if rk is not None and any(not n.endswith(f"_rank{rk}") for n in names if TP_PARTIAL_RE.search(n)):
            tp_bad["<foreign rank tag>"] = sum(c for n, c in names.items() if TP_PARTIAL_RE.search(n) and not n.endswith(f"_rank{rk}"))
    elif tp_counts:
        tp_bad = {"<undeclared tp source>": sum(tp_counts.values())}
    return tp_bad


def hazard_of_record(name: str, HAZARD_LEAVES: "tuple[str, ...] | frozenset[str] | set[str]") -> bool:
    """v1's in-place HAZARD decision for one acquired tensor name (`<module path>/<slot>`, a tap plane or a TP partial): snapshot at the hook
    (True) or batch at step end (False).  Encodes vLLM's aliasing by leaf name (`NativeHostCommitter.HAZARD_LEAVES` + the two name rules) --
    a by-name lifetime table, owned HERE; `verity_vllm.acquire.plan` names it as a RESIDUAL through this one
    function rather than restating it."""
    if TP_PARTIAL_RE.search(name) or CALL_DEPENDENT_RE.search(name):
        return True          # a collective's input partial (or a value overwritten or freed right after): snapshot now
    if name.endswith("/up"):
        return True          # the fused-MoE up value (`moe/L<k>/up`, or `<MoE module>/up` at calls): its workspace is the block's own w2 output (cache1/cache3 alias)
    base, _, idx = name.partition("/")
    leaf = base.rsplit(".", 1)[-1]
    if leaf in HAZARD_LEAVES or "embed" in leaf:                 # the caller's table (NativeHostCommitter.HAZARD_LEAVES of record); named so the by-name lint sees the rule
        return True
    if idx == "1" and "norm" in leaf.lower():
        return True
    return False


FAMILY_RULES = (   # (substring of the tensor name -> value family) for the footprint profile; first match wins
    ("fa2.m1/", "fa2_hidden_m1"), ("fa3.m1/", "fa3_hidden_m1"), ("sampled_token_ids", "sampled_ids"), ("logits", "logits"), ("embed_tokens/in", "input_ids"),
    ("embed_tokens", "embedding"), ("qkv_proj", "gemm_qkv"), ("o_proj", "gemm_o"), ("gate_up_proj", "gemm_gate_up"), ("down_proj", "gemm_down"),
    ("lm_head", "gemm_lm_head"), ("norm", "norm"), ("rotary", "rope"), ("act_fn", "act"), ("attn", "attention_out"),
)


def value_family(name: str) -> str:
    for sub, fam in FAMILY_RULES:
        if sub in name:
            return fam
    return "other"


class GcPauses:
    """`gc.callbacks` while a run is installed: each collection of at least `min_s`, as (generation, start, seconds, thread), the start on
    `time.perf_counter` like the steps' `t_step_end`, so a pause lands on its step.

    `install` first collects and freezes the heap, as vLLM does once its warm-up is done (`gc_utils.freeze_gc_heap`): the committer, its
    sources and the manifests they hold are built after that freeze, and a full collection over them took 720-960 ms on every rank's
    serving thread of GLM-5.3 TP8, stalling all ranks behind the next all-reduce.  Collections never touch a value."""

    def __init__(self, min_s: float = 1e-3) -> None:
        self.min_s = min_s
        self.pauses: list[tuple[int, float, float, str]] = []
        self.freeze_s: list[float] = []
        self._t0 = 0.0

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
        elif (dt := time.perf_counter() - self._t0) >= self.min_s:
            self.pauses.append((int(info["generation"]), self._t0, dt, threading.current_thread().name))

    def install(self) -> None:
        if self in gc.callbacks:
            return
        t0 = time.perf_counter()
        gc.collect()
        gc.freeze()
        self.freeze_s.append(time.perf_counter() - t0)
        gc.callbacks.append(self)

    def uninstall(self) -> None:
        if self in gc.callbacks:
            gc.callbacks.remove(self)

    def record(self) -> dict[str, Any]:
        return {"min_s": self.min_s, "pauses": list(self.pauses), "total_s": sum(p[2] for p in self.pauses), "freeze_s": list(self.freeze_s)}


class _Footprint:
    """Per-step device-memory timeline (`VERITY_FOOTPRINT`).  One JSON line per event; `t` is perf_counter seconds since the profiler was
    created.  Events: begin_step, end_step (serving thread), absorbed (leaves + levels computed and the copy taken; the arena block is released
    right after), plus `run` summaries written by the committer's finalize."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.t0 = time.perf_counter()
        self._f = open(path, "a")   # noqa: SIM115
        self._lock = threading.Lock()

    @staticmethod
    def cuda() -> dict[str, int]:
        if torch is None or not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved()),
                "device_free": int(free), "device_total": int(total)}

    def event(self, kind: str, **kw: Any) -> None:
        rec = {"t": round(time.perf_counter() - self.t0, 6), "event": kind, "thread": threading.current_thread().name, **kw, "cuda": self.cuda()}
        line = json.dumps(rec, default=str) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()

    @staticmethod
    def families(metas: list["TensorMeta"]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for m in metas:
            d = out.setdefault(value_family(m.name), {"bytes": 0, "tensors": 0})
            d["bytes"] += int(m.nbytes)
            d["tensors"] += 1
        return out


# ---------------------------------------------------------------------------------------------------------------------------------
# The arena block is sized from the WORKLOAD ENVELOPE at arm time, never below 512 MiB.  One engine step must fit in one block (the
# native collector has no overflow chaining); a step larger than the block is refused mid-step ("arena block too small for this step"),
# which leaves the step root incomplete.  The size changes nothing about what is bound, the packed layout, lifetimes or openings, so
# roots for a workload do not depend on it.
_ARENA_ROUND = 64 << 20
_ARENA_SAFETY = 1.25            # decode steps: measured 2.50 MB vs bound 2.72 MB (Llama-3.2-1B, 48/16)
_ARENA_SAFETY_PREFILL = 1.5     # prefill steps: measured 108.0 MB vs analytic 103.8 MB at 48 tokens (the M1 stream packs more than the
                                # StreamLayout formula per launch); the block is the MAX of both, so roots never depend on the factor
_ARENA_MIN = 512 << 20


def _cfg(model: Any, *names: str, default: int | None = None) -> int | None:
    cfg = getattr(model, "config", None)
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


def moe_bytes_per_token(model: Any, moe_layers: int) -> int:
    """Bytes per token per step the MoE block source binds: per MoE layer TOPK x (f32 weight + i32 id) + TOPK x (2I + I + H) bf16
    pair rows + the H bf16 sum (`moe_source.MoeBlockSource`)."""
    if not moe_layers:
        return 0
    hidden = _cfg(model, "hidden_size", default=0) or 0
    inter = _cfg(model, "moe_intermediate_size", "intermediate_size", default=0) or 0
    topk = _cfg(model, "num_experts_per_tok", default=1) or 1
    return int(moe_layers) * (topk * 8 + topk * (2 * inter + inter + hidden) * 2 + hidden * 2)


def schedule_stream_bound(asz: dict[str, Any], reqs: list[dict[str, Any]], sampling: dict[str, Any] | None = None) -> dict[str, Any]:
    """The bytes the collector binds over the WHOLE RUN under the workload's DECLARED schedule (each
    request prefills at its `arrive_step` in one step, then decodes one token per step for `max_tokens`), with the same per-token / per-
    sequence / per-step widths and the same FA2/FA3 hidden-m1 launch geometry as `step_bound_bytes` (`asz` = its record).  The envelope
    bound `retained_bytes_bound` is a worst case over any packing (B x max_prompt tokens of prefill, every sequence padded to the longest);
    for a known workload this is the stream of record: per step, and the hidden family's share (the bytes `--retain-exclude fa3_hidden_m1`
    leaves on the device).  Conservative where vLLM's actual packing is unknown: a step with prefill AND decode sequences pads every
    sequence of its hidden launch to the step's longest query (the FA3 varlen slab); chunked prefill is not modelled (one step per prompt)."""
    from verity_vllm.acquire.sources.hidden_source import mat_total_words, CHUNK_WORDS, fa2_kblock_n, fa3_tile

    md = asz.get("model") or {}
    heads, kv_heads = int(md.get("heads") or 1), int(md.get("kv_heads") or 1)
    head_dim, layers = int(md.get("head_dim") or 64), int(md.get("layers") or 1)
    hidden_m1 = bool(md.get("hidden_m1"))
    fa3 = str(md.get("attn_backend") or "fa2") == "fa3"
    g = max(1, heads // kv_heads)
    per_token, per_seq, per_step = int(asz["per_token_bytes"]), int(asz["per_seq_bytes"]), int(asz["per_step_bytes"])

    def hidden_words(HB: int, M: int, NB: int, BN: int) -> int:
        return -(-mat_total_words(HB, M, NB, BN, head_dim) // CHUNK_WORDS) * CHUNK_WORDS

    def launch_words(n_seq: int, seqlen_q: int, seqlen_k: int) -> int:
        if not hidden_m1 or n_seq <= 0:
            return 0
        if fa3:
            if head_dim not in (64, 128):
                return 0
            _, bn = fa3_tile(head_dim, seqlen_q, g, 1)
            return hidden_words(n_seq * kv_heads, seqlen_q * g, (seqlen_k + bn - 1) // bn, bn)
        bn = fa2_kblock_n(head_dim, 1) if head_dim in (64, 96, 128, 256) else 128
        if seqlen_q == 1 and heads > kv_heads:
            return hidden_words(n_seq * kv_heads, g, (seqlen_k + bn - 1) // bn, bn)
        return hidden_words(n_seq * heads, seqlen_q, (seqlen_k + bn - 1) // bn, bn)

    smp_max = int((sampling or {}).get("max_tokens") or 1)
    rq = []
    for r in reqs:
        n_p = len(r.get("prompt_token_ids") or []) or int(r.get("prompt_len") or 1)
        rq.append((int(r.get("arrive_step") or 0), int(n_p), int(r.get("max_tokens") or smp_max)))
    horizon = max((a + n for a, _p, n in rq), default=0)
    steps: list[dict[str, int]] = []
    total = hidden_total = hidden_prefill_total = 0
    for s in range(horizon):
        pre = [(p, n) for a, p, n in rq if a == s]                                 # prefilling this step
        dec = [(p + (s - a), n) for a, p, n in rq if a < s < a + n]                 # decoding: kv length so far
        if not pre and not dec:
            continue
        tokens = sum(p for p, _n in pre) + len(dec)
        n_seq = len(pre) + len(dec)
        m_q = max([p for p, _n in pre] + ([1] if dec else []))
        k_max = max([p for p, _n in pre] + [k for k, _n in dec])
        hid = layers * 4 * launch_words(n_seq, m_q, k_max)
        b = tokens * per_token + n_seq * per_seq + per_step + hid
        steps.append({"step": s, "tokens": tokens, "seqs": n_seq, "prefill_seqs": len(pre), "bytes": b, "hidden_bytes": hid})
        total += b
        hidden_total += hid
        if pre:
            hidden_prefill_total += hid
    big = max(steps, key=lambda d: d["bytes"]) if steps else None
    return {"requests": len(rq), "steps": len(steps), "horizon": horizon, "run_bytes": total, "hidden_bytes": hidden_total,
            "hidden_prefill_bytes": hidden_prefill_total, "largest_step": big,
            "prefill_steps": [d for d in steps if d["prefill_seqs"]][:16], "decode_step_bytes_max": max([d["bytes"] for d in steps if not d["prefill_seqs"]] or [0]),
            "model": "declared schedule: one prefill step per arrive_step group, one decode token per step; hidden launch padded to the step's longest query"}


def bounded_host_bound(ssb: dict[str, Any], *, engine_rss_bytes: int, retain: str, retain_exclude: set[str] | frozenset[str] | list[str],
                       learn_budget_bytes: int, openings: int, chunk_bytes: int, window_bytes: int, bounded: bool, open_after_release: bool,
                       host_cap_bytes: int | None, hidden_factor: float = 1.05, other_factor: float = 2.0, slack_bytes: int = 6 << 30,
                       pairs: int = 1, c2_programs_bytes: int = 0, c2_freed_before_replay: bool = True, pool_exact: bool = PINNED_EXACT,
                       replay_cache: bool = False, fork_children_bytes: int = 0, c2_residual_bytes: int = 0, pool_carry_bytes: int = 0) -> dict[str, Any]:
    """The host peak bound of a `--bounded-staging` Commit, TERM BY TERM, from the declared-schedule stream
    (`schedule_stream_bound`).  Every term is a measured quantity on the B1 1024/128 workload (cp-h100) or a budget the run
    enforces by name:
      raw     = hidden x f_hidden + other x f_other        hidden (the FA3 slab, padded to the step's longest query) sized exact to 1.2 MB on
                                                           B1 -> 1.05; other (per-token widths of every other family) measured x1.59 -> 2.0
      levels  = raw / 4                                    the tree above the leaves (32 B per 128 B leaf... every level, /4 by the layout)
      kept    = raw - hidden                               retain=host with the hidden family excluded: what stays on host (0 for retain=none)
      learn   = learn budget                               the learning steps' transient host windows; the pool keeps their capacity
      replay  = levels + learn + picks x chunk             openings-after-release runs on a FRESH pool that grows like the run's; picks =
                                                           3 x openings + 64 (after-release + both in-run phases' deferred picks), 256 B each
      slack   = engine growth after admission              measured +3,539 MiB (RSS 2,218 -> 5,757 MiB, B1) + pool rounding -> 6 GiB
      pairs   = levels + kept + learn once, for pairs >= 2   a second pair allocates its frozen pool while torch's pinned-host cache still
                                                           holds the first pair's frozen + replay blocks: B1 PAIRS=3 bounded (in-process
                                                           per-step RSS) 16.7 -> 25.6 (pair 0 + replay) -> 29.6 (pair 1 run)
                                                           -> 30.2 (pair 1 replay) -> 30.2 flat through pair 2: +4.6 GiB = 41 % of one
                                                           frozen pool, ONCE; the term takes the whole pool (bounded, converged after pair 1)
    Two more named terms, and the bound is the MAX over the process's PHASES (each a sum of the terms live in it), not one sum:
      c2_programs   the Programs of record loaded for the C2 sampled replay (`RM._programs_for`: 8 Programs / 7.85 M instances = 70,300 MiB
                    of Python objects at B8; ~6.6 GB at B1) -- an EMPIRICAL FORECAST the caller derives from the instance files (json
                    bytes x 15.0, the B8 calibration 4,924 MB -> 70,300 MiB), live from the load until commit_delta frees them after the
                    replay + linkage records (`c2_freed_before_replay`; unfreed, they stack under the replay pool)
      pool_rounding torch's CachingHostAllocator rounds every pinned block up to the next power of two (RSS +21,152 MiB outside the pool
                    at B8): (levels + kept + learn) x 1.0 worst case when the pool is torch-pinned; 0 when the pool registers exact blocks
                    (`pool_exact`, `PinnedPool.mode == "exact-registered-shared"`, RSS == bytes_total)
      fork_children the C2 sampled replay forks its evaluator children (mp Pool) while the Programs and the
                    frozen pool are live: their PRIVATE pages (copy-on-write of the Python objects they touch) are charged to the CGROUP,
                    not to the parent's VmHWM -- an EMPIRICAL term the caller passes (measured: cgroup +7,000 MiB over 33 children on
                    shared pool blocks).  The pool blocks themselves are shared mappings and are NOT copied at fork (a private
                    anonymous pinned block IS copied eagerly by Linux >= 5.9: 59,350 MiB per child, which takes the cgroup to its cap
                    and OOM-kills children); this term is the cgroup scope of the c2 phase
      phases        run = engine + levels + kept + learn + pool_rounding; c2 = run + c2_programs + fork_children; replay = run + replay (+ c2_programs when
                    not freed first); bound = max(phases) + pairs + slack.  `pairs` is 0 for the exact pool (the frozen pool is unregistered
                    and freed when the openings-after-release view dies; the replay pool becomes the next pair's and grows to the run's size)
                    and levels + kept + learn (once) for the torch pool; with `replay_cache` pairs >= 2 reuse pair 0's C2 record and the
                    c2 phase happens once (its term stays in the max either way).
    Two more EMPIRICAL terms the caller passes, both measured by the run's footprint (`rss_bytes` - `pinned_pool_bytes` per phase); with
    them a later pair's replay phase, not the c2 phase, can be the peak:
      c2_residual   what the freed Programs leave resident: 7.85 M dead small objects die into pymalloc arenas / glibc's heap, which return
                    only part (B8: gc + malloc_trim gave 29,578 MiB back of 71,445 loaded; RSS - pool after the free 31,460 MiB above the run's
                    2,732) -- live from the free to the end of the PROCESS: in the replay phase and in every later pair's phases
      pool_carry    the previous pair's pool pages still resident in the next pair: B8 pair 1's run RSS - pool = 54,239 = 2,732 (pair 0's) +
                    31,460 (c2_residual) + 20,047 -- MEASURED; the mechanism is a HYPOTHESIS (blocks of the previous pair's pool whose munmap
                    release() deferred while a tensor view was alive -- `deferred_unmaps`), to be tested by the record; once, for pairs >= 2
                    (pair 2's run RSS - pool = 51,329: no further growth)
      phases        + later_pair_replay (pairs >= 2) = run + replay + c2_residual + pool_carry; replay (pair 0) = run + replay + c2_residual (if
                    the Programs were freed first).  At B8: replay 118,303 + 31,460 = 149,763 (observed 130,895 pair 0: the
                    replay pool 39,833 < its 46,425 bound); later_pair_replay 169,810 (observed 151,097).
    A run whose RSS peak exceeds this bound is a finding against the bound (not a slack adjustment): the terms are on the record."""
    hidden_sized = int(ssb["hidden_bytes"])
    other_sized = max(0, int(ssb["run_bytes"]) - hidden_sized)
    hidden_bound, other_bound = int(hidden_sized * hidden_factor), int(other_sized * other_factor)
    raw = hidden_bound + other_bound
    excl = set(retain_exclude or ())
    hidden_excluded = hidden_bound if ({"fa3_hidden_m1", "fa2_hidden_m1"} & excl) else 0
    kept = max(0, raw - hidden_excluded) if retain == "host" else 0
    levels = raw // 4
    learn = int(learn_budget_bytes or 0)
    picks = 3 * int(openings) + 64
    per_pick = int(chunk_bytes) if bounded else int(window_bytes or 0)
    replay = (levels + learn + picks * per_pick) if open_after_release else 0
    pool_rounding = 0 if pool_exact else (levels + kept + learn)
    pairs_extra = ((levels + kept + learn) if int(pairs or 1) >= 2 else 0) if not pool_exact else 0
    c2 = max(0, int(c2_programs_bytes or 0))
    forkc = max(0, int(fork_children_bytes or 0))
    phase_run = int(engine_rss_bytes) + levels + kept + learn + pool_rounding
    phase_c2 = phase_run + c2 + forkc
    c2res = max(0, int(c2_residual_bytes or 0)) if c2_freed_before_replay else 0        # the Programs never freed: c2 itself is in the phase
    carry = max(0, int(pool_carry_bytes or 0)) if int(pairs or 1) >= 2 else 0
    phase_replay = phase_run + replay + (c2res if c2_freed_before_replay else c2)
    phases = {"run": phase_run, "c2_sampled_replay": phase_c2, "openings_after_release": phase_replay}
    if int(pairs or 1) >= 2:
        phases["later_pair_replay"] = phase_replay + carry          # pairs >= 1: the residual is already in phase_replay; + the carried pool pages
    peak_phase = max(phases, key=phases.get)
    bound = phases[peak_phase] + int(slack_bytes) + pairs_extra
    # the DOMINANT CONTRIBUTOR, so a refusal names a term and a location, never merely "predicted OOM":
    # the largest host term, and -- separately -- the largest single materialization of the stream (the whole-step plane the bound keeps
    # OFF the host under retain-exclude, opened by replay), with its step / tokens / bytes from the declared schedule
    _terms = {"engine": int(engine_rss_bytes), "levels": levels, "kept": kept, "learn": learn, "replay": replay, "slack": int(slack_bytes), "pairs": pairs_extra,
              "c2_programs": c2, "pool_rounding": pool_rounding, "fork_children": forkc, "c2_residual": c2res, "pool_carry": carry}
    _dom = max(_terms, key=_terms.get)
    _big = ssb.get("largest_step") or {}
    _hidden_where = ("excluded from host (retain-exclude), opened by replay" if hidden_excluded else
                     ("counted in kept (retain host)" if retain == "host" else "not retained (retain none), opened by replay"))
    dominant = {"host_term": _dom, "host_term_bytes": _terms[_dom], "peak_phase": peak_phase, "peak_phase_bytes": phases[peak_phase],
                "host_term_share": (_terms[_dom] / bound) if bound else None,
                "largest_materialization": {"family": "hidden stream (fa3/fa2 _hidden_m1 export, prefill)", "step": _big.get("step"),
                                            "tokens": _big.get("tokens"), "step_bytes": _big.get("bytes"), "hidden_bytes": _big.get("hidden_bytes"),
                                            "where": _hidden_where},
                "text": (f"host: {_dom} {_terms[_dom] >> 20} MiB of {bound >> 20} MiB (peak phase {peak_phase} {phases[peak_phase] >> 20} MiB); "
                         f"largest materialization: hidden stream export step "
                         f"{_big.get('step')} ({_big.get('tokens')} tok) {int(_big.get('hidden_bytes') or 0) >> 20} MiB of a {int(_big.get('bytes') or 0) >> 20} MiB step "
                         f"-- {_hidden_where}")}
    return {"engine_rss_bytes": int(engine_rss_bytes), "levels_bytes": levels, "raw_kept_bytes": kept, "raw_bound_bytes": raw, "pairs": int(pairs or 1),
            "pairs_extra_bytes": pairs_extra, "dominant_contributor": dominant,
            "excluded_prefill_hidden_bytes": hidden_excluded, "learn_budget_bytes": learn, "replay_bytes": replay, "window_bytes": int(window_bytes or 0),
            "picks": picks, "per_pick_bytes": per_pick, "slack_bytes": int(slack_bytes), "host_peak_bound_bytes": bound,
            # the two terms and the phase the bound is the max of; every number says what kind it is
            "c2_programs_bytes": c2, "c2_freed_before_replay": bool(c2_freed_before_replay), "pool_rounding_bytes": pool_rounding, "pool_exact": bool(pool_exact),
            "replay_cache": bool(replay_cache), "phases_bytes": phases, "peak_phase": peak_phase, "fork_children_bytes": forkc,
            "c2_residual_bytes": c2res, "pool_carry_bytes": carry,
            "scope": {"parent_vmhwm": "engine, levels, kept, learn, pool_rounding, c2_programs, replay, pairs, slack, c2_residual, pool_carry",
                      "cgroup_only": "fork_children (the evaluator children's private pages; the parent's VmHWM never sees them)"},
            "term_kinds": {"engine": "measured at admission (RSS of this process on the built engine)",
                           "levels": "conservative bound (raw bound / 4; raw = declared schedule x measured B1 factors)", "kept": "conservative bound (raw bound - excluded hidden)",
                           "learn": "budget the run enforces by name", "replay": "conservative bound (a fresh pool: levels + learn + picks x chunk)",
                           "c2_programs": "empirical forecast (instance-file json bytes x 15.0; B8 calibration 4,924 MB -> 70,300 MiB; the run measures it)",
                           "pool_rounding": "conservative bound (x2 worst case of torch's pow2 rounding) or 0 with the exact pool (measured: RSS == bytes_total)",
                           "fork_children": "empirical forecast, cgroup scope (the C2 evaluator children's private COW pages; attempt 1: +7,000 MiB over 33 children; the run's cgroup timeline measures it)",
                           "pairs": "conservative bound (one more frozen pool, once) for the torch pool; 0 for the exact pool (freed on release)",
                           "c2_residual": "empirical (B8 attempt 3: RSS - pool 31,460 MiB above the run's after the Programs were freed; pymalloc/glibc retention; the run measures it)",
                           "pool_carry": "empirical (B8 attempt 3: pair 1 run RSS - pool - c2_residual = 20,047 MiB = the previous pair's frozen-pool blocks with live views, deferred munmap; once for pairs >= 2)",
                           "slack": "measured engine growth after admission (B1 +3,539 MiB) rounded to 6 GiB"},
            "hidden_factor": hidden_factor, "other_factor": other_factor, "hidden_sized_bytes": hidden_sized, "other_sized_bytes": other_sized,
            "hidden_bound_bytes": hidden_bound, "other_bound_bytes": other_bound,
            "derivation": ("raw = hidden x f_hidden + other x f_other; levels = raw / 4; kept = raw - hidden (retain host + exclude); "
                           f"replay = levels + learn + picks x per_pick (per_pick = the opened leaf's {int(chunk_bytes)} B chunk when bounded, else one window) "
                           "on a fresh pool whose transient windows in flight are inside its learn term (pool window capacity = learn budget); "
                           "slack = engine growth after admission; "
                           "pairs = levels + kept + learn once for pairs >= 2 with the torch pool (its pinned cache keeps pair 0's pools; B1 PAIRS=3: +41 %, once), 0 with the exact pool; "
                           "c2_programs = the C2 sampled replay's loaded Programs (forecast from the instance files); pool_rounding = torch pow2 rounding (0 exact); "
                           "fork_children = the C2 evaluator children's private pages (cgroup scope; the pool's shared blocks are not copied at fork); "
                           "c2_residual = what the freed Programs leave resident (empirical); pool_carry = the previous pair's frozen-pool pages still mapped (deferred munmap; once, pairs >= 2); "
                           "phases: run = engine + levels + kept + learn + pool_rounding; c2 = run + c2_programs + fork_children; replay = run + replay + c2_residual (or + c2_programs unless freed first); "
                           "later_pair_replay (pairs >= 2) = replay + pool_carry; "
                           "bound = max(phases) + pairs + slack; factors measured on B1 1024/128 cp-h100 (FP8-STAGING.md §3, §6, §7)"),
            "host_memory_cap_bytes": host_cap_bytes, "margin_bytes": (host_cap_bytes - bound) if host_cap_bytes is not None else None,
            "margin_frac": ((host_cap_bytes - bound) / host_cap_bytes) if host_cap_bytes else None, "retain": retain, "retain_exclude": sorted(excl)}


def step_bound_bytes(model: Any, modules: list[tuple[str, Any]], envelope: dict[str, Any], *, hidden_m1: bool, attn_backend: str = "fa2", moe_layers: int = 0,
                     chunk: int = GPU_CHUNK_BYTES) -> dict[str, Any]:
    """Upper bound of the bytes ONE engine step binds into the arena, from the acquired-module set (the required-value manifest of the
    native path: forward outputs of ACQUIRE_CLASSES, per token) and the workload envelope (`batch`, `max_prompt`, `max_new`,
    `max_num_batched_tokens`, `max_model_len`).  Prefill: the admitted prompt tokens of one step (<= max_num_batched_tokens, <= B x max
    prompt) x bytes/token + per-sequence logits + the FA2 M1 hidden stream of every launch (superlinear in the sequence length: the
    worst case packs whole max-length prompts into the step).  Decode: B tokens x bytes/token + hidden stream at kv length max_model_len."""
    import torch

    hidden = _cfg(model, "hidden_size", default=0) or 0
    heads = _cfg(model, "num_attention_heads", default=1) or 1
    kv_heads = _cfg(model, "num_key_value_heads", default=heads) or heads
    head_dim = _cfg(model, "head_dim", default=None) or (hidden // max(1, heads))
    inter = _cfg(model, "intermediate_size", "moe_intermediate_size", default=4 * hidden) or 4 * hidden
    layers = _cfg(model, "num_hidden_layers", default=1) or 1
    vocab = _cfg(model, "vocab_size", default=32000) or 32000
    per_token = 0            # bytes per token of module outputs
    per_seq = 0              # bytes per sequence per step (logits, sampled ids)
    per_step = 0             # fixed per-step bytes (chunk padding of every acquired tensor)
    widths: dict[str, int] = {}
    # A module SHARED by every layer (Llama's one `rotary_emb`, registered under layer 0) is acquired once per
    # INVOCATION -- `layers` times per step -- not once per module: count it per layer, else the bound is `(layers - 1) x width` per token
    # short (76,800 B/token on Llama-3.2-1B, 1-3 % of a prefill step).
    n_rope = sum(1 for _n, _m in modules if LC.ROTARY in _mro_names(_m))
    rope_mult = max(1, -(-layers // n_rope)) if n_rope else 1
    for name, m in modules:
        names = _mro_names(m)
        try:
            p = next(m.parameters())
            isz = p.element_size()
        except StopIteration:
            isz = 2
        w = 0
        if LC.LOGITS in names:
            per_seq += vocab * 4
        elif LC.SAMPLER in names:
            per_seq += 8
        elif LC.EMBEDDING in names:
            w = int(getattr(m, "embedding_dim", hidden) or hidden) * isz + 4          # + the input ids (ACQUIRE_INPUT_CLASSES)
        elif names & LC.LINEAR:
            out = getattr(m, "output_size_per_partition", None) or getattr(m, "output_size", None)
            if not out and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                out = int(m.weight.shape[0])
            w = int(out or hidden) * isz
        elif names & LC.NORMS:
            hs = int(m.weight.numel()) if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor) else hidden
            w = 2 * hs * isz                                                            # fused add+norm returns (out, residual)
        elif names & LC.GATED_ACTS:
            w = inter * 2
        elif LC.ROTARY in names:
            w = (heads + kv_heads) * head_dim * 2 * rope_mult                           # returns (q, k); once per layer when shared
        elif LC.ATTENTION in names:
            w = heads * head_dim * 2
        else:
            w = hidden * 2
        per_token += w
        per_step += chunk
        cls = next((c for c in (LC.LOGITS, LC.SAMPLER, LC.EMBEDDING, LC.RMS_NORM, LC.ROTARY, LC.ATTENTION, LC.SILU_AND_MUL, LC.LINEAR_BASE) if c in names), "other")
        widths[cls] = widths.get(cls, 0) + w
    moe_pt = moe_bytes_per_token(model, moe_layers)
    if moe_pt:
        per_token += moe_pt
        per_step += moe_layers * 6 * chunk
        widths["MoeBlock"] = moe_pt
    B = max(1, int(envelope.get("batch") or 1))
    max_prompt = max(1, int(envelope.get("max_prompt") or 1))
    max_new = max(1, int(envelope.get("max_new") or 1))
    max_model_len = int(envelope.get("max_model_len") or (max_prompt + max_new))
    mnbt = envelope.get("max_num_batched_tokens")
    T_prefill = min(int(mnbt) if mnbt else B * max_prompt, B * max_prompt)

    # FA2: the M1 stream is arithmetic in the kernel tile (kBlockN, head_dim) -- 128/64 for hd 64, 64/hd otherwise (num_splits==1 kernel).
    # FA3: under FA3 (attn_backend "fa3") the slab is (sequence, KV head) and the rows are PackGQA-packed (M = seqlen_q * g); the tile
    # is tile_size_fwd_sm90's (hd 64: 192x128 prefill / 64x192 one-warpgroup decode; hd 128: 128x128 / 64x128) -- hidden_source.fa3_tile.
    from verity_vllm.acquire.sources.hidden_source import mat_total_words, CHUNK_WORDS, fa2_kblock_n, fa3_tile
    g = max(1, heads // kv_heads)
    fa3 = attn_backend == "fa3"

    def hidden_words(HB: int, M: int, NB: int, BN: int) -> int:
        return -(-mat_total_words(HB, M, NB, BN, int(head_dim)) // CHUNK_WORDS) * CHUNK_WORDS

    def launch_words(n_seq: int, seqlen_q: int, seqlen_k: int) -> int:
        if fa3:
            if int(head_dim) not in (64, 128):
                return 0
            _, bn = fa3_tile(int(head_dim), seqlen_q, g, 1)
            return hidden_words(n_seq * kv_heads, seqlen_q * g, (seqlen_k + bn - 1) // bn, bn)
        bn = fa2_kblock_n(int(head_dim), 1) if int(head_dim) in (64, 96, 128, 256) else 128
        if seqlen_q == 1 and heads > kv_heads:            # FA2 decode "swapped" kernel: rows = the query heads of a KV group
            return hidden_words(n_seq * kv_heads, g, (seqlen_k + bn - 1) // bn, bn)
        return hidden_words(n_seq * heads, seqlen_q, (seqlen_k + bn - 1) // bn, bn)

    # prefill: the tapped launch streams HB = (sequences in the step) x heads rows blocks of M = max_seqlen_q rows each, i.e. EVERY
    # sequence of the step is padded to the step's longest prompt (hidden_source.varlen_fwd: HB = b*h, M = max_seqlen_q, NB from
    # max_seqlen_k; FA3: slab (sequence, KV head), M = max_seqlen_q * g).  Worst case for T_prefill admitted tokens: one max-length prompt +
    # (B-1) one-token prompts -> B x heads x max_prompt rows.  (A "whole prompts" packing (T_prefill // max_prompt
    # sequences) under-bounds a B=8 Phi-3 256-token prefill by 2x -> "arena block too small for this step" mid-run; the bound is the stream
    # geometry itself.)
    M_pref = min(max_prompt, T_prefill)
    seqs_prefill = min(B, max(1, T_prefill - M_pref + 1))
    hid_prefill = layers * 4 * launch_words(seqs_prefill, M_pref, M_pref) if hidden_m1 else 0
    prefill = T_prefill * per_token + seqs_prefill * per_seq + per_step + hid_prefill
    hid_decode = layers * 4 * launch_words(B, 1, max_model_len) if hidden_m1 else 0
    decode = B * per_token + B * per_seq + per_step + hid_decode
    bound = max(prefill, decode)
    rnd = lambda n: -(-int(n) // _ARENA_ROUND) * _ARENA_ROUND  # noqa: E731
    decode_block = max(_ARENA_ROUND, rnd(decode * _ARENA_SAFETY))
    prefill_block = max(decode_block, rnd(prefill * _ARENA_SAFETY_PREFILL))
    block = max(_ARENA_MIN, prefill_block)   # single-size equivalent (explicit --arena-mb comparison)
    # the GPU tree keeps every step's used bytes for lazy openings (existing lifetime): upper bound over the whole workload
    retained = prefill * (-(-B * max_prompt // max(1, T_prefill))) + decode * max_new
    return {"per_token_bytes": per_token, "per_seq_bytes": per_seq, "per_step_bytes": per_step, "widths": widths,
            "rope_per_layer": True, "rope_modules": n_rope, "rope_invocations_per_step": n_rope * rope_mult,
            "prefill_tokens": T_prefill, "prefill_step_bytes": prefill, "prefill_hidden_bytes": hid_prefill,
            "decode_step_bytes": decode, "decode_hidden_bytes": hid_decode, "step_bound_bytes": bound, "block_bytes": block,
            "prefill_block_bytes": prefill_block, "decode_block_bytes": decode_block,
            "retained_bytes_bound": retained, "envelope": {"batch": B, "max_prompt": max_prompt, "max_new": max_new,
                                                           "max_num_batched_tokens": mnbt, "max_model_len": max_model_len},
            "model": {"hidden": hidden, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "inter": inter, "layers": layers,
                      "vocab": vocab, "hidden_m1": bool(hidden_m1), "attn_backend": attn_backend, "moe_layers": int(moe_layers), "moe_bytes_per_token": moe_pt}}


def _gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"


def _split_step_names(srcs: list[Any], names: Any, step: int) -> tuple[Any, dict[str, tuple[int, int]]]:
    """The step's names with each call-boundary source's own split off in turn, and every source's mismatches."""
    bad: dict[str, tuple[int, int]] = {}
    for src in srcs:
        names, mine = src.split_step_names(names, step, False)
        bad.update(mine)
    return names, bad


_MEMTRACE = settings.VERITY_MEMTRACE


def memtrace(where: str, step: int | None = None) -> None:
    """Memory diagnostics (env VERITY_MEMTRACE=1): one line with the CUDA caching-allocator state at a lifetime boundary."""
    if not _MEMTRACE:
        return
    import torch

    a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    print(f"[memtrace] {time.strftime('%H:%M:%S')} step {step if step is not None else '-':>4} {where:<28} alloc {a / 2**30:6.2f} reserved {r / 2**30:6.2f} "
          f"device-used {(total - free) / 2**30:6.2f} GiB", flush=True)


class NativeHostCommitter(WindowViews, NativeRangeOpenings, leaf_rule.LeafSchemeRule):
    name = "native_host"

    def __init__(self, engine: Any, *, program_digest: bytes | None = None, run_id: str = "run", chunk: int = DEFAULT_CHUNK,
                 hash_threads: int = 2, retain_values: bool = True, acquire_classes: set[str] | None = None, staging: str = "arena",
                 arena_bytes: int = 512 << 20, batch_copy: bool = False, fast_copy: bool = False, regen_row4: bool = False, gpu_tree: bool = False,
                 geo_digest: bytes | None = None, leaf_scheme: str = scheme.NAME, populate_threads: int | None = None,
                 commit_scheme: str | None = None) -> None:
        """`engine=None` builds the OFFLINE core (hashing, roots, openings, verification; no hooks, no CUDA) for the security tests.  `leaf_scheme`: vllm-v1 position leaves (default) or hm96-sha256/v1 hiding leaves over them (`leaf_rule.LeafSchemeRule`).  `populate_threads`: the pinned pool's (`PinnedPool`).
        `commit_scheme` rows-sha512: the offline core's GPU-tree steps (`commit_block_offline`) under `step_rows`, as the windowed collector
        commits a served run's (`NativeCollectCommitter(commit_scheme=)`, which sets its own)."""
        self.engine = engine
        self.runner = getattr(engine, "runner", None)
        self.model = getattr(engine, "model", None)
        self.chunk = int(chunk)
        self.hash_threads = int(hash_threads)
        self.retain_values = retain_values
        self.run_id = run_id
        self.program_digest = program_digest or hashlib.sha256(b"cmt-integ/no-program-digest/" + run_id.encode()).digest()
        self.acquire_classes = acquire_classes or ACQUIRE_CLASSES
        self.spans = Spans(scheme.NAME)
        self.gc_pauses = GcPauses()
        self.x03_expected = 0
        self.x03_committed = 0
        # the Program's per-layer attention Call statics (fa2_plane_classes.AttnLayerDecl, set by the driver from the
        # Build record): with them `identity_check` DERIVES each layer's hidden-plane class per step; without them the uniformity rule runs
        self.attention_layer_decls: list | None = None
        self.plane_policy = None                          # fa2_plane_classes.LayoutPolicy override (default: read from the hidden source)
        self.pool = PinnedPool(populate_threads=populate_threads)
        self._handles: list = []
        self._orig: dict[str, Any] = {}
        self._hooks = hooks.Hooks()
        self._cur: _StepJob | None = None
        self._step_index = 0
        self._init_views()
        self._q: queue.Queue = queue.Queue()
        self._done: dict[int, StepCommitment] = {}
        self._levels: dict[int, list[list[bytes]]] = {}
        self._layouts: dict[int, list[TensorMeta]] = {}
        self._streams: dict[int, memoryview] = {}
        self.sources: list[Any] = []
        self._worker: threading.Thread | None = None
        # optional CPU pinning experiment (env VERITY_PIN=1): serving thread -> one core, worker + hash threads -> other cores
        self.pin = settings.VERITY_PIN and hasattr(os, "sched_setaffinity")
        self._cores = sorted(os.sched_getaffinity(0)) if self.pin else []
        self._pool_exec = ThreadPoolExecutor(max_workers=self.hash_threads, thread_name_prefix="cmt-hash",
                                             initializer=(lambda: self._pin_thread(2)) if self.pin else None)
        self._stop = object()
        self.installed = False
        self.weights: dict | None = None
        self.errors: list[str] = []
        self.batch_copy = bool(batch_copy)
        self.staging = staging                      # "arena": D2D snapshot on the compute stream, D2H on the worker's side stream
        if staging != "arena":
            self.name = f"{self.name}_direct"          # ablation: D2H issued on the compute stream inside the hook
        if batch_copy:
            self.name = f"{self.name}_batched"
        self._rt = None
        self.cudart_path = None
        if fast_copy and staging == "arena" and engine is not None:
            self._rt, self.cudart_path = self._load_cudart()
            self._memcpy, self._memset = self._rt.cudaMemcpyAsync, self._rt.cudaMemsetAsync
            self.name = f"{self.name}_fast"
        self._dtype_names: dict = {}
        # GPU-tree steps: each step's levels and retained bytes kept per step (`_gpu_levels`, `_gpu_blocks`), openings served lazily from
        # them.  A served run's are the windowed collector's (`native_collect`, rows-sha512); offline (`commit_block_offline`) the same
        # per-step state from the CPU reference, under rows-sha512 or vllm-v1's chunk leaves (256 B chunks, hidden-path chunk header)
        self.gpu_tree = bool(gpu_tree)
        self._gpu_levels: dict[int, list] = {}
        self._gpu_blocks: dict[int, tuple] = {}
        self._gpu_t: dict[str, float] = {"leaf_ms": 0.0, "tree_ms": 0.0}
        if gpu_tree:
            self.name = f"{self.name}_gputree"
            self.chunk = GPU_CHUNK_BYTES
        self._init_leaf_scheme(leaf_scheme, gpu_tree or type(self)._commit_step is not NativeHostCommitter._commit_step)
        if commit_scheme not in (None, scheme.NAME):
            if commit_scheme != step_rows.NAME or engine is not None or not gpu_tree or self.hiding is not None:
                raise ValueError(f"commit_scheme {commit_scheme!r}: the offline core takes {step_rows.NAME} over GPU-tree steps with vllm-v1 "
                                 "leaves (engine=None, gpu_tree=True); a served run's is the windowed collector's")
            self.commit_scheme, self.rows, self.row_keys, self.chunk = step_rows.NAME, True, {}, step_rows.ROW_BYTES
            self.name = f"{self.name}+{step_rows.NAME}"
        # where absorbed raw bytes live (see RETAIN_MODES); `retain_values=False` (offline tests) maps to "none"
        # host retention goes to pinned pool memory unless VERITY_RETAIN_PINNED=0 (pageable D2H is ~1 GB/s and synchronous)
        self.retain_pinned = settings.VERITY_RETAIN_PINNED
        self.retain = r if (r := settings.VERITY_RETAIN) is not None else "device" if retain_values else "none"
        if self.retain not in RETAIN_MODES:
            raise ValueError(f"VERITY_RETAIN={self.retain!r}: expected one of {RETAIN_MODES}")
        self.retain_windows: set[tuple[int, int]] = set()   # retain=sample (windowed only): (step, window k) pairs whose raw bytes are kept
        if self.retain != "device":
            self.name = f"{self.name}+retain_{self.retain}"
        self.footprint = _Footprint(fp) if (fp := settings.VERITY_FOOTPRINT) else None
        self.released_bytes = 0                     # device bytes freed by release_staging() (reported by commit_delta)
        self.inline_d2h = settings.VERITY_INLINE_D2H
        if self.inline_d2h and self._rt is not None:
            self.name = f"{self.name}_inl"
        self.regen_row4 = bool(regen_row4)
        if regen_row4:
            self.name = f"{self.name}+row4regen"
        # `arena_bytes=None` = size the block from the workload envelope in `size_arena()` (commit_delta calls it at arm time);
        # an explicit value is honoured but checked against the same bound (refusal BEFORE the run, with needed vs available bytes)
        self.arena_explicit = arena_bytes is not None
        self.arena_sizing: dict[str, Any] | None = None
        self.arena = _DeviceArena(int(arena_bytes or _ARENA_MIN)) if (engine is not None and staging == "arena") else None
        # MANIFEST-DRIVEN input acquisition: module name -> bytes per token of its acquired input.  The required
        # manifest's promoted members (an interior Program row whose output another module reads) are observed at their first consumer's
        # input as `<consumer>/in` -- the same mechanism as the embedding's ids, no longer restricted to a module class -- and the binding map
        # binds that tensor to the producer's identity (binding.promoted_input_acquisition).  Set by commit_delta before install / size_arena.
        self.acquire_inputs: dict[str, int] = {}
        # the consumer-input pre-hook set comes from the ACQUISITION PLAN (verity_vllm.acquire.install
        # .apply_plan_to_committer sets it) instead of ACQUIRE_INPUT_CLASSES; None (no plan: TP rank committers) = the class table + `acquire_inputs`.
        self.plan_inputs: set[str] | None = None
        # VERITY_SKIP ablation: applied (`apply_skip`) to the acquisition plan's selection (`apply_plan_to_committer`); `skipped` = the
        # modules dropped per class, None until applied
        self.skip_classes = skip_classes_of_env()
        self.skipped: dict[str, list[str]] | None = None
        self._side = None                           # the worker's CUDA side stream (created lazily on the worker thread)
        if engine is not None:
            self._geo_digest = geo_digest or self._model_geo_digest()
            self._modules: list[tuple[str, Any]] = self._select_modules()
            if self.arena is not None and self.arena_explicit:
                self.arena.prealloc()               # auto mode preallocates in size_arena() once the envelope is known
        else:
            self._geo_digest = geo_digest or hashlib.sha256(b"cmt-integ/offline-geo/" + run_id.encode()).digest()
            self._modules = []

    def size_arena(self, envelope: dict[str, Any], *, hidden_m1: bool | None = None, margin_bytes: int = 1 << 30) -> dict[str, Any]:
        """Size the device staging block from the workload envelope (see `step_bound_bytes`).  Auto mode replaces the default
        block by the bound (rounded up, >= 512 MiB) and preallocates; an explicit `--arena-mb` smaller than the bound is refused HERE
        (fail closed before any step) with needed vs available bytes.  Also refuses when the two preallocated blocks do not fit next to
        the engine's reservation, and says which `gpu_memory_utilization` would.  Returns the sizing record (also kept in `arena_sizing`)."""
        import torch

        if self.arena is None:
            return {}
        if hidden_m1 is None:
            hidden_m1 = any("Hidden" in type(s).__name__ for s in getattr(self, "sources", []))
        backend = "fa3" if any(type(s).__name__ in ("HiddenM1SourceFA3", "NativeHiddenM1SourceFA3") for s in getattr(self, "sources", [])) else "fa2"
        moe_src = next((s for s in getattr(self, "sources", []) if getattr(s, "short_name", "") == "moe"), None)
        chunk = self.chunk if self.gpu_tree else GPU_CHUNK_BYTES
        rec = step_bound_bytes(self.model, self._modules, envelope, hidden_m1=bool(hidden_m1), attn_backend=backend,
                               moe_layers=(len(moe_src.layers) if moe_src is not None else 0), chunk=chunk)
        # sources that bind extra tensors per step (TP partial source) widen the bound: per token / per sequence / chunk padding
        extra_pt = extra_ps = extra_step = extra_n = 0
        for s in getattr(self, "sources", []):
            f = getattr(s, "extra_step_bytes", None)
            if f is None:
                continue
            e = f(envelope, self.model)
            extra_pt += int(e.get("per_token_bytes", 0)); extra_ps += int(e.get("per_seq_bytes", 0)); extra_n += int(e.get("tensors_per_step", 0))
            extra_step += int(e.get("per_step_bytes", 0))
        if self.acquire_inputs:
            # the manifest-driven `<consumer>/in` acquisitions: one per-token tensor per consumer per step (its width from the
            # promoted member's registry signature), minus the embedding-class inputs the module table already counts
            for n, bpt in sorted(self.acquire_inputs.items()):
                m = dict(self._modules).get(n)
                if m is not None and (_mro_names(m) & ACQUIRE_INPUT_CLASSES):
                    continue
                extra_pt += int(bpt); extra_n += 1
            rec["promoted_inputs"] = {"consumers": sorted(self.acquire_inputs), "per_token_bytes": sum(int(v) for v in self.acquire_inputs.values())}
        if extra_pt or extra_ps or extra_step or extra_n:
            B = max(1, int(envelope.get("batch") or 1))
            add_prefill = int(rec["prefill_tokens"]) * extra_pt + B * extra_ps + extra_step + extra_n * chunk
            add_decode = B * extra_pt + B * extra_ps + extra_step + extra_n * chunk
            rec["extra_sources"] = {"per_token_bytes": extra_pt, "per_seq_bytes": extra_ps, "per_step_bytes": extra_step, "tensors_per_step": extra_n,
                                    "prefill_add": add_prefill, "decode_add": add_decode}
            rec["prefill_step_bytes"] = int(rec["prefill_step_bytes"]) + add_prefill
            rec["decode_step_bytes"] = int(rec["decode_step_bytes"]) + add_decode
            rec["step_bound_bytes"] = max(int(rec["prefill_step_bytes"]), int(rec["decode_step_bytes"]))
            for k in ("prefill_block_bytes", "decode_block_bytes", "block_bytes"):
                if k in rec:
                    rec[k] = int(rec[k]) + (add_prefill if k != "decode_block_bytes" else add_decode)
            rec["per_token_bytes"] = int(rec.get("per_token_bytes", 0)) + extra_pt
            rec["retained_bytes_bound"] = int(rec.get("retained_bytes_bound", 0)) + int(rec["prefill_tokens"]) * extra_pt + B * extra_ps
        need = int(rec["block_bytes"])
        free, total = torch.cuda.mem_get_info()
        n_blocks = self.arena._n_init
        rec.update({"device_free_bytes": int(free), "device_total_bytes": int(total), "blocks_preallocated": n_blocks,
                    "explicit_block_bytes": int(self.arena.block_bytes) if self.arena_explicit else None})
        util_ok = max(0.0, (total - n_blocks * need - margin_bytes) / total)
        rec["gpu_memory_utilization_max"] = round(util_ok, 2)
        if self.arena_explicit:
            if self.arena.block_bytes < rec["step_bound_bytes"]:
                raise RuntimeError(f"native collector arena block too small for this workload: step bound {_gib(rec['step_bound_bytes'])} "
                                   f"(prefill {rec['prefill_tokens']} tok -> {_gib(rec['prefill_step_bytes'])}, decode B={rec['envelope']['batch']} -> "
                                   f"{_gib(rec['decode_step_bytes'])}) > --arena-mb block {_gib(self.arena.block_bytes)}; use --arena-mb "
                                   f"{need >> 20} (or omit it for auto sizing) and lower --engine-arg gpu_memory_utilization to <= {util_ok:.2f}")
        elif getattr(self, "windowed", False):
            # windowed staging: no per-step blocks at all -- the ring (K x W, allocated by enable_windows) IS the device staging;
            # the envelope record is still returned (enable_windows sizes K from the largest single tensor)
            rec["prealloc_bytes"] = 0
            rec["blocks_preallocated"] = 0
            rec["staging"] = "windowed"
            self.arena = _DeviceArena(1 << 20, n_blocks=0)
        else:
            # auto: n_blocks decode-sized blocks (the growth unit when the worker lags) + ONE prefill-sized block; best-fit `take`
            already = self.arena.bytes_total
            dblk, pblk = int(rec["decode_block_bytes"]), int(rec["prefill_block_bytes"])
            extra = [pblk] if pblk > dblk else []
            self.arena = _DeviceArena(dblk, n_blocks=n_blocks, extra_blocks=extra)
            need_total = n_blocks * dblk + sum(extra)
            rec["prealloc_bytes"] = need_total
            util_ok = max(0.0, (total - need_total - margin_bytes) / total)
            rec["gpu_memory_utilization_max"] = round(util_ok, 2)
            if need_total > free - margin_bytes + already:
                raise RuntimeError(f"native collector arena does not fit: {n_blocks} x {_gib(dblk)} + {_gib(sum(extra))} staging blocks (step bound "
                                   f"{_gib(rec['step_bound_bytes'])}: prefill {rec['prefill_tokens']} tok -> {_gib(rec['prefill_step_bytes'])} "
                                   f"[hidden {_gib(rec['prefill_hidden_bytes'])}], decode -> {_gib(rec['decode_step_bytes'])}) but only "
                                   f"{_gib(free)} of {_gib(total)} is free next to the engine; lower --engine-arg gpu_memory_utilization to "
                                   f"<= {util_ok:.2f} or cap the step with --engine-arg max_num_batched_tokens=<n> (chunked prefill)")
            self.arena.prealloc()
        memtrace("arena preallocated")
        self.arena_sizing = rec
        return rec

    def step_min_bytes(self, ntok: int, nreq: int) -> int:
        """The block size THIS step needs (called at step begin with the scheduler's token/request counts): a prefill-shaped step
        (more tokens than requests) gets the prefill-sized block, a pure decode step the decode-sized one."""
        rec = self.arena_sizing
        if not rec:
            return 0
        return int(rec["prefill_block_bytes"] if ntok > max(1, nreq) else rec["decode_block_bytes"])

    # ------------------------------------------------------------------ offline core (tests; no engine, no CUDA)

    def commit_offline(self, steps: list[list[tuple]]) -> RunCommitment:
        """Commit the given per-step tensors (name, raw bytes[, dtype, shape]; default uint8 of the byte length) synchronously — the exact
        hashing / layout / root code the online path runs on the worker thread, minus hooks and D2H -- for the security tests and CPU cross-checks."""
        class _Ev:
            def synchronize(self) -> None:
                return None

        self.reset()
        for tensors in steps:
            job = _StepJob(self._step_index, [], _Ev())
            self._step_index += 1
            for name, raw, *typed in tensors:
                first_leaf = (job.metas[-1].first_leaf + job.metas[-1].n_leaves) if job.metas else 0
                n_leaves = (len(raw) + self.chunk - 1) // self.chunk
                dtype, shape = (str(typed[0]), tuple(typed[1])) if typed else ("uint8", (len(raw),))
                m = TensorMeta(len(job.metas), name, dtype, shape, len(raw), self.chunk, first_leaf, n_leaves)
                m.host = memoryview(raw)
                job.metas.append(m)
            job.t_step_end = time.perf_counter()
            self._commit_step(job)
        return self.finalize()

    # ------------------------------------------------------------------ static context

    def _model_geo_digest(self) -> bytes:
        hf = self.engine.llm.llm_engine.vllm_config.model_config.hf_config
        doc = {k: getattr(hf, k, None) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                                                 "head_dim", "intermediate_size", "vocab_size", "model_type")}
        doc["repo"] = self.engine.cp.get("repo")
        doc["revision"] = self.engine.cp.get("revision")
        doc["chunk"] = self.chunk
        return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).digest()

    def _ctx_digest(self, step: int) -> bytes:
        return hashlib.sha256(f"{self.run_id}/step={step}".encode() + self._leaf_ctx(step)).digest()

    def _select_modules(self) -> list[tuple[str, Any]]:
        out = []
        for name, m in self.model.named_modules():
            names = _mro_names(m)
            if names & SKIP_CLASSES:
                continue
            if names & self.acquire_classes:
                out.append((name, m))
        # the sampler lives on the runner, not the model
        smp = getattr(self.runner, "sampler", None)
        if isinstance(smp, torch.nn.Module):
            out.append(("runner.sampler", smp))
        return out

    def apply_skip(self) -> dict[str, list[str]]:
        """VERITY_SKIP on the current selection (`_modules`): drop every selected module of a skipped class and name the committer after
        it.  Under a plan, a consumer-input site on a dropped module stays (`_input_only_modules`): the Value there is not that module's
        output.  Refused (`SkipRefused`) when a skipped class is the runner's sampler (its ids come from the `runner.sample` wrap, which
        also closes every step: no module hook to drop) or drops no selected module.  Returns {class: dropped module names}."""
        skip = self.skip_classes
        if not skip:
            return {}
        label = ",".join(sorted(skip))
        smp = getattr(getattr(self, "runner", None), "sampler", None)
        on_sampler = sorted(_mro_names(smp) & skip) if smp is not None else []
        if on_sampler:
            raise SkipRefused(f"VERITY_SKIP={label}: {on_sampler} is the runner's sampler -- its sampled ids are acquired by the runner.sample "
                              f"wrap, which also closes every step; there is no module hook to drop")
        dropped = {c: [n for n, m in self._modules if c in _mro_names(m)] for c in sorted(skip)}
        unmatched = [c for c, names in dropped.items() if not names]
        if unmatched:
            source = "the acquisition plan" if self.plan_inputs is not None else "the class tables"
            raise SkipRefused(f"VERITY_SKIP={label}: {unmatched} match none of the {len(self._modules)} modules selected by {source}; "
                              f"nothing would be dropped, so the '-skip_' name would claim an ablation that did not happen")
        self._modules = [(n, m) for n, m in self._modules if not (_mro_names(m) & skip)]
        self.skipped = dropped
        tag = f"-skip_{'_'.join(sorted(skip))}"
        if tag not in self.name:
            self.name = f"{self.name}{tag}"
        return dropped

    def _check_skip(self) -> None:
        """At install, when the selection is final: a VERITY_SKIP never applied, or a skipped class back in `_modules`, is refused."""
        if not self.skip_classes:
            return
        label = ",".join(sorted(self.skip_classes))
        if self.skipped is None:
            raise SkipRefused(f"VERITY_SKIP={label}: never applied to the selection of record (apply_plan_to_committer applies it to the "
                              f"plan's selection)")
        back = sorted(n for n, m in self._modules if _mro_names(m) & self.skip_classes)
        if back:
            raise SkipRefused(f"VERITY_SKIP={label}: {len(back)} module(s) of a skipped class are selected again at install ({back[:4]}); "
                              f"the ablation would not happen under its name")

    def _acquires_input(self, name: str, m: Any) -> bool:
        """Whether module `name`'s first positional tensor argument is acquired as `<name>/in`: by class (the embedding's ids,
        ACQUIRE_INPUT_CLASSES) or because the required manifest names it as a promoted member's consumer (`acquire_inputs`).
        [v2 acquire] under a plan (`plan_inputs` set) the plan's consumer-input sites decide, the class table is not consulted."""
        if self.plan_inputs is not None:
            return name in self.plan_inputs
        return bool(_mro_names(m) & ACQUIRE_INPUT_CLASSES) or name in self.acquire_inputs

    def _input_only_modules(self) -> list[tuple[str, Any]]:
        """The manifest's consumer modules the committer did NOT select (no acquire class): their input is acquired, their
        output is not (it is no identity of the Program).  Sorted by name; a consumer that is not a module of the built model is skipped
        here and left unbound (coverage names it).  [v2 acquire] under a plan: the plan's consumer-input sites not among the selected modules."""
        if getattr(self, "model", None) is None:
            return []
        wanted = self.plan_inputs if self.plan_inputs is not None else set(self.acquire_inputs)
        if not wanted:
            return []
        selected = {n for n, _m in self._modules}
        mods = dict(self.model.named_modules())
        return [(n, mods[n]) for n in sorted(wanted) if n not in selected and n in mods]

    @staticmethod
    def layout_digest(metas: list[TensorMeta]) -> bytes:
        # NOTE: packed mode adds stream_off (>= 0); per-tensor mode leaves it at -1 -> the two layouts never collide
        doc = [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes, m.chunk, m.first_leaf, m.n_leaves, m.stream_off] for m in metas]
        return hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode()).digest()

    # ------------------------------------------------------------------ lifecycle

    def register_weights(self) -> dict:
        """Registered input: every parameter + buffer hashed once (rows-sha512: `weights_rows`, SHA-512; the offline core's vllm-v1:
        chunked pos_leaf leaves, per-tensor subtree, fold of subtree roots, SHA-256).  Timed and reported apart from steady state,
        outside its denominator."""
        t0 = time.perf_counter()
        # host (default) | device | device-checked: device hashes rows-sha512's rows into the same per-tensor tree on the GPU, by
        # native_rows.cu (`weights_rows`); the weights tree over the tensor roots is built the same either way.  device-checked computes
        # both and refuses a mismatch.
        mode = settings.VERITY_WEIGHTS_HASH
        if mode not in ("host", "device", "device-checked"):
            raise ValueError(f"VERITY_WEIGHTS_HASH must be host | device | device-checked, got {mode!r}")
        WR = weights_rows.WeightsRows(self._geo_digest, weights_rows.key(getattr(self, "weights_tree", None))) if self.rows else None
        if mode != "host" and WR is None:
            raise ValueError(f"VERITY_WEIGHTS_HASH={mode}: the weights hash on the device under {step_rows.NAME} only (`weights_rows`)")
        roots, names, n_bytes, checked = [], [], 0, 0
        seen: set[int] = set()
        items = list(self.model.named_parameters()) + list(self.model.named_buffers())
        for name, p in items:
            if not isinstance(p, torch.Tensor) or id(p) in seen or p.numel() == 0:
                continue
            seen.add(id(p))
            src = p.detach().contiguous().reshape(-1)          # 0-dim buffers (e.g. scales) -> 1 element
            flat = src.view(torch.uint8) if src.dtype != torch.uint8 else src
            nb = int(flat.numel())
            entry = [name, _dtype_name(p), list(p.shape), nb]
            if WR is not None:
                r, c = WR.root(len(roots), entry, flat, mode)
                checked += c
            else:
                host = flat.cpu()                    # blocking D2H, once
                mv = memoryview(host.numpy()).cast("B")
                leaves = self._hash_leaves(mv)
                r = scheme.fold(leaves)
            roots.append(r)
            n_bytes += nb
            names.append(entry)
        # the tree under the root, kept off the record: a deferred replay opens the weight fields it consumes against it (replay bundle)
        self.weights_tree = WR.tree(names, roots) if WR is not None else {
            "names": names, "tensor_roots": [r.hex() for r in roots], "root": scheme.weights_root(self._geo_digest, scheme.names_digest(names), roots).hex(),
            "chunk": self.chunk, "geo_digest": self._geo_digest.hex()}
        self.weights = {"root": self.weights_tree["root"], "tensors": len(roots), "bytes": n_bytes, "wall_s": time.perf_counter() - t0,
                        "label": "measured; once per process, excluded from steady state", "hash": mode, "chunk": self.chunk,
                        "scheme": self.commit_scheme, **({"device_checked_tensors": checked} if mode == "device-checked" else {})}
        return self.weights

    def attach_source(self, src: Any) -> None:
        """An extra acquisition source (e.g. the FA2 M1 hidden stream): install/uninstall with the committer, contributes to the same step
        stream and run root, and to the coverage declaration."""
        self.sources.append(src)
        self.name = f"{self.name}+{getattr(src, 'short_name', 'hidden_m1')}"

    def flush_after_plane(self, what: str) -> bool:  # noqa: ARG002
        """Hook for acquisition sources whose tensor is overwritten before the next module hook: the arena committer
        snapshots at `_acquire` (D2D on the compute stream), so nothing is staged and nothing needs to leave early -> False.  The gather
        collector (`native_collect.NativeCollectCommitter`) overrides this with the flush."""
        return False

    def install(self) -> None:
        if self.installed:
            return
        self._check_skip()
        self._start_worker()
        self.gc_pauses.install()
        for src in self.sources:
            src.install()
        for name, m in self._modules:
            self._handles.append(m.register_forward_hook(self._make_hook(name)))
            if self._acquires_input(name, m):
                self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
        for name, m in self._input_only_modules():                                # a promoted member's consumer of no acquire class
            self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
        runner = self.runner
        for nm in ("execute_model", "sample"):
            if hasattr(runner, nm):
                self._orig[nm] = getattr(runner, nm)
        orig_exec = self._orig["execute_model"]
        me = self

        def execute_model(scheduler_output, *a, **kw):
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return orig_exec(scheduler_output, *a, **kw)
            me._end_step()                       # a step whose sample() we never saw
            me._plan_shape = (int(scheduler_output.total_num_scheduled_tokens), len(getattr(scheduler_output, "num_scheduled_tokens", ()) or ()))
            me._step_min_bytes = 0 if me.arena_explicit else me.step_min_bytes(*me._plan_shape)   # as native_collect; an explicit ring is one size
            me._begin_step()
            return orig_exec(scheduler_output, *a, **kw)

        self._hooks.patch(runner, "execute_model", execute_model)
        if "sample" in self._orig:
            orig_sample = self._orig["sample"]

            def sample(*a, **kw):
                res = orig_sample(*a, **kw)
                job = me._cur
                if job is not None:
                    # R2i: the sampled token id per step -- the V2 runner's sample() returns (SamplerOutput, ...); the Sampler module is not
                    # invoked through __call__, so a forward hook never sees it: acquire from the returned object (snapshot at once)
                    try:
                        so = res[0] if isinstance(res, tuple) else res
                        st = getattr(so, "sampled_token_ids", None)
                        if isinstance(st, torch.Tensor) and st.device.type == "cuda":
                            me._acquire("runner.sampler/sampled_token_ids", st, job)
                            for s in (s for s in me.sources if hasattr(s, "on_sample")):   # a source of the draw's inputs (served_record)
                                s.on_sample(a[1] if len(a) > 1 else kw.get("input_batch"), int(st.shape[0]), job)
                    except Exception as e:  # noqa: BLE001
                        me.errors.append(f"sampler acquire: {e!r}"[:200])
                    me._end_step()
                return res

            self._hooks.patch(runner, "sample", sample)
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self._end_step()
        self.gc_pauses.uninstall()
        for src in reversed(self.sources):
            src.uninstall()
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self._orig = {}
        self.installed = False

    def reset(self) -> None:
        """Forget the previous run's commitment (roots, levels, values) and recycle the pinned blocks; the engine stays pristine."""
        self.x03_expected = 0
        self.x03_committed = 0
        self._done = {}
        self._levels = {}
        self._layouts = {}
        self._streams = {}
        self._salts = {}
        self._gpu_blocks = {}
        self._gpu_levels = {}
        self._step_index = 0
        self.spans = Spans(scheme.NAME)
        self.gc_pauses.pauses, self.gc_pauses.freeze_s = [], []
        self._reset_views()
        self.errors = []

    def close(self) -> None:
        self.uninstall()
        self._stop_worker()
        self._pool_exec.shutdown(wait=True)

    # ------------------------------------------------------------------ acquisition (serving thread)

    def _begin_step(self) -> None:
        self._cur = _StepJob(self._step_index, [], None, com=self._window_view(), seq=self._next_seq())
        self._step_index += 1
        for src in self.sources:
            src.on_step_begin()
        if self.arena is not None:
            t0 = time.perf_counter()
            self._cur.arena = self.arena.take(int(getattr(self, "_step_min_bytes", 0) or 0))
            self.spans.add("acquisition", time.perf_counter() - t0, step=self._cur.step)
        if self.footprint is not None:
            self.footprint.event("begin_step", step=self._cur.step, arena_block_bytes=int(self._cur.arena.numel()) if self._cur.arena is not None else None,
                                 arena_total_bytes=self.arena.bytes_total if self.arena else 0, arena_free_blocks=[int(b.numel()) for b in self.arena.free] if self.arena else [],
                                 retained_device_total=self._retained_device_bytes())

    def _end_step(self) -> None:
        job = self._cur
        if job is None:
            return
        self._cur = None
        t0 = time.perf_counter()
        if job.acq_s:
            self.spans.add("acquisition", job.acq_s * 0.5, step=job.step)
            self.spans.add("movement", job.acq_s * 0.5, step=job.step, nbytes=job.acq_bytes)   # fast path: split 50/50 (not separately timed)
        if self.inline_d2h and self._rt is not None and not job.overflow and job.cursor:
            # INLINE D2H: the packed step block goes to pinned memory as ONE cudaMemcpyAsync on the compute stream, issued here by the serving
            # thread; the worker then only waits on the event and hashes -- it never touches a CUDA stream (tests the driver-contention hypothesis)
            host = self.pool.take(job.cursor)
            stream = job.stream_handle if job.stream_handle is not None else torch.cuda.current_stream().cuda_stream
            rc = self._memcpy(host.data_ptr(), job.arena.data_ptr(), job.cursor, 2, stream)
            if rc != 0:
                raise RuntimeError(f"cudaMemcpyAsync D2H rc={rc}")
            job.host_pre = host
        if job.deferred_dst:
            torch._foreach_copy_(job.deferred_dst, job.deferred_src)      # one multi-tensor-apply launch for all deferred snapshots
            self.spans.add("movement", 0.0, step=job.step, nbytes=sum(d.numel() for d in job.deferred_dst))
            job.deferred_dst, job.deferred_src = [], []
        job.event = torch.cuda.Event()
        job.event.record(torch.cuda.current_stream())
        job.t_step_end = time.perf_counter()
        job.backlog = self._q.qsize()
        self.staged_bytes += int(job.acq_bytes)
        self.spans.add("acquisition", time.perf_counter() - t0, step=job.step)
        if self.footprint is not None:
            self.footprint.event("end_step", step=job.step, used=int(job.cursor), backlog=job.backlog)
        self._q.put(job)

    ALIGN = 4
    GPU_TREE_SRC = 0                       # hidden-path chunk header field `src_mask` (0: not an FA2 tap stream); the tag field carries the step index

    def commit_block_offline(self, step: int, stream: bytes | bytearray, metas: list[TensorMeta]) -> StepCommitment:
        """CPU reference of the GPU chunk tree for ONE packed step block (tests; no CUDA): the same 256 B chunk leaves (hidden-path chunk header), the
        same production fold, the same retention policy (`retain`), the same per-step state the worker leaves behind, so `open()`,
        `release_staging()`, `verify()` and `finalize()` run unchanged on the result.  Under rows-sha512, `step_rows`'s tree over the
        block's rows, salted from a fresh step key (`row_keys`), as the windowed worker commits it."""
        chunk = self.chunk
        used = len(stream)
        padded = -(-used // chunk) * chunk
        buf = bytes(stream) + b"\0" * (padded - used)
        n = padded // chunk
        if self.rows:
            dom = self._row_domain(step, n)
            lv_bytes = step_rows.levels(dom, [buf[i * chunk:(i + 1) * chunk] for i in range(n)], self._row_key(step), 8 * chunk)
            levels = [torch.frombuffer(bytearray(b"".join(lv)), dtype=torch.uint8).reshape(len(lv), step_rows.ROOT_BYTES).clone() for lv in lv_bytes]
            mr = lv_bytes[-1][0] if lv_bytes else dom.empty_root()
        else:
            layout = self._gpu_layout(step, padded // 4)
            leaves = [scheme.chunk_leaf(scheme.chunk_header(layout, step, i, chunk // 4, self.GPU_TREE_SRC), buf[i * chunk:(i + 1) * chunk]) for i in range(n)]
            lv_bytes = scheme.fold_levels(leaves)
            levels = [torch.frombuffer(bytearray(b"".join(lv)), dtype=torch.int32).reshape(len(lv), 8).clone() for lv in lv_bytes]
            mr = lv_bytes[-1][0]
        keep = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone() if self.retain != "none" else None
        if keep is not None and self.retain == "device":
            keep = keep.cuda()
            levels = [lv.cuda() for lv in levels]
        for m in metas:
            m.stream_off = m.dev_off
        nbytes = sum(m.nbytes for m in metas)
        layout_dig = self.layout_digest(metas)
        root = (step_rows if self.rows else scheme).step_root(self.program_digest, self._ctx_digest(step), self._geo_digest, layout_dig, n, mr)
        self._gpu_levels[step] = levels
        self._gpu_blocks[step] = (keep, used)
        self._layouts[step] = metas
        t = time.perf_counter()
        self._done[step] = StepCommitment(step, root, mr, n, nbytes, len(metas), layout_dig, t, t, 0)
        self._step_index = max(self._step_index, step + 1)
        return self._done[step]

    def _retain_copy(self, arena: Any, padded: int, side: Any) -> Any:
        """The raw bytes kept for openings after the block is absorbed: device copy, host copy (D2H on the side stream; the
        arena block is recycled once the caller's done-event completes), or nothing (openings by replay).  Never touches the leaves."""
        if self.retain == "device":
            keep = torch.empty(padded, dtype=torch.uint8, device=arena.device)
            keep.copy_(arena[:padded])
            return keep
        if self.retain == "host":
            if self.retain_pinned:
                keep = self.pool.take(padded)[:padded]           # pinned pool block -> async D2H at PCIe rate (the caller syncs its done-event)
                keep.copy_(arena[:padded], non_blocking=True)
                return keep
            keep = torch.empty(padded, dtype=torch.uint8, device="cpu", pin_memory=False)
            keep.copy_(arena[:padded], non_blocking=False)      # pageable D2H: completes on the calling stream before returning
            return keep
        return None

    def _levels_to_host(self, levels: list) -> list:
        """Move the tree levels (int32 [n, 8], leaves first) to ONE pinned host block with async copies on the current (side) stream; the
        caller synchronises.  Same rows, same order, so `gpu_open` / `open_levels` read them unchanged."""
        total = sum(int(lv.numel()) for lv in levels)
        flat = self.pool.take(total * 4).view(torch.int32)
        out, off = [], 0
        for lv in levels:
            n = int(lv.numel())
            dst = flat[off:off + n]
            dst.copy_(lv.reshape(-1), non_blocking=True)
            out.append(dst.view(lv.shape))
            off += n
        return out

    def _footprint_absorbed(self, job: _StepJob, padded: int, levels: list, keep: Any) -> None:
        if self.footprint is None:
            return
        self.footprint.event("absorbed", step=job.step, used=int(job.cursor), padded=int(padded),
                             levels_bytes=sum(int(lv.nbytes) for lv in levels), levels_device=bool(levels and levels[0].is_cuda),
                             keep_bytes=int(keep.numel()) if keep is not None else 0, keep_device=(keep.is_cuda if keep is not None else None),
                             retain=self.retain, arena_block_bytes=int(job.arena.numel()) if job.arena is not None else None,
                             retained_device_total=self._retained_device_bytes(), families=_Footprint.families(job.metas),
                             t_step_end=round(job.t_step_end - self.footprint.t0, 6) if job.t_step_end else None, backlog=job.backlog)

    def _retained_device_bytes(self) -> int:
        n = sum(int(lv.nbytes) for lvls in self._gpu_levels.values() for lv in lvls if lv.is_cuda)
        n += sum(int(b.numel()) for b, _u in self._gpu_blocks.values() if b is not None and b.is_cuda)
        return n

    def release_staging(self, *, levels_to_host: bool = True) -> dict[str, int]:
        """Drop every RAW retained copy that still lives on the device (mode `device`) and, optionally, move the tree levels to host.
        After this the device holds only the arena blocks; openings need `open(..., values_from=)` (replay) or a host copy.  Returns bytes."""
        freed = {"raw_device_bytes": 0, "levels_device_bytes": 0}
        for step, (blk, used) in list(self._gpu_blocks.items()):
            if blk is not None and blk.is_cuda:
                freed["raw_device_bytes"] += int(blk.numel())
                self._gpu_blocks[step] = (None, used)
        if levels_to_host:
            for step, lvls in list(self._gpu_levels.items()):
                if lvls and lvls[0].is_cuda:
                    freed["levels_device_bytes"] += sum(int(lv.nbytes) for lv in lvls)
                    self._gpu_levels[step] = [lv.cpu() for lv in lvls]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.released_bytes += freed["raw_device_bytes"] + freed["levels_device_bytes"]
        if self.footprint is not None:
            self.footprint.event("release_staging", **freed, retained_device_total=self._retained_device_bytes())
        return freed

    def _gpu_layout(self, step: int, words: int):
        class _L:                                                 # duck-typed StreamLayout for the hidden-value chunk header: (HB, M, NB) = (step, words, 0)
            HB, M, NB = step, words, 0

        return _L

    # Tensors that vLLM (d9105ea80, Llama architecture) overwrites IN PLACE after the hook that exposes them -- these must be snapshotted at
    # hook time; everything else may be batched at step end (its buffer is kept alive by the held reference):
    #   qkv_proj output      : RotaryEmbedding rotates q and k in place inside the qkv buffer
    #   o_proj / down_proj   : fused_add_rms_norm writes the normalised output into its input buffer x
    #   RMSNorm output[1]    : the residual buffer, overwritten by the next fused norm
    #   embedding output     : becomes the layer-0 residual (same in-place fate)
    # declared lifetime facts: the correspondence annotations carry no alias / in-place facts to derive them from (plan.py: a residual)
    HAZARD_LEAVES = ("qkv_proj", "o_proj", "down_proj", "embed_tokens", "sampler")

    def _hazard(self, name: str) -> bool:
        return hazard_of_record(name, self.HAZARD_LEAVES)

    def _acquire(self, name: str, t: Any, job: _StepJob) -> None:
        if t.device.type != "cuda":
            return
        self.x03_expected += 1                       # attempted acquisition; committed counted only after the copy was issued
        self._acquire_impl(name, t, job)
        self.x03_committed += 1

    def _acquire_impl(self, name: str, t: Any, job: _StepJob) -> None:
        t0 = time.perf_counter()
        if self._rt is not None and t.is_contiguous():
            return self._acquire_by_cudart(name, t, job, t0)
        src = t.detach()
        if not src.is_contiguous():
            src = src.contiguous()          # a device copy: counted under acquisition (bytes noted)
        nbytes = src.numel() * src.element_size()
        if nbytes == 0:
            return
        src = src.reshape(-1)
        flat = src.view(torch.uint8) if src.dtype != torch.uint8 else src
        first_leaf = (job.metas[-1].first_leaf + job.metas[-1].n_leaves) if job.metas else 0
        n_leaves = (nbytes + self.chunk - 1) // self.chunk
        meta = TensorMeta(len(job.metas), name, _dtype_name(t), tuple(int(x) for x in t.shape), nbytes, self.chunk, first_leaf, n_leaves)
        t1 = time.perf_counter()
        if self.arena is not None:
            # D2D snapshot into the step's device block on the compute stream (stream-ordered before any in-place overwrite)
            off = (job.cursor + self.ALIGN - 1) // self.ALIGN * self.ALIGN
            if off + nbytes > job.arena.numel():
                job.overflow.append((job.arena, job.cursor))          # keep the filled block; continue in a fresh (bigger) one
                job.arena = self.arena.take(max(nbytes, self.arena.block_bytes))
                off = 0
            if off > job.cursor:
                job.arena[job.cursor:off].zero_()
            if "nocopy" in self.ABL:
                pass
            elif self.batch_copy and not self._hazard(name):
                # BATCHED: no launch now; the copy is issued as one multi-tensor `_foreach_copy_` at step end.  Valid only for tensors that
                # nothing overwrites in place before then (the value check is the guard); the source ref is held so the allocator cannot reuse it
                job.deferred_dst.append(job.arena[off:off + nbytes])
                job.deferred_src.append(flat)
            else:
                job.arena[off:off + nbytes].copy_(flat)
            meta.dev_off = off
            meta.dev_block = job.arena
            job.cursor = off + nbytes
        else:
            host = self.pool.take(nbytes)
            host.copy_(flat, non_blocking=True)                       # D2H on the compute stream (the "direct" ablation)
            meta.host = memoryview(host.numpy()).cast("B")
        t2 = time.perf_counter()
        job.metas.append(meta)
        self.spans.add("acquisition", (t1 - t0) + (time.perf_counter() - t2), step=job.step)
        self.spans.add("movement", t2 - t1, step=job.step, nbytes=nbytes)

    # ---- acquisition by libcudart: no torch ops on the serving thread at all -- the snapshot is one cudaMemcpyAsync (D2D, stream-ordered
    # on the compute stream) issued through libcudart via ctypes on raw data pointers (~2 us CPU instead of ~15-20 us for slice+view+copy_).
    # Bit semantics identical to the arena copy (the value check compares the committed bytes with hook-time clones).
    def _acquire_by_cudart(self, name: str, t: Any, job: _StepJob, t0: float) -> None:
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            return
        metas = job.metas
        if metas:
            last = metas[-1]
            first_leaf = last.first_leaf + last.n_leaves
        else:
            first_leaf = 0
        chunk = self.chunk
        dn = self._dtype_names.get(t.dtype)
        if dn is None:
            dn = self._dtype_names[t.dtype] = _dtype_name(t)
        meta = TensorMeta(len(metas), name, dn, tuple(t.shape), nbytes, chunk, first_leaf, (nbytes + chunk - 1) // chunk)
        off = (job.cursor + 3) & ~3
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
        job.acq_s += time.perf_counter() - t0                    # one span entry per step (see _end_step); the memcpy issue is inside it
        job.acq_bytes += nbytes

    @staticmethod
    def _load_cudart():
        import glob
        import os as _os

        cands = glob.glob(_os.path.join(_os.path.dirname(torch.__file__), "lib", "libcudart*.so*"))
        try:
            import nvidia.cuda_runtime  # type: ignore

            cands += glob.glob(_os.path.join(_os.path.dirname(nvidia.cuda_runtime.__file__), "lib", "libcudart.so*"))
        except Exception:  # noqa: BLE001
            pass
        cands += ["libcudart.so", "libcudart.so.12", "libcudart.so.13"]
        for c in cands:
            try:
                rt = ctypes.CDLL(c)
                rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
                rt.cudaMemcpyAsync.restype = ctypes.c_int
                rt.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
                rt.cudaMemsetAsync.restype = ctypes.c_int
                return rt, c
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("libcudart not found")

    def _side_stream(self):
        if self._side is None:
            self._side = torch.cuda.Stream()
        return self._side

    def _move_step_to_host(self, job: _StepJob) -> float:
        """Worker: D2H of the step's packed device stream into pinned memory on the side stream; sets meta.host views.  Returns the
        wall seconds spent (issue + wait; overlapped with the engine's compute stream)."""
        t0 = time.perf_counter()
        if job.host_pre is not None:
            job.event.synchronize()                       # the inline D2H (and every snapshot before it) has landed
            hv = memoryview(job.host_pre.numpy()).cast("B")
            for m in job.metas:
                m.stream_off = m.dev_off
                m.host = hv[m.dev_off:m.dev_off + m.nbytes]
                m.dev_block = None
            job.stream = hv[:job.cursor]
            self.arena.release(job.arena)
            job.arena = None
            return time.perf_counter() - t0
        side = self._side_stream()
        with torch.inference_mode(), torch.cuda.stream(side):
            side.wait_event(job.event)
            blocks = job.overflow + [(job.arena, job.cursor)]
            total = sum(used for _b, used in blocks)
            host = self.pool.take(total) if total else None
            hv = memoryview(host.numpy()).cast("B") if total else memoryview(b"")
            base = 0
            for blk, used in blocks:                       # the packed step stream = the used parts of the blocks, contiguous
                if used == 0:
                    continue
                host[base:base + used].copy_(blk[:used], non_blocking=True)
                for m in job.metas:
                    if m.dev_block is blk:
                        m.stream_off = base + m.dev_off
                        m.host = hv[m.stream_off:m.stream_off + m.nbytes]
                base += used
            job.stream = hv[:total]
            ev = torch.cuda.Event()
            ev.record(side)
        ev.synchronize()
        for blk, _used in blocks:
            self.arena.release(blk)
        for m in job.metas:
            m.dev_block = None
        job.overflow = []
        job.arena = None
        return time.perf_counter() - t0

    def _make_hook(self, name: str):
        me = self

        def hook(module, args, output):
            job = me._cur
            if job is None:
                return None
            try:
                if isinstance(output, torch.Tensor):
                    me._acquire(name, output, job)
                elif isinstance(output, (tuple, list)):
                    for k, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            me._acquire(f"{name}/{k}", o, job)
                    if me.regen_row4 and len(output) == 2 and isinstance(output[1], torch.Tensor) and output[1].dtype == torch.bfloat16:
                        # ROW 4 BASELINE candidate: regenerate the exact fp32 conversion of the residual word (c10 BFloat16 -> float == .float())
                        # on the GPU and bind it as its own tensor (+4 B/element).  The derived route binds the same value with no leaf.
                        me._acquire(f"{name}/row4_f32", output[1].float(), job)
                else:
                    st = getattr(output, "sampled_token_ids", None)
                    if isinstance(st, torch.Tensor):
                        me._acquire(f"{name}/sampled_token_ids", st, job)
            except Exception as e:  # noqa: BLE001
                me.errors.append(f"{name}: {e!r}"[:300])
            return None

        return hook

    def _make_pre_hook(self, name: str):
        me = self

        def pre(module, args):
            job = me._cur
            if job is None or not args:
                return None
            try:
                if isinstance(args[0], torch.Tensor):
                    me._acquire(f"{name}/in", args[0], job)
            except Exception as e:  # noqa: BLE001
                me.errors.append(f"{name}/in: {e!r}"[:300])
            return None

        return pre

    # ------------------------------------------------------------------ hashing (worker thread)

    def _hash_leaves(self, mv: memoryview) -> list[bytes]:
        chunk = self.chunk
        n = len(mv)
        if n <= 4 * chunk or self.hash_threads <= 1:
            return [scheme.pos_leaf(mv[o:o + chunk]) for o in range(0, n, chunk)]
        # split into contiguous chunk-aligned ranges for the pool (hashlib releases the GIL for >2 KB updates)
        per = max(chunk, ((n // self.hash_threads) // chunk) * chunk)
        ranges = [(o, min(o + per, n)) for o in range(0, n, per)]

        def part(r):
            lo, hi = r
            return [scheme.pos_leaf(mv[o:min(o + chunk, hi)]) for o in range(lo, hi, chunk)]

        out: list[bytes] = []
        for lst in self._pool_exec.map(part, ranges):
            out.extend(lst)
        return out

    def _pin_thread(self, group: int) -> None:
        """group 0 = serving thread (one core), 1 = worker (one core), 2 = hash pool (the rest, up to 4 cores)."""
        if not self.pin or len(self._cores) < 4:
            return
        c = self._cores
        cores = {0: {c[0]}, 1: {c[1]}, 2: set(c[2:6])}[group]
        try:
            os.sched_setaffinity(0, cores)
        except OSError:
            pass

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self.pin:
            self._pin_thread(0)
        # the CUDA current device is THREAD-LOCAL (default 0 in a new thread): the worker must run on the serving thread's
        # device, or a rank on cuda:1 (TP) creates its side stream on cuda:0 and every step fails (`cudaMemsetAsync(pad): invalid argument`,
        # 0 leaves).
        dev = torch.cuda.current_device()

        def run():
            torch.cuda.set_device(dev)
            self._pin_thread(1)
            self._run_worker()

        self._worker = threading.Thread(target=run, name="cmt-worker", daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker is None:
            return
        self._q.put(self._stop)
        self._worker.join(timeout=60)
        self._worker = None

    def _run_worker(self) -> None:
        while True:
            job = self._q.get()
            if job is self._stop:
                return
            self._absorb_job(job)

    def _absorb_step(self, job: _StepJob) -> None:
        """Worker: commit ``job`` into this committer (or window view); a step that raises is recorded failed, its blocks released."""
        try:
            self._commit_step(job)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"worker step {job.step}: {e!r}"[:300])
            for blk in [job.arena, *(b for b, _u in job.overflow)] if self.arena is not None and job.arena is not None else []:
                self.arena.release(blk)      # the serving thread waits on the ring for the blocks a failed step holds
            self._done[job.step] = StepCommitment(job.step, b"", b"", 0, 0, 0, b"", job.t_step_end, time.perf_counter(), job.backlog)

    # Ablations for the cost decomposition ONLY (env VERITY_ABL, comma list; none of them is a commitment):
    #   nocopy  : hooks + metadata + arena bookkeeping, no D2D snapshot launch      noworker: worker discards the step (no D2H, no hashing)
    #   nohash  : worker does the D2H but skips hashing / roots
    ABL = frozenset(settings.VERITY_ABL)

    def _commit_step(self, job: _StepJob) -> None:
        if self.gpu_tree and job.arena is not None:
            raise RuntimeError(f"step {job.step}: the GPU tree commits a served step under {step_rows.NAME} only, on the windowed "
                               "collector (native_collect_v2b)")
        if "noworker" in self.ABL:
            job.event.synchronize()
            for blk in [job.arena, *(b for b, _u in job.overflow)] if job.arena is not None else []:
                self.arena.release(blk)
            self._layouts[job.step] = job.metas
            self._done[job.step] = StepCommitment(job.step, b"", b"", 0, sum(m.nbytes for m in job.metas), len(job.metas), b"", job.t_step_end,
                                                  time.perf_counter(), job.backlog)
            return
        if self.arena is not None and job.arena is not None:
            mv_s = self._move_step_to_host(job)                       # side stream: D2H of the packed step, overlapped with compute
            self.spans.add("movement", mv_s, step=job.step, nbytes=0)
        else:
            job.event.synchronize()           # every direct D2H copy of this step has landed
        t0 = time.perf_counter()
        leaves: list[bytes] = []
        if job.stream is not None:
            # PACKED: leaves are fixed `chunk`-byte pieces of the step stream, spanning tensor boundaries (identity = byte range ->
            # tensors via the layout); a decode step of ~700 KB is ~11 leaves at 64 KB instead of one per tensor
            leaves = self._hash_leaves(job.stream) if (len(job.stream) and "nohash" not in self.ABL) else []
            n_stream = len(job.stream)
            for m in job.metas:
                m.first_leaf, m.n_leaves = m.stream_off // self.chunk, ((m.stream_off + m.nbytes + self.chunk - 1) // self.chunk) - (m.stream_off // self.chunk)
        elif len(job.metas) >= 2 and self.hash_threads > 1:
            futs = [self._pool_exec.submit(self._hash_leaves_single, m.host) for m in job.metas]
            for f in futs:
                leaves.extend(f.result())
        else:
            for m in job.metas:
                leaves.extend(self._hash_leaves(m.host))
        leaves = self._salted(job.step, leaves)
        levels = scheme.fold_levels(leaves) if leaves else [[scheme.empty_root()]]
        t1 = time.perf_counter()
        n = len(leaves)
        nbytes = sum(m.nbytes for m in job.metas)
        layout_dig = self.layout_digest(job.metas)
        root = scheme.step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, levels[-1][0])
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes, leaves=n)
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        self._levels[job.step] = levels
        self._layouts[job.step] = job.metas
        if job.stream is not None:
            self._streams[job.step] = job.stream
        self._done[job.step] = StepCommitment(job.step, root, levels[-1][0], n, nbytes, len(job.metas), layout_dig, job.t_step_end, t2,
                                              job.backlog, hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    def _hash_leaves_single(self, mv: memoryview) -> list[bytes]:
        chunk = self.chunk
        return [scheme.pos_leaf(mv[o:o + chunk]) for o in range(0, len(mv), chunk)]

    # ------------------------------------------------------------------ finalize / open / verify

    def finalize(self) -> RunCommitment:
        t0 = time.perf_counter()
        self._end_step()
        # wait for the worker to drain the queue (root-ready delay = this wait + the run-root fold); a detached window's, to end it
        with self.spans.span("wait"):
            self.wait_absorbed()
            while len(self._done) < self._step_index:
                time.sleep(0.0005)
        steps = [self._done[s] for s in range(self._step_index)]
        t1 = time.perf_counter()
        step_roots = [s.root for s in steps]    # b"": a placeholder step (a learn-only warm-up pass, or a failed step whose error is recorded)
        run_root = self._fold_run_root(step_roots) or b""
        t2 = time.perf_counter()
        self.spans.add("root_finalisation", t2 - t1)
        last_end = max((s.t_step_end for s in steps), default=t0)
        persistent = sum(32 * len(lv) for lvls in self._levels.values() for lv in lvls) + 32 * len(steps) + sum(map(len, self._salts.values())) + sum(
            len(json.dumps(metas.summary() if hasattr(metas, "scan") else [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes] for m in metas])) for metas in self._layouts.values())
        return RunCommitment(run_root=run_root, steps=steps, spans=self.spans.snapshot(), root_ready_delay_s=t2 - last_end,
                             t_finalize_s=t2 - t0, persistent_bytes=persistent, temporary_bytes_peak=self.pool.bytes_in_use,
                             weight_registration=self.weights,
                             extra={"values_retained_bytes": sum(s.n_bytes for s in steps), "pinned_pool_bytes": self.pool.bytes_total,
                                    "pinned_allocs": self.pool.allocs, "pinned_alloc_s": self.pool.alloc_s, "pinned_pool_mode": self.pool.mode,
                                    "pinned_populate_s": self.pool.times.get("populate_s", 0.0),
                                    "pinned_register_s": self.pool.times.get("register_s", 0.0), "pinned_populate_threads": self.pool.populate_threads,
                                    "pinned_reserved_bytes": self.pool.reserved_bytes, "pinned_reserve_unused": self.pool.reserve_unused,
                                    "pinned_pool_blocks": len(self.pool.all), "errors": self.errors[:10],
                                    "errors_n": len(self.errors), "gc": self.gc_pauses.record(),
                                    # reference integrity (Python acquisition path; the native collector overrides these in its finalize)
                                    "expected_tensors": self.x03_expected, "committed_tensors": self.x03_committed,
                                    "identity_check": self.identity_check() if self._layouts else None,
                                    "hash_threads": self.hash_threads, "chunk": self.chunk, "modules_hooked": len(self._modules), **self._leaf_record(steps),
                                    "staging": self.staging, "fast_copy": self.cudart_path, "gpu_tree": self.gpu_tree, "gpu_tree_ms": dict(self._gpu_t),
                                    "gpu_tree_retained_bytes": self._retained_device_bytes(),
                                    "host_retained_bytes": sum(int(lv.nbytes) for lvls in self._gpu_levels.values() for lv in lvls if not lv.is_cuda)
                                    + sum(int(getattr(b, "retained_bytes", None) if getattr(b, "retained_bytes", None) is not None else b.numel())
                                          for b, _u in self._gpu_blocks.values() if b is not None and not b.is_cuda),
                                    "retain": self.retain, "released_device_bytes": self.released_bytes,
                                    # per-step trace: root prefix, tensors, bytes, identity digest (names) and the first FA2 row
                                    # name -- when two paired runs disagree on the run root, this says whether the SCHEDULE (identities)
                                    # or the VALUES differed and at which step
                                    "step_trace": self._step_trace(steps),
                                    "device_arena_bytes": self.arena.bytes_total if self.arena else 0,
                                    "sources": {getattr(s, "ns_name", type(s).__name__): getattr(s, "stats", {}) for s in self.sources},
                                    "device_arena_allocs": self.arena.allocs if self.arena else 0,
                                    "device_arena_waits": self.arena.waits if self.arena else 0,
                                    "device_arena_wait_s": self.arena.wait_s if self.arena else 0.0})

    def _step_trace(self, steps: list) -> list[dict[str, Any]]:
        out = []
        for s in steps:
            metas = self._layouts.get(s.step) or []
            names = metas.names if hasattr(metas, "scan") else [m.name for m in metas]
            fa2 = next((n for n in names if n.startswith(HIDDEN_M1_PREFIXES)), None)
            out.append({"step": s.step, "root": s.root[:8].hex(), "tensors": s.n_tensors, "bytes": s.n_bytes, "leaves": s.n_leaves,
                        "layout": s.layout_digest[:8].hex(), "fa2": fa2})
        return out

    # ------------------------------------------------------------------ identities: expected manifest vs committed identities

    @staticmethod
    def _module_of(name: str) -> str:
        """Committed tensor name -> the producing module name (tuple slot `/k`, acquired input `/in`, sampler ids stripped)."""
        return name.split("/", 1)[0]

    def expected_manifest(self) -> dict[str, Any]:
        """The identities every committed step MUST cover, derived from the INSTALL-TIME module set (the acquired Instance-output modules of
        the model, selected by class -- `_select_modules`) and the attached hidden sources, not from what the collector emitted:
        every hooked module contributes >= 1 tensor per step, the sampler exactly one `sampled_token_ids`, each FA2 hidden source exactly
        one `fa2.m1/L<k>/...` row per attention layer.  Counts alone cannot see 'missing one, duplicated another'; identities can."""
        modules = [n for n, _m in self._modules]
        # under external acquisition (compiled kernel taps) the module forwards are NOT wrapped: the per-step identities
        # are the source's (P_compiled's kernel launches, `c<piece>.<kernel>`), declared by the source from the required manifest
        ext = [s for s in self.sources if hasattr(s, "expected_modules")]
        if ext and getattr(self, "external_acquisition", False):
            modules = sorted({m for s in ext for m in s.expected_modules()})
        if self.runner is not None and hasattr(self.runner, "sample") and "runner.sampler" not in modules:
            modules.append("runner.sampler")        # sampled ids are acquired from the wrapped `runner.sample`, not from a module hook
        attn_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, m in self._modules if "Attention" in _mro_names(m) and ".layers." in n})
        fa2 = any(getattr(s, "short_name", "") in ("hidden_m1", "hidden_m1_fa3") or type(s).__name__.endswith(("HiddenM1Source", "HiddenM1SourceFA3"))
                  for s in self.sources)                                   # the FA3 tap binds the same one-row-per-layer identity
        # the MoE block source binds exactly TENSORS_PER_LAYER tensors per MoE layer per step (router_w, router_ids, up, act, down, sum), + router_softmax per tapped layer
        moe_src, rs_src = (next((s for s in self.sources if getattr(s, "short_name", "") == n), None) for n in ("moe", "router_softmax"))
        moe_layers = sorted(set(moe_src.layers.values())) if moe_src is not None else []
        # the TP partial source binds exactly ONE rank-local collective input per hooked site per step (install-time site set)
        tp_src = next((s for s in self.sources if getattr(s, "short_name", "") == "tp_partial"), None)
        # the TP source is also the ONE binder of the fused-MoE block's RECEIVED all-reduce output
        # (`model.layers.<L>.mlp.experts/out`; MoERunner is not an ACQUIRE_CLASSES module), the MoE source of a module that does not reduce it
        for sites in (getattr(tp_src, "received_site_names", None), getattr(moe_src, "out_site_names", None)):
            modules += [s for s in (sites() if sites is not None else ()) if s not in modules]
        return {"modules": modules, "attention_layers": attn_layers, "fa2_hidden_per_layer": 1 if fa2 else 0, "sampler_ids": 1 if any(n == "runner.sampler" for n in modules) else 0,
                "moe_layers": moe_layers, "moe_tensors_per_layer": (getattr(moe_src, "stats", {}).get("tensors_per_layer", 6) if moe_src is not None else 0),
                "moe_hash_layers": sorted(getattr(moe_src, "hash_layers", ())),        # a hash layer's router commits one more (`moe_source`: ids, in)
                "router_softmax_layers": sorted(int(k) for k in rs_src.layers) if rs_src is not None else [], "tp_sites": sorted(tp_src.site_names()) if tp_src is not None else [], "tp_rank": (tp_src.rank if tp_src is not None else None),
                # partials per site per step: 1, or a mid-module site's call count (OlmoeAttention: 2 gathers); the calls its instance's flags rule out are listed
                "tp_site_counts": (dict(tp_src.site_counts()) if tp_src is not None and hasattr(tp_src, "site_counts") else {}), "tp_sites_ruled_out": list(getattr(tp_src, "mid_sites_ruled_out", None) or [])}

    def _moe_layer_of(self) -> Any:
        """The MoE source's `layer_of`: the layer of each name it commits (its planes, or its values at calls under a MoE module),
        which it holds to its own rule; None for every name without one.  Without the source (an offline committer), a plane's layer."""
        from verity_vllm.acquire.sources.moe_source import plane_layer
        src = next((s for s in self.sources if getattr(s, "short_name", "") == "moe"), None)
        return getattr(src, "layer_of", None) or plane_layer

    def identity_check(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """The identity check.  For every committed step: (1) every module of the expected manifest produced >= 1 committed tensor and no
        tensor came from a module outside it; (2) the FA2/FA3 hidden source produced, per attention layer, exactly the plane inventory the
        layer's class implies -- DERIVED per layer from the Program's attention Call statics x the step's active sequences when the driver
        handed `attention_layer_decls` (fa2_plane_classes: a windowed layer and a global GQA layer of one step legitimately
        differ), else the uniform rule "every layer the same positive count"; (3) exactly one sampled-ids tensor; (4) the multiset of
        module-tensor NAMES (with multiplicity: a shared RotaryEmbedding is legitimately called once per layer) is the same in every step of
        the run -- the per-step call structure of the model does not depend on the token count.  A step that drops one tensor and
        duplicates another keeps its count and FAILS (1) or (4).  With a route source (`acquire.sources.route_source`: the capture map's
        model), the names the map lists are held to the map's names for the step's routes (`routes_bad`) instead of (1) and (4), and (1)
        covers the hooked modules the map has no entry for.  -> `RunCommitment.extra['identity_check']`."""
        from collections import Counter

        from verity_vllm.acquire import fa2_plane_classes as FP
        man = manifest or self.expected_manifest()
        expected_modules = set(man["modules"])
        if not expected_modules:
            return {"ok": None, "skipped": "no install-time module set (offline core)", "steps_checked": len(self._layouts)}
        violations: list[dict[str, Any]] = []
        ref_names: Counter | None = None
        fa2_rules_used: Counter = Counter()                          # per step: "per-layer" (declarations + active set) or "uniform"
        decls = getattr(self, "attention_layer_decls", None) or []   # the Program's per-layer attention Call statics (driver-supplied)
        # per-step names held to the source's own rule, not rule (4): the call-boundary targets; the capture map's names on the step's routes
        rt_src = next((s for s in self.sources if hasattr(s, "step_modules")), None)
        cb_srcs = [s for s in self.sources if hasattr(s, "split_step_names") and s is not rt_src]
        own = frozenset().union(*(getattr(s, "own_names", ()) for s in self.sources), STEP_DEPENDENT_NAMES)   # what no module produced (KV slots)
        moe_of = self._moe_layer_of()

        # no step-reading source or attention declarations: a step's verdict reads its names, step 0's and the manifest only
        step_free, verdicts = rt_src is None and not cb_srcs and not (decls and man["fa2_hidden_per_layer"]), {}
        for step in sorted(self._layouts):
            metas = self._layouts[step]
            cols = metas.names if hasattr(metas, "scan") else None   # a cached plan's record (plan_cache): its names, no TensorMeta built
            if step_free and cols in verdicts:                         # a plan-name tuple already judged (None is never a key)
                fa2_rules_used.update({"uniform": 1} if man["fa2_hidden_per_layer"] else {})
                violations.extend([{**verdicts[cols], "step": step}] if verdicts[cols] else [])
                continue
            names = Counter(m.name for m in metas) if cols is None else Counter(cols)
            mod_names = Counter({n: c for n, c in names.items() if not n.startswith(HIDDEN_M1_PREFIXES) and not n.startswith("moe/") and moe_of(n) is None and not TP_PARTIAL_RE.search(n) and not CALL_DEPENDENT_RE.search(n) and n not in own})
            tp_bad = tp_partials_bad(man, names)
            mod_names, rt_bad = rt_src.split_step_names(mod_names, step, False, metas) if rt_src is not None else (mod_names, {})
            by_module = Counter()
            for n, c in mod_names.items():
                by_module[self._module_of(n)] += c
            step_expected = rt_src.step_modules(expected_modules) if rt_src is not None else expected_modules
            missing = sorted(step_expected - set(by_module))
            unknown = sorted(set(by_module) - expected_modules)
            fa2_layers = FP.counts_by_layer(names.elements())
            fa2_bad: dict[int, int] = {}
            fa2_detail: dict[str, Any] = {}
            if man["fa2_hidden_per_layer"]:
                seqs = self._step_sequences(step, metas if cols is None else metas.scan()) if decls else None
                if seqs is not None:
                    # each layer's class is DERIVED from the Program's attention Call statics x the step's active-sequence set
                    # (fa2_plane_classes.plane_class): a windowed layer's decode launch is un-swapped (6 planes x sequences) while a global GQA
                    # layer's is swapped (one whole row) -- uniformity across layers is a consequence for one-class Programs, not a rule
                    v = FP.check_step(decls, seqs, names.elements(), self._plane_policy())
                    fa2_bad = v["fa2_layers_bad"]
                    fa2_rules_used["per-layer"] += 1
                    if fa2_bad:
                        fa2_detail = {"rule": v["rule"], "n_seq": v["n_seq"], "max_seqlen_q": v["max_seqlen_q"], "exact_seqlens": v["exact_seqlens"],
                                      "classes": v["classes"], "layers_bad": {k: d[:3] for k, d in list(v["layers_bad"].items())[:4]},
                                      "undeclared_layers": dict(list(v["undeclared_layers"].items())[:4])}
                else:
                    # uniformity rule (no declarations, or no record of the step's active sequences): one whole-buffer row per layer, or (per-sequence
                    # compaction) 6 planes x sequences per layer -- every attention layer of the step must carry the SAME positive count
                    fa2_bad = FP.uniform_rule(fa2_layers, man["attention_layers"])
                    fa2_rules_used["uniform"] += 1
            moe_bad: dict[int, int] = {}
            if man.get("moe_layers"):
                moe_counts = Counter(k for n in names.elements() if (k := moe_of(n)) is not None)
                want, rs_layers, hash_layers = (int(man.get("moe_tensors_per_layer") or 0), set(man.get("router_softmax_layers") or []),
                                                set(man.get("moe_hash_layers") or []))
                moe_bad = {k: moe_counts.get(k, 0) for k in man["moe_layers"] if moe_counts.get(k, 0) != want + (k in rs_layers) + (k in hash_layers)}
                moe_bad.update({k: c for k, c in moe_counts.items() if k not in man["moe_layers"]})
            elif any(n.startswith("moe/") for n in names):
                moe_bad = {-1: sum(c for n, c in names.items() if n.startswith("moe/"))}      # MoE tensors from no declared source
            n_ids = names.get("runner.sampler/sampled_token_ids", 0)
            sampler_bad = bool(man["sampler_ids"]) and n_ids != man["sampler_ids"]
            mod_names, cb_bad = _split_step_names(cb_srcs, mod_names, step)
            split_bad = {k: bad for k, bad in (("call_boundaries", cb_bad), ("routes", rt_bad)) if bad}
            ref_names = mod_names if ref_names is None else ref_names
            diff = {n: (mod_names.get(n, 0), ref_names.get(n, 0)) for n in set(mod_names) | set(ref_names) if mod_names.get(n, 0) != ref_names.get(n, 0)}
            if missing or unknown or fa2_bad or moe_bad or tp_bad or sampler_bad or diff or split_bad:
                violations.append({"step": step, "missing_modules": missing[:8], "unknown_modules": unknown[:8], "fa2_layers_bad": fa2_bad,
                                   "moe_layers_bad": moe_bad, "tp_sites_bad": dict(list(tp_bad.items())[:8]), "tp_sites_bad_n": len(tp_bad),
                                   "sampler_ids": n_ids, "multiplicity_vs_step0": dict(sorted(diff.items())[:8]),
                                   **{k: v for sn, bad in split_bad.items() for k, v in ((f"{sn}_bad", dict(sorted(bad.items())[:8])), (f"{sn}_bad_n", len(bad)))},
                                   **({"fa2_detail": fa2_detail} if fa2_detail else {})})
            if step_free and cols is not None:
                verdicts[cols] = violations[-1] if violations and violations[-1]["step"] == step else None
        return {"ok": not violations, "steps_checked": len(self._layouts),
                "manifest": {"modules": len(expected_modules), "attention_layers": len(man["attention_layers"]), "fa2_hidden_per_layer": man["fa2_hidden_per_layer"],
                             "sampler_ids": man["sampler_ids"], "moe_layers": len(man.get("moe_layers") or []), "moe_tensors_per_layer": man.get("moe_tensors_per_layer", 0),
                             "tp_sites": len(man.get("tp_sites") or []), "tp_rank": man.get("tp_rank"), "tp_sites_ruled_out": man.get("tp_sites_ruled_out") or []},
                # which attention-plane rule each step ran and what it was derived from
                "fa2_rule": {"steps": dict(fa2_rules_used), "attention_decls": len(decls),
                             "policy": {k: v for k, v in self._plane_policy().__dict__.items() if k != "ms" or v} if decls else None,
                             "classes_declared": (sorted(Counter(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha")
                                                                 for d in decls).items()) if decls else None)},
                "violations": violations[:5], "violations_n": len(violations)}

    def _plane_policy(self):
        """The collector's hidden-plane staging rule (`fa2_plane_classes.LayoutPolicy`): the explicit override, else read from
        the installed hidden source -- the native FA2 source's `Fa2Wrap.compact` (per-sequence compaction), the Python source stages whole
        buffers, the FA3 sources stage whole PackGQA launches."""
        from verity_vllm.acquire import fa2_plane_classes as FP

        if getattr(self, "plane_policy", None) is not None:
            return self.plane_policy
        for s in getattr(self, "sources", None) or []:
            sn = getattr(s, "short_name", "")
            tn = type(s).__name__
            if sn == "hidden_m1_fa3" or tn.endswith("FA3"):
                return FP.LayoutPolicy(backend="fa3", compact=False, ms=bool(getattr(s, "ms", False)))
            if sn == "hidden_m1" or tn.endswith("HiddenM1Source"):
                w = getattr(s, "_w", None)
                return FP.LayoutPolicy(backend="fa2", compact=bool(getattr(w, "compact", False)) if w is not None else False, ms=bool(getattr(s, "ms", False)))
        return FP.LayoutPolicy()

    def _step_sequences(self, step: int, metas: list) -> Any:
        """The step's active-sequence set (`fa2_plane_classes.StepSequences`): the collector's row record `step_rows[step]`
        (BIND-01: [(request, row_lo, row_hi)] in launch order) when present; else (n_seq, tokens) read from the executed layout -- the
        sampled-ids member has one slot per co-scheduled sequence, the integer ids input one row per token; None when neither is known
        (the caller then runs the uniform rule for the step).  Never read from the hidden planes themselves."""
        from verity_vllm.acquire import fa2_plane_classes as FP

        rows = (getattr(self, "step_rows", None) or {}).get(int(step))
        if rows:
            return FP.StepSequences.from_rows(rows)
        n_seq = tokens = None
        for m in metas:
            numel = int(math.prod(m.shape)) if m.shape else 0
            if m.name == "runner.sampler/sampled_token_ids":
                n_seq = numel
            elif m.name.endswith("/in") and tokens is None and str(m.dtype).replace("torch.", "") in ("int32", "int64", "i32", "i64", "long", "int"):
                tokens = numel
        if n_seq is None or tokens is None:
            return None
        return FP.StepSequences.from_counts(n_seq, tokens)

    def layouts_dump(self) -> dict[int, list[dict[str, Any]]]:
        """The published per-step layouts as plain dicts ({step: [{name, dtype, shape, nbytes, numel, first_leaf, n_leaves, stream_off}]}) --
        the collector's CLAIM that the manifest's `required_manifest.coverage_check` and bind's binding map are checked against."""
        out: dict[int, list[dict[str, Any]]] = {}
        for step in sorted(self._layouts):
            out[int(step)] = [{"name": m.name, "dtype": m.dtype, "shape": list(m.shape), "nbytes": int(m.nbytes), "numel": int(math.prod(m.shape)) if m.shape else 0,
                               "first_leaf": int(m.first_leaf), "n_leaves": int(m.n_leaves), "stream_off": int(m.stream_off)} for m in self._layouts[step]]
        return out

    def identity_of(self, step: int, index: int) -> dict:
        """Verifier-side: (step, rank) -> (tensor ordinal, name, dtype, shape, byte range) from the published layout."""
        metas = self._layouts[step]
        if step in self._gpu_blocks or step in self._streams:      # packed: leaf = stream bytes [lo, hi); tensors overlapping it
            n = self._gpu_blocks[step][1] if step in self._gpu_blocks else len(self._streams[step])
            lo, hi = index * self.chunk, min((index + 1) * self.chunk, n)
            if lo >= n:
                raise IndexError((step, index))
            over = [{"tensor": m.ordinal, "name": m.name, "dtype": m.dtype, "shape": list(m.shape),
                     "first_byte": max(lo, m.stream_off) - lo, "last_byte": min(hi, m.stream_off + m.nbytes) - lo}
                    for m in metas if m.stream_off < hi and m.stream_off + m.nbytes > lo]
            return {"packed": True, "byte_lo": lo, "byte_hi": hi, "tensors": over,
                    "tensor": over[0]["tensor"] if over else -1, "name": ";".join(o["name"] for o in over)}
        for m in metas:
            if m.first_leaf <= index < m.first_leaf + m.n_leaves:
                k = index - m.first_leaf
                lo = k * m.chunk
                hi = min(lo + m.chunk, m.nbytes)
                return {"tensor": m.ordinal, "name": m.name, "dtype": m.dtype, "shape": list(m.shape), "byte_lo": lo, "byte_hi": hi}
        raise IndexError((step, index))

    @staticmethod
    def _block_holds(blk: Any, lo: int, hi: int) -> bool:
        """Does the retained block hold bytes [lo, hi)?  A contiguous tensor holds everything; the sparse host retainers (`_HostWindows`
        with retain=sample, `_HostRanges` with retain=host + exclude) answer per range."""
        if blk is None:
            return False
        holds = getattr(blk, "holds", None)
        return True if holds is None else bool(holds(lo, hi))

    def leaf_retained(self, step: int, index: int) -> bool:
        """True when leaf `index` of `step` opens from THIS committer's retained raw bytes; False when it must open by replay
        (raw bytes released / never kept / an excluded family's interior chunk)."""
        if step not in self._gpu_blocks:
            return step in self._streams or bool(self._layouts.get(step))
        blk, _used = self._gpu_blocks[step]
        lo = index * self.chunk
        return self._block_holds(blk, lo, lo + self.chunk)

    def open(self, sample: list[tuple[int, int]], *, values_from: "NativeHostCommitter | None" = None,
             paths_from_replay: bool = False) -> list[Opening]:
        """Openings for (step, leaf) positions.  The value bytes come from this committer's retained copy (device or host), or -- when the raw
        copy was released / never kept (`retain=none`), or the leaf lies in a range that was not retained (retain=sample; retain=host with
        excluded families) -- from `values_from`: a committer that REPLAYED the same committed inputs (same layouts, same bytes under BI
        determinism).  The sibling path comes from THIS committer's levels, or -- `paths_from_replay` (regenerated values AND
        authentication-path material) -- from the replay's levels; either way `verify()` checks the result against THIS run's original
        roots, so a replay that differs in any byte (value or tree) fails to open."""
        out = []
        with self.spans.span("opening_generation"):
            for step, index in sample:
                metas = self._layouts[step]
                ident = self.identity_of(step, index)
                if step in self._gpu_blocks:
                    blk, used = self._gpu_blocks[step]           # lazy D2H of the 256 B chunk (zero-padded past `used`) + level sibling path
                    lo = index * self.chunk
                    levels = self._gpu_levels.get(step)
                    if not self._block_holds(blk, lo, lo + self.chunk) or (paths_from_replay and values_from is not None):
                        if values_from is None or step not in values_from._gpu_blocks:
                            raise RuntimeError(f"open({step}, {index}): raw bytes released / not retained (retain={self.retain}) and no replay source holds step {step}")
                        rblk, rused = values_from._gpu_blocks[step]
                        if not self._block_holds(rblk, lo, lo + self.chunk):
                            raise RuntimeError(f"open({step}, {index}): the replay source does not hold the leaf's bytes either (retain={values_from.retain})")
                        if rused != used or values_from.layout_digest(values_from._layouts[step]) != self.layout_digest(metas):
                            raise RuntimeError(f"open({step}, {index}): replay layout differs from the committed layout")
                        blk = rblk                                   # the regenerated bytes (the retained copy, if any, is not consulted)
                        if paths_from_replay:
                            if step not in values_from._gpu_levels:
                                raise RuntimeError(f"open({step}, {index}): the replay source has no tree levels for step {step}")
                            levels = values_from._gpu_levels[step]
                    if levels is None:
                        raise RuntimeError(f"open({step}, {index}): this run kept no tree levels (retain_levels=False): open with a replay "
                                           "source and paths_from_replay=True")
                    value = bytes(blk[lo:lo + self.chunk].cpu().numpy().tobytes())
                    out.append(Opening(step, index, value, tuple(self._level_path(levels, index)), ident, self._gpu_salt(step, index)))
                    continue
                if ident.get("packed"):
                    value = bytes(self._streams[step][ident["byte_lo"]:ident["byte_hi"]])
                else:
                    m = metas[ident["tensor"]]
                    value = bytes(m.host[ident["byte_lo"]:ident["byte_hi"]])
                path = scheme.path_in_levels(self._levels[step], index)
                out.append(Opening(step, index, value, path, ident, self._salts_of(step, index, index + 1)))
        return out

    def _level_path(self, levels: list, index: int) -> tuple:
        """Leaf `index`'s sibling path in a GPU-tree step's levels."""
        return tuple(open_levels(levels, index))

    def verify(self, run: RunCommitment, opening: Opening) -> bool:
        """Production verifier core: shape rule + fold to the bare root, then the context binding to the ORIGINAL step root, then the
        step root's membership in the run root."""
        if self.rows:
            return self._verify_rows(run, opening)
        try:
            sc = run.steps[opening.step]
        except IndexError:
            return False
        if not 0 <= opening.index < sc.n_leaves:
            return False
        try:
            shape = scheme.path_shape(sc.n_leaves, opening.index)
        except Exception:  # noqa: BLE001
            return False
        if len(opening.path) != len(shape) or any((p is None) == has for p, has in zip(opening.path, shape, strict=True)):
            return False
        ident = self.identity_of(opening.step, opening.index)
        if ident != opening.identity or (self.hiding is None and opening.salt) or self._unsalted(opening.step, sc.n_leaves):
            return False
        if opening.step in self._gpu_blocks:
            used = self._gpu_blocks[opening.step][1]
            if len(opening.value) != self.chunk:
                return False
            padded = -(-used // self.chunk) * self.chunk
            leaf = scheme.chunk_leaf(scheme.chunk_header(self._gpu_layout(opening.step, padded // 4), opening.step, opening.index, self.chunk // 4,
                                                        self.GPU_TREE_SRC), opening.value)
            merkle = scheme.fold_path(leaf, opening.index, list(opening.path))
        else:
            leaves = self._leaf_rule([scheme.pos_leaf(opening.value)], opening.salt)
            if leaves is None or len(opening.value) != ident["byte_hi"] - ident["byte_lo"]:
                return False
            merkle = scheme.fold_path(leaves[0], opening.index, list(opening.path))
        root = scheme.step_root(self.program_digest, self._ctx_digest(opening.step), self._geo_digest,
                             self.layout_digest(self._layouts[opening.step]), sc.n_leaves, merkle)
        if root != sc.root:
            return False
        return len(run.run_root) == 32 and self._run_root(run) == run.run_root

    def _leaf_record(self, steps: list) -> dict:
        rec = super()._leaf_record(steps)
        return {**rec, "commit_scheme": self.commit_scheme} if self.rows else rec

    # ------------------------------------------------------------------ coverage

    # module (class-name) -> census root families whose NATURALLY-MATERIALIZED outputs the hook on that module binds
    FAMILY_OF_MODULE = {
        LC.LINEAR_GROUP: ("Gemm_v1",), LC.RMS_NORM: ("RMSNormFusedCuda_v2", "RMSNormTriton_v1"), LC.GEMMA_RMS_NORM: ("RMSNormFusedCuda_v2", "RMSNormTriton_v1"),
        LC.SILU_AND_MUL: ("SiluMul_v1", "SiluMul_v2"), LC.GELU_AND_MUL: ("GeluMul_v1",), LC.ROTARY: ("RoPE_v1",), LC.ATTENTION: ("Attention_v3",),
        LC.EMBEDDING: ("Embedding_v1",), LC.LOGITS: ("Gemm_v1",), LC.SAMPLER: ("TokenSelect_v1",),
    }

    def _module_class(self, name: str) -> str:
        base = name.split("/")[0]
        if base == "runner.sampler":
            return LC.SAMPLER
        for n, m in self._modules:
            if n == base:
                names = _mro_names(m)
                for k in self.FAMILY_OF_MODULE:
                    if k in names or (k == LC.LINEAR_GROUP and names & {LC.LINEAR_BASE, LC.COLUMN_LINEAR, LC.ROW_LINEAR}):
                        return k
                return type(m).__name__
        leaf = base.rsplit(".", 1)[-1]
        if not self._modules and "norm" in leaf.lower():      # offline core (tests): infer from the module path
            return LC.RMS_NORM
        return leaf

    def coverage_declaration(self, run: RunCommitment) -> dict:
        by_class: dict[str, dict] = {}
        families: set[str] = set()
        moe_of = self._moe_layer_of()
        for step, metas in self._layouts.items():
            for m in metas:
                if m.name.startswith(("fa2.m1/", "fa3.m1/", "moe/")) or moe_of(m.name) is not None or m.name.endswith("/row4_f32"):
                    continue                                  # declared separately (hidden source / MoE block source / row-4 regenerate baseline)
                mc = self._module_class(m.name)
                compiled_kernel = len(m.name) > 3 and m.name[0] == "c" and m.name[1:3].isdigit() and m.name[3] == "."   # compiled kernel: c<piece>.<kernel>/<role>
                if m.name.endswith("/in") or (compiled_kernel and m.name.endswith("/ids")):
                    cls, fam = "input_token_ids", (TOKEN_IDS_INPUT,)
                elif "sampled_token_ids" in m.name:
                    cls, fam = "sampled_token_ids", ("TokenSelect_v1",)
                elif compiled_kernel:
                    # P_compiled's VU boundaries: every non-weight argument of every Inductor kernel launch (the eager
                    # module-output families do not describe this execution; the kernel relations are compiled_relations.py's)
                    cls, fam, mc = "compiled_kernel_buffers", (), m.name.split("/", 1)[0].split(".", 1)[1].rsplit("_", 1)[0]
                else:
                    cls, fam = "instance_outputs_native", self.FAMILY_OF_MODULE.get(mc, ())
                d = by_class.setdefault(cls, {"count": 0, "bytes": 0, "tensors": 0, "modules": set(), "families": set()})
                d["bytes"] += m.nbytes
                d["tensors"] += 1
                d["modules"].add(mc)
                d["families"].update(fam)
                families.update(fam)
        classes = []
        for cls, d in by_class.items():
            classes.append({"class": cls, "how": "bound-as-leaf", "rule": None, "count": None, "bytes": d["bytes"], "tensors": d["tensors"],
                            "covers_families": sorted(d["families"]), "acquisitions": ["naturally-materialized", "registered-input"],
                            "evidence": f"forward hooks on {sorted(d['modules'])}; pos_leaf chunk={self.chunk}; per-step bind_root; run root"})
        if self.weights:
            classes.append(weights_class(self.weights, getattr(self, "weights_tree", None)))
        # derived-from-bound classes whose SOURCE is a naturally-materialized word this committer binds (norm residual bf16 -> fp32 lift):
        # declared as covered ONLY through the derived-value rule (a protocol clause; cited), which the checker reports in its own status
        if "RMSNormFusedCuda_v2" in families and self.regen_row4:
            nb = sum(m.nbytes for metas in self._layouts.values() for m in metas if m.name.endswith("/row4_f32"))
            classes.append({"class": "norm_bf16tof32_regenerated", "how": "bound-as-leaf",
                            "covers": ["rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo2"],
                            "rule": None, "count": None, "bytes": nb, "tensors": sum(1 for metas in self._layouts.values() for m in metas if m.name.endswith("/row4_f32")),
                            "evidence": "baseline candidate: residual.float() regenerated in the RMSNorm hook and bound as a tensor of the step stream"})
        if "RMSNormFusedCuda_v2" in families:
            classes.append({"class": "norm_bf16tof32_lift", "how": "derived-from-bound",
                            "covers": ["rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormfusedcuda2/int/bf16tof32_v1/bf16tof32_fo2",
                                       "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo1", "rmsnormtriton1/int/bf16tof32_v1/bf16tof32_fo2"],
                            "rule": "out/gen/r8/fa2-commit/derived_value_rule.md §1 (Bf16ToF32 injective lift of the bound bf16 residual word; "
                                    "every consumer lowered to read the source and apply phi) — a PROTOCOL CLAUSE, not a hash; reported separately",
                            "count": None, "bytes": None, "evidence": "source = the RMSNorm residual output tensor bound above"})
        if "RMSNormFusedCuda_v2" in families:
            # R10a / R10b (norm kernel interiors): DECLARED treatment only.  Every interior fp32 value of fused_add_rms_norm_kernel (squares,
            # cub block-reduction partials, variance/N, rsqrtf, x*rstd) and of the layer-0 Triton _rms_norm_kernel is a deterministic function
            # of the BOUND source row (the token's residual/embedding bf16 row, R2e / R2b) and the registered weight row under the pinned
            # kernel's fixed reduction order = the registry's exact model (`verity/ml/definitions/registry/b1.py` RMSNormFusedCuda v2, 2340/2340 rows
            # conformant).  So the derived-value route applies with a ROW-sourced phi: opening = source-row leaf(s) + weight opening + op-by-op
            # replay; no interior word needs a leaf.  The verifier for that multi-source phi is NOT implemented -> the checker
            # reports 'derived-declared', never secure.  Alternative = M1-style export of 47.6 MB/request (not built).
            classes.append({"class": "norm_interiors_derived_row", "how": "derived-from-bound", "verified": False,
                            "covers": ["rmsnormfusedcuda2/int/f32mul_v1/f32mul_fo1", "rmsnormfusedcuda2/int/f32add_v1/f32add_fo1",
                                       "rmsnormfusedcuda2/int/f32div_v1/f32div_fo1", "rmsnormfusedcuda2/int/rsqrtapprox_v1/rsqrtapprox_fo2",
                                       "rmsnormtriton1/int/f32add_v1/f32add_fo2", "rmsnormtriton1/int/f32fma_v1/f32fma_fo1",
                                       "rmsnormtriton1/int/f32mul_v1/f32mul_fo1", "rmsnormtriton1/int/f32fma_v1/f32fma_fo2",
                                       "rmsnormtriton1/int/f32add_v1/f32add_fo1", "rmsnormtriton1/int/divfullrcp_v1/divfullrcp_fo1",
                                       "rmsnormtriton1/int/mufusqrtftz_v1/mufusqrtftz_fo2", "rmsnormtriton1/int/divfullscalea_v1/divfullscalea_fo1",
                                       "rmsnormtriton1/int/f32mul_v1/f32mul_fo2"],
                            "rule": "derived-row rule (candidate, NOT verified here): phi = registry exact model of the pinned norm kernel over the bound "
                                    "source row (R2e residual / R2b embedding bf16 row) + registered weight row; opening = source-row leaf + weight "
                                    "opening + replay; verifier + binding/value tests not implemented in this lane",
                            "count": None, "bytes": None, "evidence": "source rows bound above; exact-model conformance from verity.ml.circuits registry (row-4 work)"})
        for src in self.sources:
            classes.extend(src.declaration_classes())
        notes = ["binds ONLY naturally materialised values (module outputs in HBM); FA2 interior (hidden-fa2) and norm kernel interiors "
                 "(hidden-vllm-kernel) are NOT bound by this committer — the hidden path (CMT-2) covers those classes",
                 "value identity = (step, tensor ordinal, byte range) via the published per-step layout whose digest is in the step root"]
        return declaration(self.name, "native-host-sha256", classes, notes)
