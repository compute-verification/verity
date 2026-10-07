"""Reference committer for the NATURALLY MATERIALISED interface class (Call outputs in HBM + input ids + sampled ids),
host-hashed.  Glossary (Draft 3 §12): value acquisition (the hooks) -> commitment (the position-identified Merkle roots); the
class tables `ACQUIRE_CLASSES` / `FAMILY_OF_MODULE` / `HAZARD_LEAVES` are a hand-written runtime correspondence (module class ->
Definition family) and are pinned by `tests/test_no_by_name_rules.py` until Phase 3/4 record it as data.

EXPERIMENTAL (R9 CMT-3).  This is the "correct first, then fast" reference of DANIEL-BRIEF A2 for the class the census calls
`HBM-materialized (native)`; it binds NOTHING hidden inside FA2 / the norm kernels (that is CMT-2's path) and its
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

from verity_vllm.commit import hiding, scheme
from verity_vllm.commit.committer.input_classes import TOKEN_IDS_INPUT, weights_class
from verity_vllm.commit.committer.pinned_pool import PINNED_EXACT, PinnedPool
from verity_vllm.commit.native_ranges import NativeRangeOpenings, open_levels
from verity_vllm.acquire.committer_api import Opening, RunCommitment, StepCommitment, declaration
from verity_vllm.engine import hooks
from verity_vllm.pipeline.spans import Spans

GPU_CHUNK_BYTES = 256
DEFAULT_CHUNK = 4096

# the construction-time selection: TP1 Commits replace it with the acquisition plan's (apply_plan_to_committer; compiled rows:
# CompiledGraphSource.wrap_model); TP rank committers (engine/rank_worker.py) have no plan yet and keep it
# module classes whose forward OUTPUT is an Instance output (census §3 `HBM-materialized (native)`); matched on the MRO class names
ACQUIRE_CLASSES = {
    "ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear", "LinearBase",
    "RMSNorm", "GemmaRMSNorm", "LayerNorm",
    "SiluAndMul", "GeluAndMul", "GeluAndMulSparse", "MulAndSilu",
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
    return frozenset(filter(None, os.environ.get("VERITY_SKIP", "").split(",")))


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


@dataclass
class _StepJob:
    step: int
    metas: list[TensorMeta]
    event: Any
    t_step_end: float = 0.0
    backlog: int = 0
    blocks: list = field(default_factory=list)
    arena: Any = None                       # arena staging: current device block
    cursor: int = 0                         # arena staging: bytes used in `arena`
    overflow: list = field(default_factory=list)   # arena staging: (block, used_bytes) of earlier, filled blocks of this step
    stream: Any = None                      # packed mode: memoryview of the whole packed step stream (pinned host copy)
    stream_handle: Any = None               # fast path: raw cudaStream_t of the compute stream (cached per step)
    arena_ptr: Any = None                   # fast path: data_ptr of job.arena (cached)
    host_pre: Any = None                    # inline-D2H: pinned host block already receiving the packed step (issued on the compute stream)
    acq_s: float = 0.0                      # fast path: serving-thread seconds inside the hooks (acquisition + memcpy issue), this step
    acq_bytes: int = 0
    deferred_dst: list = field(default_factory=list)   # batched acquisition: arena views awaiting the step-end _foreach_copy_
    deferred_src: list = field(default_factory=list)   # ... and their (held) sources


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
        self._extra = list(extra_blocks or [])       # X-02: additional preallocated blocks of other sizes (one prefill-sized block)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.waits = 0                                # P0 backpressure: how often / how long the serving thread waited for a block
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

    # P0 (R13) backpressure: the ring never grows past its preallocation while a fitting block is merely BUSY; the serving thread waits for
    # the worker to release one (the engine stalls instead of the collector OOMing).  Growth only when NO block of the ring can ever fit (a
    # step larger than every block), with VERITY_ARENA_GROW=1 (R12), or `wait=False`: the worker, the only releaser, must not wait on itself.
    GROW = os.environ.get("VERITY_ARENA_GROW", "0").strip() in ("1", "true", "yes")
    WAIT_TIMEOUT_S = float(os.environ.get("VERITY_ARENA_WAIT_S", "600"))

    def _best_free(self, min_bytes: int) -> int:
        best = -1
        for i, b in enumerate(self.free):
            if b.numel() >= min_bytes and (best < 0 or b.numel() < self.free[best].numel()):
                best = i
        return best

    def take(self, min_bytes: int = 0, *, wait: bool = True) -> Any:
        # best fit: the SMALLEST free block that holds the step (X-02: decode steps must not consume the prefill-sized block)
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
# R13 P0 (C-COLLECTOR-FOOTPRINT).  Two things, both OFF by default (roots/layouts/openings are unchanged in every mode):
#
#  * `VERITY_FOOTPRINT=<path.jsonl>` -- a per-step device-memory timeline written from the serving thread (begin/end of step) and the worker
#    (absorb = hashing done, release = arena block back in the pool): bytes staged by value family, arena block sizes, retained copies and
#    tree levels, torch allocated/reserved.  Design input for the streaming absorb/release work; no effect on what is bound.
#  * retention policy `VERITY_RETAIN=device|host|none` (commit_delta `--retain`): WHERE the raw staged bytes live after the step is
#    cryptographically absorbed (leaves + tree computed, root bound).  `device` = the R9..R12 behaviour (a compact device copy per step, kept
#    for lazy openings -- this is the ~9.2 GB/seq that OOMs Llama 1024/128 at B=2); `host` = the copy goes to host memory (D2H on the side
#    stream, then the arena block is recycled) and the tree levels move to host too, so the device holds only the arena blocks in flight;
#    `none` = no raw copy at all: only the authentication state (levels) is kept on host, and openings are answered by REPLAY -- a second
#    deterministic (BI) execution of the same committed inputs reproduces the same bytes and the same roots (`open(..., values_from=other)`),
#    verified against the ORIGINAL run's roots.  One-stage binding is untouched: the leaves are hashed from the arena block before any copy.
RETAIN_MODES = ("device", "host", "none", "sample")
# [R13 hopper] the attention hidden-value rows of either FlashAttention tap (FA2 v8 `fa2.m1/L<k>/...`, FA3 v9 `fa3.m1/L<k>/...`): one identity
# family for the X-03 identity check, the step trace and the footprint families
HIDDEN_M1_PREFIXES = ("fa2.m1/", "fa3.m1/")
TP_PARTIAL_RE = re.compile(r"/(part|shard)_rank\d+$")   # [R14 TP-03] <module path>/part_rank<r>: the rank-local input of that module's collective


def hazard_of_record(name: str, HAZARD_LEAVES: "tuple[str, ...] | frozenset[str] | set[str]") -> bool:
    """v1's in-place HAZARD decision for one acquired tensor name (`<module path>/<slot>`, a tap plane or a TP partial): snapshot at the hook
    (True) or batch at step end (False).  Encodes vLLM's aliasing by leaf name (`NativeHostCommitter.HAZARD_LEAVES` + the two name rules) --
    a by-name lifetime table (F-r19-int-20, Draft 3 §11.3.1), owned HERE; `verity_vllm.acquire.plan` names it as a RESIDUAL through this one
    function rather than restating it."""
    if TP_PARTIAL_RE.search(name):
        return True          # [R14 TP-03] a collective's input partial: snapshot now, before the NCCL call consumes it
    if name.startswith("moe/") and name.endswith("/up"):
        return True          # [R16 moe FLUSH-03] the fused-MoE up plane: its workspace is the block's own w2 output (cache1/cache3 alias)
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
# X-02 (R12): the arena block is sized from the WORKLOAD ENVELOPE at arm time instead of the fixed 512 MiB.  One engine step must fit
# in one block (the native collector has no overflow chaining); a step larger than the block was refused mid-step ("arena block too
# small for this step"), which left the step root incomplete.  Nothing about what is bound, the packed layout, lifetimes or openings
# changes: the block is just big enough, so roots for an unchanged workload are identical before/after.
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
    """[R13 MOE-03] bytes per token per step the MoE block source binds: per MoE layer TOPK x (f32 weight + i32 id) + TOPK x (2I + I + H) bf16
    pair rows + the H bf16 sum (`moe_source.MoeBlockSource`)."""
    if not moe_layers:
        return 0
    hidden = _cfg(model, "hidden_size", default=0) or 0
    inter = _cfg(model, "moe_intermediate_size", "intermediate_size", default=0) or 0
    topk = _cfg(model, "num_experts_per_tok", default=1) or 1
    return int(moe_layers) * (topk * 8 + topk * (2 * inter + inter + hidden) * 2 + hidden * 2)


def schedule_stream_bound(asz: dict[str, Any], reqs: list[dict[str, Any]], sampling: dict[str, Any] | None = None) -> dict[str, Any]:
    """[R17 fp8, F-dA-15 bounded admission] The bytes the collector binds over the WHOLE RUN under the workload's DECLARED schedule (each
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
    """[R17 fp8, F-dA-15 / R17-7.3] the host peak bound of a `--bounded-staging` Commit, TERM BY TERM, from the declared-schedule stream
    (`schedule_stream_bound`).  Every term is a measured quantity on the B1 1024/128 row (cp-h100, FP8-STAGING.md §3) or a budget the run
    enforces by name:
      raw     = hidden x f_hidden + other x f_other        hidden (the FA3 slab, padded to the step's longest query) sized exact to 1.2 MB on
                                                           B1 -> 1.05; other (per-token widths of every other family) measured x1.59 -> 2.0
      levels  = raw / 4                                    the tree above the leaves (32 B per 128 B leaf... every level, /4 by the layout)
      kept    = raw - hidden                               retain=host with the hidden family excluded: what stays on host (0 for retain=none)
      learn   = learn budget                               the learning steps' transient host windows; the pool keeps their capacity
      replay  = levels + learn + picks x chunk             openings-after-release runs on a FRESH pool that grows like the run's; picks =
                                                           3 x openings + 64 (after-release + both in-run stages' deferred picks), 256 B each
      slack   = engine growth after admission              measured +3,539 MiB (RSS 2,218 -> 5,757 MiB, B1) + pool rounding -> 6 GiB
      pairs   = levels + kept + learn once, for pairs >= 2   a second pair allocates its frozen pool while torch's pinned-host cache still
                                                           holds the first pair's frozen + replay blocks: B1 PAIRS=3 bounded-x (in-process
                                                           per-step RSS, try 4 pairs3) 16.7 -> 25.6 (pair 0 + replay) -> 29.6 (pair 1 run)
                                                           -> 30.2 (pair 1 replay) -> 30.2 flat through pair 2: +4.6 GiB = 41 % of one
                                                           frozen pool, ONCE; the term takes the whole pool (bounded, converged after pair 1)
    [R17 fp8 F-fp8-C2PROG / F-fp8-POW2 (FP8-STAGING §6; B8 attempt 1, VmHWM 223,776 vs bound 193,827)] two more named terms, and the
    bound is the MAX over the process's PHASES (each a sum of the terms live in it), not one sum:
      c2_programs   the Programs of record loaded for the C2 sampled replay (`RM._programs_for`: 8 Programs / 7.85 M instances = 70,300 MiB
                    of Python objects at B8; ~6.6 GB at B1) -- an EMPIRICAL FORECAST the caller derives from the instance files (json
                    bytes x 15.0, the B8 calibration 4,924 MB -> 70,300 MiB), live from the load until commit_delta frees them after the
                    replay + linkage records (`c2_freed_before_replay`; attempt 1 never did -> they stacked under the replay pool)
      pool_rounding torch's CachingHostAllocator rounds every pinned block up to the next power of two (RSS +21,152 MiB outside the pool
                    at B8): (levels + kept + learn) x 1.0 worst case when the pool is torch-pinned; 0 when the pool registers exact blocks
                    (`pool_exact`, `PinnedPool.mode == "exact-registered-shared"`, RSS == bytes_total)
      fork_children [F-fp8-FORKPIN, B8 attempt 2] the C2 sampled replay forks its evaluator children (mp Pool) while the Programs and the
                    frozen pool are live: their PRIVATE pages (copy-on-write of the Python objects they touch) are charged to the CGROUP,
                    not to the parent's VmHWM -- an EMPIRICAL term the caller passes (attempt 1: cgroup +7,000 MiB over 33 children on
                    shared pool blocks).  The pool blocks themselves are shared mappings and are NOT copied at fork (a private
                    anonymous pinned block IS copied eagerly by Linux >= 5.9: attempt 2, 59,350 MiB per child, cgroup at the cap,
                    20 OOM-killed children); this term is the cgroup scope of the c2 phase
      phases        run = engine + levels + kept + learn + pool_rounding; c2 = run + c2_programs + fork_children; replay = run + replay (+ c2_programs when
                    not freed first); bound = max(phases) + pairs + slack.  `pairs` is 0 for the exact pool (the frozen pool is unregistered
                    and freed when the openings-after-release view dies; the replay pool becomes the next pair's and grows to the run's size)
                    and levels + kept + learn (once) for the torch pool; with `replay_cache` pairs >= 2 reuse pair 0's C2 record and the
                    c2 phase happens once (its term stays in the max either way).
    [R17 fp8 F-fp8-C2RESID / F-fp8-POOLCARRY (B8 attempt 3, VmHWM 150,935 vs bound 156,654: under, but the peak was pair 2's replay phase
    151,097, not the predicted c2 phase -- the replay phase was predicted 118,303 and observed 130,895 (pair 0) / 151,097 (pair 2))] two more
    EMPIRICAL terms the caller passes, both measured by the run's footprint (`rss_bytes` - `pinned_pool_bytes` per phase):
      c2_residual   what the freed Programs leave resident: 7.85 M dead small objects die into pymalloc arenas / glibc's heap, which return
                    only part (B8: gc + malloc_trim gave 29,578 MiB back of 71,445 loaded; RSS - pool after the free 31,460 MiB above the run's
                    2,732) -- live from the free to the end of the PROCESS: in the replay phase and in every later pair's phases
      pool_carry    the previous pair's pool pages still resident in the next pair: B8 pair 1's run RSS - pool = 54,239 = 2,732 (pair 0's) +
                    31,460 (c2_residual) + 20,047 -- MEASURED; the mechanism is a HYPOTHESIS (blocks of the previous pair's pool whose munmap
                    release() deferred while a tensor view was alive -- `deferred_unmaps`), to be tested by the record; once, for pairs >= 2
                    (pair 2's run RSS - pool = 51,329: no further growth)
      phases        + later_pair_replay (pairs >= 2) = run + replay + c2_residual + pool_carry; replay (pair 0) = run + replay + c2_residual (if
                    the Programs were freed first).  B8 attempt 3 re-scored: replay 118,303 + 31,460 = 149,763 (observed 130,895 pair 0: the
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
    # [R17-29 planner, Daniel M-0309] the DOMINANT CONTRIBUTOR, so a refusal names a term and a location, never merely "predicted OOM":
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
            # [F-fp8-C2PROG / F-fp8-POW2] the two terms and the phase the bound is the max of; every number says what kind it is
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


def step_bound_bytes(model: Any, modules: list[tuple[str, Any]], envelope: dict[str, Any], *, hidden_m1: bool, attn_backend: str = "fa2", moe_layers: int = 0) -> dict[str, Any]:
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
    # [R17 stoch F-dA-15] a module SHARED by every layer (Llama's one `rotary_emb`, registered under layer 0) is acquired once per
    # INVOCATION -- `layers` times per step -- not once per module: count it per layer, else the bound is `(layers - 1) x width` per token
    # short (76,800 B/token on Llama-3.2-1B = the 1-3 % by which the R16 records' prefill steps exceeded this bound).
    n_rope = sum(1 for _n, _m in modules if "RotaryEmbedding" in _mro_names(_m))
    rope_mult = max(1, -(-layers // n_rope)) if n_rope else 1
    for name, m in modules:
        names = _mro_names(m)
        try:
            p = next(m.parameters())
            isz = p.element_size()
        except StopIteration:
            isz = 2
        w = 0
        if "LogitsProcessor" in names:
            per_seq += vocab * 4
        elif "Sampler" in names:
            per_seq += 8
        elif "VocabParallelEmbedding" in names:
            w = int(getattr(m, "embedding_dim", hidden) or hidden) * isz + 4          # + the input ids (ACQUIRE_INPUT_CLASSES)
        elif names & {"LinearBase", "ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear"}:
            out = getattr(m, "output_size_per_partition", None) or getattr(m, "output_size", None)
            if not out and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                out = int(m.weight.shape[0])
            w = int(out or hidden) * isz
        elif names & {"RMSNorm", "GemmaRMSNorm", "LayerNorm"}:
            hs = int(m.weight.numel()) if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor) else hidden
            w = 2 * hs * isz                                                            # fused add+norm returns (out, residual)
        elif names & {"SiluAndMul", "GeluAndMul", "GeluAndMulSparse", "MulAndSilu"}:
            w = inter * 2
        elif "RotaryEmbedding" in names:
            w = (heads + kv_heads) * head_dim * 2 * rope_mult                           # returns (q, k); once per layer when shared
        elif "Attention" in names:
            w = heads * head_dim * 2
        else:
            w = hidden * 2
        per_token += w
        per_step += GPU_CHUNK_BYTES
        cls = next((c for c in ("LogitsProcessor", "Sampler", "VocabParallelEmbedding", "RMSNorm", "RotaryEmbedding", "Attention", "SiluAndMul",
                                "LinearBase") if c in names), "other")
        widths[cls] = widths.get(cls, 0) + w
    moe_pt = moe_bytes_per_token(model, moe_layers)
    if moe_pt:
        per_token += moe_pt
        per_step += moe_layers * 6 * GPU_CHUNK_BYTES
        widths["MoeBlock"] = moe_pt
    B = max(1, int(envelope.get("batch") or 1))
    max_prompt = max(1, int(envelope.get("max_prompt") or 1))
    max_new = max(1, int(envelope.get("max_new") or 1))
    max_model_len = int(envelope.get("max_model_len") or (max_prompt + max_new))
    mnbt = envelope.get("max_num_batched_tokens")
    T_prefill = min(int(mnbt) if mnbt else B * max_prompt, B * max_prompt)

    # R13 fa2: the M1 stream is arithmetic in the kernel tile (kBlockN, head_dim) -- 128/64 for hd 64, 64/hd otherwise (num_splits==1 kernel).
    # R13 hopper: under FA3 (attn_backend "fa3") the slab is (sequence, KV head) and the rows are PackGQA-packed (M = seqlen_q * g); the tile
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
    # (B-1) one-token prompts -> B x heads x max_prompt rows.  (R13 fa2 FA2-02: the previous "whole prompts" packing (T_prefill // max_prompt
    # sequences) under-bounded a B=8 Phi-3 256-token prefill by 2x -> "arena block too small for this step" mid-run; the bound is the stream
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


_MEMTRACE = bool(os.environ.get("VERITY_MEMTRACE"))


def memtrace(where: str, step: int | None = None) -> None:
    """P0 diagnostics (env VERITY_MEMTRACE=1): one line with the CUDA caching-allocator state at a lifetime boundary."""
    if not _MEMTRACE:
        return
    import torch

    a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    print(f"[memtrace] {time.strftime('%H:%M:%S')} step {step if step is not None else '-':>4} {where:<28} alloc {a / 2**30:6.2f} reserved {r / 2**30:6.2f} "
          f"device-used {(total - free) / 2**30:6.2f} GiB", flush=True)


class NativeHostCommitter(NativeRangeOpenings, hiding.LeafSchemeRule):
    name = "native_host"

    def __init__(self, engine: Any, *, program_digest: bytes | None = None, run_id: str = "run", chunk: int = DEFAULT_CHUNK,
                 hash_threads: int = 2, retain_values: bool = True, acquire_classes: set[str] | None = None, staging: str = "arena",
                 arena_bytes: int = 512 << 20, batch_copy: bool = False, fast_copy: bool = False, regen_row4: bool = False, gpu_tree: bool = False,
                 geo_digest: bytes | None = None, leaf_scheme: str = scheme.NAME, populate_threads: int | None = None) -> None:
        """`engine=None` builds the OFFLINE core (hashing, roots, openings, verification; no hooks, no CUDA) for the security tests.  `leaf_scheme`: vllm-v1 position leaves (default) or hm96-sha256/v1 hiding leaves over them (`hiding.LeafSchemeRule`).  `populate_threads`: the pinned pool's (`PinnedPool`)."""
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
        self.x03_expected = 0
        self.x03_committed = 0
        # [X-03 F-r19-int-22] the Program's per-layer attention Call statics (fa2_plane_classes.AttnLayerDecl, set by the driver from the
        # Build record): with them `identity_check` DERIVES each layer's hidden-plane class per step; without them the R13 uniformity rule runs
        self.attention_layer_decls: list | None = None
        self.plane_policy = None                          # fa2_plane_classes.LayoutPolicy override (default: read from the hidden source)
        self.pool = PinnedPool(populate_threads=populate_threads)
        self._handles: list = []
        self._orig: dict[str, Any] = {}
        self._hooks = hooks.Hooks()
        self._cur: _StepJob | None = None
        self._step_index = 0
        self._q: queue.Queue = queue.Queue()
        self._done: dict[int, StepCommitment] = {}
        self._levels: dict[int, list[list[bytes]]] = {}
        self._layouts: dict[int, list[TensorMeta]] = {}
        self._streams: dict[int, memoryview] = {}
        self._padding: dict[int, Any] = {}          # [B9.1] padding steps (verity_vllm.commit.padding_steps.PaddingStep), by engine step
        self.padding_stats: dict[str, Any] | None = None   # [B9.1 / A14.3 (g)] set by finalize() when padding was applied: bytes/leaves/wall
        self.padding_hook: Any = None               # [B9.1] callable(committer) -> PaddingSteps, run inside finalize() once every step is hashed
        self.sources: list[Any] = []
        self._worker: threading.Thread | None = None
        # optional CPU pinning experiment (env VERITY_PIN=1): serving thread -> one core, worker + hash threads -> other cores
        self.pin = os.environ.get("VERITY_PIN") == "1" and hasattr(os, "sched_setaffinity")
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
        # GPU-RESIDENT hashing (CMT-2's chunk tree `hidden_gpu_tree.cu`): leaves = SHA-256(16-word chunk header || 256 B chunk) computed on the GPU
        # over the packed step stream IN THE ARENA, tree levels on the GPU, only the 32 B merkle root comes back; no D2H of the stream, no hashlib
        # threads.  Openings are served lazily: chunk bytes D2H from the retained arena block + sibling path from the GPU levels (counted under
        # opening_generation).  Leaf rule differs from pos_leaf (256 B chunks, CMT-2 header) -> the step root binds the GPU merkle root the same way.
        self.gpu_tree = bool(gpu_tree)
        self._gpu_ext = None
        self._HG = None
        self._gpu_levels: dict[int, list] = {}
        self._gpu_blocks: dict[int, tuple] = {}
        self._gpu_t: dict[str, float] = {"leaf_ms": 0.0, "tree_ms": 0.0}
        if gpu_tree:
            self.name = f"{self.name}_gputree"
            self.chunk = GPU_CHUNK_BYTES
        self._init_leaf_scheme(leaf_scheme, gpu_tree or type(self)._commit_step is not NativeHostCommitter._commit_step)
        # R13 P0: where absorbed raw bytes live (see RETAIN_MODES); `retain_values=False` (offline tests) maps to "none"
        # P0: host retention goes to pinned pool memory unless VERITY_RETAIN_PINNED=0 (pageable D2H is ~1 GB/s and synchronous)
        self.retain_pinned = os.environ.get("VERITY_RETAIN_PINNED", "1").strip() not in ("0", "false", "no")
        self.retain = os.environ.get("VERITY_RETAIN", "device" if retain_values else "none").strip().lower()
        if self.retain not in RETAIN_MODES:
            raise ValueError(f"VERITY_RETAIN={self.retain!r}: expected one of {RETAIN_MODES}")
        self.retain_windows: set[tuple[int, int]] = set()   # retain=sample (windowed only): (step, window k) pairs whose raw bytes are kept
        if self.retain != "device":
            self.name = f"{self.name}+retain_{self.retain}"
        self.footprint = _Footprint(os.environ["VERITY_FOOTPRINT"]) if os.environ.get("VERITY_FOOTPRINT") else None
        self.released_bytes = 0                     # device bytes freed by release_staging() (reported by commit_delta)
        self.inline_d2h = os.environ.get("VERITY_INLINE_D2H") == "1"
        if self.inline_d2h and self._rt is not None:
            self.name = f"{self.name}_inl"
        self.regen_row4 = bool(regen_row4)
        if regen_row4:
            self.name = f"{self.name}+row4regen"
        # X-02: `arena_bytes=None` = size the block from the workload envelope in `size_arena()` (commit_delta calls it at arm time);
        # an explicit value is honoured but checked against the same bound (refusal BEFORE the run, with needed vs available bytes)
        self.arena_explicit = arena_bytes is not None
        self.arena_sizing: dict[str, Any] | None = None
        self.arena = _DeviceArena(int(arena_bytes or _ARENA_MIN)) if (engine is not None and staging == "arena") else None
        # [R19 int-20, F-r19-int-20] MANIFEST-DRIVEN input acquisition: module name -> bytes per token of its acquired input.  The required
        # manifest's promoted members (an interior Program row whose output another module reads) are observed at their first consumer's
        # input as `<consumer>/in` -- the same mechanism as the embedding's ids, no longer restricted to a module class -- and the binding map
        # binds that tensor to the producer's identity (binding.promoted_input_acquisition).  Set by commit_delta before install / size_arena.
        self.acquire_inputs: dict[str, int] = {}
        # [v2 acquire, Phase 4a] the consumer-input pre-hook set comes from the ACQUISITION PLAN (verity_vllm.acquire.install
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
        """X-02: size the device staging block from the workload envelope (see `step_bound_bytes`).  Auto mode replaces the default
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
        rec = step_bound_bytes(self.model, self._modules, envelope, hidden_m1=bool(hidden_m1), attn_backend=backend,
                               moe_layers=(len(moe_src.layers) if moe_src is not None else 0))
        # [R14 TP-03] sources that bind extra tensors per step (TP partial source) widen the bound: per token / per sequence / chunk padding
        extra_pt = extra_ps = extra_n = 0
        for s in getattr(self, "sources", []):
            f = getattr(s, "extra_step_bytes", None)
            if f is None:
                continue
            e = f(envelope, self.model)
            extra_pt += int(e.get("per_token_bytes", 0)); extra_ps += int(e.get("per_seq_bytes", 0)); extra_n += int(e.get("tensors_per_step", 0))
        if self.acquire_inputs:
            # [R19 int-20] the manifest-driven `<consumer>/in` acquisitions: one per-token tensor per consumer per step (its width from the
            # promoted member's registry signature), minus the embedding-class inputs the module table already counts
            for n, bpt in sorted(self.acquire_inputs.items()):
                m = dict(self._modules).get(n)
                if m is not None and (_mro_names(m) & ACQUIRE_INPUT_CLASSES):
                    continue
                extra_pt += int(bpt); extra_n += 1
            rec["promoted_inputs"] = {"consumers": sorted(self.acquire_inputs), "per_token_bytes": sum(int(v) for v in self.acquire_inputs.values())}
        if extra_pt or extra_ps or extra_n:
            B = max(1, int(envelope.get("batch") or 1))
            add_prefill = int(rec["prefill_tokens"]) * extra_pt + B * extra_ps + extra_n * GPU_CHUNK_BYTES
            add_decode = B * extra_pt + B * extra_ps + extra_n * GPU_CHUNK_BYTES
            rec["extra_sources"] = {"per_token_bytes": extra_pt, "per_seq_bytes": extra_ps, "tensors_per_step": extra_n, "prefill_add": add_prefill, "decode_add": add_decode}
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
            # P0 (R13) windowed staging: no per-step blocks at all -- the ring (K x W, allocated by enable_windows) IS the device staging;
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
        """X-02: the block size THIS step needs (called at step begin with the scheduler's token/request counts): a prefill-shaped step
        (more tokens than requests) gets the prefill-sized block, a pure decode step the decode-sized one."""
        rec = self.arena_sizing
        if not rec:
            return 0
        return int(rec["prefill_block_bytes"] if ntok > max(1, nreq) else rec["decode_block_bytes"])

    # ------------------------------------------------------------------ offline core (tests; no engine, no CUDA)

    def commit_offline(self, steps: list[list[tuple]], *, padding: list | None = None) -> RunCommitment:
        """Commit the given per-step tensors (name, raw bytes[, dtype, shape]; default uint8 of the byte length) synchronously — the exact
        hashing / layout / root code the online path runs on the worker thread, minus hooks and D2H -- for the security tests and CPU cross-checks.
        `padding`: [B9.1] PaddingSteps -- or a callable(committer) -> PaddingSteps run after the steps are committed -- registered
        before `finalize()` (see `add_padding_steps`)."""
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
        if padding:                                   # a callable sees the committed steps (leaf counts of a mixed step) like the pod does
            self.add_padding_steps(padding(self) if callable(padding) else padding)
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
        ACQUIRE_INPUT_CLASSES) or [R19 int-20] because the required manifest names it as a promoted member's consumer (`acquire_inputs`).
        [v2 acquire] under a plan (`plan_inputs` set) the plan's consumer-input sites decide, the class table is not consulted."""
        if self.plan_inputs is not None:
            return name in self.plan_inputs
        return bool(_mro_names(m) & ACQUIRE_INPUT_CLASSES) or name in self.acquire_inputs

    def _input_only_modules(self) -> list[tuple[str, Any]]:
        """[R19 int-20] the manifest's consumer modules the committer did NOT select (no acquire class): their input is acquired, their
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
        """Registered input: every parameter + buffer hashed once (chunked pos_leaf leaves, per-tensor subtree, fold of subtree roots).
        Timed and reported apart from steady state (BRIEF §3 denominator rule)."""
        t0 = time.perf_counter()
        # [R16 sys SYS-05] host (default, the path below) | device | device-checked: the same leaves (pos_leaf over self.chunk-byte pieces of
        # the tensor's bytes) and the same per-tensor fold on the GPU (verity_vllm.commit.committer.leafhash / native_leafhash.cu); the root
        # construction below is untouched.  device-checked computes both and refuses a mismatch (validation runs record the equality).
        mode = os.environ.get("VERITY_WEIGHTS_HASH", "host")
        if mode not in ("host", "device", "device-checked"):
            raise ValueError(f"VERITY_WEIGHTS_HASH must be host | device | device-checked, got {mode!r}")
        LH = None
        if mode != "host":
            from verity_vllm.commit.committer import leafhash as LH  # noqa: N812
            LH.ext()                                          # JIT once; a build failure here fails the registration by name, not silently host
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
            if mode == "host" or not flat.is_cuda:
                host = flat.cpu()                    # blocking D2H, once
                mv = memoryview(host.numpy()).cast("B")
                leaves = self._hash_leaves(mv)
                r = scheme.fold(leaves)
            else:
                r = LH.tensor_root_device(flat, self.chunk)
                if mode == "device-checked":
                    rh = LH.tensor_root_host(flat, self.chunk)
                    if rh != r:
                        raise RuntimeError(f"weights registration: device root != host root for {name} ({nb} B, chunk {self.chunk}): "
                                           f"{r.hex()[:16]} vs {rh.hex()[:16]} -- refusing to commit (SYS-05 device-checked)")
                    checked += 1
            roots.append(r)
            n_bytes += nb
            names.append([name, _dtype_name(p), list(p.shape), nb])
        root = scheme.weights_root(self._geo_digest, scheme.names_digest(names), roots)
        # the tree under the root, kept off the record: a deferred replay opens the weight fields it consumes against it (replay bundle)
        self.weights_tree = {"names": names, "tensor_roots": [r.hex() for r in roots], "root": root.hex(), "chunk": self.chunk,
                             "geo_digest": self._geo_digest.hex()}
        self.weights = {"root": root.hex(), "tensors": len(roots), "bytes": n_bytes, "wall_s": time.perf_counter() - t0,
                        "label": "measured; once per process, excluded from steady state", "hash": mode, "chunk": self.chunk,
                        **({"device_checked_tensors": checked} if mode == "device-checked" else {})}
        return self.weights

    def attach_source(self, src: Any) -> None:
        """An extra acquisition source (e.g. the FA2 M1 hidden stream): install/uninstall with the committer, contributes to the same step
        stream and run root, and to the coverage declaration."""
        self.sources.append(src)
        self.name = f"{self.name}+{getattr(src, 'short_name', 'hidden_m1')}"

    def flush_after_plane(self, what: str) -> bool:  # noqa: ARG002
        """[R16 moe FLUSH-03] hook for acquisition sources whose tensor is overwritten before the next module hook: the arena committer
        snapshots at `_acquire` (D2D on the compute stream), so nothing is staged and nothing needs to leave early -> False.  The gather
        collector (`native_collect.NativeCollectCommitter`) overrides this with the flush."""
        return False

    def install(self) -> None:
        if self.installed:
            return
        self._check_skip()
        self._start_worker()
        for src in self.sources:
            src.install()
        for name, m in self._modules:
            self._handles.append(m.register_forward_hook(self._make_hook(name)))
            if self._acquires_input(name, m):
                self._handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
        for name, m in self._input_only_modules():                                # [R19 int-20] a promoted member's consumer of no acquire class
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
            me._step_min_bytes = 0 if me.arena_explicit else me.step_min_bytes(   # X-02 as native_collect; an explicit ring is one size
                int(scheduler_output.total_num_scheduled_tokens), len(getattr(scheduler_output, "num_scheduled_tokens", ()) or ()))
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
        for src in self.sources:
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
        self._padding = {}
        self.padding_stats = None
        self._step_index = 0
        self.spans = Spans(scheme.NAME)
        self.pool.reset()
        self.errors = []

    def close(self) -> None:
        self.uninstall()
        self._stop_worker()
        self._pool_exec.shutdown(wait=True)

    # ------------------------------------------------------------------ acquisition (serving thread)

    def _begin_step(self) -> None:
        self._cur = _StepJob(self._step_index, [], None)
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
        self.spans.add("acquisition", time.perf_counter() - t0, step=job.step)
        if self.footprint is not None:
            self.footprint.event("end_step", step=job.step, used=int(job.cursor), backlog=job.backlog)
        self._q.put(job)

    ALIGN = 4
    GPU_TREE_SRC = 0                       # CMT-2 header field `src_mask` (0: not an FA2 tap stream); the tag field carries the step index
    layout = "chunk-leaf-v1"               # the GPU tree's leaf rule, which padding leaves follow; `native_collect` also implements chunk-leaf-v2

    def _gpu_ext_load(self):
        if self._gpu_ext is None:
            import sys as _sys

            _tree = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            src_dir = os.path.join(_tree, "out", "gen", "r9", "cmt-hidden", "src")
            if not os.path.isfile(os.path.join(src_dir, "hidden_gpu.py")):
                # [R18 infra] a tree whose r9 copy was never materialised (pod_hidden_gpu.sh not run) imports the SHIPPED copy of the same
                # tree -- never another tree's, never a default path
                src_dir = os.path.join(_tree, "verity_vllm", "commit", "committer", "hidden_gpu_src")
            if src_dir not in _sys.path:
                _sys.path.insert(0, src_dir)
            import hidden_gpu as HG  # CMT-2 driver: JIT-builds hidden_gpu_tree.cu on first use

            # [R18 infra, F-r17-tp-06] the module that LOADED must be this tree's, and so must verity.ml.circuits / verity_vllm / vllm-poc as
            # this process resolves them: PRECHECK_FAIL_SOURCE_IDENTITY by name at the import site, before the extension is built or used
            from verity_vllm.pipeline import source_identity as _sid

            _sid.assert_loaded_module_is_ours(HG, "hidden_gpu", _tree)
            # [R18 infra, r18-int-2; rev S-r18-1 n1] the document of THIS check (hidden_gpu loaded:true from its resolved path, plus
            # verity.ml.circuits / verity_vllm / vllm-poc as loaded in this process) is written beside the attempt when the driver names the
            # dir ($VERITY_SOURCE_IDENTITY_DIR, set by commit_delta from --out): <dir>/source_identity.gpu_ext_load.json.  The start-of-run
            # record has hidden_gpu loaded:false; this one is the evidence that what LOADED at the import site was this tree's.
            _sid_dir = os.environ.get("VERITY_SOURCE_IDENTITY_DIR")
            _sid_out = os.path.join(_sid_dir, "source_identity.gpu_ext_load.json") if _sid_dir else None
            _sid_doc = _sid.precheck("commit.gpu_ext_load", _tree, out=_sid_out)
            _sid_doc.setdefault("loaded_modules", {})["hidden_gpu"] = {"file": getattr(HG, "__file__", None), "repo": getattr(HG, "REPO", None)}
            if _sid_out:
                with open(_sid_out, "w") as _f:
                    json.dump(_sid_doc, _f, indent=1, default=str)

            # a killed build's torch `lock` (every later worker once hung in file_baton.wait on one) is cleared inside HG.ext(), under
            # the build dir's lock (native_jit.jit_load)
            self._gpu_ext = HG.ext()
            self._HG = HG
        return self._gpu_ext

    def _commit_step_gpu(self, job: _StepJob) -> None:
        """Worker: GPU-resident leaves + tree over the packed step block on the side stream; keeps the block for lazy openings."""
        from verity_vllm.commit import hidden_stream as hs

        e = self._gpu_ext_load()
        side = self._side_stream()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(side):
            side.wait_event(job.event)
            blocks = job.overflow + [(job.arena, job.cursor)]
            if len(blocks) > 1:                                   # overflow: concatenate the used parts into one block (rare; counted)
                total = sum(u for _b, u in blocks)
                cat = self.arena.take(total, wait=False)          # the block that fits may be one this step holds
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
            padded = -(-used // GPU_CHUNK_BYTES) * GPU_CHUNK_BYTES
            if padded > job.arena.numel():
                big = self.arena.take(padded, wait=False)
                big[:used].copy_(job.arena[:used])
                self.arena.release(job.arena)
                job.arena = big
            if padded > used:
                job.arena[used:padded].zero_()
            for m in () if hasattr(job.metas, "scan") else job.metas:     # a cached plan's record (plan_cache) is placed already
                m.stream_off, m.dev_block = m.dev_off, None
            stream_i32 = job.arena[:padded].view(torch.int32)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            ev[0].record(side)
            memtrace("worker: before leaves", job.step)
            leaves = e.chunk_leaves(stream_i32, GPU_CHUNK_BYTES // 4, job.step, self.GPU_TREE_SRC, job.step, padded // 4, 0)
            ev[1].record(side)
            levels = e.tree_levels(leaves)
            ev[2].record(side)
            memtrace("worker: before keep-copy", job.step)
            keep = self._retain_copy(job.arena, padded, side)     # R13 P0: compact retained copy on device | host | none (lazy openings / replay)
            mr_t = levels[-1].to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
            done.synchronize()                                    # keep-copy + root D2H complete before the arena block can be reused
            if self.retain != "device":
                levels = self._levels_to_host(levels)             # authentication state to host; the device keeps only arena blocks
                torch.cuda.current_stream().synchronize()
            self._footprint_absorbed(job, padded, levels, keep)
            self.arena.release(job.arena)
            memtrace("worker: block released", job.step)
        mr = mr_t.numpy().astype("uint32").tobytes() if hasattr(mr_t, "numpy") else bytes(mr_t)
        self._gpu_t["leaf_ms"] += ev[0].elapsed_time(ev[1])
        self._gpu_t["tree_ms"] += ev[1].elapsed_time(ev[2])
        t1 = time.perf_counter()
        n = int(leaves.shape[0])
        nbytes = job.metas.nbytes if hasattr(job.metas, "scan") else sum(m.nbytes for m in job.metas)
        layout_dig = self.layout_digest(job.metas)
        root = scheme.step_root(self.program_digest, self._ctx_digest(job.step), self._geo_digest, layout_dig, n, mr)
        t2 = time.perf_counter()
        self.spans.add("hashing", t1 - t0, step=job.step, nbytes=nbytes, leaves=n)
        self.spans.add("root_finalisation", t2 - t1, step=job.step)
        self._gpu_levels[job.step] = levels
        self._gpu_blocks[job.step] = (keep, used)
        self._layouts[job.step] = job.metas
        job.arena = None
        self._done[job.step] = StepCommitment(job.step, root, mr, n, nbytes, len(job.metas), layout_dig, job.t_step_end, t2, job.backlog,
                                              hashing_s=t1 - t0, root_finalisation_s=t2 - t1)

    def commit_block_offline(self, step: int, stream: bytes | bytearray, metas: list[TensorMeta]) -> StepCommitment:
        """CPU reference of the GPU chunk tree for ONE packed step block (tests; no CUDA): the same 256 B chunk leaves (CMT-2 header), the
        same production fold, the same retention policy (`retain`), the same per-step state the worker leaves behind, so `open()`,
        `release_staging()`, `verify()` and `finalize()` run unchanged on the result."""
        chunk = self.chunk
        used = len(stream)
        padded = -(-used // chunk) * chunk
        buf = bytes(stream) + b"\0" * (padded - used)
        layout = self._gpu_layout(step, padded // 4)
        leaves = [scheme.chunk_leaf(scheme.chunk_header(layout, step, i, chunk // 4, self.GPU_TREE_SRC), buf[i * chunk:(i + 1) * chunk]) for i in range(padded // chunk)]
        lv_bytes = scheme.fold_levels(leaves)
        levels = [torch.frombuffer(bytearray(b"".join(lv)), dtype=torch.int32).reshape(len(lv), 8).clone() for lv in lv_bytes]
        keep = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone() if self.retain != "none" else None
        if keep is not None and self.retain == "device":
            keep = keep.cuda()
            levels = [lv.cuda() for lv in levels]
        for m in metas:
            m.stream_off = m.dev_off
        mr = lv_bytes[-1][0]
        n = len(leaves)
        nbytes = sum(m.nbytes for m in metas)
        layout_dig = self.layout_digest(metas)
        root = scheme.step_root(self.program_digest, self._ctx_digest(step), self._geo_digest, layout_dig, n, mr)
        self._gpu_levels[step] = levels
        self._gpu_blocks[step] = (keep, used)
        self._layouts[step] = metas
        t = time.perf_counter()
        self._done[step] = StepCommitment(step, root, mr, n, nbytes, len(metas), layout_dig, t, t, 0)
        self._step_index = max(self._step_index, step + 1)
        return self._done[step]

    def _retain_copy(self, arena: Any, padded: int, side: Any) -> Any:
        """The raw bytes kept for openings after the block is absorbed: device copy (R9 behaviour), host copy (D2H on the side stream; the
        arena block is recycled once the caller's done-event completes), or nothing (openings by replay).  Never touches the leaves."""
        if self.retain == "device":
            keep = torch.empty(padded, dtype=torch.uint8, device=arena.device)
            keep.copy_(arena[:padded])
            return keep
        if self.retain == "host":
            if self.retain_pinned:
                keep = self.pool.take(padded)[:padded]           # P0: pinned pool block -> async D2H at PCIe rate (the caller syncs its done-event)
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
                             levels_bytes=sum(int(lv.numel()) * 4 for lv in levels), levels_device=bool(levels and levels[0].is_cuda),
                             keep_bytes=int(keep.numel()) if keep is not None else 0, keep_device=(keep.is_cuda if keep is not None else None),
                             retain=self.retain, arena_block_bytes=int(job.arena.numel()) if job.arena is not None else None,
                             retained_device_total=self._retained_device_bytes(), families=_Footprint.families(job.metas),
                             t_step_end=round(job.t_step_end - self.footprint.t0, 6) if job.t_step_end else None, backlog=job.backlog)

    def _retained_device_bytes(self) -> int:
        n = sum(int(lv.numel()) * 4 for lvls in self._gpu_levels.values() for lv in lvls if lv.is_cuda)
        n += sum(int(b.numel()) for b, _u in self._gpu_blocks.values() if b is not None and b.is_cuda)
        return n

    def release_staging(self, *, levels_to_host: bool = True) -> dict[str, int]:
        """R13 P0: drop every RAW retained copy that still lives on the device (mode `device`) and, optionally, move the tree levels to host.
        After this the device holds only the arena blocks; openings need `open(..., values_from=)` (replay) or a host copy.  Returns bytes."""
        freed = {"raw_device_bytes": 0, "levels_device_bytes": 0}
        for step, (blk, used) in list(self._gpu_blocks.items()):
            if blk is not None and blk.is_cuda:
                freed["raw_device_bytes"] += int(blk.numel())
                self._gpu_blocks[step] = (None, used)
        if levels_to_host:
            for step, lvls in list(self._gpu_levels.items()):
                if lvls and lvls[0].is_cuda:
                    freed["levels_device_bytes"] += sum(int(lv.numel()) * 4 for lv in lvls)
                    self._gpu_levels[step] = [lv.cpu() for lv in lvls]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.released_bytes += freed["raw_device_bytes"] + freed["levels_device_bytes"]
        if self.footprint is not None:
            self.footprint.event("release_staging", **freed, retained_device_total=self._retained_device_bytes())
        return freed

    def _gpu_layout(self, step: int, words: int):
        from verity_vllm.commit import hidden_stream as hs

        v2 = bool(getattr(self, "layout_v2", False))               # chunk-leaf-v2 (R13 P0, behind --layout): M := 0, the length is bound by n_leaves

        class _L:                                                 # duck-typed StreamLayout for the CMT-2 chunk header: (HB, M, NB) = (step, words, 0)
            HB, M, NB = step, (0 if v2 else words), 0

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
        self.x03_expected += 1                       # X-03: attempted acquisition; committed counted only after the copy was issued
        self._acquire_impl(name, t, job)
        self.x03_committed += 1

    def _acquire_impl(self, name: str, t: Any, job: _StepJob) -> None:
        t0 = time.perf_counter()
        if self._rt is not None and t.is_contiguous():
            return self._acquire_fast(name, t, job, t0)
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

    # ---- fast acquisition: no torch ops on the serving thread at all -- the snapshot is one cudaMemcpyAsync (D2D, stream-ordered on the
    # compute stream) issued through libcudart via ctypes on raw data pointers (~2 us CPU instead of ~15-20 us for slice+view+copy_).  Bit
    # semantics identical to the arena copy (the value check compares the committed bytes with hook-time clones).
    def _acquire_fast(self, name: str, t: Any, job: _StepJob, t0: float) -> None:
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
        # [R13 tp, F-tp-07] the CUDA current device is THREAD-LOCAL (default 0 in a new thread): the worker must run on the serving thread's
        # device, or a rank on cuda:1 (TP) creates its side stream on cuda:0 and every step fails (`cudaMemsetAsync(pad): invalid argument`,
        # 0 leaves).  TP1 (device 0) is unchanged.
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
            try:
                self._commit_step(job)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"worker step {job.step}: {e!r}"[:300])
                self._done[job.step] = StepCommitment(job.step, b"", b"", 0, 0, 0, b"", job.t_step_end, time.perf_counter(), job.backlog)

    # Ablations for the cost decomposition ONLY (env VERITY_ABL, comma list; none of them is a commitment):
    #   nocopy  : hooks + metadata + arena bookkeeping, no D2D snapshot launch      noworker: worker discards the step (no D2H, no hashing)
    #   nohash  : worker does the D2H but skips hashing / roots
    ABL = frozenset(filter(None, os.environ.get("VERITY_ABL", "").split(",")))

    def _commit_step(self, job: _StepJob) -> None:
        if self.gpu_tree and job.arena is not None:
            return self._commit_step_gpu(job)
        if "noworker" in self.ABL:
            job.event.synchronize()
            if job.arena is not None:
                self.arena.release(job.arena)
                for blk, _ in job.overflow:
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

    # ------------------------------------------------------------------ [B9.1] padding steps -- the inactive rows of a padded Commit
    # (LIFTING-SPEC v0.7 B9.1; verity_vllm/commit/padding_steps.py: one rule, this committer and the laptop conversion are its two
    # consumers).  Nothing here observes a value: every padding word is the canonical bot of its lifted width, synthesized by ENABLEMENT
    # from RequiredValues(Lift(P)) outside the request's executed prefix of record.

    def add_padding_steps(self, padding: list) -> None:
        """Register the padding steps (`PaddingStep`s from `padding_steps.padding_step_commit`, built with THIS committer's program digest,
        run id, geo digest and layout) before `finalize()`.  A step nobody executed becomes a step of its own; a MIXED step (index below the
        executed count; B > 1) is refolded from this committer's retained leaf digests + the padding leaves (`refold_step`) -- no active
        byte is re-read.  The padding must be laid out after the executed leaves (`leaf_base == n_leaves` of the step), else refused."""
        self._refuse_padding(padding)
        for ps in padding:
            if ps.step in self._padding:
                raise ValueError(f"padding step {ps.step} registered twice")
            self._padding[int(ps.step)] = ps

    def _leaf_digests_of(self, step: int) -> list[bytes]:
        if step in self._gpu_levels:
            lv0 = self._gpu_levels[step][0]
            return [lv0[i].cpu().numpy().astype("uint32").tobytes() for i in range(int(lv0.shape[0]))]
        return list(self._levels[step][0])

    def _apply_padding(self, steps: list[StepCommitment]) -> list[StepCommitment]:
        from verity_vllm.commit import padding_steps as PSm

        for s in sorted(self._padding):
            ps = self._padding[s]
            # host-only metas: `dev_off -1` -- a padding tensor has no bytes in the step's device block (a GPU-tree mixed step keeps one),
            # so a reader of its bytes (`commit.opened.OpenedReader`: `leaf_span` on `stream_off`) opens the padding leaves, never the executed block at offset 0
            pad_metas = [TensorMeta(m.ordinal, m.name, m.dtype, tuple(m.shape), m.nbytes, m.chunk, m.first_leaf, m.n_leaves,
                                    host=memoryview(ps.stream)[m.stream_off - ps.stream_base:m.stream_off - ps.stream_base + m.nbytes],
                                    stream_off=m.stream_off, dev_off=-1) for m in ps.metas]
            t = time.perf_counter()
            if s < len(steps):                                      # a MIXED step: other requests executed it -> refold, one root
                sc = steps[s]
                if not ps.appended or ps.leaf_base != sc.n_leaves:
                    raise RuntimeError(f"padding of step {s}: laid out after {ps.leaf_base} leaves but the executed step has {sc.n_leaves} "
                                       f"(appended={ps.appended}) -- the padding must be built on this committer's step (B9.1 (3))")
                levels, root = PSm.refold_step(self._leaf_digests_of(s), self._layouts[s], ps, program_digest=self.program_digest,
                                               run_id=self.run_id, geo_digest=self._geo_digest)
                self._layouts[s] = list(self._layouts[s]) + pad_metas
                self._levels[s] = levels                             # host levels of the refolded tree (padding openings read these)
                if s in self._gpu_levels:                            # the executed leaves' openings read the tensor levels: refolded too
                    self._gpu_levels[s] = [torch.frombuffer(bytearray(b"".join(lv)), dtype=torch.int32).reshape(len(lv), 8).clone() for lv in levels]
                if s in self._streams and s not in self._gpu_blocks:  # host packed: the step stream grows by the padding block (chunk-aligned)
                    old = bytes(self._streams[s])
                    self._streams[s] = memoryview(old + b"\0" * (ps.stream_base - len(old)) + ps.stream)
                n = sc.n_leaves + ps.n_leaves
                steps[s] = StepCommitment(s, root, levels[-1][0], n, sc.n_bytes + ps.n_bytes, sc.n_tensors + len(pad_metas),
                                          self.layout_digest(self._layouts[s]), sc.t_step_end, time.perf_counter(), sc.backlog_at_step_end,
                                          hashing_s=sc.hashing_s, root_finalisation_s=sc.root_finalisation_s + (time.perf_counter() - t))
                self._done[s] = steps[s]
            else:                                                   # a step nobody executed: the padding IS the step
                if s != len(steps):
                    raise RuntimeError(f"padding step {s} is not the next step ({len(steps)} committed): the padded Program's steps are contiguous")
                if ps.appended:
                    raise RuntimeError(f"padding step {s} was laid out as appended (leaf_base {ps.leaf_base}) but no executed step {s} exists")
                self._layouts[s] = pad_metas
                self._levels[s] = ps.levels
                self._streams[s] = memoryview(ps.stream)
                sc = StepCommitment(s, ps.root, ps.merkle_root, ps.n_leaves, ps.n_bytes, len(pad_metas), self.layout_digest(pad_metas),
                                    t, time.perf_counter(), 0, root_finalisation_s=time.perf_counter() - t)
                self._done[s] = sc
                steps.append(sc)
                self._step_index = s + 1
            self.spans.add("padding", time.perf_counter() - t, step=s, nbytes=ps.n_bytes, leaves=ps.n_leaves)
        return steps

    def _padding_leaf(self, step: int, index: int):
        """The PaddingStep owning leaf `index` of `step`, or None (an executed leaf)."""
        ps = self._padding.get(step)
        return ps if (ps is not None and index >= ps.leaf_base) else None

    # ------------------------------------------------------------------ finalize / open / verify

    def finalize(self) -> RunCommitment:
        t0 = time.perf_counter()
        self._end_step()
        # wait for the worker to drain the queue (root-ready delay = this wait + the run-root fold)
        with self.spans.span("wait"):
            while len(self._done) < self._step_index:
                time.sleep(0.0005)
        steps = [self._done[s] for s in range(self._step_index)]
        t1 = time.perf_counter()
        ex_steps, ex_bytes, ex_leaves = len(steps), sum(s.n_bytes for s in steps), sum(s.n_leaves for s in steps)
        if self.padding_hook is not None:               # [B9.1] the pod consumer: the hook sees every executed step's leaf count and layout
            hook, self.padding_hook = self.padding_hook, None
            self.add_padding_steps(hook(self) or [])
        t_hook = time.perf_counter()
        if self._padding:
            steps = self._apply_padding(steps)          # [B9.1] the inactive rows, synthesized by enablement, BEFORE the run-root fold
        if self._padding:                               # [B9.1 / A14.3 (g)] the padding's cost beside the executed Commit, measured on THIS run
            pads = list(self._padding.values())
            self.padding_stats = {
                "executed_steps": ex_steps, "executed_bytes": ex_bytes, "executed_leaves": ex_leaves,
                "padding_steps_total": len(pads), "padding_steps_appended": sum(1 for ps in pads if ps.appended),
                "padding_steps_own": sum(1 for ps in pads if not ps.appended), "padding_bytes": sum(ps.n_bytes for ps in pads),
                "padding_leaves": sum(ps.n_leaves for ps in pads), "padding_tensors": sum(len(ps.metas) for ps in pads),
                "steps_after": len(steps), "bytes_after": sum(s.n_bytes for s in steps), "leaves_after": sum(s.n_leaves for s in steps),
                "hook_wall_s": t_hook - t1, "apply_wall_s": time.perf_counter() - t_hook,
                "padding_bytes_over_executed": (sum(ps.n_bytes for ps in pads) / ex_bytes) if ex_bytes else None}
        step_roots = [s.root for s in steps]    # b"": a placeholder step (a learn-only warm-up pass, or a failed step whose error is recorded)
        run_root = scheme.run_root(self.program_digest, self._geo_digest, step_roots) if all(len(r) == 32 for r in step_roots) else b""
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
                                    "errors_n": len(self.errors),
                                    # X-03 reference integrity (Python acquisition path; the native collector overrides these in its finalize)
                                    "expected_tensors": self.x03_expected, "committed_tensors": self.x03_committed,
                                    "identity_check": self.identity_check() if self._layouts else None,
                                    "hash_threads": self.hash_threads, "chunk": self.chunk, "modules_hooked": len(self._modules), **self._leaf_record(steps),
                                    "staging": self.staging, "fast_copy": self.cudart_path, "gpu_tree": self.gpu_tree, "gpu_tree_ms": dict(self._gpu_t),
                                    "gpu_tree_retained_bytes": self._retained_device_bytes(),
                                    "host_retained_bytes": sum(int(lv.numel()) * 4 for lvls in self._gpu_levels.values() for lv in lvls if not lv.is_cuda)
                                    + sum(int(getattr(b, "retained_bytes", None) if getattr(b, "retained_bytes", None) is not None else b.numel())
                                          for b, _u in self._gpu_blocks.values() if b is not None and not b.is_cuda),
                                    "retain": self.retain, "released_device_bytes": self.released_bytes,
                                    # per-step trace (R13 p0b): root prefix, tensors, bytes, identity digest (names) and the first FA2 row
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

    # ------------------------------------------------------------------ X-03 identities (R13): expected manifest vs committed identities

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
        # [R15 compiled S3] under external acquisition (compiled kernel taps) the module forwards are NOT wrapped: the per-step identities
        # are the source's (P_compiled's kernel launches, `c<piece>.<kernel>`), declared by the source from the required manifest
        ext = [s for s in self.sources if hasattr(s, "expected_modules")]
        if ext and getattr(self, "external_acquisition", False):
            modules = sorted({m for s in ext for m in s.expected_modules()})
        if self.runner is not None and hasattr(self.runner, "sample") and "runner.sampler" not in modules:
            modules.append("runner.sampler")        # sampled ids are acquired from the wrapped `runner.sample`, not from a module hook
        attn_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, m in self._modules if "Attention" in _mro_names(m) and ".layers." in n})
        fa2 = any(getattr(s, "short_name", "") in ("hidden_m1", "hidden_m1_fa3") or type(s).__name__.endswith(("HiddenM1Source", "HiddenM1SourceFA3"))
                  for s in self.sources)                                   # [R13 hopper] the FA3 tap binds the same one-row-per-layer identity
        # [R13 MOE-03] the MoE block source binds exactly TENSORS_PER_LAYER tensors per MoE layer per step (router_w, router_ids, up, act, down, sum), + router_softmax per tapped layer
        moe_src, rs_src = (next((s for s in self.sources if getattr(s, "short_name", "") == n), None) for n in ("moe", "router_softmax"))
        moe_layers = sorted(set(moe_src.layers.values())) if moe_src is not None else []
        # [R14 TP-03] the TP partial source binds exactly ONE rank-local collective input per hooked site per step (install-time site set)
        tp_src = next((s for s in self.sources if getattr(s, "short_name", "") == "tp_partial"), None)
        # [R15 tp TP-06, manbind M-0350 C3] the TP source is also the ONE binder of the fused-MoE block's RECEIVED all-reduce output
        # (`model.layers.<L>.mlp.experts/out`; MoERunner is not an ACQUIRE_CLASSES module): those sites are expected modules of every step
        if tp_src is not None and hasattr(tp_src, "received_site_names"):
            modules += [s for s in tp_src.received_site_names() if s not in modules]
        return {"modules": modules, "attention_layers": attn_layers, "fa2_hidden_per_layer": 1 if fa2 else 0, "sampler_ids": 1 if any(n == "runner.sampler" for n in modules) else 0,
                "moe_layers": moe_layers, "moe_tensors_per_layer": (getattr(moe_src, "stats", {}).get("tensors_per_layer", 6) if moe_src is not None else 0),
                "router_softmax_layers": sorted(int(k) for k in rs_src.layers) if rs_src is not None else [], "tp_sites": sorted(tp_src.site_names()) if tp_src is not None else [], "tp_rank": (tp_src.rank if tp_src is not None else None),
                # [R19 moe F-r19-moe-1] partials per site per step: 1, or a mid-module site's call count (OlmoeAttention: 2 gathers); the calls its instance's flags rule out are listed
                "tp_site_counts": (dict(tp_src.site_counts()) if tp_src is not None and hasattr(tp_src, "site_counts") else {}), "tp_sites_ruled_out": list(getattr(tp_src, "mid_sites_ruled_out", None) or [])}

    def identity_check(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """X-03 on IDENTITIES.  For every committed step: (1) every module of the expected manifest produced >= 1 committed tensor and no
        tensor came from a module outside it; (2) the FA2/FA3 hidden source produced, per attention layer, exactly the plane inventory the
        layer's class implies -- DERIVED per layer from the Program's attention Call statics x the step's active sequences when the driver
        handed `attention_layer_decls` (fa2_plane_classes; F-r19-int-22: a windowed layer and a global GQA layer of one step legitimately
        differ), else the R13 rule "every layer the same positive count"; (3) exactly one sampled-ids tensor; (4) the multiset of
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
        fa2_rules_used: Counter = Counter()                          # per step: "per-layer" (declarations + active set) or "uniform" (R13)
        decls = getattr(self, "attention_layer_decls", None) or []   # the Program's per-layer attention Call statics (driver-supplied)
        # [B9.1 / R17-42; pad fix (e), the second #3 pod run on b542ef5fc] a PADDED run's layouts carry the padding tensors (host-only metas,
        # `dev_off -1`, named `<executed name>/<rid>/bot`, fix (c)) beside or instead of the executed ones.  A padding tensor materializes an
        # identity the executed prefix left INACTIVE: it is not a module call, so it is not one of the "tensors a module produced" (1), (4) count;
        # (a) at a MIXED step the executed metas alone are checked as before (the batched launch is still one tensor per identity name) and the
        # padding metas are counted apart; (b) at a PADDING-ONLY step (nobody executed it) the check reads the padding: every identity NAME of
        # step 0 must be materialized as ⊥ (set equality, not multiplicity: each inactive request contributes one), every expected module
        # covered, the sampler-ids identity present -- a step that drops one identity's ⊥ still FAILS by name.  Never a relabel: the
        # padding's words are checked == bot by `padding_steps.padded_population`, its placement by the composition check.
        padded_run = bool(getattr(self, "_padding", None))
        padding_only_steps: list[int] = []
        padding_tensors_n = 0
        # per-step names held to the source's own rule, not rule (4): the call-boundary targets; the capture map's names on the step's routes
        rt_src = next((s for s in self.sources if hasattr(s, "step_modules")), None)
        cb_src = next((s for s in self.sources if hasattr(s, "split_step_names") and s is not rt_src), None)

        def _identity_of_padding(n: str) -> str:                     # "<executed name>/<rid>/bot" -> "<executed name>"
            return n[:-len("/bot")].rsplit("/", 1)[0] if n.endswith("/bot") and "/" in n[:-len("/bot")] else n

        for step in sorted(self._layouts):
            metas_all = self._layouts[step]
            cols = metas_all.names if hasattr(metas_all, "scan") else None   # a cached plan's record (plan_cache): its names, no TensorMeta built
            ps = self._padding.get(step) if padded_run else None
            # a padding meta: the step has a PaddingStep, the meta sits in the padding block of the step stream (`stream_off >= stream_base`;
            # an executed arena meta reads -1, an executed packed meta < stream_base) and carries the padding name (fix (c))
            is_pad = (lambda m: m.name.endswith("/bot") and m.stream_off >= ps.stream_base) if ps is not None else (lambda m: False)
            pad_metas = [m for m in metas_all if is_pad(m)] if cols is None else []
            metas = [m for m in metas_all if not is_pad(m)] if cols is None else metas_all
            padding_only = bool(pad_metas) and not metas
            padding_tensors_n += len(pad_metas)
            if padding_only:
                padding_only_steps.append(int(step))
            names_all = Counter(m.name for m in metas_all) if cols is None else Counter(cols)   # fa2 per-layer uniformity reads every plane of the step (rule (a) planes included)
            names = Counter(m.name for m in metas) if cols is None else names_all
            mod_names = Counter({n: c for n, c in names.items() if not n.startswith(HIDDEN_M1_PREFIXES) and not n.startswith("moe/") and not TP_PARTIAL_RE.search(n)})
            pad_ident = Counter(_identity_of_padding(m.name) for m in pad_metas)
            pad_mod_names = Counter({n: 1 for n in pad_ident if not n.startswith(HIDDEN_M1_PREFIXES) and not n.startswith("moe/") and not TP_PARTIAL_RE.search(n)})
            if padding_only:
                mod_names = pad_mod_names                                # the identities materialized as ⊥, one each (set semantics below)
            # [R14 TP-03] rank-local collective inputs: exactly one `tp/<site>/{part,shard}_rank<r>` per expected site, none from elsewhere
            tp_bad: dict[str, int] = {}
            tp_sites = man.get("tp_sites") or []
            tp_counts = Counter(n.rsplit("/", 1)[0] for n in names.elements() if TP_PARTIAL_RE.search(n))
            if tp_sites:
                tp_want = man.get("tp_site_counts") or {}                     # [R19 moe] a mid-module site binds its static call count per step
                tp_bad = {s: tp_counts.get(s, 0) for s in tp_sites if tp_counts.get(s, 0) != int(tp_want.get(s, 1))}
                tp_bad.update({s: c for s, c in tp_counts.items() if s not in tp_sites})
                rk = man.get("tp_rank")
                if rk is not None and any(not n.endswith(f"_rank{rk}") for n in names if TP_PARTIAL_RE.search(n)):
                    tp_bad["<foreign rank tag>"] = sum(c for n, c in names.items() if TP_PARTIAL_RE.search(n) and not n.endswith(f"_rank{rk}"))
            elif tp_counts:
                tp_bad = {"<undeclared tp source>": sum(tp_counts.values())}
            mod_names, rt_bad = rt_src.split_step_names(mod_names, step, padding_only) if rt_src is not None else (mod_names, {})
            by_module = Counter()
            for n, c in mod_names.items():
                by_module[self._module_of(n)] += c
            step_expected = rt_src.step_modules(expected_modules) if rt_src is not None else expected_modules
            missing = sorted(step_expected - set(by_module))
            unknown = sorted(set(by_module) - expected_modules)
            fa2_layers = FP.counts_by_layer(names_all.elements())
            fa2_bad: dict[int, int] = {}
            fa2_detail: dict[str, Any] = {}
            if man["fa2_hidden_per_layer"]:
                seqs = (FP.StepSequences(()) if padding_only else self._step_sequences(step, metas if cols is None else metas.scan())) if decls else None
                if seqs is not None:
                    # [F-r19-int-22] each layer's class is DERIVED from the Program's attention Call statics x the step's active-sequence set
                    # (fa2_plane_classes.plane_class): a windowed layer's decode launch is un-swapped (6 planes x sequences) while a global GQA
                    # layer's is swapped (one whole row) -- uniformity across layers is a consequence for one-class Programs, not a rule
                    v = FP.check_step(decls, seqs, names_all.elements(), self._plane_policy())
                    fa2_bad = v["fa2_layers_bad"]
                    fa2_rules_used["per-layer"] += 1
                    if fa2_bad:
                        fa2_detail = {"rule": v["rule"], "n_seq": v["n_seq"], "max_seqlen_q": v["max_seqlen_q"], "exact_seqlens": v["exact_seqlens"],
                                      "classes": v["classes"], "layers_bad": {k: d[:3] for k, d in list(v["layers_bad"].items())[:4]},
                                      "undeclared_layers": dict(list(v["undeclared_layers"].items())[:4])}
                else:
                    # R13 rule (no declarations, or no record of the step's active sequences): one whole-buffer row per layer, or (per-sequence
                    # compaction) 6 planes x sequences per layer -- every attention layer of the step must carry the SAME positive count
                    fa2_bad = FP.uniform_rule(fa2_layers, man["attention_layers"])
                    fa2_rules_used["uniform"] += 1
            moe_bad: dict[int, int] = {}
            # a padding-only step's MoE planes: one identity per (layer, member) -- SET semantics, as for mod_names (revb F-r17b-72: Counter(pad_ident)
            # counted one per padding META = per inactive request, so >= 2 inactive requests read n_inactive x moe_tensors_per_layer -> false refusal)
            moe_src = (Counter({n: 1 for n in pad_ident}) if padding_only else names)
            if man.get("moe_layers"):
                moe_counts = Counter(int(n.split("/L", 1)[1].split("/", 1)[0]) for n in moe_src.elements() if n.startswith("moe/L"))
                want, rs_layers = int(man.get("moe_tensors_per_layer") or 0), set(man.get("router_softmax_layers") or [])
                moe_bad = {k: moe_counts.get(k, 0) for k in man["moe_layers"] if moe_counts.get(k, 0) != want + (k in rs_layers)}
                moe_bad.update({k: c for k, c in moe_counts.items() if k not in man["moe_layers"]})
            elif any(n.startswith("moe/") for n in moe_src):
                moe_bad = {-1: sum(c for n, c in moe_src.items() if n.startswith("moe/"))}      # MoE tensors from no declared source
            n_ids = (1 if "runner.sampler/sampled_token_ids" in pad_ident else 0) if padding_only else names.get("runner.sampler/sampled_token_ids", 0)
            sampler_bad = bool(man["sampler_ids"]) and n_ids != man["sampler_ids"]
            mod_names, cb_bad = cb_src.split_step_names(mod_names, step, padding_only) if cb_src is not None else (mod_names, {})
            split_bad = {k: bad for k, bad in (("call_boundaries", cb_bad), ("routes", rt_bad)) if bad}
            ref_names = mod_names if ref_names is None else ref_names
            if padding_only:                                              # (b): set equality of identity names with step 0, not multiplicity
                diff = {n: (mod_names.get(n, 0), ref_names.get(n, 0)) for n in set(mod_names) ^ set(ref_names)}
            else:
                diff = {n: (mod_names.get(n, 0), ref_names.get(n, 0)) for n in set(mod_names) | set(ref_names) if mod_names.get(n, 0) != ref_names.get(n, 0)}
            if missing or unknown or fa2_bad or moe_bad or tp_bad or sampler_bad or diff or split_bad:
                violations.append({"step": step, "missing_modules": missing[:8], "unknown_modules": unknown[:8], "fa2_layers_bad": fa2_bad,
                                   "moe_layers_bad": moe_bad, "tp_sites_bad": dict(list(tp_bad.items())[:8]), "tp_sites_bad_n": len(tp_bad),
                                   "sampler_ids": n_ids, "multiplicity_vs_step0": dict(sorted(diff.items())[:8]),
                                   **{k: v for sn, bad in split_bad.items() for k, v in ((f"{sn}_bad", dict(sorted(bad.items())[:8])), (f"{sn}_bad_n", len(bad)))},
                                   **({"fa2_detail": fa2_detail} if fa2_detail else {}),
                                   **({"padding_only": padding_only, "padding_tensors": len(pad_metas)} if pad_metas else {})})
        return {"ok": not violations, "steps_checked": len(self._layouts),
                **({"padding": {"steps_with_padding": len(self._padding), "padding_only_steps": padding_only_steps, "padding_tensors": padding_tensors_n,
                                "rule": "padding tensors (<name>/<rid>/bot, host-only) are counted apart: a mixed step's executed metas are checked as an unpadded step's; "
                                        "a padding-only step must materialize every identity NAME of step 0 as bot (set equality) and cover every expected module"}}
                   if padded_run else {}),
                "manifest": {"modules": len(expected_modules), "attention_layers": len(man["attention_layers"]), "fa2_hidden_per_layer": man["fa2_hidden_per_layer"],
                             "sampler_ids": man["sampler_ids"], "moe_layers": len(man.get("moe_layers") or []), "moe_tensors_per_layer": man.get("moe_tensors_per_layer", 0),
                             "tp_sites": len(man.get("tp_sites") or []), "tp_rank": man.get("tp_rank"), "tp_sites_ruled_out": man.get("tp_sites_ruled_out") or []},
                # [F-r19-int-22] which attention-plane rule each step ran and what it was derived from
                "fa2_rule": {"steps": dict(fa2_rules_used), "attention_decls": len(decls),
                             "policy": {k: v for k, v in self._plane_policy().__dict__.items() if k != "ms" or v} if decls else None,
                             "classes_declared": (sorted(Counter(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha")
                                                                 for d in decls).items()) if decls else None)},
                "violations": violations[:5], "violations_n": len(violations)}

    def _plane_policy(self):
        """[F-r19-int-22] the collector's hidden-plane staging rule (`fa2_plane_classes.LayoutPolicy`): the explicit override, else read from
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
        """[F-r19-int-22] the step's active-sequence set (`fa2_plane_classes.StepSequences`): the collector's row record `step_rows[step]`
        (BIND-01: [(request, row_lo, row_hi)] in launch order) when present; else (n_seq, tokens) read from the executed layout -- the
        sampled-ids member has one slot per co-scheduled sequence, the integer ids input one row per token; None when neither is known
        (the caller then runs the R13 uniform rule for the step).  Never read from the hidden planes themselves."""
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
        """R14: the published per-step layouts as plain dicts ({step: [{name, dtype, shape, nbytes, numel, first_leaf, n_leaves, stream_off}]}) --
        the collector's CLAIM that lane manifest's `required_manifest.coverage_check` and bind's binding map are checked against."""
        out: dict[int, list[dict[str, Any]]] = {}
        for step in sorted(self._layouts):
            out[int(step)] = [{"name": m.name, "dtype": m.dtype, "shape": list(m.shape), "nbytes": int(m.nbytes), "numel": int(math.prod(m.shape)) if m.shape else 0,
                               "first_leaf": int(m.first_leaf), "n_leaves": int(m.n_leaves), "stream_off": int(m.stream_off)} for m in self._layouts[step]]
        return out

    def identity_of(self, step: int, index: int) -> dict:
        """Verifier-side: (step, rank) -> (tensor ordinal, name, dtype, shape, byte range) from the published layout."""
        metas = self._layouts[step]
        ps = self._padding_leaf(step, index)
        if ps is not None:                                          # [B9.1] a padding leaf: bytes of the padding block, tensors from its metas
            lo, hi = index * self.chunk, min((index + 1) * self.chunk, ps.stream_base + ps.n_bytes)
            over = [{"tensor": m.ordinal, "name": m.name, "dtype": m.dtype, "shape": list(m.shape),
                     "first_byte": max(lo, m.stream_off) - lo, "last_byte": min(hi, m.stream_off + m.nbytes) - lo}
                    for m in metas if m.stream_off < hi and m.stream_off + m.nbytes > lo]
            return {"packed": True, "padding": True, "byte_lo": lo, "byte_hi": hi, "tensors": over,
                    "tensor": over[0]["tensor"] if over else -1, "name": ";".join(o["name"] for o in over)}
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
        """[R17 fp8] True when leaf `index` of `step` opens from THIS committer's retained raw bytes; False when it must open by replay
        (raw bytes released / never kept / an excluded family's interior chunk)."""
        if self._padding_leaf(step, index) is not None:
            return True                                   # [B9.1] a padding leaf's bytes are the synthesized block's, always held
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
        determinism).  The sibling path comes from THIS committer's levels, or -- `paths_from_replay` (R17-5: regenerated values AND
        authentication-path material) -- from the replay's levels; either way `verify()` checks the result against THIS run's original
        roots, so a replay that differs in any byte (value or tree) fails to open."""
        out = []
        with self.spans.span("opening_generation"):
            for step, index in sample:
                metas = self._layouts[step]
                ident = self.identity_of(step, index)
                ps = self._padding_leaf(step, index)
                if ps is not None:                                   # [B9.1] a padding leaf: its bytes are the synthesized block's, path from the (re)folded levels
                    lo = (index - ps.leaf_base) * self.chunk
                    value = bytes(ps.stream[lo:lo + self.chunk])
                    out.append(Opening(step, index, value, scheme.path_in_levels(self._levels[step], index), ident))
                    continue
                if step in self._gpu_blocks:
                    blk, used = self._gpu_blocks[step]           # lazy D2H of the 256 B chunk (zero-padded past `used`) + level sibling path
                    lo = index * self.chunk
                    levels = self._gpu_levels[step]
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
                    value = bytes(blk[lo:lo + self.chunk].cpu().numpy().tobytes())
                    path = self._HG.gpu_open(levels, index) if self._HG is not None else open_levels(levels, index)
                    out.append(Opening(step, index, value, tuple(path), ident))
                    continue
                if ident.get("packed"):
                    value = bytes(self._streams[step][ident["byte_lo"]:ident["byte_hi"]])
                else:
                    m = metas[ident["tensor"]]
                    value = bytes(m.host[ident["byte_lo"]:ident["byte_hi"]])
                path = scheme.path_in_levels(self._levels[step], index)
                out.append(Opening(step, index, value, path, ident, self._salts_of(step, index, index + 1)))
        return out

    def verify(self, run: RunCommitment, opening: Opening) -> bool:
        """Production verifier core: shape rule + fold to the bare root, then the context binding to the ORIGINAL step root, then the
        step root's membership in the run root."""
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
        ps = self._padding_leaf(opening.step, opening.index)
        if ps is not None:                                           # [B9.1] a padding leaf: the padding block's leaf rule (NB = 1 header, or pos_leaf)
            from verity_vllm.commit import padding_steps as PSm

            if len(opening.value) != ident["byte_hi"] - ident["byte_lo"]:      # a chunk (the GPU tree's blocks are chunk multiples; a host tail may be short)
                return False
            leaf = PSm.leaf_digest_of(ps, opening.value, opening.index - ps.leaf_base)
            merkle = scheme.fold_path(leaf, opening.index, list(opening.path))
        elif opening.step in self._gpu_blocks:
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

    # ------------------------------------------------------------------ coverage

    # module (class-name) -> census root families whose NATURALLY-MATERIALIZED outputs the hook on that module binds
    FAMILY_OF_MODULE = {
        "Linear": ("Gemm_v1",), "RMSNorm": ("RMSNormFusedCuda_v2", "RMSNormTriton_v1"), "GemmaRMSNorm": ("RMSNormFusedCuda_v2", "RMSNormTriton_v1"),
        "SiluAndMul": ("SiluMul_v1", "SiluMul_v2"), "GeluAndMul": ("GeluMul_v1",), "RotaryEmbedding": ("RoPE_v1",), "Attention": ("Attention_v3",),
        "VocabParallelEmbedding": ("Embedding_v1",), "LogitsProcessor": ("Gemm_v1",), "Sampler": ("TokenSelect_v1",),
    }

    def _module_class(self, name: str) -> str:
        base = name.split("/")[0]
        if base == "runner.sampler":
            return "Sampler"
        for n, m in self._modules:
            if n == base:
                names = _mro_names(m)
                for k in self.FAMILY_OF_MODULE:
                    if k in names or (k == "Linear" and names & {"LinearBase", "ColumnParallelLinear", "RowParallelLinear"}):
                        return k
                return type(m).__name__
        leaf = base.rsplit(".", 1)[-1]
        if not self._modules and "norm" in leaf.lower():      # offline core (tests): infer from the module path
            return "RMSNorm"
        return leaf

    def coverage_declaration(self, run: RunCommitment) -> dict:
        by_class: dict[str, dict] = {}
        families: set[str] = set()
        for step, metas in self._layouts.items():
            for m in metas:
                if m.name.startswith(("fa2.m1/", "fa3.m1/", "moe/")) or m.name.endswith("/row4_f32"):
                    continue                                  # declared separately (hidden source / MoE block source / row-4 regenerate baseline)
                mc = self._module_class(m.name)
                compiled_kernel = len(m.name) > 3 and m.name[0] == "c" and m.name[1:3].isdigit() and m.name[3] == "."   # R15 S3: c<piece>.<kernel>/<role>
                if m.name.endswith("/in") or (compiled_kernel and m.name.endswith("/ids")):
                    cls, fam = "input_token_ids", (TOKEN_IDS_INPUT,)
                elif "sampled_token_ids" in m.name:
                    cls, fam = "sampled_token_ids", ("TokenSelect_v1",)
                elif compiled_kernel:
                    # [R15 compiled S3] P_compiled's VU boundaries: every non-weight argument of every Inductor kernel launch (the eager
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
            # kernel's fixed reduction order = the registry's exact model (`verity_vllm/program/registry/b1.py` RMSNormFusedCuda v2, 2340/2340 rows
            # conformant).  So the derived-value route applies with a ROW-sourced phi: opening = source-row leaf(s) + weight opening + op-by-op
            # replay; no interior word needs a leaf.  The verifier for that multi-source phi is NOT implemented in this lane -> the checker
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
