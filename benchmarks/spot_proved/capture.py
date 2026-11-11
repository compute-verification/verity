"""A vLLM general plugin that records one replayed step's interior, per TP rank, in an engine that serves with CUDA graphs.

    python capture.py site SITE --capture DIR --steps s[,s2]|--steps-from STEPS.json [--open dense:1,moe:1] [--mla-tokens 8] [--keep none|opened|all [--keep-steps s,...]] [--sampler] [--table L[,L2]]
    PYTHONPATH=SITE <a vLLM engine>

Registered as a ``vllm.general_plugins`` entry point (`site` writes the dist-info, and the capture's settings into
``SITE/spot_capture.json``, which the plugin reads beside itself), so it loads in every worker. It runs on
the V2 model runner, vLLM's default at the integration's pin. A step is counted when a worker's ``execute_model`` has
scheduled tokens and is not a dummy run. At step s (0-based), when the step dispatches to a CUDA graph (FULL, PIECEWISE or
breakable), the runner's graph call is preceded by one more forward of the same model on the same input buffers, attention
metadata, slot mappings and padded size, with the forward context's graph mode set to NONE; that forward is the one
recorded. A step the engine runs without a graph is recorded as it runs. What is recorded:

* every dispatched op (`Recorder`, a TorchDispatchMode): over the whole forward when the model is not compiled (the pin's
  default for GLM-5 and DeepSeek-V3.2, whose breakable CUDA graphs turn compilation off), or inside each
  ``PiecewiseBackend.__call__`` when it is, where values that pass only between two Inductor Triton kernels are not
  recorded. Inside vLLM's Python custom ops (``vllm::``) the ops they dispatch are recorded too, each naming its
  ``parent``. Each tensor argument is recorded before the call, and each return and written argument after it. A parameter
  is named, not copied. A tensor above ``BIG`` bytes (a KV cache) is recorded by shape only. An op whose ``Tensor[]``
  return can hold an undefined tensor (`OUTSIDE_MODE`: FA3's ``fwd``) is recorded the same way by a wrapper that runs it
  outside the mode, which couldn't hand such a list back;
* calls no dispatcher sees: each Triton fused-MoE launch (``invoke_fused_moe_triton_kernel``: A, its scales, the expert
  ids, the sorted token ids, the routed weights, the config, and C after the launch), and each sparse MLA kernel call
  (FlashMLA's ``_bf16_flash_mla_kernel``, or FA3's ``flash_attn_varlen_func`` in the FlashAttention sparse backend: the
  queries, the global top-k rows, the outputs, and the KV rows those indices read for ``--mla-tokens`` tokens: FlashMLA's
  first ones, FA3's with the most keys (``seqused_k``), the ones with the most splits, named in ``kv_gathered_rows``), and
  each other kernel `PY_KERNELS` lists, launched from Python (GLM-5.3's fused q/kv norm and RoPE with its cache writes, its
  fused query RoPE and quantization, the router's GEMM at up to 16 tokens, and the sparse backend's conversion of each
  token's top-k positions to global slots through its request's page table): its arguments by name, a paged cache by
  shape only, and after the call its returns and the arguments it writes.

Then the graph call runs as served, and its hidden states must equal the recorded forward's bit for bit (``check``). The
extra forward writes the same KV values to the same slots, so the step and every later one are unchanged.

Each recorded tensor is named by its SHA-512 when recorded: its bytes are copied to the host, where ``--hash-threads``
threads hash them while the forward goes on, holding at most ``--host-mib`` of copies at once. A whole paged cache an op
takes (`CACHES`) is hashed on its own device instead, by `chunks_sha512` (the SHA-512s of its `CHUNK`-byte pieces, one
device thread each), and never leaves it; the first ``--gpu-check`` a rank hashes there are hashed on the host too and must
agree. No tensor's bytes wait in memory for the coins: under ``--keep opened`` each is spooled to ``DIR/rank<r>/spool/`` as
it is hashed, and once the coins have opened layers the opened layers' tensors move, once each, to
``DIR/rank<r>/blobs/<sha512>`` and the rest are deleted; ``--keep all`` writes every one up to ``--keep-max-mib`` as it is
recorded and spools the rest; ``--keep none`` writes none (the step's commitments only). ``--keep-steps`` names the steps
``--keep`` applies to, any other keeping one level less (all, then opened, then none). Nothing is written or spooled once
the disk would fall below ``--min-free-gib``, and an opened tensor that wasn't is counted in ``commit.json``'s ``missing``.
``DIR/rank<r>/step<s>/ops.jsonl`` lists the events in order. ``step<s>/meta.json`` holds the step (tokens, padding, graph
mode, the scheduled requests and their token counts, the batch's request ids in its order), the check, where the step's
time and bytes went, and the rank's interior root for the step: SHA-512 over the SHA-512s of ``ops.jsonl``, of
``params.json``, which gives each parameter a recorded op read (its shape, dtype and SHA-512 over its bytes, hashed once per
engine and again only if its storage, layout or version changes), and of the batch (``batch_sha512``: the request ids in
order and their scheduled tokens). Every recorded event names its blobs by SHA-512 and its decoder layer, so the root
binds every recorded byte. With
``--allreduce-tree``, ``step<s>/allreduce_tree.json`` beside it holds the node's multimem reduction tree and where it was
read, for the node's registration to commit and the match gate to bind. Then
`_commit_and_open`: the step's root over the ranks' roots, the coins rank 0 samples only after that, and the layers they
open, whose parameters each rank writes beside the step (``commit.json``, ``open/``).

With ``--table L[,L2]``, layers with their own indexer, each recorded step also writes on every rank, outside the step's
root, each long row's history table at those layers (`_table_convert`): for each request whose last token of the step reads
more than ``--table-keys`` keys, its n latent rows as the layer's paged cache holds them once the step's own are written, the
n f32 scores its top-k call ranked, the top-k the sparse backend's slot conversion maps (the canonical top-k), and the rows
that attention read, under ``step<s>/table/`` (`_write_tables`), and the step's ``meta.json`` names the layers. `table.py`
checks that they join. The tables are private: ``SITE`` and ``DIR`` must lie outside any run directory, whose files custody
uploads, and are made mode 0700 (`private_dir`); give the steps by ``--steps-from`` to keep them off the command line.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

BIG = 1 << 30
#: with ``--keep all``, every parameter the step read up to this size is written too (``commit.json``'s ``kept_params``: the
#: norms' and biases' weights of the layers the coins didn't open, which a gate over every call reads)
SMALL_PARAM = 1 << 20
#: aten ops that only allocate, view or restride: their outputs alias or are uninitialized, never a value of the step
SKIP = {"empty", "empty_strided", "empty_like", "as_strided", "view", "_unsafe_view", "reshape", "alias", "detach", "t",
        "permute", "transpose", "expand", "unsqueeze", "squeeze", "select", "slice", "_reinterpret_tensor", "lift_fresh",
        "split", "split_with_sizes", "unbind", "chunk", "narrow", "view_as", "resize_", "set_", "record_stream"}
#: the paged KV caches an op takes whole: hashed into the root but never written, since the rows a call reads are the sparse
#: MLA tap's ``kv_gathered`` and no gate reads the rest
CACHES = {"_vllm_fa3_C::fwd": ("k", "v"), "_C_cache_ops::cp_gather_indexer_k_quant_cache": ("kv_cache",)}
#: the piece of a whole cache one device thread hashes (`chunks_sha512`): a multiple of 8, so each piece loads as 64-bit words
CHUNK = 16 << 10
#: `chunks_sha512` on the device, one thread a piece, from ``sha512.cuh`` (`site` copies it beside this file)
CHUNKS_CU = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "sha512.cuh"

__global__ void spot_chunks(const uint8_t* __restrict__ data, long long nbytes, long long chunk, long long n, uint8_t* out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long lo = i * chunk, len = nbytes - lo < chunk ? nbytes - lo : chunk, full = len / 128;
  const uint64_t* p = reinterpret_cast<const uint64_t*>(data + lo);
  uint64_t st[8], w[16];
  sha::iv(st);
  for (long long b = 0; b < full; ++b) {
#pragma unroll
    for (int j = 0; j < 16; ++j) w[j] = sx::bswap64(__ldg(reinterpret_cast<const unsigned long long*>(p) + b * 16 + j));
    sha::compress(st, w);
  }
  const int r = (int)(len - full * 128), nb = r + 17 <= 128 ? 1 : 2;
  uint8_t blk[256];
  for (int k = 0; k < 256; ++k) blk[k] = k < r ? data[lo + full * 128 + k] : k == r ? 0x80 : 0;
  sha::put_be64(blk + nb * 128 - 8, (uint64_t)len * 8);
  for (int b = 0; b < nb; ++b) {
    for (int j = 0; j < 16; ++j) w[j] = sha::be64(blk + b * 128 + 8 * j);
    sha::compress(st, w);
  }
  for (int k = 0; k < 8; ++k) sha::put_be64(out + i * 64 + 8 * k, st[k]);
}

torch::Tensor chunks(torch::Tensor flat, int64_t chunk) {
  TORCH_CHECK(flat.is_cuda() && flat.scalar_type() == torch::kUInt8 && flat.is_contiguous() && chunk % 8 == 0);
  TORCH_CHECK(flat.numel() == 0 || reinterpret_cast<uintptr_t>(flat.data_ptr<uint8_t>()) % 8 == 0);
  const int64_t nbytes = flat.numel(), n = (nbytes + chunk - 1) / chunk;
  auto out = torch::empty({n * 64}, flat.options());
  if (n) spot_chunks<<<(unsigned)((n + 127) / 128), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      flat.data_ptr<uint8_t>(), nbytes, chunk, n, out.data_ptr<uint8_t>());
  return out;
}
"""


#: the capture's settings, which `site` writes beside its copy of this file
CONFIG = "spot_capture.json"
#: the node's multimem reduction tree and where it was read, beside each step's ``params.json`` (stage.py's `TREE_FILE`)
TREE_FILE = "allreduce_tree.json"


def reduction_tree(text: str) -> list:
    """A binary reduction tree over ranks 0..W-1, as JSON pairs or as kernels' probe writes it, ``(((0,2),3),1)``."""
    tree = json.loads(text.replace("(", "[").replace(")", "]"))

    def leaves(x: Any) -> list[int]:
        if isinstance(x, int) and not isinstance(x, bool):
            return [x]
        if not (isinstance(x, list) and len(x) == 2):
            raise ValueError(f"allreduce tree {text}: a node is a rank or a pair of nodes, not {x!r}")
        return leaves(x[0]) + leaves(x[1])
    got = leaves(tree)
    if sorted(got) != list(range(len(got))):
        raise ValueError(f"allreduce tree {text}: its leaves must be the ranks 0..{len(got) - 1}, each once")
    return tree


def config() -> dict | None:
    """`site`'s settings, or None where none sit beside this file: the plugin then leaves the engine as it is."""
    p = Path(__file__).with_name(CONFIG)
    return json.loads(p.read_text()) if p.is_file() else None


class _State:
    def __init__(self, cfg: dict) -> None:
        self.dir = Path(cfg["capture"])
        self.targets = set(cfg["steps"])
        self.mla_tokens = int(cfg["mla_tokens"])
        self.sampler = bool(cfg.get("sampler"))
        self.allreduce_tree: dict | None = cfg.get("allreduce_tree")
        self.sample_at: int | None = None
        self.smp: dict | None = None
        self.open = dict(cfg["open"])
        self.step = -1
        self.active = False
        self.recording = False
        self.runner: Any = None
        self.sched: Any = None
        self.input_batch: Any = None
        self.attn: Any = None
        self.slot_maps: Any = None
        self.params: dict[tuple, str] = {}
        self.storages: dict[int, tuple[int, str]] = {}
        self.param_of: dict[str, Any] = {}
        self.used_params: set[str] = set()
        #: each parameter's `_param_digests` record for the engine's life, under the identity it was hashed at
        self.digests: dict[str, tuple[tuple, dict]] | None = {}
        self.seen: set[str] = set()
        self.spooled: dict[str, int] = {}
        self.dropped: dict[str, int] = {}
        self.keep = cfg["keep"]
        self.keep_steps = None if cfg.get("keep_steps") is None else set(cfg["keep_steps"])
        self.keep_max = int(cfg.get("keep_max_mib", 64)) << 20
        self.min_free = int(cfg.get("min_free_gib", 40)) << 30
        self.floor_hit = False
        self.threads = int(cfg.get("hash_threads", 0))
        self.host_budget = int(cfg.get("host_mib", 8192)) << 20
        self.cache_hash = cfg.get("cache_hash", "gpu")
        self.gpu_check = int(cfg.get("gpu_check", 4))
        self.gpu_checked = 0
        self.dev: Any = None
        self.hasher: _Hasher | None = None
        import threading
        self.lock = threading.RLock()
        self.writing = 0
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.events: list[dict] = []
        self.piece = -1
        self.parents: list[int] = []
        self.layer: int | None = None
        self.quiet = False
        self.whole = False
        self.check: dict | None = None
        self.t0 = time.time()
        self.rank = 0
        self.errors: list[str] = []
        self.bytes = 0
        self.table = {int(x) for x in cfg.get("table") or ()}
        self.table_keys = int(cfg.get("table_keys", 2048))
        self.tab: dict | None = None
        self.tables: list[dict] = []

    def root(self) -> Path:
        return self.dir / f"rank{self.rank}"

    @property
    def step_keep(self) -> str:
        """What this step keeps: ``--keep`` at the ``--keep-steps`` (default every step), one level less at any other."""
        if self.keep_steps is None or self.step in self.keep_steps:
            return self.keep
        return {"all": "opened"}.get(self.keep, "none")

    @property
    def keep_all(self) -> bool:
        """Whether this step writes every recorded tensor: ``--keep all``, at the ``--keep-steps`` (default every step)."""
        return self.step_keep == "all"

    def room(self, nbytes: int, root: Path | None = None) -> bool:
        """Whether ``nbytes`` more can go to the capture's disk and leave ``--min-free-gib`` free (with what is being written);
        once not, nothing more is written or spooled this step. Called under `lock`."""
        if self.floor_hit:
            return False
        import shutil
        root = root or self.root()
        root.mkdir(parents=True, exist_ok=True)
        self.floor_hit = shutil.disk_usage(root).free - self.writing - nbytes < self.min_free
        return not self.floor_hit

    def write_now(self, nbytes: int, mode: str | None = None, root: Path | None = None) -> bool:
        """Whether a blob is written as it is recorded: on a step kept whole, up to ``--keep-max-mib``, while the capture's
        disk keeps ``--min-free-gib`` free. Any other blob is spooled until the coins and kept only in an opened layer."""
        return (mode or self.step_keep) == "all" and nbytes <= self.keep_max and self.room(nbytes, root)

    def add(self, what: str, seconds: float = 0.0, nbytes: int = 0) -> None:
        """Adds to the step's time and bytes under ``what`` (meta.json's ``time_s`` and ``bytes``), from any thread."""
        with self.lock:
            self.t[what] = self.t.get(what, 0.0) + seconds
            if nbytes:
                self.n[what] = self.n.get(what, 0) + nbytes


class _Hasher:
    """SHA-512s of recorded tensors' host copies on ``--hash-threads`` threads while the forward goes on (at 0, each as it is
    recorded, so its reference is complete when its op is). A copy is made only once the copies held come to at most
    ``--host-mib`` with it (a larger one waits until none is held), so host memory stays bounded however many tokens the
    step has; `drain` waits for every job."""

    def __init__(self, threads: int, budget: int) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="spot-hash") if threads > 0 else None
        self.cv = threading.Condition()
        self.budget, self.held, self.peak = budget, 0, 0
        self.jobs: list = []

    def take(self, n: int) -> None:
        with self.cv:
            while self.held and self.held + n > self.budget:
                self.cv.wait()
            self.held += n
            self.peak = max(self.peak, self.held)

    def give(self, n: int) -> None:
        with self.cv:
            self.held -= n
            self.cv.notify_all()

    def run(self, n: int, fn, *args) -> None:
        def job():
            try:
                fn(*args)
            finally:
                self.give(n)
        if self.pool is None:
            job()
        else:
            self.jobs.append(self.pool.submit(job))

    def drain(self) -> None:
        from concurrent.futures import wait
        jobs, self.jobs = self.jobs, []
        wait(jobs)
        for j in jobs:
            j.result()


S: _State | None = None


def register() -> None:
    global S
    cfg = config()
    if cfg is None or S is not None:
        return
    S = _State(cfg)
    try:
        _install()
    except Exception:  # noqa: BLE001 -- a plugin that fails to install must say so and leave the engine as served
        print("[spot-capture] install failed:\n" + traceback.format_exc(), file=sys.stderr, flush=True)


def _note(what: str) -> None:
    print(f"[spot-capture] {what}", file=sys.stderr, flush=True)


def _wrap(owner: Any, name: str, make) -> None:
    orig = getattr(owner, name)
    setattr(owner, name, make(orig))


def _install() -> None:
    from vllm.compilation import piecewise_backend as pb
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu import model_runner as mr

    def execute_model(orig):
        def run(self, scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
            reqs = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            real = ((not dummy) and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
                    and not any(str(r).startswith("_warmup") for r in reqs))
            if real:
                S.step += 1
                S.active = S.step in S.targets
                S.runner, S.sched = self, scheduler_output
                S.sample_at = S.step if S.active and S.sampler else None
                if S.active:
                    _bind_model_state(self)
            try:
                return orig(self, scheduler_output, *a, **kw)
            finally:
                if S.active and S.events:
                    try:
                        _finish(self)
                    except Exception:  # noqa: BLE001
                        print(f"[spot-capture] rank {S.rank} step {S.step}: the capture failed, serving goes on\n"
                              f"{traceback.format_exc()}", flush=True)
                        _reset()
                S.active = False
        return run

    def prepare_inputs(orig):
        def run(self, *a, **kw):
            out = orig(self, *a, **kw)
            if S.active:
                S.input_batch = out
            return out
        return run

    def slot_maps(orig):
        def run(*a, **kw):
            out = orig(*a, **kw)
            if S.active:
                S.slot_maps = out
            return out
        return run

    def run_fullgraph(orig):
        def run(self, batch_desc, *a, **kw):
            if not S.active:
                return orig(self, batch_desc, *a, **kw)
            mine = _eager(S.runner, None, "FULL", batch_desc)
            out = orig(self, batch_desc, *a, **kw)
            _check(mine, out, "FULL", batch_desc)
            return out
        return run

    def run_pw_graph(orig):
        def run(self, model, model_inputs, *a, **kw):
            if not S.active:
                return orig(self, model, model_inputs, *a, **kw)
            mine = _eager(S.runner, model_inputs, "PIECEWISE", None)
            out = orig(self, model, model_inputs, *a, **kw)
            _check(mine, out, "PIECEWISE", None)
            return out
        return run

    def pb_call(orig):
        def call(self, *args):
            if not S.recording or S.whole:
                return orig(self, *args)
            S.piece = int(getattr(self, "piecewise_compile_index", -1))
            with Recorder():
                return orig(self, *args)
        return call

    if S.sampler:
        _install_sampler()
    _wrap(mr.GPUModelRunner, "execute_model", execute_model)
    _wrap(mr.GPUModelRunner, "prepare_inputs", prepare_inputs)
    _wrap(mr, "build_slot_mappings_by_layer", slot_maps)
    manager = getattr(cgu, "ModelCudaGraphManager", cgu.CudaGraphManager)
    _wrap(manager, "run_fullgraph", run_fullgraph)
    _wrap(manager, "run_pw_graph", run_pw_graph)
    _wrap(pb.PiecewiseBackend, "__call__", pb_call)
    _install_moe()
    _install_mla()
    _install_py_kernels()
    _install_outside_mode()


def _install_sampler() -> None:
    """Rank 0's sampler at each recorded step (``--sampler``), read only: the logits it takes, the logits entering top-k/top-p
    (after penalties and temperature) and leaving it, the Gumbel sample's per-row seed, position and temperature, and the chosen
    tokens, under ``step<s>/sampler/`` with their SHA-512s in ``sampler.json``, outside the step's root."""
    import torch
    from vllm.v1.worker.gpu.sample import sampler as smp

    def host(t: Any) -> Any:
        return None if t is None else t.detach().to("cpu", copy=True)

    def call(orig):
        def run(self, logits, input_batch, *a, **kw):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if S.sample_at is None or rank != 0:
                return orig(self, logits, input_batch, *a, **kw)
            S.smp = {"logits": host(logits)}
            out = None
            try:
                out = orig(self, logits, input_batch, *a, **kw)
                return out
            finally:
                rec, step, S.smp, S.sample_at = S.smp, S.sample_at, None, None
                try:
                    rec["chosen"] = host(getattr(out, "sampled_token_ids", None))
                    _write_sampler(step, rec, list(getattr(input_batch, "req_ids", None) or []))
                except Exception:  # noqa: BLE001
                    _note(f"step {step}: the sampler record failed, serving goes on\n{traceback.format_exc()}")
        return run

    def top_k_top_p(orig):
        def run(logits, top_k, top_p, *a, **kw):
            if S.smp is None:
                return orig(logits, top_k, top_p, *a, **kw)
            S.smp.update(topp_in=host(logits), top_k=host(top_k), top_p=host(top_p))
            out = orig(logits, top_k, top_p, *a, **kw)
            S.smp["topp_out"] = host(out)
            return out
        return run

    def gumbel(orig):
        def run(logits, expanded_idx_mapping, temperature, seed, pos, *a, **kw):
            out = orig(logits, expanded_idx_mapping, temperature, seed, pos, *a, **kw)
            if S.smp is not None:
                idx = expanded_idx_mapping.long()
                S.smp.update(row_state=host(idx), seed=host(seed[idx]), temperature=host(temperature[idx]), pos=host(pos),
                             gumbel_out=host(out), gumbel_args={k: v for k, v in kw.items() if isinstance(v, (bool, int, float))})
            return out
        return run

    _wrap(smp.Sampler, "__call__", call)
    _wrap(smp, "apply_top_k_top_p", top_k_top_p)
    _wrap(smp, "gumbel_sample", gumbel)


def _write_sampler(step: int, rec: dict, req_ids: list[str]) -> None:
    d = S.dir / "rank0" / f"step{step}" / "sampler"
    d.mkdir(parents=True, exist_ok=True)
    if rec.get("topp_out") is not None:
        rec["topp_kept"] = rec["topp_out"] != float("-inf")
    doc: dict = {"step": step, "req_ids": req_ids, "gumbel_args": rec.pop("gumbel_args", None), "tensors": {},
                 "path": "gumbel" if "gumbel_out" in rec else "flashinfer (no per-request seed in the batch)"}
    for k, t in rec.items():
        if t is None:
            continue
        raw = _flat_bytes(t).numpy().tobytes()
        (d / f"{k}.bin").write_bytes(raw)
        doc["tensors"][k] = {"shape": list(t.shape), "dtype": str(t.dtype), "sha512": hashlib.sha512(raw).hexdigest()}
    (d / "sampler.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    _note(f"step {step}: the sampler's {len(doc['tensors'])} tensors over {len(req_ids)} requests -> {d}")


def _bind_model_state(runner: Any) -> None:
    ms = runner.model_state
    if getattr(ms, "_spot_bound", False):
        return
    orig = ms.prepare_attn

    def prepare_attn(*a, **kw):
        out = orig(*a, **kw)
        if S.active:
            S.attn = out
        return out
    ms.prepare_attn = prepare_attn
    ms._spot_bound = True
    model = runner.model
    import re
    for name, mod in model.named_modules():
        if m := re.search(r"(?:^|\.)layers\.(\d+)$", name):
            def pre(_mod, _args, n=int(m[1])):
                S.layer = n

            def post(_mod, _args, _out):
                S.layer = None
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
    fwd = model.forward

    def forward(*a, **kw):
        if not S.active or S.recording or _graph_mode() != "NONE":
            return fwd(*a, **kw)
        out = _recorded(runner, "NONE", lambda: fwd(*a, **kw))
        S.check = {"mode": "NONE", "recorded_forward": out is not None, "served_forward": True}
        return out if out is not None else fwd(*a, **kw)
    model.forward = forward


def _graph_mode() -> str:
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return "NONE"
    mode = get_forward_context().cudagraph_runtime_mode
    return getattr(mode, "name", str(mode))


def _install_moe() -> None:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    try:
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
    except ImportError as e:
        _note(f"no fused_moe.experts.triton_moe ({e}): only fused_moe's own launch is wrapped")
        triton_moe = None
    names = ("A", "B", "C", "A_scale", "B_scale", "topk_weights", "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
             "mul_routed_weight", "top_k", "config")

    def expert(orig):
        def launch(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            args = dict(zip(names, a))
            args.update({k: v for k, v in kw.items() if k in names or k == "block_shape"})
            ins = {k: _ref(v) for k, v in args.items() if k != "C"}
            out = orig(*a, **kw)
            _event("moe_launch", ins, {"C": _ref(args["C"])})
            return out
        return launch

    for mod in (fm, triton_moe):
        if mod is not None and hasattr(mod, "invoke_fused_moe_triton_kernel"):
            _wrap(mod, "invoke_fused_moe_triton_kernel", expert)


def _install_mla() -> None:
    try:
        from vllm.v1.attention.backends.mla import flashmla_sparse as fs
    except ImportError as e:
        _note(f"no sparse MLA backend ({e}): no sparse MLA kernel call is recorded")
        return
    import torch

    def kernel(orig):
        def run(self, q, kv_rows, topk_indices, *a, **kw):
            if not S.recording:
                return orig(self, q, kv_rows, topk_indices, *a, **kw)
            rest = dict(zip(("topk_length", "actual_num_heads"), a))
            rest.update(kw)
            ins = {"q": _ref(q), "kv_rows": _ref(kv_rows), "topk_indices": _ref(topk_indices), **{k: _ref(v) for k, v in rest.items()}}
            n = min(S.mla_tokens, int(topk_indices.shape[0]))
            if n:
                with _quiet():
                    idx = topk_indices[:n].long().clamp(min=0)
                    rows = kv_rows.reshape(-1, kv_rows.shape[-1])[idx.reshape(-1)].reshape(n, idx.shape[1], -1)
                ins["kv_gathered"] = _ref(rows)
                ins["kv_gathered_tokens"] = n
            out = orig(self, q, kv_rows, topk_indices, *a, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            _event("sparse_mla_kernel", ins, {f"r{i}": _ref(t) for i, t in enumerate(outs) if isinstance(t, torch.Tensor)})
            _table_read(lambda idx: kv_rows.reshape(-1, kv_rows.shape[-1])[idx], topk_indices)
            return out
        return run

    if hasattr(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel"):
        _wrap(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel", kernel)
    try:
        from vllm.v1.attention.backends.mla import flashattn_mla_sparse as fa
    except ImportError as e:
        _note(f"no FlashAttention sparse MLA backend ({e}): only FlashMLA's kernel is recorded")
        return

    def varlen(orig):
        def run(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            ins = {k: _ref(v) for k, v in kw.items() if k not in ("k", "v")}
            k, v, table = kw["k"], kw["v"], kw["block_table"]
            ins["k"] = {"shape": list(k.shape), "dtype": str(k.dtype), "kv_rows": True}
            n = min(S.mla_tokens, int(table.shape[0]))
            if n:
                with _quiet():
                    used = kw.get("seqused_k")
                    pick = (torch.sort(torch.argsort(used[:table.shape[0]].long(), descending=True, stable=True)[:n]).values
                            if used is not None else torch.arange(n, device=table.device))
                    idx = table[pick].long().clamp(min=0).reshape(-1)
                    rows = k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1)
                    gathered = torch.cat([rows[1][idx], rows[0][idx]], dim=-1).reshape(n, table.shape[1], -1)
                ins["kv_gathered"] = _ref(gathered)
                ins["kv_gathered_tokens"] = n
                ins["kv_gathered_rows"] = [int(x) for x in pick.tolist()]
            out = orig(*a, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            _event("fa3_sparse_mla", ins, {f"r{i}": _ref(t) for i, t in enumerate(outs) if isinstance(t, torch.Tensor)})
            _table_read(lambda idx: torch.cat([v.reshape(v.shape[0], -1)[idx], k.reshape(k.shape[0], -1)[idx]], dim=-1), table)
            return out
        return run

    _wrap(fa, "flash_attn_varlen_func", varlen)


#: kernels the served forward launches from Python, which no dispatcher sees: the module its caller looks the name up in, the
#: name (also the event's op), its positional arguments' names, the arguments it writes, and the paged caches it writes
#: whole, recorded by shape only
PY_KERNELS = (
    ("vllm.models.deepseek_v32.attention", "fused_norm_rope",
     ("positions", "q_c", "q_rms_norm_w", "q_rms_eps", "kv_c", "kv_rms_norm_w", "kv_rms_eps", "k_pe", "k_rope_cos_sin_cache",
      "index_k", "index_k_layer_norm_w", "index_k_layer_norm_bias", "index_k_layer_norm_eps", "index_k_rope_cos_sin_cache",
      "topk_indices_buffer"),
     ("q_c_out", "kv_c_out", "k_pe_out", "index_k_out", "topk_indices_buffer"), ("mla_kv_cache", "indexer_k_cache")),
    ("vllm.models.deepseek_v32.attention", "fused_q",
     ("positions", "q_pe", "q_pe_cos_sin_cache", "index_q", "index_q_cos_sin_cache", "ql_nope", "q_scale", "index_weights",
      "index_weights_softmax_scale", "index_weights_head_scale"), (), ()),
    ("vllm.model_executor.kernels.linear.cute_dsl.ll_bf16", "ll_bf16_gemm", ("hidden_states", "router_weight", "output_dtype"),
     (), ()),
    ("vllm.v1.attention.backends.mla.flashattn_mla_sparse", "triton_convert_req_index_to_global_index",
     ("req_id", "block_table", "token_indices"), (), ()),
)


def _py_kernel(op: str, names: tuple[str, ...], written: tuple[str, ...], caches: tuple[str, ...]):
    def make(orig):
        def run(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            args = {**dict(zip(names, a)), **kw}
            ins = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "kv_rows": True} if k in caches and v is not None
                   else _ref(v) for k, v in args.items()}
            out = orig(*a, **kw)
            outs = {f"r{i}": _ref(t) for i, t in enumerate(out if isinstance(out, tuple) else (out,))}
            outs.update({k: _ref(args[k]) for k in written if args.get(k) is not None})
            _event(op, ins, outs)
            if op in _TABLE and S.layer in S.table:
                _TABLE[op](args)
            return out
        return run
    return make


def _install_py_kernels() -> None:
    import importlib
    for mod, name, names, written, caches in PY_KERNELS:
        try:
            owner = importlib.import_module(mod)
        except ImportError as e:
            _note(f"no {mod} ({e}): {name} is not recorded")
            continue
        if hasattr(owner, name):
            _wrap(owner, name, _py_kernel(name, names, written, caches))


#: the indexer's top-k ops (`dsa_topk.OPS`) by the position of the output among their arguments, the logits first: prefill's
#: (logits, rowStarts, rowEnds, indices, numRows, ...) with keys [rowStarts, rowEnds), decode's (logits, next_n, seqLens,
#: indices, numRows, ...) with keys [0, seqLens), and (logits, lengths, output, ...) with keys [0, lengths)
TOPK_OPS = {"_C::top_k_per_row_prefill": 3, "_C::top_k_per_row_decode": 3, "_C::persistent_topk": 2,
            "_C::cooperative_topk": 2}


def _table_norm(args: dict) -> None:
    """A ``--table`` layer's `fused_norm_rope`, once it has written the step's own latent rows: the paged latent cache, the top-k
    buffer the layer's top-k calls write into, and the rows to keep: each request's last token of the step (vLLM lays the
    step's tokens out by request, in the batch's order, each its scheduled count of them) whose row reads more than
    ``--table-keys`` keys."""
    ids = [str(x) for x in getattr(S.input_batch, "req_ids", None) or []]
    counts = getattr(S.sched, "num_scheduled_tokens", None) or {}
    pos, want, end = args.get("positions"), {}, 0
    with _quiet():
        for r, rid in enumerate(ids):
            c = int(counts.get(rid, 0))
            end += c
            if c and (p := int(pos[end - 1])) + 1 > S.table_keys:
                want[end - 1] = {"layer": S.layer, "request": rid, "batch_index": r, "token": end - 1, "position": p, "n": p + 1}
    S.tab = {"layer": S.layer, "cache": args.get("mla_kv_cache"), "buf": args.get("topk_indices_buffer"), "want": want,
             "dtype": args.get("mla_kv_cache_dtype", "auto")}


def _table_topk(op: str, flat: list) -> None:
    """A top-k call of a ``--table`` layer: the scores of each kept token's row it ranks, and the row's place in its output. A
    call's output is a view of the layer's top-k buffer, its offset there naming its first token; a call whose output is not
    one ranks a kept token's row where exactly one of its rows has that token's count of keys."""
    import torch
    t = S.tab
    if t is None or t["layer"] != S.layer or not t["want"]:
        return
    out, logits, buf = flat[TOPK_OPS[op]], flat[0], t["buf"]
    with _quiet():
        rows = int(flat[4]) if TOPK_OPS[op] == 3 else int(logits.shape[0])
        if op == "_C::top_k_per_row_prefill":
            lo, hi = flat[1][:rows].long(), flat[2][:rows].long()
        elif op == "_C::top_k_per_row_decode":
            if int(flat[1]) != 1:
                S.errors.append(f"step {S.step} table layer {S.layer}: {op} with next_n {int(flat[1])}, one token a row is read")
                return
            hi = flat[2].reshape(-1)[:rows].long()
            lo = torch.zeros_like(hi)
        else:
            hi = flat[1].reshape(-1)[:rows].long()
            lo = torch.zeros_like(hi)
        same = buf is not None and out.untyped_storage().data_ptr() == buf.untyped_storage().data_ptr()
        width = buf.stride(0) * buf.element_size() if same else 0
        off = out.data_ptr() - buf.data_ptr() if same else -1
        view = same and off >= 0 and off % width == 0 and out.stride(0) == buf.stride(0)
        for i, e in t["want"].items():
            found = ([i - off // width] if 0 <= i - off // width < rows else []) if view else (
                torch.nonzero(hi - lo == e["n"]).reshape(-1).tolist())
            if len(found) != 1:
                continue
            row = found[0]
            if "op" in e:
                e["why"] = f"{e['op']} and {op} both rank its row"
            if logits.dtype != torch.float32:
                e["why"] = f"{op} ranks {logits.dtype} logits, not the f32 scores DsaTopKMaskCounted ranks"
            e.update(op=op, op_row=row, by="view" if view else "keys", keys=[int(lo[row]), int(hi[row])], op_out=out,
                     scores=logits[row, int(lo[row]):int(hi[row])].clone())


def _table_convert(args: dict) -> None:
    """A ``--table`` layer's top-k-to-slot conversion: for each kept token, its n latent rows as the paged cache holds them
    (row i at position i, [kv_c | k_pe], the step's own included), the n scores its top-k call ranked, and the top-k the
    conversion maps, which is that call's row as the canonical top-k left it (`dsa_topk`'s plugin rewrites the op's output in
    place once it returns). A token whose request, keys or top-k don't join, or whose cache holds other than the bf16 rows
    ``DsaLatentRow_v1`` outputs (an fp8 cache), is an error of the step."""
    import torch
    t = S.tab
    if t is None or t["layer"] != S.layer:
        return
    req, table, topk = args["req_id"], args["block_table"], args["token_indices"]
    cache, bs = t["cache"], int(args.get("BLOCK_SIZE") or 64)
    with _quiet():
        for i, e in sorted(t["want"].items()):
            out, why = e.pop("op_out", None), e.pop("why", None)
            if why is None and (i >= req.shape[0] or int(req[i]) != e["batch_index"]):
                why = f"the conversion's token {i} is not of batch index {e['batch_index']}"
            elif why is None and out is None:
                why = "no top-k call of the layer ranks its row"
            elif why is None and e["keys"][1] - e["keys"][0] != e["n"]:
                why = f"{e['op']} ranks keys [{e['keys'][0]}, {e['keys'][1]})"
            elif why is None and not torch.equal(out[e["op_row"]], topk[i]):
                why = f"the conversion's top-k of token {i} is not {e['op']}'s row {e['op_row']}"
            elif why is None and (cache is None or cache.dim() != 3 or cache.shape[1] != bs):
                why = f"no paged latent cache [blocks, {bs}, words] at the layer's fused_norm_rope"
            elif why is None and (cache.dtype != torch.bfloat16 or t["dtype"] not in ("auto", "bfloat16")):
                why = f"a {cache.dtype} cache written as {t['dtype']}: a table holds bf16 rows only"
            if why:
                S.errors.append(f"step {S.step} table layer {e['layer']} request {e['request']} token {i}: {why}")
                continue
            p = torch.arange(e["n"], device=cache.device)
            S.tables.append({**e, "kv": cache[table[e["batch_index"]].long()[p // bs], p % bs].clone(),
                             "topk": topk[i].clone()})
    t["want"] = {}


def _table_read(rows_at: Callable[[Any], Any], slots: Any) -> None:
    """The rows the layer's sparse attention read for each table token: ``rows_at`` of the token's row of global slots."""
    for e in S.tables:
        if e["layer"] == S.layer and "read" not in e:
            with _quiet():
                e["read"] = rows_at(slots[e["token"]].long().clamp(min=0)).clone()


_TABLE = {"fused_norm_rope": _table_norm, "triton_convert_req_index_to_global_index": _table_convert}


#: what marks a research run directory (``research.runs``), every file of which custody may upload to the store
RUN_MARKS = ("job.json", "remote.json")


def run_dir_of(p: Path) -> Path | None:
    """The research run directory ``p`` lies in (``p`` or an ancestor holding a ``RUN_MARKS`` file), else None."""
    p = Path(p).resolve()
    return next((q for q in (p, *p.parents) if any((q / m).is_file() for m in RUN_MARKS)), None)


def private_dir(p: Path, what: str) -> Path:
    """``p``, made mode 0700; refused (SystemExit) inside a run directory, whose files custody uploads."""
    if (run := run_dir_of(p)) is not None:
        raise SystemExit(f"{what} {p} lies inside the run directory {run}, whose files custody uploads to the store: give it "
                         "a private directory outside the run directory")
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.chmod(0o700)
    return p


def _private_write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    path.chmod(0o600)


def _write_tables(d: Path) -> dict[str, Any]:
    """``step<s>/table/`` (outside the step's root, as the sampler's records are): each ``--table`` row's ``<k>.kv`` (n rows of
    u16 words, the cache's bf16 bits), ``<k>.scores`` (n f32), ``<k>.topk`` (i32) and ``<k>.read`` (the rows attention read, one
    a top-k slot), little-endian, with their shapes and SHA-512s in ``table.json``; the directory 0700 and its files 0600. A
    capture directory that others can read or that lies inside a run directory gets no table and an error instead. Returns
    what the step's ``meta.json`` records of them: the layers (decoder layer ``i`` of ``layers.<i>``), the key count, the rows
    written and ``table.json``'s SHA-512."""
    out = {"layers": sorted(S.table), "keys": S.table_keys, "rows": 0}
    run, mode = run_dir_of(S.dir), S.dir.stat().st_mode & 0o777
    if run is not None or mode & 0o077:
        why = f"inside the run directory {run}" if run is not None else f"mode {mode:o}, not 0700"
        S.errors.append(f"table: no table written, the capture directory is {why}")
        S.tables.clear()
        return {**out, "refused": why}
    td = d / "table"
    td.mkdir(parents=True, exist_ok=True, mode=0o700)
    td.chmod(0o700)
    doc = {"rank": S.rank, "step": S.step, "keys": S.table_keys, "tables": []}
    for k, e in enumerate(S.tables):
        files = {}
        for name in ("kv", "scores", "topk", "read"):
            v = e.pop(name, None)
            if v is not None:
                raw = _flat_bytes(v).cpu().numpy()
                _private_write(td / f"{k}.{name}", raw.tobytes())
                files[name] = {"shape": list(v.shape), "dtype": str(v.dtype), "sha512": hashlib.sha512(raw).hexdigest()}
        doc["tables"].append({**e, "files": files})
    raw = json.dumps(doc, indent=1, sort_keys=True).encode()
    _private_write(td / "table.json", raw)
    return {**out, "rows": len(doc["tables"]), "sha512": hashlib.sha512(raw).hexdigest()}


def _recorder_base():
    from torch.utils._python_dispatch import TorchDispatchMode
    return TorchDispatchMode


class Recorder(_recorder_base()):
    """Records each dispatched op, and inside vLLM's Python custom ops (``vllm::``) the ops they dispatch, with ``parent``."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        import torch
        kwargs = kwargs or {}
        packet = getattr(func, "_overloadpacket", None)
        base = getattr(packet, "__name__", str(func)).split(".")[-1]
        ns = getattr(func, "namespace", "aten")
        if S.quiet or (ns == "aten" and base in SKIP):
            return func(*args, **kwargs)

        def call(seq: int) -> Any:
            if ns == "vllm" and torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), "CUDA"):
                S.parents.append(seq)
                try:
                    with self:
                        return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA), *args, **kwargs)
                finally:
                    S.parents.pop()
            return func(*args, **kwargs)
        return _record_op(f"{ns}::{base}", func._schema, args, kwargs, call)


def _record_op(op: str, schema: Any, args: tuple, kwargs: dict, call) -> Any:
    """One op's event: its arguments by schema name before ``call(seq)``, and its returns and written arguments after it."""
    flat = list(args) + [kwargs.get(a.name) for a in schema.arguments[len(args):]]
    cache = CACHES.get(op, ())
    ins = {a.name: _ref(v, write=a.name not in cache) for a, v in zip(schema.arguments, flat)}
    seq = _event(op, ins, {})
    out = call(seq)
    outs = {f"ret{i}": _ref(t) for i, t in enumerate(out if isinstance(out, (tuple, list)) else (out,))}
    for a, v in zip(schema.arguments, flat):
        if a.alias_info is not None and a.alias_info.is_write:
            outs[a.name] = _ref(v, write=a.name not in cache)
    S.events[seq]["out"] = outs
    if S.tab is not None and op in TOPK_OPS:
        _table_topk(op, flat)
    return out


#: ops returning a ``Tensor[]`` that can hold an undefined tensor (FA3's split-KV accumulators when a call doesn't split), which
#: a Python dispatch mode can't hand back: it sees the undefined one as None, and C++ refuses a None in a ``Tensor[]``
#: ("Unable to cast Python instance of type <class 'list'>"). Each is recorded as `Recorder` would, by a wrapper on
#: ``torch.ops.<ns>.<name>`` that runs the op with the Python modes off.
OUTSIDE_MODE = (("_vllm_fa3_C", "fwd"),)


def _install_outside_mode() -> None:
    import torch
    from torch.utils._python_dispatch import _disable_current_modes, _get_current_dispatch_mode
    for ns, name in OUTSIDE_MODE:
        space = getattr(torch.ops, ns)
        try:
            packet = getattr(space, name)
        except AttributeError as e:
            _note(f"no {ns}::{name} ({e}): it is recorded under the mode")
            continue

        def run(*a, _packet=packet, _op=f"{ns}::{name}", **kw):
            if S.quiet or not isinstance(_get_current_dispatch_mode(), Recorder):
                return _packet(*a, **kw)

            def call(seq: int) -> Any:
                with _disable_current_modes():
                    return _packet(*a, **kw)
            return _record_op(_op, _packet.default._schema, a, kw, call)
        setattr(space, name, run)


def _params(runner: Any) -> None:
    if S.params:
        return
    for name, t in list(runner.model.named_parameters()) + list(runner.model.named_buffers()):
        if t.numel() == 0:
            continue
        S.params[(t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))] = name
        S.param_of[name] = t
        lo = t.untyped_storage().data_ptr()
        S.storages.setdefault(lo, (lo + t.untyped_storage().nbytes(), name))


def _param_digests(names: Any, cache: dict | None) -> dict:
    """Each named parameter's shape, dtype and SHA-512 over its bytes (``params.json``), on ``--hash-threads`` threads. With
    ``cache`` (the engine's, `_State.digests`) each tensor is hashed once for the engine's life, and again only if its
    storage, shape, strides, dtype or version counter changed, which serving never does."""
    from concurrent.futures import ThreadPoolExecutor

    def digest(name: str) -> tuple[str, dict]:
        t = S.param_of[name]
        key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), t._version)
        hit = cache.get(name) if cache is not None else None
        if hit is not None and hit[0] == key:
            return name, hit[1]
        raw = _flat_bytes(t).cpu().numpy()
        rec = {"shape": list(t.shape), "dtype": str(t.dtype), "sha512": hashlib.sha512(memoryview(raw)).hexdigest()}
        S.add("params_hashed", 0.0, raw.nbytes)
        if cache is not None:
            cache[name] = (key, rec)
        return name, rec
    with ThreadPoolExecutor(max(1, S.threads)) as pool:
        return dict(pool.map(digest, sorted(names)))


def _param_view(v: Any) -> dict | None:
    """A tensor that is a parameter, or a view of one's storage (a transposed or reshaped weight), named with its view."""
    key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype))
    if key in S.params:
        S.used_params.add(S.params[key])
        return {"param": S.params[key], "shape": list(v.shape), "dtype": str(v.dtype)}
    if v.numel() == 0:
        return None
    hit = S.storages.get(v.untyped_storage().data_ptr())
    if hit is None:
        return None
    S.used_params.add(hit[1])
    base = S.param_of[hit[1]]
    return {"param": hit[1], "shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride()),
            "byte_offset": v.data_ptr() - base.data_ptr()}


class _quiet:
    """Tensor work of the capture's own (copies, gathers): not recorded."""

    def __enter__(self):
        self.was, S.quiet = S.quiet, True

    def __exit__(self, *exc):
        S.quiet = self.was


def _ref(v: Any, write: bool = True) -> Any:
    with _quiet():
        return _ref_quiet(v, write)


def _write_file(p: Path, raw: Any) -> None:
    import threading
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{threading.get_ident()}")
    tmp.write_bytes(memoryview(raw).cast("B"))
    os.replace(tmp, p)


def _write_blob(h: str, raw: Any, root: Path | None = None) -> None:
    p = (root or S.root()) / "blobs" / h[:2] / h
    if not p.exists():
        _write_file(p, raw)


def chunks_sha512(raw: Any, chunk: int = CHUNK) -> str:
    """A whole cache's name: SHA-512 over a tag, ``chunk`` and the byte count (8 bytes each, big-endian), then the SHA-512 of
    each ``chunk`` bytes in order, the last piece possibly shorter. Each piece hashes on its own device thread (`CHUNKS_CU`);
    this is the host's."""
    mv = memoryview(raw).cast("B")
    return _chunks_root(b"".join(hashlib.sha512(mv[i:i + chunk]).digest() for i in range(0, len(mv), chunk)), len(mv), chunk)


def _chunks_root(digests: bytes, nbytes: int, chunk: int) -> str:
    return hashlib.sha512(b"verity/spot/chunks-sha512\0" + chunk.to_bytes(8, "big") + nbytes.to_bytes(8, "big")
                          + digests).hexdigest()


def _device_chunks() -> Any:
    """`CHUNKS_CU` built for the visible devices, once (torch's extension cache keeps it across engines; its builder runs
    ``ninja`` and ``nvcc`` from PATH and CUDA_HOME), or None where it can't be: then a whole cache is chunk-hashed on the
    host, under the same name."""
    if S.dev is None:
        try:
            from torch.utils.cpp_extension import load_inline
            here = Path(__file__).resolve()
            hdr = next(p for p in (here.with_name("sha512.cuh"), here.parents[2] / "verity/ml/kernels/sha512.cuh") if p.is_file())
            tag = hashlib.sha512(CHUNKS_CU.encode() + hdr.read_bytes()).hexdigest()[:16]
            t = time.time()
            S.dev = load_inline(f"spot_chunks_{tag}", cpp_sources="torch::Tensor chunks(torch::Tensor flat, int64_t chunk);",
                                cuda_sources=CHUNKS_CU, functions=["chunks"], extra_include_paths=[str(hdr.parent)],
                                extra_cuda_cflags=["-O3"])
            S.add("device_build", time.time() - t)
        except Exception as e:  # noqa: BLE001 -- the host computes the same names, slower
            _note(f"rank {S.rank}: no device chunk hashing ({e!r}); whole caches are chunk-hashed on the host")
            S.dev = False
    return S.dev or None


def _keep(h: str, raw: Any, root: Path, mode: str) -> None:
    """A written tensor's bytes, once a step, under the rank's ``root`` and the step's keep ``mode`` (both as recorded): on
    the disk now (`write_now`), spooled until the coins (``--keep opened``, or over ``--keep all``'s cap), or not kept
    (``--keep none``, or past the disk floor: counted in `dropped`)."""
    n = raw.nbytes
    with S.lock:
        if h in S.seen:
            return
        S.seen.add(h)
        S.bytes += n
        now = S.write_now(n, mode, root)
        spool = not now and mode != "none" and S.room(n, root)
        if now or spool:
            S.writing += n
        elif mode != "none":
            S.dropped[h] = n
    if not (now or spool):
        return
    t = time.time()
    try:
        if now:
            _write_blob(h, raw, root)
        else:
            _write_file(root / "spool" / h, raw)
    finally:
        with S.lock:
            S.writing -= n
            if spool:
                S.spooled[h] = n
            S.add("write_now" if now else "spool", time.time() - t, n)


def _hash_job(raw: Any, ref: dict, write: tuple[Path, str] | None, check: str | None = None) -> None:
    t = time.time()
    if check is None:
        ref["sha512"] = hashlib.sha512(memoryview(raw).cast("B")).hexdigest()
        S.add("hash_host", time.time() - t, raw.nbytes)
        if write is not None:
            _keep(ref["sha512"], raw, *write)
        return
    got = chunks_sha512(raw)
    S.add("hash_check", time.time() - t, raw.nbytes)
    if got != check:
        S.errors.append(f"a {ref.get('dtype')} {ref.get('shape')} cache's device chunk hash {check[:16]} is not the "
                        f"host's {got[:16]}: its name is the host's, and the device hashes no more caches")
        ref["chunks_sha512"], S.dev = got, False


def _host_copy(v: Any, ref: dict, write: bool, check: str | None = None) -> None:
    """``v``'s bytes on the host, once the hashing pool has room (`_Hasher`), hashed there (`_hash_job`)."""
    if S.hasher is None:
        S.hasher = _Hasher(S.threads, S.host_budget)
    nb = v.numel() * v.element_size()
    t = time.time()
    S.hasher.take(nb)
    S.add("host_wait", time.time() - t)
    try:
        t = time.time()
        x = _flat_bytes(v)
        raw = (x.cpu() if x.device.type == "cuda" else x.clone()).numpy()
        S.add("host_copy", time.time() - t, nb)
    except BaseException:
        S.hasher.give(nb)
        raise
    S.hasher.run(nb, _hash_job, raw, ref, (S.root(), S.step_keep) if write else None, check)


def _flat_bytes(t: Any) -> Any:
    """``t``'s bytes in row-major order, as a flat uint8 view. ``contiguous`` leaves a one-element view its stride (a size-1
    dimension is contiguous under any stride), which a uint8 ``view`` refuses, so such a tensor is copied."""
    import torch
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() and flat.stride(0) != 1:
        flat = flat.clone(memory_format=torch.contiguous_format)
    return flat.view(torch.uint8)


def _cache_ref(v: Any, d: dict) -> dict:
    """A whole cache's reference, named by `chunks_sha512` on its device (or on the host where no kernel builds)."""
    flat = _flat_bytes(v)
    if flat.numel() and flat.data_ptr() % 8:
        flat = flat.clone()
    ref = {**d, "chunk": CHUNK}
    dev = _device_chunks()
    if dev is None:
        t = time.time()
        ref["chunks_sha512"] = chunks_sha512(flat.cpu().numpy())
        S.add("hash_cache_host", time.time() - t, flat.numel())
        return ref
    t = time.time()
    ref["chunks_sha512"] = _chunks_root(dev.chunks(flat, CHUNK).cpu().numpy().tobytes(), flat.numel(), CHUNK)
    S.add("hash_device", time.time() - t, flat.numel())
    if S.gpu_checked < S.gpu_check:
        S.gpu_checked += 1
        _host_copy(flat, ref, False, check=ref["chunks_sha512"])
    return ref


def _blobs(v: Any):
    if isinstance(v, dict):
        if isinstance(v.get("sha512"), str) and "shape" in v:
            yield v["sha512"]
        for x in v.values():
            yield from _blobs(x)
    elif isinstance(v, list):
        for x in v:
            yield from _blobs(x)


def _ref_quiet(v: Any, write: bool = True) -> Any:
    import torch
    if isinstance(v, torch.Tensor):
        if v.device.type == "cuda" and (pv := _param_view(v)) is not None:
            return pv
        nb = v.numel() * v.element_size()
        d = {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
        if v.device.type != "cuda" and v.device.type != "cpu":
            return {**d, "device": v.device.type}
        if nb > BIG:
            return {**d, "big": nb}
        if not write and v.device.type == "cuda" and S.cache_hash == "gpu":
            return _cache_ref(v, d)
        ref = dict(d)
        _host_copy(v, ref, write)
        return ref
    if isinstance(v, (list, tuple)):
        return [_ref(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _ref(x) for k, x in v.items()}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return {"repr": repr(v)[:200]}


def _event(kind: str, ins: dict, outs: dict) -> int:
    seq = len(S.events)
    S.events.append({"seq": seq, "piece": S.piece, "layer": S.layer, "parent": S.parents[-1] if S.parents else None, "op": kind,
                     "in": ins, "out": outs})
    return seq


def _recorded(runner: Any, mode: str, forward) -> Any:
    """Runs ``forward()`` with recording on: under a `Recorder` for the whole forward unless the model is compiled (then
    `PiecewiseBackend` enters one per piece). Returns the hidden states, or None if the forward raised."""
    import contextlib
    import torch
    S.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _params(runner)
    S.t0 = time.time()
    cmode = runner.vllm_config.compilation_config.mode
    S.whole = getattr(cmode, "name", str(cmode)) != "VLLM_COMPILE"
    S.recording = True
    try:
        with Recorder() if S.whole else contextlib.nullcontext():
            out = forward()
        hidden = out[0] if isinstance(out, tuple) else out
        S.piece = -1
        _event("model_output", {}, {"hidden": _ref(hidden)})
        return out
    except Exception:  # noqa: BLE001
        S.errors.append(traceback.format_exc()[-4000:])
        return None
    finally:
        S.recording = False
        S.add("forward", time.time() - S.t0)


def _eager(runner: Any, model_inputs: dict | None, mode: str, batch_desc: Any) -> Any:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
    ib = S.input_batch

    def forward():
        if model_inputs is None:
            inputs = {"input_ids": ib.input_ids, "positions": ib.positions, "inputs_embeds": None, "intermediate_tensors": None,
                      **runner.model_state.prepare_inputs(ib, runner.req_states)}
            with set_forward_context(S.attn, runner.vllm_config, num_tokens=ib.num_tokens_after_padding,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                     batch_descriptor=BatchDescriptor(num_tokens=ib.num_tokens_after_padding,
                                                                      has_lora=runner.lora_config is not None,
                                                                      num_active_loras=getattr(batch_desc, "num_active_loras", 0)),
                                     slot_mapping=S.slot_maps, is_padding=getattr(ib, "is_padding", None)):
                return runner.model(**inputs)
        fc = get_forward_context()
        was = fc.cudagraph_runtime_mode
        fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
        try:
            return runner.model(**model_inputs)
        finally:
            fc.cudagraph_runtime_mode = was

    out = _recorded(runner, mode, forward)
    if out is None:
        return None
    return (out[0] if isinstance(out, tuple) else out).clone()


def _check(mine: Any, out: Any, mode: str, batch_desc: Any) -> None:
    import torch
    got = out[0] if isinstance(out, tuple) else out
    ib = S.input_batch
    n = int(getattr(ib, "num_tokens", got.shape[0] if got is not None else 0))
    S.check = {"mode": mode, "tokens": n, "padded": int(getattr(ib, "num_tokens_after_padding", 0)),
               "recorded_forward": mine is not None}
    if mine is not None and isinstance(got, torch.Tensor):
        S.check.update(equal_tokens=bool(torch.equal(mine[:n], got[:n])), equal_padded=bool(torch.equal(mine, got)),
                       differ_words=int((mine[:n].view(torch.int16) != got[:n].view(torch.int16)).sum().item())
                       if mine.dtype == torch.bfloat16 else None)
        if not S.check["equal_tokens"] and mine.shape == got.shape:
            rows = (mine[:n] != got[:n]).reshape(n, -1).any(dim=1).nonzero().reshape(-1).tolist()
            S.check["differ_rows"] = [int(r) for r in rows]


def _host_memory() -> dict:
    """This process's resident and peak resident host memory (``/proc/self/status``), in bytes."""
    p = Path("/proc/self/status")
    if not p.is_file():
        return {}
    got = dict(line.split(":", 1) for line in p.read_text().splitlines() if ":" in line)
    return {k: int(got[k].split()[0]) << 10 for k in ("VmRSS", "VmHWM") if k in got}


def _drain() -> None:
    """Waits for every recorded tensor's hash (and its write or spool), raising the first job's error."""
    if S.hasher is not None:
        S.hasher.drain()


def _reset() -> None:
    """Clears a step's state, waiting for its hashing jobs first, so a failed step leaves nothing to the next."""
    try:
        _drain()
    except Exception as e:  # noqa: BLE001
        _note(f"rank {S.rank} step {S.step}: a hashing job failed ({e!r})")
    import shutil
    shutil.rmtree(S.root() / "spool", ignore_errors=True)
    S.events, S.errors, S.check = [], [], None
    S.tab, S.tables = None, []
    S.seen.clear()
    S.spooled.clear()
    S.dropped.clear()
    S.floor_hit = False
    S.used_params.clear()
    S.t, S.n = {}, {}
    if S.hasher is not None:
        S.hasher.peak = 0


def _batch_sha512(req_ids: list[str], scheduled: dict[str, int]) -> str:
    """SHA-512 over the step's batch, which the rank root commits: its request ids in batch order and each request's scheduled
    tokens (`stage.batch_sha512` recomputes it)."""
    doc = json.dumps({"req_ids": req_ids, "scheduled": scheduled}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha512(f"verity/spot/batch\0{doc}".encode()).hexdigest()


def _finish(runner: Any) -> None:
    t = time.time()
    _drain()
    S.add("hash_drain", time.time() - t)
    d = S.root() / f"step{S.step}"
    d.mkdir(parents=True, exist_ok=True)
    tables = None
    if S.table:
        t = time.time()
        tables = _write_tables(d)
        S.add("tables", time.time() - t)
    t = time.time()
    lines = "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in S.events).encode()
    (d / "ops.jsonl").write_bytes(lines)
    S.add("write_ops", time.time() - t, len(lines))
    sched = S.sched
    reqs = {str(k): int(v) for k, v in getattr(sched, "num_scheduled_tokens", {}).items()}
    t = time.time()
    params = _param_digests(S.used_params, S.digests)
    S.add("params", time.time() - t)
    (d / "params.json").write_text(json.dumps(params, indent=0, sort_keys=True))
    if S.allreduce_tree is not None:
        (d / TREE_FILE).write_text(json.dumps(S.allreduce_tree, sort_keys=True))
    ops_sha, params_sha = hashlib.sha512(lines).hexdigest(), hashlib.sha512((d / "params.json").read_bytes()).hexdigest()
    ids = [str(x) for x in getattr(S.input_batch, "req_ids", None) or []]
    batch_sha = _batch_sha512(ids, reqs)
    meta = {"rank": S.rank, "step": S.step, "check": S.check, "scheduled": reqs, "req_ids": ids, "events": len(S.events),
            "blob_bytes": S.bytes, "errors": S.errors, "seconds": round(time.time() - S.t0, 1),
            "keep": S.step_keep, "cache_hash": "device" if S.dev else "host",
            "time_s": {k: round(v, 3) for k, v in sorted(S.t.items())}, "bytes": dict(sorted(S.n.items())),
            "host": {**_host_memory(), "copies_peak": S.hasher.peak if S.hasher else 0},
            "ops_sha512": ops_sha, "params_sha512": params_sha, "batch_sha512": batch_sha,
            "rank_root": hashlib.sha512(f"verity/spot/rank-root\0{ops_sha}\0{params_sha}\0{batch_sha}".encode()).hexdigest()}
    if tables is not None:
        meta["table"] = tables
    (d / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    commit = _commit_and_open(runner, d, meta["rank_root"])
    print(f"[spot-capture] rank {S.rank} step {S.step}: {len(S.events)} events, {meta['seconds']} s, check {meta['check']}, "
          f"errors {len(S.errors)}, root {commit.get('root', '')[:16]}, opened layers {commit.get('layers')}", flush=True)
    _reset()


def _commit_and_open(runner: Any, d: Path, rank_root: str) -> dict:
    """The step's interior root over every rank's, then the coins, flipped by rank 0 only after every rank's root is fixed, and
    the layers they open: ``--open`` (default ``dense:1,moe:1``) layers of each kind, whose parameters each rank
    writes under ``open/`` (the bytes its sampled proof blocks read). Written to ``commit.json``."""
    import torch.distributed as dist
    group = None
    try:
        from vllm.distributed.parallel_state import get_world_group
        group = get_world_group().cpu_group
    except Exception as e:  # noqa: BLE001
        _note(f"no vLLM world group ({e!r}): the roots and coins go over torch.distributed's default group")
        group = None
    world = dist.get_world_size() if dist.is_initialized() else 1
    roots = [None] * world
    if world > 1:
        dist.all_gather_object(roots, rank_root, group=group)
    else:
        roots = [rank_root]
    root = hashlib.sha512(("verity/spot/step-root\0" + "\0".join(roots)).encode()).hexdigest()
    t_root = time.time()
    coins = [os.urandom(64).hex() if S.rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(coins, src=dist.get_global_rank(group, 0) if group is not None else 0, group=group)
    coins_hex = coins[0]
    hf = getattr(runner.model_config, "hf_text_config", None) or runner.model_config.hf_config
    n_layers = int(getattr(hf, "num_hidden_layers"))
    dense_k = min(int(getattr(hf, "first_k_dense_replace", 0) or 0), n_layers)
    pools = {"dense": list(range(dense_k)), "moe": list(range(dense_k, n_layers))}
    want = S.open
    layers: dict[str, list[int]] = {}
    for kind, n in want.items():
        pool = list(pools.get(kind, []))
        layers[kind] = []
        for i in range(min(int(n), len(pool))):
            j = int.from_bytes(hashlib.sha512(bytes.fromhex(coins_hex) + f"open/{kind}/{i}".encode()).digest(), "big") % len(pool)
            layers[kind].append(pool.pop(j))
    t_open = time.time()
    opened = sorted(x for v in layers.values() for x in v)
    marks = tuple(f"layers.{x}." for x in opened)
    kept = {h for e in S.events if e.get("layer") in opened for h in _blobs([e.get("in"), e.get("out")])}
    spool = S.root() / "spool"
    kept_bytes, missing = 0, []
    for h in sorted(kept):
        if h in S.spooled:
            p = S.root() / "blobs" / h[:2] / h
            p.parent.mkdir(parents=True, exist_ok=True)
            os.replace(spool / h, p)
            kept_bytes += S.spooled[h]
        elif h in S.dropped:
            missing.append(S.dropped[h])
    import shutil
    shutil.rmtree(spool, ignore_errors=True)
    withheld = [n for h, n in {**S.spooled, **S.dropped}.items() if h not in kept]
    od = d / "open"
    files, small = {}, {}
    if S.step_keep != "none":
        od.mkdir(exist_ok=True)
        names = [name for name in sorted(S.param_of) if any(m in f".{name}" for m in marks)]
        if S.keep_all:
            names += [n for n in sorted(S.used_params - set(names))
                      if S.param_of[n].numel() * S.param_of[n].element_size() <= SMALL_PARAM]
        digests = _param_digests(names, S.digests)
        for name in names:
            t = S.param_of[name]
            tw = time.time()
            raw = _flat_bytes(t).cpu().numpy()
            (od / f"{name}.bin").write_bytes(memoryview(raw))
            S.add("write_params", time.time() - tw, raw.nbytes)
            (files if any(m in f".{name}" for m in marks) else small)[name] = {**digests[name], "stride": list(t.stride())}
    S.add("open", time.time() - t_open)
    mode = S.step_keep
    commit = {"rank": S.rank, "step": S.step, "rank_roots": roots, "root": root, "t_root": t_root, "coins": coins_hex,
              "t_coins_received": time.time(), "coins_by": "rank 0's os.urandom(64), drawn after every rank's root was gathered",
              "layers": layers, "opened": files, "kept_params": small,
              "kept": ("nothing: --keep none commits the step only" if mode == "none" else
                       "every recorded tensor" if mode == "all" and not withheld and not missing else
                       f"every recorded tensor but {len(withheld)} outside the opened layers, {sum(withheld)} bytes "
                       f"({'past the disk floor' if S.floor_hit else 'each over the keep cap'})" if mode == "all" else
                       f"the opened layers' {len(kept) - len(missing)} tensors, {kept_bytes} bytes"),
              "missing": {"tensors": len(missing), "bytes": sum(missing)},
              # every layer's calls can be gated: a withheld tensor is one over the keep cap, never one lost to the floor
              "whole": mode == "all" and not S.floor_hit, "keep_max": S.keep_max if mode == "all" else None,
              "time_s": {"open": round(time.time() - t_open, 3), "write_params": round(S.t.get("write_params", 0.0), 3)}}
    (d / "commit.json").write_text(json.dumps(commit, indent=1, sort_keys=True))
    return commit


def site(d: Path, cfg: dict) -> Path:
    """A directory that, on PYTHONPATH, registers this file as the ``vllm.general_plugins`` entry ``verity_spot_capture``,
    with ``cfg`` (`main`'s settings) beside it as `CONFIG`, and ``sha512.cuh`` beside it for `CHUNKS_CU`. With ``table``
    layers, ``d`` and the capture directory are made private (`private_dir`): the tables, and the steps in `CONFIG`, are."""
    import shutil
    if cfg.get("table"):
        private_dir(Path(cfg["capture"]), "the --table capture directory")
        private_dir(d, "the --table site directory")
    info = d / "verity_spot_capture-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-spot-capture\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_spot_capture = spot_capture:register\n")
    shutil.copyfile(__file__, d / "spot_capture.py")
    here = Path(__file__).resolve()
    hdr = next((p for p in (here.with_name("sha512.cuh"), here.parents[2] / "verity/ml/kernels/sha512.cuh") if p.is_file()), None)
    if hdr is not None:
        shutil.copyfile(hdr, d / "sha512.cuh")
    (d / CONFIG).write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return d


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="capture.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("site", help="write the plugin's site directory, its settings beside it")
    s.add_argument("dir", type=Path)
    s.add_argument("--capture", required=True, help="where each rank's capture goes")
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--steps", help="the 0-based steps to record, comma-separated")
    which.add_argument("--steps-from", type=Path, metavar="FILE",
                       help="the steps to record from a private file, `table.py steps`' output, off the command line")
    s.add_argument("--open", default="dense:1,moe:1", help="how many layers of each kind the coins open, KIND:N,...")
    s.add_argument("--mla-tokens", type=int, default=8, help="how many of a sparse MLA call's tokens have their gathered KV rows recorded (FA3: those with the most keys)")
    s.add_argument("--keep", choices=("none", "opened", "all"), default="opened",
                   help="all: write every recorded tensor; opened: the opened layers'; none: no tensor, the commitments only")
    s.add_argument("--keep-steps", help="the steps --keep applies to (default: every step); the others keep one level less "
                   "(all, then opened, then none)")
    s.add_argument("--keep-max-mib", type=int, default=64,
                   help="with --keep all, a larger tensor is written only in an opened layer (hashed into the root either way)")
    s.add_argument("--min-free-gib", type=int, default=40,
                   help="a step stops writing and spooling tensors once its disk would fall below this")
    s.add_argument("--hash-threads", type=int, default=8, help="host threads hashing each rank's recorded tensors (0: inline)")
    s.add_argument("--host-mib", type=int, default=8192,
                   help="the most of a rank's recorded tensors' host copies held at once, awaiting their hashes")
    s.add_argument("--cache-hash", choices=("gpu", "host"), default="gpu",
                   help="gpu: a whole paged cache is named by chunks_sha512 on its device; host: by SHA-512 on the host")
    s.add_argument("--gpu-check", type=int, default=4,
                   help="how many whole caches each rank hashes on the host too, held equal to the device's")
    s.add_argument("--sampler", action="store_true", help="also record rank 0's sampler at each step (outside the step's root)")
    s.add_argument("--allreduce-tree", metavar="TREE",
                   help=f"the node's multimem reduction tree as its probe read it, written at each step as {TREE_FILE}")
    s.add_argument("--allreduce-tree-source", default="", help="where TREE was read (the probe's record)")
    s.add_argument("--table", default="", metavar="LAYER[,LAYER]",
                   help="layers with their own indexer whose long rows' history tables each step writes (outside the step's "
                        "root): each request's last token's latent rows, scores, canonical top-k and the rows attention read")
    s.add_argument("--table-keys", type=int, default=2048,
                   help="a row is long past this many keys (the indexer's top-k)")
    a = ap.parse_args(argv)
    steps = ({int(x) for x in a.steps.split(",") if x.strip()} if a.steps is not None
             else {int(e["step"]) for e in json.loads(a.steps_from.read_text())["steps"]})
    cfg = {"capture": str(Path(a.capture).resolve()), "steps": sorted(steps),
           "open": {k: int(n) for k, n in (x.split(":") for x in a.open.split(",") if x)}, "mla_tokens": a.mla_tokens,
           "keep": a.keep, "keep_steps": None if a.keep_steps is None else sorted({int(x) for x in a.keep_steps.split(",") if x.strip()}),
           "keep_max_mib": a.keep_max_mib, "min_free_gib": a.min_free_gib, "sampler": a.sampler,
           "hash_threads": a.hash_threads, "host_mib": a.host_mib, "cache_hash": a.cache_hash, "gpu_check": a.gpu_check,
           "allreduce_tree": {"tree": reduction_tree(a.allreduce_tree), "source": a.allreduce_tree_source}
           if a.allreduce_tree else None,
           "table": sorted({int(x) for x in a.table.split(",") if x.strip()}), "table_keys": a.table_keys}
    print(site(a.dir, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
