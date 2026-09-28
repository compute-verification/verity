"""Instantiate the ACTUAL pinned vLLM model class on the `meta` device through vLLM's own config + model registry, and
wrap one engine step (forward -> compute_logits -> greedy argmax) as an ordinary module `torch.export` can trace.

Everything here is model-agnostic structural scaffolding (counted in the REPORT): no layer order, no op sequence, no
family flags. The architecture comes from whatever `vllm_config.model_config.architectures` resolves to.

Environment knobs of the pinned vLLM that the construction sets (declared, not hidden; `engine.env.EXPORT`):
  VLLM_BATCH_INVARIANT=1     the admitted profile (selects the BI code paths: fused_add_rms_norm / Triton rms_norm / BI GEMM)
  VLLM_USE_LAYERNAME=0       layer names as plain `str` op arguments (torch>=2.11 default wraps them in an opaque `LayerName`
                             object, which export would lift as a custom object); the vLLM source documents this switch.
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=1  a shared expert runs on the main stream: `SharedExperts.__init__` would otherwise make a CUDA
    aux stream (no driver on a GPU-free Build host); the stream schedules the same kernels on the same values.
  HF_HUB_OFFLINE=1           the model config is the seeded snapshot; no network.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any

from verity_vllm.engine import env as engine_env

engine_env.pin(engine_env.EXPORT)

import torch  # noqa: E402
from torch import nn  # noqa: E402

PIN = {"model": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "dtype": "bfloat16"}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_vllm_config(model: str = PIN["model"], revision: str = PIN["revision"], dtype: str = PIN["dtype"], max_model_len: int = 2048,
                     attention_backend: str | None = None, tensor_parallel_size: int = 1, **kw) -> Any:
    """vLLM's own `EngineArgs.create_engine_config()` for the eager, bf16 profile at `tensor_parallel_size` (no engine, no GPU allocation).
    `attention_backend` is the TargetProfile's knob (`EngineArgs(attention_backend=AttentionBackendEnum[name])`; None = vLLM's
    selector) — the ONLY attention-backend input this vLLM pin reads (env `VLLM_ATTENTION_BACKEND` is not defined in its `envs`).
    [R13 tp] `tensor_parallel_size` is the row's TP structural input: the parallel layers shard from `parallel_config` + the TP group."""
    from vllm.engine.arg_utils import EngineArgs
    if attention_backend is not None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        kw["attention_backend"] = AttentionBackendEnum[attention_backend]
    if tensor_parallel_size > 1:
        kw.setdefault("disable_custom_all_reduce", True)   # the admitted TP path on the pod (pynccl only); recorded as an engine arg
    args = EngineArgs(model=model, revision=revision, tokenizer_revision=revision, dtype=dtype, enforce_eager=True, tensor_parallel_size=tensor_parallel_size,
                      max_model_len=max_model_len, gpu_memory_utilization=0.2, skip_tokenizer_init=True, **kw)
    return args.create_engine_config()


class WithPeers(nn.Module):
    """[R13 tp] A rank's wrapper with its PEER INPUTS: `forward(prompt, *peers)` hands the peers (the other rank's collective partials,
    in collective call order) to the rank's `CollectiveBus` and runs the inner wrapper; the k-th collective of this forward consumes
    peer k (see `tp_ops.CollectiveBus`).  A stochastic inner wrapper's own inputs come before the peers, in its order
    (`SeededWithPeers`, `SplitsWithPeers`; `with_peers` picks the form): the Program's parameters are the TP1 Program's, then the peers."""

    def __init__(self, inner: nn.Module, bus) -> None:
        super().__init__()
        self.inner, self.bus = inner, bus

    @property
    def DERIVE_SOURCE(self) -> str:
        return f"verity-vllm/with-peers/v1[{self.inner.DERIVE_SOURCE}]"

    def forward(self, prompt: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt,), peers)

    def _run(self, own: tuple, peers: tuple):
        self.bus.reset("trace", list(peers))
        out = self.inner(*own)
        if self.bus.cursor * (self.bus.world - 1) != len(peers):    # [lane vllm-tp-n] world-1 peer inputs per collective
            raise RuntimeError(f"rank {self.bus.rank}: {self.bus.cursor} collectives consumed x {self.bus.world - 1} peers, {len(peers)} peer inputs given")
        return out


class SeededWithPeers(WithPeers):
    """`WithPeers` over a stochastic wrapper whose split count is the single request's constant: `forward(prompt, seed, *peers)`."""

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt, seed), peers)


class SplitsWithPeers(WithPeers):
    """`WithPeers` over a stochastic wrapper with a `splits` input: `forward(prompt, seed, splits, *peers)`."""

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, splits: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt, seed, splits), peers)


def with_peers(inner: nn.Module, inputs: tuple, bus) -> tuple[nn.Module, tuple]:
    """A tensor-parallel rank's traced wrapper over `inner` (a `make_wrapper` module) and its example inputs: `inner`'s own inputs (the
    prompt, and a stochastic wrapper's `seed` and `splits`), then the bus's example peers in collective order."""
    forms = {1: WithPeers, 2: SeededWithPeers, 3: SplitsWithPeers}
    if len(inputs) not in forms:
        raise ValueError(f"with_peers: a wrapper takes 1 to 3 inputs of its own (prompt, seed, splits), {type(inner).__name__} takes {len(inputs)}")
    return forms[len(inputs)](inner, bus), tuple(inputs) + tuple(bus.example_peers())


def probe_collectives(inner: nn.Module, inputs: tuple, vllm_config, num_tokens: int, bus, profile=None) -> list[dict]:
    """[R13 tp] One probe export of the rank's wrapper with the bus in `probe` mode (graph discarded): records every collective's kind,
    shape, dtype and call order — the peer-input signature of the Program.  Same mechanism and shims as the real export (the fused CUDA
    ops have fake impls under export's FakeTensorMode, not Meta kernels for a plain meta forward)."""
    from .export_compat import export_compat
    from .triton_capture import intercept_triton_launches
    bus.reset("probe")
    with torch.no_grad(), fake_forward_context(vllm_config, num_tokens), torch.device("meta"), export_compat(profile), intercept_triton_launches():
        torch.export.export(inner, tuple(inputs), strict=False)
    return list(bus.record)


_DIST_READY = False
#: `_free_port` closes its socket before the store binds the port, so concurrent derives (a Build runs world x shapes of them) can be
#: handed the same port: the loser's bind fails EADDRINUSE before any process group exists, and a fresh port is a clean retry.
_PORT_TRIES = 8


def ensure_distributed(vllm_config, backend: str = "gloo") -> str:
    """vLLM's TP/PP groups for world size 1 (the model classes call `get_tensor_model_parallel_world_size()` / `get_pp_group()`)."""
    global _DIST_READY
    if _DIST_READY:
        return "already"
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    with set_current_vllm_config(vllm_config):
        for attempt in range(_PORT_TRIES):
            try:
                init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{_free_port()}", local_rank=0, backend=backend)
                break
            except Exception as e:  # noqa: BLE001 -- torch's DistNetworkError; only the port race is retried
                if "EADDRINUSE" not in str(e) or attempt == _PORT_TRIES - 1:
                    raise
        ensure_model_parallel_initialized(1, 1)
    _DIST_READY = True
    return backend


def instantiate_meta(vllm_config) -> nn.Module:
    """The registered model class for `vllm_config.model_config` (vLLM's registry), constructed on `meta` (shapes only)."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.utils.torch_utils import set_default_torch_dtype
    model_cls, arch = get_model_architecture(vllm_config.model_config)
    # worker parity (vllm/v1/worker/worker_base.py: "IR op priority and torch-wrap state are constant for the worker's lifetime"):
    # without these, vLLM's `ir.ops.*` (e.g. GemmaRMSNorm's unfused norm) stay opaque `vllm_ir.*` torch ops instead of running
    # their Python implementation as the engine does. Generic scaffolding, no model fact.
    try:
        import vllm.ir
        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(vllm_config.compilation_config.ir_enable_torch_wrap)
    except Exception:  # noqa: BLE001
        pass
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype), torch.device("meta"):
        model = model_cls(vllm_config=vllm_config)
    model.eval()
    model._verity_arch = arch  # annotation only
    return model


def retie_lm_head_as_the_loader_does(model: nn.Module, vllm_config) -> dict:
    """[R15 fp8] Loader parity for a checkpoint whose config says `tie_word_embeddings` but which SHIPS an `lm_head.weight` of its own
    (quantisation tooling keeps a redundant copy: Qwen3-4B-Instruct-2507-FP8).  vLLM's `ModelConfig.maybe_untie_word_embeddings`
    then builds the head untied (`hf_config.tie_word_embeddings = False`, `word_embeddings_untied_by_checkpoint = True`), loads both
    tensors, and `maybe_retie_word_embeddings` (model_loader/utils.py, `process_weights_after_loading`) re-ties them -- ONE shared
    Parameter -- iff `torch.equal(lm_head.weight, embed_tokens.weight)`.  A meta instantiation never loads, so without this the
    derived Program carries two Inputs (embed, lm_head) where the served model has one root: a structural mismatch at Match.

    The decision is reproduced exactly as the loader makes it, from the pinned checkpoint's bytes (a checkpoint fact, read like its
    config.json -- never assumed): the two tensors are read from the snapshot's safetensors on the CPU and compared with `torch.equal`;
    equal -> vLLM's own `_UntiedLMHead.tie` on the meta model (the head's `weight` becomes the embedding's Parameter, which
    `derive._weights_of` collapses to one leaf with an alias).  Not equal -> left untied, as the loader leaves it (and the checkpoint
    is then genuinely untied: two fields).  Returns the fact as consulted for the report; `{"applies": False}` when vLLM did not untie."""
    fact: dict[str, Any] = {"applies": False}
    mc = vllm_config.model_config
    if not bool(getattr(mc, "word_embeddings_untied_by_checkpoint", False)):
        return fact
    from vllm.model_executor.model_loader.weight_tying import _get_untied_lm_head
    untied = _get_untied_lm_head(model)
    fact.update(applies=True, config_tie_word_embeddings=True, untied_for_loading=True, head=None if untied is None else untied.name)
    if untied is None:
        fact["note"] = "no single unquantised (lm_head, embed_tokens) pair of one shape: the loader leaves the model untied"
        return fact
    import glob
    import json as _json
    import struct
    from safetensors import safe_open
    root = mc.model
    if not os.path.isdir(root):
        raise RuntimeError(f"retie_lm_head: model path {root!r} is not a local snapshot directory (HF_HUB_OFFLINE resolution failed)")
    names = {"embed": None, "lm_head": None}
    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = set(_json.loads(fh.read(n)))
        for tag, nm in (("embed", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
            if nm in keys:
                names[tag] = (f, nm)
    if names["embed"] is None or names["lm_head"] is None:
        raise RuntimeError(f"retie_lm_head: checkpoint {root!r} lacks one of model.embed_tokens.weight / lm_head.weight in {files}")
    tensors = {}
    for tag, (f, nm) in names.items():
        with safe_open(f, framework="pt", device="cpu") as sf:
            tensors[tag] = sf.get_tensor(nm)
    equal = tensors["embed"].shape == tensors["lm_head"].shape and tensors["embed"].dtype == tensors["lm_head"].dtype \
        and bool(torch.equal(tensors["embed"], tensors["lm_head"]))
    fact.update(checkpoint_files={k: os.path.basename(v[0]) for k, v in names.items()},
                shapes={k: list(t.shape) for k, t in tensors.items()}, dtypes={k: str(t.dtype) for k, t in tensors.items()},
                lm_head_equals_embed=equal, retied=equal,
                rule="vllm.model_executor.model_loader.weight_tying.maybe_retie_word_embeddings: torch.equal -> tie_weights")
    del tensors
    if equal:
        untied.tie(model)
        fact["shared_parameter"] = bool(untied.lm_head.weight is untied.embed_tokens.weight)
    return fact


@contextmanager
def fake_forward_context(vllm_config, num_tokens: int):
    """vLLM's `set_forward_context` with per-layer attention metadata = None: the attention op is opaque to export (its
    body never runs), so only the *presence* of a context is structural. Nothing here reads shapes of a KV cache."""
    from vllm.forward_context import set_forward_context
    static = vllm_config.compilation_config.static_forward_context
    attn_metadata = {name: None for name in static}
    with set_forward_context(attn_metadata, vllm_config, num_tokens=num_tokens):
        yield


class EngineStep(nn.Module):
    """One engine step for a single request at positions pos0..pos0+M-1: token ids -> model forward -> logits of the
    last row (vLLM's LogitsProcessor gathers the sampled row) -> greedy argmax (int32 token id).

    Generic: takes any vLLM `*ForCausalLM` (forward(input_ids, positions), compute_logits(hidden)). The row policy
    (`logits of the last row only`) is vLLM's `logits_indices` for one request; it is structure, not architecture."""

    DERIVE_SOURCE = "verity-vllm/engine-step/v1"

    def __init__(self, model: nn.Module, pos0: int, M: int, sampled_rows: str = "last") -> None:
        super().__init__()
        self.model = model
        self.pos0, self.M, self.sampled_rows = pos0, M, sampled_rows

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.pos0, self.pos0 + self.M, dtype=torch.int64)
        hidden = self.model(input_ids, positions)
        if self.sampled_rows == "last":
            hidden = hidden[-1:]
        logits = self.model.compute_logits(hidden)
        return torch.argmax(logits, dim=-1).to(torch.int32)


class GreedyRequest(nn.Module):
    """One whole request as one module (the flat, post-hoc shape): prefill of `LP` tokens (positions 0..LP-1; vLLM's model
    forward over every row, LM head on the sampled/last row only), then exactly `T` decode steps at positions LP..LP+T-1,
    each consuming the previous step's greedy sample. Returns the T+1 sampled ids (int32). The KV cache is implicit vLLM
    state: the frontend's per-layer history (UnifiedAttentionWithOutput[vllm]) accumulates across the steps of this graph.

    Generic: any vLLM `*ForCausalLM`; `LP`, `T` and greedy sampling are the request's structural inputs (BRIEF: prompt
    length, max_tokens, sampling kind)."""

    DERIVE_SOURCE = "verity-vllm/greedy-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int) -> None:
        super().__init__()
        self.model, self.LP, self.T = model, LP, T

    def _step(self, ids: torch.Tensor, pos0: int) -> torch.Tensor:
        positions = torch.arange(pos0, pos0 + ids.shape[0], dtype=torch.int64)
        hidden = self.model(ids, positions)
        logits = self.model.compute_logits(hidden[-1:])
        return torch.argmax(logits, dim=-1).to(torch.int32)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        tok = self._step(prompt, 0)
        toks = [tok]
        for s in range(self.T):
            tok = self._step(tok, self.LP + s)
            toks.append(tok)
        return torch.cat(toks)


class PaddedGreedyRequest(nn.Module):
    """The FIXED padded request rectangle `prefill | decode 1 .. decode D_MAX` (BRIEF "Padding"; serve3-spec §3–§8) over the
    actual vLLM model: liveness is a value — `live_0 = 1`, `live_{g+1} = live_g & ~(y_g == EOS)` on the raw sample; the
    emitted slot is `live ? y : PAD(0)`; the next step consumes the muxed slot; after every decode step the engine's
    step-boundary cache clear `vllm_iface.unified_cache_pad(live_in, zero, slot)` makes every layer's appended K/V row
    `live_in ? row : 0` against ONE zero per step. Nothing inside the step body is guarded (§6). Returns (slots, live flags),
    D_MAX+1 each.

    The cache clear is the one padding primitive this wrapper needs that vLLM does not expose: vLLM's KV cache is implicit
    engine state (written by `unified_kv_cache_update` inside the attention layer), so the wrapper cannot mux the rows the way
    `PaddedGreedyServe` does with its own `unified_cache_write`; `unified_cache_pad` addresses the state by slot for every
    layer at once — model-agnostic (no layer names, no counts)."""

    DERIVE_SOURCE = "verity-vllm/padded-greedy-request/v1"

    def __init__(self, model: nn.Module, LP: int, D_MAX: int, eos_id: int) -> None:
        super().__init__()
        self.model, self.LP, self.D_MAX, self.eos_id = model, LP, D_MAX, eos_id

    def _logits(self, ids: torch.Tensor, pos0: int) -> torch.Tensor:
        positions = torch.arange(pos0, pos0 + ids.shape[0], dtype=torch.int64)
        hidden = self.model(ids, positions)
        return self.model.compute_logits(hidden[-1:])

    def _event(self, logits: torch.Tensor, live: torch.Tensor):
        y = torch.argmax(logits, dim=-1).to(torch.int32)  # the raw sample, [1]
        tok = torch.where(live, y, 0)  # PAD_TOKEN_ID = 0, keyed on the event's liveness
        live = live & ~(y == self.eos_id)  # absorbing
        return tok, live

    def forward(self, prompt: torch.Tensor):
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        live = torch.ones(1, dtype=torch.bool)  # live_0: a request always samples y_0
        tok, live = self._event(self._logits(prompt, 0), live)
        toks, lives = [tok], [live]
        for s in range(1, self.D_MAX + 1):  # decode step s at position LP+s-1
            slot = self.LP + s - 1
            live_in = live
            zero = torch.zeros((), dtype=torch.bfloat16)  # the step's one zero (broadcast by every row mux)
            logits = self._logits(tok, slot)
            tok, live = self._event(logits, live_in)
            iface.unified_cache_pad(live_in, zero, slot)  # every layer's row `slot` := live_in ? row : 0 (step boundary)
            toks.append(tok)
            lives.append(live)
        return torch.cat(toks), torch.cat(lives)


class StochasticEngineStep(nn.Module):
    """[R15 stoch] `EngineStep` with the row's STOCHASTIC policy: the last row's logits -> `vllm_iface.gumbel_topp_token_select(logits,
    seed, temperature, top_p, pos, splits)` (vLLM d9105ea80 `Sampler.sample` with temperature + top-p and an explicit per-request seed:
    f32 widening -> `_temperature_kernel` -> `_apply_topp_split` -> `_gumbel_sample_kernel` + argmax).  `seed` (int64 [1]) is a forward
    INPUT (the request's `SamplingParams.seed`: a runtime value of the Program); temperature / top_p are the declared policy (structural
    inputs, constants of the Program); `pos` = the sampled logits row's position (pos0 + M - 1).  [R16 stoch, coord M-0169] `splits`
    (int32 [1]) is a forward INPUT too: the event's split count S = SplitsFor_v1(|live|, num_SMs), a DERIVED value of the workload.
    In a single-request workload `S` is the constant SplitsFor_v1(1, num_SMs) and the forward takes no `splits`."""

    DERIVE_SOURCE = "verity-vllm/stochastic-engine-step/v1"

    def __init__(self, model: nn.Module, pos0: int, M: int, temperature: float, top_p: float, S: int | None = None) -> None:
        super().__init__()
        self.model = model
        self.pos0, self.M = pos0, M
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.S = S

    def forward(self, input_ids: torch.Tensor, seed: torch.Tensor, splits: torch.Tensor | None = None) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        positions = torch.arange(self.pos0, self.pos0 + self.M, dtype=torch.int64)
        hidden = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden[-1:])
        return iface.gumbel_topp_token_select(logits, seed, self.temperature, self.top_p, self.pos0 + self.M - 1,
                                              None if self.S is not None else splits[0:1], S=self.S)


class StochasticRequest(nn.Module):
    """[R15 stoch] `GreedyRequest` with the row's STOCHASTIC policy (temperature + top-p, explicit seed): prefill of `LP` tokens, then
    exactly `T` decode steps, every sampled token `gumbel_topp_token_select(logits, seed, temperature, top_p, pos, splits[s])` with `pos`
    the position of the logits row (LP-1 for the prefill sample, LP-1+k for the k-th sample: vLLM's `positions[logits_indices]`, the
    stream offset of `tl.randint(seed, pos)`).  `seed` is a forward INPUT (one per request; B2/B8 rows have one per component).

    [R16 stoch, coord M-0169 / global M-0189 (1)] `splits` (int32 [T+1]) is the request-owned DERIVED-S INPUT: entry s is the split
    count of sampling event s (the prefill sample = event 0), S_t = SplitsFor_v1(|live(t)|, num_SMs) at the engine step t the event runs
    in -- a value of the WORKLOAD (the live set), bound at Match to the split kernels' constexpr and checked against the relation there
    (GM-01 G6 'sampler geometry').  The Program carries no S schedule: the forward reads `splits[s:s+1]` per event (a static view of
    the Input).  Parameter order [prompt, seed, splits] so the workload Program's layout stays [prompts, seeds, splits, weights...].

    [single-request workloads] With `S` given (the workload's one request: S = SplitsFor_v1(1, num_SMs) at
    every event) the forward takes [prompt, seed] and every select's split count is that constant (`Const32[S]` in the Program)."""

    DERIVE_SOURCE = "verity-vllm/stochastic-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int, temperature: float, top_p: float, S: int | None = None) -> None:
        super().__init__()
        self.model, self.LP, self.T = model, LP, T
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.S = S

    def _step(self, ids: torch.Tensor, pos0: int, seed: torch.Tensor, splits: torch.Tensor | None) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        positions = torch.arange(pos0, pos0 + ids.shape[0], dtype=torch.int64)
        hidden = self.model(ids, positions)
        logits = self.model.compute_logits(hidden[-1:])
        return iface.gumbel_topp_token_select(logits, seed, self.temperature, self.top_p, pos0 + ids.shape[0] - 1, splits, S=self.S)

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, splits: torch.Tensor | None = None) -> torch.Tensor:
        entry = (lambda s: None) if self.S is not None else (lambda s: splits[s:s + 1])   # noqa: E731
        tok = self._step(prompt, 0, seed, entry(0))
        toks = [tok]
        for s in range(self.T):
            tok = self._step(tok, self.LP + s, seed, entry(s + 1))
            toks.append(tok)
        return torch.cat(toks)


def single_request_splits(sampling: dict) -> int | None:
    """The single-request constant S of a stochastic policy: `sampling["splits"]` when it is an int, the
    policy's geometry is one request's (`sampler_batch == 1`) and the int is SplitsFor_v1(1, num_SMs) of that geometry; else None."""
    from verity_vllm.program.registry.topp_geometry import splits_for_v1
    S, g = sampling.get("splits"), sampling.get("geometry") or {}
    if isinstance(S, bool) or not isinstance(S, int) or g.get("sampler_batch") != 1 or g.get("num_sms") is None:
        return None
    return S if S == int(splits_for_v1(1, int(g["num_sms"]))) else None


def make_wrapper(kind: str, model: nn.Module, *, pos0: int = 8, M: int = 1, LP: int = 8, T: int = 4, eos_id: int = 0,
                 sampling: dict | None = None) -> tuple[nn.Module, tuple]:
    """(module, example inputs on meta) for `kind` in {step, request, padded}.  [X-05] `sampling` is the row's declared policy: the greedy
    wrappers author `torch.argmax`; [R15 stoch] `{"kind": "stochastic", "temperature", "top_p", "top_k": 0, "seeded": True, "splits": "derived"}`
    authors `vllm_iface.gumbel_topp_token_select` with the seed as a second forward input and [R16, coord M-0169] the derived split
    count `splits` (int32, one entry per sampling event: [1] for the step wrapper, [T+1] for the request wrapper) as a third (step /
    request wrappers; padded is greedy-only).  [single-request workloads] A numeric `splits` is the single-request constant S =
    SplitsFor_v1(1, num_SMs) (`build.sampler_geometry`, `geometry.sampler_batch == 1`): the wrapper takes no `splits` input.  Any other
    numeric `splits` (R15's static S / declared schedule) is REFUSED, as is any other policy (derive_step refuses earlier)."""
    kind_s = (sampling or {"kind": "greedy"}).get("kind")
    if kind_s == "stochastic":
        s = sampling or {}
        if int(s.get("top_k", 0) or 0) != 0 or not s.get("seeded", False):
            raise ValueError(f"make_wrapper authors the seeded temperature + top-p select only (top_k == 0, seeded): {sampling}")
        S = single_request_splits(s)
        if s.get("splits") != "derived" and S is None:
            raise ValueError(f"make_wrapper: sampling.splits must be 'derived' (S_t = SplitsFor_v1(|live(t)|, num_SMs), a request-owned Input of "
                             f"the Program; coord M-0169) or, for a single-request workload, the constant SplitsFor_v1(1, num_SMs) of its geometry "
                             f"-- any other static S or a declared schedule is not a Program the Match can hold, got {s.get('splits')!r}")
        temp, p = float(s["temperature"]), float(s.get("top_p", 1.0))
        seed_in = torch.empty((1,), dtype=torch.int64, device="meta")
        if kind == "step":
            ids = torch.empty((M,), dtype=torch.int32, device="meta")
            if S is not None:
                return StochasticEngineStep(model, pos0, M, temp, p, S=S), (ids, seed_in)
            return StochasticEngineStep(model, pos0, M, temp, p), (ids, seed_in, torch.empty((1,), dtype=torch.int32, device="meta"))
        if kind == "request":
            prompt = torch.empty((LP,), dtype=torch.int32, device="meta")
            if S is not None:
                return StochasticRequest(model, LP, T, temp, p, S=S), (prompt, seed_in)
            return StochasticRequest(model, LP, T, temp, p), (prompt, seed_in, torch.empty((T + 1,), dtype=torch.int32, device="meta"))
        raise ValueError(f"wrapper {kind!r} has no stochastic form (padded is greedy-only)")
    if kind_s != "greedy":
        raise ValueError(f"make_wrapper authors greedy token selection only; the row's sampling policy is {sampling}")
    if kind == "step":
        return EngineStep(model, pos0, M), (torch.empty((M,), dtype=torch.int32, device="meta"),)
    if kind == "request":
        return GreedyRequest(model, LP, T), (torch.empty((LP,), dtype=torch.int32, device="meta"),)
    if kind == "padded":
        return PaddedGreedyRequest(model, LP, T, eos_id), (torch.empty((LP,), dtype=torch.int32, device="meta"),)
    raise ValueError(kind)


#: environment variables this construction SETS (module top) — recorded as consulted only if the pinned vLLM's `envs` defines them
_ENV_SET_BY_BUILDER = ("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL", "TOKENIZERS_PARALLELISM", "VLLM_ATTENTION_BACKEND",
                       "VLLM_USE_DEEP_GEMM")   # [R15 fp8] the block-scaled FP8 kernel knob `TargetProfile.fp8_block_gemm` applies


def consulted_facts(vllm_config) -> dict:
    """Configuration facts the construction actually CONSULTED that the exported graph does not carry: the vLLM config's attention
    backend knob, KV block size / dtype, model dtype, eager mode, the platform class — plus, split honestly, the environment:
    `env_consulted` = variables the builder sets AND the pinned vLLM's `envs` module defines (so vLLM reads them);
    `env_ignored`   = variables set in the process that this vLLM pin does NOT define (e.g. `VLLM_ATTENTION_BACKEND`, dead in d9105ea80)
    — never reported as declared facts. The target facts themselves (capability, SM count, BI switch) live in `target.declared/applied`."""
    import os
    facts: dict = {}
    try:
        import vllm.envs as envs
        defined = set(getattr(envs, "environment_variables", {}) or {})
    except Exception:  # noqa: BLE001
        defined = set()
    facts["env_consulted"] = {k: os.environ.get(k) for k in _ENV_SET_BY_BUILDER if k in defined and k in os.environ}
    facts["env_ignored"] = {k: {"value": os.environ.get(k), "reason": "not defined in this vLLM pin's `vllm.envs`; never read"}
                            for k in _ENV_SET_BY_BUILDER if k not in defined and k in os.environ and k.startswith("VLLM_")}
    facts["env_set_not_vllm"] = {k: os.environ.get(k) for k in _ENV_SET_BY_BUILDER if k not in defined and k in os.environ and not k.startswith("VLLM_")}
    try:
        facts["model_dtype"] = str(vllm_config.model_config.dtype)
        facts["enforce_eager"] = bool(vllm_config.model_config.enforce_eager)
    except Exception:  # noqa: BLE001
        pass
    try:
        facts["kv_block_size"] = vllm_config.cache_config.block_size
        facts["kv_cache_dtype"] = str(vllm_config.cache_config.cache_dtype)
    except Exception:  # noqa: BLE001
        pass
    ac = getattr(vllm_config, "attention_config", None)
    if ac is not None:
        b = getattr(ac, "backend", None)
        facts["attention_backend_knob"] = None if b is None else getattr(b, "name", str(b))
    try:
        from vllm.platforms import current_platform
        facts["platform"] = current_platform.device_name if hasattr(current_platform, "device_name") else str(type(current_platform).__name__)
        facts["platform_class"] = type(current_platform).__qualname__
    except Exception:  # noqa: BLE001
        pass
    return facts


@contextmanager
def no_fx_stack_traces():
    """Export without FX stack traces: nothing reads `node.meta["stack_trace"]`, and recording one per node was a tenth of a derive."""
    import torch.fx.config as fx_config
    from verity_vllm.engine import hooks
    with hooks.Hooks() as h:
        h.patch(fx_config, "do_not_emit_stack_traces", True)
        yield


def export_module(mod: nn.Module, inputs: tuple, vllm_config, num_tokens: int, strict: bool = False, intercept_triton: bool = True, launch_log=None,
                  profile=None, applied_out: dict | None = None):
    """`torch.export.export(strict=False)` of a wrapper module on meta under vLLM's forward context + the export shims for the
    TargetProfile `profile` (default profile if None). `intercept_triton`: direct Python-level Triton launches become opaque
    `verity_cba::triton_launch` nodes (triton_capture.py). `applied_out`, if given, receives `read_back_applied()` taken INSIDE the
    export context right after the export — what vLLM saw while the graph was produced (the driver refuses if it differs from the profile)."""
    from .export_compat import export_compat, read_back_applied
    from .triton_capture import intercept_triton_launches
    with torch.no_grad(), fake_forward_context(vllm_config, num_tokens), torch.device("meta"), export_compat(profile, mod), no_fx_stack_traces():
        if intercept_triton:
            with intercept_triton_launches(launch_log):
                ep = torch.export.export(mod, inputs, strict=strict)
        else:
            ep = torch.export.export(mod, inputs, strict=strict)
        if applied_out is not None:
            applied_out.update(read_back_applied())
        return ep


def export_step(model: nn.Module, vllm_config, pos0: int, M: int, strict: bool = False, intercept_triton: bool = True, launch_log=None):
    """One engine step (see `EngineStep`)."""
    step, ids = make_wrapper("step", model, pos0=pos0, M=M)
    return export_module(step, ids, vllm_config, M, strict, intercept_triton, launch_log)


def op_histogram(ep) -> dict[str, int]:
    h: dict[str, int] = {}
    for n in ep.graph.nodes:
        if n.op == "call_function":
            k = str(n.target)
            h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


__all__ = ["PIN", "make_vllm_config", "ensure_distributed", "instantiate_meta", "fake_forward_context", "EngineStep", "GreedyRequest",
           "PaddedGreedyRequest", "make_wrapper", "export_module", "export_step", "op_histogram", "consulted_facts",
           "attention_impls", "export_facts", "BI_TRITON_KERNELS"]


def attention_impls(model: nn.Module) -> dict:
    """OBSERVED (not declared) attention implementation per `Attention` layer of the instantiated model: the selector's output
    (`layer.impl` class, `layer.backend`), the FA version, window / soft cap / alibi / sinks, kv dtype, and whether the impl runs
    in batch-invariant mode (single-split combine). Keyed by `layer_name` — what the exported `unified_attention_with_output`
    node carries — so the attention rule can select the Definition from the layer that will run, not from an env string."""
    from vllm.model_executor.layers.attention import Attention
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, Attention):
            continue
        impl = mod.impl
        backend = getattr(mod, "backend", None)
        obs = {"impl": f"{type(impl).__module__}.{type(impl).__qualname__}", "backend": getattr(backend, "name", str(backend)),
               "attn_type": getattr(getattr(impl, "attn_type", None), "value", str(getattr(impl, "attn_type", None))),
               "kv_cache_dtype": getattr(impl, "kv_cache_dtype", None), "layer_kv_cache_dtype": getattr(mod, "kv_cache_dtype", None),
               "num_heads": getattr(impl, "num_heads", None), "num_kv_heads": getattr(impl, "num_kv_heads", None), "head_size": getattr(impl, "head_size", None),
               "scale": getattr(impl, "scale", None)}
        for k in ("vllm_flash_attn_version", "sliding_window", "logits_soft_cap", "alibi_slopes", "sinks", "batch_invariant_enabled",
                  "dcp_world_size", "pcp_world_size", "num_queries_per_kv", "kv_sharing_target_layer_name"):
            v = getattr(impl, k, "<absent>")
            obs[k] = list(v) if isinstance(v, tuple) else (None if isinstance(v, torch.Tensor) and v.numel() == 0 else ("tensor" if isinstance(v, torch.Tensor) else v))
        out[getattr(mod, "layer_name", name)] = obs
    return out


BI_TRITON_KERNELS = ("vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent",
                     "vllm.model_executor.determinism.batch_invariant._rms_norm_kernel")


def export_facts(op_hist: dict, launch_kernels) -> dict:
    """Graph-level facts the rules select on, DERIVED from the export: the dispatch `_C.fused_add_rms_norm` was reached through.
    At the pinned vLLM, `RMSNorm.forward_cuda` reaches that op only via `rms_norm_batch_invariant` (its non-BI branch is
    `forward_native`, plain aten), and the same `init_batch_invariance()` installs the Triton overrides whose launches the graph
    shows; the aten override library being installed is the in-process trace of that call. Evidence recorded verbatim."""
    fused_ops = [k for k in op_hist if "fused_add_rms_norm" in k]
    bi_launches = sorted(k for k in set(launch_kernels) if k in BI_TRITON_KERNELS)
    lib_installed = None
    try:
        from vllm.model_executor.determinism import batch_invariant as bi
        lib_installed = bool(getattr(bi, "_batch_invariant_MODE", False))
    except Exception:  # noqa: BLE001
        pass
    switch = None
    try:
        import vllm.envs as envs
        switch = bool(envs.VLLM_BATCH_INVARIANT)  # the very predicate `RMSNorm.forward_cuda` branches on (read in this process)
    except Exception:  # noqa: BLE001
        pass
    native_norm = [k for k in op_hist if k.startswith("aten.rsqrt") or k.startswith("aten.pow")]  # forward_native's signature
    # the aten override library (`enable_batch_invariant_mode`) is installed by the engine worker, not by a meta instantiation:
    # recorded as evidence, not required — the graph itself shows the BI Python paths (Triton launches) that only that switch opens
    if fused_ops and bi_launches and not native_norm:
        dispatch = "batch_invariant"
    elif not fused_ops:
        dispatch = "absent"
    else:
        dispatch = "eager"
    return {"fused_add_rms_norm_dispatch": dispatch,
            "evidence": {"fused_add_rms_norm_ops": fused_ops, "bi_triton_kernels_launched": bi_launches, "override_library_installed": lib_installed,
                         "forward_cuda_switch_in_process": switch,
                         "forward_native_ops_present": native_norm,
                         "mean_dim_override": mean_dim_override_facts(),
                         "source": "vllm/model_executor/layers/layernorm.py RMSNorm.forward_cuda -> rms_norm_batch_invariant -> ops.fused_add_rms_norm (BI branch); "
                                   "forward_native otherwise; determinism/batch_invariant.py init_batch_invariance/enable_batch_invariant_mode"}}


def mean_dim_override_facts() -> dict | None:
    """[R15 dense] The installed vLLM's `aten::mean.dim` batch-invariant override, read in-process (never assumed): the Triton
    `mean_kernel`'s qualified name and source sha (`JITFunction.src`, the same text the eager capture hashes as `src_sha`), the
    `mean_dim` / `mean_batch_invariant` Python sources' shas (BLOCK_SIZE, the `(M, N, 1)` view, the per-dim reduction order), and
    whether `enable_batch_invariant_mode` registers `aten::mean.dim` at all (grep of its source).  `MeanDimBIRule` checks these against
    `MEAN_PINS`; None when vLLM is not importable (the rule then refuses)."""
    try:
        import hashlib
        import inspect
        from vllm.model_executor.determinism import batch_invariant as bi
        k = bi.mean_kernel
        fn = getattr(k, "fn", k)
        registers = "aten::mean.dim" in inspect.getsource(bi.enable_batch_invariant_mode)
        return {"kernel": f"{fn.__module__}.{fn.__qualname__}",
                "src_sha256": hashlib.sha256(str(getattr(k, "src", "")).encode("utf-8")).hexdigest(),
                "mean_dim_src_sha256": hashlib.sha256(inspect.getsource(bi.mean_dim).encode()).hexdigest(),
                "mean_batch_invariant_src_sha256": hashlib.sha256(inspect.getsource(bi.mean_batch_invariant).encode()).hexdigest(),
                "enable_batch_invariant_mode_registers_mean_dim": registers}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
