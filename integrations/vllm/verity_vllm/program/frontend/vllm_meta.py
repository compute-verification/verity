"""Instantiate the ACTUAL pinned vLLM model class on the `meta` device through vLLM's own config + model registry, and
wrap one engine step (forward -> compute_logits -> greedy argmax) as an ordinary module `torch.export` can trace.

Everything here is model-agnostic structural scaffolding (counted in the REPORT): no layer order, no op sequence, no
family flags. The architecture comes from whatever `vllm_config.model_config.architectures` resolves to.

Environment knobs of the pinned vLLM that the construction sets (declared, not hidden; `engine.env.EXPORT`; batch invariance
is not among them: `pipeline/build.derive_step` applies the target's `VLLM_BATCH_INVARIANT`, the row's execution label's, before
this module is imported):
  VLLM_USE_LAYERNAME=0       layer names as plain `str` op arguments (torch>=2.11 default wraps them in an opaque `LayerName`
                             object, which export would lift as a custom object); the vLLM source documents this switch.
  VLLM_DISABLE_SHARED_EXPERTS_STREAM=1  a shared expert runs on the main stream: `SharedExperts.__init__` would otherwise make a CUDA
    aux stream (no driver on a GPU-free Build host); the stream schedules the same kernels on the same values.
  HF_HUB_OFFLINE=1           the model config is the seeded snapshot; no network.
"""

from __future__ import annotations

import socket
import sys
from contextlib import contextmanager, nullcontext
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
    `tensor_parallel_size` is the row's TP structural input: the parallel layers shard from `parallel_config` + the TP group."""
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
    """A rank's wrapper with its PEER INPUTS: `forward(prompt, *peers)` hands the peers (the other rank's collective partials,
    in collective call order) to the rank's `CollectiveBus` and runs the inner wrapper; the k-th collective of this forward consumes
    peer k (see `export_ops.CollectiveBus`).  A stochastic inner wrapper's own inputs come before the peers, in its order
    (`SeededWithPeers`, `SplitsWithPeers`), and so does a `schedule` witness (`ScheduledWithPeers`); `with_peers` picks the form: the
    Program's parameters are the TP1 Program's, then the schedule, then the peers."""

    def __init__(self, inner: nn.Module, bus) -> None:
        super().__init__()
        self.inner, self.bus = inner, bus

    @property
    def DERIVE_SOURCE(self) -> str:
        return f"verity-vllm/with-peers/v1[{self.inner.DERIVE_SOURCE}]"

    def forward(self, prompt: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt,), peers)

    def _run(self, own: tuple, peers: tuple, schedule: torch.Tensor | None = None):
        self.bus.reset("trace", list(peers), schedule)
        out = self.inner(*own)
        if self.bus.cursor * (self.bus.world - 1) != len(peers):    # world-1 peer inputs per collective
            raise RuntimeError(f"rank {self.bus.rank}: {self.bus.cursor} collectives consumed x {self.bus.world - 1} peers, {len(peers)} peer inputs given")
        if self.bus.schedule_cursor != len(self.bus.schedule_record):
            raise RuntimeError(f"rank {self.bus.rank}: {self.bus.schedule_cursor} align calls read the schedule, the probe recorded "
                               f"{len(self.bus.schedule_record)}")
        return out


class SeededWithPeers(WithPeers):
    """`WithPeers` over a stochastic wrapper whose split count is the single request's constant: `forward(prompt, seed, *peers)`."""

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt, seed), peers)


class SplitsWithPeers(WithPeers):
    """`WithPeers` over a stochastic wrapper with a `splits` input: `forward(prompt, seed, splits, *peers)`."""

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, splits: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt, seed, splits), peers)


class ScheduledWithPeers(WithPeers):
    """`WithPeers` over a greedy wrapper whose forward reads the `schedule` witness (`CollectiveBus.schedule_key`, each
    `moe_align_block_size` call's key per pair): `forward(prompt, schedule, *peers)`."""

    @property
    def DERIVE_SOURCE(self) -> str:
        return f"verity-vllm/scheduled-with-peers/v1[{self.inner.DERIVE_SOURCE}]"

    def forward(self, prompt: torch.Tensor, schedule: torch.Tensor, *peers: torch.Tensor):
        return self._run((prompt,), peers, schedule)


def with_peers(inner: nn.Module, inputs: tuple, bus) -> tuple[nn.Module, tuple]:
    """A tensor-parallel rank's traced wrapper over `inner` (a `make_wrapper` module) and its example inputs: `inner`'s own inputs (the
    prompt, and a stochastic wrapper's `seed` and `splits`), the `schedule` when the probe recorded an align call that reads it, then
    the bus's example peers in collective order."""
    forms = {1: WithPeers, 2: SeededWithPeers, 3: SplitsWithPeers}
    if getattr(inner, "C", 0):
        raise ValueError(f"with_peers: a rank Program with a cached prefix (C={inner.C}) has no peer form yet (its prefix is the rank's own "
                         f"KV heads, an input before the peers)")
    if len(inputs) not in forms:
        raise ValueError(f"with_peers: a wrapper takes 1 to 3 inputs of its own (prompt, seed, splits), {type(inner).__name__} takes {len(inputs)}")
    if bus.schedule_record:
        if len(inputs) != 1:
            raise ValueError(f"with_peers: the schedule witness has a greedy wrapper's form only (prompt, schedule, *peers), not one "
                             f"with {len(inputs)} inputs of its own")
        return ScheduledWithPeers(inner, bus), tuple(inputs) + (bus.example_schedule(),) + tuple(bus.example_peers())
    return forms[len(inputs)](inner, bus), tuple(inputs) + tuple(bus.example_peers())


def probe_collectives(inner: nn.Module, inputs: tuple, vllm_config, num_tokens: int, bus, profile=None) -> list[dict]:
    """One probe export of the rank's wrapper with the bus in `probe` mode (graph discarded): records every collective's kind,
    shape, dtype and call order — the peer-input signature of the Program.  Same mechanism and shims as the real export (the fused CUDA
    ops have fake impls under export's FakeTensorMode, not Meta kernels for a plain meta forward)."""
    from .export_compat import export_compat
    from .triton_capture import intercept_triton_launches
    bus.reset("probe")
    with torch.no_grad(), fake_forward_context(vllm_config, num_tokens), torch.device("meta"), export_compat(profile, inner), intercept_triton_launches():
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
            except Exception as e:  # torch's DistNetworkError; only the port race is retried
                if "EADDRINUSE" not in str(e) or attempt == _PORT_TRIES - 1:
                    raise
        ensure_model_parallel_initialized(1, 1)
    _DIST_READY = True
    return backend


class _CudaFactoriesOnMeta(torch.overrides.TorchFunctionMode):
    """A factory call that names the CUDA device (`device=current_platform.device_type`) makes its tensor on `meta`: vLLM allocates a
    few buffers so in `__init__`, past the default device (DeepSeek-V3.2's `topk_indices_buffer`, deepseek_v2.py:1421-1427, which
    the layers read only for its shape), and a GPU-free Build host has no CUDA device to put them on."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        d = kwargs.get("device")
        if d is not None and torch.device(d).type == "cuda":
            kwargs = {**kwargs, "device": torch.device("meta")}
        return func(*args, **kwargs)


def instantiate_meta(vllm_config) -> nn.Module:
    """The registered model class for `vllm_config.model_config` (vLLM's registry), constructed on `meta` (shapes only); a buffer
    vLLM puts on the CUDA device explicitly is on `meta` too (`_CudaFactoriesOnMeta`), and a CUDA stream or event it makes is an
    inert stand-in (`export_compat.cuda_streams_inert`)."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.utils.torch_utils import set_default_torch_dtype

    from .export_compat import cuda_streams_inert
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
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype), torch.device("meta"), \
            _CudaFactoriesOnMeta(), cuda_streams_inert():
        model = model_cls(vllm_config=vllm_config)
    model.eval()
    model._verity_arch = arch  # annotation only
    return model


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

    Generic: any vLLM `*ForCausalLM`; `LP`, `T` and greedy sampling are the request's structural inputs (prompt
    length, max_tokens, sampling kind)."""

    DERIVE_SOURCE = "verity-vllm/greedy-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int) -> None:
        super().__init__()
        self.model, self.LP, self.T = model, LP, T

    def _step(self, ids: torch.Tensor, pos0: int, s: int) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import launch_region
        with launch_region(s, "tokens"):
            positions = torch.arange(pos0, pos0 + ids.shape[0], dtype=torch.int64)
            hidden = self.model(ids, positions)
        with launch_region(s, "sampled"):
            logits = self.model.compute_logits(hidden[-1:])
            return torch.argmax(logits, dim=-1).to(torch.int32)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        return self._decode(self._step(prompt, 0, 0))

    def _decode(self, tok: torch.Tensor) -> torch.Tensor:
        toks = [tok]
        for s in range(self.T):
            tok = self._step(tok, self.LP + s, s + 1)
            toks.append(tok)
        return torch.cat(toks)


class CachedGreedyRequest(GreedyRequest):
    """`GreedyRequest` for a request whose first `C` prompt rows vLLM's prefix cache serves: `forward(prompt, prefix)` hands `prefix`
    (bf16 [L, 2, C, KVH*D]: every attention layer's cached key and value rows, layers in forward order) to `vllm_iface.kv_prefix`, so
    each layer's history opens with them; prefills the uncached suffix prompt[C:] at positions C..LP-1 (the engine's launch rows), then
    decodes as `GreedyRequest`.  The whole prompt stays the token input (Match wires prompt rows by position); no gate reads rows 0..C-1."""

    DERIVE_SOURCE = "verity-vllm/cached-greedy-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int, C: int) -> None:
        super().__init__(model, LP, T)
        self.C = C

    def forward(self, prompt: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        iface.kv_prefix(prefix)
        return self._decode(self._step(prompt[self.C:], self.C, 0))


class PaddedGreedyRequest(nn.Module):
    """The FIXED padded request rectangle `prefill | decode 1 .. decode D_MAX` over the
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
    """`EngineStep` with the row's STOCHASTIC policy: the last row's logits -> `vllm_iface.gumbel_topp_token_select(logits,
    seed, temperature, top_p, pos, splits)` (vLLM d9105ea80 `Sampler.sample` with temperature + top-p and an explicit per-request seed:
    f32 widening -> `_temperature_kernel` -> `_apply_topp_split` -> `_gumbel_sample_kernel` + argmax).  `seed` (int64 [1]) is a forward
    INPUT (the request's `SamplingParams.seed`: a runtime value of the Program); temperature / top_p are the declared policy (structural
    inputs, constants of the Program); `pos` = the sampled logits row's position (pos0 + M - 1).  `splits`
    (int32 [1]) is a forward INPUT too: the event's split count S = SplitsFor_v1(|live|, num_SMs), a DERIVED value of the workload.
    With `S` given (the registered split count, `topp_geometry.registered_splits`) the forward takes no `splits`."""

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
    """`GreedyRequest` with the row's STOCHASTIC policy (temperature + top-p, explicit seed): prefill of `LP` tokens, then
    exactly `T` decode steps, every sampled token `gumbel_topp_token_select(logits, seed, temperature, top_p, pos, splits[s])` with `pos`
    the position of the logits row (LP-1 for the prefill sample, LP-1+k for the k-th sample: vLLM's `positions[logits_indices]`, the
    stream offset of `tl.randint(seed, pos)`).  `seed` is a forward INPUT (one per request; B2/B8 rows have one per component).

    `splits` (int32 [T+1]) is the request-owned DERIVED-S INPUT: entry s is the split
    count of sampling event s (the prefill sample = event 0), S_t = SplitsFor_v1(|live(t)|, num_SMs) at the engine step t the event runs
    in -- a value of the WORKLOAD (the live set), bound at Match to the split kernels' constexpr and checked against the relation there
    (GM-01 G6 'sampler geometry').  The Program carries no S schedule: the forward reads `splits[s:s+1]` per event (a static view of
    the Input).  Parameter order [prompt, seed, splits] so the workload Program's layout stays [prompts, seeds, splits, weights...].

    [registered S] With `S` given (`topp_geometry.registered_splits(num_SMs)` = SplitsFor_v1(1, num_SMs), every event of every
    workload) the forward takes [prompt, seed] and every select's split count is that constant (`Const32[S]` in the Program)."""

    DERIVE_SOURCE = "verity-vllm/stochastic-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int, temperature: float, top_p: float, S: int | None = None) -> None:
        super().__init__()
        self.model, self.LP, self.T = model, LP, T
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.S = S

    def _step(self, ids: torch.Tensor, pos0: int, seed: torch.Tensor, splits: torch.Tensor | None, s: int) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import launch_region
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        with launch_region(s, "tokens"):
            positions = torch.arange(pos0, pos0 + ids.shape[0], dtype=torch.int64)
            hidden = self.model(ids, positions)
        with launch_region(s, "sampled"):
            logits = self.model.compute_logits(hidden[-1:])
            return iface.gumbel_topp_token_select(logits, seed, self.temperature, self.top_p, pos0 + ids.shape[0] - 1, splits, S=self.S)

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, splits: torch.Tensor | None = None) -> torch.Tensor:
        return self._request(prompt, 0, seed, splits)

    def _request(self, prompt: torch.Tensor, C: int, seed: torch.Tensor, splits: torch.Tensor | None) -> torch.Tensor:
        entry = (lambda s: None) if self.S is not None else (lambda s: splits[s:s + 1])
        tok = self._step(prompt[C:] if C else prompt, C, seed, entry(0), 0)
        toks = [tok]
        for s in range(self.T):
            tok = self._step(tok, self.LP + s, seed, entry(s + 1), s + 1)
            toks.append(tok)
        return torch.cat(toks)


class CachedStochasticRequest(StochasticRequest):
    """`StochasticRequest` with a cached prefix of `C` rows (see `CachedGreedyRequest`): `forward(prompt, seed, [splits,] prefix)`, the
    prefix after the request's other inputs."""

    DERIVE_SOURCE = "verity-vllm/cached-stochastic-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int, temperature: float, top_p: float, C: int, S: int | None = None) -> None:
        super().__init__(model, LP, T, temperature, top_p, S=S)
        self.C = C

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        from verity_vllm.program.frontend.rules.vllm_iface import iface
        iface.kv_prefix(rest[-1])
        return self._request(prompt, self.C, seed, rest[0] if len(rest) == 2 else None)


def prefix_input(model: nn.Module, C: int) -> torch.Tensor:
    """The example `prefix` input of a cached request wrapper: bf16 [L, 2, C, KVH*D] over the model's L attention layers (one KV
    geometry for every layer; MLA's latent cache has no such prefix yet)."""
    impls = attention_impls(model)
    if not impls or any(v.get("kind") == "mla" for v in impls.values()):
        raise ValueError(f"a cached prefix is a dense attention cache's rows; this model's attention layers are {sorted({v.get('kind', 'dense') for v in impls.values()})}")
    geo = {(v["num_kv_heads"], v["head_size"]) for v in impls.values()}
    if len(geo) != 1:
        raise ValueError(f"a cached prefix holds one KV geometry for every layer; the layers have (KVH, D) in {sorted(geo)}")
    (kvh, d), = geo
    return torch.empty((len(impls), 2, C, kvh * d), dtype=torch.bfloat16, device="meta")


def registered_policy_splits(sampling: dict) -> int | None:
    """The registered split count S of a stochastic policy: `sampling["splits"]` when it is an int equal to
    `topp_geometry.registered_splits` of the policy's geometry's num_SMs (`build.sampler_geometry`); else None."""
    from verity.ml.definitions.registry.topp_geometry import registered_splits
    S, g = sampling.get("splits"), sampling.get("geometry") or {}
    if isinstance(S, bool) or not isinstance(S, int) or g.get("num_sms") is None:
        return None
    return S if S == int(registered_splits(int(g["num_sms"]))) else None


def cached_wrapper(mod: nn.Module, inputs: tuple, model: nn.Module, C: int) -> tuple[nn.Module, tuple]:
    """The request wrapper `mod` (with its example `inputs`) for a request whose first `C` prompt rows the prefix cache serves
    (`CachedGreedyRequest`, `CachedStochasticRequest`), the prefix (`prefix_input`) its last input."""
    if type(mod) not in (GreedyRequest, StochasticRequest) or not 0 < C < mod.LP:
        raise ValueError(f"a cached prefix of C={C} rows needs a request wrapper and 0 < C < LP, got {type(mod).__name__}")
    if isinstance(mod, StochasticRequest):
        mod = CachedStochasticRequest(model, mod.LP, mod.T, mod.temperature, mod.top_p, C, S=mod.S)
    else:
        mod = CachedGreedyRequest(model, mod.LP, mod.T, C)
    return mod, (*inputs, prefix_input(model, C))


def make_wrapper(kind: str, model: nn.Module, *, pos0: int = 8, M: int = 1, LP: int = 8, T: int = 4, eos_id: int = 0,
                 sampling: dict | None = None, C: int = 0) -> tuple[nn.Module, tuple]:
    """(module, example inputs on meta) for `kind` in {step, request, padded}; `C` > 0: the request wrapper's `cached_wrapper`.
    [X-05] `sampling` is the row's declared policy: the greedy wrappers author `torch.argmax`; `{"kind": "stochastic", "temperature", "top_p", "top_k": 0, "seeded": True, "splits": "derived"}`
    authors `vllm_iface.gumbel_topp_token_select` with the seed as a second forward input and the derived split
    count `splits` (int32, one entry per sampling event: [1] for the step wrapper, [T+1] for the request wrapper) as a third (step /
    request wrappers; padded is greedy-only).  A numeric `splits` is the registered S = SplitsFor_v1(1, num_SMs) the served engine
    pins (`build.sampler_geometry`, `registered_policy_splits`): the wrapper takes no `splits` input.  Any other numeric `splits`
    (the earlier static S / declared schedule) is REFUSED, as is any other policy (derive_step refuses earlier)."""
    if C:
        return cached_wrapper(*make_wrapper(kind, model, pos0=pos0, M=M, LP=LP, T=T, eos_id=eos_id, sampling=sampling), model, C)
    kind_s = (sampling or {"kind": "greedy"}).get("kind")
    if kind_s == "stochastic":
        s = sampling or {}
        if int(s.get("top_k", 0) or 0) != 0 or not s.get("seeded", False):
            raise ValueError(f"make_wrapper authors the seeded temperature + top-p select only (top_k == 0, seeded): {sampling}")
        S = registered_policy_splits(s)
        if s.get("splits") != "derived" and S is None:
            raise ValueError(f"make_wrapper: sampling.splits must be the registered SplitsFor_v1(1, num_SMs) of its geometry (the S the served "
                             f"engine pins) or 'derived' (a request-owned Input of the Program) -- any other static S or a declared "
                             f"schedule is not a Program the Match can hold, got {s.get('splits')!r}")
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


#: environment variables this construction SETS (module top, or the target's `env()`) — recorded as consulted only if the pinned vLLM's `envs` defines them
_ENV_SET_BY_BUILDER = ("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL", "TOKENIZERS_PARALLELISM", "VLLM_ATTENTION_BACKEND",
                       "VLLM_USE_DEEP_GEMM")   # the block-scaled FP8 kernel knob `TargetProfile.fp8_block_gemm` applies


def consulted_facts(vllm_config) -> dict:
    """Configuration facts the construction actually CONSULTED that the exported graph does not carry: the vLLM config's attention
    backend knob, KV block size / dtype, model dtype, eager mode, the platform class — plus, split honestly, the environment:
    `env_consulted` = variables the builder sets AND the pinned vLLM's `envs` module defines (so vLLM reads them);
    `env_ignored`   = variables set in the process that this vLLM pin does NOT define (e.g. `VLLM_ATTENTION_BACKEND`, dead in d9105ea80)
    — never reported as declared facts. The target facts themselves (capability, SM count, BI switch) live in `target.declared/applied`."""
    from verity_vllm import settings
    facts: dict = {}
    try:
        import vllm.envs as envs
        defined = set(getattr(envs, "environment_variables", {}) or {})
    except Exception:  # noqa: BLE001
        defined = set()
    env = {k: settings.raw(k) for k in _ENV_SET_BY_BUILDER if settings.is_set(k)}
    facts["env_consulted"] = {k: v for k, v in env.items() if k in defined}
    facts["env_ignored"] = {k: {"value": v, "reason": "not defined in this vLLM pin's `vllm.envs`; never read"}
                            for k, v in env.items() if k not in defined and k.startswith("VLLM_")}
    facts["env_set_not_vllm"] = {k: v for k, v in env.items() if k not in defined and not k.startswith("VLLM_")}
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
    export context right after the export — what vLLM saw while the graph was produced (the driver refuses if it differs from the profile).
    Under a declared launch batch the graph keeps each node's `launch_region` tag."""
    import torch.fx.traceback as fxt
    from .export_compat import export_compat, read_back_applied
    from .rules.vllm_bindings.observations import launch_batch_declared
    from .triton_capture import intercept_triton_launches
    with (torch.no_grad(), fake_forward_context(vllm_config, num_tokens), torch.device("meta"), export_compat(profile, mod), no_fx_stack_traces(),
          fxt.preserve_node_meta() if launch_batch_declared() else nullcontext()):
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
           "CachedGreedyRequest", "CachedStochasticRequest", "cached_wrapper", "prefix_input", "PaddedGreedyRequest", "make_wrapper", "export_module", "export_step", "op_histogram", "consulted_facts",
           "attention_impls", "export_facts", "BI_TRITON_KERNELS"]


def attention_impls(model: nn.Module) -> dict:
    """OBSERVED (not declared) attention implementation per `Attention` layer of the instantiated model: the selector's output
    (`layer.impl` class, `layer.backend`), the FA version, window / soft cap / alibi / sinks, kv dtype, and whether the impl runs
    in batch-invariant mode (single-split combine). Keyed by `layer_name` — what the exported `unified_attention_with_output`
    node carries — so the attention rule can select the Definition from the layer that will run, not from an env string.  Each
    `MLAAttention` layer has an entry with `"kind": "mla"`: its impl, backend name, kv dtype, heads, scale, context-parallel sizes,
    dense-MHA prefill backend (with its FA version, whether it pads v to the qk head and its scale), the FA3 split count its impl's module calls `flash_attn_varlen_func` with (`num_splits`: the one
    `engine.sparse_mla`'s pin marks, None for vLLM's heuristic), and from the layer's vLLM config the query heads per rank its
    metadata builder keys the decode threshold on and `sparse_mla_force_mqa`, what `rules/vllm_dsa.py` reads.  Any other attention
    layer (`AttentionLayerBase`: DeepSeek-V4's sparse MLA, its window, indexer and compressor-state caches) has an entry with `"kind":
    "layer"` under its prefix: its class, its backend's name and its kv cache dtype."""
    from vllm.model_executor.layers.attention import Attention, MLAAttention
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
        if callable(kb := getattr(impl, "k_scale_bits", None)):      # the int8 backend's K scales (engine.int8_attention_vllm)
            obs["int8_k_scales"] = [b & 0xFFFFFFFF for b in kb(getattr(mod, "layer_name", name))]
        out[getattr(mod, "layer_name", name)] = obs
    for name, mod in model.named_modules():
        if not isinstance(mod, MLAAttention):
            continue
        impl = mod.impl
        backend = getattr(mod, "attn_backend", None)
        pb = getattr(mod, "prefill_backend", None)
        fa = getattr(sys.modules.get(type(impl).__module__), "flash_attn_varlen_func", None)
        vc = mod._vllm_config
        out[getattr(mod, "layer_name", name)] = {
            "prefill_backend": None if pb is None else f"{type(pb).__module__}.{type(pb).__qualname__}",
            "prefill_fa_version": getattr(pb, "vllm_flash_attn_version", None),
            "prefill_requires_v_padding": getattr(pb, "requires_v_padding", None), "prefill_scale": getattr(pb, "scale", None),
            "kind": "mla", "impl": f"{type(impl).__module__}.{type(impl).__qualname__}",
            "backend": backend.get_name() if hasattr(backend, "get_name") else str(backend),
            "kv_cache_dtype": getattr(impl, "kv_cache_dtype", None), "layer_kv_cache_dtype": getattr(mod, "kv_cache_dtype", None),
            "num_heads": getattr(impl, "num_heads", None), "scale": getattr(impl, "scale", None),
            "kv_lora_rank": getattr(mod, "kv_lora_rank", None), "qk_rope_head_dim": getattr(mod, "qk_rope_head_dim", None),
            "dcp_world_size": getattr(impl, "dcp_world_size", None), "use_pcp": getattr(mod, "use_pcp", None),
            "use_sparse": getattr(mod, "use_sparse", None), "num_splits": getattr(fa, "verity_sparse_mla_splits", None),
            "q_heads_per_rank": vc.model_config.get_num_attention_heads(vc.parallel_config),
            "sparse_mla_force_mqa": bool(vc.attention_config.sparse_mla_force_mqa)}
    base = getattr(sys.modules.get("vllm.model_executor.layers.attention_layer_base"), "AttentionLayerBase", None)
    for name, mod in model.named_modules():
        if base is None or isinstance(mod, (Attention, MLAAttention)) or not isinstance(mod, base):
            continue
        out[getattr(mod, "prefix", name)] = {"kind": "layer", "impl": f"{type(mod).__module__}.{type(mod).__qualname__}",
                                             "backend": mod.get_attn_backend().get_name(), "kv_cache_dtype": getattr(mod, "kv_cache_dtype", None)}
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
    """The installed vLLM's `aten::mean.dim` batch-invariant override, read in-process (never assumed): the Triton
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
