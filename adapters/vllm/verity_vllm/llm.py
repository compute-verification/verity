"""``verity_vllm.LLM``: a pinned vLLM engine that is constructed and called like ``vllm.LLM``.

The engine is built by :func:`verity_vllm.engine.vllm_adapter.build_engine` from the model's pin in
``manifests/checkpoints.json`` (:class:`verity_vllm.config.EnginePin`), the path every ``verity-vllm`` stage builds its
engine on.  ``generate`` returns vLLM's own outputs; the records sit beside them on the object.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any

from verity_vllm import config
from verity_vllm import protocols as PO
from verity_vllm.config import EnginePin
from verity_vllm.engine import served
from verity_vllm.engine import spec as engine_spec
from verity_vllm.engine.class_admission import Admission
from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.errors import Refused

#: The names the library's other modules and files outside the adapter read here.  Outside the distribution the class is
#: ``from verity_vllm import LLM``.
__all__ = ["LLM", "protocol_context", "resolve_options"]


@dataclass(frozen=True)
class Supported:
    """One constructor option: its accepted type, the default and what it sets."""

    type: type | tuple[type, ...]
    default: Any
    help: str


#: The constructor options ``LLM`` accepts, beside ``model`` and ``revision``.  Anything else raises ``TypeError``.
SUPPORTED: dict[str, Supported] = {
    "tensor_parallel_size": Supported(int, 1, "GPUs the model is sharded over"),
    "max_num_seqs": Supported(int, 8, "requests the scheduler runs together"),
    "max_num_batched_tokens": Supported(int, None, "tokens per scheduler step (default: vLLM's for the model length)"),
    "long_prefill_token_threshold": Supported(int, None, "the most tokens of one prompt a scheduler step prefills (default: vLLM's, "
                                                         "0: no cap)"),
    "max_model_len": Supported(int, None, "context length (default: the pin's; -1: the longest request the KV pool holds)"),
    "gpu_memory_utilization": Supported((float, int), None, "fraction of device memory the engine may take (default: the "
                                                            "engine profile's)"),
    "kv_cache_memory_bytes": Supported(int, None, "the KV cache's size in bytes, in place of sizing it from free memory "
                                                  "(default: vLLM's sizing)"),
    "enable_chunked_prefill": Supported(bool, None, "False holds a prefill that doesn't fit a step instead of cutting it into chunks "
                                                    "(default: vLLM's, on)"),
    "enable_prefix_caching": Supported(bool, None, "True lets a request read another's cached KV blocks for a shared prefix "
                                                   "(default: off)"),
    "async_scheduling": Supported(bool, None, "False schedules each step once the previous step's output is in (LAG 0) "
                                              "(default: vLLM's resolution, on)"),
    "enforce_eager": Supported(bool, True, "True runs eagerly; False compiles with torch.compile and CUDA graphs"),
    "execution": Supported((str, type(None)), None, "the execution label the engine is built for (engine.served.EXECUTIONS), which "
                                                    "sets its batch invariance and enforce_eager (default: bi-eager, or bi-compiled "
                                                    "with enforce_eager False)"),
    "engine_class": Supported((str, type(None)), None, "a registered engine class (engine.profiles.engine_classes), whose world, "
                                                       "execution, engine limits, prefix cache, experts and environment the engine "
                                                       "runs at, and "
                                                       "whose window's limits its admission holds (default: none)"),
    "dtype": Supported((str, type(None)), None, "served dtype override (default: the checkpoint profile's)"),
    "quantization": Supported((str, type(None)), None, "vLLM quantization override, such as fp8 (default: the checkpoint's)"),
    "compilation_config": Supported((dict, type(None)), None, "vLLM compilation and CUDA-graph settings"),
    "attention_backend": Supported((str, type(None)), None, "vLLM's attention backend, such as FLASH_ATTN (default: vLLM's choice)"),
    "attention_config": Supported((dict, type(None)), None, "vLLM attention settings, such as {'flash_attn_version': 2}"),
    "moe_backend": Supported((str, type(None)), None, "vLLM's MoE backend, such as triton (default: vLLM's auto)"),
    "seed": Supported(int, 0, "the engine seed"),
    "protocols": Supported((str, dict, list, tuple), None, "protocol options installed on the engine: 'pouw:<scheme>,pous:<scheme>' "
                                                             "or {name: scheme} (verity_vllm.protocols); default none"),
    "protocols_config": Supported(dict, None, "each enabled protocol's settings, {name: {...}}; never hashed"),
}
#: Options passed to vLLM only when the caller sets them; unset, the engine profile's value stands.
ENGINE_ARGS = ("max_num_batched_tokens", "long_prefill_token_threshold", "max_model_len", "gpu_memory_utilization",
               "kv_cache_memory_bytes", "enable_chunked_prefill", "enable_prefix_caching", "async_scheduling", "dtype",
               "quantization", "compilation_config", "attention_backend", "attention_config", "moe_backend", "seed")
#: Explicit overrides that can select kernels outside a protocol option's Build.
KERNEL_PROFILE_OVERRIDES = ("dtype", "quantization", "compilation_config", "attention_backend", "attention_config", "moe_backend")
#: The keyword arguments ``generate`` accepts beside the prompts and sampling parameters.
GENERATE_SUPPORTED = ("use_tqdm",)


def resolve_options(options: dict[str, Any]) -> dict[str, Any]:
    """Every supported option with the caller's values over the defaults; ``TypeError`` names an unsupported option or a
    value of the wrong type."""
    unknown = sorted(set(options) - set(SUPPORTED))
    if unknown:
        raise TypeError(f"verity_vllm.LLM does not support {', '.join(map(repr, unknown))}; "
                        f"supported options: {', '.join(SUPPORTED)}")
    out = {k: s.default for k, s in SUPPORTED.items()}
    for k, v in options.items():
        want = SUPPORTED[k].type
        if (isinstance(v, bool) and want is not bool) or not isinstance(v, want):
            name = want.__name__ if isinstance(want, type) else want[0].__name__
            raise TypeError(f"verity_vllm.LLM option {k!r} must be {name}, got {v!r}")
        out[k] = v
    if out["tensor_parallel_size"] < 1 or out["max_num_seqs"] < 1:
        raise ValueError("verity_vllm.LLM: tensor_parallel_size and max_num_seqs must be at least 1")
    if out["execution"] is not None:
        eager = served.execution(out["execution"])[0]["enforce_eager"]
        if options.get("enforce_eager", eager) != eager:
            raise ValueError(f"verity_vllm.LLM: execution {out['execution']!r} builds with enforce_eager={eager}, not "
                             f"{options['enforce_eager']}")
        out["enforce_eager"] = eager
    return out


def class_options(options: dict[str, Any]) -> dict[str, Any]:
    """``options`` with its ``engine_class``'s settings filled in (`engine.profiles.engine_classes.options`); `errors.Refused`
    names every option that contradicts the class, every kernel override it doesn't fix, a ``max_num_seqs`` over its and a
    launch that doesn't state its context limit (``max_model_len``: a length, or -1), which the class leaves to it."""
    name = options.get("engine_class")
    if name is None:
        return dict(options)
    resolve_options(options)
    execution = options.get("execution") if options.get("execution") in EC.EXECUTIONS.get(name, {}) else None
    fixed, most = EC.options(name, execution), EC.max_num_seqs(name)
    bad = [f"{k}={options[k]!r}, where the class serves {v!r}" for k, v in fixed.items() if k in options and options[k] != v]
    bad += [f"{k}={options[k]!r}: the class serves the engine profile's" for k in KERNEL_PROFILE_OVERRIDES
            if k not in fixed and options.get(k) is not None]
    if options.get("max_num_seqs", most) > most:
        bad.append(f"max_num_seqs={options['max_num_seqs']}, over the class's {most}")
    limit = options.get("max_model_len")
    if limit is None or not (limit > 0 or limit == -1):
        bad.append(f"max_model_len={limit!r}: the class serves at its launch's context limit, so state one (a length, or -1 for "
                   "the longest request the engine's KV pool holds, vLLM's --max-model-len auto)")
    if bad:
        raise Refused(bad, what=f"engine class {name}")
    return {**fixed, **options, "max_num_seqs": options.get("max_num_seqs", most)}


def protocol_context(opts: dict[str, Any], explicit: dict[str, Any] | None = None) -> Any:
    """The protocol options' serving context for resolved options, None when none is enabled; ``ValueError`` names every
    refusal, before any engine exists.  The settings an adapter takes from the built engine (`protocol_options.engine_fills`)
    are named in its ``filled`` and read as set until `LLM` fills them."""
    enabled = config.parse_protocols(opts["protocols"])
    if not enabled:
        if opts["protocols_config"]:
            raise ValueError("verity_vllm.LLM: protocols_config without protocols")
        return None
    ctx = PO.Context(stage="serve", enabled=enabled, world=opts["tensor_parallel_size"], enforce_eager=opts["enforce_eager"],
                     settings=PO.load_settings(opts["protocols_config"]))
    ctx = dataclasses.replace(ctx, filled=PO.engine_fills(ctx))
    overrides = tuple(k for k in KERNEL_PROFILE_OVERRIDES if explicit and explicit.get(k) is not None)
    bad = []
    for entry in enabled:
        try:
            needs_build = PO.requires_build(entry, ctx)
        except PO.Refused:
            continue
        if needs_build and overrides:
            bad.append(f"{entry}: engine overrides {', '.join(overrides)} can change the served kernels and no longer match its "
                       "Build; use the pinned engine profile")
    bad.extend(PO.refusals(ctx))
    if bad:
        raise PO.Refused(bad)
    return ctx


@dataclass(frozen=True)
class GenerateRecord:
    """One ``generate`` call: the sampling parameters as given and, per request, its prompt and generated token ids."""

    sampling: list[str]
    requests: list[dict[str, Any]]

    @classmethod
    def of(cls, outputs: list[Any], sampling_params: Any) -> GenerateRecord:
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params]
        return cls(sampling=[repr(s) for s in sps],
                   requests=[{"request_id": o.request_id, "prompt_token_ids": list(o.prompt_token_ids or []),
                              "outputs": [{"token_ids": list(c.token_ids), "finish_reason": c.finish_reason} for c in o.outputs]}
                             for o in outputs])


class LLM:
    """A vLLM engine for a pinned model, built and called like ``vllm.LLM``::

        from vllm import SamplingParams
        from verity_vllm import LLM

        llm = LLM("unsloth/Llama-3.2-1B", revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da")
        outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))

    ``model`` is a Hugging Face repo id and ``revision`` its full commit sha; both must be pinned in
    ``manifests/checkpoints.json`` with the snapshot on this host, else the constructor raises ``ValueError``.  The dtype,
    tokenizer, attention backend and the rest of the engine come from the pin's engine profile, as in every
    ``verity-vllm`` stage; prefix caching is off unless ``enable_prefix_caching=True``.

    Supported options (keyword only; any other raises ``TypeError`` naming it and this list):

    * ``tensor_parallel_size`` (int, default 1): GPUs the model is sharded over.
    * ``max_num_seqs`` (int, default 8): requests the scheduler runs together.
    * ``max_num_batched_tokens`` (int): tokens per scheduler step; default vLLM's for the model length.
    * ``long_prefill_token_threshold`` (int): the most tokens of one prompt a scheduler step prefills, the rest in later
      steps; default vLLM's, 0, no cap.
    * ``max_model_len`` (int): context length; default the pin's, and -1 the longest request the KV pool holds, which vLLM fits
      when the engine starts.
    * ``gpu_memory_utilization`` (float): fraction of device memory the engine may take; default the engine profile's.
    * ``kv_cache_memory_bytes`` (int): the KV cache's size in bytes, in place of sizing it from free memory; default vLLM's
      sizing.
    * ``enable_chunked_prefill`` (bool): False holds a prefill that doesn't fit a step instead of cutting it; default vLLM's.
    * ``enable_prefix_caching`` (bool): True lets a request read the KV blocks another computed for the same prefix; default
      off.
    * ``async_scheduling`` (bool): False schedules each step once the previous step's output is in (LAG 0); default vLLM's
      resolution, on.
    * ``enforce_eager`` (bool, default True): True runs eagerly; False compiles with torch.compile and CUDA graphs, in a
      fresh Inductor cache for this process.
    * ``execution`` (str): the execution label the engine is built for, a key of ``engine.served.EXECUTIONS``, which sets
      its batch invariance and ``enforce_eager`` (``"compiled"`` is vLLM's serving defaults); default ``"bi-eager"``, or
      ``"bi-compiled"`` with ``enforce_eager=False``.  An ``enforce_eager`` the label contradicts raises ``ValueError``.
    * ``engine_class`` (str): a registered engine class (`engine.profiles.engine_classes`), such as ``"G"``: the engine runs at
      the class's ``tensor_parallel_size``, an ``execution`` it serves (default its own; class G's ``"compiled"`` at its CUDA
      graphs' ``compilation_config``), served engine limits, ``moe_backend`` and prefix cache off, with
      ``max_num_seqs`` at most the class's (its default), under the class's environment (class G's
      ``VERITY_DSA_TOPK=canonical``), set over the process's, at the ``max_model_len`` the launch states (a length, or -1:
      the longest request the engine's KV pool holds).  An option that contradicts the class, another
      kernel override, a launch with no ``max_model_len``, or a checkpoint of another model type raises `errors.Refused` before
      the engine is built, and so does a
      built engine whose vLLM config or environment doesn't carry the class's settings after it.  The built engine's admission
      (`engine.class_admission`) refuses a prompt over the engine's ``max_model_len``, and
      ``close_window()`` refuses while a request is in flight; a refusal ends the window.  Default none.
    * ``dtype`` (str): served dtype override; default the checkpoint profile's.
    * ``quantization`` (str): vLLM quantization override, such as ``"fp8"``; default the checkpoint's.
    * ``compilation_config`` (dict): vLLM compilation and CUDA-graph settings.
    * ``attention_backend`` (str): vLLM's attention backend, such as ``"FLASH_ATTN"``; default vLLM's choice for the device.
    * ``attention_config`` (dict): vLLM attention settings, such as ``{"flash_attn_version": 2}``.
    * ``moe_backend`` (str): vLLM's MoE backend, such as ``"triton"``; default vLLM's ``"auto"``.
    * ``seed`` (int, default 0): the engine seed.
    * ``protocols`` (str or mapping): protocol options installed on the engine once it is built
      (:mod:`verity_vllm.protocols`), ``"pouw:<scheme>,pous:<scheme>"`` or ``{"pous": "<scheme>"}``; default none.
      A set the options refuse raises ``ValueError`` before the engine is built.  Sampled proofs runs in the Commit stage
      of a row, not here.  Settings an option takes from the engine are filled from this one after the build
      (`protocol_options.from_engine`), and one it can't fill there is refused by name: PoUW's H1
      (``protocols="pouw:eps1-sm120-h1fx1", protocols_config={"pouw": {"rows": T, "steps": S}}``) gets the engine's
      model runner and, without a ``challenger``, in-process coins, which credit nothing.
    * ``protocols_config`` (dict): each enabled protocol's settings, ``{name: {...}}``; never hashed.

    ``generate(prompts, sampling_params=None, use_tqdm=True)`` returns vLLM's ``list[RequestOutput]`` unchanged and
    appends a :class:`GenerateRecord` to ``records``.  ``pin`` is the resolved :class:`~verity_vllm.config.EnginePin`
    and ``engine_record`` the engine as built: the ``vllm.LLM`` keyword arguments, the environment flags the profile
    set, and the execution vLLM reports back.  ``protocols`` is the installed
    :class:`~verity_vllm.protocols.Composition` (None without protocols); ``close()`` uninstalls it, and
    ``reinstall(protocols_config=None)`` installs the same set afresh (H1's next window).  The engine is always built
    here; an engine built elsewhere is not accepted.
    """

    def __init__(self, model: str, *, revision: str, **options: Any) -> None:
        options = class_options(options)
        opts = resolve_options(options)
        self.options = opts
        self.engine_overrides = {k: options[k] for k in KERNEL_PROFILE_OVERRIDES if options.get(k) is not None}
        self.pin = EnginePin.resolve(model, revision, world=opts["tensor_parallel_size"])
        if opts["engine_class"] is not None:
            EC.check_model(opts["engine_class"], self.pin.model, self.pin.revision)
        ctx = protocol_context(opts, options)
        from verity_vllm.engine import vllm_adapter as va

        engine_args = {k: opts[k] for k in ENGINE_ARGS if k in options}
        if self.pin.world > 1:
            engine_args["tensor_parallel_size"] = self.pin.world
        env: dict[str, Any] = {} if opts["engine_class"] is None else {"env": EC.env(opts["engine_class"])}
        self._engine = va.build_engine(config.CHECKPOINTS, max_num_seqs=opts["max_num_seqs"], case=self.pin.case,
                                       engine_args=engine_args,
                                       execution=opts["execution"] or va.execution_label_of(opts["enforce_eager"]), **env)
        self.records: list[GenerateRecord] = []
        self.spec_digest: str | None = None
        self.protocols: PO.Composition | None = None
        self._explicit, self._installs = dict(options), 0
        self.admission = None
        if opts["engine_class"] is not None:
            bad = EC.effective_refusals(opts["engine_class"], self._engine.effective or {}, dict(self._engine.env_flags or {}),
                                        opts["max_model_len"])
            if bad:
                raise Refused(bad, what=f"engine class {opts['engine_class']}")
            self.admission = Admission(opts["engine_class"]).install(self._engine.llm)
        if ctx is not None:
            self._install(ctx)

    def _install(self, ctx: Any) -> None:
        filled = PO.from_engine(ctx, self._engine.llm, epoch=self._installs)
        self._installs += 1
        self.protocols = PO.install(self._engine.model, filled)

    def reinstall(self, protocols_config: dict[str, Any] | None = None) -> Any:
        """Close the installed protocol options and install the constructor's ``protocols`` afresh on this engine, with
        ``protocols_config`` in place of the constructor's when given, and return the new composition.  The settings
        taken from the engine are filled again, with fresh in-process coins of the next epoch: for H1, one window.
        ``ValueError`` without ``protocols``, or naming every refusal of the new settings."""
        opts = {**self.options, **({} if protocols_config is None else {"protocols_config": protocols_config})}
        if not config.parse_protocols(opts["protocols"]):
            raise ValueError("verity_vllm.LLM.reinstall: this LLM was built without protocols")
        ctx = protocol_context(opts, self._explicit)
        self.close()
        self.protocols = None
        self._install(ctx)
        return self.protocols

    @classmethod
    def from_spec(cls, spec: Any) -> LLM:
        """The engine an engine spec describes (`pipeline.engine_spec`: a path to its TOML, or a loaded `EngineSpec`),
        with the spec's digest in ``engine_record["spec_digest"]``.  A spec that isn't strictly of the declared shape raises
        ``verity_config.ConfigError`` before the engine is built."""
        if not isinstance(spec, engine_spec.EngineSpec):
            spec = engine_spec.load(spec)
        llm = cls(spec.model, revision=spec.revision, **spec.options())
        llm.spec_digest = spec.digest
        return llm

    def close(self) -> None:
        """Uninstall the protocol options.  Idempotent."""
        if self.protocols is not None:
            self.protocols.close()

    def close_window(self) -> None:
        """Close a Commit window on this engine.  Under an engine class, `errors.Refused` while a request is in flight or once the
        class has refused one (`engine.class_admission`); without one there is nothing to hold."""
        if self.admission is not None:
            self.admission.close_window()

    @property
    def engine_record(self) -> dict[str, Any]:
        e = self._engine
        return {"pin": asdict(self.pin), "engine_kwargs": dict(e.kwargs), "env_flags": dict(e.env_flags),
                "execution": e.execution, "effective": e.effective, "build_s": e.build_s,
                "engine_overrides": dict(self.engine_overrides), "spec_digest": self.spec_digest,
                "engine_class": self.options["engine_class"],
                "protocols": None if self.protocols is None else self.protocols.record()}

    def generate(self, prompts: Any, sampling_params: Any = None, **kwargs: Any) -> list[Any]:
        unknown = sorted(set(kwargs) - set(GENERATE_SUPPORTED))
        if unknown:
            raise TypeError(f"verity_vllm.LLM.generate does not support {', '.join(map(repr, unknown))}; "
                            f"supported: prompts, sampling_params, {', '.join(GENERATE_SUPPORTED)}")
        outputs = self._engine.llm.generate(prompts, sampling_params, **kwargs)
        self.records.append(GenerateRecord.of(outputs, sampling_params))
        return outputs

    def get_tokenizer(self) -> Any:
        return self._engine.llm.get_tokenizer()

    @property
    def vllm(self) -> Any:
        """The ``vllm.LLM`` this wraps, for a caller that drives its ``llm_engine`` step by step
        (`engine.arrivals.run_requests_arrivals`)."""
        return self._engine.llm

    @property
    def model(self) -> Any:
        """The served model in this process (the in-process model runner's, `engine.handle.model_runner`): the object
        protocol options install on."""
        return self._engine.model
