"""``verity_vllm.LLM``: a pinned vLLM engine that is constructed and called like ``vllm.LLM``.

The engine is built by :func:`verity_vllm.engine.vllm_adapter.build_engine` from the model's pin in
``manifests/checkpoints.json`` (:class:`verity_vllm.config.EnginePin`), the path every ``verity-vllm`` stage builds its
engine on.  ``generate`` returns vLLM's own outputs; the records sit beside them on the object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from verity_vllm import config
from verity_vllm.config import EnginePin


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
    "max_model_len": Supported(int, None, "context length (default: the pin's)"),
    "gpu_memory_utilization": Supported((float, int), None, "fraction of device memory the engine may take (default: the "
                                                            "engine profile's)"),
    "enforce_eager": Supported(bool, True, "True runs eagerly; False compiles with torch.compile and CUDA graphs"),
    "dtype": Supported((str, type(None)), None, "served dtype override (default: the checkpoint profile's)"),
    "quantization": Supported((str, type(None)), None, "vLLM quantization override, such as fp8 (default: the checkpoint's)"),
    "compilation_config": Supported((dict, type(None)), None, "vLLM compilation and CUDA-graph settings"),
    "seed": Supported(int, 0, "the engine seed"),
    "protocols": Supported((str, dict, list, tuple), None, "protocol options installed on the engine: 'pouw:<scheme>,pous:<scheme>' "
                                                             "or {name: scheme} (verity_vllm.protocol_options); default none"),
    "protocols_config": Supported(dict, None, "each enabled protocol's settings, {name: {...}}; never hashed"),
}
#: Options passed to vLLM only when the caller sets them; unset, the engine profile's value stands.
ENGINE_ARGS = ("max_num_batched_tokens", "max_model_len", "gpu_memory_utilization", "dtype", "quantization",
               "compilation_config", "seed")
#: Explicit overrides that can select kernels outside a protocol option's Build.
KERNEL_PROFILE_OVERRIDES = ("dtype", "quantization", "compilation_config")
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
    return out


def protocol_context(opts: dict[str, Any], explicit: dict[str, Any] | None = None) -> Any:
    """The protocol options' serving context for resolved options, None when none is enabled; ``ValueError`` names every
    refusal, before any engine exists."""
    enabled = config.parse_protocols(opts["protocols"])
    if not enabled:
        if opts["protocols_config"]:
            raise ValueError("verity_vllm.LLM: protocols_config without protocols")
        return None
    from verity_vllm import protocol_options as PO
    ctx = PO.Context(stage="serve", enabled=enabled, world=opts["tensor_parallel_size"], enforce_eager=opts["enforce_eager"],
                     settings=PO.load_settings(opts["protocols_config"]))
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
    ``verity-vllm`` stage; prefix caching is off.

    Supported options (keyword only; any other raises ``TypeError`` naming it and this list):

    * ``tensor_parallel_size`` (int, default 1): GPUs the model is sharded over.
    * ``max_num_seqs`` (int, default 8): requests the scheduler runs together.
    * ``max_num_batched_tokens`` (int): tokens per scheduler step; default vLLM's for the model length.
    * ``max_model_len`` (int): context length; default the pin's.
    * ``gpu_memory_utilization`` (float): fraction of device memory the engine may take; default the engine profile's.
    * ``enforce_eager`` (bool, default True): True runs eagerly; False compiles with torch.compile and CUDA graphs, in a
      fresh Inductor cache for this process.
    * ``dtype`` (str): served dtype override; default the checkpoint profile's.
    * ``quantization`` (str): vLLM quantization override, such as ``"fp8"``; default the checkpoint's.
    * ``compilation_config`` (dict): vLLM compilation and CUDA-graph settings.
    * ``seed`` (int, default 0): the engine seed.
    * ``protocols`` (str or mapping): protocol options installed on the engine once it is built
      (:mod:`verity_vllm.protocol_options`), ``"pouw:<scheme>,pous:<scheme>"`` or ``{"pous": "<scheme>"}``; default none.
      A set the options refuse raises ``ValueError`` before the engine is built.  Sampled proofs runs in the Commit stage
      of a row, not here.
    * ``protocols_config`` (dict): each enabled protocol's settings, ``{name: {...}}``; never hashed.

    ``generate(prompts, sampling_params=None, use_tqdm=True)`` returns vLLM's ``list[RequestOutput]`` unchanged and
    appends a :class:`GenerateRecord` to ``records``.  ``pin`` is the resolved :class:`~verity_vllm.config.EnginePin`
    and ``engine_record`` the engine as built: the ``vllm.LLM`` keyword arguments, the environment flags the profile
    set, and the execution vLLM reports back.  ``protocols`` is the installed
    :class:`~verity_vllm.protocol_options.Composition` (None without protocols); ``close()`` uninstalls it.  The engine
    is always built here; an engine built elsewhere is not accepted.
    """

    def __init__(self, model: str, *, revision: str, **options: Any) -> None:
        opts = resolve_options(options)
        self.options = opts
        self.engine_overrides = {k: options[k] for k in KERNEL_PROFILE_OVERRIDES if options.get(k) is not None}
        self.pin = EnginePin.resolve(model, revision, world=opts["tensor_parallel_size"])
        ctx = protocol_context(opts, options)
        from verity_vllm.engine import vllm_adapter as va

        engine_args = {k: opts[k] for k in ENGINE_ARGS if k in options}
        if self.pin.world > 1:
            engine_args["tensor_parallel_size"] = self.pin.world
        self._engine = va.build_engine(config.CHECKPOINTS, max_num_seqs=opts["max_num_seqs"], case=self.pin.case,
                                       engine_args=engine_args, execution=va.execution_label_of(opts["enforce_eager"]))
        self.records: list[GenerateRecord] = []
        self.spec_digest: str | None = None
        self.protocols = None
        if ctx is not None:
            from verity_vllm import protocol_options as PO
            self.protocols = PO.install(self._engine.model, ctx)

    @classmethod
    def from_spec(cls, spec: Any) -> LLM:
        """The engine an engine spec describes (`pipeline.engine_spec`: a path to its TOML, or a loaded `EngineSpec`),
        with the spec's digest in ``engine_record["spec_digest"]``.  A spec that isn't strictly of the declared shape raises
        ``verity_config.ConfigError`` before the engine is built."""
        from verity_vllm.pipeline import engine_spec

        if not isinstance(spec, engine_spec.EngineSpec):
            spec = engine_spec.load(spec)
        llm = cls(spec.model, revision=spec.revision, **spec.options())
        llm.spec_digest = spec.digest
        return llm

    def close(self) -> None:
        """Uninstall the protocol options.  Idempotent."""
        if self.protocols is not None:
            self.protocols.close()

    @property
    def engine_record(self) -> dict[str, Any]:
        e = self._engine
        return {"pin": asdict(self.pin), "engine_kwargs": dict(e.kwargs), "env_flags": dict(e.env_flags),
                "execution": e.execution, "effective": e.effective, "build_s": e.build_s,
                "engine_overrides": dict(self.engine_overrides), "spec_digest": self.spec_digest,
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
        """The served model in this process (the in-process model runner's, `engine_profile.get_model_runner`): the object
        protocol options install on."""
        return self._engine.model
