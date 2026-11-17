"""vLLM benchmark arms: stock engines, protocol options, and hook modules."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

from verity_vllm.engine.hooks import Hooks

from verity_bench.vllm.presets import NO_CONTROL, SERVED_EXECUTIONS, VLLM, ArmSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arm:
    """One engine configuration measured by the benchmark."""

    id: str
    protocols: tuple[str, ...] = ()
    precision: str | None = None
    control: str | None = None
    module_spec: str | None = None
    kernel_digest: str | None = None
    experimental: bool = False
    resident: bool = True
    #: a protocol arm's protocols config for `verity_vllm.LLM` (`presets.ArmSettings`), "$NAME" values unexpanded
    protocols_config: Mapping[str, Mapping[str, Any]] | None = None
    #: a served variant's execution (`served_variant`), or an experimental arm's own (its ``execution``): its engine is
    #: built at vLLM's serving defaults, eager or compiled, whatever the spec's `eager` and the preset's
    #: `serving_defaults`; None builds it as they say
    execution: str | None = None
    #: an experimental arm's name for the engine its own ``engine_overrides`` build (its ``engine_label``), recorded in
    #: its identity only: it never changes how the harness builds the engine, unlike ``execution``
    engine_label: str | None = None
    #: the engine settings (`presets.ARM_ENGINE`) and request sampling (`presets.ARM_SAMPLING`) the arm overrides for
    #: itself, a `presets.VLLM` value taking `vllm serve`'s (`serve_defaults.resolve`)
    engine: Mapping[str, Any] | None = None
    sampling: Mapping[str, Any] | None = None
    #: gated, then timed, each in an engine lifetime of its own after the other arms (`runner.ALONE`): its engine takes
    #: what vLLM leaves no other engine beside it
    alone: bool = False

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k not in OWN or v}


#: The fields an arm's record carries only when set, so the record of an arm without them is what it was before them
OWN = ("engine", "sampling", "alone")


@dataclass
class Target:
    """What an experimental arm may inspect and patch.

    Every patch belongs to ``hooks``. The runner checks this after ``install``
    and always uninstalls the set when the engine closes. ``phase``,
    ``point`` and ``tokens`` describe the run ``begin_run`` opens: ``gate``,
    ``warmup`` or ``measure``, at that sweep point's id, serving that many
    prompt and output tokens.
    """

    arm: Arm
    engine: Any
    model: Any
    hooks: Hooks
    phase: str = ""
    point: str = ""
    tokens: int = 0


BUILTINS = {
    "stock": Arm("stock", precision="bfloat16"),
    # Different precisions need not generate identical greedy tokens.  The
    # stock arms are independent references; protocol arms use stock FP8 as
    # their bit-exact control.
    "stock-fp8": Arm("stock-fp8", precision="fp8"),
}
#: What `vllm serve MODEL` takes by default at the pin, as far as the harness sets it: vLLM's compile, scheduler, KV
#: sizing, utilization, prefix caching, context length and dtype (`serve_defaults.resolve`). The requests stay the
#: workload's, greedy with `ignore_eos`, so every arm serves the same requests; `serve_defaults.differences` records the
#: `vllm serve` defaults that leaves out.
SERVE_ENGINE = {k: VLLM for k in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "kv_cache_memory_bytes",
                                  "gpu_memory_utilization", "enable_prefix_caching", "dtype")}
SERVE = {"execution": "compiled", "sampling": {"detokenize": True}, "alone": True}
BUILTINS |= {
    "stock@serve": Arm("stock@serve", precision="bfloat16", engine={**SERVE_ENGINE, "quantization": VLLM}, **SERVE),
    "stock-fp8@serve": Arm("stock-fp8@serve", precision="fp8", engine=SERVE_ENGINE, **SERVE),
}


def _overrides(settings: ArmSettings) -> dict[str, Any]:
    return {k: dict(v) for k in OWN[:2] if (v := getattr(settings, k, None))}


def _protocols(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    return tuple(str(x) for x in value)


def load_object(spec: str) -> Any:
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"experimental arm {spec!r}: expected pkg.mod:NAME")
    mod: ModuleType = importlib.import_module(module)
    try:
        return getattr(mod, name)
    except AttributeError:
        raise ValueError(f"experimental arm {spec!r}: {module} has no {name!r}") from None


def experimental_arm(spec: str) -> Arm:
    obj = load_object(spec)
    if isinstance(obj, Arm):
        return Arm(**{**obj.to_json(), "module_spec": spec, "experimental": True})
    get = obj.get if isinstance(obj, Mapping) else lambda k, d=None: getattr(obj, k, d)
    name = get("id") or get("name") or spec.partition(":")[2].lower().replace("_", "-")
    if not callable(get("install")):
        raise ValueError(f"experimental arm {spec!r}: NAME must expose install(target)")
    kernel_digest = get("kernel_digest")
    execution = get("execution")
    if execution is not None and execution not in SERVED_EXECUTIONS:
        raise ValueError(f"experimental arm {spec!r}: execution {execution!r} is not one of {list(SERVED_EXECUTIONS)}")
    return Arm(
        id=str(name),
        protocols=_protocols(get("protocols")),
        precision=get("precision"),
        control=get("control"),
        module_spec=spec,
        kernel_digest=kernel_digest if isinstance(kernel_digest, str) else None,
        experimental=True,
        resident=bool(get("resident", True)),
        execution=execution,
        engine_label=get("engine_label") if isinstance(get("engine_label"), str) else None,
    )


def plugin(arm: Arm) -> Any | None:
    return None if arm.module_spec is None else load_object(arm.module_spec)


def build_bound(arm: Arm, world: int = 1) -> tuple[str, ...]:
    """The arm's protocol entries that bind the served computation to a row Build (`protocol_options.requires_build`). An
    entry whose protocol `verity_vllm.protocols` refuses (unknown, or its adapter not importable here) isn't one: it is
    logged, and the engine's own build refuses it by name."""

    if not arm.protocols:
        return ()
    from verity.errors import Refused
    from verity_vllm import protocols as PO

    ctx = PO.Context(stage="serve", enabled=arm.protocols, world=world)
    bound = []
    for entry in arm.protocols:
        try:
            if PO.requires_build(entry, ctx):
                bound.append(entry)
        except Refused as e:
            log.info("arm %r: protocol entry %r refused (%s), so it binds no Build here; the engine's build refuses it",
                     arm.id, entry, e)
    return tuple(bound)


def _protocol_arm(name: str, settings: ArmSettings | None = None) -> Arm:
    """A protocol arm: the preset's precision and stock FP8 as its control, unless its `ArmSettings` say otherwise."""

    entries = tuple(x for x in name.split("+") if x)
    if not entries or any(":" not in e for e in entries):
        raise ValueError(f"unknown benchmark arm {name!r}; use a built-in, protocol:scheme, or --arm-module")
    s = settings or ArmSettings()
    control = "stock-fp8" if s.control is None else None if s.control == NO_CONTROL else s.control
    config = {k: dict(v) for k, v in s.protocols_config.items()} or None
    return Arm(name, protocols=entries, precision=s.precision, control=control, protocols_config=config, **_overrides(s))


def served_variant(name: str, settings: ArmSettings | None = None) -> Arm:
    """``BUILTIN@LABEL``: the built-in arm's engine at vLLM's serving defaults in execution LABEL (`SERVED_EXECUTIONS`),
    so one spec times one engine in two executions (`stock-fp8` eager against `stock-fp8@compiled`). Its control is the
    built-in unless its `ArmSettings` say otherwise."""

    base, _, label = name.partition("@")
    plain = sorted(b for b in BUILTINS if "@" not in b)
    if base not in plain or label not in SERVED_EXECUTIONS:
        raise ValueError(f"served variant {name!r}: expected BUILTIN@LABEL with BUILTIN one of {plain} and "
                         f"LABEL one of {list(SERVED_EXECUTIONS)}")
    s = settings or ArmSettings()
    control = base if s.control is None else None if s.control == NO_CONTROL else s.control
    return Arm(**{**BUILTINS[base].to_json(), "id": name, "control": control, "execution": label, **_overrides(s)})


def protocols_config(arm: Arm, environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """The arm's protocols config as `verity_vllm.LLM` takes it: each ``"$NAME"`` value read from `environ` (a missing
    one raises, naming it), and PoUW's ``coins`` an in-process challenger of epoch 0 (`pouw.challenged`) when PoUW is
    enabled, since its install takes the epoch's salt from a live coin and a config can't carry one."""

    def expand(value: Any, where: str) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            if value[1:] not in environ:
                raise ValueError(f"arm {arm.id!r}: {where} reads ${value[1:]}, which the point's environment doesn't set "
                                 "(the spec's setup exports it)")
            return environ[value[1:]]
        return value

    config = {p: {k: expand(v, f"{p}.{k}") for k, v in s.items()} for p, s in (arm.protocols_config or {}).items()}
    if not any(e.partition(":")[0] == "pouw" for e in arm.protocols):
        return config
    from verity_vllm.protocols.pouw import challenged

    return challenged(config, arm.protocols)


def resolve_arms(names: str | list[str] | tuple[str, ...], module_specs: tuple[str, ...] = (),
                 settings: Mapping[str, ArmSettings] | None = None) -> tuple[Arm, ...]:
    """Resolve the comma-separated arm ids, replacing matching ids with experimental modules; a protocol arm takes its
    `ArmSettings` from `settings`, by id."""

    requested = [x.strip() for x in (names.split(",") if isinstance(names, str) else names) if x.strip()]
    custom = {a.id: a for a in map(experimental_arm, module_specs)}
    duplicate_modules = len(custom) != len(module_specs)
    if duplicate_modules:
        raise ValueError("two --arm-module values expose the same arm id")
    settings = settings or {}
    stray = sorted(set(settings) - set(requested))
    if stray:
        raise ValueError(f"arm settings name arms not in --arms: {stray}")
    out = [custom.get(name) or BUILTINS.get(name)
           or (served_variant if "@" in name else _protocol_arm)(name, settings.get(name)) for name in requested]
    if any(name in settings and (name in custom or name in BUILTINS) for name in requested):
        raise ValueError("arm settings are a protocol arm's or a served variant's; built-in and experimental arms carry "
                         "their own")
    if len({a.id for a in out}) != len(out):
        raise ValueError("benchmark arm ids must be unique")
    unused = sorted(set(custom) - {a.id for a in out})
    if unused:
        raise ValueError(f"--arm-module defines arms not named by --arms: {unused}")
    ids = {a.id for a in out}
    missing = sorted({a.control for a in out if a.control is not None} - ids)
    if missing:
        raise ValueError(f"benchmark controls must also be named by --arms: {missing}")
    return tuple(out)
