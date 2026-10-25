"""vLLM benchmark arms: stock engines, protocol options, and hook modules."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Mapping

from verity_vllm.engine.hooks import Hooks

from verity_bench.vllm.presets import NO_CONTROL, SERVED_EXECUTIONS, ArmSettings


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
    #: a served variant's own checkpoint (`presets.ArmSettings`), `model` at `revision`; None loads the preset's
    model: str | None = None
    revision: str | None = None
    #: an experimental arm's name for the engine its own ``engine_overrides`` build (its ``engine_label``), recorded in
    #: its identity only: it never changes how the harness builds the engine, unlike ``execution``
    engine_label: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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
    # The eager int8_exact engine without a protocol: PoUW H1's token control, since H1's arm serves its own y.
    "stock-int8": Arm("stock-int8", precision="int8_exact"),
}
#: PoUW's H1 + fx1 on the int8_exact engine (`pouw_h1.SCHEME`), through `verity_vllm.LLM`'s ``protocols``: the LLM fills
#: its model runner and in-process coins from the built engine, and the harness gives each run its own window (`h1_config`)
H1 = "pouw:eps1-sm120-h1fx1"
#: H1's pouw settings a spec can't name: the harness supplies ``steps`` for each window and `verity_vllm.LLM` the rest
H1_SUPPLIED = ("coins", "rounds", "runner", "steps", "challenger", "session", "gate")


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
    """The arm's protocol entries that bind the served computation to a row Build (`protocol_options.requires_build`)."""

    if not arm.protocols:
        return ()
    from verity_vllm import protocols as PO

    ctx = PO.Context(stage="serve", enabled=arm.protocols, world=world)
    bound = []
    for entry in arm.protocols:
        try:
            if PO.requires_build(entry, ctx):
                bound.append(entry)
        except PO.Refused:
            continue
    return tuple(bound)


def _protocol_arm(name: str, settings: ArmSettings | None = None) -> Arm:
    """A protocol arm: the preset's precision and stock FP8 as its control, unless its `ArmSettings` say otherwise."""

    entries = tuple(x for x in name.split("+") if x)
    if not entries or any(":" not in e for e in entries):
        raise ValueError(f"unknown benchmark arm {name!r}; use a built-in, protocol:scheme, or --arm-module")
    s = settings or ArmSettings()
    control = "stock-fp8" if s.control is None else None if s.control == NO_CONTROL else s.control
    config = {k: dict(v) for k, v in s.protocols_config.items()} or None
    return Arm(name, protocols=entries, precision=s.precision, control=control, protocols_config=config)


def served_variant(name: str, settings: ArmSettings | None = None) -> Arm:
    """``BUILTIN@LABEL``: the built-in arm's engine at vLLM's serving defaults in execution LABEL (`SERVED_EXECUTIONS`),
    so one spec times one engine in two executions (`stock-fp8` eager against `stock-fp8@compiled`). Its control is the
    built-in, and its checkpoint the preset's, unless its `ArmSettings` say otherwise."""

    base, _, label = name.partition("@")
    if base not in BUILTINS or label not in SERVED_EXECUTIONS:
        raise ValueError(f"served variant {name!r}: expected BUILTIN@LABEL with BUILTIN one of {sorted(BUILTINS)} and "
                         f"LABEL one of {list(SERVED_EXECUTIONS)}")
    s = settings or ArmSettings()
    control = base if s.control is None else None if s.control == NO_CONTROL else s.control
    return Arm(**{**BUILTINS[base].to_json(), "id": name, "control": control, "execution": label, "model": s.model,
                  "revision": s.revision})


def protocols_config(arm: Arm, environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """The arm's protocols config as `verity_vllm.LLM` takes it: each ``"$NAME"`` value read from `environ` (a missing
    one raises, naming it), and PoUW's ``coins`` an in-process challenger of epoch 0 (`pouw.challenged`) when PoUW is
    enabled, since its install draws the epoch's salt from a live coin and a config can't carry one; H1 takes its coins from
    `verity_vllm.LLM` instead, one window each (`h1_config`)."""

    def expand(value: Any, where: str) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            if value[1:] not in environ:
                raise ValueError(f"arm {arm.id!r}: {where} reads ${value[1:]}, which the point's environment doesn't set "
                                 "(the spec's setup exports it)")
            return environ[value[1:]]
        return value

    config = {p: {k: expand(v, f"{p}.{k}") for k, v in s.items()} for p, s in (arm.protocols_config or {}).items()}
    if is_h1(arm) or not any(e.partition(":")[0] == "pouw" for e in arm.protocols):
        return config
    from verity_vllm.protocols import pouw

    return pouw.challenged(config, arm.protocols)


def is_h1(arm: Arm) -> bool:
    return H1 in arm.protocols


def window_steps(point: Any, rows: int) -> int:
    """S for one run of `point` at `rows` tokens a step: an upper bound on the engine steps that run takes. A step that
    schedules fewer than `rows` tokens leaves no request waiting on its prefill, so each such step moves every running
    request on by its whole prefill or one token: at most ceil(tokens / rows) full steps, and at most the longest output,
    the arrival steps and one step a request besides."""

    shapes = point.requests()
    tokens = sum(r.prompt_tokens + r.output_tokens for r in shapes)
    return (-(-tokens // rows) + max(r.output_tokens for r in shapes) + max(r.arrive_step for r in shapes)
            + len(shapes) + 8)


def h1_config(config: Mapping[str, Mapping[str, Any]], steps: int) -> dict[str, dict[str, Any]]:
    """H1's protocols config for one window of `steps` engine steps (S), as `verity_vllm.LLM` and its ``reinstall`` take
    it: only the public parameters. The LLM fills the engine's model runner and the window's coins, held in this process
    (`pouw_h1_settings.from_engine`), which a verifier doesn't trust: the window credits nothing (`pouw_h1.IN_PROCESS`),
    so it measures cost only."""

    pouw = dict(config.get("pouw", {}))
    supplied = sorted(set(pouw) & set(H1_SUPPLIED))
    if supplied:
        raise ValueError(f"{H1}: the pouw settings {supplied} are each window's, which the harness and verity_vllm.LLM "
                         "supply")
    return {**{k: dict(v) for k, v in config.items()}, "pouw": {**pouw, "steps": steps}}


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
