"""Benchmark arms: stock engines, protocol-option sets, and experimental hook modules."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Mapping

from verity_vllm.engine.hooks import Hooks


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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Target:
    """What an experimental arm may inspect and patch.

    Every patch belongs to ``hooks``. The runner checks this after ``install``
    and always uninstalls the set when the engine closes.
    """

    arm: Arm
    engine: Any
    model: Any
    hooks: Hooks


BUILTINS = {
    "stock": Arm("stock", precision="bfloat16"),
    # Different precisions need not generate identical greedy tokens.  The
    # stock arms are independent references; protocol arms use stock FP8 as
    # their bit-exact control.
    "stock-fp8": Arm("stock-fp8", precision="fp8"),
}


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
    return Arm(
        id=str(name),
        protocols=_protocols(get("protocols")),
        precision=get("precision"),
        control=get("control"),
        module_spec=spec,
        kernel_digest=kernel_digest if isinstance(kernel_digest, str) else None,
        experimental=True,
        resident=bool(get("resident", True)),
    )


def plugin(arm: Arm) -> Any | None:
    return None if arm.module_spec is None else load_object(arm.module_spec)


def _protocol_arm(name: str) -> Arm:
    entries = tuple(x for x in name.split("+") if x)
    if not entries or any(":" not in e for e in entries):
        raise ValueError(f"unknown benchmark arm {name!r}; use a built-in, protocol:scheme, or --arm-module")
    return Arm(name, protocols=entries, control="stock-fp8")


def resolve_arms(names: str | list[str] | tuple[str, ...], module_specs: tuple[str, ...] = ()) -> tuple[Arm, ...]:
    """Resolve the comma-separated arm ids, replacing matching ids with experimental modules."""

    requested = [x.strip() for x in (names.split(",") if isinstance(names, str) else names) if x.strip()]
    custom = {a.id: a for a in map(experimental_arm, module_specs)}
    duplicate_modules = len(custom) != len(module_specs)
    if duplicate_modules:
        raise ValueError("two --arm-module values expose the same arm id")
    out = [custom.get(name) or BUILTINS.get(name) or _protocol_arm(name) for name in requested]
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
