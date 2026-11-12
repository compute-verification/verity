"""Pin one name a vLLM module or op namespace is called through (`pin_op`): replace it once, before anything calls it, and
put back what was there.

A pin names its ``site``, the ``name`` there, and ``make(original)``, which builds the replacement from what the site
holds:

- a module's global, a function the module's own code calls by name (``moe_align_block_size`` in vLLM's Marlin MoE
  experts module): pinned once the module's own code has run, now if it is imported and else when it is
  (`verity_vllm.engine.hooks.at_import`), so nothing is imported for it. With ``caller``, the pin is refused unless the
  module's function ``caller`` loads ``name`` as its global (`loads_global`): a caller that reaches the function another
  way would bypass the pin.
- an op of a ``torch.ops`` namespace (``site`` ``"torch.ops._C"``): pinned now, so the library that registers the op is
  imported first. The op's overloads (``.default``) stay the op's own, as attributes of the replacement, so code that reads
  them finds them, and a call through one of them bypasses the pin.

A pin is refused while another live patch (`verity_vllm.engine.hooks.live`) replaces the same name at the site, and while
another pin of it is on or armed. It returns a `Pin`, whose ``uninstall`` puts back what the site held, or disarms a pin
whose module hasn't been imported.
"""
from __future__ import annotations

import dis
import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verity_vllm.engine import hooks

__all__ = ["Pin", "loads_global", "pin_op"]

OPS = "torch.ops."


def loads_global(fn: Any, name: str) -> bool:
    """Whether ``fn`` (unwrapped), or a function nested in it, loads ``name`` as its module's global: what replacing the
    module's ``name`` reaches. An attribute of the same name (``ops.name``) is not one."""
    code = getattr(inspect.unwrap(fn), "__code__", None)
    todo = [code] if isinstance(code, types.CodeType) else []
    while todo:
        code = todo.pop()
        if any(i.opname == "LOAD_GLOBAL" and i.argval == name for i in dis.get_instructions(code)):
            return True
        todo += [c for c in code.co_consts if isinstance(c, types.CodeType)]
    return False


@dataclass(eq=False)
class Pin:
    """``site``'s ``name``, replaced through ``hooks`` (one `verity_vllm.engine.hooks.Hooks`, tagged by the pin)."""

    site: str
    name: str
    hooks: hooks.Hooks

    @property
    def installed(self) -> bool:
        """Whether the replacement is in place (False while the pin waits for its module, and once uninstalled)."""
        return any(isinstance(e, hooks.Patch) for e in self.hooks.entries)

    @property
    def original(self) -> Any:
        """What the site held under ``name`` when the pin went in, or `verity_vllm.engine.hooks.ABSENT` before."""
        patches = [e for e in self.hooks.entries if isinstance(e, hooks.Patch)]
        return patches[0].original if patches else hooks.ABSENT

    def uninstall(self) -> None:
        """Put back what the site held, or disarm the pin if its module hasn't been imported. Idempotent."""
        self.hooks.uninstall()


def pin_op(site: str, name: str, make: Callable[[Any], Any], *, caller: str | None = None) -> Pin:
    """Replace ``name`` at ``site`` (a module, or ``"torch.ops.<namespace>"``) with ``make(original)``, refused as the
    module says."""
    tag = f"pin-op:{site}:{name}"
    if hooks.live(tag):
        raise ValueError(f"{site}.{name} is already pinned: uninstall that Pin first")
    pin = Pin(site, name, hooks.Hooks(tag=tag))
    if site.startswith(OPS):
        if caller is not None:
            raise ValueError(f"caller names a function of a module; {site} is an op namespace")
        _place(pin, _namespace(site, name), lambda op: _keeping_overloads(make(op), op))
    else:
        pin.hooks.at_import(site, lambda module: _place(pin, module, make, caller))
    return pin


def _namespace(site: str, name: str) -> Any:
    import torch
    ns = getattr(torch.ops, site.removeprefix(OPS))
    try:
        getattr(ns, name)
    except (AttributeError, RuntimeError) as e:
        raise RuntimeError(f"refused to pin {site}.{name}: no such op is registered; import the library that registers it "
                           "before pinning it") from e
    return ns


def _keeping_overloads(fn: Any, op: Any) -> Any:
    for overload in op.overloads():
        setattr(fn, overload, getattr(op, overload))
    return fn


def _place(pin: Pin, owner: Any, make: Callable[[Any], Any], caller: str | None = None) -> None:
    where = f"{pin.site}.{pin.name}"
    if caller is not None and not loads_global(getattr(owner, caller, None), pin.name):
        raise RuntimeError(f"refused to pin {where}: {pin.site}.{caller} doesn't load {pin.name} as its global, so its calls "
                           "would bypass the pin; pin it where this vLLM calls it")
    if on := [p for p in hooks.live() if isinstance(p, hooks.Patch) and p.owner is owner and p.name == pin.name]:
        raise RuntimeError(f"refused to pin {where} under {on[-1].value!r}, which is already on: pin it before anything "
                           "replaces it")
    pin.hooks.wrap(owner, pin.name, make)
