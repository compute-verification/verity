"""A module's torch forward hooks as entries of a source's `verity_vllm.engine.hooks.Hooks`.

A source registers its forward hooks where it installs its patches.  Each is a `Service` of the source's ``Hooks``: while
it is on, `verity_vllm.engine.hooks.live` lists it beside the patches, and the source's ``Hooks.uninstall`` removes it
with them, newest first.  The hook, its keywords and the order of registration are the source's: torch's
``register_forward_pre_hook`` and ``register_forward_hook`` are called here, at once, with what it passes.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from verity_vllm.engine.hooks import Hooks, Service


def _remove(handle: Any) -> None:
    handle.remove()


def forward_pre_hook(into: Hooks, module: Any, hook: Callable[..., Any], **kwargs: Any) -> Service:
    """``module.register_forward_pre_hook(hook, **kwargs)`` as an entry of ``into``: uninstalling it removes the hook."""
    return into.service(f"forward_pre_hook of {type(module).__qualname__}", partial(module.register_forward_pre_hook, hook, **kwargs), _remove)


def forward_hook(into: Hooks, module: Any, hook: Callable[..., Any], **kwargs: Any) -> Service:
    """``module.register_forward_hook(hook, **kwargs)`` as an entry of ``into``: uninstalling it removes the hook."""
    return into.service(f"forward_hook of {type(module).__qualname__}", partial(module.register_forward_hook, hook, **kwargs), _remove)


__all__ = ["forward_hook", "forward_pre_hook"]
