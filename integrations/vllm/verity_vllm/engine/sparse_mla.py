"""FA3's sparse MLA at one split, the serving switch ``VERITY_SPARSE_MLA_NUM_SPLITS`` (`ENV`).

vLLM's `FlashAttnMLASparseImpl.forward_mqa` (`FLASH_ATTN_MLA_SPARSE`: every decode step and a prefill over more than TOPK keys)
calls FA3 with no ``num_splits``, so FA3's heuristic splits each row's keys by the batch shape and combines the partial rows: a
reduction no Definition states, whose words change with the split count.  `SparseMlaHead_v1` is FA3 at ``num_splits=1``.

The switch is off by default: unset or ``"0"`` serves vLLM's standard setup (the heuristic), ``"1"`` pins one split, and any
other value is refused.  When it is on, `install` wraps the ``flash_attn_varlen_func`` that backend module calls, its own module
global, so FA3's dense prefill and every other caller keep vLLM's arguments.  Every process that builds a served engine pins it
before the model's first forward (a CUDA graph captures the call's arguments).
Importing this module `arm`\\ s the pin: it is installed when the backend module is imported, and nothing is imported for it, so a
process that loads this module without serving (a committer, a CPU test) imports no vLLM.  Each tensor-parallel rank imports it
through `engine.rank_worker` when vLLM mixes the extension into its Worker; the driver (`engine.build.build_engine`) arms it
before vLLM builds the model (a forked rank inherits the armed finder), and the meta inventory `install`\\ s it outright.
`num_splits` reads the switch back from the process: `meta_inventory`'s selections, which the capture map's ``target.switches``
holds.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from typing import Any

from verity_vllm.engine import hooks

ENV = "VERITY_SPARSE_MLA_NUM_SPLITS"
SITE = "vllm.v1.attention.backends.mla.flashattn_mla_sparse"
NAME = "flash_attn_varlen_func"
SPLITS = 1

_PIN: hooks.Patch | None = None


def wanted() -> int | None:
    """The split count `ENV` asks for: `SPLITS` for ``"1"``, None (vLLM's heuristic) when unset or ``"0"``."""
    v = os.environ.get(ENV, "0").strip()
    if v not in ("0", str(SPLITS)):
        raise ValueError(f"{ENV}={v!r}: {SPLITS} pins FA3's sparse MLA to one split, 0 leaves vLLM's heuristic; nothing else is modelled")
    return None if v == "0" else SPLITS


def _one_split(original: Any) -> Any:
    def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_splits"] = SPLITS
        return original(*args, **kwargs)

    flash_attn_varlen_func.verity_sparse_mla_splits = SPLITS
    return flash_attn_varlen_func


def install(module: Any = None) -> hooks.Patch | None:
    """Pin `SITE`'s ``flash_attn_varlen_func`` to `SPLITS` (``module``: that module, imported when None), once per process;
    None when the switch is off (`ENV` unset or ``"0"``)."""
    global _PIN
    if wanted() is None:
        return None
    if _PIN is not None and _PIN.installed:
        return _PIN
    if module is None:
        module = importlib.import_module(SITE)
    _PIN = hooks.wrap(module, NAME, _one_split)
    return _PIN


class _OnImport(importlib.abc.MetaPathFinder):
    """Finds ``site`` as the path finder would and runs `install` on the module once its own code has run."""

    def __init__(self, site: str) -> None:
        self.site = site

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name != self.site:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        def exec_module(run: Any) -> Any:
            def exec_then_pin(module: Any) -> None:
                run(module)
                install(module)
            return exec_then_pin

        hooks.wrap(spec.loader, "exec_module", exec_module)
        return spec


def arm(site: str = SITE) -> None:
    """`install` now if ``site`` is imported, else when it is: nothing is imported here.  Once per process; nothing when the
    switch is off."""
    if wanted() is None:
        return
    if site in sys.modules:
        install(sys.modules[site])
    elif not any(isinstance(f, _OnImport) and f.site == site for f in sys.meta_path):
        sys.meta_path.insert(0, _OnImport(site))


def uninstall() -> None:
    """Put the backend's ``flash_attn_varlen_func`` back and disarm the import hook."""
    global _PIN
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _OnImport)]
    if _PIN is not None:
        _PIN.uninstall()
    _PIN = None


def num_splits() -> int | None:
    """The split count this process serves FA3's sparse MLA with: `SPLITS` while the pin is installed, None (vLLM's heuristic)
    otherwise."""
    return SPLITS if _PIN is not None and _PIN.installed else None


arm()

__all__ = ["ENV", "NAME", "SITE", "SPLITS", "arm", "install", "num_splits", "uninstall", "wanted"]
