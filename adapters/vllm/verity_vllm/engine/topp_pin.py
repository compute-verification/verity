"""vLLM's split top-p pipeline at the registered split count, in every served engine.

vLLM d9105ea80's `_apply_topp_split` (`SITE`) picks its split count from the sampler batch, S = the largest power of two <= 32 with
S * batch <= num_SMs, and the keep set its kernels compute depends on S (`verity.ml.definitions.registry.topp_split`, bit-exact at every S): a
request's batch-mates would pick its nucleus at top_p < 1.  `install` wraps `_apply_topp_split` so every launch runs at
`topp_geometry.registered_splits(num_SMs)` whatever the batch: it hands vLLM's doubling loop num_sm = S * batch, at which the loop
stops at exactly S.  The stats and partials workspaces are sized for 64 rows of 32 splits, so every batch the split pipeline serves
fits.  The collector reads each launch's S back from the kernels' constexpr (`commit.committer.topp_splits`) and the Match holds it
to the registered S (`check.match.sampler_geometry`).

The pin is always on.  Importing this module `arm`\\ s it, as `engine.sparse_mla` arms its switch: it is installed when vLLM's module
is imported, and nothing is imported for it.  The driver (`engine.build.build_engine`) and each tensor-parallel rank
(`pipeline.tp.rank_worker`) import it before vLLM builds the model.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any

from verity_vllm.engine import hooks

SITE = "vllm.v1.sample.ops.topk_topp_triton"
NAME = "_apply_topp_split"

_PIN: hooks.Patch | None = None


def _at_registered_splits(original: Any) -> Any:
    from verity.ml.definitions.registry.topp_geometry import registered_splits

    def _apply_topp_split(logits: Any, k: Any, p: Any, mask_value: float, num_sm: int) -> Any:
        return original(logits, k, p, mask_value, registered_splits(num_sm) * int(logits.shape[0]))

    _apply_topp_split.verity_topp_pin = True
    return _apply_topp_split


def install(module: Any = None) -> hooks.Patch:
    """Pin `SITE`'s `_apply_topp_split` to the registered split count (``module``: that module, imported when None), once per
    process."""
    global _PIN
    if _PIN is not None and _PIN.installed:
        return _PIN
    if module is None:
        module = importlib.import_module(SITE)
    _PIN = hooks.wrap(module, NAME, _at_registered_splits)
    return _PIN


class _OnImport(hooks.OnImport):
    """Runs `install` on ``site`` once its own code has run."""

    def __init__(self, site: str) -> None:
        super().__init__(site, install)


def arm(site: str = SITE) -> None:
    """`install` now if ``site`` is imported, else when it is: nothing is imported here.  Once per process."""
    if site in sys.modules:
        install(sys.modules[site])
    elif not any(isinstance(f, _OnImport) and f.site == site for f in sys.meta_path):
        sys.meta_path.insert(0, _OnImport(site))


def uninstall() -> None:
    """Put vLLM's `_apply_topp_split` back and disarm the import hook."""
    global _PIN
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _OnImport)]
    if _PIN is not None:
        _PIN.uninstall()
    _PIN = None


def pinned() -> bool:
    """Whether this process serves the split pipeline at the registered split count."""
    return _PIN is not None and _PIN.installed


arm()

__all__ = ["NAME", "SITE", "arm", "install", "pinned", "uninstall"]
