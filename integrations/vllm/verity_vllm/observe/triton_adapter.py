"""Direct Triton launches, observed at `triton.runtime.jit.JITFunction.run` (pinned Triton 3.7.1).

Triton 3.7.1's `run(self, *args, grid, warmup, **kwargs)` binds the call with a per-device *binder*
(`kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]`;
`bound_args, specialization, options = binder(*args, **kwargs)`): `bound_args` is the `{name: value}` dict of
**every** kernel parameter (constexprs included; tensors are still the torch tensors, Triton takes `data_ptr()`
only inside the launcher), `options` are the launch options passed by the caller (`num_warps`, `num_stages`, ...),
and a callable `grid` is evaluated as `grid(bound_args)`.  The wrapper installed here calls the same binder on the
same arguments (a pure, memoized function), runs the original `run`, and records `TritonLaunch(id, parent = the
innermost open dispatcher call or None, step, phase, fn, src_sha, args, constexprs, grid, num_warps, num_stages,
stream)`.  Warm-up compilations (`warmup=True`) are not launches and are not recorded.  Nothing about the launch is
changed: same arguments, same grid object, same kernel object returned.

Fallback (not needed on the pinned wheel): `triton.knobs.runtime.launch_enter_hook` + `JITFunction.pre_run_hooks`.
"""

from __future__ import annotations

import hashlib
from typing import Any

from verity_vllm.engine import hooks

from .events import TritonLaunch
from .observer import Recorder, encode_arg

_STATE: dict[str, Any] = {"orig_run": None, "rec": None, "patch": None}
_SRC_SHA: dict[int, str] = {}
_CONSTEXPR_NAMES: dict[int, frozenset[str]] = {}


def install(rec: Recorder | None) -> None:
    """Wrap `JITFunction.run` once (idempotent) and bind the recorder; `rec=None` keeps the wrapper but disables
    recording (pass-through)."""
    if _STATE["patch"] is None:
        _STATE["patch"] = hooks.triton_launch(lambda orig: _run)
        _STATE["orig_run"] = _STATE["patch"].original
    _STATE["rec"] = rec


def set_recorder(rec: Recorder | None) -> None:
    _STATE["rec"] = rec


def uninstall() -> None:
    if _STATE["patch"] is not None:
        _STATE["patch"].uninstall()
        _STATE["patch"] = None
    _STATE["rec"] = None


def is_installed() -> bool:
    return _STATE["patch"] is not None


def src_sha(jit_fn: Any) -> str:
    k = id(jit_fn)
    s = _SRC_SHA.get(k)
    if s is None:
        try:
            src = jit_fn.src
        except AttributeError:
            src = ""
        s = hashlib.sha256(src.encode("utf-8")).hexdigest()
        _SRC_SHA[k] = s
    return s


def constexpr_names(jit_fn: Any) -> frozenset[str]:
    k = id(jit_fn)
    names = _CONSTEXPR_NAMES.get(k)
    if names is None:
        try:
            names = frozenset(p.name for p in jit_fn.params if p.is_constexpr)
        except (AttributeError, TypeError):
            names = frozenset()
        _CONSTEXPR_NAMES[k] = names
    return names


def fn_name(jit_fn: Any) -> str:
    f = jit_fn.fn
    return f"{f.__module__}.{f.__qualname__}"


def _run(self, *args, grid, warmup, **kwargs):
    rec: Recorder | None = _STATE["rec"]
    orig = _STATE["orig_run"]
    if rec is None or not rec.active:
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs)
    bound: dict | None = None
    options: dict = {}
    try:
        from triton.runtime.driver import driver

        device = driver.active.get_current_device()
        binder = self.device_caches[device][-1]
        bound, _specialization, options = binder(*args, **kwargs)
        bound = dict(bound)
        options = dict(options) if isinstance(options, dict) else {}
    except Exception as e:  # noqa: BLE001
        rec.note("triton binder failed; recording positional arguments", fn=fn_name(self), error=repr(e)[:300])
    kernel = orig(self, *args, grid=grid, warmup=warmup, **kwargs)
    if warmup:
        return kernel
    try:
        _record(self, rec, kernel, bound, options, args, grid)
    except Exception as e:  # noqa: BLE001
        rec.note("triton launch record failed", fn=fn_name(self), error=repr(e)[:300])
    return kernel


def _record(jit_fn: Any, rec: Recorder, kernel: Any, bound: dict | None, options: dict, args: tuple, grid: Any) -> None:
    eid = rec.new_id()
    if bound is not None:
        args_enc = {k: encode_arg(v, rec.storage) for k, v in bound.items()}
        g = grid(bound) if callable(grid) else grid
    else:
        args_enc = {f"arg{i}": encode_arg(a, rec.storage) for i, a in enumerate(args)}
        g = grid if not callable(grid) else None
    if g is None:
        grid_list: list[int] = []
    elif isinstance(g, (list, tuple)):
        grid_list = [int(v) for v in g]
    else:
        grid_list = [int(g)]
    cnames = constexpr_names(jit_fn)
    constexprs = {k: v for k, v in args_enc.items() if k in cnames}
    md = getattr(kernel, "metadata", None)
    num_warps = options.get("num_warps", getattr(md, "num_warps", None))
    num_stages = options.get("num_stages", getattr(md, "num_stages", None))
    rec.counts["triton"] += 1
    rec.emit(TritonLaunch(eid, rec.parent(), rec.step, rec.phase, fn_name(jit_fn), src_sha(jit_fn), args_enc, constexprs,
                          grid_list, None if num_warps is None else int(num_warps),
                          None if num_stages is None else int(num_stages), rec.stream(), rec.module_hint()))
    if rec.after_launch is not None:
        rec.after_launch(eid, fn_name(jit_fn), bound if bound is not None else {f"arg{i}": a for i, a in enumerate(args)})
