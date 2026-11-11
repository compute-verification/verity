"""Autotune equivalence of an Inductor kernel's candidate launch configs -- shown on the committed data, in the treatment process.

The rule for compiled kernels: if the autotune choice affects the numerics, it is part of the specialization identity and
must be pinned; if it does not, show it (same outputs bit-for-bit) and record it.  Inductor's pointwise heuristic offers more than one
launch config (XBLOCK x num_warps) and benchmarks them ONCE per process, keeping the winner -- so two processes (the control and the
treatment arm, or the production run and a later one) may execute two different specializations of the SAME generated source (Qwen L40S B8
run 9: `triton_poi_fused_mul_silu_slice_t_1` control XBLOCK=512/8 warps vs treatment XBLOCK=1024/4 warps; the deny-list identity then says
DIFFERS, correctly: the SASS differs).

What this module shows, per kernel with >1 candidate config, on a sampled launch's COMMITTED inputs: the kernel's OWN compiled source
(Inductor's `CachingAutotuner._precompile_config(cfg)` -> the same compile path and options as the executed launcher) launched under EVERY
candidate config the heuristic offers reproduces the COMMITTED outputs bit for bit.  The executed config is replayed too (the control of the
method: if the argument derivation were wrong it would not reproduce the committed output and the launch is 'not shown', never 'shown').

Scope (fail closed): the launch's scalar arguments are derived from the committed outputs only where the derivation is unambiguous --
pointwise kernels with one output (`xnumel` = its element count) and reductions (`xnumel` = rows, `r0_numel` = row width); anything else
(`ks*`, `ynumel`, multi-output pointwise) is recorded as not derivable and the merge keeps refusing for such a kernel."""
from __future__ import annotations

import re
from typing import Any

_AUT: dict[str, Any] | None = None
_SCAN_ERROR: str | None = None
_LAUNCHERS: dict[tuple[str, str], Any] = {}
_CANDS: dict[str, tuple[list[Any] | None, str | None]] = {}
MAX_LAUNCHES_PER_KERNEL = 8
NO_AUTOTUNER = "no autotuner bound to this kernel in the process"


def cfg_dict(c: Any) -> dict[str, Any]:
    return {"kwargs": {str(a): int(b) for a, b in (getattr(c, "kwargs", None) or {}).items() if isinstance(b, int)},
            "num_warps": int(getattr(c, "num_warps", 0) or 0), "num_stages": int(getattr(c, "num_stages", 0) or 0)}


def cfg_label(d: dict[str, Any]) -> str:
    kw = d.get("kwargs") or {}
    return " ".join(f"{k}={v}" for k, v in sorted(kw.items())) + f" warps={d.get('num_warps')} stages={d.get('num_stages')}"


def _scan() -> dict[str, Any]:
    """This process's autotuners by kernel name; when Inductor's can't be imported, none, with the cause in ``_SCAN_ERROR``."""
    global _SCAN_ERROR
    out: dict[str, Any] = {}
    try:
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception as e:  # noqa: BLE001
        _SCAN_ERROR = f"{NO_AUTOTUNER}: Inductor's autotuner is not importable ({e!r})"[:200]
        return out
    _SCAN_ERROR = None
    seen: set[int] = set()
    for src_name in ("modules", "modules_no_attr", "cache"):
        coll = getattr(PyCodeCache, src_name, None)
        items = list(coll.values()) if isinstance(coll, dict) else list(coll or [])
        for m in items:
            if id(m) in seen:
                continue
            seen.add(id(m))
            for name, obj in list(vars(m).items()):
                if isinstance(obj, CachingAutotuner):
                    kname = getattr(getattr(obj, "fn", None), "__name__", None) or name
                    out.setdefault(kname, obj)
    return out


def autotuner(kernel: str) -> Any | None:
    """The CachingAutotuner bound to `kernel` in THIS process (the object whose launchers the compiled pieces / CUDA graphs execute)."""
    global _AUT
    if _AUT is None or kernel not in _AUT:
        _AUT = _scan()
    return _AUT.get(kernel)


def _decorator_hints(aut: Any) -> tuple[Any, int]:
    """(tile_hint, min_elem_per_thread) parsed from the generated module's @triton_heuristics.pointwise(...) text (None/0 when the
    decorator names none).  Raises when the module can't be read: defaults in place of its hints could offer other candidates."""
    from torch._inductor.runtime.hints import TileHint

    tile_hint, mept = None, 0
    path = aut.fn.fn.__code__.co_filename
    with open(path) as f:
        text = f.read()
    name = aut.fn.__name__
    i = text.find(f"def {name}(")
    deco = text[max(0, text.rfind("@triton_heuristics.", 0, i)):i] if i >= 0 else ""
    m = re.search(r"tile_hint=TileHint\.(\w+)", deco)
    if m:
        tile_hint = getattr(TileHint, m.group(1), None)
    m = re.search(r"min_elem_per_thread=(\d+)", deco)
    if m:
        mept = int(m.group(1))
    return tile_hint, mept


def candidates(kernel: str) -> tuple[list[Any] | None, str | None]:
    """Every launch config the kernel's heuristic offers in this process (Inductor's own `pointwise/reduction/persistent_reduction(...,
    return_configs=True)` with the autotuner's recorded size_hints / triton_meta / inductor_meta), with None; or None and why the
    set is not reconstructable."""
    if kernel in _CANDS:
        return _CANDS[kernel]
    aut = autotuner(kernel)
    res: list[Any] | None = None
    why: str | None = None
    try:
        from torch._inductor.runtime import triton_heuristics as th
        from torch._inductor.runtime.hints import HeuristicType

        if aut is None:
            why = _SCAN_ERROR or NO_AUTOTUNER
        else:
            ht = getattr(aut, "heuristic_type", None)
            im, tm, sh = dict(aut.inductor_meta or {}), dict(aut.triton_meta or {}), aut.size_hints
            if getattr(aut, "device_props", None) is not None:
                tm["device"] = aut.device_props          # the constructor normalised triton_meta['device'] to an index; the heuristics read DeviceProperties
            if ht == HeuristicType.POINTWISE:
                tile_hint, mept = _decorator_hints(aut)
                if tile_hint is None and set(sh or {}) - {"x"}:
                    why = "2-D pointwise kernel without a tile hint: the hint drives the candidate set"
                else:
                    res = list(th.pointwise(sh, tm, tile_hint=tile_hint, filename=aut.filename, min_elem_per_thread=mept, inductor_meta=im, return_configs=True))
            elif ht == HeuristicType.REDUCTION:
                res = list(th.reduction(sh, reduction_hint=im.get("reduction_hint", False), triton_meta=tm, filename=aut.filename, inductor_meta=im, return_configs=True))
            elif ht == HeuristicType.PERSISTENT_REDUCTION:
                res = list(th.persistent_reduction(sh, reduction_hint=im.get("reduction_hint", False), triton_meta=tm, filename=aut.filename, inductor_meta=im, return_configs=True))
            else:
                res = [l.config for l in (aut.launchers or []) if getattr(l, "config", None) is not None]   # templates etc.: one fixed config
    except Exception as e:  # noqa: BLE001
        res, why = None, f"reconstructing the heuristic's candidates raised: {e!r}"[:200]
    _CANDS[kernel] = (res, why)
    return res, why


def executed(kernel: str) -> list[dict[str, Any]]:
    aut = autotuner(kernel)
    return [cfg_dict(l.config) for l in (getattr(aut, "launchers", None) or []) if getattr(l, "config", None) is not None] if aut is not None else []


def _launcher(kernel: str, aut: Any, cfg: Any) -> Any:
    key = (kernel, cfg_label(cfg_dict(cfg)))
    if key not in _LAUNCHERS:
        _LAUNCHERS[key] = aut._precompile_config(cfg).make_launcher()
    return _LAUNCHERS[key]


def replay_under_candidates(kernel: str, ins: list, outs: dict[str, Any], roles: dict[int, str], output_roles: set[str]) -> dict[str, Any]:
    """Launch `kernel`'s own compiled source under every candidate config on the launch's committed inputs; compare to the committed
    outputs bit for bit.  Returns a record (never raises): {"shown": bool, "why": ..., "candidates": [...], "executed": [...], "results": [...]}"""
    import torch

    rec: dict[str, Any] = {"kernel": kernel, "shown": False, "why": None, "candidates": [], "executed": executed(kernel), "results": []}
    aut = autotuner(kernel)
    if aut is None:
        rec["why"] = _SCAN_ERROR or NO_AUTOTUNER
        return rec
    cands, why = candidates(kernel)
    rec["heuristic"] = str(getattr(aut, "heuristic_type", None))
    # config independence is STRUCTURAL only for a pointwise kernel (one thread per element: XBLOCK/num_warps set scheduling
    # and masking, not the arithmetic); for a reduction the config sets the summation/fma order, so bit-equality on sampled launches is a
    # property of the data, not of the kernel -- recorded, but the merge lifts a cross-arm config difference only when `structural`
    rec["structural"] = rec["heuristic"].endswith("POINTWISE")
    if cands is None:
        rec["why"] = f"candidate configs not reconstructable: {why}"
        return rec
    rec["candidates"] = [cfg_dict(c) for c in cands]
    if len(cands) <= 1:
        rec["why"] = "one candidate config: no autotune choice"
        return rec
    sig = dict((aut.triton_meta or {}).get("signature") or {})
    names = list(sig)
    out_idx = sorted(i for i, r in roles.items() if r in output_roles)
    outs_by_idx = {i: outs.get(roles[i]) for i in out_idx}
    if any(v is None for v in outs_by_idx.values()) or not out_idx:
        rec["why"] = "committed outputs missing"
        return rec
    first_out = outs_by_idx[out_idx[0]]
    from torch._inductor.runtime.hints import HeuristicType
    ht = getattr(aut, "heuristic_type", None)
    args: list[Any] = []
    bufs: dict[int, Any] = {}
    derived: dict[str, int] = {}
    for i, n in enumerate(names):
        ty = str(sig[n])
        if ty == "constexpr":
            continue
        if ty.startswith("*"):
            if i in outs_by_idx:
                bufs[i] = torch.empty_like(outs_by_idx[i])
                args.append(bufs[i])
            elif i < len(ins) and ins[i] is not None:
                args.append(ins[i])
            else:
                rec["why"] = f"pointer argument {n} (index {i}) has no committed/registered tensor"
                return rec
        elif n == "xnumel":
            if ht == HeuristicType.POINTWISE:
                if len(out_idx) != 1:
                    rec["why"] = "multi-output pointwise kernel: xnumel not derivable from one output"
                    return rec
                derived[n] = int(first_out.numel())
            else:
                derived[n] = int(first_out.reshape(-1, first_out.shape[-1]).shape[0])
            args.append(derived[n])
        elif n == "r0_numel":
            derived[n] = int(first_out.shape[-1])
            args.append(derived[n])
        elif re.fullmatch(r"xnumel_\d+", n) and ht == HeuristicType.POINTWISE:
            # Inductor COMBO kernel (two pointwise sub-kernels fused horizontally: Qwen's RoPE-k + qkv-bias `triton_poi_fused_1`): sub-kernel i
            # iterates its own output's elements -- xnumel_i = numel of the i-th output pointer; the executed-config replay validates it
            j = int(n.rsplit("_", 1)[1])
            if j >= len(out_idx):
                rec["why"] = f"combo sub-kernel {n} has no {j}-th output"
                return rec
            derived[n] = int(outs_by_idx[out_idx[j]].numel())
            args.append(derived[n])
        else:
            rec["why"] = f"scalar argument {n} not derivable"
            return rec
    rec["derived_scalars"] = derived
    stream = torch.cuda.current_stream().cuda_stream
    ex_labels = {cfg_label(d) for d in rec["executed"]}
    all_exact, executed_exact = True, False
    for c in cands:
        d = cfg_dict(c)
        lb = cfg_label(d)
        r: dict[str, Any] = {"config": d, "executed": lb in ex_labels}
        try:
            launcher = _launcher(kernel, aut, c)
            for b in bufs.values():
                b.fill_(float("nan")) if b.is_floating_point() else b.zero_()
            launcher(*args, stream=stream)
            torch.cuda.synchronize()
            nd = [int((bufs[i] != outs_by_idx[i]).sum()) for i in out_idx]
            r["exact"] = all(x == 0 for x in nd)
            r["n_diff"] = nd
        except Exception as e:  # noqa: BLE001
            r["exact"] = False
            r["error"] = repr(e)[:200]
        all_exact = all_exact and bool(r["exact"])
        executed_exact = executed_exact or (r["executed"] and bool(r["exact"]))
        rec["results"].append(r)
    rec["executed_in_candidates"] = all(lb in {cfg_label(cfg_dict(c)) for c in cands} for lb in ex_labels) and bool(ex_labels)
    rec["shown"] = bool(all_exact and executed_exact and rec["executed_in_candidates"])
    rec["why"] = None if rec["shown"] else ("a candidate config does not reproduce the committed output" if executed_exact else
                                             "the EXECUTED config does not reproduce the committed output from the derived arguments -> method not validated on this launch")
    return rec
