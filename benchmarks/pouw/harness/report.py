"""The report's arithmetic, per shape and timing method, from the medians: each baseline family's fastest kernel and its
TFLOPS, and for each arm, over its precision's fastest plain GEMM, the hash-free and total ratios, hashing (total less
hash-free: hashing and XOF, incremental), the hashing rate, the per-epoch weight side with the reuse count R₁% at which it
falls under 1% of the baseline call, and the same ratios over every other family."""

from __future__ import annotations

import math

#: the families an arm's headline divides by, by its precision: the fastest of them at the shape
HEADLINE_FAMILIES = {"fp8-e4m3": ("fp8-e4m3", "fp8-e4m3-rowwise"), "nvfp4": ("nvfp4",)}
METHODS = ("graph", "burst")


def reuse_under(weight_ms: float, base_ms: float, share: float = 0.01) -> int:
    """The least number of calls per (weight, epoch) at which the weight side, spread over them, costs at most `share` of
    one baseline call: R ≥ weight_ms / (share · base_ms)."""
    if base_ms <= 0:
        raise ValueError("base_ms must be positive")
    return max(1, math.ceil(weight_ms / (share * base_ms) - 1e-9))


def best_family(baselines: dict[str, dict], families: tuple[str, ...], method: str) -> tuple[str, float] | None:
    got = {f: baselines[f]["ms"][method] for f in families if f in baselines and baselines[f]["ms"].get(method)}
    if not got:
        return None
    f = min(got, key=got.get)
    return f, got[f]


def baseline_rows(shape, baselines: dict[str, dict]) -> dict:
    """Per family and method: the fastest kernel (`kernel[method]`, its source and config), its per-call ms and TFLOPS
    (2mkn / time)."""
    out = {}
    for fam, b in baselines.items():
        out[fam] = {m: {"ms": ms, "tflops": 2 * shape.mkn / (ms * 1e-3) / 1e12, "kernel": b["kernel"].get(m),
                        "source": b.get("source", {}).get(m), "config": b.get("config", {}).get(m)}
                    for m, ms in b["ms"].items() if ms}
    return out


def arm_row(shape, precision: str, work, arm_ms: dict[str, dict[str, float]], weight_ms: dict[str, float] | None,
            baselines: dict[str, dict]) -> dict:
    """One arm at one shape. `arm_ms[role][method]` are the medians of its flag sets (`hash`, `nohash`, extras),
    `weight_ms[method]` its weight side's, `baselines[family]["ms"][method]` the families'."""
    fams = HEADLINE_FAMILIES.get(precision, (precision,))
    row: dict = {"precision": precision, "w_ref": work.w_ref, "credited": work.credited, "unit": work.unit,
                 "gamma": work.gamma, "tensor_macs_over_mkn": {p: v / shape.mkn for p, v in work.tensor_macs.items()},
                 "hashed_bytes": work.hashed_bytes, "weight_hashed_bytes": work.weight_hashed_bytes}
    for method in METHODS:
        head = best_family(baselines, fams, method)
        total, free = arm_ms.get("hash", {}).get(method), arm_ms.get("nohash", {}).get(method)
        if head is None or total is None or free is None:
            continue
        fam, base = head
        hashing = total - free
        hb = sum(work.hashed_bytes.values())
        r = {"baseline": fam, "baseline_ms": base, "total_ms": total, "hash_free_ms": free, "hashing_ms": hashing,
             "total_x": total / base, "hash_free_x": free / base, "hashing_x": hashing / base,
             "hashing_bytes_per_s": hb / (hashing * 1e-3) if hashing > 0 and hb else None,
             "extra_roles_x": {role: v[method] / base for role, v in arm_ms.items()
                               if role not in ("hash", "nohash") and v.get(method)},
             "over": {f: {"total_x": total / b["ms"][method], "hash_free_x": free / b["ms"][method]}
                      for f, b in baselines.items() if b["ms"].get(method)}}
        if weight_ms and weight_ms.get(method) is not None:
            w = weight_ms[method]
            r.update({"weight_side_ms": w, "weight_side_x": w / base, "reuse_1pct": reuse_under(w, base)})
        row[method] = r
    return row
