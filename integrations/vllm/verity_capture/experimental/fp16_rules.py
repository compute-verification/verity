"""[r8 lane E-cleanup 2026-09-18 — CANDIDATE, not a gate] fp16 column (adapter fix (d), WINDOW-20260917 M11/M12): what the frozen
checkers cannot say about a served-float16 record, derived from the committed record and clearly labelled.

  1. `dtype_refusal_census(resolution)`  — M11 evidence: the Unsupported resolutions of a record grouped by the pattern guard's
     reason; `all_dtype_refusals` is True iff EVERY refusal is a bf16-guard message (the column is refused, not mis-modelled).
     The fp16 twins (`lane/w-fp16` 8f21264cc, quarantine tree) are NOT merged here (brief: no shared integration merges).
  2. `check_cos_sin_words(C, words, dtype)` — the analytic rotary-table bound check of `analytic.check_cos_sin` generalised to the
     table's own dtype: identical arithmetic (fp32 pipeline error bound), with the rounding / ulp / decoding taken in `dtype`
     ("bfloat16" reproduces `analytic.check_cos_sin` exactly; "float16" is the fp16 root's candidate validator).  The frozen
     `root_policy` keeps failing an fp16 run's `cos_sin_cache` root ("no captured cos_sin_cache.npy": its validator is bf16-only);
     with `m1_capture.dump_cos_sin` now observing the fp16 table as `capture/cos_sin_cache.float16.npy` + `cos_sin_cache.json`,
     `candidate_cos_sin_root(run)` evaluates this check over that observation.
  3. `served_dtype_labels(run)` — M12: the served dtype as each label of the record says it (spec.json, run.json engine args, the
     workload's engine_args_required, the profile manifest) and whether they agree.

    python -m verity_capture.experimental.fp16_rules --run-dir RUN_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from verity_capture import analytic as A

BF16_GUARD_MARKERS = ("is bf16", "bf16 →", "bf16 [", "bf16 only", "row-major bf16", "over row-contiguous bf16", "the bf16 kernel")
DTYPES = {"bfloat16": {"bits": 16, "mant": 7}, "float16": {"bits": 16, "mant": 10}}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


# --- 1. M11: the refusal census -------------------------------------------------------------------------------------------------

def dtype_refusal_census(resolution: dict) -> dict:
    """Unsupported resolutions grouped by (op, reason); `all_dtype_refusals` iff every reason is a bf16-guard message."""
    ex = resolution.get("unsupported_examples") or []
    ex = list(ex.values()) if isinstance(ex, dict) else list(ex)
    groups: Counter = Counter()
    for e in ex:
        groups[(str(e.get("op") or e.get("pattern")), str(e.get("why") or e.get("reason") or ""))] += 1
    by_reason = [{"op": op, "reason": why, "examples": n, "bf16_guard": any(m in why for m in BF16_GUARD_MARKERS)}
                 for (op, why), n in groups.most_common()]
    non_guard = [r for r in by_reason if not r["bf16_guard"]]
    return {"total_unsupported": int(((resolution.get("totals") or {}).get("unsupported")) or 0), "distinct_reasons": len(by_reason),
            "by_reason": by_reason, "all_dtype_refusals": bool(by_reason) and not non_guard, "non_guard_reasons": non_guard}


# --- 2. the analytic table in the table's own dtype -------------------------------------------------------------------------------

def f32_to_words(x: np.ndarray, dtype: str) -> np.ndarray:
    """IEEE round-to-nearest-even of fp32 values to `dtype`, as uint16 words."""
    if dtype == "bfloat16":
        return A.f32_to_bf16_words(x)
    if dtype == "float16":
        return np.ascontiguousarray(x, dtype=np.float32).astype(np.float16).view(np.uint16)     # numpy: RNE
    raise ValueError(dtype)


def words_to_f32(w: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return A.bf16_words_to_f32(w)
    if dtype == "float16":
        return np.ascontiguousarray(w, dtype=np.uint16).view(np.float16).astype(np.float32)
    raise ValueError(dtype)


def _monotone(w: np.ndarray) -> np.ndarray:
    """sign-magnitude 16-bit words -> integers monotone in the value (|Δ| = distance in ulps of that format)."""
    w = w.astype(np.int64)
    return np.where((w & 0x8000) != 0, -(w & 0x7FFF), w)


def _half_ulp(values_f32: np.ndarray, dtype: str) -> np.ndarray:
    """half an ulp of `dtype` at each value (float64)."""
    if dtype == "bfloat16":
        return 0.5 * np.abs(np.spacing(values_f32)).astype(np.float64) * 65536.0            # as analytic.check_cos_sin
    return 0.5 * np.abs(np.spacing(values_f32.astype(np.float16))).astype(np.float64)


def check_cos_sin_words(C: dict[str, Any], captured: str | Path | np.ndarray, dtype: str, *, rows: int = 4096) -> dict[str, Any]:
    """`analytic.check_cos_sin` with the rounding / ulp / decoding taken in `dtype` (bfloat16 reproduces it; float16 is new)."""
    if dtype not in DTYPES:
        return {"status": "error", "reason": f"no analytic table in dtype {dtype!r}"}
    got = np.load(captured) if not isinstance(captured, (np.ndarray,)) else captured
    out: dict[str, Any] = {"status": "fail", "dtype": dtype, "shape": list(got.shape), "pow_ulps": A.POW_ULPS, "trig_ulps": A.TRIG_ULPS,
                           "candidate": "verity_capture.experimental.fp16_rules.check_cos_sin_words (r8 lane E-cleanup)"}
    D = A.rotary_dim(C)
    n = got.shape[0] if got.ndim == 2 else 0
    try:
        inv = A._inv_freq(C)
    except NotImplementedError as e:
        out.update(status="error", reason=str(e))
        return out
    out["expected_shape"] = [int(C["MAX_POS"]), D]
    out["shape_ok"] = got.ndim == 2 and tuple(got.shape) == (int(C["MAX_POS"]), D) and got.dtype == np.uint16
    if not out["shape_ok"]:
        out["reason"] = f"captured {got.dtype} {got.shape} vs analytic {out['expected_shape']} for D={D} MAX_POS={C['MAX_POS']}"
        return out
    hist = np.zeros(9, dtype=np.int64)
    n_diff = n_outside = 0
    max_ulp_seen = 0
    max_excess = 0.0
    offending: list[dict] = []
    ulp_inv = np.spacing(inv).astype(np.float64)
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        t = np.arange(lo, hi, dtype=np.float32)
        freqs = (t[:, None] * inv[None, :]).astype(np.float32)
        cos = np.cos(freqs, dtype=np.float32)
        sin = np.sin(freqs, dtype=np.float32)
        exp_words = np.concatenate([f32_to_words(cos, dtype), f32_to_words(sin, dtype)], axis=1)
        g = got[lo:hi]
        d = np.abs(_monotone(g) - _monotone(exp_words))
        hist += np.bincount(np.minimum(d, 8).reshape(-1), minlength=9)
        n_diff += int((d > 0).sum())
        max_ulp_seen = max(max_ulp_seen, int(d.max()) if d.size else 0)
        darg = (t.astype(np.float64)[:, None] * (A.POW_ULPS * ulp_inv)[None, :]) + np.spacing(freqs).astype(np.float64)
        darg = np.concatenate([darg, darg], axis=1)
        exact = np.concatenate([cos, sin], axis=1).astype(np.float64)
        slope = np.concatenate([np.abs(sin), np.abs(cos)], axis=1).astype(np.float64)
        f32v = np.concatenate([cos, sin], axis=1)
        bound = slope * darg + 0.5 * darg ** 2 + A.TRIG_ULPS * np.abs(np.spacing(f32v)).astype(np.float64)
        gv32 = words_to_f32(g, dtype)
        excess = np.abs(gv32.astype(np.float64) - exact) - (bound + _half_ulp(gv32, dtype))
        bad = excess > 0
        n_outside += int(bad.sum())
        if bad.any():
            max_excess = max(max_excess, float(excess[bad].max()))
            for p, c in np.argwhere(bad)[: max(0, 8 - len(offending))]:
                offending.append({"pos": int(lo + p), "col": int(c), "got": int(g[p, c]), "expected": int(exp_words[p, c]),
                                  "ulp": int(d[p, c]), "excess": float(excess[p, c])})
    out.update(ulp_histogram={str(i) if i < 8 else "8+": int(c) for i, c in enumerate(hist) if c}, n_diff=n_diff, n_elements=int(n * D),
               max_ulp=max_ulp_seen, n_outside_bound=n_outside, max_excess=max_excess, first_offending=offending,
               status="pass" if n_outside == 0 else "fail")
    return out


def candidate_cos_sin_root(run: Path, C: dict[str, Any]) -> dict:
    """The fp16 `cos_sin_cache` root over the dtype-qualified observation (`capture/cos_sin_cache.json` sidecar), if present."""
    side = run / "capture" / "cos_sin_cache.json"
    if not side.exists():
        return {"status": "not-observed", "reason": "no capture/cos_sin_cache.json sidecar (record predates the dtype-qualified dump, or no rotary buffer)"}
    meta = _read(side)
    p = run / "capture" / str(meta.get("file") or "cos_sin_cache.npy")
    if not p.exists():
        return {"status": "not-observed", "reason": f"{p.name} not in this (compact) record", "sidecar": meta}
    chk = check_cos_sin_words(C, p, str(meta.get("dtype")))
    return {"status": chk["status"], "sidecar": meta, "check": chk}


# --- 3. M12: the served-dtype labels ---------------------------------------------------------------------------------------------

def served_dtype_labels(run: Path) -> dict:
    labels: dict[str, Any] = {}
    spec = _read(run / "spec.json") if (run / "spec.json").exists() else {}
    labels["spec.dtype"] = spec.get("dtype")
    rj = _read(run / "run.json") if (run / "run.json").exists() else {}
    labels["run.engine_args_requested.dtype"] = ((rj.get("engine_args_requested") or {}) if isinstance(rj.get("engine_args_requested"), dict) else {}).get("dtype")
    labels["workload.engine_args_required.dtype"] = ((rj.get("workload") or {}).get("engine_args_required") or {}).get("dtype")
    vers = _read(run / "capture" / "versions.json") if (run / "capture" / "versions.json").exists() else {}
    labels["versions.engine_kwargs.dtype"] = ((vers.get("engine_kwargs") or {}) if isinstance(vers, dict) else {}).get("dtype")
    man = _log_header_profile(run / "capture" / "log.jsonl.gz")
    labels["manifest.dtype"] = man.get("dtype") if isinstance(man, dict) else None
    labels["manifest.engine_kwargs.dtype"] = ((man.get("engine_kwargs") or {}) if isinstance(man, dict) else {}).get("dtype")
    labels["hf_config.torch_dtype"] = ((spec.get("hf_config") or {}) if isinstance(spec.get("hf_config"), dict) else {}).get("torch_dtype")
    served = (labels["workload.engine_args_required.dtype"] or labels["run.engine_args_requested.dtype"]
              or labels["versions.engine_kwargs.dtype"])
    stated_keys = ("spec.dtype", "manifest.dtype", "manifest.engine_kwargs.dtype")
    stated = {k: labels[k] for k in stated_keys if labels.get(k) is not None}
    return {"labels": labels, "served_dtype_requested": served,
            "consistent": (served is None) or all(v == served for v in stated.values()),
            "mislabelled": [k for k, v in stated.items() if served is not None and v != served],
            "unlabelled": [k for k in stated_keys if labels.get(k) is None]}


def _log_header_profile(log: Path) -> dict | None:
    """The profile manifest from the capture log's RunHeader (first line), without the events module (torch-free, cheap)."""
    if not log.exists():
        return None
    import gzip
    try:
        with gzip.open(log, "rt") as f:
            h = json.loads(f.readline())
        return h.get("profile") if isinstance(h, dict) else None
    except Exception:  # noqa: BLE001 - informational label
        return None


def report(run_dir: str | Path, out_dir: str | Path) -> dict:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = _read(run / "resolution.json") if (run / "resolution.json").exists() else {}
    census = dtype_refusal_census(res)
    labels = served_dtype_labels(run)
    C = ((_read(run / "run.json") if (run / "run.json").exists() else {}).get("C")) or {}
    root = candidate_cos_sin_root(run, C) if C else {"status": "not-observed", "reason": "run.json has no C"}
    gates = _read(run / "gates.json") if (run / "gates.json").exists() else {}
    doc = {"schema": "verity-capture/fp16-candidate-report/v1", "utc": _utc(), "run": run.name, "recorded_verdict": gates.get("verdict"),
           "recorded_gates": {k: v.get("status") for k, v in (gates.get("gates") or {}).items()},
           "M11_refusal_census": census, "M12_served_dtype_labels": labels, "cos_sin_root_candidate": root,
           "note": "CANDIDATE report — not a verdict; the fp16 twins are not merged, so no fp16 cell can pass G2 here."}
    (out / "fp16.candidate.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = [f"# fp16 CANDIDATE report — `{run.name}`", "", f"recorded verdict **{gates.get('verdict')}** {doc['recorded_gates']}", "",
          f"## M11 — refusal census: {census['total_unsupported']} Unsupported, {census['distinct_reasons']} distinct reasons, all bf16-guard refusals: **{census['all_dtype_refusals']}**", "",
          "| op | reason | examples | bf16 guard |", "| --- | --- | --- | --- |"]
    for r in census["by_reason"]:
        md.append(f"| `{r['op']}` | {r['reason'].replace('|', '/')} | {r['examples']} | {r['bf16_guard']} |")
    md += ["", f"## M12 — served-dtype labels: requested **{labels['served_dtype_requested']}**, consistent: **{labels['consistent']}**, mislabelled: {labels['mislabelled']}, unlabelled: {labels['unlabelled']}", "",
           "~~~json", json.dumps(labels["labels"], indent=1), "~~~", "",
           f"## `cos_sin_cache` root (fp16 analytic table, candidate): **{root['status']}** — {root.get('reason') or root.get('check', {}).get('n_outside_bound')}", ""]
    (out / "CANDIDATE.md").write_text("\n".join(md))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = report(a.run_dir, a.out)
    print(json.dumps({k: doc[k] for k in ("run", "recorded_verdict")} | {"all_dtype_refusals": doc["M11_refusal_census"]["all_dtype_refusals"],
                      "labels": doc["M12_served_dtype_labels"]["labels"], "cos_sin_root": doc["cos_sin_root_candidate"]["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
