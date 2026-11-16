"""The band and P2 v1 side by side, from the two `bench-result.json` of one session (`pod.sh IMPL=side-by-side`): encode, the
decode pass and its overhead per `T`, the honest audit's answer times and margin against its limit, and the recompute control's
margin, with each arm's gates.  An arm whose gates failed contributes no number (its measurements are withheld, as
`bench.py` withholds them).

    python benchmarks/pous/side_by_side.py BAND.json P2.json --out SIDE.json [--markdown SIDE.md]

With `RESEARCH_RUN_DIR` set, `result.json` carries both arms' rows, `band_*` and `p2_*`, and validation passes only if both arms did.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "verity-pous/side-by-side/v1"


def _get(doc: dict[str, Any], *path: Any) -> Any:
    cur: Any = doc
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p, cur.get(str(p)) if not isinstance(p, str) else None)
    return cur


def metrics(doc: dict[str, Any]) -> dict[str, Any]:
    """One arm's comparable numbers, `None` where the arm did not measure them or its gates failed."""
    if not doc.get("passed"):
        return {}
    m = doc.get("metrics") or {}
    out = {"encode_GBps": m.get("encode_GBps"), "ewb_GBps": (m.get("ewb") or 0) / 1e9 or None,
           "decode_pass_s": m.get("decode_pass_s")}
    for t in _get(m, "loop", "tokens") or []:
        out[f"decode_overhead_T{t}"] = _get(m, "loop", "overhead", t)
    for q in ("p50", "p99.9", "max"):
        out[f"audit_answer_{q}_s"] = _get(m, "audit", "elapsed_s", q)
    out["audit_limit_s"] = (_get(m, "deadline", "limit_ns") or 0) / 1e9 or None
    out["audit_margin_min_s"] = _get(m, "deadline", "honest_margin_min_s")
    out["recompute_answer_p99.9_s"] = _get(m, "recompute_control", "elapsed_s", "p99.9")
    out["recompute_margin_min_s"] = _get(m, "recompute_control", "margin_s", "min")
    return out


def table(band: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    a, b = metrics(band), metrics(p2)
    rows = []
    for name in list(dict.fromkeys([*a, *b])):
        x, y = a.get(name), b.get(name)
        rows.append({"metric": name, "band": x, "p2": y, "p2_over_band": y / x if x and y is not None else None})
    return {"schema": SCHEMA, "band": {"scheme": band.get("scheme"), "passed": band.get("passed"), "gates":
                                       {g: v.get("passed") for g, v in (band.get("gates") or {}).items()}},
            "p2": {"scheme": p2.get("scheme"), "passed": p2.get("passed"), "gates":
                   {g: v.get("passed") for g, v in (p2.get("gates") or {}).items()}},
            "weight_set": band.get("weight_set"), "hardware": band.get("hardware"), "rows": rows}


def markdown(t: dict[str, Any]) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.4g}"  # noqa: E731
    lines = [f"| metric | band ({t['band']['scheme']}) | P2 ({t['p2']['scheme']}) | P2 / band |", "|---|---|---|---|"]
    lines += [f"| {r['metric']} | {fmt(r['band'])} | {fmt(r['p2'])} | {fmt(r['p2_over_band'])} |" for r in t["rows"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("band")
    ap.add_argument("p2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default="")
    a = ap.parse_args(argv)
    band, p2 = json.loads(Path(a.band).read_text()), json.loads(Path(a.p2).read_text())
    t = table(band, p2)
    Path(a.out).write_text(json.dumps(t, indent=1) + "\n")
    if a.markdown:
        Path(a.markdown).write_text(markdown(t))
    print(markdown(t), flush=True)
    passed = bool(band.get("passed") and p2.get("passed"))
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = [{"name": f"{arm}_{r['metric']}", "value": r[arm], "unit": "s" if r["metric"].endswith("_s") else
                 ("GB/s" if r["metric"].endswith("GBps") else "x")} for r in t["rows"] for arm in ("band", "p2")
                if r[arm] is not None] if passed else []
        rel = os.path.relpath(Path(a.out).resolve(), Path(run_dir).resolve())
        Path(run_dir, "result.json").write_text(json.dumps({
            "schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID"),
            "workload_fingerprint": {"benchmark": SCHEMA, "weight_set": t["weight_set"], "hardware": t["hardware"],
                                     "schemes": [t["band"]["scheme"], t["p2"]["scheme"]]},
            "validation": {"status": "passed" if passed else "failed", "checker": "benchmarks/pous/side_by_side.py",
                           "detail": json.dumps({"band": t["band"]["passed"], "p2": t["p2"]["passed"]}), "evidence": [rel]},
            "measurements": rows, "measurement_files": [rel], "artifacts": [rel]}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
