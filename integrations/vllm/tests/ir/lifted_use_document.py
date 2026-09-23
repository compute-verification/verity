"""The LIFTED VU-partition use document of a REAL row's request Program (LIFTING-SPEC v0.5.1 A4.4 / A5 / A7 (iii); revb M-0042 B1's condition):
`Lift(P)` of a `build_request*` dir under the row's DECLARED lag, the default tiling `default(w_max_lifted)` evaluated on it, the untouched
`partition.validate_vu_partition` at `w_max_lifted`, and `L.lifted_vu_use_document` -> `query_lifted.json` + `query_use_lifted.json` carrying
`use.kind = vu-partition-lifted`, `rules_version = lifted-v1` and the `width_summary` {w_max_lifted, max_seen, w_out_lifted_max, n_out
histogram, ...}.  Beside it (information, never the verdict): the same validator at the active-reference 32 on the unlifted P.

    python -m tests.ir.lifted_use_document --build-request <dir> --out <dir> [--workload-program <row>/build_workload/workload_program.json | --lag N]
                                                  [--eos 50279] [--cap 32] [--max-sets 50000]

Lifted-padded, Program-side: no record fact is read except the Build's own declaration (descriptor, params, cap, EOS, lag).  The LAG is
the Build's DECLARED value through the one rule (`L.declared_lag_of_row`); `--lag` only for a synthetic source, and it must agree with a
row that declares one.  Nothing is inferred from a trace."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from verity_vllm.lifting import record as R
from verity_vllm.ir import partition as PT
from verity_vllm.ir import workload as W
from verity.ir.defs import REGISTRY
from verity_vllm.registry import lifted as L

SCHEMA = "verity/pad/lifted-use-document/v1"


def _params(build_request: Path):
    ip = build_request / "instances.json.gz"
    if not ip.exists():
        return None
    with gzip.open(ip, "rt") as f:
        return [(str(n), str(t)) for n, t in (json.load(f).get("params") or [])] or None


def _declared(build_request: Path, workload_program: str | None, lag_arg: int | None, eos_arg: str | None, cap_arg: int | None):
    """cap / EOS / lag from the row's own declaration: `result.json` (LP, T) and `workload_program.json` (stop_rule, lag static)."""
    wp = json.loads(Path(workload_program).read_text()) if workload_program else None
    lag_row, field = L.declared_lag_of_row(wp) if wp is not None else (None, None)
    if lag_arg is not None and lag_row is not None and int(lag_arg) != int(lag_row):
        raise SystemExit(f"--lag {lag_arg} disagrees with the row's declared lag {lag_row} ({field}); the row's declaration wins, drop --lag")
    lag = lag_row if lag_row is not None else lag_arg
    if lag is None:
        raise SystemExit("no declared LAG: give --workload-program <row>/build_workload/workload_program.json (the Build's declaration) or --lag for a synthetic source")
    eos = [int(x) for x in eos_arg.split(",")] if eos_arg else list((wp or {}).get("stop_rule", {}).get("eos_token_ids") or [])
    if not eos:
        raise SystemExit("no EOS set: --eos or the row's stop_rule.eos_token_ids")
    cap = cap_arg
    if cap is None:
        rj = build_request / "result.json"
        if rj.exists():
            res = json.loads(rj.read_text())  # cb-a/derive_step/v3: top-level LP, T
            t = res.get("T")
            if t is not None:
                cap = int(t) + 1  # the Build's (LP, T): T decode steps after the prefill event -> cap = T + 1 sampling events
        if cap is None and wp is not None:
            caps = {int(r["cap"]) for r in wp.get("requests", [])}
            if len(caps) == 1:
                cap = caps.pop()
    if cap is None:
        raise SystemExit("no cap: --cap or the Build's result.json (LP, T)")
    return int(lag), (field or ("--lag" if lag_arg is not None else None)), eos, int(cap), wp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload-program", default=None)
    ap.add_argument("--lag", type=int, default=None)
    ap.add_argument("--eos", default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--boundary-prefix", default=L.BOUNDARY_PREFIX)
    ap.add_argument("--max-sets", type=int, default=L.N_OUT_HISTOGRAM_MAX_SETS)
    ap.add_argument("--no-reference", action="store_true", help="skip the active-reference validator at 32 on the unlifted P")
    a = ap.parse_args(argv)
    br, out = Path(a.build_request), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lag, lag_field, eos, cap, wp = _declared(br, a.workload_program, a.lag, a.eos, a.cap)
    desc = R.load_descriptor(br)
    params = _params(br)
    comp = W.request_component(desc, REGISTRY, params)
    lifted = L.lift_request_component(comp.fn, eos, max_tokens=cap, lag=lag, boundary_prefix=a.boundary_prefix)
    P_l = L.as_program(lifted)
    t1 = time.time()
    q = f"default({L.W_MAX_LIFTED})"
    v = PT.validate_vu_partition(P_l, q, L.W_MAX_LIFTED)
    t2 = time.time()
    qdoc, use = L.lifted_vu_use_document(q, P_l, v, histogram_max_sets=a.max_sets)
    t3 = time.time()
    ref = None
    if not a.no_reference:
        P_u = L.as_program(comp.fn)  # the same lowering of a body-only specialization the lifted side uses
        v_u = PT.validate_vu_partition(P_u, f"default({L.W_MAX})", L.W_MAX)
        ref = {"query": f"default({L.W_MAX})", "w_max": L.W_MAX, "status": v_u.status, "count": v_u.count,
               "width": {"status": v_u.width.status, "max_seen": v_u.width.max_seen, "violators": len(v_u.width.violators), "unsupported": len(v_u.width.unsupported)},
               "note": "the ACTIVE-REFERENCE validator on the UNLIFTED P (information; not a lifted verdict, not mixed with it)"}
    (out / "query_lifted.json").write_text(json.dumps(qdoc, indent=1, sort_keys=True))
    (out / "query_use_lifted.json").write_text(json.dumps(use, indent=1, sort_keys=True))
    ws = use["width_summary"]
    rep = {"schema": SCHEMA, "label": L.LABEL, "rules_version": L.LIFTED_RULES_VERSION, "spec": "LIFTING-SPEC v0.5.1 A4.4/A5/A7(iii)",
           "build_request": str(br), "program_digest": comp.program_digest, "lifted_program_digest": R.lifted_program_digest(lifted),
           "lag": lag, "lag_declared_field": lag_field, "eos": eos, "cap": cap,
           "workload_digest": (wp or {}).get("workload_digest"),
           "query": q, "use_kind": use["use"]["kind"], "verdict_status": v.status, "ok": v.status == "ok",
           "width_summary": ws, "sets": v.count, "partition_status": v.partition.status, "width_status": v.width.status,
           "active_reference_at_32_on_unlifted_P": ref,
           "timing_s": {"lift": round(t1 - t0, 1), "validate": round(t2 - t1, 1), "use_document": round(t3 - t2, 1)},
           "files": {"query": str(out / "query_lifted.json"), "use": str(out / "query_use_lifted.json")}}
    (out / "lifted_use_document.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("ok", "verdict_status", "lag", "cap", "use_kind", "timing_s")} |
                     {k: ws.get(k) for k in ("w_max_lifted", "max_seen", "w_out_lifted_max", "n_out_max", "growth_max", "w_out_payload_max", "complete", "sets_visited")}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
