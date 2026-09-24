"""Rebuild gate (T2, CPU / meta device): re-derive a recorded row's Programs with the CURRENT frontend under the recorded structural
inputs and compare `program_digest` to the record; report the `correspondence_digest` the new frontend emits beside it.

    python -m verity_vllm.harness.rebuild_digest_gate --row-dir /workspace/rows/<row> --out /workspace/v2corr/<row> \
        --workload workloads/<row>.json [--wrappers step,request] [--python /workspace/venv312/bin/python]

Per wrapper the recorded `build_<wrapper>/result.json` supplies the derive's inputs exactly as Build ran them (model pin, M / pos0 or
LP / T, max_model_len, the declared TargetProfile, the launch context, tp / rank); `derive_step` is run again into `<out>/build_<wrapper>`
and the two `result.json` are compared: `program_digest` equal or not (the gate), `correspondence_digest` (new), Call counts and
descriptor sizes (recorded vs new -- the correspondence lives in the descriptor's annotations, so `descriptor.json.gz` grows while
`program_digest` must not move).  Writes `<out>/gate.json` and prints one line per wrapper; exit 0 iff every compared digest is equal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _rec(row_dir: str, wrapper: str) -> dict | None:
    p = os.path.join(row_dir, f"build_{wrapper}", "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _argv(py: str, rec: dict, wrapper: str, out: str, workload: str | None) -> list[str]:
    pin = rec["model_pin"]
    kind = wrapper.split("_", 1)[0]          # `request_LP1024_T46` (a per-shape request build) is a `request` derive
    argv = [py, "-m", "verity_vllm.harness.derive_step", "--model", pin["model"], "--revision", pin["revision"], "--wrapper", kind,
            "--max-model-len", str(pin["max_model_len"]), "--out", out, "--profile", rec.get("profile") or "b1-eager-v3"]
    if kind == "step":
        argv += ["--M", str(rec["M"]), "--pos0", str(rec["pos0"])]
    else:
        argv += ["--LP", str(rec["LP"]), "--T", str(rec["T"])]
    if rec.get("eos"):
        argv += ["--eos", str(rec["eos"])]
    tp = rec.get("tp") or {}
    if (tp.get("world") or pin.get("tp") or 1) > 1:
        argv += ["--tp", str(tp.get("world") or pin["tp"]), "--rank", str(tp.get("rank") or pin.get("rank") or 0)]
    declared = ((rec.get("target") or {}).get("declared"))
    if declared:
        argv += ["--target", json.dumps(declared, sort_keys=True)]
    if workload:
        argv += ["--workload", workload]
    lc = rec.get("launch_context")
    if lc:
        if lc.get("conflict"):
            argv += ["--launch-max-seqlen-q", "conflict"]
        elif lc.get("max_seqlen_q_by_step"):
            argv += ["--launch-max-seqlen-q", json.dumps({str(k): v for k, v in lc["max_seqlen_q_by_step"].items()})]
    return argv


def _size(p: str) -> int | None:
    return os.path.getsize(p) if os.path.exists(p) else None


def _wrappers(row_dir: str, spec: str) -> list[str]:
    """`step,request` as given, or `all`: every recorded `build_<wrapper>/` of the row (step, request, per-shape `request_LP<n>_T<m>`)."""
    if spec != "all":
        return [x for x in spec.split(",") if x]
    ws = [d[len("build_"):] for d in sorted(os.listdir(row_dir)) if d.startswith("build_") and os.path.exists(os.path.join(row_dir, d, "result.json"))]
    return sorted(ws, key=lambda w: (w not in ("step", "request"), w))


def _one(a, w: str) -> tuple[str, dict, bool]:
    rec = _rec(a.row_dir, w)
    if rec is None:
        return w, {"skipped": "no recorded result.json"}, True
    out = os.path.join(a.out, f"build_{w}")
    argv = _argv(a.python, rec, w, out, a.workload)
    t0 = time.time()
    with open(os.path.join(a.out, f"build_{w}.log"), "w") as log:
        rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
    wall = round(time.time() - t0, 1)
    new = _rec(a.out, w) or {"ok": False, "message": f"no result.json (rc={rc})"}
    rp, np_ = rec.get("program") or {}, new.get("program") or {}
    equal = bool(rp.get("digest")) and rp.get("digest") == np_.get("digest")
    entry = {"argv": argv[1:], "rc": rc, "wall_s": wall, "ok": new.get("ok"), "unsupported": new.get("unsupported"), "message": new.get("message"),
             "program_digest": {"recorded": rp.get("digest"), "new": np_.get("digest"), "equal": equal},
             "correspondence_digest": np_.get("correspondence_digest"), "correspondence": np_.get("correspondence"),
             "n_root_nodes": {"recorded": rp.get("n_root_nodes"), "new": np_.get("n_root_nodes")},
             "descriptor_bytes_sans_annotations": {"recorded": rp.get("descriptor_bytes"), "new": np_.get("descriptor_bytes")},
             "descriptor_json_gz_bytes": {"recorded": _size(os.path.join(a.row_dir, f"build_{w}", "descriptor.json.gz")), "new": _size(os.path.join(out, "descriptor.json.gz"))},
             "derive_secs": {"recorded": rec.get("derive_secs"), "new": new.get("derive_secs")},
             "max_rss_mb": {"recorded": rec.get("max_rss_mb"), "new": new.get("max_rss_mb")}}
    print(f"{w}: program_digest {'EQUAL' if equal else 'DIFFERENT'} recorded={rp.get('digest')} new={np_.get('digest')} | correspondence_digest={np_.get('correspondence_digest')} "
          f"| calls {rp.get('n_root_nodes')}->{np_.get('n_root_nodes')} | descriptor.json.gz {entry['descriptor_json_gz_bytes']['recorded']}->{entry['descriptor_json_gz_bytes']['new']} B "
          f"| derive {rec.get('derive_secs')}->{new.get('derive_secs')} s | rss {rec.get('max_rss_mb')}->{new.get('max_rss_mb')} MB | rc={rc} ok={new.get('ok')} {new.get('message') or ''}", flush=True)
    return w, entry, equal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload", default=None)
    ap.add_argument("--wrappers", default="step,request", help="comma list, or `all` for every recorded build_<wrapper>/ of the row")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--jobs", type=int, default=1, help="derives run concurrently (each is one process)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report: dict = {"schema": "vllm-v2/rebuild-digest-gate/v1", "row_dir": a.row_dir, "out": a.out, "wrappers": {}}
    ws = _wrappers(a.row_dir, a.wrappers)
    ok_all = True
    if a.jobs <= 1:
        results = [_one(a, w) for w in ws]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(lambda w: _one(a, w), ws))
    for w, entry, equal in results:
        report["wrappers"][w] = entry
        ok_all &= equal
        if "skipped" in entry:
            print(f"{w}: SKIP (no recorded build_{w}/result.json)")
    n_eq = sum(1 for _w, e, eq in results if "skipped" not in e and eq)
    n_cmp = sum(1 for _w, e, _eq in results if "skipped" not in e)
    report.update(ok=ok_all, compared=n_cmp, equal=n_eq)
    json.dump(report, open(os.path.join(a.out, "gate.json"), "w"), indent=1, default=str)
    print(f"gate: {n_eq}/{n_cmp} program_digest equal -> {'OK' if ok_all else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
