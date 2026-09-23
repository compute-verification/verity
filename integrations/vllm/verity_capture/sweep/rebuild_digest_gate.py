"""Rebuild gate (T2, CPU / meta device): re-derive a recorded row's Programs with the CURRENT frontend under the recorded structural
inputs and compare `program_digest` to the record; report the `correspondence_digest` the new frontend emits beside it.

    python -m verity_capture.sweep.rebuild_digest_gate --row-dir /workspace/rows/<row> --out /workspace/v2corr/<row> \
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
    argv = [py, "-m", "verity_capture.experimental.cb_a.derive_step", "--model", pin["model"], "--revision", pin["revision"], "--wrapper", wrapper,
            "--max-model-len", str(pin["max_model_len"]), "--out", out, "--profile", rec.get("profile") or "b1-eager-v3"]
    if wrapper == "step":
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload", default=None)
    ap.add_argument("--wrappers", default="step,request")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report: dict = {"schema": "vllm-v2/rebuild-digest-gate/v1", "row_dir": a.row_dir, "out": a.out, "wrappers": {}}
    ok_all = True
    for w in [x for x in a.wrappers.split(",") if x]:
        rec = _rec(a.row_dir, w)
        if rec is None:
            report["wrappers"][w] = {"skipped": "no recorded result.json"}
            print(f"{w}: SKIP (no recorded build_{w}/result.json)")
            continue
        out = os.path.join(a.out, f"build_{w}")
        argv = _argv(a.python, rec, w, out, a.workload)
        t0 = time.time()
        with open(os.path.join(a.out, f"build_{w}.log"), "w") as log:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
        wall = round(time.time() - t0, 1)
        new = _rec(a.out, w) or {"ok": False, "message": f"no result.json (rc={rc})"}
        rp, np_ = rec.get("program") or {}, new.get("program") or {}
        equal = bool(rp.get("digest")) and rp.get("digest") == np_.get("digest")
        ok_all &= equal
        entry = {"argv": argv[1:], "rc": rc, "wall_s": wall, "ok": new.get("ok"), "unsupported": new.get("unsupported"), "message": new.get("message"),
                 "program_digest": {"recorded": rp.get("digest"), "new": np_.get("digest"), "equal": equal},
                 "correspondence_digest": np_.get("correspondence_digest"), "correspondence": np_.get("correspondence"),
                 "n_root_nodes": {"recorded": rp.get("n_root_nodes"), "new": np_.get("n_root_nodes")},
                 "descriptor_bytes_sans_annotations": {"recorded": rp.get("descriptor_bytes"), "new": np_.get("descriptor_bytes")},
                 "descriptor_json_gz_bytes": {"recorded": _size(os.path.join(a.row_dir, f"build_{w}", "descriptor.json.gz")), "new": _size(os.path.join(out, "descriptor.json.gz"))},
                 "derive_secs": {"recorded": rec.get("derive_secs"), "new": new.get("derive_secs")},
                 "max_rss_mb": {"recorded": rec.get("max_rss_mb"), "new": new.get("max_rss_mb")}}
        report["wrappers"][w] = entry
        print(f"{w}: program_digest {'EQUAL' if equal else 'DIFFERENT'} recorded={rp.get('digest')} new={np_.get('digest')} | correspondence_digest={np_.get('correspondence_digest')} "
              f"| calls {rp.get('n_root_nodes')}->{np_.get('n_root_nodes')} | descriptor.json.gz {entry['descriptor_json_gz_bytes']['recorded']}->{entry['descriptor_json_gz_bytes']['new']} B "
              f"| derive {rec.get('derive_secs')}->{new.get('derive_secs')} s | rss {rec.get('max_rss_mb')}->{new.get('max_rss_mb')} MB | rc={rc} ok={new.get('ok')} {new.get('message') or ''}", flush=True)
    report["ok"] = ok_all
    json.dump(report, open(os.path.join(a.out, "gate.json"), "w"), indent=1, default=str)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
