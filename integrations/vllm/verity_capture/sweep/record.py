"""sweep/record.py — write / update one `verity-sweep/row/v1` record (SCHEMA §3) or a `verity-sweep/failure/v1` record (§4).

    python -m verity_capture.sweep.record row --row-id <id> --lane bc --pod cp-bc --tier 1 --workload workloads/<id>.json \\
        [--set status=QUEUED] [--set build.status=PASS] [--set-json bench='{"rounds": 3}'] [--from-json <path>=<key>]
    python -m verity_capture.sweep.record failure --failure-id F-bc-01 --row-id <id> --stage build --class "…" --cmd "…" --first-error "…" \\
        --artifact "pod:cp-bc:/workspace/cp/sweep/<id>/build.log" [--same-as F-dA-02]

`--set a.b=c` assigns a dotted key (values parsed as JSON when they are JSON, else strings); `updated_at` is stamped on every write.
`row --init` fills `config` / `workload_actual` from the workload manifest's `sweep` block and the row id.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROWS = REPO / "out" / "gen" / "sweep" / "rows"
FAILS = REPO / "out" / "gen" / "sweep" / "failures"
MODEL_DISPLAY = {"smollm2-135m": "SmolLM2-135M", "llama32-1b": "Llama-3.2-1B", "qwen25-15b": "Qwen2.5-1.5B", "phi3-mini": "Phi-3-mini-4k",
                 "gemma2-2b": "Gemma-2-2B", "mistral-7b": "Mistral-7B-v0.3", "olmoe-1b-7b": "OLMoE-1B-7B-0924", "qwen3-4b": "Qwen3-4B-Instruct-2507",
                 "qwen3-4b-fp8": "Qwen3-4B-Instruct-2507-FP8", "qwen3-30b-a3b": "Qwen3-30B-A3B"}
HW = {"l40s": ("L40S", "NVIDIA L40S", [8, 9]), "a100": ("A100", "NVIDIA A100-SXM4-80GB", [8, 0]), "h100": ("H100", "NVIDIA H100 80GB HBM3", [9, 0]),
      "rtx4090": ("RTX 4090", "NVIDIA GeForce RTX 4090", [8, 9])}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short=9", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_row_id(row_id: str) -> dict:
    parts = row_id.split("__")
    if len(parts) != 10:
        raise SystemExit(f"row id must have 10 fields: {row_id}")
    model, dtype, hw, tp, b, i, o, wl, samp, ex = parts
    return {"model": model, "dtype": dtype, "hw": hw, "tp": int(tp[2:]), "batch": int(b[1:]), "max_input": int(i[1:]), "max_output": int(o[1:]),
            "workload": wl, "sampling": samp, "execution": ex}


def _parse(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def set_dotted(d: dict, key: str, value) -> None:
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


def init_row(row_id: str, lane: str, pod: str, tier: int, workload: str | None) -> dict:
    f = parse_row_id(row_id)
    hw_name, gpu_name, cc = HW.get(f["hw"], (f["hw"], f["hw"], None))
    cfg = {"model": f["model"], "model_display": MODEL_DISPLAY.get(f["model"], f["model"]), "hf_repo": None, "revision": None, "registry_role": None,
           "moe": f["model"] in ("olmoe-1b-7b", "qwen3-30b-a3b"), "dtype": f["dtype"], "hardware": hw_name, "compute_capability": cc, "gpu_name": gpu_name,
           "tp": f["tp"], "batch": f["batch"], "max_input": f["max_input"], "max_output": f["max_output"],
           "workload": f["workload"], "workload_manifest": workload, "workload_seed": None,
           "sampling": f["sampling"], "sampling_params": {"temperature": 0, "top_p": 1.0, "seeds": []},
           "execution": f["execution"], "engine_args": {"enforce_eager": True, "VLLM_BATCH_INVARIANT": 1, "max_num_seqs": f["batch"]},
           "vllm_revision": "d9105ea80", "verity_commit": git_sha()}
    actual = {"input_lengths": [], "output_caps": [], "generated_lengths": [], "arrival_offsets_s": [], "sampling_seeds": []}
    if workload:
        w = json.loads((REPO / workload).read_text() if not Path(workload).is_absolute() else Path(workload).read_text())
        sw = w.get("sweep", {})
        cfg.update({"hf_repo": w.get("model"), "revision": w.get("tokenizer_revision"), "registry_role": w.get("case"), "workload_seed": sw.get("seed")})
        sp = sw.get("sampling_params") or {}
        cfg["sampling_params"] = {"temperature": sp.get("temperature", 0), "top_p": sp.get("top_p", 1.0), "seeds": sw.get("sampling_seeds", [])}
        cfg["engine_args"].update(w.get("engine_args_required") or {})
        actual.update({"input_lengths": sw.get("input_lengths", []), "output_caps": sw.get("output_caps", []),
                       "arrival_offsets_s": sw.get("arrival_offsets_s", []), "sampling_seeds": sw.get("sampling_seeds", [])})
    return {"schema": "verity-sweep/row/v1", "row_id": row_id, "lane": lane, "tier": tier, "pod": pod, "status": "QUEUED", "updated_at": now(),
            "config": cfg, "workload_actual": actual,
            "build": {"status": "-", "program_digest": None, "artifact_identity": None, "artifact": None, "cmd": None, "wall_s": None, "failure_id": None},
            "match": {"status": "-", "comparator": None, "verdict": None, "control_vs_instrumented_tokens_equal": None, "artifact": None, "cmd": None, "failure_id": None},
            "commit": {"status": "-", "committer": None, "layout": None, "required_classes": None, "bound_classes": None, "openings_checked": None,
                       "openings_ok": None, "backlog_max": None, "tokens_equal": None, "run_root": None, "artifact": None, "cmd": None, "failure_id": None},
            "bench": {"status": "-", "rounds": None, "warmup": None, "control_s": [], "treatment_s": [], "control_mean_s": None, "treatment_mean_s": None,
                      "overhead_rel": None, "overhead_abs_ms_per_step": None, "decode_steps": None, "profiled": False, "artifact": None, "cmd": None},
            "code_changes_required": [], "notes": ""}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    r = sub.add_parser("row")
    r.add_argument("--row-id", required=True)
    r.add_argument("--lane", default="bc")
    r.add_argument("--pod", default="cp-bc")
    r.add_argument("--tier", type=int, default=1)
    r.add_argument("--workload", default=None, help="manifest path (repo-relative) used by --init")
    r.add_argument("--init", action="store_true", help="(re)create the record from the row id + manifest (keeps nothing)")
    r.add_argument("--set", action="append", default=[], help="dotted.key=value (JSON-parsed when possible)")
    r.add_argument("--from-json", action="append", default=[], help="<file>:<dotted.key.in.file>=<dotted.key.in.record>")
    r.add_argument("--attempt", default=None, help="R13: append an ATTEMPT (json file: {campaign, revision, build, match, commit, bench, status, notes, "
                   "evidence}) to `attempts[]`; the row's top-level stage results/status become this attempt's (earlier attempts stay)")
    f = sub.add_parser("failure")
    f.add_argument("--failure-id", required=True)
    f.add_argument("--row-id", required=True)
    f.add_argument("--stage", required=True, choices=["build", "match", "commit", "bench", "infra"])
    f.add_argument("--class", dest="klass", required=True)
    f.add_argument("--cmd", required=True)
    f.add_argument("--first-error", required=True)
    f.add_argument("--artifact", required=True)
    f.add_argument("--same-as", action="append", default=[])
    a = ap.parse_args(argv)

    if a.kind == "failure":
        FAILS.mkdir(parents=True, exist_ok=True)
        rec = {"schema": "verity-sweep/failure/v1", "failure_id": a.failure_id, "row_id": a.row_id, "stage": a.stage, "class": a.klass,
               "cmd": a.cmd, "commit": git_sha(), "first_error": a.first_error[:2000], "artifact": a.artifact, "same_as": a.same_as, "created_at": now()}
        p = FAILS / f"{a.failure_id}.json"
        p.write_text(json.dumps(rec, indent=1) + "\n")
        print(p)
        return 0

    ROWS.mkdir(parents=True, exist_ok=True)
    p = ROWS / f"{a.row_id}.json"
    # R15 write-time validation (lane report; brief §10): this writer may create/annotate a panel row record but never type a stage
    # verdict — displayed statuses come from `verity_capture.sweep.finalize` (evidence dir → derived statuses).
    try:
        from .finalize import panel_ids, STAGE_STATUS, BENCH_STATUS
    except Exception:  # pragma: no cover
        panel_ids, STAGE_STATUS, BENCH_STATUS = (lambda root="out/gen/sweep": frozenset()), ("-",), ("-",)
    panel = panel_ids(str(ROWS.parent))
    if panel and a.row_id not in panel:
        print(f"REJECTED {a.row_id}: row_id: not one of the {len(panel)} panel rows (out/gen/sweep/PANEL.json) — no panel growth", file=sys.stderr)
        return 2
    if a.attempt:
        print(f"REJECTED {a.row_id}: attempts: `record.py --attempt` types stage statuses — refused since R15 (TABLE-RULES: nobody types PASS). "
              f"Ship the evidence directory and run `python -m verity_capture.sweep.finalize {a.row_id} <evidence-dir> --kind execution --lane <id> --pod <pod> --revision <sha> --release-tag <tag>`", file=sys.stderr)
        return 2
    for s_ in a.set:
        k, _, v = s_.partition("=")
        if k in ("status", "build.status", "match.status", "commit.status", "bench.status") or k.endswith(".status"):
            print(f"REJECTED {a.row_id}: {k}: typed status {_parse(v)!r} refused — the finalizer derives displayed statuses (allowed values it writes: "
                  f"{BENCH_STATUS if k.startswith('bench') else STAGE_STATUS})", file=sys.stderr)
            return 2
    rec = init_row(a.row_id, a.lane, a.pod, a.tier, a.workload) if (a.init or not p.exists()) else json.loads(p.read_text())
    for fj in a.from_json:
        src, _, keys = fj.partition(":")
        skey, _, dkey = keys.partition("=")
        d = json.loads(Path(src).read_text())
        for k in skey.split("."):
            d = d[k]
        set_dotted(rec, dkey or skey, d)
    if a.attempt:
        att = json.loads(Path(a.attempt).read_text())
        att.setdefault("campaign", "r13")
        att.setdefault("revision", git_sha())
        att.setdefault("at", now())
        prev = rec.get("attempts") or []
        if not prev:                                                   # first append: the pre-R13 top-level result becomes attempt 0
            prev = [{"campaign": rec.get("campaign", "r12"), "revision": rec["config"].get("verity_commit"), "at": rec.get("updated_at"),
                     "status": rec.get("status"), "build": rec.get("build"), "match": rec.get("match"), "commit": rec.get("commit"),
                     "bench": rec.get("bench"), "notes": rec.get("notes")}]
        rec["attempts"] = prev + [att]
        for k in ("build", "match", "commit", "bench"):
            if k in att:
                rec[k] = att[k]
        if "status" in att:
            rec["status"] = att["status"]
        rec["campaign"] = att["campaign"]
        if att.get("notes") is not None:
            rec["notes"] = att["notes"]
        if att.get("code_changes_required") is not None:
            rec["code_changes_required"] = att["code_changes_required"]
        if att.get("infra_fixes") is not None:
            rec["infra_fixes"] = att["infra_fixes"]
    for s in a.set:
        k, _, v = s.partition("=")
        set_dotted(rec, k, _parse(v))
    rec["updated_at"] = now()
    rec["config"]["verity_commit"] = rec["config"].get("verity_commit") or git_sha()
    p.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{p.name}: status={rec['status']} build={rec['build']['status']} match={rec['match']['status']} commit={rec['commit']['status']} bench={rec['bench']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
