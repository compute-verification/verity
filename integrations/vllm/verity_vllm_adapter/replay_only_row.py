"""Table 1 row for a replay-agreement-only experiment (CONTRACT §16.1 rung (a)): a replay that ran the
production computation with **no interception and no commitments** and is compared with production on
tokens and logprobs only. Nothing in §9's check fields was attempted (authentication, relations,
negatives, coverage: 0/0), so the row is ``NOT_RUN`` by the §10 rule with the reason as its blocker; the
agreement counts live in the README and in ``bundle/replay.json`` (``unforced_sampler_agreement``,
``logprobs_vs_production``, ``kernel_identity``). Never a substitute for a verified row.

    python -m verity_vllm_adapter.replay_only_row --replay R --production P --case B0 --worker W1c \
        --adapter-revision REV --out results/<experiment_id>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity_vllm import rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="replay run dir (replay.json, manifest.json)")
    ap.add_argument("--production", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--worker", default="W1c")
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--out-root", required=True, help="results root; the experiment directory is created inside")
    ap.add_argument("--utc", default=None, help="override the id timestamp (yyyymmddThhmmZ)")
    args = ap.parse_args()
    rp = Path(args.replay)
    prod = Path(args.production)
    replay = json.loads((rp / "replay.json").read_text())
    manifest = json.loads((rp / "manifest.json").read_text())
    profile_id = replay["profile_id"]
    utc = args.utc or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    exp_id = rows.experiment_id(args.case, profile_id, utc)
    common = rows.Common(exp_id, args.case, profile_id, args.worker, args.adapter_revision)
    prod_rows = [json.loads(l) for l in (prod / "production.jsonl").read_text().splitlines() if l.strip()]
    steps = json.loads((prod / "steptrace.json").read_text()).get("summary", {}) if (prod / "steptrace.json").exists() else {}
    workload = {
        "requests": len(prod_rows),
        "generated_positions": sum(len(r.get("output_token_ids", [])) for r in prod_rows),
        "input_len_range": [min(len(r["prompt_token_ids"]) for r in prod_rows), max(len(r["prompt_token_ids"]) for r in prod_rows)],
        "peak_concurrency_achieved": int(steps.get("peak_concurrency_achieved", 0) or 0),
    }
    wx = prod / "workload_exercised.json"
    workload_exercised = json.loads(wx.read_text()) if wx.exists() else None
    from verity_vllm_adapter.verify import production_changes_of  # JSON-only helper

    rr = dict(replay.get("replay_restrictions") or {})
    if replay.get("no_captures"):
        rr["interception"] = "none (--no-captures: dispatch mode never entered, autotuner untouched; token/logprob agreement and kernel identity only)"
    restrictions = [f"{k}={v}" for k, v in rr.items() if not isinstance(v, dict)]
    ua = replay.get("unforced_sampler_agreement") or {}
    lp = replay.get("logprobs_vs_production") or {}
    ki = (replay.get("kernel_identity") or {}).get("vs_production") or {}
    zero = {"exact": 0, "attempted": 0}
    reason = (
        f"replay-agreement only, no interception and no commitments by design (rung (a)): production tokens vs replay unforced argmax "
        f"{ua.get('agree')}/{ua.get('positions')}, chosen logprobs bit-equal on {lp.get('equal')}/{lp.get('requests_compared')} requests, "
        f"generated kernel sources identical to production: {ki.get('kernel_sources_identical')}"
    )
    t1 = rows.table1_row(
        common,
        workload=workload,
        production_changes=production_changes_of(manifest),
        replay_restrictions=restrictions,
        token_selection=zero,
        boundary_noninterference={"identical": False, "compared": 0, "mismatched_scalars": 0, "reference_method": "not attempted (no capture)"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks={"same_runtime": zero, "independent": zero},
        structural={"result": "not attempted", "failing_invariant": None},
        negative={"rejected": 0, "attempted": 0},
        blocker={"kind": "NOT_RUN", "reason": reason},
        workload_exercised=workload_exercised,
    )
    out = Path(args.out_root) / exp_id
    out.mkdir(parents=True, exist_ok=True)
    rows.write_jsonl(out / "table1.jsonl", [t1])
    (out / "manifest.json").write_bytes((rp / "manifest.json").read_bytes())
    print(exp_id)
    print(json.dumps({"result": t1["result"], "first_blocker": t1["first_blocker"]}, indent=1))


if __name__ == "__main__":
    main()
