"""For each candidate line of /tmp/cir255_cands.txt with gate `check`, fetch each run's records (no --all) and print what
result.json says of the pytest shard's circuit_check suite and of the circuit-check step."""
import json
import os
import subprocess
import sys
from pathlib import Path

RUNS = Path.home() / ".research" / "runs"
REPO = "/agent/repos/verity"


def fetch(rid: str) -> Path:
    d = RUNS / rid
    if not (d / "result.json").exists():
        subprocess.run(["uv", "run", "-q", "--project", f"{REPO}/tools/research", "research", "fetch", rid],
                       cwd=REPO, capture_output=True, timeout=300)
    return d


def info(rid: str) -> dict:
    d = fetch(rid)
    out = {"run": rid}
    try:
        job = json.loads((d / "job.json").read_text())
        out["node"] = (job.get("cost") or {}).get("machine")
    except (OSError, ValueError):
        pass
    try:
        slot = json.loads((d / "slot.json").read_text())
        out["slot"] = (slot.get("slot"), slot.get("cores"), slot.get("since"), slot.get("took"))
    except (OSError, ValueError):
        pass
    try:
        r = json.loads((d / "result.json").read_text())
    except (OSError, ValueError):
        out["result"] = "absent"
        return out
    for s in r.get("steps", []):
        if s["step"] == "pytest":
            out["pytest_s"] = s.get("seconds")
        if s["step"] == "circuit-check":
            c = s.get("cache") or {}
            u = c.get("units") or {}
            out["cc_s"] = s.get("seconds")
            out["cc_targets"] = (c.get("hits"), c.get("stored"), c.get("changed"))
            out["cc_units"] = (u.get("reused"), len(u.get("rebuilt") or []))
            out["cc_parts"] = s.get("parts")
    pt = (r.get("reuse") or {}).get("steps", {}).get("pytest") or {}
    for x in pt.get("reran", []):
        if x.get("suite") == "tools/verity/circuit_check":
            out["suite_s"] = x.get("seconds")
            out["suite_forced"] = [f[:140] for f in x.get("forced_by", [])]
    for x in pt.get("reused_suites", []) if isinstance(pt.get("reused_suites"), list) else []:
        if "circuit_check" in str(x):
            out["suite_reused"] = x
    out["deferred_parts"] = [p for p in (r.get("deferred") or {}).get("parts", []) if p.startswith(("pytest", "circuit-check"))]
    return out


lines = [ln.split() for ln in Path("/tmp/cir255_cands.txt").read_text().splitlines() if ln.strip()]
want = sys.argv[1:] or None
for ln in lines:
    at, verdict, sha, base, gate, ids = ln[0], ln[1] + " " + ln[2], ln[3], ln[4], ln[5], ln[6]
    if gate != "check" or (want and sha not in want):
        continue
    print(json.dumps({"at": at, "verdict": verdict, "sha": sha, "base": base}))
    for rid in ids.split(","):
        print("   ", json.dumps(info(rid)))
    sys.stdout.flush()
