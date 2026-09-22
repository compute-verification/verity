#!/bin/bash
# Laptop-side: collect one W10 state row (collect.sh + the state-row extras) into fixtures/results/<experiment_id>/.
# usage: collect_w10.sh <CASE> <TAG>     (env: PODSSH, POD_HOST, POD_PORT, POD_KEY as for collect.sh)
set -eu
CASE=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${3:-$HERE/../../results}
export PODSSH=${PODSSH:-/tmp/podssh-w10.sh} POD_HOST=${POD_HOST:-root@47.47.180.32} POD_PORT=${POD_PORT:-10972} POD_KEY=${POD_KEY:-$HOME/.runpod/ssh/vllm-poc-key}
EXP=$(bash "$HERE/collect.sh" "$CASE" "$TAG" "$ROOT" | tail -1)
DEST=$ROOT/$EXP
R=/workspace/results/$CASE/$TAG
RS="rsync -az --max-size=5m -e \"ssh -i $POD_KEY -p $POD_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR\""
for f in compare_fresh.json recompute_check.json cached_check.json cached_vs_fresh.json workload.json workload.json.design.json; do
  eval $RS "$POD_HOST:$R/$f" "$DEST/" 2>/dev/null || true
done
if $PODSSH "test -d $R/fresh"; then
  eval $RS --include='*.json' --include='*.log' --exclude='*' "$POD_HOST:$R/fresh/" "$DEST/fresh/"
fi
# R4: the production-only budget probes (r4-probe-<blocks>) and any replay attempt kept aside (ref1-*)
for P in $($PODSSH "ls -d $R-probe-* $R/ref1-* 2>/dev/null"); do
  case $P in
    *-probe-*) mkdir -p "$DEST/probes/$(basename $P)"; eval $RS --include='*.json' --include='*.log' --include='*.jsonl' --exclude='*' "$POD_HOST:$P/production/" "$DEST/probes/$(basename $P)/";;
    *) eval $RS --include='*.json' --include='*.log' --exclude='*' "$POD_HOST:$P/" "$DEST/$(basename $P)/";;
  esac
done
# the state-row witnesses at the experiment root, for the doc and the render
python3 - "$DEST" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
out = {}
we = d / "production" / "workload_exercised.json"
if we.exists():
    w = json.loads(we.read_text())
    out["production"] = {k: w.get(k) for k in ("preemptions", "prefill_chunks_per_request", "max_tokens_in_a_step", "requests_with_cached_tokens",
                                                "num_cached_tokens_per_request", "engine_steps", "peak_concurrency_achieved")}
m = d / "production" / "manifest.json"
if m.exists():
    out["engine_overrides"] = json.loads(m.read_text()).get("engine_overrides")
rj = d / "bundle" / "replay.json"
if rj.exists():
    r = json.loads(rj.read_text())
    out["replay"] = {"state_rows": r.get("state_rows"), "capture": r.get("capture"), "trajectory_followed": r.get("trajectory_followed"),
                     "logprobs_vs_production": r.get("logprobs_vs_production"), "restrictions": r.get("replay_restrictions", {}).get("engine_args")}
for f in ("compare.json", "compare_fresh.json", "recompute_check.json", "cached_check.json", "cached_vs_fresh.json"):
    p = d / f
    if p.exists():
        j = json.loads(p.read_text())
        out[f[:-5]] = {k: v for k, v in j.items() if not isinstance(v, (list, dict))} | ({"pairs": [{k: v for k, v in pp.items() if k != "examples"} for pp in j["pairs"]]} if "pairs" in j else {})
t1 = d / "table1.jsonl"
if t1.exists():
    row = json.loads(t1.read_text().splitlines()[0])
    out["table1"] = {k: row.get(k) for k in ("experiment_id", "result", "first_blocker", "relation_checks", "structural", "negative", "boundary_noninterference", "token_selection")}
(d / "state_row.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(json.dumps(out.get("table1"), default=str)[:300])
EOF
echo "$EXP"
