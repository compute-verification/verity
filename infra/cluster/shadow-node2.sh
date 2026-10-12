#!/usr/bin/env bash
# Step 3 of the one-cluster cutover: `cluster agent --mode shadow` beside vy-nebius-2's gpu-lease for HOURS (default 8).
#
# Run from the repository root (`research run --on vy-nebius-2 --cwd source ... -- bash infra/cluster/shadow-node2.sh 8`).
# While it runs it writes only /workspace/pouw/infra/cluster/shadow/<run id>/ (node2-ops' condition: under 1 GB, covered
# by the node's hourly backup). It reads /run/gpu-lease, /workspace/pouw/fill/status.txt and, at the end, read-only copies
# of the node's own logs for the same span (the sampler's util/*.jsonl, node_ops' quiet.jsonl, preempted.log), taken
# outside a timed window. Then it evaluates the pass bar and copies everything into $RESEARCH_RUN_DIR/out/ for custody.
# `touch /workspace/pouw/infra/cluster/shadow/STOP` ends it early, cleanly.
set -u
hours=${1:-8}
base=/workspace/pouw/infra/cluster/shadow
out=$base/${RESEARCH_RUN_ID:-local-$$}
status=/workspace/pouw/fill/status.txt
mkdir -p "$out/node"
touch "$out/.started"
export PYTHONPATH=$PWD/infra/cluster/src
cluster() { nice -n 19 python3 -m cluster "$@"; }

cluster agent infra/cluster/descriptions/nebius.toml --mode shadow --out "$out" --hours "$hours" --stop-file "$base/STOP"
rc=$?

while head -1 "$status" 2>/dev/null | grep -q 'timed True'; do sleep 5; done
find /workspace/pouw/infra/util -maxdepth 1 -name '*.jsonl' -newer "$out/.started" -exec nice -n 19 cp {} "$out/node/" \;
nice -n 19 cp /workspace/pouw/infra/logs/quiet.jsonl /run/gpu-lease/preempted.log "$out/node/" 2>/dev/null
nice -n 19 gzip -f "$out"/node/*.jsonl "$out"/node/*.log 2>/dev/null

cluster shadow evaluate "$out" --quiet-log "$out/node/quiet.jsonl.gz" --sampler "$out"/node/20*.jsonl.gz > "$out/evaluation.json"
erc=$?
echo "cluster shadow: agent exit $rc, evaluation exit $erc (0 = the pass bar holds), $(du -sh "$out" | cut -f1) in $out"
if [ -n "${RESEARCH_RUN_DIR:-}" ]; then
  mkdir -p "$RESEARCH_RUN_DIR/out"
  cp -r "$out/." "$RESEARCH_RUN_DIR/out/"
fi
exit "$rc"
