#!/usr/bin/env bash
# vy-store-evict (vy-nebius-1, hourly as ubuntu, vy-store-evict.timer from cluster_up.sh): free /workspace by evicting only local
# copies the remote store holds. `research data evict --runs` drops blobs of $VY_EVICT_STORE (/workspace/jobs/store, the job
# pods' store, owned by ubuntu) whose artifacts are PRESERVED on the remote, and the preserved big files of finished run dirs
# under $VY_EVICT_RUNS (/workspace/jobs/runs), oldest first, until $VY_EVICT_FREE_GB (2500, keeping the disk near 54% or under)
# are free. Nothing that isn't preserved is touched. Falling short of the mark is not a failure (nothing more is preserved);
# output that isn't the tool's JSON is. One line per run goes to the journal (journalctl -u vy-store-evict).
#   VY_RESEARCH_TOOL: the research tool to run (default the newest snapshot under /workspace/research/tool).
set -uo pipefail
FREE=${VY_EVICT_FREE_GB:-2500}
STORE=${VY_EVICT_STORE:-/workspace/jobs/store}
RUNS=${VY_EVICT_RUNS:-/workspace/jobs/runs}
TOOL=${VY_RESEARCH_TOOL:-$(ls -td /workspace/research/tool/*/ 2> /dev/null | head -1)}
TOOL=${TOOL%/}
if [ -z "$TOOL" ] || [ ! -d "$TOOL/research" ]; then
  echo "vy-store-evict: no research tool (VY_RESEARCH_TOOL, or a snapshot under /workspace/research/tool)" >&2
  exit 1
fi
avail() { df -B1G --output=avail /workspace | tail -1 | tr -d ' '; }
before=$(avail)
out=$(cd /tmp && RESEARCH_STORE=$STORE PYTHONPATH=$TOOL timeout 3000 python3 -m research data evict --target-free-gb "$FREE" --runs \
  --runs-dir "$RUNS" --json 2>&1)
summary=$(printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
s, r = d.get("store") or {}, d.get("runs") or {}
blobs, files = s.get("evicted_bytes", 0) / 1e9, r.get("evicted_bytes", 0) / 1e9
print("%.1f GB of store blobs, %.1f GB of run files; target reached: %s" % (blobs, files, s.get("reached_target", r.get("reached_target"))))
' 2> /dev/null)
if [ -z "$summary" ]; then
  echo "vy-store-evict: research data evict failed: $(printf '%s' "$out" | tail -2 | tr '\n' ' ')" >&2
  exit 1
fi
echo "vy-store-evict: free $before -> $(avail) GiB (mark $FREE GB): evicted $summary ($(basename "$TOOL"))"
