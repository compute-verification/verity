#!/usr/bin/env bash
# vy-store-evict (vy-nebius-1, hourly as ubuntu, vy-store-evict.timer from cluster_up.sh): free /workspace by evicting only local
# copies the remote store holds. `research data evict --runs` drops blobs of $VY_EVICT_STORE (/workspace/jobs/store, the job
# pods' store, owned by ubuntu) whose artifacts are PRESERVED on the remote, and the preserved big files of finished run dirs
# under $VY_EVICT_RUNS (/workspace/jobs/runs), oldest first, until $VY_EVICT_FREE_GB (2500, keeping the disk near 54% or under)
# are free. Nothing that isn't preserved is touched. Falling short of the mark is not a failure (nothing more is preserved);
# output that isn't the tool's JSON is. One line per run goes to the journal (journalctl -u vy-store-evict).
# With $VY_EVICT_SRC (research's unit, store_evict_src.conf: /workspace/research/src) it also deletes the `research run` source
# trees there recoverable on the node (a ref of $VY_EVICT_ORIGIN, default <root>/git/verity.git) that nothing uses, every hour whatever the mark
# (`--src-dir`, store/evict.py `evict_src`); the line says how many went and why the rest stayed.
# A tool too old for --src-dir still evicts the rest, and the line says so. Python runs from / so nothing in a writable cwd imports.
#   VY_RESEARCH_TOOL: the research tool to run (default the newest snapshot under /workspace/research/tool).
set -uo pipefail
FREE=${VY_EVICT_FREE_GB:-2500}
STORE=${VY_EVICT_STORE:-/workspace/jobs/store}
RUNS=${VY_EVICT_RUNS:-/workspace/jobs/runs}
SRCDIR=${VY_EVICT_SRC:-}
TOOL=${VY_RESEARCH_TOOL:-$(ls -td /workspace/research/tool/*/ 2> /dev/null | head -1)}
TOOL=${TOOL%/}
if [ -z "$TOOL" ] || [ ! -d "$TOOL/research" ]; then
  echo "vy-store-evict: no research tool (VY_RESEARCH_TOOL, or a snapshot under /workspace/research/tool)" >&2
  exit 1
fi
args=(--target-free-gb "$FREE" --runs --runs-dir "$RUNS")
srcnote=
if [ -n "$SRCDIR" ] && grep -q -- '"--src-dir"' "$TOOL/research/store/cli.py" 2> /dev/null; then
  args+=(--src-dir "$SRCDIR")
  [ -n "${VY_EVICT_ORIGIN:-}" ] && args+=(--src-origin "$VY_EVICT_ORIGIN")
elif [ -n "$SRCDIR" ]; then
  srcnote="; src trees: not evicted, $(basename "$TOOL") has no --src-dir"
fi
avail() { df -B1G --output=avail /workspace | tail -1 | tr -d ' '; }
before=$(avail)
out=$(cd / && RESEARCH_STORE=$STORE PYTHONPATH=$TOOL timeout 3000 python3 -m research data evict "${args[@]}" --json 2>&1)
summary=$(printf '%s' "$out" | python3 -c '
import json, sys
d = json.load(sys.stdin)
s, r, c = d.get("store") or {}, d.get("runs") or {}, d.get("src")
blobs, files = s.get("evicted_bytes", 0) / 1e9, r.get("evicted_bytes", 0) / 1e9
line = "%.1f GB of store blobs, %.1f GB of run files; target reached: %s" % (blobs, files, s.get("reached_target", r.get("reached_target")))
if c is not None:
    gone, kept = c.get("deleted") or [], c.get("kept") or []
    line += "; src trees: deleted %d (%.1f GB, %d inodes), kept %d (%s)" % (len(gone), c.get("deleted_bytes", 0) / 1e9, c.get("deleted_inodes", 0),
        len(kept), " ".join("%s:%d" % kv for kv in sorted((c.get("kept_by") or {}).items())) or "none")
    if c.get("error"):
        line += ", nothing more deleted: " + c["error"]
print(line)
' 2> /dev/null)
if [ -z "$summary" ]; then
  echo "vy-store-evict: research data evict failed: $(printf '%s' "$out" | tail -2 | tr '\n' ' ')" >&2
  exit 1
fi
echo "vy-store-evict: free $before -> $(avail) GiB (mark $FREE GB): evicted $summary$srcnote ($(basename "$TOOL"))"
