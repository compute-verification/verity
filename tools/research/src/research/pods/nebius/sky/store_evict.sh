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
# The node's runs have no private remote, so a fixture whose blob is gone fails its suite whenever the public export lacks it
# (rec-coins, 11 Oct). Every artifact in `fixtures/artifacts.json` at the $VY_EVICT_KEEP_REFS newest `refs/research/src/*` commits
# of $VY_EVICT_GIT (each tree `research run` shipped here) keeps its blobs (`--keep-registry`). When no registry can be read, or the
# tool has no --keep-registry, no store blob (and no run file) is evicted, and the line says why.
# With $VY_EVICT_PYCACHE (research's unit: its PYTHONPYCACHEPREFIX, /workspace/research/cache/pycache) it also removes every directory
# there whose source directory is gone (an evicted source tree, a uv build env, a run's scratch), which nothing else prunes: 1.45M
# dead files took node 1 to 63.5% inodes on 11 Oct. Only a definite ENOENT removes one; a source it can't stat stays.
#   VY_RESEARCH_TOOL: the research tool to run (default the newest snapshot under /workspace/research/tool).
set -uo pipefail
FREE=${VY_EVICT_FREE_GB:-2500}
STORE=${VY_EVICT_STORE:-/workspace/jobs/store}
RUNS=${VY_EVICT_RUNS:-/workspace/jobs/runs}
SRCDIR=${VY_EVICT_SRC:-}
GITDIR=${VY_EVICT_GIT:-/workspace/research/git/verity.git}
KEEP_REFS=${VY_EVICT_KEEP_REFS:-20}
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
keep=$(mktemp)
trap 'rm -f "$keep"' EXIT
kept=$(python3 - "$GITDIR" "$KEEP_REFS" "$keep" 2>&1 <<'EOF'
import json, subprocess, sys
g, n, out = sys.argv[1], sys.argv[2], sys.argv[3]
git = ["git", "-c", "safe.directory=*", "--git-dir", g]
shas = subprocess.run([*git, "for-each-ref", "--sort=-committerdate", f"--count={n}", "--format=%(objectname)", "refs/research/src"],
                      capture_output=True, text=True, check=True).stdout.split()
ids = {}
for s in shas:
    p = subprocess.run([*git, "show", f"{s}:fixtures/artifacts.json"], capture_output=True, text=True)
    if p.returncode == 0:
        ids.update({k: v for k, v in json.loads(p.stdout).items() if k.startswith("art:")})
if not ids:
    sys.exit(f"no fixtures/artifacts.json at the newest refs/research/src commits of {g}")
json.dump(ids, open(out, "w"))
print(len(ids))
EOF
)
if [[ "$kept" =~ ^[0-9]+$ ]] && grep -q -- '"--keep-registry"' "$TOOL/research/store/cli.py" 2> /dev/null; then
  args+=(--keep-registry "$keep")
  keepnote="; kept $kept registry fixtures"
else
  [[ "$kept" =~ ^[0-9]+$ ]] && kept="$(basename "$TOOL") has no --keep-registry"
  args+=(--min-mb 1e12)
  keepnote="; no store blob or run file evicted: $(printf '%s' "$kept" | tail -1)"
fi
avail() { df -B1G --output=avail /workspace | tail -1 | tr -d ' '; }
before=$(avail)
out=$(cd / && RESEARCH_STORE=$STORE PYTHONPATH=$TOOL timeout 3000 python3 -m research data evict "${args[@]}" --json 2>&1)
pycnote=
if [ -n "${VY_EVICT_PYCACHE:-}" ]; then
  pycnote="; pycache: $(cd / && python3 -I - "$VY_EVICT_PYCACHE" 2>&1 <<'EOF' | tail -1
import os, shutil, sys
top = os.path.abspath(sys.argv[1])
dirs = files = 0
for d, subs, names in os.walk(top):
    keep = []
    for s in subs:
        mirror = os.path.join(d, s)
        try:
            os.lstat(mirror[len(top):])
            keep.append(s)
        except FileNotFoundError:
            for _, _, fs in os.walk(mirror):
                files += len(fs)
            shutil.rmtree(mirror, ignore_errors=True)
            dirs += 1
        except OSError:
            keep.append(s)
    subs[:] = keep
print(f"removed {dirs} dirs ({files} files) whose source is gone")
EOF
)"
fi
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
echo "vy-store-evict: free $before -> $(avail) GiB (mark $FREE GB): evicted $summary$srcnote$keepnote$pycnote ($(basename "$TOOL"))"
