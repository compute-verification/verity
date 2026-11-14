#!/usr/bin/env bash
# launch.sh LABEL SCRIPT [ARGS...]: a research run at the architecture worktree's pushed head, running SCRIPT (copied into $OUT) from
# adapters/vllm under watch.sh. MEM (GB) and TIMEOUT (s) pass through when set. Logs to /tmp/cir/LABEL.log.
set -u
label=$1; script=$2; shift 2
s64=$(base64 -w0 "$script"); w64=$(base64 -w0 /tmp/cir/watch.sh)
cmd="git rev-parse HEAD | tee \$OUT/head.txt; git status --short > \$OUT/status.txt; uv sync -q --all-packages --extra torch-cpu </dev/null || exit 3; echo $s64 | base64 -d > \$OUT/script.py; echo $w64 | base64 -d > \$OUT/watch.sh; cd adapters/vllm && bash \$OUT/watch.sh job ../../.venv/bin/python \$OUT/script.py \$OUT $*"
opts=(--on auto --cwd source)
[ -n "${MEM:-}" ] && opts+=(--mem-gb "$MEM")
[ -n "${TIMEOUT:-}" ] && opts+=(--timeout "$TIMEOUT")
cd /agent/repos/verity-arch && uv run -q --project /agent/repos/verity/tools/research research run "${opts[@]}" -- bash -lc "$cmd" > "/tmp/cir/$label.log" 2>&1
