#!/usr/bin/env bash
# Pearl-C4's m64-n512-k2048 decode re-time (Llama-3.2-1B's k/v; B row 386, R1-rejected in r20261001-134930-22d2) into a booked
# whole-node line of node 2's fill/windows, with the re-time's harness and methodology (llama8b_window.sh timed: 20 reps,
# locked-2100, finalists 3, dependent chain, gpu-lease 8 --timed, the harness on GPU 0, host threads on 48-91):
#
#   bash runtree/launch_decode_row.sh WINDOW_AT [LEASE_MIN]      e.g. 2026-10-01T22:30:00Z 10
#
# Run it from a clean checkout of this branch, in tmux: the local `research run` follows the run. Then pin the runner itself,
# as the re-time did: ssh node 2, `taskset -acp 48-91 <runner_pid of runs/<id>/launch.json>`.
set -euo pipefail
at=${1:?WINDOW_AT, e.g. 2026-10-01T22:30:00Z} lease=${2:-10}
here=$(git rev-parse --show-toplevel)
[[ -z $(git -C "$here" status --porcelain) ]] || { echo "the checkout is dirty" >&2; exit 1; }
[[ $(git -C "$here" rev-parse HEAD) == $(git -C "$here" rev-parse @{u}) ]] || { echo "HEAD is not on origin" >&2; exit 1; }
d=$(mktemp -d)
uv run --project /workspace research data fetch art:7418102e56c56ff4cf8234937dabf8fb39578feca4a1302357df711fc1e31e71 --to "$d" > /dev/null
echo "dd01ae5e5ac74ad65f62d63261ce5293b0fbf43be3d9c10e59cfd260b95f95ff  $d/pearl_c4.cubin" | sha256sum -c --quiet
t=$(( $(date -u -d "$at" +%s) - $(date -u +%s) + lease * 60 + 1800 ))
cd "$here"
exec uv run --project /workspace research run --on vy-nebius-2 --project verity --campaign pouw --custody-r2 --custody-ttl 8h \
  --source "$here" --cwd source --send "$d/pearl_c4.cubin" --no-sampler --timeout "$t" \
  --env WINDOW_AT="$at" --env LEASE_MIN="$lease" --env ITEMS=narrow_decode64:64 --env LINEARS_narrow_decode64=n512-k2048 \
  --declared-output 'llama8b/*.json' --declared-output 'llama8b/*.jsonl' --declared-output 'llama8b/*.log' \
  --declared-output 'llama8b/*/*.json' --declared-output 'llama8b/*/*.log' --declared-output 'llama8b/*/verify/*' \
  --declared-output 'llama8b/*/ledger/*' --declared-output 'llama8b/*/main-verify/*/*/*/*' \
  -- bash runtree/llama8b_window.sh timed
