#!/usr/bin/env bash
# (Re)start the live verifier (`python -m backends.direct.ligero.live serve`) on a CPU pod from the tree this script sits in.
# Idempotent: bootstraps only what is missing, rebuilds ligero-verify (cargo is incremental), restarts the server.
#
#   research pods create --name vy-verifier-<dc> --cpu cpu3m --vcpu 8 --disk 50 --expose-port 7000 --data-center <DC of the provers>
#   cd <worktree to serve, e.g. the integration branch>
#   research pods sync vy-verifier-<dc> --dest /workspace/lv-src
#   research pods ssh vy-verifier-<dc> -- bash /workspace/lv-src/backends/direct/ligero/live_serve.sh
#   research pods list          # the public tcp mapping of :7000 -> the provers' --verifier tcp://IP:PORT
#
# Leaves on the pod:
#   /workspace/venv312            Python 3.12 + torch (CPU wheel) + numpy + blake3 + pytest (live.py imports protocol/serialize)
#   /workspace/bin/ligero-verify  cargo --release build of $SRC/backends/ligero-verify (target dir /workspace/cargo-target)
#   /workspace/live/start.sh      the server in a restart loop; log /workspace/live/serve.out; per-session lines in
#                                 /workspace/live/sessions/serve.log, one line per finished session in .../index.jsonl
#
# Knobs (environment): LIVE_JOBS=8 (parallel ligero-verify processes; ~1 per vCPU), LIVE_PORT=7000, LIVE_TARGET_BITS=128,
#   LIVE_FORCE=1 (restart even while sessions are connected: they fail), SKIP_BUILD=1 (keep /workspace/bin/ligero-verify).
#   The verifier identity in every verdict is live-verifier@<commit of the synced tree> (.research-source.json).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
W=/workspace
PY=$W/venv312/bin/python
PORT=${LIVE_PORT:-7000}
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" CARGO_TARGET_DIR=$W/cargo-target
mkdir -p $W/bin $W/live/sessions
stage() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

open=$(ss -Htn state established "( sport = :$PORT )" 2>/dev/null | wc -l)
if [ "$open" -gt 0 ] && [ "${LIVE_FORCE:-0}" != 1 ]; then
  echo "REFUSED: $open connection(s) on :$PORT (a session is running; check $W/live/sessions/serve.log). LIVE_FORCE=1 overrides." >&2
  exit 2
fi

stage "machine: $(nproc) vCPU, DC ${RUNPOD_DC_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_DC_ID=//p')}, tree $SRC"
command -v rsync >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync >/dev/null 2>&1; }   # later syncs: rsync, not tar
if ! command -v cargo >/dev/null 2>&1; then
  stage "rustup stable"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/tmp/rustup.log 2>&1 || { tail /tmp/rustup.log; exit 1; }
fi
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  stage "cargo build --release backends/ligero-verify (background)"
  ( cd "$SRC/backends/ligero-verify" && cargo build --release 2>&1 | tail -2 && cp "$CARGO_TARGET_DIR/release/ligero-verify" $W/bin/ligero-verify.new \
      && mv $W/bin/ligero-verify.new $W/bin/ligero-verify ) > /tmp/live_serve_build.log 2>&1 &
  BUILD=$!
fi
if ! "$PY" -c "import torch, numpy, blake3" 2>/dev/null; then
  stage "uv + python 3.12 + torch (CPU) + numpy + blake3"
  command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  uv python install 3.12 >/dev/null 2>&1
  [ -x "$PY" ] || uv venv $W/venv312 --python 3.12 >/dev/null
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0" 2>&1 | tail -1
  uv pip install --python "$PY" numpy blake3 pytest 2>&1 | tail -1
fi
if [ -n "${BUILD:-}" ]; then
  wait $BUILD || { cat /tmp/live_serve_build.log; echo "BUILD FAILED" >&2; exit 1; }
  cat /tmp/live_serve_build.log
fi
[ -x $W/bin/ligero-verify ] || { echo "no $W/bin/ligero-verify" >&2; exit 1; }
sha256sum $W/bin/ligero-verify

COMMIT=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('commit') or '')" "$SRC/.research-source.json" 2>/dev/null)
[ -n "$COMMIT" ] || COMMIT=$(cat "$SRC/COMMIT" 2>/dev/null)
echo "$COMMIT" > "$SRC/COMMIT"
cat > $W/live/start.sh <<EOF
#!/bin/bash
export PYTHONPATH="$SRC/packages/verity/src:$SRC/backends/numerical/python:$SRC/tools/research/src:$SRC"
export RESEARCH_GIT_COMMIT=\$(cat "$SRC/COMMIT")
cd "$SRC"
while true; do
  "$PY" -m backends.direct.ligero.live serve --listen 0.0.0.0:$PORT --out $W/live/sessions --ligero-verify $W/bin/ligero-verify \\
      --jobs ${LIVE_JOBS:-8} --threads 1 --target-bits ${LIVE_TARGET_BITS:-128} >> $W/live/serve.out 2>&1
  echo "[\$(date -u +%FT%TZ)] server exited rc=\$?; restarting in 2s" >> $W/live/serve.out
  sleep 2
done
EOF
chmod +x $W/live/start.sh
stage "restart live serve on :$PORT as live-verifier@${COMMIT:0:12}"
pkill -f "$W/live/start.sh" 2>/dev/null; pkill -f "ligero.live serve --listen 0.0.0.0:$PORT" 2>/dev/null; sleep 1
setsid nohup bash $W/live/start.sh > /dev/null 2>&1 < /dev/null &
for _ in $(seq 20); do ss -Hltn "( sport = :$PORT )" | grep -q . && break; sleep 0.5; done
tail -1 $W/live/serve.out
ss -Hltn "( sport = :$PORT )" | grep -q . && echo SERVING || { echo "NOT LISTENING (see $W/live/serve.out)" >&2; exit 1; }
