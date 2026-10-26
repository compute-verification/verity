#!/usr/bin/env bash
# One remote bandwidth session over C* on the demo card (bandwidth.py session), served from MEM-69's encoded store through
# kernels' staging decode, and the cheat that skips a layer's reads; cwd = the source tree's root:
#
#   research run --on vy-dc-h200-2 --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=bw-h200 [--env CONNECT=127.0.0.1:PORT] [--send store-layout.json] [--send p2lib.tar] \
#     -- bash benchmarks/pous/p2_v1/bw_session.sh
#
# The verifier is `p2_party` on this host by default (VERIFIER, a shell command speaking --stdio), which certifies
# nothing; CONNECT=HOST:PORT (and CHEAT_CONNECT for the cheat) reaches the auditor's `p2_party --listen` instead
# (vy-auditor-2's gate port, through the tunnel the gate opens), one session a listener. The registration the verifier
# holds is the tracked bandwidth-mem69-v2.json, and the store must be it. ENC is only ever read. The lib is kernels'
# build (p2lib.tar, LIB_SHA256), the layout CMP-95 item 2's (LAYOUT_SHA512), each from the run's inputs or, unsent,
# kernels' run KSTAGE's on this host.
#
# Outside the lease: both held to their pins, the store's registration to the tracked one, and the toy's session on the
# CPU (P2's reference decode, the same verifier) as a preflight. Under one untimed lease of at most LEASE_MIN minutes
# (default 10), refused near INF-242's window (Tue 27 Oct 08:00-10:00Z): MODE's sessions (default both), the honest one
# (WINDOWS x PASSES, SAMPLES openings a window, expect accept) and then the cheat (SKIP_LAYER's reads skipped, one window
# of one pass, expect reject). Each step's exit code is in $OUT/rc.txt and each session's verdict line in
# $OUT/verdicts.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/pous/p2_v1
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
CUDA=${CUDA:-/usr/local/cuda-13.0}
ENC=${ENC:-/workspace/research/pous-store/mem69-v2}
NODE=${NODE:-vy-dc-h200-2}
REG=$D/bandwidth-mem69-v2.json
KSTAGE=${KSTAGE:-/workspace/research/runs/r20261026-073433-59f7}
LIB_SHA256=${LIB_SHA256:-6f9789ca44f23592a5c0d1239b10fd81b395f6d94b30bf5246987e225d847353}
LAYOUT_SHA512=${LAYOUT_SHA512:-45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783}
MODE=${MODE:-both}
WINDOWS=${WINDOWS:-2} PASSES=${PASSES:-1} SAMPLES=${SAMPLES:-1024} WINDOW_S=${WINDOW_S:-1} DEADLINE_S=${DEADLINE_S:-1}
SKIP_LAYER=${SKIP_LAYER:-51} CHEAT_SAMPLES=${CHEAT_SAMPLES:-1024}
LEASE_MIN=${LEASE_MIN:-10}
VERIFIER=${VERIFIER:-$PY -m verity.core.protocols.remote_bandwidth.p2_party --registration $REG --stdio}
CONNECT=${CONNECT:-} CHEAT_CONNECT=${CHEAT_CONNECT:-}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LIB_TAR=$IN/p2lib.tar; [ -f "$LIB_TAR" ] || LIB_TAR=$KSTAGE/inputs/p2lib.tar
LAYOUT=$IN/store-layout.json; [ -f "$LAYOUT" ] || LAYOUT=$KSTAGE/inputs/store-layout.json
export O D PY ENC NODE REG LAYOUT S MODE WINDOWS PASSES SAMPLES WINDOW_S DEADLINE_S SKIP_LAYER CHEAT_SAMPLES LEASE_MIN \
  VERIFIER CONNECT CHEAT_CONNECT PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/benchmarks/pous${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"; : > "$O/verdicts.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

[ "$(sha256sum < "$LIB_TAR" | cut -c1-64)" = "$LIB_SHA256" ] && mkdir -p "$S/lib" && tar -C "$S/lib" -xf "$LIB_TAR" \
  && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256) \
  && [ "$(sha512sum < "$LAYOUT" | cut -c1-128)" = "$LAYOUT_SHA512" ]
step ship $?
echo "lib $LIB_TAR layout $LAYOUT" > "$O/inputs.txt"
"$PY" "$D/bandwidth.py" registration --store "$ENC" --node "$NODE" --out "$O/registration.json" > "$O/registration.log" 2>&1 \
  && "$PY" - "$REG" "$O/registration.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
sys.exit(0 if a == b else f"the store's registration {b['digest'][:16]} is not the tracked {a['digest'][:16]}")
EOF
step registration $?
"$PY" "$D/bandwidth.py" toy "$S/toy" > "$O/toy.log" 2>&1 \
  && "$PY" "$D/bandwidth.py" session --store "$S/toy/store" --layout "$S/toy/layout.json" --node p2-toy-node \
    --registration "$S/toy/registration.json" --device cpu --ref-decode --samples 16 --window-s 0.001 --deadline-s 0.001 \
    --verifier "$PY -m verity.core.protocols.remote_bandwidth.p2_party --registration $S/toy/registration.json --stdio" \
    --out "$O/toy" --expect accept >> "$O/toy.log" 2>&1
step toy $?
grep -qE "^(ship|registration|toy) rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt" "$O/registration.log" "$O/toy.log"; exit 2; }
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
gpu-lease status > "$O/lease-before.txt" 2>&1
export LIB=$S/lib/lib

echo "bw_session: asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  now=$(date -u +%s); w0=$(date -u -d 2026-10-27T08:00:00Z +%s); w1=$(date -u -d 2026-10-27T10:00:00Z +%s)
  if [ $((now + 60 * LEASE_MIN)) -gt "$w0" ] && [ "$now" -lt "$w1" ]; then
    echo "a lease from now could reach INF-242 window (Tue 27 Oct 08:00-10:00Z): released unused" >> "$O/rc.txt"
    step window 3; exit 3
  fi
  base=(session --store "$ENC" --layout "$LAYOUT" --lib "$LIB" --node "$NODE" --registration "$REG"
        --window-s "$WINDOW_S" --deadline-s "$DEADLINE_S")
  if [ "$MODE" != cheat ]; then
    if [ -n "$CONNECT" ]; then to=(--connect "$CONNECT"); else to=(--verifier "$VERIFIER"); fi
    "$PY" "$D/bandwidth.py" "${base[@]}" "${to[@]}" --windows "$WINDOWS" --passes "$PASSES" --samples "$SAMPLES" \
      --out "$O/honest" --expect accept > "$O/honest.log" 2> "$O/honest.err"
    step honest $?; tail -n 1 "$O/honest.log" >> "$O/verdicts.txt"
  fi
  if [ "$MODE" != honest ]; then
    if [ -n "$CHEAT_CONNECT" ]; then to=(--connect "$CHEAT_CONNECT"); else to=(--verifier "$VERIFIER"); fi
    "$PY" "$D/bandwidth.py" "${base[@]}" "${to[@]}" --windows 1 --passes 1 --samples "$CHEAT_SAMPLES" \
      --skip-layer "$SKIP_LAYER" --out "$O/cheat" --expect reject > "$O/cheat.log" 2> "$O/cheat.err"
    step cheat $?; tail -n 1 "$O/cheat.log" >> "$O/verdicts.txt"
  fi
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
cat "$O/rc.txt" "$O/verdicts.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
