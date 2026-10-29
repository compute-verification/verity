#!/usr/bin/env bash
# One remote bandwidth session over C* on the demo card (bandwidth.py session), served from the mem69-v2 encoded store through
# kernels' staging decode, and the cheat that skips a layer's reads; cwd = the source tree's root:
#
#   OUT=DIR RESEARCH_RUN_DIR=RUN GPU_LEASE_WHO=bw-h200 [PARTY='ssh ... VERB' PARTY_HOST=ADDRESS] \
#     bash benchmarks/pous/p2_v1/bw_session.sh      # on the demo card's node; RUN/inputs/ may hold store-layout.json, p2lib.tar
#
# The verifier is `p2_party` on this host by default (VERIFIER, a shell command speaking --stdio), which certifies
# nothing. PARTY is the auditor's instead: the ssh command to the auditor host's gate verb that starts `p2_party --listen`
# there, once a session (the honest one, then the cheat), whose port this host reaches at PARTY_HOST and whose stdout
# ends with the verifier's own record of the answer (`bandwidth.py session --party`). Or CONNECT=HOST:PORT reaches a
# `p2_party --listen --sessions 2` already listening, the cheat at CHEAT_CONNECT (by default the same listener), with
# TUNNEL ('SSH_OPTIONS USER@HOST') through `ssh -L` to TUNNEL's host, CONNECT then the address there. The registration
# the verifier holds is the tracked bandwidth-mem69-v2.json, and the store must be it, its plan the layout's tensors'
# units (not the padding past lm_head). ENC is only ever read. The lib
# is kernels' build: p2lib.tar from the run's inputs (LIB_SHA256), else that build's out/ on this host (BUILD), else
# build.sh's in scratch, each held to its MANIFEST.sha256 and its sources to the kernels' build's
# (LIB_SOURCES_SHA256); the layout is the card's frozen store layout (LAYOUT_SHA512), from the run's inputs or kernels' run KSTAGE's
# on this host.
#
# Outside the lease: both held to their pins, the store's registration to the tracked one, and the toy's session on the
# CPU (P2's reference decode, the same verifier) as a preflight. Under one untimed lease of at most LEASE_MIN minutes
# (default 10), refused near the card's booked window (`w0` to `w1`): MODE's sessions (default both), the honest one
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
BUILD=${BUILD:-/workspace/research/runs/r20261025-233631-4164/out}
LIB_SHA256=${LIB_SHA256:-6f9789ca44f23592a5c0d1239b10fd81b395f6d94b30bf5246987e225d847353}
# kernels' build (BUILD): its libp2v1.so, and the sha256 of its source.txt's source lines (this tree's at b7250b3831)
LIB_SO_SHA256=8380423f24bb469a06a8aeb7f3f7cad7989febad9bf220e1999cbebd48211db5
LIB_SOURCES_SHA256=${LIB_SOURCES_SHA256:-6ce1eb44d059949d391d96764f0ddc0810de9a70702c2883ba5d4bd3dac80634}
LAYOUT_SHA512=${LAYOUT_SHA512:-45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783}
MODE=${MODE:-both}
WINDOWS=${WINDOWS:-2} PASSES=${PASSES:-1} SAMPLES=${SAMPLES:-1024} WINDOW_S=${WINDOW_S:-1} DEADLINE_S=${DEADLINE_S:-1}
SKIP_LAYER=${SKIP_LAYER:-51} CHEAT_SAMPLES=${CHEAT_SAMPLES:-1024}
LEASE_MIN=${LEASE_MIN:-10}
VERIFIER=${VERIFIER:-$PY -m verity.core.protocols.remote_bandwidth.p2_party --registration $REG --stdio}
CONNECT=${CONNECT:-}; CHEAT_CONNECT=${CHEAT_CONNECT:-$CONNECT}; TUNNEL=${TUNNEL:-}
PARTY=${PARTY:-} PARTY_HOST=${PARTY_HOST:-}
S=$(mktemp -d)
trap 'for c in "$S"/ctl-*; do [ -S "$c" ] && ssh -S "$c" -O exit tunnel 2>/dev/null; done; rm -rf "$S"' EXIT
LIB_TAR=$IN/p2lib.tar; [ -f "$LIB_TAR" ] || LIB_TAR=$KSTAGE/inputs/p2lib.tar
LAYOUT=$IN/store-layout.json; [ -f "$LAYOUT" ] || LAYOUT=$KSTAGE/inputs/store-layout.json
export O D PY ENC NODE REG LAYOUT S MODE WINDOWS PASSES SAMPLES WINDOW_S DEADLINE_S SKIP_LAYER CHEAT_SAMPLES LEASE_MIN \
  VERIFIER CONNECT CHEAT_CONNECT PARTY PARTY_HOST PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/benchmarks/pous${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"; : > "$O/verdicts.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

mkdir -p "$S/lib"
if [ -f "$LIB_TAR" ]; then
  [ "$(sha256sum < "$LIB_TAR" | cut -c1-64)" = "$LIB_SHA256" ] && tar -C "$S/lib" -xf "$LIB_TAR"; LIB_FROM=$LIB_TAR
elif [ -f "$BUILD/MANIFEST.sha256" ]; then
  cp -r "$BUILD/." "$S/lib/"; LIB_FROM=$BUILD
else
  bash benchmarks/torch_gpu/p2_stage/build.sh "$S/lib" > "$O/build.log" 2>&1; LIB_FROM="build.sh at this commit"
fi
(cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256) \
  && [ "$(tail -n +2 "$S/lib/source.txt" | sha256sum | cut -c1-64)" = "$LIB_SOURCES_SHA256" ] \
  && [ "$(sha512sum < "$LAYOUT" | cut -c1-128)" = "$LAYOUT_SHA512" ]
step ship $?
cp "$S/lib/source.txt" "$S/lib/toolchain.txt" "$S/lib/MANIFEST.sha256" "$O/" 2>/dev/null
so=$(sha256sum < "$S/lib/lib/libp2v1.so" | cut -c1-64)
echo "lib from $LIB_FROM: libp2v1.so $so (kernels' build's: $([ "$so" = "$LIB_SO_SHA256" ] && echo same || echo other)); layout $LAYOUT" \
  > "$O/inputs.txt"
"$PY" "$D/bandwidth.py" registration --store "$ENC" --node "$NODE" --layout "$LAYOUT" --out "$O/registration.json" \
  > "$O/registration.log" 2>&1 \
  && "$PY" - "$REG" "$O/registration.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
sys.exit(0 if a == b else f"the store's registration {b['digest'][:16]}, plan {b.get('plan')}, is not the tracked "
                          f"{a['digest'][:16]}, plan {a.get('plan')}")
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
forward() {  # HOST:PORT as TUNNEL's host sees it, forwarded to a port of this host's 127.0.0.1, which it prints
  local lport
  lport=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # shellcheck disable=SC2086
  ssh -f -N -M -S "$S/ctl-$lport" -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=20 \
    -L "127.0.0.1:$lport:$1" $TUNNEL > /dev/null 2>> "$O/tunnel.err" && echo "127.0.0.1:$lport"
}
if [ -n "$TUNNEL" ] && [ -n "$CONNECT" ]; then
  c=$(forward "$CONNECT"); step tunnel $?
  if [ "$CHEAT_CONNECT" = "$CONNECT" ]; then CHEAT_CONNECT=$c; else CHEAT_CONNECT=$(forward "$CHEAT_CONNECT"); fi
  CONNECT=$c
  echo "through $TUNNEL: the honest session at $CONNECT, the cheat at $CHEAT_CONNECT" >> "$O/inputs.txt"
  grep -q "^tunnel rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt" "$O/tunnel.err"; exit 2; }
fi

echo "bw_session: asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  now=$(date -u +%s); w0=$(date -u -d 2026-10-27T08:00:00Z +%s); w1=$(date -u -d 2026-10-27T10:00:00Z +%s)
  if [ $((now + 60 * LEASE_MIN)) -gt "$w0" ] && [ "$now" -lt "$w1" ]; then
    echo "a lease from now could reach the booked window (Tue 27 Oct 08:00-10:00Z): released unused" >> "$O/rc.txt"
    step window 3; exit 3
  fi
  base=(session --store "$ENC" --layout "$LAYOUT" --lib "$LIB" --node "$NODE" --registration "$REG"
        --window-s "$WINDOW_S" --deadline-s "$DEADLINE_S")
  if [ "$MODE" != cheat ]; then
    to=(--verifier "$VERIFIER"); [ -n "$CONNECT" ] && to=(--connect "$CONNECT")
    [ -n "$PARTY" ] && to=(--party "$PARTY" --party-host "$PARTY_HOST")
    "$PY" "$D/bandwidth.py" "${base[@]}" "${to[@]}" --windows "$WINDOWS" --passes "$PASSES" --samples "$SAMPLES" \
      --out "$O/honest" --expect accept > "$O/honest.log" 2> "$O/honest.err"
    step honest $?; tail -n 1 "$O/honest.log" >> "$O/verdicts.txt"
  fi
  if [ "$MODE" != honest ]; then
    to=(--verifier "$VERIFIER"); [ -n "$CHEAT_CONNECT" ] && to=(--connect "$CHEAT_CONNECT")
    [ -n "$PARTY" ] && to=(--party "$PARTY" --party-host "$PARTY_HOST")
    "$PY" "$D/bandwidth.py" "${base[@]}" "${to[@]}" --windows 1 --passes 1 --samples "$CHEAT_SAMPLES" \
      --skip-layer "$SKIP_LAYER" --out "$O/cheat" --expect reject > "$O/cheat.log" 2> "$O/cheat.err"
    step cheat $?; tail -n 1 "$O/cheat.log" >> "$O/verdicts.txt"
  fi
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
cat "$O/rc.txt" "$O/verdicts.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
