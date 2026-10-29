#!/usr/bin/env bash
# A served Llama-3.1-8B int8 Commit window's legs on one GPU, one after another: pouw_h1_gpu.py --workload fill --served
# --commit --negatives --receipts --credit --live-dump (h1u.cu's -hbc build), under `gpu-lease 1`. LEGS: space-separated
# v1 | j10r | v1-live | j10r-live (j10r: --j10r, J10-R's tile unit). A -live leg draws every window's coins from a live
# gate session (--peer) and needs GATE (the lane's forced-command ssh line to vy-challenger-1), CHALLENGER (the window's
# 127.0.0.1:PORT) and PEER (this node's name in the challenger's /etc/vy/peers). It opens a period at its Commit window
# (--period, PERIOD, default commit-<UTC>-<leg>: cap 1, its circuit, draw K, T and --length, the --declare's fixed length
# that core's credit holds the window to, and --circuit-files from CIRCUITS: a pouw_h1_circuits --live 32,4 directory
# of the declared Program, sent with the run), closed once that window's session has ended; it refuses to start while the
# key's lane (GATE_LANE, default vllm) has a period open, and saves each live leg's Commit session tree (GATE fetch,
# $OUT/<leg>/commit-session.tgz) for pouw_h1_gpu.py --check-challenger. The units kernel is built for the node's card
# (arch.sh: ARCH, 120a or 90a); LEG_ARGS are more pouw_h1_gpu.py flags for every leg (another int8 pin: --base-role
# LLAMA32_1B --role LLAMA32_1B_INT8 --repo verity-int8/Llama-3.2-1B-INT8) and PY the node's Python (default
# /workspace/jobs/venv312's; an H200 node's is /workspace/cp/venv312's).
#
#   research run --on NODE --project verity --source . --cwd source --mem-gb 128 --timeout 4h --declared-output 'out/*' \
#     --env LEGS=j10r-live [--env GATE=... --env CHALLENGER=127.0.0.1:7811 --env PEER=NODE --env CIRCUITS=DIR] \
#     -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by LANE \
#     --run-max 12600 -- bash benchmarks/pouw/served_h1/commit_legs.sh
set -u
. "$(dirname "$0")/arch.sh"
S=$(mktemp -d); trap 'rm -rf "$S" || true' EXIT
/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GENCODE" -Xcompiler -fPIC \
  -shared -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 -o "$S/h1u_hbc.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-hbc.log" 2>&1 || exit 1
sha256sum "$S/h1u_hbc.so" | tee "$OUT/builds.txt"
[ -n "${GATE:-}" ] && $GATE pubkey > "$OUT/gate-pubkey.json"
if [ -n "${GATE:-}" ]; then
  # periods of a lane never overlap: the gate refuses a -live leg's period-open while the key's lane (GATE_LANE) has one open
  $GATE ledger > "$OUT/gate-ledger-before.jsonl"
  python3 - "$OUT/gate-ledger-before.jsonl" "${GATE_LANE:-vllm}" > "$OUT/open-periods.txt" <<'PY' || { cat "$OUT/open-periods.txt"; exit 3; }
import json, sys
o = {}
for l in open(sys.argv[1]):
    e = json.loads(l); b = e.get("body") or {}
    if e.get("kind") == "period-open" and b.get("lane") == sys.argv[2]: o[b.get("period")] = b.get("lane")
    if e.get("kind") == "period-close": o.pop(b.get("period"), None)
print("[legs] lane", sys.argv[2], "open periods:", json.dumps(o)); sys.exit(1 if o else 0)
PY
fi
[ -n "${CIRCUITS:-}" ] && CIRCUITS=$(realpath "$CIRCUITS")
cd integrations/vllm
rc_all=0
for leg in ${LEGS:-v1}; do
  mkdir -p "$OUT/$leg"
  read -r -a extra <<< "${LEG_ARGS:-}"
  case "$leg" in j10r*) extra+=(--j10r) ;; esac
  case "$leg" in *-live) extra+=(--challenger "${CHALLENGER:?}" --gate "${GATE:?}" --peer "${PEER:?}"
                                 --period "${PERIOD:-commit-$(date -u +%Y%m%dT%H%M%SZ)-$leg}" --circuits "${CIRCUITS:?}") ;; esac
  echo "[legs] leg $leg start $(date -u +%FT%TZ)"
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-commit-legs} gpu-lease 1 --wait --max-min 120 --mem-gb 96 -- \
    env PYTHONPATH="$PWD/../..:$PWD/../../tools/config:$PWD" HF_HOME=/workspace/hf "${PY:-/workspace/jobs/venv312/bin/python}" -m tests.protocols.pouw_h1_gpu \
    --workload fill --served --commit --negatives --receipts "$S/h1u_hbc.so" "${extra[@]}" --credit \
    --live-dump "$OUT/$leg/live" --declare 32,3,32,4 --out "$OUT/$leg" --at 2026-10-21T00:00:00Z --by "${BY:-commit-legs}" \
    > "$OUT/$leg/stdout.log" 2> "$OUT/$leg/stderr.log"
  rc=$?
  echo "[legs] leg $leg rc $rc $(date -u +%FT%TZ)" | tee -a "$OUT/legs.txt"
  [ $rc -ne 0 ] && rc_all=$rc
done
if [ -n "${GATE:-}" ]; then
  $GATE ledger > "$OUT/gate-ledger.jsonl"; $GATE pins > "$OUT/gate-pins.jsonl"
  # each live leg's Commit session tree (GATE fetch S), what pouw_h1_gpu --check-challenger reads off the gate host
  for leg in ${LEGS:-v1}; do
    case "$leg" in *-live)
      s=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["h1_commit"]["session"]["session"])' \
            "$OUT/$leg/pouw_h1_gpu.json") && $GATE fetch "$s" > "$OUT/$leg/commit-session.tgz"
      echo "[legs] leg $leg commit session ${s:-?} fetched rc $?" | tee -a "$OUT/legs.txt" ;;
    esac
  done
fi
exit $rc_all
