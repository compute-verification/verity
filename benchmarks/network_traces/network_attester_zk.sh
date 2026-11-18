#!/usr/bin/env bash
# The network attester's bucket rows read by the proof system (network_attester_zk.py) on a CPU machine, on hidden_zk's builds
# (its job.sh, sourced: Flock's flock-circuit and the Lean verity-verify-proof) with the ZK server's Lean frame. From a clone:
#   env WZ_MEM_GB=64 CHECK_SLOTS=SLOTS python3 tools/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/network_attester_zk.sh
#   as a job of at most 180 minutes on 16 CPUs and 64 GB.
# The run: a network-attester-serve rehearsal against the fake endpoint (`network_attester_serve.py rehearse --fake`), whose network
# attester commits and signs each window's bucket rows while serving; then `network_attester_zk.py prove` on it.
# env: WZ_MEM_GB (what job.sh's stages may take, within the slot's)  WZ_REQUESTS (rehearse's --requests, 6)
#      WZ_UNITS (network_attester_zk.py's --units, 16)  WZ_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  WZ_DRY=1
#      (the statements and the host's read of them only)
# Kept in $OUT/run: what the gate publishes of the rehearsal (`auditor.py publish`: the deployment's public keys,
# the certificates, the cells' registrations less their wiring and cost, and outcome.json); $OUT/wzk:
# network_attester_zk's summary alone. A member's registrations (the network attester's with its wiring, its own
# with its units' connection and cells) and the verifier's registered reads open the wiring, so they stay in $W.
set -uo pipefail
export HZ_CONFIGS=wzk HZ_MEM_GB=${WZ_MEM_GB:?}
WZ_CACHE=${WZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$WZ_CACHE" 2>/dev/null && [ -w "$WZ_CACHE" ]; then export HZ_CACHE=$WZ_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/protocols/proofs/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/protocols/proofs/spot/verifier/lean && lake build spot-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_ZK_SERVER_LEAN=${LEAN%/*}/flock-zk-server
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_ZK_SERVER_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-zk-server)) \
  > "$OUT/build-zk-server.log" 2>&1
[ -x "$FC_ZK_SERVER_LEAN" ] || { tail -20 "$OUT/build-zk-server.log"; log "wzk: no ZK server frame"; exit 7; }

RUN=$W/run
log "wzk: network-attester-serve rehearsal (fake endpoint, requests ${WZ_REQUESTS:-6})"
(cd benchmarks/network_traces && "$PY" -m verity.protocols.resource_accounting.network.attester.network_attester_serve rehearse --fake --out "$RUN" --requests "${WZ_REQUESTS:-6}" \
  --rate 1 --tail 5 --max-tokens 64 --prompt-max 64 --start-in 2) 2>&1 | tee -a "$OUT/wzk.log" | tail -3
[ -f "$RUN/check.json" ] || { log "wzk: rehearsal FAILED"; exit 7; }
mkdir -p "$OUT/run"
"$PY" -m verity.protocols.compose.verifier.live_challenger.auditor publish "$RUN" "$OUT/run" > "$OUT/run/outcome.json" || { log "wzk: the publisher (auditor.py publish) exited $? on $RUN"; exit 6; }
log "wzk: proving, units ${WZ_UNITS:-16}${WZ_DRY:+, dry}"
"$PY" benchmarks/network_traces/zk_drivers.py network_attester_zk prove --run "$RUN" --bin "$BIN" --lean "$LEAN" --out "$W/wzk" \
  --units "${WZ_UNITS:-16}" ${WZ_DRY:+--dry} 2>&1 | tee -a "$OUT/wzk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/wzk"
[ -f "$W/wzk/summary.json" ] && cp "$W/wzk/summary.json" "$OUT/wzk/"
log "wzk: done, rc $rc"
exit "$rc"
