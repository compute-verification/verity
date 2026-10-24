#!/usr/bin/env bash
# The warden's bucket rows read by the proof system (warden_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh,
# sourced: C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame. From a clone:
#   research run --on NODE --source . --project verity --as LANE --cwd clone --cpus 16 --mem-gb 64 --timeout 180m \
#     --env WZ_MEM_GB=64 -- env CHECK_SLOTS=SLOTS python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/warden_zk.sh
# The run: a warden-serve rehearsal against the fake endpoint (`warden_serve.py rehearse --fake`), whose warden commits
# and signs each window's bucket rows while serving; then `warden_zk.py prove` on it.
# env: WZ_MEM_GB (what job.sh's stages may take, within the slot's)  WZ_REQUESTS (rehearse's --requests, 6)
#      WZ_UNITS (warden_zk.py's --units, 16)  WZ_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  WZ_DRY=1
#      (the statements and the host's read of them only)
# Kept in $OUT/run: the rehearsal's public files (deployment, certified records, registrations, check; never its records
# or salts); $OUT/wzk: warden_zk's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=wzk HZ_MEM_GB=${WZ_MEM_GB:?}
WZ_CACHE=${WZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$WZ_CACHE" 2>/dev/null && [ -w "$WZ_CACHE" ]; then export HZ_CACHE=$WZ_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "wzk: no firewall frame"; exit 7; }

RUN=$W/run
log "wzk: warden-serve rehearsal (fake endpoint, requests ${WZ_REQUESTS:-6})"
(cd benchmarks/network_traces && "$PY" warden_serve.py rehearse --fake --out "$RUN" --requests "${WZ_REQUESTS:-6}" \
  --rate 1 --tail 5 --max-tokens 64 --prompt-max 64 --start-in 2) 2>&1 | tee -a "$OUT/wzk.log" | tail -3
[ -f "$RUN/check.json" ] || { log "wzk: rehearsal FAILED"; exit 7; }
mkdir -p "$OUT/run"
for f in deployment.json certified.jsonl cells.jsonl check.json warden.json; do [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/run/"; done
log "wzk: proving, units ${WZ_UNITS:-16}${WZ_DRY:+, dry}"
"$PY" benchmarks/network_traces/warden_zk.py prove --run "$RUN" --bin "$BIN" --lean "$LEAN" --out "$W/wzk" \
  --units "${WZ_UNITS:-16}" ${WZ_DRY:+--dry} 2>&1 | tee -a "$OUT/wzk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/wzk"
(cd "$W/wzk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/wzk/")
log "wzk: done, rc $rc"
exit "$rc"
