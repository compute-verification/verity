#!/usr/bin/env bash
# N2's head checks in zero knowledge (n2_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced: C-Flock's
# flock-circuit and the Lean flock-verify) with the firewall's Lean frame, N2's network-records and the sampled-proofs
# verifier built in the tree. From a clone:
#   research run --on vy-nebius-cpu-5 --source . --project verity --as LANE --cwd clone --cpus 32 --mem-gb 128 \
#     --env N2_MEM_GB=128 -- bash benchmarks/network_traces/n2_zk.sh
# The runs: certifier-serve's honest and extra runs, from N2_RUNS (a tarball sent with `--send`, holding honest/ and
# extra/ as `certifier_serve.py rehearse` writes them) or else made here by `n2_zk.py virtual` (stub endpoint, injected
# clock).
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_RUNS (the tarball's name under the run's inputs/)
#      N2_REQUESTS (virtual's --requests)  N2_POINTS (n2_zk.py's --points, 1 by default)  N2_CACHE (the build cache, kept
#      across runs where writable)  N2_DRY=1 (the instances and their references only)
# Kept in $OUT/n2zk: n2_zk's records and logs, the public files and everything the verifier received; never a prover's
# file (inst-*) nor a circuit (its pin is kept), nor a session's archive. $OUT/runs: the runs' public files.
set -uo pipefail
export HZ_CONFIGS=n2zk HZ_MEM_GB=${N2_MEM_GB:?}
if [ -n "${N2_CACHE:-}" ] && mkdir -p "$N2_CACHE" 2>/dev/null && [ -w "$N2_CACHE" ]; then export HZ_CACHE=$N2_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
(cd verity/core/protocols/network/lean && lake build network-records) > "$OUT/build-network-records.log" 2>&1 \
  || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$SRC/verity/core/protocols/network/lean/.lake/build/bin/network-records
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2zk: no firewall frame"; exit 7; }

RUNS=$W/runs
if [ -n "${N2_RUNS:-}" ]; then
  mkdir -p "$RUNS" && tar -xzf "${RESEARCH_RUN_DIR:?}/inputs/$N2_RUNS" -C "$RUNS" || { log "n2zk: no runs in $N2_RUNS"; exit 7; }
  log "n2zk: runs from $N2_RUNS: $(ls "$RUNS" | tr '\n' ' ')"
else
  log "n2zk: virtual runs (requests ${N2_REQUESTS:-40})"
  "$PY" benchmarks/network_traces/n2_zk.py virtual --out "$RUNS" --requests "${N2_REQUESTS:-40}" 2>&1 | tee -a "$OUT/n2zk.log" \
    || { log "n2zk: virtual FAILED"; exit 7; }
fi
mkdir -p "$OUT/runs"
(cd "$RUNS" && find . -type f \( -name '*.json' -o -name 'certified.jsonl' \) -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/runs/")
EXTRA=; [ -d "$RUNS/extra" ] && EXTRA=$RUNS/extra
log "n2zk: proving, points ${N2_POINTS:-1}${N2_DRY:+, dry}"
"$PY" benchmarks/network_traces/n2_zk.py prove --run "$RUNS/honest" ${EXTRA:+--extra "$EXTRA"} --bin "$BIN" --lean "$LEAN" \
  --out "$W/n2zk" --points "${N2_POINTS:-1}" ${N2_DRY:+--dry} 2>&1 | tee -a "$OUT/n2zk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/n2zk"
(cd "$W/n2zk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/n2zk/")
log "n2zk: done, rc $rc"
exit "$rc"
