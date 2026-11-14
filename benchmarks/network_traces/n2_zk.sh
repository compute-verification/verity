#!/usr/bin/env bash
# N2's verdict of record from zero-knowledge proofs of its registered proof block (n2_zk.py) on a CPU machine, on
# hidden_zk's builds (its job.sh, sourced: Flock's flock-circuit and the Lean flock-verify) with the ZK server's Lean
# frame, N2's network-records and SPOT's verifier built in the tree. From a clone, as a job of 96 GB whose
# input is the run's tarball n2-run.tar.gz:
#   env N2_MEM_GB=96 N2_RUN=n2-run.tar.gz bash benchmarks/network_traces/n2_zk.sh
# The run: an active network certifier's directory, as its server writes it, from N2_RUN, a tarball sent with `--send`
# holding it at its top.
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_RUN (the tarball's name under the run's inputs/)
#      N2_ROUNDS (n2_zk.py's --rounds, proved one after another: one firewalled
#      session at a time)  N2_DEBUG_CLEAR=1 (each verdict also carries the in-the-clear comparison)  N2_CACHE (the build cache,
#      kept across runs where writable)  N2_DRY=1 (the execution phases and their references only)
# Kept in $OUT/n2zk: n2_zk's records and logs, the public files and everything the verifier received; never a prover's
# file (inst-*, prover-instances.bin) nor a circuit (its pin is kept). $OUT/run: the run's public files, never its
# request values (traffic.json's prompts, client.json's responses).
set -uo pipefail
export HZ_CONFIGS=n2zk HZ_MEM_GB=${N2_MEM_GB:?}
if [ -n "${N2_CACHE:-}" ] && mkdir -p "$N2_CACHE" 2>/dev/null && [ -w "$N2_CACHE" ]; then export HZ_CACHE=$N2_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/protocols/proofs/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/protocols/proofs/spot/verifier/lean && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
(cd verity/protocols/accounting/certifier/verifier/lean && lake build network-records) > "$OUT/build-network-records.log" 2>&1 \
  || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$SRC/verity/protocols/accounting/certifier/verifier/lean/.lake/build/bin/network-records
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2zk: no firewall frame"; exit 7; }

RUN=$W/run
mkdir -p "$RUN" && tar -xzf "${RESEARCH_RUN_DIR:?}/inputs/${N2_RUN:?}" -C "$RUN" || { log "n2zk: no run in $N2_RUN"; exit 7; }
log "n2zk: the run from $N2_RUN: $(ls "$RUN" | tr '\n' ' ')"
mkdir -p "$OUT/run"
for f in deployment.json cells.jsonl certified.jsonl check.json warden.json; do
  [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/run/"
done
log "n2zk: proving${N2_DEBUG_CLEAR:+, debug clear}${N2_DRY:+, dry}"
"$PY" benchmarks/network_traces/n2_zk.py prove --warden-run "$RUN" \
  --bin "$BIN" --lean "$LEAN" --out "$W/n2zk" ${N2_ROUNDS:+--rounds "$N2_ROUNDS"} \
  ${N2_DEBUG_CLEAR:+--debug-clear} ${N2_DRY:+--dry} 2>&1 | tee -a "$OUT/n2zk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/n2zk"
(cd "$W/n2zk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name prover-instances.bin -print0 \
  | xargs -0 -r -I{} cp --parents {} "$OUT/n2zk/")
log "n2zk: done, rc $rc"
exit "$rc"
