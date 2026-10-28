#!/usr/bin/env bash
# network-e2e's runner, `verity bench network_e2e ARGS`, ARGS e2e.py's (README.md has the commands). With --no-prove,
# --proofs DIR or --traffic varied (whose advice profile n2_zk doesn't prove) it runs e2e.py alone (N2's network-records
# is built on first use, with lake on the PATH). Otherwise it
# runs under `research run` on a CPU node: it builds what n2_zk.sh builds (hidden_zk's job.sh, sourced: Flock's
# flock-circuit and the Lean flock-verify, then the firewall's Lean frame, N2's network-records and the sampled-proofs
# verifier in the tree, so --cwd clone) and runs e2e.py with them, the report in $OUT. From a clone:
#   research run --pool research --as LANE --pool-wait 3600 --timeout 5h --source . --project verity --cwd clone \
#     --send RUNS.tgz -- bash -c 'bash benchmarks/network_e2e/run.sh --from "$RESEARCH_RUN_DIR/inputs/RUNS.tgz"'
# env: E2E_MEM_GB (what job.sh's stages may take, within the slot's; 32 by default)  E2E_CACHE (the build cache, kept
#      across runs where writable; by default the node's /workspace/research/cache/pouw-hidden-zk)
# Kept in $OUT: report.json, report.md, e2e.log, the builds' logs, and n2zk/, n2_zk's records and everything the
# verifier received, never a prover's instance file, a circuit or a session's archive (e2e.keep).
set -uo pipefail
cd "$(dirname "$0")/../.."
case " $* " in
  *" --no-prove "* | *" --proofs "* | *" --traffic varied "*)
    export PATH=$HOME/.elan/bin:$PATH
    exec "${PYTHON:-python3}" benchmarks/network_e2e/e2e.py "$@" ;;
esac
: "${RESEARCH_RUN_DIR:?proving runs under research run on a node (benchmarks/network_e2e/README.md); --no-prove decides N2 alone}"
export HZ_CONFIGS=n2zk HZ_MEM_GB=${E2E_MEM_GB:-32}
CACHE=${E2E_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$CACHE" 2>/dev/null && [ -w "$CACHE" ]; then export HZ_CACHE=$CACHE; fi
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
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "network-e2e: no firewall frame"; exit 7; }
log "network-e2e: $*"
"$PY" benchmarks/network_e2e/e2e.py "$@" --bin "$BIN" --lean "$LEAN" --out "$OUT" --work "$W/e2e" 2>&1 \
  | tee -a "$OUT/e2e.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
log "network-e2e: done, rc $rc"
exit "$rc"
