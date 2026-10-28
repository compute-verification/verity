#!/usr/bin/env bash
# The served-rows tie in zero knowledge (served_tie_zk.py, or served_statics_zk.py) on a sealed served window, on a CPU machine, on
# hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify) with the ZK server's Lean
# frame and the sampled-proofs verifier built in the tree. From a clone, the outputs in $RESEARCH_RUN_DIR/out, in up to
# 180 minutes:
#   STZK_MEM_GB=56 STZK_RUN=RUN_DIR STZK_WINDOW=5 bash benchmarks/prove_path/served_tie_zk.sh
# env: STZK_MEM_GB (what job.sh's stages may take, within the slot's)  STZK_RUN, STZK_WINDOW (the sealed run and window)
#      STZK_FORM (tie, the default: served_tie_zk.py, the routed tie; statics: served_statics_zk.py, the rows held to
#      their Programs' statics)  STZK_ARGS (the driver's prove flags: --g, and the routed form's --blocks, --per)
#      STZK_CACHE (the build cache, kept across runs where writable; default pouw-hidden-zk's)  STZK_DRY=1 (the window
#      and its references only)
# Kept in $OUT/stzk: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=stzk HZ_MEM_GB=${STZK_MEM_GB:?}
STZK_CACHE=${STZK_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$STZK_CACHE" 2>/dev/null && [ -w "$STZK_CACHE" ]; then export HZ_CACHE=$STZK_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "stzk: no firewall frame"; exit 7; }

DRIVER=benchmarks/prove_path/served_${STZK_FORM:-tie}_zk.py
[ -f "$DRIVER" ] || { log "stzk: no driver $DRIVER (STZK_FORM is tie or statics)"; exit 2; }
log "stzk: proving ${STZK_RUN:?} w${STZK_WINDOW:?} with $DRIVER ${STZK_ARGS:-}${STZK_DRY:+, dry}"
# shellcheck disable=SC2086
"$PY" "$DRIVER" prove --serve-run "$STZK_RUN" --window "$STZK_WINDOW" ${STZK_ARGS:-} \
  --bin "$BIN" --lean "$LEAN" --out "$W/stzk" ${STZK_DRY:+--dry} 2>&1 | tee -a "$OUT/stzk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/stzk"
(cd "$W/stzk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/stzk/")
log "stzk: done, rc $rc"
exit "$rc"
