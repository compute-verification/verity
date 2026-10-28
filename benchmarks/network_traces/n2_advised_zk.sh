#!/usr/bin/env bash
# N2's advised unit in zero knowledge (n2_advised_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame, N2's network-records and the
# sampled-proofs verifier built in the tree, as n2_zk.sh runs n2_zk.py. From a clone, in one of the network
# slots on vy-nebius-cpu-1:
#   research run --on vy-nebius-cpu-1 --source . --project verity --as LANE --cwd clone --timeout 180m \
#     --env N2_MEM_GB=56 -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- bash benchmarks/network_traces/n2_advised_zk.sh
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_SEED (the advised tests' session, 7 by default)
#      N2_E2E (--e2e: network-e2e's output directory, or a .tgz of it, unpacked into the work directory, e.g. one sent
#      with --send to $RESEARCH_RUN_DIR/inputs/)  N2_WINDOWS (--windows, with N2_E2E; every window by default)
#      N2_CLOCK_STAND_IN=1 (--clock-stand-in, with N2_E2E: a run without clocks.jsonl is proved on its proxy log's
#      syncs, a labelled stand-in, instead of refused)
#      N2_ADVICE_FROM_IO=1 (--advice-from-io, with N2_E2E: a run with no endpoint advice, such as a gate's rehearsal,
#      on an advice synthesized from its certified I/O, labelled so)
#      N2_DRAW (--draw K: each member's period sampled, K of its windows drawn after the period is committed and only
#      those proved; every window by default, and the verifier's own coins)  N2_DELTA (--delta)
#      N2_CHALLENGER=local|gate:USER@HOST (--challenger, with N2_DRAW: the draws on the auditor's challenger's coins,
#      one session for every member, pinned on main's first-parent history fetched from the clone's origin); with
#      gate:…, N2_PEER (this node as the gate's /etc/vy/peers names it), N2_CHALLENGER_KEY (the challenger's key record,
#      challenger-key/v1) and N2_GATE_KEY (the lane's gate key)
#      N2_MEMBERS (--members, comma-separated; every member by default)
#      N2_PORT (--port; a free port a session by default; the members are proved one after another)
#      N2_CACHE (the build cache, kept across runs where writable)  N2_DEBUG_CLEAR=1 (--debug-clear)
#      N2_DRY=1 (the instances and references only)
# Kept in $OUT/n2adv: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=n2adv HZ_MEM_GB=${N2_MEM_GB:?}
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
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2adv: no firewall frame"; exit 7; }

E2E=${N2_E2E:-}
if [ -f "$E2E" ]; then
  mkdir -p "$W/e2e" && tar xzf "$E2E" -C "$W/e2e" || exit 7
  # the run's root holds rehearsal/honest; the advice's windows carry report.json files of their own
  E2E=$(find "$W/e2e" -path '*/rehearsal/honest/plan.json' -print -quit)
  [ -n "$E2E" ] || { log "n2adv: $N2_E2E holds no rehearsal/honest/plan.json"; exit 7; }
  E2E=${E2E%/rehearsal/honest/plan.json}
fi
CH=()
if [ -n "${N2_CHALLENGER:-}" ]; then   # the verifier's own copy of main's first-parent history, which the session's pin must be on
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > "$OUT/fetch-main.log" 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> "$OUT/fetch-main.log" 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > "$OUT/main.txt" || { tail -5 "$OUT/fetch-main.log"; exit 4; }
  CH=(--challenger "$N2_CHALLENGER" --main "$OUT/main.txt")
  case $N2_CHALLENGER in gate:*)
    [ -n "${N2_PEER:-}" ] && [ -n "${N2_CHALLENGER_KEY:-}" ] || { log "n2adv: N2_CHALLENGER=gate:… needs N2_PEER and N2_CHALLENGER_KEY"; exit 4; }
    CH+=(--peer "$N2_PEER" --challenger-key "$N2_CHALLENGER_KEY" ${N2_GATE_KEY:+--gate-key "$N2_GATE_KEY"});;
  esac
fi
log "n2adv: proving, ${E2E:+e2e $E2E windows ${N2_WINDOWS:-all}${N2_CLOCK_STAND_IN:+ clock stand-in}${N2_ADVICE_FROM_IO:+ advice from io}, }seed" \
  "${N2_SEED:-7}, members ${N2_MEMBERS:-all}, draw ${N2_DRAW:-all}${N2_CHALLENGER:+ on $N2_CHALLENGER}${N2_DRY:+, dry}"
"$PY" benchmarks/network_traces/n2_advised_zk.py prove --seed "${N2_SEED:-7}" ${E2E:+--e2e "$E2E"} \
  ${N2_WINDOWS:+--windows "$N2_WINDOWS"} ${N2_CLOCK_STAND_IN:+--clock-stand-in} ${N2_ADVICE_FROM_IO:+--advice-from-io} \
  ${N2_DRAW:+--draw "$N2_DRAW"} "${CH[@]}" ${N2_DELTA:+--delta "$N2_DELTA"} \
  ${N2_MEMBERS:+--members "$N2_MEMBERS"} \
  --bin "$BIN" --lean "$LEAN" --out "$W/n2adv" \
  ${N2_PORT:+--port "$N2_PORT"} ${N2_DEBUG_CLEAR:+--debug-clear} ${N2_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/n2adv.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/n2adv"
(cd "$W/n2adv" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/n2adv/")
log "n2adv: done, rc $rc"
exit "$rc"
