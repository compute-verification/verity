#!/usr/bin/env bash
# The network chain's run of record (chain_record.py) on a CPU machine, from the gate's fetched tarballs, on hidden_zk's
# builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify, the verifier of record) with the
# firewall's Lean frame, N2's network-records and the sampled-proofs verifier built in the tree. From a clone, in a slot
# of network-accounting's pool:
#   research run --on vy-nebius-cpu-2 --source . --project verity --as LANE --cwd clone --timeout 240m \
#     --send GATE.tgz -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- env CR_MEM_GB=128 CR_GATES=GATE.tgz bash benchmarks/network_traces/chain_record.sh
# Steps, each kept under $OUT and read by the report:
#   custody  each gate run's files against its end.json (chain_record.py custody)
#   certs    the client's kept certificates.jsonl (or certified.jsonl) and clocks.jsonl, verified (chain_record.py certs)
#   n2       N2's units proved by C-Flock --zk and decided by its verifier of record, Lean included (n2_zk.py prove, or
#            CR_N2_CMD: N2-ZK's sampled draw on signed clocks)
#   scan     the scan's chained units, --zk, held by Lean verify --zk --registered (scan_zk.py prove, with
#            --warden-run when CR_WARDEN_RUN names the gate's warden run and scan_zk takes it: IO-TIE)
#   report   $OUT/report/report.json and report.md (chain_record.py report)
# env: CR_MEM_GB (what job.sh's stages may take, within the slot's)
#      CR_GATES (space-separated: a tarball's name under the run's inputs/, an art: id, or a directory)
#      CR_CLIENT (the client's kept files, a path under the unpacked gates; by default the directory holding
#      certificates.jsonl, else the certifier rehearsal's honest directory)  CR_DEPLOYED=1 (certs refuses a key that
#      isn't LMS at lms.DEPLOYED, H = 20)
#      CR_N2_ROUNDS (n2_zk.py's --rounds)  CR_N2_JOBS (members proved at once, 2)  CR_N2_CMD (replaces n2_zk.py prove;
#      run with HONEST, CLIENT, BIN, LEAN and N2OUT set, writing N2OUT/summary.json)
#      CR_WARDEN_RUN (the warden rehearsal's run, a path under the unpacked gates)  CR_SCAN_ARGS (more scan_zk.py
#      arguments, e.g. "--draw 8")  CR_SCAN_CMD (replaces scan_zk.py prove; SCANOUT/summary.json)
#      CR_SERIAL=1 (n2 then scan; at once by default)  CR_CACHE (the build cache; hidden_zk's by default)
# Kept in $OUT: the steps' records and logs, the public files and everything each verifier received; never a prover's
# file (inst-*, prover-instances.bin, instances.bin) nor a circuit (its pin is kept), a session's archive or a signer's
# state.
set -uo pipefail
export HZ_CONFIGS=chain HZ_MEM_GB=${CR_MEM_GB:?}
CR_CACHE=${CR_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$CR_CACHE" 2>/dev/null && [ -w "$CR_CACHE" ]; then export HZ_CACHE=$CR_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
CR=benchmarks/network_traces/chain_record.py

# ---- the gates -------------------------------------------------------------------------------------------------------
G=$W/gates
mkdir -p "$G"
k=0
for g in ${CR_GATES:?CR_GATES: the gate tarballs, art: ids or directories}; do
  k=$((k + 1)); d=$G/g$k; mkdir -p "$d"
  case $g in
    art:*) research data fetch "$g" --to "$d" > "$OUT/fetch-g$k.log" 2>&1 || { log "chain: fetch $g FAILED"; exit 3; } ;;
    *) f=$g; [ -e "$f" ] || f=${RESEARCH_RUN_DIR:?}/inputs/$g
       if [ -d "$f" ]; then cp -a "$f/." "$d/"; else tar -xf "$f" -C "$d" || { log "chain: no tarball $g"; exit 3; }; fi ;;
  esac
  log "chain: gate $k $g: $(find "$d" -name end.json | sed "s|$d/||" | tr '\n' ' ')"
done
HONEST=$(find "$G" -path '*/out/honest/plan.json' -print -quit); HONEST=${HONEST%/plan.json}
[ -n "$HONEST" ] || { log "chain: no certifier rehearsal (out/honest/plan.json) in $CR_GATES"; exit 3; }
if [ -n "${CR_CLIENT:-}" ]; then CLIENT=$(find "$G" -path "*$CR_CLIENT" -print -quit)
else CLIENT=$(find "$G" -name certificates.jsonl -printf '%h\n' -quit); CLIENT=${CLIENT:-$HONEST}; fi
WARDEN=; [ -z "${CR_WARDEN_RUN:-}" ] || WARDEN=$(find "$G" -path "*$CR_WARDEN_RUN" -print -quit)
log "chain: certifier run ${HONEST#$G/}, client ${CLIENT#$G/}, warden run ${WARDEN:-none}"
mkdir -p "$OUT/gates"
(cd "$G" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' \) -size -64M -print0 \
  | xargs -0 -r -I{} cp --parents {} "$OUT/gates/")

setup
"$PY" "$CR" custody --gates "$G" --out "$OUT/custody.json" 2>&1 | tee -a "$OUT/job.log" | cut -c1-2000
"$PY" "$CR" certs --run "$HONEST" --client "$CLIENT" ${CR_DEPLOYED:+--deployed} --out "$OUT/certs.json" 2>&1 \
  | tee -a "$OUT/job.log" | cut -c1-2000
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
(cd verity/core/protocols/network/lean && lake build network-records) > "$OUT/build-network-records.log" 2>&1 \
  || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$SRC/verity/core/protocols/network/lean/.lake/build/bin/network-records
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "chain: no firewall frame"; exit 7; }

# ---- n2 and scan -------------------------------------------------------------------------------------------------------
keep() {   # keep FROM TO: the public files and what the verifier received, never a prover's file, circuit or archive
  mkdir -p "$2"
  (cd "$1" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name prover-instances.bin ! -name instances.bin \
    ! -name signer-state.json ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$2/")
}
n2() {
  export HONEST CLIENT BIN LEAN N2OUT=$W/n2
  log "chain: n2 ${CR_N2_CMD:+(CR_N2_CMD) }on ${HONEST#$G/}"
  if [ -n "${CR_N2_CMD:-}" ]; then bash -c "$CR_N2_CMD"
  else "$PY" benchmarks/network_traces/n2_zk.py prove --run "$HONEST" --bin "$BIN" --lean "$LEAN" --out "$N2OUT" \
    --jobs "${CR_N2_JOBS:-2}" ${CR_N2_ROUNDS:+--rounds "$CR_N2_ROUNDS"}; fi > "$OUT/n2.log" 2>&1
  echo $? > "$W/n2.rc"; keep "$N2OUT" "$OUT/n2"; log "chain: n2 done, rc $(cat "$W/n2.rc")"
}
scan() {
  export WARDEN BIN LEAN SCANOUT=$W/scan
  local wr=
  if [ -n "$WARDEN" ]; then
    if "$PY" benchmarks/network_traces/scan_zk.py prove --help 2>/dev/null | grep -q -- --warden-run; then wr="--warden-run $WARDEN"
    else log "chain: scan_zk takes no --warden-run in this tree, so the scan proves its own run"; fi
  fi
  log "chain: scan ${CR_SCAN_CMD:+(CR_SCAN_CMD) }${wr:-on the synthetic warden run of scan_zk}"
  if [ -n "${CR_SCAN_CMD:-}" ]; then bash -c "$CR_SCAN_CMD"
  else "$PY" benchmarks/network_traces/scan_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$SCANOUT" $wr \
    ${CR_SCAN_ARGS:-}; fi > "$OUT/scan.log" 2>&1
  echo $? > "$W/scan.rc"; keep "$SCANOUT" "$OUT/scan"; log "chain: scan done, rc $(cat "$W/scan.rc")"
  [ -n "$wr" ] && echo "$WARDEN" > "$W/scan.warden"
}
if [ -n "${CR_SERIAL:-}" ]; then n2; scan; else n2 & scan & wait; fi

# ---- the report --------------------------------------------------------------------------------------------------------
"$PY" - "$OUT/job.json" "$G" <<EOF
import json, os, sys
json.dump({"run": os.environ.get("RESEARCH_RUN_ID"), "tree": "${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}",
           "gates": "${CR_GATES}".split(), "lean": "$(sha256sum "$LEAN" | cut -c1-64) $LEAN", "bin": "$(sha256sum "$BIN" | cut -c1-64)",
           "n2_cmd": os.environ.get("CR_N2_CMD") or "n2_zk.py prove", "scan_cmd": os.environ.get("CR_SCAN_CMD") or "scan_zk.py prove",
           "rc": {"n2": open("$W/n2.rc").read().strip(), "scan": open("$W/scan.rc").read().strip()}}, open(sys.argv[1], "w"), indent=1)
EOF
"$PY" "$CR" report --custody "$OUT/custody.json" --certs "$OUT/certs.json" --n2 "$OUT/n2/summary.json" \
  --scan "$OUT/scan/summary.json" --job "$OUT/job.json" $( [ -f "$W/scan.warden" ] && echo --warden-run "$(cat "$W/scan.warden")") \
  --out "$OUT/report" | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
cat "$OUT/report/report.md" | tee -a "$OUT/job.log" | cut -c1-600
log "chain: done, rc $rc"
exit "$rc"
