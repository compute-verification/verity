#!/usr/bin/env bash
# The network chain's run of record (chain_record.py) on a CPU machine, from the gate's fetched tarballs, on hidden_zk's
# builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify, the verifier of record) with the
# firewall's Lean frame, N2's network-records and the sampled-proofs verifier built in the tree. From a clone, in one of
# the network slots:
#   research run --on vy-nebius-cpu-2 --source . --project verity --as LANE --cwd clone --timeout 240m \
#     --send GATE.tgz -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- env CR_MEM_GB=128 CR_GATES=GATE.tgz bash benchmarks/network_traces/chain_record.sh
# Steps, each kept under $OUT and read by the report:
#   custody  each gate run's files against its end.json (chain_record.py custody)
#   certs    the client's kept certificates.jsonl (or certified.jsonl) and clocks.jsonl, verified (chain_record.py certs)
#   endpoint with CR_ENDPOINT, the endpoint's own advice bound to the run and the I/O tie on it, both in the clear
#            (network-e2e's e2e.py --no-prove: chain.served_advice and chain.tie)
#   n2       N2's units proved by C-Flock --zk and decided by its verifier of record, Lean included: CR_N2_CMD
#            (a sampled draw on signed clocks); else on the advice profile n2_advised_zk.py --e2e on the bound
#            advice and the certifier's clock certificates, and on the others n2_zk.py prove
#   scan     the scan's chained units, --zk, held by Lean verify --zk --registered (scan_zk.py prove, with
#            --warden-run when CR_WARDEN_RUN names the gate's warden run and scan_zk takes it)
#   tie      on the warden path, the I/O tie on the warden's ingress, proved (n2_tie_zk.py prove --class warden)
#   report   $OUT/report/report.json and report.md (chain_record.py report)
#   guard    on every exit, before custody publishes the run dir, the record's leak guard (leak_guard.py): each file in it
#            holding a gate run's witness moved to the quarantine, the run refused by name (exit 3; $OUT/leak-guard.json)
# CR_PATH=warden is the warden path: every step reads the active warden's own run (CR_WARDEN_RUN, or the one gate run
# holding a certified.jsonl) and nothing of a certifier rehearsal's, which in CR_GATES is refused by name: certs
# --warden-run, endpoint (chain_record.py endpoint, the advice bound to the certified I/O off the warden's cells),
# n2_zk.py prove --warden-run, the tie and the scan; the report requires every one (chain_record.WARDEN).
# env: CR_MEM_GB (what job.sh's stages may take, within the slot's)
#      CR_GATES (space-separated: a tarball's name under the run's inputs/, an art: id, or a directory)
#      CR_CLIENT (the client's kept files, a path under the unpacked gates; by default the certifier rehearsal's
#      own client/, else its honest directory, never a file of another gate run)  CR_DEPLOYED=1 (certs refuses a key that
#      isn't LMS at lms.DEPLOYED, H = 20)
#      CR_ENDPOINT (the endpoint's records, network_e2e chain.ENDPOINT's four files: a tarball's name under inputs/,
#      an art: id, or a directory; on the warden path it may be the served node's endpoint report instead, a
#      report.json beside no advice_trace.jsonl, which the endpoint step then reads rather than runs)
#      CR_N2_ROUNDS (n2_zk.py's --rounds)  CR_N2_WINDOWS (n2_advised_zk.py's --windows)
#      CR_N2_MEMBERS (its --members)  CR_N2_STANDIN=simulated (a dry run's: on the advice profile when the
#      advice doesn't bind, the advised unit on n2_advised_zk's simulated link, labelled a stand-in)
#      CR_N2_ARGS (more n2_advised_zk.py arguments, e.g. "--draw 4", a sampled period on the verifier's own coins)
#      CR_N2_CMD (replaces the above; run with HONEST, CLIENT, BIN, LEAN, E2E and N2OUT set, writing N2OUT/summary.json)
#      CR_WARDEN_RUN (the warden rehearsal's run, a path under the unpacked gates)  CR_SCAN_ARGS (more scan_zk.py
#      arguments, e.g. "--draw 8")  CR_SCAN_CMD (replaces scan_zk.py prove; SCANOUT/summary.json)
#      CR_PATH=certifier-only (the records are the certifier rehearsal's alone, with no warden rows: the scan isn't run
#      or required, and the report names what that path stubs, chain_record.STUBBED)
#      CR_PATH=warden (above)  CR_EOS_IDS (the warden path's endpoint step: the model's EOS token ids, comma-separated,
#      e.g. 128001 for Llama-3.2-1B; a warden run records none)  CR_TIE_WINDOWS (the tie's link-windows, the run's all
#      by default)  CR_TIE_GPU=1 and CR_N2_GPU=1 (the tie's and N2's provers on the GPU)  CR_N2_DRAW (n2_zk.py's
#      --draw K: N2's period sampled, K of the link-windows drawn on the verifier's coins; every one by default)
#      CR_AUDIT (the served units' compute audit windows, space-separated directories: the report's --audit, d1's K)
#      CR_PERIOD_ANSWER (certifier/v0's network-period check answer on the period, X.answer.json with its case
#      X.case.json beside it, a path or a name under inputs/) and CR_SERVER (the node the block ran on): network-period
#      is built, and an accepted report whose answer is this chain's and network-period's own on its case also writes
#      $OUT/report/block.json, the block record's certifier entry (chain_record.block_entry)
#      CR_SCAN_ONLY=1 (the scan alone, its report given no N2 summary: "scan only, N2 not given"; the record's N2 runs
#      in jobs of its own)  CR_CACHE (the build cache; hidden_zk's by default)
#      CR_QUARANTINE (where what the record mustn't hold goes, 0700 and outside the run dir: by default quarantine/RUN
#      beside the runs root, /workspace/research/quarantine/RUN on a node, kept until its owner removes it)
# n2 runs, then (on the warden path) the tie, then scan: outer proofs go through the firewall one at a time
# (n2_zk.ONE_SESSION), so no two of their provers' firewalled sessions overlap.
# Kept in $OUT: the steps' records and logs, the public files and everything each verifier received; never a prover's
# file (inst-*, prover-instances.bin, instances.bin) nor a circuit (its pin is kept), a session's archive, a signer's
# state, a gate run's witness/ (custody uploads all of $OUT; a warden run's witness is its lane key's only) nor a
# rehearsal's request values (plan.json's and probe.json's requests, io.json's certified I/O, out-witness.jsonl's rows).
# Custody publishes the whole run dir once this exits (out/, the runner's logs, and undeclared files, inputs/ among
# them), so none of it may hold the gates' witness: a gate sent under inputs/ is moved to the quarantine once unpacked,
# and on every exit leak_guard.py moves out whatever else holds a piece of it. On a signal (a timeout's SIGTERM leaves
# seconds, not a scan) or when the guard can't run, out/, inputs/ and the logs are withheld whole instead.
set -uo pipefail
export HZ_CONFIGS=chain HZ_MEM_GB=${CR_MEM_GB:?}
CR_CACHE=${CR_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$CR_CACHE" 2>/dev/null && [ -w "$CR_CACHE" ]; then export HZ_CACHE=$CR_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python:$SRC/tools/research/src
CR=benchmarks/network_traces/chain_record.py

# ---- the record's leak guard ---------------------------------------------------------------------------------------
G=$W/gates RUN=$RESEARCH_RUN_DIR SIGNALED=
mkdir -p "$G"
Q=${CR_QUARANTINE:-$(dirname "$(dirname "$RUN")")/quarantine/$(basename "$RUN")}
withhold() {   # withhold WHY: out/, inputs/ and the logs moved to the quarantine whole, deleted if they can't be moved
  log "chain: $1: out/, inputs/ and the logs withheld from the record (in $Q/files)"
  local f
  for f in out inputs stdout.log stderr.log; do
    [ -e "$RUN/$f" ] || continue
    mv "$RUN/$f" "$Q/files/" 2>/dev/null || rm -rf "${RUN:?}/$f"
  done
}
leak_guard() {   # on every exit: the guard on the run dir, then job.sh's removal of $W (the unpacked gates with it)
  local rc=$? g
  # a signal now ends the guard (its exit then withholds), not this handler
  trap - EXIT; trap 'SIGNALED=SIGTERM' TERM; trap 'SIGNALED=SIGINT' INT
  if [ -n "$SIGNALED" ]; then withhold "$SIGNALED ended the run before the leak guard read it"
  else
    "${PY:-python3}" "$SRC/benchmarks/network_traces/leak_guard.py" "$RUN" "$Q" --gates "$G" \
      --report "$OUT/leak-guard.json" > "$W/leak-guard.out" 2> "$W/leak-guard.err"
    g=$?
    if [ "$g" = 0 ] || [ "$g" = 3 ]; then log "chain: $(cat "$W/leak-guard.out")"
    else withhold "the leak guard failed (exit $g: $(tail -n 1 "$W/leak-guard.err" | cut -c1-300))"; fi
    [ "$g" = 0 ] || [ "$rc" != 0 ] || rc=3
  fi
  rm -rf "$W"
  exit "$rc"
}
trap 'SIGNALED=SIGTERM; exit 143' TERM
trap 'SIGNALED=SIGINT; exit 130' INT
trap leak_guard EXIT
mkdir -p "$Q/files" && chmod 700 "$Q" || { log "chain: no quarantine at $Q"; exit 3; }

# ---- the gates -------------------------------------------------------------------------------------------------------
unpack() {   # unpack SRC DIR: a tarball under the run's inputs/ (or a path), an art: id, or a directory, into DIR
  mkdir -p "$2"
  case $1 in
    art:*) research data fetch "$1" --to "$2" > "$OUT/fetch-${2##*/}.log" 2>&1 || { log "chain: fetch $1 FAILED"; exit 3; } ;;
    *) local f=$1; [ -e "$f" ] || f=${RESEARCH_RUN_DIR:?}/inputs/$1
       if [ -d "$f" ]; then cp -a "$f/." "$2/"; else tar -xf "$f" -C "$2" || { log "chain: no tarball $1"; exit 3; }; fi ;;
  esac
}
k=0
for g in ${CR_GATES:?CR_GATES: the gate tarballs, art: ids or directories}; do
  k=$((k + 1)); unpack "$g" "$G/g$k"
  sent=$g; [ -e "$sent" ] || sent=$RUN/inputs/$g
  if [ -e "$sent" ] && [[ $(realpath "$sent") == "$(realpath "$RUN")"/* ]]; then
    rel=$(realpath --relative-to="$(realpath "$RUN")" "$sent"); mkdir -p "$(dirname "$Q/files/$rel")"
    mv "$sent" "$Q/files/$rel" && log "chain: gate $k's $rel, which holds its witness, moved to the quarantine"
  fi
  log "chain: gate $k $g: $(find "$G/g$k" -name end.json | sed "s|$G/g$k/||" | tr '\n' ' ')"
done
if [ "${CR_PATH:-}" = warden ]; then
  PROXY=$(find "$G" \( -name plan.json -o -name calibration.json -o -name proxy.json.gz \) -print -quit)
  [ -z "$PROXY" ] || { log "chain: the warden path reads no certifier rehearsal's run (its records through" \
    "active.Proxy), and ${PROXY#$G/} is one's"; exit 3; }
  if [ -n "${CR_WARDEN_RUN:-}" ]; then WARDEN=$(find "$G" -path "*$CR_WARDEN_RUN" -print -quit)
  else WARDEN=$(find "$G" -name certified.jsonl -printf '%h\n' -quit); WARDEN=${WARDEN%/witness}; fi
  [ -n "$WARDEN" ] || { log "chain: the warden path: no warden run (a certified.jsonl) in $CR_GATES"; exit 3; }
  HONEST= CLIENT= PROFILE=
  log "chain: the warden path, warden run ${WARDEN#$G/}"
else
  HONEST=$(find "$G" -path '*/out/honest/plan.json' -print -quit); HONEST=${HONEST%/plan.json}
  [ -n "$HONEST" ] || { log "chain: no certifier rehearsal (out/honest/plan.json) in $CR_GATES"; exit 3; }
  if [ -n "${CR_CLIENT:-}" ]; then CLIENT=$(find "$G" -path "*$CR_CLIENT" -print -quit)
  else CLIENT=$HONEST/client; [ -d "$CLIENT" ] || CLIENT=$HONEST; fi
  WARDEN=; [ -z "${CR_WARDEN_RUN:-}" ] || WARDEN=$(find "$G" -path "*$CR_WARDEN_RUN" -print -quit)
  log "chain: certifier run ${HONEST#$G/}, client ${CLIENT#$G/}, warden run ${WARDEN:-none}"
fi
mkdir -p "$OUT/gates"
(cd "$G" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' \) -size -64M ! -path '*/witness/*' \
  ! -name plan.json ! -name probe.json ! -name io.json ! -name out-witness.jsonl -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/gates/")

setup
"$PY" "$CR" custody --gates "$G" --out "$OUT/custody.json" 2>&1 | tee -a "$OUT/job.log" | cut -c1-2000
if [ "${CR_PATH:-}" = warden ]; then
  "$PY" "$CR" certs --warden-run "$WARDEN" ${CR_DEPLOYED:+--deployed} --out "$OUT/certs.json" 2>&1 \
    | tee -a "$OUT/job.log" | cut -c1-2000
else
  "$PY" "$CR" certs --run "$HONEST" --client "$CLIENT" ${CR_DEPLOYED:+--deployed} --out "$OUT/certs.json" 2>&1 \
    | tee -a "$OUT/job.log" | cut -c1-2000
  PROFILE=$("$PY" -c 'import json, sys; print((json.load(open(sys.argv[1])).get("occupancy") or {}).get("kind"))' \
    "$HONEST/plan.json")
fi
E2E=$W/e2e
if [ -n "${CR_ENDPOINT:-}" ] && [ "${CR_PATH:-}" = warden ]; then
  unpack "$CR_ENDPOINT" "$W/endpoint"
  EP=$(find "$W/endpoint" -name advice_trace.jsonl -printf '%h\n' -quit)
  REP=$(find "$W/endpoint" -name report.json -print -quit)
  mkdir -p "$OUT/endpoint"
  if [ -z "$EP" ] && [ -n "$REP" ]; then
    log "chain: endpoint report ${REP#$W/endpoint/} ($CR_ENDPOINT), the served node's (served_session.py endpoint)"
    cp "$REP" "$OUT/endpoint/report.json"
  else
    [ -n "${CR_EOS_IDS:-}" ] || { log "chain: the warden path's endpoint step needs CR_EOS_IDS, the model's EOS" \
      "token ids (128001 for Llama-3.2-1B)"; exit 3; }
    log "chain: endpoint ${EP:-none} ($CR_ENDPOINT): its advice bound to the certified I/O off ${WARDEN#$G/}'s cells"
    "$PY" "$CR" endpoint --warden-run "$WARDEN" --endpoint "${EP:-$W/endpoint}" --eos "$CR_EOS_IDS" \
      --out "$OUT/endpoint/report.json" > "$OUT/endpoint.log" 2>&1
    log "chain: endpoint done, rc $?"
    cp "${EP:-$W/endpoint}/advice.log" "$OUT/endpoint/" 2>/dev/null
  fi
elif [ -n "${CR_ENDPOINT:-}" ]; then
  unpack "$CR_ENDPOINT" "$W/endpoint"
  EP=$(find "$W/endpoint" -name advice_trace.jsonl -printf '%h\n' -quit)
  log "chain: endpoint ${EP:-none} ($CR_ENDPOINT): its advice bound to ${HONEST#$G/} and the I/O tie, in the clear"
  "$PY" benchmarks/network_e2e/e2e.py --from "${HONEST%/out/honest}" --no-prove --no-negatives --endpoint-run "${EP:-$W/endpoint}" \
    --out "$E2E" --work "$W/e2e-work" > "$OUT/endpoint.log" 2>&1
  log "chain: endpoint done, rc $?"
  mkdir -p "$OUT/endpoint"; cp "$E2E"/report.json "$E2E"/report.md "$E2E"/endpoint/advice.log "$OUT/endpoint/" 2>/dev/null
  [ -d "$E2E/rehearsal" ] || ln -s "${HONEST%/honest}" "$E2E/rehearsal"
fi
BOUND=$("$PY" -c 'import json, sys; b = json.load(open(sys.argv[1]))["chain"]["bits"]; print(int(bool(b["bound"] and b["rebuilt"])))' \
  "$E2E/report.json" 2>/dev/null || echo 0)
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
(cd verity/core/protocols/network/lean && lake build network-records ${CR_PERIOD_ANSWER:+network-period}) \
  > "$OUT/build-network-records.log" 2>&1 || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$SRC/verity/core/protocols/network/lean/.lake/build/bin/network-records
[ -z "${CR_PERIOD_ANSWER:-}" ] || export VERITY_NETWORK_PERIOD_EXE=$SRC/verity/core/protocols/network/lean/.lake/build/bin/network-period
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
  export HONEST CLIENT BIN LEAN E2E N2OUT=$W/n2
  local advised="$PY benchmarks/network_traces/n2_advised_zk.py prove --bin $BIN --lean $LEAN --out $N2OUT \
    ${CR_N2_MEMBERS:+--members $CR_N2_MEMBERS} ${CR_N2_ARGS:-}"
  if [ -n "${CR_N2_CMD:-}" ]; then cmd=$CR_N2_CMD
  elif [ "${CR_PATH:-}" = warden ]; then cmd="$PY benchmarks/network_traces/n2_zk.py prove --warden-run $WARDEN \
    --bin $BIN --lean $LEAN --out $N2OUT ${CR_N2_GPU:+--gpu} ${CR_N2_DRAW:+--draw $CR_N2_DRAW}"
  elif [ "$PROFILE" != advice ]; then cmd="$PY benchmarks/network_traces/n2_zk.py prove --run $HONEST --bin $BIN \
    --lean $LEAN --out $N2OUT ${CR_N2_ROUNDS:+--rounds $CR_N2_ROUNDS}"
  elif [ "$BOUND" = 1 ]; then cmd="$advised --e2e $E2E ${CR_N2_WINDOWS:+--windows $CR_N2_WINDOWS}"
  else
    why="the run is on the advice profile, whose counts N2's advised unit decides on the endpoint's advice, and "
    if [ -z "${CR_ENDPOINT:-}" ]; then why+="no endpoint records were given (CR_ENDPOINT)"
    else why+="the endpoint's advice doesn't bind to the run: $("$PY" -c 'import json, sys
b = json.load(open(sys.argv[1]))["chain"]["bits"]; print((b.get("bound_why") or b.get("why") or "")[:400])' "$E2E/report.json" 2>/dev/null)"; fi
    if [ "${CR_N2_STANDIN:-}" = simulated ]; then
      echo "the advised unit on n2_advised_zk's simulated link (its certifier signs each window's sync), not the gate's: $why" \
        > "$W/n2.standin"; cmd=$advised
    else echo "$why" > "$W/n2.why"; log "chain: n2 not run: $why"; echo 2 > "$W/n2.rc"; return; fi
  fi
  log "chain: n2: $cmd"
  bash -c "$cmd" > "$OUT/n2.log" 2>&1
  echo $? > "$W/n2.rc"; keep "$N2OUT" "$OUT/n2"; log "chain: n2 done, rc $(cat "$W/n2.rc")"
  [ "$(cat "$W/n2.rc")" = 0 ] || tail -n 1 "$OUT/n2.log" | cut -c1-600 > "$W/n2.why"
}
scan() {
  export WARDEN BIN LEAN SCANOUT=$W/scan
  if [ "${CR_PATH:-}" = certifier-only ]; then
    echo "the certifier-only path: no warden rows to scan, so no scan" > "$W/scan.why"; echo 2 > "$W/scan.rc"
    log "chain: scan not run (CR_PATH=certifier-only)"; return
  fi
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
  [ "$(cat "$W/scan.rc")" = 0 ] || tail -n 1 "$OUT/scan.log" | cut -c1-600 > "$W/scan.why"
  [ -n "$wr" ] && echo "$WARDEN" > "$W/scan.warden"
}
tie() {
  [ "${CR_PATH:-}" = warden ] || return 0
  local k=${CR_TIE_WINDOWS:-$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["windows"])' \
    "$(find "$WARDEN" -name deployment.json -print -quit)")}
  local cmd="$PY benchmarks/network_traces/n2_tie_zk.py prove --class warden --warden-run $WARDEN --windows $k \
    --bin $BIN --lean $LEAN --out $W/tie ${CR_TIE_GPU:+--gpu}"
  log "chain: tie: $cmd"
  bash -c "$cmd" > "$OUT/tie.log" 2>&1
  echo $? > "$W/tie.rc"; keep "$W/tie" "$OUT/tie"; log "chain: tie done, rc $(cat "$W/tie.rc")"
  [ "$(cat "$W/tie.rc")" = 0 ] || tail -n 1 "$OUT/tie.log" | cut -c1-600 > "$W/tie.why"
}
if [ -n "${CR_SCAN_ONLY:-}" ]; then scan; else n2; tie; scan; fi

# ---- the report --------------------------------------------------------------------------------------------------------
"$PY" - "$OUT/job.json" "$W" <<EOF
import json, os, sys
from pathlib import Path
w = Path(sys.argv[2])
text = lambda f: (w / f).read_text().strip() if (w / f).is_file() else None
json.dump({"run": os.environ.get("RESEARCH_RUN_ID"), "tree": "${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}",
           "gates": "${CR_GATES}".split(), "endpoint": os.environ.get("CR_ENDPOINT"), "profile": "$PROFILE",
           "path": os.environ.get("CR_PATH"),
           "lean": "$(sha256sum "$LEAN" | cut -c1-64) $LEAN", "bin": "$(sha256sum "$BIN" | cut -c1-64)",
           "n2_cmd": os.environ.get("CR_N2_CMD"), "scan_cmd": os.environ.get("CR_SCAN_CMD"),
           "scan_only": bool(os.environ.get("CR_SCAN_ONLY")),
           "rc": {"n2": text("n2.rc"), "scan": text("scan.rc"), "tie": text("tie.rc")},
           "why": {"n2": text("n2.why"), "scan": text("scan.why"), "tie": text("tie.why")},
           "stand_in": {"n2": text("n2.standin")}}, open(sys.argv[1], "w"), indent=1)
EOF
PA=${CR_PERIOD_ANSWER:-}; [ -z "$PA" ] || [ -e "$PA" ] || PA=${RESEARCH_RUN_DIR:?}/inputs/$PA
"$PY" "$CR" report --custody "$OUT/custody.json" --certs "$OUT/certs.json" --endpoint "$OUT/endpoint/report.json" \
  $( [ -n "${CR_SCAN_ONLY:-}" ] || echo --n2 "$OUT/n2/summary.json") --scan "$OUT/scan/summary.json" --job "$OUT/job.json" \
  $( [ -f "$W/scan.warden" ] && echo --warden-run "$(cat "$W/scan.warden")") ${CR_AUDIT:+--audit $CR_AUDIT} \
  $( [ "${CR_PATH:-}" = warden ] && echo --tie "$OUT/tie/summary.json") \
  ${PA:+--period-answer "$PA"} ${CR_SERVER:+--server "$CR_SERVER"} --out "$OUT/report" | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
cat "$OUT/report/report.md" | tee -a "$OUT/job.log" | cut -c1-600
log "chain: done, rc $rc"
exit "$rc"
