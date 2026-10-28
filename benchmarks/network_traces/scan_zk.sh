#!/usr/bin/env bash
# The step-indexed scan composed in zero knowledge (scan_zk.py, w63's class or SC_CLASS's) on a CPU machine, on hidden_zk's builds (its
# job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the
# sampled-proofs verifier built in the tree. From a clone, in one of the network slots:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env SC_MEM_GB=128 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/scan_zk.sh
# env: SC_MEM_GB (what job.sh's stages may take, within the slot's)  SC_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  SC_DRY=1 (the run and the host's read of every round only)
#      SC_CLASS=w63|test (`prove --class`: test is the certifier tests' class, a CPU window)  SC_GROUP=G (`prove
#      --group G`: link-windows 0 … G − 1 in one registration)
#      SC_WARDEN_RUN=DIR (a warden_serve run's directory, `prove --warden-run`)  SC_REHEARSE=1 (first a rehearsal,
#      `warden_serve.py rehearse --fake --ids` at the class's schedule, SC_REQUESTS requests at SC_RATE a second, the
#      fake's id-form answer SC_FIRST_MS after each (past the warden's delay: a void run), SC_PROMPT_MAX and
#      SC_MAX_TOKENS its traffic's sizes, and prove its run)  SC_DRAW=K (`prove --draw K`: each round draws
#      stratified:K; stratified:N by default)
#      SC_CHALLENGER=local|gate:USER@HOST (required unless SC_DRY: every round's draw on a session of its own of the
#      challenger's, `prove --challenger`, pinned on main's first-parent history fetched from the clone's origin;
#      gate: also SC_PEER, this node as the gate's /etc/vy/peers names it, SC_CHALLENGER_KEY, the deployment's
#      challenger-key/v1 record, and SC_GATE_KEY, the lane's gate key)
#      SC_RSS=1 (each flock-circuit process's own peak RSS, its command and wall, a line each in $OUT/rss.jsonl: a8's
#      max_child_rss_gb is the largest of every session's so far)
#      SC_PORT (`prove --port`, the sessions' first loopback port: by default 7811 + 16 × the lowest core the slot
#      pins the job to, so that runs in two slots of one machine never share a session's port)
# Kept in $OUT/sczk: the driver's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive, nor a signer's state (the deployment's or the tamper's). A
# rehearsal's public files go in $OUT/warden, its records and salts (the prover's) nowhere.
set -uo pipefail
export HZ_CONFIGS=sczk HZ_MEM_GB=${SC_MEM_GB:?}
SC_CACHE=${SC_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$SC_CACHE" 2>/dev/null && [ -w "$SC_CACHE" ]; then export HZ_CACHE=$SC_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "sczk: no firewall frame"; exit 7; }
if [ -n "${SC_RSS:-}" ]; then
  cat > "$W/flock-circuit-rss" <<EOF
#!$PY
import json, resource, subprocess, sys, time
t = time.monotonic()
rc = subprocess.call(["$BIN", *sys.argv[1:]])
with open("$OUT/rss.jsonl", "a") as f:
    f.write(json.dumps({"command": sys.argv[1:3], "rc": rc, "wall_s": round(time.monotonic() - t, 2),
                        "peak_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)}) + "\n")
sys.exit(rc)
EOF
  chmod +x "$W/flock-circuit-rss"
  BIN=$W/flock-circuit-rss
fi

if [ -n "${SC_REHEARSE:-}" ]; then
  log "sczk: rehearsing the warden at scan_zk's schedule"
  # shellcheck disable=SC2046
  "$PY" benchmarks/network_traces/warden_serve.py rehearse --fake --ids --out "$W/warden" \
    --requests "${SC_REQUESTS:-12}" --rate "${SC_RATE:-2}" --tail "${SC_TAIL:-6}" --start-in 2 \
    ${SC_FIRST_MS:+--first-ms "$SC_FIRST_MS"} ${SC_PROMPT_MAX:+--prompt-max "$SC_PROMPT_MAX"} \
    ${SC_MAX_TOKENS:+--max-tokens "$SC_MAX_TOKENS"} \
    $("$PY" benchmarks/network_traces/scan_zk.py schedule ${SC_CLASS:+--class "$SC_CLASS"}) 2>&1 \
    | tee -a "$OUT/warden-rehearse.log" | cut -c1-4000
  log "sczk: rehearsal rc ${PIPESTATUS[0]}"
  mkdir -p "$OUT/warden"
  for f in deployment.json cells.jsonl certified.jsonl check.json client.json warden.json serve.log traffic.json; do
    [ -f "$W/warden/$f" ] && cp "$W/warden/$f" "$OUT/warden/"
  done
  SC_WARDEN_RUN=$W/warden
fi
CH=()
if [ -n "${SC_CHALLENGER:-}" ]; then   # the verifier's own copy of main's first-parent history, which the session's pin must be on
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > "$OUT/fetch-main.log" 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> "$OUT/fetch-main.log" 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > "$OUT/main.txt" || { tail -5 "$OUT/fetch-main.log"; exit 4; }
  CH=(--challenger "$SC_CHALLENGER" --main "$OUT/main.txt")
  case $SC_CHALLENGER in gate:*)
    [ -n "${SC_PEER:-}" ] && [ -n "${SC_CHALLENGER_KEY:-}" ] || { log "sczk: SC_CHALLENGER=gate:… needs SC_PEER and SC_CHALLENGER_KEY"; exit 4; }
    CH+=(--peer "$SC_PEER" --challenger-key "$SC_CHALLENGER_KEY" ${SC_GATE_KEY:+--gate-key "$SC_GATE_KEY"});;
  esac
elif [ -z "${SC_DRY:-}" ]; then
  log "sczk: SC_CHALLENGER (local|gate:USER@HOST) names the challenger every round draws on: nothing draws on the" \
    "verifier's own coins"; exit 4
fi
SC_PORT=${SC_PORT:-$("$PY" -c 'import os; print(7811 + 16 * min(os.sched_getaffinity(0)))')}
log "sczk: proving${SC_CLASS:+ at $SC_CLASS}${SC_GROUP:+, group $SC_GROUP}${SC_DRY:+, dry}${SC_WARDEN_RUN:+, warden run $SC_WARDEN_RUN}${SC_DRAW:+, draw $SC_DRAW}${SC_CHALLENGER:+, every round on a session of $SC_CHALLENGER}, sessions from port $SC_PORT"
"$PY" benchmarks/network_traces/scan_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/sczk" --port "$SC_PORT" ${SC_DRY:+--dry} \
  ${SC_CLASS:+--class "$SC_CLASS"} ${SC_GROUP:+--group "$SC_GROUP"} \
  ${SC_WARDEN_RUN:+--warden-run "$SC_WARDEN_RUN"} ${SC_DRAW:+--draw "$SC_DRAW"} "${CH[@]}" 2>&1 \
  | tee -a "$OUT/sczk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/sczk"
(cd "$W/sczk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'instances.bin' ! -path '*/archive/*' \
  ! -name '*signer-state.json*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/sczk/")
log "sczk: done, rc $rc"
exit "$rc"
