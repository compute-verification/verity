#!/usr/bin/env bash
# PoUW's piece of a served H1 window with its sampled proof blocks proved (served_verdicts.sh's POUW_CMD): drawn.py (the CLI of
# verity_vllm.protocols.pouw.h1.drawn, reading --checkpoint with the live challenger's model registry) stages each
# sampled call's J10-R units on the window's own registered values, and each sampled proof block of a lanes-form K as its lanes and
# combine; each staged statement is proved by one Flock --zk session sampling all its instances (hidden_zk/job.sh's
# `session`, at the slot its circuit file's META names), one session at a time, its prover under the ZK server
# (FC_FIREWALL=1, FC_HOST_WITNESS=1: the Lean session loop flock-firewall, live PROTOCOL.md §10.4, whose Hello is set up
# on the statement's own program.json and partition.json, the Program and partition credit verifies it under, so what
# leaves for the live verifier is the loop's), as h1_prove/job.sh, live.sh and h1_lanes/served.sh prove theirs; the
# live verifier (`serve --zk --operator loopback`) listens on 127.0.0.1 beside it. drawn.py writes the statements file and
# checks it as credit does, and then `served_verdicts.py pouw` judges the window with it as --statements (the Lean verdict
# of record refuses the lanes form until its mirror: H1_DRAWN_ONLY=1 stops at drawn.py's check).
#   POUW_CMD='bash benchmarks/pouw/h1_prove/drawn.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/pouw/h1_prove/drawn.sh SERVE_DIR WINDOW PIECE_DIR
# The window must be served with its receipts (serve-openai --h1-receipts, served_verdicts.sh's H1_RECEIPTS=1): a J10-R
# call statement's o0 is opened at its call's receipt table receipt@<c>, and staging refuses a window without it.
# env: H1_CIRCUITS (pouw_h1_circuits' --out, with the sampled K's k<K> file, --strata K, or its lane<K> and combine<K>
#      files, --lanes K)  H1_CHECKPOINT (the served model's
#      safetensors dir, b/<K>'s rows) or H1_WEIGHTS (an npz of w<i>, s<i>)  H1_GATE and H1_MODEL (the auditor's model-key
#      for b/<K>, the server's --h1-model; or H1_WEIGHTS_KEY, a key file)  H1_DRAW_K (else period.json's)
#      FLOCK_PROVE, FLOCK_SELFTEST, FLOCK_VERIFY (prebuilt; else job.sh builds them under HZ_CACHE)  FC_FIREWALL_LEAN (the
#      ZK server's flock-firewall, prebuilt; else built beside FLOCK_VERIFY)  H1_DRAWN_MEM_GB (the job's budget, default
#      90% of the host's memory)  H1_ZK_SERVER_GB (a session's live verifier, `serve --zk`, its limit, default 125)
#      H1_ZK_LEAN_GB (what the prover's share leaves the ZK server's Lean processes, default 145)  H1_FIREWALL (1; 0 the
#      unfirewalled --zk path, for a cost comparison only: no ZK server serves it)
#      PY (served_verdicts.sh's venv python, default .venv's)  H1_DRAWN_ONLY=1: write and check the statements, no verdict
#      H1_DRAWN_CHECK=0: no drawn.py check, for a caller whose credit verifies the statements itself
#      H1_DRAWN_WORK (a private directory on the node: the scratch directory goes in it and is kept, so the proof files
#      statements.json names outlive drawn.sh for that caller; it holds the staged witness)
#      H1_DRAWN_STAGED (a work dir `drawn.py stage` already wrote, its plan.json: proved and checked as it is, no staging;
#      drawn_cost.sh's DC_REAL=0 fixture, whose stand-in tile the CLI's word reference can't restage)
# Memory: staging is capped at 90% of H1_DRAWN_MEM_GB. A session's live verifier and prover then run side by side, each
# under an address-space limit (`ulimit -S -v`): the verifier gets H1_ZK_SERVER_GB and the prover the rest of
# H1_DRAWN_MEM_GB less H1_ZK_LEAN_GB. The ZK server's Lean processes are the prover's children, and under an
# address-space limit their runtime can't create its threads, so they start through a wrapper that lifts the soft limit
# they inherit: they are bounded by the H1_ZK_LEAN_GB the prover's limit leaves them and the job's cgroup, not by a
# limit of their own. An unfirewalled K 2048 lane session (2^27 slots: 24.4M ANDs, k_log 26, m 29; a combine is 4223
# ANDs) peaked at 65.4 GiB resident in the verifier and 61.3 GiB in the prover;
# under the ZK server, 64.2 GiB in the prover and 77.8 GiB summed over the ZK server's two Lean
# processes. A J10-R tile call of one unit (k_log 27, m 30) peaked at 92.2 GiB in the verifier on
# either path (whose 88 GiB limit it outgrew), 98.5 GiB in the prover and 108.1 GiB in the Lean
# processes. The defaults are those k_log 27 peaks plus ~35% for the verifier and the Lean
# processes, since an address space outgrows resident memory by the allocator's arenas and the threads' stacks.
# drawn.sh refuses before staging when the prover's limit is below its 99 GB resident peak, so a k_log 27 window wants
# H1_DRAWN_MEM_GB >= 125 + 145 + 99 = 369 (H1_DRAWN_MEM_GB=400: the prover's limit is 130, its peak plus 32%).
# Out: PIECE_DIR/drawn/ (statements.json and job.log, and sessions/<statement>/: the live verifier's log, the prover's
# LIVE verdict lines, both sides' wall seconds, peak RSS and CPU seconds, and the job cgroup's CPU seconds over the
# session (cgroup.json, the ZK server's Lean processes among them), kept whether the session passed or failed;
# serve is the verifier, so none of it is witness), then served_verdicts.py pouw's files in PIECE_DIR. The staging (stage/: the plan
# and each statement's witness files and session) and job.sh's other logs, the prover's stderr among them, go to a
# `mktemp -d` scratch directory removed on exit (kept with H1_DRAWN_WORK): PIECE_DIR is under the run's $OUT, and a staged
# witness never goes there.
set -uo pipefail
SD=${1:?SERVE_DIR} WIN=${2:?WINDOW} P=${3:?PIECE_DIR}
cd "$(dirname "$0")/../../.." || exit 1
VPY=${PY:-$(pwd)/.venv/bin/python}
mkdir -p "$P/drawn"
P=$(cd "$P" && pwd)
TOP=${H1_DRAWN_WORK:-${TMPDIR:-/tmp}}
(umask 077; mkdir -p "$TOP") && SCR=$(mktemp -d "$TOP/h1-drawn.XXXXXX") || exit 1
export RESEARCH_RUN_DIR=$SCR HZ_CONFIGS=drawn HZ_RUNS=1 HZ_WARM=0 HZ_PYTHON=$VPY
export HZ_MEM_GB=${H1_DRAWN_MEM_GB:-$(awk '/MemTotal/ {print int($2 * 0.9 / 1048576)}' /proc/meminfo)}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
trap 'cp "$OUT/job.log" "$P/drawn/" 2>/dev/null; rm -rf "$W"; [ -n "${H1_DRAWN_WORK:-}" ] || rm -rf "$SCR"' EXIT
FW=${H1_FIREWALL:-1}
SERVE_GB=${H1_ZK_SERVER_GB:-125} LEAN_GB=0
[ "$FW" = 1 ] && LEAN_GB=${H1_ZK_LEAN_GB:-145}
PROVE_GB=$(( MEM_GB - SERVE_GB - LEAN_GB ))
[ "$PROVE_GB" -ge 99 ] || { log "drawn: the prover's cap, $MEM_GB - $SERVE_GB (H1_ZK_SERVER_GB) - $LEAN_GB (H1_ZK_LEAN_GB) =" \
  "$PROVE_GB GB, is below a k_log 27 session's 99 GB peak: raise H1_DRAWN_MEM_GB (a larger host, or a CPU node) or," \
  "for a smaller K, lower H1_ZK_SERVER_GB"; exit 1; }
vlimit() { echo "ulimit -S -v $(( $1 * 1024 * 1024 ));"; }
setup
if [ "$FW" = 1 ]; then
  FWL=${FC_FIREWALL_LEAN:-${LEAN%/*}/flock-firewall}
  if [ ! -x "$FWL" ]; then
    command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
    (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FWL" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
      > "$OUT/build-firewall.log" 2>&1
  fi
  [ -x "$FWL" ] || { tail -20 "$OUT/build-firewall.log" 2>/dev/null; log "drawn: no ZK server frame at $FWL (lake build" \
    "flock-firewall beside FLOCK_VERIFY, or FC_FIREWALL_LEAN)"; exit 1; }
  sha256sum "$FWL" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
  printf '#!/usr/bin/env bash\nulimit -S -v "$(ulimit -H -v)" && exec %q "$@"\n' "$FWL" > "$W/flock-firewall" \
    && chmod 700 "$W/flock-firewall" || exit 1
  export FC_FIREWALL_LEAN=$W/flock-firewall
fi
log "drawn: ZK server $([ "$FW" = 1 ] && echo "on ($FWL)" || echo "OFF (H1_FIREWALL=0)"), caps: verifier $SERVE_GB GB," \
  "prover $PROVE_GB GB, its ZK server's Lean $LEAN_GB GB left"
export PYTHONPATH=$SRC:$SRC/adapters/vllm:$SRC/verity/protocols/flock/prover/python${PYTHONPATH:+:$PYTHONPATH}
WK=${H1_DRAWN_STAGED:-$SCR/stage} ST=$P/drawn/statements.json
args=(--circuits "${H1_CIRCUITS:?H1_CIRCUITS: pouw_h1_circuits --out, with the drawn K file}")
[ -n "${H1_DRAW_K:-}" ] && args+=(--k "$H1_DRAW_K")
st=("${args[@]}" --work "$WK")
[ -n "${H1_CHECKPOINT:-}" ] && st+=(--checkpoint "$H1_CHECKPOINT")
[ -n "${H1_WEIGHTS:-}" ] && st+=(--weights "$H1_WEIGHTS")
if [ -n "${H1_WEIGHTS_KEY:-}" ]; then st+=(--weights-key "$H1_WEIGHTS_KEY")
elif [ -n "${H1_GATE:-}" ] && [ -n "${H1_MODEL:-}" ]; then st+=(--gate "$H1_GATE" --model "$H1_MODEL"); fi
if [ -n "${H1_DRAWN_STAGED:-}" ]; then
  [ -s "$WK/plan.json" ] || { log "drawn: H1_DRAWN_STAGED=$WK holds no plan.json"; exit 1; }
  log "drawn: window $WIN of $SD, staged in $WK (H1_DRAWN_STAGED)"
else
  log "drawn: window $WIN of $SD, staging"
  $TIME "$OUT/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $J/../h1_prove/drawn.py stage $(printf '%q ' "$SD" "$WIN" "${st[@]}")" \
    2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: staging REFUSED"; exit 1; }
fi
keep() {   # keep O NAME: the session's verifier-side log, its verdict lines and timings, none of them witness
  local K=$P/drawn/sessions/$2 f
  mkdir -p "$K"
  for f in serve.log serve.time prove.time sessions.du cgroup.json; do [ -e "$1/$f" ] && cp "$1/$f" "$K/"; done
  grep '^LIVE' "$1/prove.out" > "$K/verdicts.txt" 2>/dev/null
}
# the job's cgroup's CPU microseconds: a session's ZK server processes are its prover's children that it never waits for,
# so timed.py's cpu_s leaves them out, and cgroup.json's cgroup_cpu_s (verifier, prover and ZK server) counts them
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
cg_cpu() { awk '$1 == "usage_usec" {print $2}' "$CG/cpu.stat" 2>/dev/null; }
for d in $("$PY" -c 'import json, sys; p = json.load(open(sys.argv[1])); print(" ".join(f"{c["dir"]}:{c["instances"]}" for c in [*p["calls"], *p.get("leaves", [])]))' "$WK/plan.json"); do
  D=${d%:*} N=${d##*:} O=$OUT/$(basename "${d%:*}")
  mkdir -p "$O" && rm -rf "$D/sessions"
  log "drawn: $(basename "$D"): --zk session over its $N drawn units"
  if [ "$FW" = 1 ]; then
    [ -s "$D/program.json" ] && [ -s "$D/partition.json" ] \
      || { log "drawn: $(basename "$D"): no program.json and partition.json to set the ZK server's Hello up on"; exit 1; }
    export FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_STATEMENT=verity/flock-circuit
    export FC_FIREWALL_HELLO_ARGS="--partition $D/partition.json --program $D/program.json"
  fi
  c0=$(cg_cpu)
  session "$D" "$O"; rc=$?
  c1=$(cg_cpu)
  [ -n "$c0" ] && [ -n "$c1" ] && echo "{\"cgroup_cpu_s\": $(( (c1 - c0) / 1000000 ))}" > "$O/cgroup.json"
  keep "$O" "$(basename "$D")"
  log "drawn: $(basename "$D"): serve {$(grep -o '"wall_s": [^}]*' "$O/serve.time" 2>/dev/null)}" \
      "prove {$(grep -o '"wall_s": [^}]*' "$O/prove.time" 2>/dev/null)} $(cat "$O/cgroup.json" 2>/dev/null)"
  [ "$rc" = 0 ] && grep -q '"accepted":true' "$O/prove.out" \
    || { log "drawn: $(basename "$D"): --zk session FAILED: $(tail -1 "$O/prove.out") $(tail -1 "$O/serve.log")"; exit 1; }
done
unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
"$PY" "$J/../h1_prove/drawn.py" statements --work "$WK" --out "$ST" 2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
if [ "${H1_DRAWN_CHECK:-1}" != 0 ]; then
  "$PY" "$J/../h1_prove/drawn.py" check "$SD" "$WIN" "${args[@]}" --statements "$ST" --lean "$LEAN" 2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: credit refuses the statements"; exit 1; }
fi
[ "${H1_DRAWN_ONLY:-0}" = 1 ] && exit 0
"$VPY" benchmarks/one_stage/served_verdicts.py pouw "$SD" "$WIN" "$P" --statements "$ST" \
  ${H1_LIVE_EVIDENCE:+--live-evidence "$H1_LIVE_EVIDENCE"}
