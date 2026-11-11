#!/usr/bin/env bash
# PoUW's piece of a served H1 window with its sampled proof blocks proved (served_verdicts.sh's POUW_CMD): drawn.py (the CLI of
# verity_vllm.protocols.pouw.h1.drawn, reading --checkpoint with the live challenger's model registry) stages each
# sampled call's J10-R units on the window's own registered values, and each sampled proof block of a lanes-form K as its lanes and
# combine; each staged statement is proved by one loopback Flock --zk session sampling all its instances
# (hidden_zk/job.sh's `session`, at the slot its circuit file's META names), drawn.py writes the statements file and
# checks it as credit does, and then `served_verdicts.py pouw` judges the window with it as --statements (the Lean verdict
# of record refuses the lanes form until its mirror: H1_DRAWN_ONLY=1 stops at drawn.py's check).
#   POUW_CMD='bash benchmarks/pouw/h1_prove/drawn.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/pouw/h1_prove/drawn.sh SERVE_DIR WINDOW PIECE_DIR
# env: H1_CIRCUITS (pouw_h1_circuits' --out, with the sampled K's k<K> file, --strata K, or its lane<K> and combine<K>
#      files, --lanes K)  H1_CHECKPOINT (the served model's
#      safetensors dir, b/<K>'s rows) or H1_WEIGHTS (an npz of w<i>, s<i>)  H1_GATE and H1_MODEL (the auditor's model-key
#      for b/<K>, the server's --h1-model; or H1_WEIGHTS_KEY, a key file)  H1_DRAW_K (else period.json's)
#      FLOCK_PROVE, FLOCK_SELFTEST, FLOCK_VERIFY (prebuilt; else job.sh builds them under HZ_CACHE)  H1_DRAWN_MEM_GB (the
#      job's budget, default 90% of the host's memory)  H1_ZK_SERVER_GB (a session's ZK server cap, default 88)
#      PY (served_verdicts.sh's venv python, default .venv's)  H1_DRAWN_ONLY=1: write and check the statements, no verdict
#      H1_DRAWN_CHECK=0: no drawn.py check, for a caller whose credit verifies the statements itself
#      H1_DRAWN_WORK (a private directory on the node: the scratch directory goes in it and is kept, so the proof files
#      statements.json names outlive drawn.sh for that caller; it holds the staged witness)
# Memory: staging is capped at 90% of H1_DRAWN_MEM_GB. A session's ZK server and prover then run side by side, each under
# an address-space cap (job.sh's vlimit): the server gets H1_ZK_SERVER_GB and the prover the rest of H1_DRAWN_MEM_GB. The
# default is sized to a K 2048 lane session (2^27 slots: 24.4M ANDs, k_log 26, m 29; a combine is 4223 ANDs), which peaked
# at 65.4 GiB resident in the server and 61.3 GiB in the prover:
# each side's cap is its peak plus ~35%, since an address space outgrows resident memory by the allocator's arenas and
# the threads' stacks. So a K 2048 window wants H1_DRAWN_MEM_GB >= 88 + 83 = 171 (a ~195 GB H200 host's 90% is ~174),
# and drawn.sh refuses before staging only when the prover's share is below its 62 GB resident peak. A larger K's lane
# is measured again before it runs.
# Out: PIECE_DIR/drawn/ (statements.json and job.log, and sessions/<statement>/: the ZK server's log, the prover's LIVE
# verdict lines and both sides' wall seconds and peak RSS, kept whether the session passed or failed; the server is the
# verifier, so none of it is witness), then served_verdicts.py pouw's files in PIECE_DIR. The staging (stage/: the plan
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
SERVE_GB=${H1_ZK_SERVER_GB:-88}
PROVE_GB=$(( MEM_GB - SERVE_GB ))
[ "$PROVE_GB" -ge 62 ] || { log "drawn: the prover's share, $MEM_GB - $SERVE_GB (H1_ZK_SERVER_GB) = $PROVE_GB GB, is below a K 2048" \
  "lane session's 62 GB peak: raise H1_DRAWN_MEM_GB (a larger host, or a CPU node) or, for a smaller K, lower H1_ZK_SERVER_GB"; exit 1; }
setup
export PYTHONPATH=$SRC:$SRC/adapters/vllm:$SRC/verity/protocols/flock/prover/python${PYTHONPATH:+:$PYTHONPATH}
WK=$SCR/stage ST=$P/drawn/statements.json
args=(--circuits "${H1_CIRCUITS:?H1_CIRCUITS: pouw_h1_circuits --out, with the drawn K file}")
[ -n "${H1_DRAW_K:-}" ] && args+=(--k "$H1_DRAW_K")
st=("${args[@]}" --work "$WK")
[ -n "${H1_CHECKPOINT:-}" ] && st+=(--checkpoint "$H1_CHECKPOINT")
[ -n "${H1_WEIGHTS:-}" ] && st+=(--weights "$H1_WEIGHTS")
if [ -n "${H1_WEIGHTS_KEY:-}" ]; then st+=(--weights-key "$H1_WEIGHTS_KEY")
elif [ -n "${H1_GATE:-}" ] && [ -n "${H1_MODEL:-}" ]; then st+=(--gate "$H1_GATE" --model "$H1_MODEL"); fi
log "drawn: window $WIN of $SD, staging"
$TIME "$OUT/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $J/../h1_prove/drawn.py stage $(printf '%q ' "$SD" "$WIN" "${st[@]}")" \
  2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: staging REFUSED"; exit 1; }
keep() {   # keep O NAME: the session's verifier-side log, its verdict lines and timings, none of them witness
  local K=$P/drawn/sessions/$2 f
  mkdir -p "$K"
  for f in serve.log serve.time prove.time sessions.du; do [ -e "$1/$f" ] && cp "$1/$f" "$K/"; done
  grep '^LIVE' "$1/prove.out" > "$K/verdicts.txt" 2>/dev/null
}
for d in $("$PY" -c 'import json, sys; p = json.load(open(sys.argv[1])); print(" ".join(f"{c["dir"]}:{c["instances"]}" for c in [*p["calls"], *p.get("leaves", [])]))' "$WK/plan.json"); do
  D=${d%:*} N=${d##*:} O=$OUT/$(basename "${d%:*}")
  mkdir -p "$O" && rm -rf "$D/sessions"
  log "drawn: $(basename "$D"): --zk session over its $N drawn units"
  session "$D" "$O"; rc=$?
  keep "$O" "$(basename "$D")"
  log "drawn: $(basename "$D"): serve {$(grep -o '"wall_s": [^}]*' "$O/serve.time" 2>/dev/null)}" \
      "prove {$(grep -o '"wall_s": [^}]*' "$O/prove.time" 2>/dev/null)}"
  [ "$rc" = 0 ] && grep -q '"accepted":true' "$O/prove.out" \
    || { log "drawn: $(basename "$D"): --zk session FAILED: $(tail -1 "$O/prove.out") $(tail -1 "$O/serve.log")"; exit 1; }
done
"$PY" "$J/../h1_prove/drawn.py" statements --work "$WK" --out "$ST" 2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
if [ "${H1_DRAWN_CHECK:-1}" != 0 ]; then
  "$PY" "$J/../h1_prove/drawn.py" check "$SD" "$WIN" "${args[@]}" --statements "$ST" --lean "$LEAN" 2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: credit refuses the statements"; exit 1; }
fi
[ "${H1_DRAWN_ONLY:-0}" = 1 ] && exit 0
"$VPY" benchmarks/one_stage/served_verdicts.py pouw "$SD" "$WIN" "$P" --statements "$ST" \
  ${H1_LIVE_EVIDENCE:+--live-evidence "$H1_LIVE_EVIDENCE"}
