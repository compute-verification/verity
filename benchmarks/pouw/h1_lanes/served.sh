#!/usr/bin/env bash
# One unit a served H1 window's own draw selected, proved as its 32 lanes, each a Flock --zk statement under the Lean
# firewall, and verified by the Lean verifier against the window's registered roots (served.py's docstring).
#   research run --on NODE --project verity --campaign C --source . --cwd source --mem-gb G --timeout 600m \
#     --declared-output 'out/*' --env H1L_MEM_GB=G --env H1S_CALLS=... [...] -- bash benchmarks/pouw/h1_lanes/served.sh
#   1. hidden_zk/job.sh's builds and the firewall's Lean frame (flock-firewall)
#   2. the drawn units (h1_prove/served.py: core's draw on the window's signed coin) and served.py extract: the drawn unit's
#      openings from the prover's saved window, into the job's scratch (private), never out/
#   3. served.py prepare: the lane statements and the negatives staged (WORK/plan.tsv)
#   4. one loopback --zk session per statement, H1L_PAR at a time, each lane's prover under the firewall (FC_FIREWALL=1:
#      the Lean session loop, its level-0 draw and Lean finish) unless H1S_FIREWALL=0; the negatives first, without it
#      (H1S_UNFIREWALLED: what they check is the verifier's refusal)
#   5. the plan's Lean checks, each with the lane Program, its partition, its public file and registered reads: with
#      H1L_VERIFY_EACH=1 (the default) each statement's right after its session, holding its session slot, so a refusal shows
#      after the first lane (the first firewalled lane runs before the negatives); with 0, all after the last session
#   6. served.py summarize: out/summary.json
# env: H1L_MEM_GB (the job's --mem-gb)
#      the window: H1S_CALLS (pouw-h1-calls.json)  H1S_SESSION (the gate's fetched session out/)  H1S_LEDGER (ledger heads,
#      space-separated)  H1S_PIN (the challenger key's public_sha512)  H1S_PINS (the host's pins log)  H1S_MAIN (the
#      verifier's own `git rev-list --first-parent origin/main`)  H1S_DESCRIPTOR (window-program.json.gz)  H1S_PROVER (the
#      window's prover directory, pouw_h1_prover's: key, calls/, state/)  H1S_PY (the H1 venv's python, which unpickles it)
#      or H1S_SMOKE=K: a stand-in window at K (served.py smoke), the harness's own check
#      or H1S_OPENED (an extract's private openings: one node's share of the lanes, H1L_LANES, the negatives skipped
#      with H1S_SKIP=tampered-zy,other-call-a); H1S_EXTRACT_ONLY=FILE: keep the extract's openings there (0600) and stop
#      or H1S_OPENED_AT=FILE: a node path (a private directory) the openings are copied to while prepare lowers the lane
#      circuit, waited for up to H1S_WAIT seconds (14400), then moved into the scratch; with H1S_DESCRIPTOR, prepare's
#      lane statement is the one derived from that window Program (served.py prepare --descriptor)
#      H1L_SESSION_GB (~350 at K=4096), H1L_LEAN_GB (115), H1L_STAGGER (60), H1L_RAMP (900): admit a firewalled session only with that much
#      free in the run's cgroup, its slice and the node, beyond what the sessions still ramping up will take
#   a session counts as proved only if its live verifier accepted it (prove.out's LIVE line); otherwise out/TAG/FAILED says
#   why (the live verifier's and the prover's exits, a kill's signal), it gets no Lean check, and the job exits 1
#      H1L_K (4096)  H1L_PAR (8: sessions at a time)  H1L_LPAR (8: Lean checks at a time)  H1L_JOBS (8: prepare's stagers)
#      H1L_VERIFY_EACH (1)
#      H1L_LANES (all, or 0,5,...)  H1S_FIREWALL (1)  H1S_UNFIREWALLED ("tampered-zy other-call-a")  H1S_KEEP (tags whose sessions out/ keeps: "lane0 tampered-zy", or all;
#      each as soon as its session ends)  H1S_KEEP_CIRCUIT (a node directory each lowered circuit is kept in after prepare,
#      as SHA512.txt, a hard link where it can be: the scratch, the circuit's other name, goes at exit)
#      H1S_FIREWALL_TGZ (`git archive --format=tar.gz REV verity/core/service/flock`: the firewall frame built from REV's
#      Lean, which must speak this tree's prover protocol; the verifier stays this tree's)
#      H1S_FIREWALL_TGZ_DIR (a node directory of such tarballs, read again before each firewalled session: the session runs
#      the frame of the last *.tgz by name, so a later frame dropped in mid-share serves the lanes after it. Move a tarball
#      in whole (mv), never write it in place; one that doesn't build is logged and the previous frame kept. Each session's
#      frame is in out/TAG/firewall.sha256 and every frame's binary in out/binaries.sha256)
set -uo pipefail
export HZ_CONFIGS=h1-served HZ_MEM_GB=${H1L_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
H=$SRC/benchmarks/pouw/h1_lanes
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk:$SRC/benchmarks/pouw/h1_prove
PAR=${H1L_PAR:-8} LPAR=${H1L_LPAR:-8} K=${H1L_K:-4096} FW=${H1S_FIREWALL:-1}
WK=$W/work OP=$W/opened.json
mkdir -p "$OUT/lean" "$WK"
setup
fw_frame() {   # fw_frame TGZ: the firewall frame built from TGZ, cached by the tarball's hash; its binary's path on stdout
  local fd
  fd=$HZ_CACHE/firewall-$(sha256sum "$1" | cut -c1-16)
  (exec 8>"$fd.lock"; flock 8; [ -x "$fd/.lake/build/bin/flock-firewall" ] || { rm -rf "$fd" && mkdir -p "$fd" \
    && tar -xzf "$1" -C "$fd" --strip-components=4 && cd "$fd" && lake build flock-firewall; }) \
    < /dev/null >> "$OUT/build-firewall.log" 2>&1
  echo "$fd/.lake/build/bin/flock-firewall"
}
fw_last() { ls "${H1S_FIREWALL_TGZ_DIR:?}"/*.tgz 2>/dev/null | LC_ALL=C sort | tail -1; }
FWDIR=${H1S_FIREWALL_TGZ_DIR:-}
if [ "$FW" = 1 ] && [ -n "$FWDIR" ] && [ -z "${H1S_FIREWALL_TGZ:-}" ]; then H1S_FIREWALL_TGZ=$(fw_last); fi
FWTGZ=
if [ "$FW" = 1 ] && [ -n "${H1S_FIREWALL_TGZ:-}" ]; then
  log "firewall frame from $H1S_FIREWALL_TGZ ($(sha256sum "$H1S_FIREWALL_TGZ" | cut -c1-16)), not this tree's"
  FWTGZ=$(sha256sum "$H1S_FIREWALL_TGZ" | cut -c1-64)
  export FC_FIREWALL_LEAN
  FC_FIREWALL_LEAN=$(fw_frame "$H1S_FIREWALL_TGZ")
elif [ "$FW" = 1 ]; then
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
fi
if [ "$FW" = 1 ]; then
  [ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "no firewall frame at $FC_FIREWALL_LEAN"; exit 1; }
  sha256sum "$FC_FIREWALL_LEAN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi

# ---- the drawn unit's openings ---------------------------------------------------------------------------------------------
if [ -n "${H1S_SMOKE:-}" ]; then
  K=$H1S_SMOKE
  log "h1-served: SMOKE, a stand-in window at K=$K (the harness's own check, no served window)"
  "$PY" "$H/served.py" smoke --k "$K" --out "$OP" > "$OUT/extract.out" 2> "$OUT/extract.err" || { log "smoke FAILED: $(tail -3 "$OUT/extract.err")"; exit 1; }
elif [ -n "${H1S_OPENED_AT:-}" ]; then
  # the openings arrive at this node path (a private directory) while prepare lowers the lane circuit
  OP=$H1S_OPENED_AT
  log "h1-served: the openings are awaited at $OP (up to ${H1S_WAIT:-14400} s)"
elif [ -n "${H1S_OPENED:-}" ]; then
  # an extract's private openings, moved into the scratch: a copy left in the run dir would be uploaded with its record
  install -m 600 "$H1S_OPENED" "$OP" || exit 1
  case $H1S_OPENED in "$RESEARCH_RUN_DIR"/*) rm -f "$H1S_OPENED" \
    "${RESEARCH_RUN_DIR%/runs/*}/requests/${RESEARCH_RUN_DIR##*/}/inputs/${H1S_OPENED##*/}" ;; esac
  log "h1-served: the openings of $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['window'][:16], 'unit', d['unit']['unit'], 'K', d['k'])" "$OP")"
else
  HPP=$SRC:$SRC/integrations/vllm:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/h1_prove
  ledgers=()
  for l in ${H1S_LEDGER:?}; do ledgers+=(--ledger "$l"); done
  PYTHONPATH=$HPP "${H1S_PY:?}" "$SRC/benchmarks/pouw/h1_prove/served.py" --calls "${H1S_CALLS:?}" --session "${H1S_SESSION:?}" \
    "${ledgers[@]}" --pin "${H1S_PIN:?}" --pins "${H1S_PINS:?}" --main "${H1S_MAIN:?}" --descriptor "${H1S_DESCRIPTOR:?}" \
    --out "$OUT/drawn.json" > "$OUT/drawn.out" 2> "$OUT/drawn.err" \
    || { log "draw FAILED: $(tail -3 "$OUT/drawn.err") $(tail -3 "$OUT/drawn.out")"; exit 1; }
  log "drawn: $(head -c 1500 "$OUT/drawn.out")"
  PYTHONPATH=$HPP $TIME "$OUT/extract.time" "$H1S_PY" "$H/served.py" extract --calls "$H1S_CALLS" --drawn "$OUT/drawn.json" \
    --descriptor "$H1S_DESCRIPTOR" --prover "${H1S_PROVER:?}" --k "$K" --out "$OP" --public "$OUT/extract.json" \
    > "$OUT/extract.out" 2> "$OUT/extract.err" || { log "extract FAILED: $(tail -3 "$OUT/extract.err")"; exit 1; }
  log "extract: $(tail -1 "$OUT/extract.out" | cut -c1-600) $(cat "$OUT/extract.time")"
  if [ -n "${H1S_EXTRACT_ONLY:-}" ]; then
    install -m 600 "$OP" "$H1S_EXTRACT_ONLY" && log "openings kept at $H1S_EXTRACT_ONLY (private), sha256 $(sha256sum < "$OP" | cut -c1-64)"
    exit $?
  fi
fi

# ---- the statements ------------------------------------------------------------------------------------------------------
log "h1-served: K=$K tile=${H1_TILE:-v1} firewall $FW, sessions $PAR at a time, Lean $LPAR at a time"
DESC=()
[ -z "${H1S_SMOKE:-}" ] && [ -n "${H1S_DESCRIPTOR:-}" ] && DESC=(--descriptor "$H1S_DESCRIPTOR")
$TIME "$OUT/prepare.time" "$PY" "$H/served.py" prepare --opened "$OP" --work "$WK" --jobs "${H1L_JOBS:-8}" \
  --lanes "${H1L_LANES:-all}" --skip "${H1S_SKIP:-}" --name "h1-served-${RESEARCH_RUN_ID:-local}" --k "$K" \
  ${DESC[@]+"${DESC[@]}"} \
  ${H1S_OPENED_AT:+--wait "${H1S_WAIT:-14400}"} > "$OUT/prepare.out" 2> "$OUT/prepare.err"
rc=$?
if [ -n "${H1S_OPENED_AT:-}" ] && [ $rc = 0 ]; then mv "$H1S_OPENED_AT" "$W/opened.json"; OP=$W/opened.json; fi
tail -4 "$OUT/prepare.out" | cut -c1-2000 | tee -a "$OUT/job.log"
cp "$WK/plan.json" "$WK/prepare.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -5 "$OUT/prepare.err")"; exit 1; }
log "prepare: $(cat "$OUT/prepare.time")"
mkdir -p "$OUT/verifier" && cp "$WK"/verifier/* "$OUT/verifier/"
if [ -n "${H1S_KEEP_CIRCUIT:-}" ]; then
  mkdir -p "$H1S_KEEP_CIRCUIT"
  "$PY" -c "import json, sys; [print(c['circuit_sha512'], c['dir']) for c in json.load(open(sys.argv[1]))['checks'].values()]" \
    "$WK/plan.json" | sort -u -k1,1 | while read -r sha d; do
    k=$H1S_KEEP_CIRCUIT/$sha.txt c=$(readlink -f "$d/circuit.txt")
    if [ ! -e "$k" ] && ! ln "$c" "$k" 2>/dev/null; then cp "$c" "$k.tmp.$$" && mv -n "$k.tmp.$$" "$k"; rm -f "$k.tmp.$$"; fi
    log "circuit $sha kept at $k ($(stat -c %s "$k" 2>/dev/null) B)"
  done
fi
keep() {   # keep TAG: TAG's sessions and pin into out/, if H1S_KEEP names it (or is all)
  case " ${H1S_KEEP:-lane0 tampered-zy} " in *" $1 "*|" all ") ;; *) return 0 ;; esac
  [ -d "$WK/st/$1/sessions" ] && [ ! -e "$OUT/$1/sessions" ] && cp -r "$WK/st/$1/sessions" "$OUT/$1/" \
    && cp "$WK/st/$1/pin.txt" "$OUT/$1/"
}

# ---- proofs: one --zk session per statement ------------------------------------------------------------------------------
VLIMIT=$(declare -f vlimit)
if [ "$FW" = 1 ]; then
  export FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_STATEMENT=verity/flock-circuit
  export FC_FIREWALL_HELLO_ARGS="--partition $WK/verifier/lane.partition.json --program $WK/verifier/lane.program.json"
  # the firewall's Lean processes are the prover's children, and under an address-space cap (ulimit -v) their runtime
  # can't create its threads: a firewalled session is bounded by the run's cgroup (--mem-gb) alone
  vlimit() { :; }
fi
N=1
export RAYON_NUM_THREADS=$(( TH / PAR > 0 ? TH / PAR : 1 ))
UNFW=" ${H1S_UNFIREWALLED:-tampered-zy other-call-a} "
# H1L_SESSION_GB: a firewalled session starts only while that much is free, H1L_STAGGER seconds after the last, so
# H1L_PAR is a cap and memory decides. A firewalled session at K=4096 holds the prover (137 GiB), its live verifier
# (141 GiB) and the firewall's Lean process (~40 GB), ~350 GB together, and reaches that only after ~H1L_RAMP seconds
# (900): each session younger than that is charged its whole H1L_SESSION_GB against what's free, and so is each Lean
# check younger than that (H1L_LEAN_GB, 115: a lane's check starts as its session ends, just when the next session is
# admitted). Free is the least of the run's cgroup (capped at
# H1L_MEM_GB), every cgroup above it with a memory.max (the user's slice) and the node's MemAvailable: the kernel's OOM
# killer answers to all three, and kills the largest process, the live verifier, first
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
free_gb() {
  local m c f g=$CG a
  m=$(cat "$CG/memory.max" 2>/dev/null) c=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
  case $m in ''|max) m=$(( MEM_GB << 30 )) ;; esac
  [ "$m" -gt $(( MEM_GB << 30 )) ] && m=$(( MEM_GB << 30 ))
  f=$(( m - c ))
  while [ "${g%/*}" != "$g" ] && g=${g%/*} && [ "$g" != /sys/fs/cgroup ]; do
    m=$(cat "$g/memory.max" 2>/dev/null) c=$(cat "$g/memory.current" 2>/dev/null || echo 0)
    [[ $m =~ ^[0-9]+$ ]] && [ $(( m - c )) -lt "$f" ] && f=$(( m - c ))
  done
  a=$(awk '$1 == "MemAvailable:" {print $2 * 1024}' /proc/meminfo)
  [ -n "$a" ] && [ "$a" -lt "$f" ] && f=$a
  echo $(( f >> 30 ))
}
STARTS=()
young() {   # sessions started less than H1L_RAMP seconds ago
  local now n=0 s
  printf -v now '%(%s)T' -1
  for s in ${STARTS[@]+"${STARTS[@]}"}; do [ $(( now - s )) -lt "${H1L_RAMP:-900}" ] && n=$(( n + 1 )); done
  echo "$n"
}
young_lean() {   # Lean checks started less than H1L_RAMP seconds ago (out/lean/TAG.start, written as the session ends)
  local now n=0 f s
  printf -v now '%(%s)T' -1
  for f in "$OUT"/lean/*.start; do
    [ -e "$f" ] && read -r s < "$f" && [ $(( now - s )) -lt "${H1L_RAMP:-900}" ] && n=$(( n + 1 ))
  done
  echo "$n"
}
oom_kills() { awk '$1 == "oom_kill" {print $2}' /proc/vmstat; }
live_failed() {   # live_failed O: why O's session wasn't accepted live, on stdout, or exit 1 if it was. The prover exits 0
  # whatever the live verifier answered, and a killed live verifier or firewall leaves no session for Lean to check
  "$PY" - "$1" <<'EOF'
import json, signal, sys
o = sys.argv[1]
live = None
try:
    for ln in open(o + "/prove.out", errors="replace"):
        if ln.startswith("LIVE\t"):
            live = json.loads(ln.split("\t", 1)[1])
except (OSError, ValueError):
    pass
if live is not None and live.get("accepted") is True:
    sys.exit(1)
def ended(k, name):
    try:
        e = json.load(open(f"{o}/{k}.time")).get("exit")
    except (OSError, ValueError):
        return f"{name}: no record"
    if e == -15 and k == "serve":
        return f"{name} stopped by the harness after the prover"
    if isinstance(e, int) and e < 0:
        return f"{name} killed by {signal.Signals(-e).name}" + (" (the kernel's OOM killer, or a cancel)" if e == -9 else "")
    return f"{name} exit {e}"
why = "no LIVE line" if live is None else f"LIVE accepted={live.get('accepted')}: {str(live.get('why'))[:300]}"
print(f"{why}; {ended('serve', 'the live verifier')}; {ended('prove', 'the prover')}")
EOF
}
EACH=${H1L_VERIFY_EACH:-1}
eval "${VLIMIT/vlimit/lean_vlimit}"
lean_check() {   # lean_check TAG CAP: the plan's Lean check of TAG, in the foreground
  local kind tag D pub reg prog part _expect s
  IFS=$'\t' read -r kind tag D pub reg prog part _expect < <(awk -F'\t' -v t="$1" '$1 == "C" && $2 == t' "$WK/plan.tsv")
  [ -n "${tag:-}" ] || return 0
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; return 0; }
  env -u FC_FIREWALL -u FC_HOST_WITNESS -u FC_FIREWALL_STATEMENT -u FC_FIREWALL_HELLO_ARGS \
    $TIME "$OUT/lean/$tag.time" bash -c "$(lean_vlimit "$2") exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement verity/flock-circuit --public $pub --partition $part --program $prog --registered $reg --session $s" \
    > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err"
}
order() {   # the plan's statements in session order
  local u="$UNFW" first=""
  [ "$EACH" = 1 ] && first=$(awk -F'\t' -v u="$u" '$1 == "S" && !index(u, " " $2 " ") {print $2; exit}' "$WK/plan.tsv")
  awk -F'\t' -v f="$first" '$1 == "S" && $2 == f' "$WK/plan.tsv"
  awk -F'\t' -v u="$u" -v f="$first" '$1 == "S" && $2 != f && index(u, " " $2 " ")' "$WK/plan.tsv"
  awk -F'\t' -v u="$u" -v f="$first" '$1 == "S" && $2 != f && !index(u, " " $2 " ")' "$WK/plan.tsv"
}
while IFS=$'\t' read -r kind tag D; do
  [ "$kind" = S ] || continue
  mkdir -p "$OUT/$tag"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  if [[ $UNFW == *" $tag "* ]]; then fw=0; else fw=$FW; fi
  if [ "$fw" = 1 ] && [ -n "${H1L_SESSION_GB:-}" ]; then
    while [ "$(jobs -rp | wc -l)" -gt 0 ] \
      && [ "$(free_gb)" -lt $(( H1L_SESSION_GB * (1 + $(young)) + ${H1L_LEAN_GB:-115} * $(young_lean) )) ]; do sleep 15; done
    log "$tag: $(free_gb) GB free (the least of the run's cgroup, its slice and the node), $(jobs -rp | wc -l) sessions running, $(young) of them and $(young_lean) Lean checks ramping"
    printf -v now '%(%s)T' -1
    STARTS+=("$now")
  fi
  if [ "$fw" = 1 ] && [ -n "$FWDIR" ]; then
    t=$(fw_last)
    if [ -n "$t" ] && h=$(sha256sum "$t" | cut -c1-64) && [ "$h" != "$FWTGZ" ]; then
      log "$tag: a new firewall frame in $FWDIR, $t (${h:0:16}): building"
      f=$(fw_frame "$t")
      if [ -x "$f" ]; then
        FWTGZ=$h FC_FIREWALL_LEAN=$f
        log "$tag: firewall frame now from $t (${h:0:16})"
        sha256sum "$FC_FIREWALL_LEAN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
      else
        log "$tag: $t (${h:0:16}) did not build, the frame stays $FC_FIREWALL_LEAN: $(tail -3 "$OUT/build-firewall.log")"
        FWTGZ=$h
      fi
    fi
  fi
  [ "$fw" = 1 ] && sha256sum "$FC_FIREWALL_LEAN" > "$OUT/$tag/firewall.sha256"
  log "prove $tag (firewall $fw)"
  ( [ "$fw" = 1 ] || unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
    k0=$(oom_kills)
    if session "$D" "$OUT/$tag" && ! why=$(live_failed "$OUT/$tag"); then
      [ "$EACH" = 1 ] && printf '%(%s)T\n' -1 > "$OUT/lean/$tag.start"
      log "proved $tag: $(cat "$OUT/$tag/prove.time")"
      keep "$tag"
      if [ "$EACH" = 1 ]; then
        lean_check "$tag" $(( STAGE_GB / PAR ))
        log "lean $tag: $(tail -1 "$OUT/lean/$tag.out" 2>/dev/null | cut -c1-300) $(cat "$OUT/lean/$tag.time" 2>/dev/null)"
      fi
    else
      [ -n "${why:-}" ] || why=$(live_failed "$OUT/$tag")
      log "prove $tag FAILED, not accepted live: $why; the node's OOM kills $k0 -> $(oom_kills) during it; $(tail -2 "$OUT/$tag/prove.err" | tr '\n' ' ' | cut -c1-400)"
      echo "$why" > "$OUT/$tag/FAILED"
      keep "$tag"
    fi ) &
  [ "$fw" = 1 ] && [ -n "${H1L_SESSION_GB:-}" ] && sleep "${H1L_STAGGER:-60}"
done < <(order)
wait
unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
eval "$VLIMIT"
log "proved: $(ls -d "$WK"/st/*/sessions/*/ 2>/dev/null | wc -l) sessions"

# ---- Lean: the plan's checks -----------------------------------------------------------------------------------------------
cap=$(( STAGE_GB / LPAR ))
while IFS=$'\t' read -r kind tag D pub reg prog part _expect; do
  [ "$kind" = C ] || continue
  [ -e "$OUT/lean/$tag.time" ] && continue
  # a session the live verifier didn't accept is no evidence either way: Lean refuses a cut-off one, which a negative expects
  [ -e "$OUT/$tag/FAILED" ] && { echo "not accepted live: $(cat "$OUT/$tag/FAILED")" > "$OUT/lean/$tag.err"; continue; }
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$LPAR" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement verity/flock-circuit --public $pub --partition $part --program $prog --registered $reg --session $s" \
    > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
done < "$WK/plan.tsv"
wait
for tag in $(awk -F'\t' '$1 == "S" {print $2}' "$WK/plan.tsv"); do keep "$tag"; done

"$PY" "$H/served.py" summarize --work "$WK" --out "$OUT" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
failed=$(cd "$OUT" && ls -d */FAILED 2>/dev/null | cut -d/ -f1 | tr '\n' ' ')
[ -n "$failed" ] && { log "h1-served: NOT ACCEPTED LIVE, so not checked: $failed"; rc=1; }
log "h1-served: done (exit $rc)"
exit "$rc"
