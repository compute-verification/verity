#!/usr/bin/env bash
# The Lean verifier's seconds on a served block type's sessions, on one core, by the statement's blocks.
#   1. hidden_zk/job.sh's builds (flock-circuit's proving build on CPUs, the Lean flock-verify, a venv), and flock-read
#      (flock_read/Main.lean) in a scratch Lake package against a copy of the built verifier
#   2. flock_sessions.py stage: the block type's flat statement at `full`, as served_units/stage.py stages a census type,
#      and from the same lowering the statement's first N blocks for each N given
#   3. per N: one loopback --zk session of the CPU prover sampling all N blocks (job.sh's `session`), then
#      `flock-verify verify --zk` on it FS_REPS times, as the proof server's verdict pool calls it (the statement and
#      the public file by their archive keys), pinned to one core under /usr/bin/time -v; for the first N, a second
#      session of the same statement and one `verify` handed both, a call of two sessions
#   4. sessions.tar, where it is sent: a proof server's saved sessions of the full statement (one directory a session),
#      all handed to one `verify` FS_REPS times, then the first of them alone
#   5. flock-read on the circuit (the read's phases), one `verify` under `perf record` where the node has perf, and
#      flock_sessions.py summary
# `verify` reads the statement once a call (its `inputs` line), then sets up and checks each session it was handed (the
# session's own lines), so a call's seconds split into a part paid once a circuit and a part paid a session; summary.json
# has both and their growth with the blocks. A census statement binds no Program, so the verifier refuses every session
# here for want of --program once every check has passed.
# Run from the source tree's root:
#   research run --as proofs --project verity --on auto --mem-gb G --source . --cwd source --declared-output 'out/*' \
#     [--send sessions.tar] -- env FS_MEM_GB=G bash benchmarks/verifier_cost/flock_sessions.sh 'census:NAME' [N...]
# N...: the block counts (default 1 2 4; one at or above `full` is left out).
# env: FS_MEM_GB (the job's --mem-gb)  FS_REPS=2  FS_CORE (the core every verify is pinned to; default the last one this
#      process may run on)  FS_PIN (a prefix the staged circuit's SHA-512 must have)  FS_CACHE (job.sh's HZ_CACHE)
# $OUT gets stage.json and pin.txt (stage.py's), per call TAG.{err,out,time} (TAG nN-repI, nN-pair, saved-repI or saved-one), per N
# nN/ (job.sh's session files and session.json), phases.jsonl, perf-symbols.txt and summary.json.
set -uo pipefail
SPEC=${1:?usage: flock_sessions.sh census:NAME [N...]}
shift
NS=${*:-1 2 4}
REPS=${FS_REPS:-2}
export HZ_CONFIGS=sessions HZ_MEM_GB=${FS_MEM_GB:?set it to the --mem-gb of the job} HZ_RUNS=1 HZ_WARM=0
[ -z "${FS_CACHE:-}" ] || export HZ_CACHE=$FS_CACHE
# shellcheck source=../pouw/hidden_zk/job.sh
source benchmarks/pouw/hidden_zk/job.sh
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
export FC_FIREWALL=0
unset FC_HOST_WITNESS
I=$RESEARCH_RUN_DIR/inputs
V=benchmarks/verifier_cost
spent() { grep -E 'User time|System time|Elapsed|Maximum resident|Exit status' "$1" | sed 's/^\s*//; s/ (h:mm:ss or m:ss)//' | paste -sd ';'; }

uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || { tail -20 "$OUT/sync.log"; exit 2; }
D=$W/st
log "flock-sessions: $SPEC at full and at ($NS) blocks, $REPS verifies each"
# shellcheck disable=SC2086
( /usr/bin/time -v -o "$OUT/stage.time" nice -n 19 uv run --no-sync python "$V/flock_sessions.py" stage "$SPEC" "$D" $NS \
    > "$OUT/stage.out" 2> "$OUT/stage.err"; echo $? > "$W/stage.rc" ) &
staging=$!
setup

READ=$W/read/.lake/build/bin/flock-read
# job.sh's build is the verifier beside the packages it requires by path (lean_copy), its .lake a link into the verifier's
# own directory there: all of it is copied, and flock-read requires the verifier at its place in the copy
BUILT=$(cd "$(dirname "$LEAN")/../../.." && pwd -P)
PKG=$(cd -P "$(dirname "$LEAN")/../../.." && pwd)
PKG=$W/verifier${PKG#"$BUILT"}
cp -a "$BUILT" "$W/verifier" && mkdir -p "$W/read" && cp "$V/flock_read/Main.lean" "$PKG/lean-toolchain" "$W/read/" \
  && printf 'name = "flock_read"\ndefaultTargets = ["flock-read"]\n\n[[require]]\nname = "flock_verifier"\npath = "%s"\n\n[[lean_exe]]\nname = "flock-read"\nroot = "Main"\n' \
    "$PKG" > "$W/read/lakefile.toml" && (cd "$W/read" && lake build flock-read) > "$OUT/build-read.log" 2>&1
[ -x "$READ" ] || { log "flock-read did not build: $(tail -3 "$OUT/build-read.log" 2> /dev/null | tr '\n' '|' | cut -c1-300)"; READ=; }

wait $staging
rc=$(cat "$W/stage.rc" 2> /dev/null || echo 99)
log "staged, exit $rc: $(spent "$OUT/stage.time"); $(grep '^flock_sessions' "$OUT/stage.err" | tr '\n' ';')"
[ "$rc" = 0 ] || { tail -20 "$OUT/stage.err"; exit "$rc"; }
cp "$D/stage.json" "$D/pin.txt" "$OUT/"
PIN=$(cat "$D/pin.txt")
log "pin ${PIN:0:16}, circuit $(stat -c %s "$D/circuit.txt") bytes"
[[ $PIN == "${FS_PIN:-}"* ]] || { log "the staged circuit's pin is not FS_PIN's ($FS_PIN)"; exit 3; }

CORE=${FS_CORE:-$(taskset -pc $$ | sed 's/.*: //; s/.*[,-]//')}
A=$W/archive
/usr/bin/time -v -o "$OUT/archive-put.time" taskset -c "$CORE" "$LEAN" archive-put "$A" "$D/circuit.txt" > "$OUT/archive-put.out" 2>&1
CK=$(awk 'NR == 1 {print $1}' "$OUT/archive-put.out")
log "archive-put of the circuit on core $CORE: $(spent "$OUT/archive-put.time")"

verify() {   # verify TAG PUB SESSION...: one `verify` of the sessions on CORE, its phase lines and verdicts to the log
  local tag=$1 pub=$2 pk
  shift 2
  pk=$(taskset -c "$CORE" "$LEAN" archive-put "$A" "$pub" | awk 'NR == 1 {print $1}')
  # every argument after --session is a session's directory; verifier_env is job.sh's, the pool's allocator setting
  # shellcheck disable=SC2046
  /usr/bin/time -v -o "$OUT/$tag.time" taskset -c "$CORE" $(verifier_env) "$LEAN" verify --statement verity/flock-circuit --archive "$A" \
    --circuit "$CK" --public "$pk" --zk --session "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  log "$tag exit $?: $(grep -E ' ms$' "$OUT/$tag.err" | sed 's|^/.*/||' | tr '\n' ';') $(spent "$OUT/$tag.time")"
}

SAVED=()
if [ -f "$I/sessions.tar" ]; then
  mkdir -p "$W/saved" && tar -xf "$I/sessions.tar" -C "$W/saved"
  for d in "$W"/saved/*/; do [ -f "$d/session.json" ] && SAVED+=("${d%/}"); done
  log "saved sessions: ${#SAVED[@]} (${SAVED[*]##*/})"
  [ ${#SAVED[@]} -gt 0 ] || { log "sessions.tar holds no directory with a session.json"; exit 4; }
  for i in $(seq 1 "$REPS"); do verify "saved-rep$i" "$D/pub.bin" "${SAVED[@]}"; done
  verify saved-one "$D/pub.bin" "${SAVED[0]}"
fi

FIRST=
for N in $NS; do
  d=$D-n$N O=$OUT/n$N
  [ -f "$d/pub.bin" ] || { log "n$N: not staged"; continue; }
  mkdir -p "$O" && ln -s "$D/circuit.txt" "$D/pin.txt" "$d/"
  session "$d" "$O" || { log "n$N: the CPU session failed: $(tail -2 "$O/prove.err" | tr '\n' '|' | cut -c1-300)"; continue; }
  s=$(ls -dt "$d"/sessions/*/ 2> /dev/null | head -1)
  [ -n "$s" ] || { log "n$N: no session record"; continue; }
  cp "$s/session.json" "$O/"
  log "n$N: proved: $(cat "$O/prove.time")"
  for i in $(seq 1 "$REPS"); do verify "n$N-rep$i" "$d/pub.bin" "${s%/}"; done
  if [ -z "$FIRST" ]; then   # the verifier refuses a session key it has seen in the call, so the pair is two sessions
    mkdir -p "$O/second"
    if session "$d" "$O/second"; then
      s2=$(ls -dt "$d"/sessions/*/ | head -1)
      [ "$s2" = "$s" ] || verify "n$N-pair" "$d/pub.bin" "${s%/}" "${s2%/}"
    else
      log "n$N: the second CPU session failed"
    fi
    FIRST="$d/pub.bin ${s%/}"
  fi
done

if [ -n "$READ" ]; then
  /usr/bin/time -v -o "$OUT/read.time" taskset -c "$CORE" "$READ" "$D/circuit.txt" "$D/pub.bin" verity/flock-circuit > "$OUT/read.out" 2> "$OUT/read.err"
  grep '^PHASE ' "$OUT/read.out" | sed 's/^PHASE //' > "$OUT/phases.jsonl"
  log "flock-read: $(spent "$OUT/read.time")"
fi
[ ${#SAVED[@]} -eq 0 ] || FIRST="$D/pub.bin ${SAVED[0]}"
if [ -n "$FIRST" ] && command -v perf > /dev/null; then
  pk=$(taskset -c "$CORE" "$LEAN" archive-put "$A" "${FIRST% *}" | awk 'NR == 1 {print $1}')
  # shellcheck disable=SC2046
  taskset -c "$CORE" perf record -F 99 -o "$W/perf.data" -- $(verifier_env) "$LEAN" verify --statement verity/flock-circuit --archive "$A" \
    --circuit "$CK" --public "$pk" --zk --session "${FIRST#* }" > "$OUT/perf-verify.out" 2> "$OUT/perf-verify.err"
  perf report -i "$W/perf.data" --stdio --no-children --sort symbol --percent-limit 0.3 2> /dev/null | grep -v '^#' | grep -v '^$' \
    > "$OUT/perf-symbols.txt"
fi
"$PY" "$V/flock_sessions.py" summary "$OUT" | tee -a "$OUT/job.log"
log "flock-sessions: done"
