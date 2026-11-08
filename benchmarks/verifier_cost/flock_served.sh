#!/usr/bin/env bash
# The Lean verifier on a served statement: flock_served.py stages one kind of a saved served step as one_stage/a11.py
# stages it (flat), or with --layout MODE as a templated statement of the same unit (with --structural, struct_lower's
# template, the flat unit never built). With a session sent, this times `flock-verify archive-put` and `flock-verify
# verify --zk --partition --program` on it as a0.lean_verify runs them (under `perf record` where the node has perf).
# With --phases or --layout, flock-served (flock_served/Main.lean)
# times verify's work on the statement before any session a step at a time, and on a flat one flock-read
# (flock_read/Main.lean) its parse's phases. Run from a clone (the builds write the verifier's .lake):
#   research run --as proofs --project verity --on auto --mem-gb 128 --timeout 7200 --source . --cwd clone \
#     --send values.tar.gz [--send session.tar] -- bash benchmarks/verifier_cost/flock_served.sh KIND [N] [--phases] \
#     [--layout MODE [--structural]]
# N: the kind's first N saved units, as a11's --per-kind staged them (the session's population). $OUT gets build.json
# (flock_served.py's), archive-put.{log,time}, verify.{log,time}, perf.txt and perf-samples.txt.gz, served.jsonl and
# served.time, and on a flat statement with --phases phases.jsonl and read.time.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
kind=${1:?usage: flock_served.sh KIND [N] [--phases] [--layout MODE [--structural]]}
shift
n= phases= layout= structural=
while [ $# -gt 0 ]; do
  case $1 in
    --phases) phases=1 ;;
    --layout) layout=$2; shift ;;
    --structural) structural=1 ;;
    *) n=$1 ;;
  esac
  shift
done
I=${RESEARCH_RUN_DIR:?}/inputs
values=$(ls "$I"/*.tar.gz | head -1)
session=$(ls "$I"/*.tar 2> /dev/null | head -1 || true)
L=$PWD/verity/core/service/flock
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
(cd "$L" && lake build flock-verify) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
FV=$L/.lake/build/bin/flock-verify
scratch() { # scratch NAME: NAME's Main.lean in a scratch Lake package against the checkout's verifier
  mkdir -p "$S/$1"
  cp "benchmarks/verifier_cost/${1//-/_}/Main.lean" "$L/lean-toolchain" "$S/$1/"
  printf 'name = "%s"\ndefaultTargets = ["%s"]\n\n[[require]]\nname = "flock_verifier"\npath = "%s"\n\n[[lean_exe]]\nname = "%s"\nroot = "Main"\n' \
    "${1//-/_}" "$1" "$L" "$1" > "$S/$1/lakefile.toml"
  (cd "$S/$1" && lake build "$1") >> "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
}
w=$S/w
/usr/bin/time -v uv run python benchmarks/verifier_cost/flock_served.py "$values" "$kind" "$w" $n ${layout:+--layout "$layout"} \
  ${structural:+--structural} > "$O/stage.out" 2> "$O/stage.err" || { tail -40 "$O/stage.err" >&2; exit 1; }
cp "$w/build.json" "$O/build.json"
cat "$O/build.json"
stmt=verity/flock-circuit${layout:+/tmpl}
pub=$(ls "$w"/stmt/pub-*.bin)
/usr/bin/time -v -o "$O/archive-put.time" "$FV" archive-put "$S/archive" "$w/circuit.txt" "$pub" "$w/partition.json" \
  "$w/program.json" > "$O/archive-put.log"
key() { awk -v f="$(basename "$1")" '{ n = split($2, p, "/"); if (p[n] == f) print $1 }' "$O/archive-put.log"; }
if [ -n "$session" ]; then
  mkdir -p "$S/session"
  tar -xf "$session" -C "$S/session"
  sdir=$(dirname "$(find "$S/session" -name session.json | head -1)")
  cmd=("$FV" verify --statement "$stmt" --archive "$S/archive" --circuit "$(key "$w/circuit.txt")" --public "$(key "$pub")"
       --partition "$(key "$w/partition.json")" --program "$(key "$w/program.json")" --zk --session "$sdir")
  if command -v perf > /dev/null; then
    /usr/bin/time -v -o "$O/verify.time" perf record -F 49 -o "$S/perf.data" -- "${cmd[@]}" > "$O/verify.log" 2>&1 || true
    perf report -i "$S/perf.data" --stdio --no-children --sort symbol --percent-limit 0.3 2> /dev/null | head -400 > "$O/perf.txt" || true
    perf script -i "$S/perf.data" -F time,ip,sym 2> /dev/null | gzip > "$O/perf-samples.txt.gz" || true
  else
    /usr/bin/time -v -o "$O/verify.time" "${cmd[@]}" > "$O/verify.log" 2>&1 || true
  fi
  cat "$O/verify.log"
fi
if [ -n "$phases" ] || [ -n "$layout" ]; then
  scratch flock-served
  /usr/bin/time -v -o "$O/served.time" "$S/flock-served/.lake/build/bin/flock-served" "$w/circuit.txt" "$pub" "$stmt" \
    "$w/partition.json" "$w/program.json" > "$O/served.out" || true
  grep '^PHASE ' "$O/served.out" | sed 's/^PHASE //' > "$O/served.jsonl" || true
  cat "$O/served.jsonl"
fi
if [ -n "$phases" ] && [ -z "$layout" ]; then
  scratch flock-read
  /usr/bin/time -v -o "$O/read.time" "$S/flock-read/.lake/build/bin/flock-read" "$w/circuit.txt" "$pub" "$stmt" \
    > "$O/read.out" || true
  grep '^PHASE ' "$O/read.out" | sed 's/^PHASE //' > "$O/phases.jsonl" || true
  cat "$O/phases.jsonl"
fi
