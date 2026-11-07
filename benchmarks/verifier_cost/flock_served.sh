#!/usr/bin/env bash
# The Lean verifier on a served flat statement: flock_served.py stages one kind of a saved served step as one_stage/a11.py
# stages it, then this times `flock-verify archive-put` and `flock-verify verify --zk --partition --program` on an a11
# session over it, as a0.lean_verify runs them (under `perf record` where the node has perf), and with --phases flock-read's
# phases of the read (flock_read/Main.lean). Run from a clone (the builds write the verifier's .lake), sending the values
# and the session's directory as a tarball:
#   research run --as proofs --project verity --on auto --mem-gb 128 --timeout 7200 --source . --cwd clone \
#     --send values.tar.gz --send session.tar -- bash benchmarks/verifier_cost/flock_served.sh KIND [N] [--phases]
# N: the kind's first N saved units, as a11's --per-kind staged them (the session's population). $OUT gets build.json
# (flock_served.py's), archive-put.{log,time}, verify.{log,time}, perf.txt, and with --phases phases.jsonl and read.time.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
kind=${1:?usage: flock_served.sh KIND [N] [--phases]}
shift
n= phases=
for a in "$@"; do
  case $a in --phases) phases=$a ;; *) n=$a ;; esac
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
w=$S/w
/usr/bin/time -v uv run python benchmarks/verifier_cost/flock_served.py "$values" "$kind" "$w" $n > "$O/stage.out" 2> "$O/stage.err" \
  || { tail -40 "$O/stage.err" >&2; exit 1; }
cp "$w/build.json" "$O/build.json"
cat "$O/build.json"
pub=$(ls "$w"/stmt/pub-*.bin)
/usr/bin/time -v -o "$O/archive-put.time" "$FV" archive-put "$S/archive" "$w/circuit.txt" "$pub" "$w/partition.json" \
  "$w/program.json" > "$O/archive-put.log"
key() { awk -v f="$(basename "$1")" '{ n = split($2, p, "/"); if (p[n] == f) print $1 }' "$O/archive-put.log"; }
if [ -n "$session" ]; then
  mkdir -p "$S/session"
  tar -xf "$session" -C "$S/session"
  sdir=$(dirname "$(find "$S/session" -name session.json | head -1)")
  cmd=("$FV" verify --statement verity/flock-circuit --archive "$S/archive" --circuit "$(key "$w/circuit.txt")"
       --public "$(key "$pub")" --partition "$(key "$w/partition.json")" --program "$(key "$w/program.json")" --zk --session "$sdir")
  if command -v perf > /dev/null; then
    /usr/bin/time -v -o "$O/verify.time" perf record -F 49 -o "$S/perf.data" -- "${cmd[@]}" > "$O/verify.log" 2>&1 || true
    perf report -i "$S/perf.data" --stdio --no-children --sort symbol --percent-limit 0.3 2> /dev/null | head -400 > "$O/perf.txt" || true
    perf script -i "$S/perf.data" -F time,sym 2> /dev/null | gzip > "$O/perf-samples.txt.gz" || true
  else
    /usr/bin/time -v -o "$O/verify.time" "${cmd[@]}" > "$O/verify.log" 2>&1 || true
  fi
  cat "$O/verify.log"
fi
if [ "$phases" = "--phases" ]; then
  mkdir -p "$S/read"
  cp benchmarks/verifier_cost/flock_read/Main.lean "$L/lean-toolchain" "$S/read/"
  cat > "$S/read/lakefile.toml" <<EOF
name = "flock_read"
defaultTargets = ["flock-read"]

[[require]]
name = "flock_verifier"
path = "$L"

[[lean_exe]]
name = "flock-read"
root = "Main"
EOF
  (cd "$S/read" && lake build flock-read) >> "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
  /usr/bin/time -v -o "$O/read.time" "$S/read/.lake/build/bin/flock-read" "$w/circuit.txt" "$pub" verity/flock-circuit \
    > "$O/read.out" || true
  grep '^PHASE ' "$O/read.out" | sed 's/^PHASE //' > "$O/phases.jsonl" || true
  cat "$O/phases.jsonl"
fi
