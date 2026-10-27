#!/usr/bin/env bash
# Q_call v1's cost on Serve_v1: `flock-verify qcall-program` at each X and the one-stage `partition` piece under
# `Q_call` v1 at X = 16, on one config's word-level descriptor, each under `/usr/bin/time -v`, and a `perf` profile of
# `qcall-program` at the first X where the node has perf. Run from a clone (the builds write .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-4 --source . --cwd clone -- \
#     bash benchmarks/verifier_cost/qcall.sh b0 16 32
# $OUT gets build.log, sizes.json, qcall-X.{json.sha256,time,head} per X, partition.{json,time} and perf-X.txt.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
cfg=${1:?usage: qcall.sh CONFIG X...}
shift
uv sync -q --all-packages --extra torch-cpu
{ (cd verity/core/service/flock && lake build flock-verify) && \
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-difftest); } > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log" >&2; exit 1; }
FV=$PWD/verity/core/service/flock/.lake/build/bin/flock-verify
SP=$PWD/verity/core/service/sampled_proofs/.lake/build/bin/sampled-proofs-difftest
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
uv run python - "$cfg" "$d" "$O" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/verifier_cost")
import run
from verity.ml.circuits import partition_object as PO
cfg, d, o = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
prog = run.serve(cfg, "word")
text = run.text_of(prog)
(d / "desc.json").write_text(text)
(d / "case.jsonl").write_text(json.dumps(run._partition_case(json.loads(text), PO.call_query(16))) + "\n")
distinct = {PO.template_id(fn): fn.gates for _, fn in PO.calls(prog)}
(o / "sizes.json").write_text(json.dumps({"config": cfg, "description_bytes": len(text.encode()),
    "unrolled_gates": prog.root.gates, "call_definition_gates": sum(distinct.values()), "calls": distinct}, indent=1))
EOF
for X in "$@"; do
  /usr/bin/time -v -o "$O/qcall-$X.time" "$FV" qcall-program "$d/desc.json" "$X" > "$d/q.json" || true
  sha256sum "$d/q.json" | cut -d' ' -f1 > "$O/qcall-$X.json.sha256"
  head -c 4000 "$d/q.json" > "$O/qcall-$X.head"
  if [ "$X" = "$1" ] && command -v perf > /dev/null; then
    perf record -F 199 -g -o "$d/perf.data" -- "$FV" qcall-program "$d/desc.json" "$X" > /dev/null 2> "$d/perf.err" || true
    perf report -i "$d/perf.data" --stdio --no-children --percent-limit 1 2> /dev/null | head -300 > "$O/perf-$X.txt" || true
  fi
done
/usr/bin/time -v -o "$O/partition.time" "$SP" < "$d/case.jsonl" | head -c 4000 > "$O/partition.json" || true
grep -h "Elapsed\|Maximum resident" "$O"/*.time
