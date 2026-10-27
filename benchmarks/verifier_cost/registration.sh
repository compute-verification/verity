#!/usr/bin/env bash
# R3's cost against the window's ports: registration.py writes, per N, a valid v1 audit with N more ports (a MoE window's
# shape; TPEVAL's w13 window at chunk cap 64 is 1,536 members, about 12k ports), and registration-time
# (registration/Main.lean, built in a scratch Lake package against the checkout's verifier) times `Registration.check`
# (the port index) and `Registration.checkRef` (R3 as first written) on it, refusing where their codes differ. `checkRef`
# runs only up to REF_MAX ports (default 12288). Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/registration.sh 1024 4096 12288 49152
# $OUT gets registration.jsonl (one row per N and function: ports, roots, ms, peak RSS, codes) and inputs.jsonl.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: registration.sh N..." >&2; exit 2; }
L=$PWD/verity/core/service/sampled_proofs
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/time"
cp benchmarks/verifier_cost/registration/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "registration_time"
defaultTargets = ["registration-time"]

[[require]]
name = "core_sampled_proofs"
path = "$L"

[[lean_exe]]
name = "registration-time"
root = "Main"
EOF
(cd "$S/time" && lake build registration-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/registration.py "$S/in" "$@" | tee "$O/inputs.jsonl"
for n in "$@"; do
  ref=(); [ "$n" -le "${REF_MAX:-12288}" ] || ref=(--no-ref)
  "$S/time/.lake/build/bin/registration-time" "$S/in/reg-$n.json" "${ref[@]}" | grep '^PHASE ' | sed 's/^PHASE //' \
    | tee -a "$O/registration.jsonl"
done
