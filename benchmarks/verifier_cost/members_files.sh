#!/usr/bin/env bash
# The per-member form's pass over every member's files against the members of one circuit file: members-files-time
# (members_files/Main.lean, built in a scratch Lake package against the checkout's verifier) times `membersFiles` as the
# compiled check runs it and the pass as stated (`Member.files` a member, up to REF_MAX members, default 4096) on one
# per-member case, its first member handed in N times over, and exits 1 where the two disagree or a member's files are
# not held. The case is MF_CASE's (a driver's case-v2-members.json), or the one members_files.py makes by running the
# two-member audit of benchmarks/one_stage's test on Flock's prover (check_build.sh bin: the machine must hold the Flock
# checkout and the toolchain check's flock-circuit-build step leaves) and this tree's Lean verifier and ZK server; the
# compiled check must accept it. Run from a clone (the builds write the verifiers' .lake):
#   research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 -- \
#     bash benchmarks/verifier_cost/members_files.sh 1 2 64 512 4096 32768
# $OUT gets members_files.jsonl (the case's row, then one row per N and pass: members, microseconds, peak RSS, whether
# every member's files were held, the result's digest), inputs.jsonl and each build's log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: members_files.sh N..." >&2; exit 2; }
L=$PWD/verity/protocols/spot/verifier/lean
F=$PWD/verity/protocols/flock/verifier
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/time"
cp benchmarks/verifier_cost/members_files/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOT
name = "members_files_time"
defaultTargets = ["members-files-time"]

[[require]]
name = "core_spot"
path = "$L"

[[lean_exe]]
name = "members-files-time"
root = "Main"
EOT
(cd "$S/time" && lake build members-files-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
if [ -n "${MF_CASE:-}" ]; then
  from=(--case "$MF_CASE")
else
  (cd "$F" && lake build flock-verify flock-firewall) > "$O/verifier-build.log" 2>&1 \
    || { tail -60 "$O/verifier-build.log" >&2; exit 1; }
  bash verity/protocols/flock/prover/check_build.sh bin "$S/flock-circuit" > "$O/prover-build.log" 2>&1 \
    || { tail -20 "$O/prover-build.log" >&2; exit 1; }
  B=$F/.lake/build/bin
  from=(--bin "$S/flock-circuit" --lean "$B/flock-verify" --firewall "$B/flock-firewall")
fi
uv run python benchmarks/verifier_cost/members_files.py "$S/in" "${from[@]}" | tee "$O/inputs.jsonl"
"$S/time/.lake/build/bin/members-files-time" "$S/in/params.json" "$S/in/commitment.bin" "$S/in/blobs.txt" "$@" \
  --ref-max "${REF_MAX:-4096}" | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/members_files.jsonl"
