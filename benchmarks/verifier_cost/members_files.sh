#!/usr/bin/env bash
# The per-member form's pass over every member's files against the members of one circuit file: members-files-time
# (members_files/Main.lean, built in a scratch Lake package against the checkout's verifier) times `membersFiles` as the
# compiled check runs it and the pass as stated (`Member.files` a member, up to REF_MAX members, default 150) on one
# per-member case, at its first N members of one circuit file (`all`: every member of the case), and exits 1 where the
# two disagree or a member's files are not held. The case is MF_CASE's (a driver's case-v2-members.json), or the one
# members_files.py makes: a stand-in window of MEMBERS members (default 300) over two templates
# (benchmarks/one_stage/members_standin.py's), audited on Flock's prover (check_build.sh bin: the machine must hold the
# Flock checkout and the toolchain check's flock-circuit-build step leaves), this tree's Lean verifier and ZK server.
# MEMBERS may name several sizes ("300 1200"): a window, a case and its rows each, so the rows at one N are the pass
# against the window's members at a fixed count of members handed. A stand-in window's program does not grow with its
# members (its root holds the templates' nodes), so these rows do not vary the program's size.
# v2's check must accept each case. Run from a clone (the builds write the verifiers' .lake):
#   research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 -- \
#     bash benchmarks/verifier_cost/members_files.sh 1 2 16 150 all
#   research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 --env "MEMBERS=300 1200" -- \
#     bash benchmarks/verifier_cost/members_files.sh 1 16 150
# $OUT gets members_files.jsonl (for each case its row: its members, its program's bytes and root nodes and the
# microseconds of one evaluation of its partition's query's strata; then one row per N and pass: members, circuit files,
# microseconds, peak RSS, whether every member's files were held, the result's digest), inputs.jsonl (each case, the
# audit's seconds and the whole check's) and each build's log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: members_files.sh N|all..." >&2; exit 2; }
L=$PWD/verity/protocols/proofs/spot/verifier/lean
F=$PWD/verity/protocols/proofs/flock/verifier
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
if [ -z "${MF_CASE:-}" ]; then
  # what the audit can't do without, first: each ends at once where the machine lacks it, before any Lean is built
  git rev-list --first-parent -1 HEAD > /dev/null \
    || { echo "no git history here, and the audit's local challenger is pinned on the checkout's" >&2; exit 1; }
  bash verity/protocols/proofs/flock/prover/check_build.sh bin "$S/flock-circuit" > "$O/prover-build.log" 2>&1 \
    || { tail -20 "$O/prover-build.log" >&2; exit 1; }
fi
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
if [ -z "${MF_CASE:-}" ]; then
  (cd "$F" && lake build verity-verify-proof flock-firewall) > "$O/verifier-build.log" 2>&1 \
    || { tail -60 "$O/verifier-build.log" >&2; exit 1; }
fi
sizes=("$@")
# one case's rows: its directory, then members_files.py's arguments for the case
rows() {
  local d=$1
  shift
  uv run python benchmarks/verifier_cost/members_files.py "$d" "$@" | tee -a "$O/inputs.jsonl"
  "$S/time/.lake/build/bin/members-files-time" "$d/params.json" "$d/commitment.bin" "$d/blobs.txt" "${sizes[@]}" \
    --ref-max "${REF_MAX:-150}" | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/members_files.jsonl"
}
if [ -n "${MF_CASE:-}" ]; then
  rows "$S/in" --case "$MF_CASE"
else
  for m in ${MEMBERS:-300}; do
    rows "$S/in-$m" --members "$m" --bin "$S/flock-circuit" --lean-bin "$F/.lake/build/bin"
  done
fi
