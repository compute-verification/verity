#!/usr/bin/env bash
# The per-member form's pass over every member's files against the members of one circuit file: members-files-time
# (members_files/Main.lean, built in a scratch Lake package against the checkout's verifier) times `membersFiles` as the
# compiled check runs it and the pass as stated (`Member.files` a member, up to REF_MAX members, default 150) on one
# per-member case, at its first N members of one circuit file (`all`: every member of the case), and exits 1 where the
# two disagree or a member's files are not held. The case is MF_CASE's (a driver's case-v2-members.json), or the one
# members_files.py makes: a stand-in window of MEMBERS members (default 300) over two templates
# (benchmarks/one_stage/members_standin.py's), audited on Flock's prover (check_build.sh bin: the machine must hold the
# Flock checkout and the toolchain check's flock-circuit-build step leaves), this tree's Lean verifier and ZK server.
# v2's check must accept the case. Run from a clone (the builds write the verifiers' .lake):
#   research run --as proofs --project verity --on auto --source . --cwd clone --mem-gb 32 -- \
#     bash benchmarks/verifier_cost/members_files.sh 1 2 16 150 all
# $OUT gets members_files.jsonl (the case's row, then one row per N and pass: members, circuit files, microseconds, peak
# RSS, whether every member's files were held, the result's digest), inputs.jsonl (the case, the audit's seconds and the
# whole check's) and each build's log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: members_files.sh N|all..." >&2; exit 2; }
L=$PWD/verity/protocols/spot/verifier/lean
F=$PWD/verity/protocols/flock/verifier
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
if [ -n "${MF_CASE:-}" ]; then
  from=(--case "$MF_CASE")
else
  # what the audit can't do without, first: each ends at once where the machine lacks it, before any Lean is built
  git rev-list --first-parent -1 HEAD > /dev/null \
    || { echo "no git history here, and the audit's local challenger is pinned on the checkout's" >&2; exit 1; }
  bash verity/protocols/flock/prover/check_build.sh bin "$S/flock-circuit" > "$O/prover-build.log" 2>&1 \
    || { tail -20 "$O/prover-build.log" >&2; exit 1; }
  from=(--members "${MEMBERS:-300}" --bin "$S/flock-circuit" --lean-bin "$F/.lake/build/bin")
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
  (cd "$F" && lake build flock-verify flock-firewall) > "$O/verifier-build.log" 2>&1 \
    || { tail -60 "$O/verifier-build.log" >&2; exit 1; }
fi
uv run python benchmarks/verifier_cost/members_files.py "$S/in" "${from[@]}" | tee "$O/inputs.jsonl"
"$S/time/.lake/build/bin/members-files-time" "$S/in/params.json" "$S/in/commitment.bin" "$S/in/blobs.txt" "$@" \
  --ref-max "${REF_MAX:-150}" | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/members_files.jsonl"
