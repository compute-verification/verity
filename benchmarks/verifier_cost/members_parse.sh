#!/usr/bin/env bash
# The per-member parse's cost against the members: members-parse-time (members_parse/Main.lean, built in a scratch Lake
# package against the checkout's verifier) makes, per N, N members of one range of units each with no unit in two, times
# `OneStageZk.membersOf` on them and then on the same members with one more whose range starts inside the last
# member's, and fails where the parse gives back other members than those made or accepts the unit in two. The parse
# reads each member once and sorts the members' ranges, so its time must grow no faster than N log N. It reads no file.
# Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --cwd clone -- \
#     bash benchmarks/verifier_cost/members_parse.sh 1000 25000 100000 1000000
# $OUT gets members_parse.jsonl (one row per N and case: members, ms, peak RSS).
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: members_parse.sh N..." >&2; exit 2; }
L=$PWD/verity/protocols/proofs/spot/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
mkdir -p "$S/time"
cp benchmarks/verifier_cost/members_parse/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "members_parse_time"
defaultTargets = ["members-parse-time"]

[[require]]
name = "core_spot"
path = "$L"

[[lean_exe]]
name = "members-parse-time"
root = "Main"
EOF
(cd "$S/time" && lake build members-parse-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
"$S/time/.lake/build/bin/members-parse-time" "$@" | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/members_parse.jsonl"
