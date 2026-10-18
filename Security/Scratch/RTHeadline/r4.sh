#!/usr/bin/env bash
# RT-HEADLINE r4 scratch (never lands): build what R4.lean imports at #2482's r4 head, then check it.
#   research run --queue --source . --project verity --cwd clone -- bash Security/Scratch/RTHeadline/r4.sh
set -uo pipefail
root=$(pwd); out=${OUT:-/tmp/rth-out}; mkdir -p "$out"; fail=0
bash tools/verity/lean/setup.sh Security/Proofs > "$out/setup-proofs.log" 2>&1 || fail=1
tail -5 "$out/setup-proofs.log"
export PATH="$HOME/.elan/bin:$PATH"
cd "$root/Security/Proofs" || exit 1
lake build Proofs.Accounting.Headline.Luck > "$out/build.log" 2>&1
rc=$?; [ $rc -eq 0 ] || { fail=1; tail -40 "$out/build.log"; }
lake env lean "$root/Security/Scratch/RTHeadline/R4.lean" > "$out/R4.txt" 2>&1
rc=$?; [ $rc -eq 0 ] || fail=1
cat "$out/R4.txt"
exit $fail
