#!/usr/bin/env bash
# RT-HEADLINE scratch (never lands): build what the scratch imports, then check it.
#   research run --queue --source . --project verity --cwd clone -- bash Security/Scratch/RTHeadline/run.sh
set -uo pipefail
root=$(pwd); out=${OUT:-/tmp/rth-out}; mkdir -p "$out"; fail=0
bash tools/verity/lean/setup.sh Security/Proofs > "$out/setup-proofs.log" 2>&1 || fail=1
tail -5 "$out/setup-proofs.log"
export PATH="$HOME/.elan/bin:$PATH"
cd "$root/Security/Proofs" || exit 1
lake build Proofs.Accounting.Headline.Luck Proofs.Accounting.TS.ProofsSound Models.Accounting.Guarantees.Headline \
  > "$out/build.log" 2>&1
rc=$?; [ $rc -eq 0 ] || { fail=1; tail -40 "$out/build.log"; }
lake env lean "$root/Security/Scratch/RTHeadline/Headline.lean" > "$out/Headline.txt" 2>&1
rc=$?; [ $rc -eq 0 ] || fail=1
cat "$out/Headline.txt"
exit $fail
