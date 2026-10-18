#!/usr/bin/env bash
# RT-I8 scratch (never lands): build the scratch files' imports and elaborate each, outputs in $OUT.
#   research run --queue --source . --project verity --cwd clone -- bash Security/Scratch/RTI8/run.sh
set -uo pipefail
root=$(pwd)
out=${OUT:-/tmp/rti8-out}
mkdir -p "$out"
fail=0
bash tools/verity/lean/setup.sh Security/Proofs > "$out/setup-proofs.log" 2>&1 || fail=1
echo "setup-proofs fail=$fail" | tee -a "$out/rc.txt"
export PATH="$HOME/.elan/bin:$PATH"
cd "$root/Security/Proofs" || exit 1
lake build Proofs.Accounting.TS.Drawn Models.Accounting.Assumptions.Compose > "$out/build.log" 2>&1
rc=$?; [ $rc -eq 0 ] || fail=1
echo "build rc=$rc" | tee -a "$out/rc.txt"
tail -20 "$out/build.log"
for f in Drawn Compose; do
  lake env lean "$root/Security/Scratch/RTI8/$f.lean" > "$out/$f.txt" 2>&1
  rc=$?; [ $rc -eq 0 ] || fail=1
  echo "$f rc=$rc" | tee -a "$out/rc.txt"
  cat "$out/$f.txt"
done
exit $fail
