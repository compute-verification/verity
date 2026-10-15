#!/usr/bin/env bash
set -euo pipefail

git rev-parse HEAD | tee "$OUT/head.txt"
bash tools/verity/lean/setup.sh Security/Proofs > "$OUT/setup.txt" 2>&1
export PATH="$HOME/.elan/bin:$PATH"
cd Security/Proofs
lake build Proofs.Flock.C3.Compose.SrcOKFamily Proofs.Flock.C3.Compose.AtHoloHeld Proofs.Flock.C3.Compose.DecodeFix \
  > "$OUT/build.txt" 2>&1 || { tail -80 "$OUT/build.txt"; exit 1; }
tail -5 "$OUT/build.txt"
set +e
lake env lean ../../p1-probe.lean > "$OUT/probes.txt" 2>&1
rc=$?
set -e
cat "$OUT/probes.txt"
echo "probe exit $rc" | tee -a "$OUT/probes.txt"
exit "$rc"
