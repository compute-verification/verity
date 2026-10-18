#!/usr/bin/env bash
# hash-redteam scratch: RowProbe.lean on a node (research run --cwd clone), output to $OUT/probes.txt.
set -euo pipefail
cp scratch/hash-redteam/cases.json Security/Properties/cases.json
bash tools/verity/lean/setup.sh Security/Properties
cd Security/Properties
lake build Definitions.Hash PouwCommit.Rows
ln -sf "${OUT:-.}/progress.txt" progress.txt 2>/dev/null || true
export RT_ONLY=${RT_ONLY:-}
time lake env lean ../../scratch/hash-redteam/RowProbe.lean 2>&1 | tee "${OUT:-.}/probes.txt"
