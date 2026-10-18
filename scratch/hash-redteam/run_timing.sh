#!/usr/bin/env bash
# hash-redteam scratch: Timing.lean on a node, its log to $OUT/timing.txt as it goes.
set -euo pipefail
bash tools/verity/lean/setup.sh Security/Properties
cd Security/Properties
lake build Definitions.Hash PouwCommit.Rows
ln -sf "${OUT:-.}/timing.txt" timing.txt 2>/dev/null || true
lake env lean ../../scratch/hash-redteam/Timing.lean 2>&1 | tee "${OUT:-.}/timing-lean.txt"
