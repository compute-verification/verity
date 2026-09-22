#!/usr/bin/env bash
# [E6 CURVE 2026-09-13] Pull ONLY the slim re-pin files from the volume to the laptop (disk: ~3 GB free; population.v2.json / pipeline files stay on the
# volume): per size instance/{protocol_instance,expected_population,inventory,pins,coverage_table}.json + inventory.md, SHA256SUMS, IDENTITY.json,
# BUILD.log; rows/<size>/ROWS_IDENTITY.json + ROWS.log; rule_checks/, bounds/, PINS.md, program_digests.json, chain.log -> out/scale/e6/cv/.
# Works while the pod is alive (via its ssh) -- or from ANY pod mounting the volume: pass the ssh wrapper as $1.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
SSHW=${1:-out/scale/e6/pods/cv-e6-repin.ssh}
DST=out/scale/e6/cv; mkdir -p "$DST/pinned" "$DST/rows"
"$SSHW" 'cd /workspace/cv/e6 && tar -cz --ignore-failed-read \
   pinned/*/instance/protocol_instance.json pinned/*/instance/expected_population.json pinned/*/instance/inventory.json pinned/*/instance/inventory.md \
   pinned/*/instance/pins.json pinned/*/instance/coverage_table.json pinned/*/SHA256SUMS pinned/*/IDENTITY.json pinned/*/BUILD.log pinned/*/FAILED.json \
   pinned/rule_checks pinned/bounds pinned/PINS.md pinned/program_digests.json pinned/REFUSED.json pinned/tests_a1_population.log \
   legacy/*/instance/protocol_instance.json legacy/*/instance/expected_population.json legacy/*/IDENTITY.json legacy/*/BUILD.log legacy/*/SHA256SUMS legacy/*/FAILED.json \
   rows/*/ROWS_IDENTITY.json rows/*/ROWS.log rows/*/produced/boundary_compare.json chain.log TREE_HASH GEOMETRY.json 2>/dev/null' \
  | tar -xz -C "$DST"
du -sh "$DST"; ls "$DST/pinned"
