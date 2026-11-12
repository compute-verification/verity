#!/usr/bin/env bash
# two_stage_small.sh on the greedy select in 20 units (V = 2560, four outputs a span), the second stage sampling every
# proof block of each template in a sampled span, but for one template: 19 of the 20 member statements then hold only
# sampled proof blocks, each row a registered read of the interior record, and each one's session samples every unit
# of its statement. One template stays at one proof block of its four a span: the second stage refuses counts that
# prove every cell whole. Exit and records as two_stage_small.sh's; the counts it ran at are $OUT/counts.json.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 24 --max-min 120 -- bash benchmarks/one_stage/two_stage_small_whole.sh
set -euo pipefail
O=${OUT:?research run sets OUT}; mkdir -p "$O"
V=2560 UNITS=20 WHOLE=4
python3 - "$V" "$UNITS" "$WHOLE" > "$O/counts.json" <<'EOF'
import json
import sys

v, units, whole = (int(x) for x in sys.argv[1:])
n = v // units
k = {"GreedyChunk_v2{N=%d,OFF=%d}" % (n, off): whole for off in range(0, v - n, n)}
k["GreedyChunk_v2{N=%d,OFF=0}" % n] = 1
k["GreedyCombine_v2{V=%d,N=%d}" % (v, n)] = whole
print(json.dumps(k, indent=1))
EOF
SMALL_ARGS="--inline --units $UNITS --v $V" SMALL_FIRST="--k $O/counts.json" \
  exec bash "$(dirname "$0")/two_stage_small.sh"
