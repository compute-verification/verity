#!/usr/bin/env bash
# A 70B arm from a target map whose values are a clip rate, absmax, deflR (`quant.QLinear`'s deflated matrix, at 2%) or
# `out`: an `out` matrix is uncredited, served exact (absmax codes, bf16 activations, no noise); the rest are at their
# rate (unlisted ones at 2%) under design U's noise.
#   bash benchmarks/hardness_defence/arm70b-out.sh NAME TARGETS.json [QAT_ARGS...]
set -euo pipefail
name=$1 targets=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${OUT:?}"
map=$OUT/$name.targets.json
exact=$(python3 - "$targets" "$map" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
json.dump({k: "absmax" if v == "out" else v for k, v in t.items()}, open(sys.argv[2], "w"), indent=1)
print("+".join(k.removeprefix("model.").removesuffix(".weight") for k, v in t.items() if v == "out"))
EOF
)
echo "$name: $(grep -c . "$map") lines in the map, exact: ${exact:0:300}"
bash "$here/arm70b.sh" "$name" --target 0.02 --rotate --targets "$map" --serve "e2${exact:+:$exact}" "$@"
