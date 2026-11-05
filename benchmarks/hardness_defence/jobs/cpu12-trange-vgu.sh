#!/usr/bin/env bash
# HT-GENEROUS: trange.py on the 70B v, gate and up whose exact split, priced at two digits instead of three, would fall
# under the line at the recommended point's credits (L2 and L3 v, L1 and L3 gate, L3 up), at the ranks it is priced at
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
source "$here/../hardness_public/env.sh"
mkdir -p "${OUT:?}"
T="$here/trange.py"
R=1,2,4,6,8,10,12,14,16,32,48,64,96,128,192,256
pids=()
go() { name=$1; shift; python -u "$T" "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1 & pids+=($!); }
go 70b-v --hf unsloth/Meta-Llama-3.1-70B --layers 2,3 --match v_proj --ranks $R --workers 2
go 70b-gate --hf unsloth/Meta-Llama-3.1-70B --layers 1,3 --match gate_proj --ranks $R --workers 2
go 70b-up --hf unsloth/Meta-Llama-3.1-70B --layers 3 --match up_proj --ranks $R --workers 1
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
