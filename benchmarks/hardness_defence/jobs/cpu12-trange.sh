#!/usr/bin/env bash
# HT-GENEROUS: the range T = (E − 2)·B of the exact split reaches for drawn E, against the worst case its digits are
# priced for (trange.py), on every credited 70B q and k and every 8B q and k, at the ranks their exact split is priced at
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
source "$here/../hardness_public/env.sh"
mkdir -p "${OUT:?}"
T="$here/trange.py"
pids=()
go() { name=$1; shift; python -u "$T" "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1 & pids+=($!); }
go 70b-q --hf unsloth/Meta-Llama-3.1-70B --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 --match q_proj --ranks 1,8,16,32,48,64,96,112,128,160,192,256,384,512 --workers 40
go 70b-k --hf unsloth/Meta-Llama-3.1-70B --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79 --match k_proj --ranks 1,8,16,24,32,40,48,52,56,60,64,68,72,76,80,96,112,128,160,192,256 --workers 79
go 8b-q --hf unsloth/Meta-Llama-3.1-8B --match q_proj --ranks 1,8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,512 --workers 32
go 8b-k --hf unsloth/Meta-Llama-3.1-8B --match k_proj --ranks 1,8,16,24,32,40,48,52,56,60,64,68,72,76,80,96,112,128,160,192,256 --workers 32
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
