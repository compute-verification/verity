#!/usr/bin/env bash
# β(C) (beta.py) on the exact-int8 Llama-3.1-8B request Program (job.sh's Build, digest 8c683b2a..., LP=256, T=31), with the
# input term shared over the 8B row's KV pool (share.py's M_dyn for 8 engines). ART=art:... reads another Build's tree,
# M_NODE=BYTES another pool, CHUNKS="N ..." the coarse bounds before the fine one.
#
#   research run --queue --cpus 4 --mem-gb 48 --max-min 300 --source . --project verity \
#     --cwd clone/integrations/vllm -- bash ../../benchmarks/stacked_share/beta_job.sh
set -euo pipefail
export PY=/workspace/jobs/venv312/bin/python3 PYTHONPATH=.:../..:../../tools/research/src HF_HUB_OFFLINE=1
HERE=../../benchmarks/stacked_share
ART=${ART:-art:c2086c4c7da0041dcbf0776357bce6c3d7127bef6d6245c8652a772024af240f}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
python3 -m research data fetch "$ART" --to "$d/build" --path descriptor.json.gz > "$OUT/fetch.log" 2>&1
desc=$(find "$d/build" -name descriptor.json.gz | head -1)
[ -n "$desc" ] || { cat "$OUT/fetch.log"; exit 3; }
chunks=()
for n in ${CHUNKS:-4096 256}; do chunks+=(--chunk "$n"); done
/usr/bin/time -v "$PY" "$HERE/beta.py" "$desc" --out "$OUT/beta-8b.json" --m-dyn-node "${M_NODE:-609801469952}" "${chunks[@]}" \
  2> >(tee "$OUT/beta.log" >&2)
