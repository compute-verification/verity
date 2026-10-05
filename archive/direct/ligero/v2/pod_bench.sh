set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=.:benchmarks/numerical/python:.
PY=/workspace/venv312/bin/python   # + cupy-cuda12x, blake3 (uv pip install, 2026-09-22: GPU Merkle; without them the host-hash fallback costs ~4 s/proof)
BI=/workspace/bi/v1
for L in ${LS:-16384 32768}; do
  $PY -m archive.direct.ligero.v2.run bench --root $BI --device cuda --l $L --total-vus 4096 --all-subbatches --out "$RESEARCH_RUN_DIR/bench_l${L}.json"
done
