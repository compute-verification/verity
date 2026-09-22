set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=packages/verity/src:backends/numerical/python:.
PY=python3   # system python3: torch 2.4.1+cu124, cupy 14.2, blake3 (venv312 lacks cupy/blake3 -> host-hash fallback)
BI=/workspace/bi/v1
for L in ${LS:-16384 32768}; do
  $PY -m backends.direct.ligero.v2.run bench --root $BI --device cuda --l $L --total-vus 4096 --all-subbatches --out "$RESEARCH_RUN_DIR/bench_l${L}.json"
done
