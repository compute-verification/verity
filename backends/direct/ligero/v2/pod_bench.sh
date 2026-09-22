set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=packages/verity/src:backends/numerical/python:.
PY=/workspace/venv312/bin/python
BI=/workspace/bi/v1
L=${L:-16384}
$PY -m backends.direct.ligero.v2.run bench --root $BI --device cuda --l $L --total-vus 4096 --all-subbatches --out "$RESEARCH_RUN_DIR/bench_l${L}.json"
