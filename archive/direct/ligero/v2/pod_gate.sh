set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=verity:benchmarks/numerical/python:.
PY=/workspace/venv312/bin/python   # + cupy-cuda12x, blake3 (uv pip install, 2026-09-22: GPU Merkle; without them the host-hash fallback costs ~4 s/proof)
BI=/workspace/bi/v1
if [ ! -f $BI/vu-k1536.x.u16 ]; then mkdir -p /workspace/bi; $PY -m verity_numerical.bench.instances build --out $BI --seeds fixtures/bench-instances/v1/seeds --procs $(nproc); fi
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m archive.direct.ligero.v2.run shape --l 16384 | head -12
$PY -m archive.direct.ligero.v2.run gate --root $BI --device cuda --vus 8 --l 128 --out "$RESEARCH_RUN_DIR/gate.json"
