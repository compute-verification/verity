set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=packages/verity/src:backends/numerical/python:.
PY=python3   # system python3: torch 2.4.1+cu124, cupy 14.2, blake3 (venv312 lacks cupy/blake3 -> host-hash fallback)
BI=/workspace/bi/v1
if [ ! -f $BI/vu-k1536.x.u16 ]; then mkdir -p /workspace/bi; $PY -m verity_numerical.bench.instances build --out $BI --seeds fixtures/bench-instances/v1/seeds --procs $(nproc); fi
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m backends.direct.ligero.v2.run shape --l 16384 | head -12
$PY -m backends.direct.ligero.v2.run gate --root $BI --device cuda --vus 8 --l 128 --out "$RESEARCH_RUN_DIR/gate.json"
