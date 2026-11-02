#!/usr/bin/env bash
# flock-backend's preflight check: a short integration check of verity/ml/flock on one GPU pod. Builds flock-pure-gpu with
# the CUDA prover (SM), then:
#   CPU selftests at SELF (8 64): flock-pure-gpu on bf16-hopper, fp8-ada, fp8-hopper, bf16-ampere with blake3 and sha256 row
#   leaves;
#   GPU selftests at SELF: flock-pure-gpu on GPU_REL with blake3 and sha256 leaves.
# env: SM=89 SELF="8 64" GPU_REL=fp8-ada RELS="bf16-hopper fp8-ada fp8-hopper bf16-ampere"
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-preflight}/out; W=/workspace/flock-backend; mkdir -p $O $W
SM=${SM:-89}; SELF=${SELF:-8 64}; GPU_REL=${GPU_REL:-fp8-ada}; RELS=${RELS:-bf16-hopper fp8-ada fp8-hopper bf16-ampere}
MODE=build GPU=1 SM=$SM bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
F=/workspace/flock-gpu-link/flock
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER_=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); export RAYON_NUM_THREADS=$TH
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python
BI=$W/bench-instances/v1; FZ=$REPO/fixtures/bench-instances/v1
[ -f $BI/vu-k1536.x.u16 ] || python3 -m verity_numerical.bench.instances build --out $BI --seeds $FZ/seeds --procs $TH 2>&1 | tail -3
python3 - $BI $FZ <<'PYEOF' | tee $O/bench-instances.txt || exit 4
import json, sys
from pathlib import Path
from verity_numerical.bench.instances import sha256_file
bi, fz = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((fz / "manifest.json").read_text())
built = {n: f for t in man["tiers"].values() for n, f in t.get("files", {}).items() if not f.get("committed")}
bad = [n for n, f in built.items() if not (bi / n).is_file() or sha256_file(bi / n) != f["sha256"]]
assert not bad, f"rebuilt arrays differ from the committed manifest: {bad}"
for n in built:
    (fz / n).unlink(missing_ok=True)
    (fz / n).symlink_to(bi / n)
print(f"{len(built)} rebuilt arrays match the committed manifest")
PYEOF
P=$F/target/release/flock-pure-gpu
sha256sum $P | tee $O/binary.sha256; nvidia-smi -L | tee $O/gpu.txt
preflight() {  # name bin rel leaf gpu
  local name=$1 bin=$2 rel=$3 leaf=$4 g=$5
  for n in $SELF; do
    f=$W/inst-$rel-$leaf-$n.bin
    [ -f $f ] || python3 -m verity_flock.instances $rel $n $f --procs $TH --leaf $leaf > /dev/null
    python3 -c "from verity_flock import lowering; open('$W/net-$rel.txt','w').write(lowering.netlist('$rel'))"
    $bin selftest ${g:+--gpu} --instances $f --netlist $W/net-$rel.txt > $O/selftest-$name-$n.txt 2>&1
    s=$(grep -h '^SELFTEST' $O/selftest-$name-$n.txt | tail -1)
    np=$(grep -c '"pass":true' $O/selftest-$name-$n.txt); nf=$(grep -c '"pass":false' $O/selftest-$name-$n.txt)
    echo "PREFLIGHT	$name	n=$n	pass=$np	fail=$nf	${s:0:160}" | tee -a $O/preflight.tsv
  done
}
for rel in $RELS; do for leaf in blake3 sha256; do preflight cpu-$rel-$leaf $P $rel $leaf ""; done; done
preflight gpu-$GPU_REL-blake3 $P $GPU_REL blake3 1
preflight gpu-$GPU_REL-sha256 $P $GPU_REL sha256 1
echo "PREFLIGHT_SUMMARY	$(awk -F'\t' '{f+=substr($5,6)} END {print NR" selftest runs, "f" failing cases"}' $O/preflight.tsv)" | tee -a $O/preflight.tsv
true
