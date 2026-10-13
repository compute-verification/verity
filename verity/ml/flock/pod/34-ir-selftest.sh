#!/usr/bin/env bash
# flock-ir-lowering: flock-ir-frame's selftest (honest sessions and every negative) with the device prover (--gpu) on a pod,
# on instance files staged here from the input set sent as inputs/<content digest>.tar.
#   TEMPLATE N (instances staged)                       (GPU build for the pod's own SM)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=${RESEARCH_RUN_DIR:-/tmp/flock-ir-selftest}/out; W=/workspace/flock-ir; mkdir -p $O $W
TEMPLATE=${TEMPLATE:?}; N=${N:-2}
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-90}
MODE=build GPU=1 SM=$SM bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
# the cgroup's CPU quota (v2 cpu.max, else v1 cfs_quota_us / cfs_period_us, as timing_guard.cpu_quota), rounded up: more
# threads than the quota only get throttled
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER_=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); }
TH=$( { [ "$Q" = max ] || [ "$Q" = -1 ]; } && nproc || echo $(( (Q + PER_ - 1) / PER_ )) ); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock
(cd $F && cargo build --release -p flock-live --features gpu --bin flock-ir-frame -j $TH) > $O/build-ir.txt 2>&1 || { tail -40 $O/build-ir.txt; exit 1; }
B=$F/target/release/flock-ir-frame; sha256sum $B | tee $O/binary.sha256
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
rm -rf $W/st-set && mkdir -p $W/st-set && tar -xf $(ls $I/*.tar | head -1) -C $W/st-set || exit 1
SET=$(dirname $(find $W/st-set -name manifest.json | head -1))
nvidia-smi -L | tee $O/host.txt
python3 - "$SET" "$O/stage" "$N" "$TEMPLATE" <<'EOF'
import sys
from verity_numerical.bench import lowerings
from verity_numerical.bench.input_sets import InputSet
from verity_flock import ir_lower
s = InputSet.open(sys.argv[1])
ir_lower.write_tables(sys.argv[2] + "/tables")
print(lowerings.registry("C-Flock").module(sys.argv[4]).stage(s, 0, int(sys.argv[3]), sys.argv[2]))
EOF
$B selftest --gpu --instances $(ls $O/stage/frame-*.bin) --netlist $O/stage/net.txt --tables $O/stage/tables > $O/selftest.txt 2> $O/selftest.err
grep SELFTEST $O/selftest.txt
grep -c '"pass":true' $O/selftest.txt; grep '"pass":false' $O/selftest.txt | cut -c1-300
grep -q '"all_pass":true' $O/selftest.txt
