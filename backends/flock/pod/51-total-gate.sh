#!/usr/bin/env bash
# flock-backend: the total unit's GPU gate on the prover pod before any total cell: flock-pure-gpu selftest (CPU and --gpu) on
# instances.write_probe files (finite, +-inf, NaN and mixed VUs) at K = 1536 and 2048, and the NaN / inf negatives with the
# prover on the GPU.   REL=bf16-ampere-total SM=89 VUS="8 64"
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-total-gate}/out; W=/workspace/flock-backend; mkdir -p $O $W
REL=${REL:-bf16-ampere-total}; VUS=${VUS:-8 64}
MODE=build GPU=1 bash backends/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python
B=/workspace/flock-gpu-link/flock/target/release/flock-pure-gpu; sha256sum $B | tee $O/binary.sha256
python3 -c "from verity_flock import lowering as L; open('$O/net.txt','w').write(L.netlist('$REL')); assert L.digest('$REL') == L.PINS['$REL']"
for k in 1536 2048; do
  for n in $VUS; do
    python3 -c "from verity_flock import instances as I; I.write_probe('$O/probe-$k-$n.bin', '$REL', $n, $k)"
    for g in "" "--gpu"; do
      $B selftest $g --instances $O/probe-$k-$n.bin --netlist $O/net.txt > $O/selftest-$k-$n${g:+-gpu}.txt 2>&1
      echo "GATE	k$k n$n ${g:-cpu}	pass=$(grep -c '"pass":true' $O/selftest-$k-$n${g:+-gpu}.txt)	fail=$(grep -c '"pass":false' $O/selftest-$k-$n${g:+-gpu}.txt)	$(grep -h '^SELFTEST' $O/selftest-$k-$n${g:+-gpu}.txt | tail -1)" | tee -a $O/gate.tsv
    done
  done
done
python3 -m verity_flock.negatives --bin $B --relation $REL --vus 64 --out $O/negatives --prove-arg=--gpu 2>&1 | tee $O/negatives.txt
echo "GATE_SUMMARY	$(awk -F'\t' '{f+=substr($4,6)} END {print NR" selftest runs, "f" failing cases"}' $O/gate.tsv); $(tail -1 $O/negatives.txt)" | tee -a $O/gate.tsv
