#!/usr/bin/env bash
# Non-producer re-verification of A-fs results (A-GKR alone, Fiat-Shamir coins, x and W private): verity-gkr-verify built from
# this tree, the statement rebuilt here from the captured set (gpu.v2.export circuits at the set's K + bench_result.write_statement
# of the set's y words), compared with the producer's statement files, then every recorded proof verified against it with the
# relation pinned, plus negatives.  CPU only.
#   per TAG: inputs/set-TAG.tgz  inputs/afs-TAG.tgz (proofs/rep*.bin)  inputs/afs-TAG.sha256 (the producer's statement files)
#   AFS="TAG ..."
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:?}; W=/workspace/verify-afs; O=$RD/out/afs; mkdir -p $O $W
Q=-1; PER=100000
if [ -r /sys/fs/cgroup/cpu.max ]; then read -r Q PER < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1; fi
NT=$(( Q > 0 ? Q / PER : $(nproc) )); [ "$NT" -lt 1 ] && NT=1
export PYTHONPATH="$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/tools/research/src:$REPO:$REPO/backends/gkr"
command -v cc >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential pkg-config >/dev/null; }
[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env
VB=$W/verity-gkr-verify
( cd backends/gkr/verifier && cargo build --release -j $NT 2>&1 | grep -E '^error|Finished' | head -3 && cp target/release/verity-gkr-verify $VB )
[ -x $VB ] || { echo "verity-gkr-verify build failed"; exit 1; }
sha256sum $VB | tee $O/verifier.sha256
PY=$W/venv/bin/python
if ! $PY -c "import numpy, blake3" 2>/dev/null; then
  command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"; uv venv $W/venv --python 3.12 >/dev/null 2>&1; uv pip install --python $PY numpy blake3 2>&1 | tail -1
fi
bad=0
for TAG in $AFS; do
  X=$W/$TAG; rm -rf $X; mkdir -p $X/set $X/rec; C=$O/$TAG; mkdir -p $C; : > $C/verify.jsonl; : > $C/negatives.jsonl
  tar -xzf $RD/inputs/set-$TAG.tgz -C $X/set; tar -xzf $RD/inputs/afs-$TAG.tgz -C $X/rec
  SET=$(dirname "$(find $X/set -name manifest.json | head -1)")
  read -r K N < <($PY -c "import json,sys;m=json.load(open(sys.argv[1]+'/manifest.json'));print(m['relation']['statics']['K'], m['n'])" "$SET")
  REL=bf16-ampere-k$K
  ( cd backends/gkr && $PY -m gpu.v2.export circuits --model ampere_bf16_m16n8k16 --k $K --out $X/export > /dev/null ) || { echo "$TAG export failed"; bad=$((bad+1)); continue; }
  ( cd backends/gkr && $PY - "$X/export" "$X/statement" "$SET" "$N" <<'EOF'
import sys
from pathlib import Path
import numpy as np
import bench_result as br
from gpu import commit as CM
d, out, s, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
x, w, y, _iset, _ref = CM.input_set(s, n)
br.write_statement(d, out, y.astype(np.uint16))
print(f"statement {out}: {n} VUs, K {x.shape[1]}")
EOF
  ) || { bad=$((bad+1)); continue; }
  ( cd $X/statement && for f in *; do echo "$(sha256sum $f | cut -c1-64)  statement/$f"; done ) | sort -k2 > $C/mine.sha256
  sort -k2 $RD/inputs/afs-$TAG.sha256 > $C/producer.sha256
  if cmp -s $C/mine.sha256 $C/producer.sha256; then echo "$TAG K=$K N=$N statement: $(wc -l < $C/mine.sha256) files equal to the producer's" | tee -a $C/summary.txt
  else echo "$TAG statement DIFFERS: $(diff $C/mine.sha256 $C/producer.sha256 | tr '\n' ' ')" | tee -a $C/summary.txt; bad=$((bad+1)); fi
  v() { $VB verify --dir $1 --proof $2 --vus $3 --threads $NT --relation $4 --json $5 > /dev/null 2>&1; echo $?; }
  for p in $(ls $X/rec/proofs/rep*.bin | sort); do
    j=$C/$(basename $p .bin).json; rc=$(v $X/statement $p $N $REL $j)
    $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps({'proof':sys.argv[2],'rc':int(sys.argv[3]),'accepted':d.get('accepted') is True and d.get('circuit_pinned') is True and int(sys.argv[3])==0,'relation':d.get('relation'),'vus':d.get('vus'),'prime_coins':d.get('prime_coins'),'error':d.get('error'),'verify_seconds':d.get('verify_seconds')}))" $j $(basename $p) $rc | tee -a $C/verify.jsonl
  done
  ok=$(grep -c '"accepted": true' $C/verify.jsonl); tot=$(wc -l < $C/verify.jsonl)
  echo "$TAG proofs: $ok of $tot accepted (relation $REL pinned)" | tee -a $C/summary.txt; [ $ok = $tot ] && [ $tot -gt 0 ] || bad=$((bad+1))
  # negatives on rep0: each must be refused
  P0=$X/rec/proofs/rep0.bin; T=$X/neg; mkdir -p $T
  cp $P0 $T/flipped.bin; $PY -c "import sys;p=sys.argv[1];b=bytearray(open(p,'rb').read());b[len(b)//2]^=1;open(p,'wb').write(b)" $T/flipped.bin
  cp -r $X/statement $T/pub; $PY -c "import sys;p=sys.argv[1]+'/public.bin';b=bytearray(open(p,'rb').read());b[-1]^=1;open(p,'wb').write(b)" $T/pub
  neg() { local name=$1 dir=$2 proof=$3 rel=$4; rc=$(v $dir $proof $N $rel $T/$name.json)
    acc=$($PY -c "import json,sys;d=json.load(open(sys.argv[1])) if __import__('os').path.exists(sys.argv[1]) else {};print(d.get('accepted') is True and int(sys.argv[2])==0)" $T/$name.json $rc)
    echo "{\"neg\": \"$name\", \"rejected\": $([ $acc = True ] && echo false || echo true), \"rc\": $rc, \"error\": $($PY -c "import json,sys,os;print(json.dumps((json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}).get('error')))" $T/$name.json)}" | tee -a $C/negatives.jsonl; }
  neg proof_byte_flipped $X/statement $T/flipped.bin $REL
  neg public_word_changed $T/pub $P0 $REL
  neg claimed_as_k1536_relation $X/statement $P0 bf16-ampere
  for other in $AFS; do [ $other != $TAG ] && [ -d $W/$other/statement ] && neg other_sets_statement $W/$other/statement $P0 $REL; done
  nr=$(grep -c '"rejected": true' $C/negatives.jsonl); nt=$(wc -l < $C/negatives.jsonl)
  echo "$TAG negatives: $nr of $nt rejected" | tee -a $C/summary.txt; [ $nr = $nt ] || bad=$((bad+1))
  rm -rf $X/rec $T
done
echo "AFS TOTAL bad $bad" | tee $O/summary.txt
[ $bad = 0 ]
