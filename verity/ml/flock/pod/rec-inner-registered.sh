#!/usr/bin/env bash
# Route 1's honest replay (scratch: cursor/rec-inner-registered-replay-2f12, rec-restage merged with
# cursor/rec-inner-registered-95d4). The universal unit's inner statement at CLS (rec_inner.py universal), its description
# registered and named in its header (verity_flock.inner_registered register; other claims the same rows), then
# 87-rec-private.sh end to end on that staging (STAGED=), whose folds read the description from the registration
# (--registered). Then the crafted refusals, each inner-fold at the honest session's points: other (a description that does
# not open the registered root), extra (a registration of one row more than the table), plain (the staging before
# registration: no registered rows) and unheld (no --registered); and test_inner_registered.py's crafted statements (order,
# ref, coverage and the rest) on this tree's flock-verify. Run with research run --cwd clone. env: CLS=16,16,8 N=64 NW=8
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
CLS=${CLS:-16,16,8}; N=${N:-64}; NW=${NW:-8}
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
FW=$FLOCK_WORK; L=$FW/cache/rec-v0; W=$FW/flock-circuit; REPO=$(pwd)
O=${RESEARCH_RUN_DIR:-/tmp/rir-job}/out; mkdir -p $O/crafted
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
D=$FW/data/rec-inner-registered/${RESEARCH_RUN_ID:-local-$$}; rm -rf $D; mkdir -p $D || exit 1
command -v research >/dev/null && research keep $D --owner @proofs --expires 2026-10-23 --priority low --why "route 1 replay $CLS" > /dev/null 2>&1
S=$(mktemp -d); trap 'rm -rf $S' EXIT
UV=$(command -v uv || echo $HOME/.local/bin/uv); mkdir -p $W || exit 1
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$HOME/.elan/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm

# stage, keep the unregistered statement (plain) and a copy for a registration past its table (extra), register, claim
/usr/bin/time -v -o $O/time-stage.txt python3 verity/ml/flock/pod/rec_inner.py universal --cls $CLS --n $N --seed 0 --out $D \
  > $O/stage-universal.json 2> $O/stage-universal.err || { tail -20 $O/stage-universal.err; exit 1; }
for c in plain extra; do
  mkdir -p $D/$c && cp $D/stage/pub-$N.bin $D/stage/inst-$N.bin $D/stage/partition.json $D/$c/ && ln -s $D/stage/circuit.txt $D/$c/ || exit 1
done
/usr/bin/time -v -o $O/time-register.txt python3 -m verity_flock.inner_registered register $D/stage > $O/register.json || exit 1
python3 -m verity_flock.inner_registered claim $D/other --from $D/stage > $O/claim-other.json || exit 1
python3 -m verity_flock.inner_registered register $D/extra --extra 1 > $O/register-extra.json || exit 1
cp $D/stage/registered.json $O/
python3 - $D $N > $O/header-bytes.json <<'EOF' || exit 1
import json, sys
from pathlib import Path
from verity_flock import inner_registered as IR
d, n = Path(sys.argv[1]), sys.argv[2]
out = {}
for f in (f"pub-{n}.bin", f"inst-{n}.bin"):
    (hs, _), (hp, _) = IR._split(d / "stage" / f), IR._split(d / "plain" / f)
    out[f] = {"registered": (d / "stage" / f).stat().st_size, "plain": (d / "plain" / f).stat().st_size,
              "header_registered": hs.get("registered"), "plain_has_registered": "registered" in hp}
print(json.dumps(out, sort_keys=True))
EOF

# the honest run, end to end, on the registered staging
TAG=reg-${CLS//,/-}-n$N
bash verity/ml/flock/pod/87-rec-private.sh STAGED=$D CLS=$CLS N=$N NW=$NW TAG=$TAG > $O/87.log 2>&1
echo $? > $O/87.exit

# the crafted refusals, at the honest session's points, by the run's own inner-fold
TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
  -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
IF=$L/tools-$TSHA/inner-fold; PTS=$FW/data/rec-private/$TAG/points-honest.json
craft() {
  local name=$1 dir=$2; shift 2
  $IF --plain-leaves --partition $dir/partition.json "$@" verity/flock-circuit $dir/circuit.txt $dir/pub-$N.bin $PTS os \
    > $S/$name.out 2> $O/crafted/$name.err
  echo $? > $O/crafted/$name.exit
}
if [ -x $IF ] && [ -s $PTS ]; then
  craft honest $D/stage --registered $D/stage/registered.json
  craft other $D/other --registered $D/stage/registered.json
  craft extra $D/extra --registered $D/extra/registered.json
  craft plain $D/plain --registered $D/stage/registered.json
  craft unheld $D/stage
  python3 - $O/crafted > $O/crafted.json <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
print(json.dumps({p.stem: {"exit": int(p.read_text()), "err": (d / f"{p.stem}.err").read_text().strip()[-600:]}
                  for p in sorted(d.glob("*.exit"))}, sort_keys=True, indent=1))
EOF
else
  echo "no inner-fold ($IF) or no honest points ($PTS)" > $O/crafted.json
fi

# test_inner_registered.py's crafted statements on this tree's flock-verify (87's build of the same sources, copied in)
( exec 9>$L/lean.lock; flock 9; [ -d $L/lean-src/.lake ] && cp -r $L/lean-src/.lake verity/core/service/flock/ )
$UV venv -q --python 3.12 $S/py && $UV pip install -q --python $S/py/bin/python numpy blake3 pytest || exit 1
$S/py/bin/python -m pytest -q -m slow -p no:cacheprovider verity/ml/flock/tests/test_inner_registered.py > $O/test_inner_registered.txt 2>&1
echo $? > $O/test_inner_registered.exit
exit $(cat $O/87.exit)
