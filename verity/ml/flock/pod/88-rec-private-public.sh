#!/usr/bin/env bash
# rec-private's public views (lane rec-v0): what the auditor holds of two netlists of one class, 87-rec-private.sh's registered
# netlist and its other one (one unread no-op gate changed). V*'s statements of the other netlist's inner session at every level
# (rec_vstage stage --no-algebra, under the salts 87 staged its algebra with), then for each of V*'s statements (every level and
# algebra part) its circuit's SHA-512 and where its public file differs from the registered netlist's (rec_inner.py differ),
# and the same for the inner statement's public files. Into out/public/ (public.json).
# Run after 87 on its $FLOCK_WORK/data/rec-private/$TAG (research run --cwd source; 87 runs it last).
# env: TAG=16-16-8-n64  CPUS=32  FLOCK_WORK (as 87)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
O=${RESEARCH_RUN_DIR:-/tmp/rec-v0-job}/out; P=$O/public; mkdir -p $P || exit 1
W=$FLOCK_WORK/flock-circuit
export PATH=$W/py/bin:$PATH PYTHONPATH=$(pwd):$(pwd)/catalog:$(pwd)/kernels:$(pwd)/experimental:$(pwd)/benchmarks/numerical/python:$(pwd)/verity/ml/flock/python
DD=$FLOCK_WORK/data/rec-private/${TAG:-16-16-8-n64}
[ -f $DD/cap.json ] && [ -d $DD/v ] && [ -d $DD/vo ] || { echo "no 87 staging under $DD"; exit 1; }
CAP=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cap'])" $DD/cap.json)
PY=verity/ml/flock/pod/rec_inner.py

if [ ! -d $DD/vo/L0 ]; then
  /usr/bin/time -v -o $P/time-other-levels.txt python3 -m verity_flock.rec_vstage stage $DD/inner-other $DD/fold-other.json $DD/vo \
    --points $DD/points-other.json --salts $DD/vo-acc-salts.bin --cap $CAP --no-algebra > $P/other-levels.json 2> $P/other-levels.err
  echo "vstage other levels exit $?"
fi
python3 $PY differ $(ls $DD/u/stage/pub-*.bin | head -1) $(ls $DD/u/other/pub-*.bin | head -1) > $P/inner.json
for C in $(cd $DD/v && ls -d L* alg/p* | sort -V); do
  [ -f $DD/vo/$C/pub.bin ] || continue
  n=${C//\//-}
  python3 $PY differ $DD/v/$C/pub.bin $DD/vo/$C/pub.bin > $P/$n.json
  sha512sum $DD/v/$C/circuit.txt $DD/vo/$C/circuit.txt | cut -d' ' -f1 | uniq | wc -l > $P/$n.circuits
done
python3 - $P <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
out = {"inner": json.loads((p / "inner.json").read_text())}
for f in sorted(p.glob("*.circuits")):
    out[f.stem] = {**json.loads((p / f"{f.stem}.json").read_text()), "same_circuit": f.read_text().strip() == "1"}
(p / "public.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v.get("same_circuit"), v["same_size"], v["header_differing"]) for k, v in out.items()}))
EOF
