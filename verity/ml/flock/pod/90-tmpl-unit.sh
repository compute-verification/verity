#!/usr/bin/env bash
# A templated statement end to end (PRF-133): RoPE d64 staged as verity/flock-circuit/tmpl (tests/tmpl_statement.py), its
# unit read at layout LAYOUT by Lean's `flock-tmpl-unit` (tests/lean/TmplUnit.lean: the rows at Tmpl.posMap, an ORDER
# line), proved by the CPU flock-circuit (--tmpl-unit) in one loopback session per mode, then Lean's `flock-verify verify` on each. The verifier
# refuses the statement once every check has passed, so each verdict's reason must be exactly that refusal. Run from the
# verity tree (research run --cwd source); CPU only. Outputs under out/: host.txt, template.json, tmpl.txt, stage.json,
# statement-<mode>.txt, serve-<mode>.log, prove-<mode>.txt and .err, lean-<mode>.log (its VERDICT line), verdicts.json.
# env: CPUS=16  LAYOUT=iv  MODES="m0 zk"  FLOCK_WORK=/workspace, or /workspace/jobs/pc-read-gadget where /workspace isn't
#      writable (the prover build and its venv, shared as 76-read-gadget.sh's)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${TU_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export TU_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=${FC_COINS:-os}
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/tmpl-unit}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/pc-read-gadget
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/tmpl-unit; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$TU_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-}"; } > $O/host.txt 2>&1

# Lean's flock-verify, and beside it the prover's flock-tmpl-unit (tests/lean/TmplUnit.lean, which unrolls the unit and so
# is no part of the verifier of record), built once per source, while the prover builds
LSHA=$( (cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum;
  cat verity/ml/flock/tests/lean/TmplUnit.lean) | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify; UNIT_BIN=$L/lean-$LSHA/flock-tmpl-unit
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ] || [ ! -x $UNIT_BIN ]; then
    rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
      && cp verity/ml/flock/tests/lean/TmplUnit.lean $L/lean-src/ \
      && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> $L/lean-src/lakefile.toml \
      && (cd $L/lean-src && lake build flock-verify flock-tmpl-unit) && mkdir -p $L/lean-$LSHA \
      && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN && cp $L/lean-src/.lake/build/bin/flock-tmpl-unit $UNIT_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

# the CPU prover, cached per pod under its sources' key (76-read-gadget.sh's)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm

D=$S/stage; N=8; LAYOUT=${LAYOUT:-iv}
python3 verity/ml/flock/tests/tmpl_statement.py template --out $D > $O/template.json 2> $O/template.err || { tail -20 $O/template.err; exit 1; }
wait $LP; [ -x $LEAN_BIN ] && [ -x $UNIT_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
$UNIT_BIN $D/unit.tmpl --layout $LAYOUT --out $D/unit.placed > $O/tmpl.txt 2>&1 || { cat $O/tmpl.txt; exit 1; }
grep -h '^TMPL ' $O/tmpl.txt | cut -c6- > $D/tmpl.json
python3 verity/ml/flock/tests/tmpl_statement.py stage --out $D --tmpl $D/tmpl.json --layout $LAYOUT > $O/stage.json 2> $O/stage.err \
  || { tail -20 $O/stage.err; exit 1; }
mkdir -p $D/tables
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
WANT="refused: verity/flock-circuit/tmpl, a templated statement: a statement form the soundness proof does not cover"
echo '{}' > $O/verdicts.json
for mode in ${MODES:-m0 zk}; do
  Z=; [ $mode = zk ] && Z=--zk
  C=(--coins os); [ $mode = zk ] && C=()
  U=(--circuit $D/circuit.txt --tmpl-unit $D/unit.placed --tables $D/tables)
  $B statement $Z "${U[@]}" --instances $D/pub-$N.bin > $O/statement-$mode.txt 2>&1
  PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  $B serve $Z --listen 127.0.0.1:$PORT --out $S/sessions-$mode --instances $D/pub-$N.bin "${U[@]}" --pin $PIN --operator tmpl-unit \
    --sessions 1 > $O/serve-$mode.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $O/serve-$mode.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  $B prove $Z --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin "${U[@]}" --runs 1 --warm 0 > $O/prove-$mode.txt 2> $O/prove-$mode.err
  for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill $SP 2>/dev/null; wait $SP
  SESS=$(dirname $(ls -1 $S/sessions-$mode/*/session.json | LC_ALL=C sort | tail -1)) || { tail -20 $O/prove-$mode.err; continue; }
  $LEAN_BIN verify $Z "${C[@]}" --statement verity/flock-circuit/tmpl --circuit $D/circuit.txt --public $D/pub-$N.bin --tables $D/tables \
    --session $SESS > $O/lean-$mode.log 2>&1
  echo "$mode rc $?"
  grep -h '^VERDICT' $O/lean-$mode.log | head -1 | cut -c1-600
  python3 - $O/verdicts.json $mode "$WANT" $O/lean-$mode.log $O/prove-$mode.txt <<'EOF'
import json, sys
out, mode, want, log, prove = sys.argv[1:]
v = [json.loads(l[8:]) for l in open(log) if l.startswith("VERDICT ")]
live = [json.loads(l.split("\t", 1)[1]) for l in open(prove) if l.startswith("LIVE")]
doc = json.load(open(out))
doc[mode] = {"verdict": v[0] if v else None, "pass": bool(v) and not v[0]["accepted"] and v[0]["why"] == want,
             "live_why": live[0].get("why") if live else None}
json.dump(doc, open(out, "w"), indent=1, sort_keys=True)
EOF
done
cat $O/verdicts.json
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d and all(v["pass"] for v in d.values()) else 1)' $O/verdicts.json
