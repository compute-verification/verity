#!/usr/bin/env bash
# A templated statement end to end: RoPE d64 staged as verity/flock-circuit/tmpl (tests/tmpl_statement.py), its
# unit read at layout LAYOUT by Lean's `flock-tmpl-unit` (tests/lean/TmplUnit.lean: the rows at Tmpl.posMap, an ORDER
# line), proved by the CPU flock-circuit (--tmpl-unit) in one loopback session per mode, then Lean's `flock-verify verify` on each. The verifier
# covers a templated statement only under --zk (`ProvedScope.scopeOkT`): the zk session, sampled (--draw subset:N, every
# unit) and verified with the statement's partition and program (tmpl_statement.py's partition.json and program.json),
# must be accepted, and the m0 one refused as a templated statement without --zk, by Lean and by the live server (its
# LIVE line) alike. The zkfw session is the zk one proved behind the ZK server (FC_FIREWALL=1, Lean's flock-firewall
# setting its Hello up at the templated statement with the partition and program), and must be accepted too. Each
# session's coin tree is sampled and opened by its own challenger (challenger.sh: serve --tree-from,
# the m0 session's under FC_COINS=tree), and its record held to that challenger's release log. Run from the
# verity tree (research run --cwd source); CPU only. Outputs under out/: host.txt, template.json, tmpl.txt, stage.json,
# statement-<mode>.txt, serve-<mode>.log, prove-<mode>.txt and .err, challenger-<mode>/ (its record, releases.jsonl and
# audit.json), lean-<mode>.log (its VERDICT line), verdicts.json.
# Pins its own slices where it can take slice locks (node 1); on a vy-cpu pod, run it in a check slot (--pool), which pins it.
# env: CPUS=16  LAYOUT=iv  MODES="m0 zk zkfw"  FLOCK_WORK=/workspace, or where /workspace isn't writable
#      /workspace/jobs/pc-read-gadget (node 1, shared as 76-read-gadget.sh's) else /workspace/cache/tmpl-unit (the prover
#      build and its venv)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${TU_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/protocols/flock/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-16}" || exit 1
    export TU_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  # no slice locks here (a vy-cpu pod): the check slot (research run --pool) has pinned the job to its own cores
  export TU_PINNED=slot RAYON_NUM_THREADS=$(nproc)
fi
export FC_COINS=tree
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/tmpl-unit}/out; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/tmpl-unit; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/tmpl-unit; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$TU_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-}"; } > $O/host.txt 2>&1

# Lean's flock-verify, and beside it the prover's flock-tmpl-unit (tests/lean/TmplUnit.lean, which unrolls the unit and so
# is no part of the verifier), built once per source, while the prover builds
LSHA=$(python3 verity/protocols/flock/pod/lake_closure.py key verity/protocols/flock/verifier $REPO/verity/protocols/flock/prover/tests/lean/TmplUnit.lean)
LEAN_BIN=$L/lean-$LSHA/flock-verify; UNIT_BIN=$L/lean-$LSHA/flock-tmpl-unit; FW_BIN=$L/lean-$LSHA/flock-firewall
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ] || [ ! -x $UNIT_BIN ] || [ ! -x $FW_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/flock/pod/lake_closure.py copy verity/protocols/flock/verifier $L/lean-src) \
      && cp verity/protocols/flock/prover/tests/lean/TmplUnit.lean $LEAN_SRC/ \
      && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> $LEAN_SRC/lakefile.toml \
      && (cd $LEAN_SRC && lake build flock-verify flock-tmpl-unit flock-firewall) && mkdir -p $L/lean-$LSHA \
      && cp $LEAN_SRC/.lake/build/bin/flock-verify $LEAN_BIN && cp $LEAN_SRC/.lake/build/bin/flock-tmpl-unit $UNIT_BIN \
      && cp $LEAN_SRC/.lake/build/bin/flock-firewall $FW_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!
CHALLENGER_REPO=$REPO CHALLENGER_WORK=$S/challenger-work
. verity/protocols/flock/prover/pod/challenger.sh
challenger_build

# the CPU prover, cached per pod under its sources' key (76-read-gadget.sh's)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/protocols/flock/pod/20-gpu-link.sh verity/protocols/flock/pod/60-circuit.sh verity/protocols/flock/prover/*.patch verity/protocols/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/flock/prover/cuda verity/protocols/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
CHALLENGER_PY=python3
challenger_built || exit 1

D=$S/stage; N=8; LAYOUT=${LAYOUT:-iv}
python3 verity/protocols/flock/prover/tests/tmpl_statement.py template --out $D > $O/template.json 2> $O/template.err || { tail -20 $O/template.err; exit 1; }
wait $LP; [ -x $LEAN_BIN ] && [ -x $UNIT_BIN ] && [ -x $FW_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
$UNIT_BIN $D/unit.tmpl --layout $LAYOUT --out $D/unit.placed > $O/tmpl.txt 2>&1 || { cat $O/tmpl.txt; exit 1; }
grep -h '^TMPL ' $O/tmpl.txt | cut -c6- > $D/tmpl.json
python3 verity/protocols/flock/prover/tests/tmpl_statement.py stage --out $D --tmpl $D/tmpl.json --layout $LAYOUT > $O/stage.json 2> $O/stage.err \
  || { tail -20 $O/stage.err; exit 1; }
mkdir -p $D/tables
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
echo '{}' > $O/verdicts.json
for mode in ${MODES:-m0 zk zkfw}; do
  Z=; [ $mode != m0 ] && Z=--zk
  C=(--coins tree); [ $mode != m0 ] && C=()
  DRAW=(); [ $mode != m0 ] && DRAW=(--draw subset:$N)
  PENV=(); [ $mode = zkfw ] && PENV=(FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_LEAN=$FW_BIN
    FC_FIREWALL_STATEMENT=verity/flock-circuit/tmpl "FC_FIREWALL_HELLO_ARGS=--partition $D/partition.json --program $D/program.json")
  U=(--circuit $D/circuit.txt --tmpl-unit $D/unit.placed --tables $D/tables)
  $B statement $Z "${U[@]}" --instances $D/pub-$N.bin > $O/statement-$mode.txt 2>&1
  challenger_start $O/challenger-$mode || exit 1
  PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  $B serve $Z --listen 127.0.0.1:$PORT --out $S/sessions-$mode --instances $D/pub-$N.bin "${U[@]}" --pin $PIN --operator tmpl-unit \
    "${DRAW[@]}" --sessions 1 "${TREE_FROM[@]}" > $O/serve-$mode.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $O/serve-$mode.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  env "${PENV[@]}" $B prove $Z --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin "${U[@]}" --runs 1 --warm 0 > $O/prove-$mode.txt 2> $O/prove-$mode.err
  for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill $SP 2>/dev/null; wait $SP
  challenger_end $O/challenger-$mode; echo "$mode challenger exit $?"
  SESS=$(dirname $(ls -1 $S/sessions-$mode/*/session.json | LC_ALL=C sort | tail -1)) || { tail -20 $O/prove-$mode.err; continue; }
  challenger_audit $O/challenger-$mode $SESS
  $LEAN_BIN verify $Z "${C[@]}" --statement verity/flock-circuit/tmpl --circuit $D/circuit.txt --public $D/pub-$N.bin --tables $D/tables \
    --partition $D/partition.json --program $D/program.json --session $SESS > $O/lean-$mode.log 2>&1
  echo "$mode rc $?"
  grep -h '^VERDICT' $O/lean-$mode.log | head -1 | cut -c1-600
  python3 - $O/verdicts.json $mode $O/lean-$mode.log $O/prove-$mode.txt $O/challenger-$mode/audit.json <<'EOF'
import json, sys
from pathlib import Path
out, mode, log, prove, audit = sys.argv[1:]
want = None if mode != "m0" else \
    "refused: a templated statement without --zk: a statement form the soundness proof does not cover"
v = [json.loads(l[8:]) for l in open(log) if l.startswith("VERDICT ")]
live = [json.loads(l.split("\t", 1)[1]) for l in open(prove) if l.startswith("LIVE")]
audit = json.loads(Path(audit).read_text()) if Path(audit).exists() else None
lean_ok = bool(v) and (v[0]["accepted"] is True if want is None else not v[0]["accepted"] and v[0]["why"] == want)
live_ok = bool(live) and (live[0].get("accepted") is True if want is None else live[0].get("why") == want)
# the accepted session's coins are its challenger's; the refused one's audit is kept, not judged
audit_ok = want is not None or bool(audit and audit["ok"])
doc = json.load(open(out))
doc[mode] = {"verdict": v[0] if v else None, "pass": lean_ok and live_ok and audit_ok, "lean_pass": lean_ok,
             "live_pass": live_ok, "audit_pass": audit_ok, "audit": audit, "live_why": live[0].get("why") if live else None}
json.dump(doc, open(out, "w"), indent=1, sort_keys=True)
EOF
done
cat $O/verdicts.json
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d and all(v["pass"] for v in d.values()) else 1)' $O/verdicts.json
