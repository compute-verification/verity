#!/usr/bin/env bash
# A served row's proof units ranked by gates (served_units.py, from the Boolean export of the served rows) and one unit class
# written as a universal-unit netlist (served_netlist.py), on a CPU node:
#   research run --on vy-nebius-cpu-1 --project verity --source . --cwd source --declared-output 'out/**' -- \
#     python3 tools/verity/check/slot.py -- bash benchmarks/private_circuit/served.sh DEF='GemmCoordinate_v1{K=8192}'
# env: EXPORT (the export's art id)  ROW (its row key)  DEF (a registered Definition id for served_netlist.py)
# A netlist over 64 MiB goes into the store (research data put --preserve) and out/netlist.art names it.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
EXPORT=${EXPORT:-art:cf2fc51314ed8b32bafe1a18e683e136892aabc0f39e1ea8711ea61dc133c0ea}
ROW=${ROW:-llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager}
REPO=$(pwd); O=$RESEARCH_RUN_DIR/out; mkdir -p $O
T=$(mktemp -d); trap 'rm -rf $T' EXIT
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
$UV venv -q --python 3.12 $T/py && $UV pip install -q --python $T/py/bin/python numpy blake3 || exit 1
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src
research data fetch $EXPORT --to $T/export --path commitments.json --path "$ROW.boolean.json" --path index.json > $O/fetch.log 2>&1 || { cat $O/fetch.log; exit 1; }
echo "$EXPORT $ROW" > $O/source.txt
$T/py/bin/python benchmarks/private_circuit/served_units.py $T/export "$ROW" --out $O > $O/served_units.log 2>&1 || { tail -20 $O/served_units.log; exit 1; }
if [ -n "${DEF:-}" ]; then
  /usr/bin/time -v -o $O/time-netlist.txt $T/py/bin/python benchmarks/private_circuit/served_netlist.py --definition "$DEF" --out $T/net \
    > $O/served_netlist.log 2>&1; rc=$?
  cp $T/net/netlist.summary.json $O/ 2>/dev/null
  if [ -f $T/net/netlist.json.gz ]; then
    sha256sum $T/net/netlist.json.gz | tee $O/netlist.sha256
    if [ $(stat -c %s $T/net/netlist.json.gz) -gt $((64 << 20)) ]; then
      research data put --kind fixture/v1 --file $T/net/netlist.json.gz --preserve \
        --meta "{\"what\": \"universal-unit netlist of $DEF's largest Q_word unit class\", \"run\": \"$RESEARCH_RUN_ID\"}" > $O/netlist.art 2> $O/put.err
      cat $O/netlist.art $O/put.err
    else
      cp $T/net/netlist.json.gz $O/
    fi
  fi
  exit $rc
fi
