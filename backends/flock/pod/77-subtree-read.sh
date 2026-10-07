#!/usr/bin/env bash
# The subtree read (`verity.primitives.commitments.gates.merkle.MerkleSubtreeRead`, read_gadget.py): its slot types against the references, the
# merkle tests (slow ones included), circuit-check of its two bindings, and every read of a run of 1, 16 or 256 cells in a store
# of 2^30 to 2^33 cells built whole and tallied (`read_gadget.py measure`). Run from the verity tree (research run --cwd source);
# CPU only. Outputs under out/ (flat): host.txt, check.json, pytest.log, circuit-check.json and .log, measure.jsonl,
# measure-summary.json, measure.log, time-measure.txt (peak RSS), rc.tsv (each step's exit code). The uv environment goes in a
# scratch directory removed at exit.
# env: CPUS=24 (cores held, cpu-slices.sh)  JOBS (measure's workers; default CPUS - 8: pytest takes 6, circuit-check 2)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${ST_PINNED:-}" ]; then
  . backends/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-24}" || exit 1
  export ST_PINNED=$SLICE_CPUS
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
O=${RESEARCH_RUN_DIR:-/tmp/subtree-read}/out; mkdir -p $O
T=$(mktemp -d); trap 'rm -rf $T' EXIT
N=$(echo $ST_PINNED | tr ',' '\n' | wc -l)
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$ST_PINNED slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"
  free -g | head -2; cat /proc/loadavg; df -i . | tail -1; } > $O/host.txt 2>&1
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }
[ -x /usr/bin/time ] || { echo "no /usr/bin/time: the measurement's peak memory is part of it" >&2; exit 1; }
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
export UV_PROJECT_ENVIRONMENT=$T/venv
RUN="$UV run -q --frozen --extra torch-cpu"
$RUN python -c 'import circuit_check, verity_flock, xdist' > $O/env.log 2>&1 || { tail -30 $O/env.log; exit 1; }
$RUN python backends/flock/pod/read_gadget.py check > $O/check.json || { cat $O/check.json; exit 1; }

( $RUN python -m pytest -q -p no:cacheprovider -n 6 verity/primitives/commitments/tests/test_boolean_merkle.py > $O/pytest.log 2>&1
  echo "pytest	$?" >> $O/rc.tsv ) &
( $RUN python -m circuit_check -j 2 --out $O/circuit-check.json \
    'MerkleSubtreeRead_v1{ARITY=2,D=2,J=1}' 'MerkleSubtreeRead_v1{ARITY=4,D=1,J=0}' > $O/circuit-check.log 2>&1
  echo "circuit-check	$?" >> $O/rc.tsv ) &
/usr/bin/time -v -o $O/time-measure.txt $RUN python backends/flock/pod/read_gadget.py measure --jobs ${JOBS:-$((N > 12 ? N - 8 : 4))} \
  --out $O > $O/measure.log 2>&1
echo "measure	$?" >> $O/rc.tsv
wait
cat $O/rc.tsv
! grep -qv '	0$' $O/rc.tsv
