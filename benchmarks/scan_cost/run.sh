#!/usr/bin/env bash
# The scan's cost (README.md): one scan of the synthetic run at the benchmark's identity (the class w63k4, every block
# proved, the job on 21 cores, the Lean verifier on one), by benchmarks/network_traces/scan_zk.sh, then its result
# (result.py write): the two numbers with their stages, written where the guard holds. From a clone, in one slot of
# the network pool, as a job of at most 180 minutes:
#   research run --on NODE --project verity --source . --cwd clone --pool network --as LANE --mem-gb 128 \
#     --timeout 10800 -- bash benchmarks/scan_cost/run.sh
# env: SC_GROUP=G (scan_zk.sh's: G execution phases, each a registration; a setting, in the result beside the identity)
#      SC_CORES=N (21, the least a slot of the network pool gives: the job is pinned to the first N cores it has, and
#      a job with fewer is refused; another N is another identity, which compares with no result at 21)
#      SC_CLASS (w63k4, the benchmark's; `test`, network attestation's tests' class, rehearses this runner, and its
#      result compares with no w63k4 one)  SC_MEM_GB (128)  SC_CACHE  SC_PORT (scan_zk.sh's)
# Every other SC_ setting is cleared, those of scan_zk.sh that change what is proved among them, and so are
# OMP_NUM_THREADS and OMP_THREAD_LIMIT, which nproc answers with in place of the pin and the scan sizes its provers'
# threads by: the identity is not the caller's to change.
# Exit: 0, the guard holds and the result is written; the scan's own exit where it failed; 1, the guard does not hold
# or the result could not be written; 2, the job cannot run at the identity. The scan's kept files stay under $OUT
# whatever the exit, but for the verifier's copies of a round's circuits and program (ROUND/verifier/NAME-circuit.txt
# and ROUND/verifier/program.json): the scan writes them in its scratch and keeps them with the rest, they are most of a
# scan's kept bytes, and no number reads them. The scan takes no directory for them, so each is removed here, once the
# scan has ended, where a lean-verify.log of its round names its SHA-512 (the verifier's command names its circuit
# and its program by digest); one that no log names stays. The sessions' prove.log and lean-verify.log, which the
# stages are read from, are a child's logs and never in $OUT: the scan keeps them in its SC_KEEP, a 0700 directory
# made here outside the run dir and removed when this exits, after result.py has read it.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 2
: "${RESEARCH_RUN_DIR:?run.sh is the command of a research run (its header)}"
N=${SC_CORES:-21}
case $N in ''|*[!0-9]*|0*) echo "scan_cost: SC_CORES=$N is not a count of cores" >&2; exit 2;; esac
HAVE=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))') || exit 2
PIN=$(echo "$HAVE" | cut -d, -f1-"$N")
GOT=$(echo "$PIN" | tr , '\n' | grep -c .)
if [ "$GOT" -lt "$N" ]; then
  echo "scan_cost: this job has $GOT cores and the benchmark's identity is $N: run it in a slot that gives $N or" \
    "more (each of the network pool's does); SC_CORES=$GOT runs here, at an identity of its own" >&2
  exit 2
fi
echo "scan_cost: pinned to $N cores ($PIN) of the $(echo "$HAVE" | tr , '\n' | grep -c .) this job has"
for v in $(compgen -e SC_); do
  case $v in SC_GROUP|SC_CLASS|SC_MEM_GB|SC_CACHE|SC_PORT) ;; *) unset "$v";; esac
done
unset OMP_NUM_THREADS OMP_THREAD_LIMIT
KEEP=$(mktemp -d "${TMPDIR:-/tmp}/scan-cost-keep.XXXXXX") || { echo "scan_cost: no private directory for the scan's keep" >&2; exit 2; }
trap 'rm -rf "$KEEP"' EXIT
SC_CLASS="${SC_CLASS:-w63k4}" SC_MEM_GB="${SC_MEM_GB:-128}" SC_RSS=1 SC_LEAN_ONE=1 SC_KEEP=$KEEP \
  taskset -c "$PIN" bash benchmarks/network_traces/scan_zk.sh
rc=$?
echo "scan_cost: the scan exited $rc"
KEPT=$RESEARCH_RUN_DIR/out/sczk
if [ -d "$KEPT" ]; then
  find "$KEPT" -type f \( -path '*/verifier/*-circuit.txt' -o -path '*/verifier/program.json' \) -print0 \
    | while IFS= read -r -d '' f; do
        digest=$(sha512sum "$f" | cut -d' ' -f1) rel=${f#"$KEPT"/}
        logs=$(find "$KEEP/sczk/${rel%%/*}" -name lean-verify.log 2>/dev/null)
        if [ -z "$logs" ]; then
          echo "scan_cost: kept ${f#"$RESEARCH_RUN_DIR"/}: the scan's keep ($KEEP) holds no lean-verify.log of its round"
        elif echo "$logs" | tr '\n' '\0' | xargs -0 grep -qF -e "$digest"; then
          rm -f "$f"
        else
          echo "scan_cost: kept ${f#"$RESEARCH_RUN_DIR"/}: no lean-verify.log of its round names its SHA-512"
        fi
      done
  echo "scan_cost: $(du -sm "$KEPT" | cut -f1) MB of the scan's files kept, without the verifier's copies a kept log names"
fi
export PYTHONPATH=$ROOT:$ROOT/benchmarks/harness:$ROOT/tools/config
PY=(python3)   # the result's writer is the harness kit's, which this job's python3 may be too old to import
python3 -c 'import verity_bench.kit' 2>/dev/null || PY=(uv run -q --no-project --python 3.12 python)
PATH=$PATH:$HOME/.local/bin:$HOME/.cargo/bin "${PY[@]}" benchmarks/scan_cost/result.py write "$RESEARCH_RUN_DIR" --scan-exit "$rc" --keep "$KEEP"
wrote=$?
[ "$rc" -ne 0 ] && exit "$rc"
exit "$wrote"
