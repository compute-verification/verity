#!/usr/bin/env bash
# The node-side jobs of the erase calibration, run on a GPU node from benchmarks/pous/erase_calib with RESEARCH_RUN_DIR set.
# Each driver step takes its own preemptible `gpu-lease 1` on GPUs $ERASE_GPUS with its host threads on cores $ERASE_CPUS of
# NUMA node 0 (off node 2's merge-check slot, cores 0-47; the research user may run on 0-91), and writes small JSON to $RESEARCH_RUN_DIR/out. No lease is held
# past $ERASE_STOP_EPOCH. A validate or sweep step that is preempted (143) or reaches its --max-min (124) is taken up again and
# resumes from its JSON; a host step does not resume, so it takes one lease and is sized to fit it; a soak takes a fresh
# lease (and a fresh tag) per chunk until the stop epoch.
#
#   bash node.sh smoke                          validate, a 1 GiB + max sweep of chacha8/seq, a 1 GiB host pass
#   bash node.sh sweep SEED_RUN [driver args]   the size x generator x layout x order sweep
#   bash node.sh host [driver args]             host-DRAM fills and the GPU-to-pinned fill
#   bash node.sh soak TAG SEED_RUN [args]       back-to-back max-size fill+digest in <= 28 min chunks until the stop epoch
set -u
cd "$(dirname "$0")"
PY=${ERASE_PY:-/workspace/jobs/venv312/bin/python}
OUT=${RESEARCH_RUN_DIR:?}/out
STOP=${ERASE_STOP_EPOCH:?seconds since the epoch after which no lease is held}
STOP=${STOP%.*}
GPUS=${ERASE_GPUS:-0,1,2,3}
CPUS=${ERASE_CPUS:-48-91}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/erase-calib-triton-cache}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting:erase-calib}
PIN=()
command -v numactl >/dev/null && PIN=(numactl --physcpubind="$CPUS" --preferred=0)
mkdir -p "$OUT"

# lease MINUTES driver-args...: one leased driver step, at most MINUTES and never past the stop epoch (75 if no time is left)
lease() {
    local want=$1 left rc
    shift
    left=$(( (STOP - $(date +%s)) / 60 - 1 ))
    if [ "$left" -lt 2 ]; then
        echo "node.sh: stop epoch reached before driver $1" >&2
        return 75
    fi
    gpu-lease 1 --wait --preemptible --on "$GPUS" --max-min $(( want < left ? want : left )) -- \
        "${PIN[@]}" "$PY" driver.py "$@" --out "$OUT" --stop-epoch "$STOP"
    rc=$?
    echo "node.sh: driver $1 exit $rc at $(date -u +%H:%M:%SZ)" >&2
    return $rc
}

# step MINUTES driver-args...: a lease, taken up again after a preemption or a --max-min stop
step() {
    local rc
    while :; do
        lease "$@"
        rc=$?
        case $rc in
            124|143) continue ;;
            *) return $rc ;;
        esac
    done
}

phase=${1:?smoke|sweep|host|soak}
shift
case $phase in
    smoke)
        step 10 validate && step 15 sweep --sizes 1,max --gens chacha8 --layouts seq --orders seq,perm --trials 20 \
            && step 10 host --host-gib 1 --procs 1,4 --host-reps 2 --d2h-gib 2
        ;;
    sweep)
        seed_run=${1:?SEED_RUN}
        shift
        step 10 validate && step 28 sweep --seed-run "$seed_run" "$@"
        ;;
    host)
        lease 28 host "$@"
        ;;
    soak)
        tag=${1:?TAG}
        seed_run=${2:?SEED_RUN}
        shift 2
        step 10 validate
        rc=$?
        chunk=0
        while [ $rc -eq 0 ] && [ $(( STOP - $(date +%s) )) -gt 300 ]; do
            lease 28 soak --tag "$tag-c$chunk" --seed-run $(( seed_run + chunk * 1000000 )) --minutes 27 "$@"
            case $? in
                0|124|143) ;;
                *) break ;;
            esac
            chunk=$((chunk + 1))
        done
        [ $rc -eq 0 ]
        ;;
    *)
        echo "node.sh: unknown phase $phase" >&2
        exit 2
        ;;
esac
rc=$?
"$PY" driver.py result --out "$OUT"
exit $rc
