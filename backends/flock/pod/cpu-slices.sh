# Sourced by 74-gemm-hill.sh. Node 1 starts every `provers` job on the same cores (128-175), so jobs pinned side by side need
# cores of their own. hold_slices splits the job's cores into slices of SLICE_SIZE (16) and takes a lock (flock, under
# SLICE_LOCKS) on every slice the job will pin, held by the job's open descriptors until its last process exits; another
# job waits for them. In a Kubernetes container it then pins every process of the container (the harness included) to them.
#   hold_slices CPUSET N: the slices CPUSET touches, taken in order (blocking), pinning CPUSET;
#   hold_slices "" N:     any ceil(N / SLICE_SIZE) free slices, pinning the first N of their cores.
# Sets SLICE_CPUS (the cores to pin), SLICES (the slices held) and SLICE_WAIT_S. Must run in the job's own shell: a subshell
# would release the locks when it exits.
hold_slices() {
  local want=$1 n=${2:-48} L=${SLICE_LOCKS:-/workspace/jobs/slices} z=${SLICE_SIZE:-16} t0 s fd lo hi
  local -a all held fds
  t0=$(date +%s.%N)
  mkdir -p "$L" && chmod 1777 "$L" 2>/dev/null
  read -ra all < <(python3 -c "import os, sys; z = int(sys.argv[1]); c = sorted(os.sched_getaffinity(0)); print(' '.join(f'{c[i]}-{c[min(i + z - 1, len(c) - 1)]}' for i in range(0, len(c), z)))" "$z")
  open_() { [ -e "$L/$1.lock" ] || { : > "$L/$1.lock"; chmod 666 "$L/$1.lock"; } 2>/dev/null; exec {fd}<"$L/$1.lock"; }
  if [ -n "$want" ]; then
    # every core of CPUSET must be the job's own: taskset pins any core of the pod's cgroup, and no slice lock covers one
    # outside the job's cpuset (the dispatcher's, e.g. a provers task's 128-175)
    local w; w=$(python3 -c "import os, sys; w = {int(x) for p in sys.argv[1].split(',') for x in range(int(p.partition('-')[0]), int(p.partition('-')[2] or p.partition('-')[0]) + 1)}; print(','.join(map(str, sorted(w)))); sys.exit(0 if w <= os.sched_getaffinity(0) else 1)" "$want") \
      || { echo "hold_slices: CPUSET=$want is not within this job's cores (slices ${all[*]}): not pinning cores no lock covers" >&2; return 1; }
    for s in "${all[@]}"; do
      lo=${s%-*}; hi=${s#*-}
      python3 -c "import sys; lo, hi = int(sys.argv[2]), int(sys.argv[3]); sys.exit(0 if any(lo <= int(c) <= hi for c in sys.argv[1].split(',') if c) else 1)" "$w" "$lo" "$hi" || continue
      open_ "$s"
      flock -n "$fd" || { echo "hold_slices: waiting for cores $s" >&2; flock "$fd"; }
      held+=("$s"); fds+=("$fd")
    done
    SLICE_CPUS=$want
  else
    local need=$(( (n + z - 1) / z ))
    while :; do
      held=(); fds=()
      for s in "${all[@]}"; do
        [ ${#held[@]} -ge $need ] && break
        open_ "$s"
        if flock -n "$fd"; then held+=("$s"); fds+=("$fd"); else exec {fd}<&-; fi
      done
      [ ${#held[@]} -ge $need ] && break
      # never hold some while waiting for the rest: two jobs each holding half would wait on each other
      for fd in "${fds[@]}"; do exec {fd}<&-; done
      echo "hold_slices: $need of ${#all[@]} slices wanted, ${#held[@]} free; waiting" >&2
      sleep 15
    done
    SLICE_CPUS=$(python3 -c "import sys; c = [x for s in sys.argv[2:] for x in range(int(s.split('-')[0]), int(s.split('-')[1]) + 1)]; print(','.join(map(str, c[:int(sys.argv[1])])))" "$n" "${held[@]}")
  fi
  SLICES=$(IFS=,; echo "${held[*]}")
  SLICE_WAIT_S=$(python3 -c "import sys, time; print(round(time.time() - float(sys.argv[1]), 1))" "$t0")
  export SLICE_CPUS SLICES SLICE_WAIT_S
  # In a Kubernetes container every process is the job's, but the research run harness, its telemetry sampler and uv start on
  # the pod's whole affinity (node 1's provers range, 128-175) and would load another job's slice: pin them to this one too
  if [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
    local cg p
    cg=$(cat /proc/self/cgroup)
    for p in /proc/[0-9]*; do
      [ "$(cat "$p/cgroup" 2>/dev/null)" = "$cg" ] && taskset -a -p -c "$SLICE_CPUS" "${p#/proc/}" > /dev/null 2>&1
    done
  fi
  return 0
}
