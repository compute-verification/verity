# Sourced from the tree's root. cuda_13_3 LOG [LOCK]: CUDA_TK set and exported to a CUDA 13.3 toolkit: CUDA_TK where it is
# set, else the machine's (benchmarks/kernels/machine.py toolkit 13.3), else the one
# verity/protocols/proof/flock/prover/pod/cuda-13.3.sh finds or unpacks under FLOCK_WORK, one unpacking at a time under
# the lock file LOCK where given. What the finder and the unpacker say is appended to LOG. Returns 1 where none is found.
cuda_13_3() {
  local -
  set +euo pipefail
  local log=$1 lock=${2:-} unpack=verity/protocols/proof/flock/prover/pod/cuda-13.3.sh
  if [ -z "${CUDA_TK:-}" ] && ! CUDA_TK=$(python3 benchmarks/kernels/machine.py toolkit 13.3 2>> "$log"); then
    CUDA_TK=
    if [ -n "$lock" ]; then (exec 8>"$lock"; flock 8; . "$unpack") >> "$log" 2>&1; else (. "$unpack") >> "$log" 2>&1; fi \
      && . "$unpack" >> "$log" 2>&1
  fi
  [ -n "${CUDA_TK:-}" ] && export CUDA_TK
}
