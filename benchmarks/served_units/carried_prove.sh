#!/usr/bin/env bash
# prove_time.sh's prove step where a scheduler's job holds one card and the node has no CUDA toolkit, no gpu-lease and no
# git: each statement under FC/statements proved by a carried flock-circuit under --zk --gpu, WARM + RUNS sessions (default
# one, cold). Run from the tree's root. Its sessions, each with the statement's public files, go to OUT/kept/NAME for
# lean_check.sh off the node; OUT/summary.tsv has each statement's prover exit, wall seconds and peak card memory.
# env: OUT  COMMIT (the tree's, written to OUT/commit.txt)  FC (a directory: bin/flock-circuit, lib64/ with the libraries
#      its ldd names, statements/NAME/ as prove_time.sh's PT_STATEMENTS takes them)  WARM=0  RUNS=1  VCPUS=4
#      PT_DEVICE=gpu (cpu: the same sessions on the CPU prover, a rehearsal off the card)
set -uo pipefail
: "${OUT:?}" "${COMMIT:?}" "${FC:?}"
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
mkdir -p "$OUT" "$S/bin" && echo "$COMMIT" > "$OUT/commit.txt" && cp -r "$FC/statements" "$S/statements" || exit 1
printf '#!/usr/bin/env bash\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift; exec "$@"\n' > "$S/bin/gpu-lease"
chmod +x "$S/bin/gpu-lease"
# the prove step imports no package of the stager's venv, which prove_time.sh would otherwise fetch
mkdir -p "$S/fw/py/bin" && ln -s "$(command -v python3)" "$S/fw/py/bin/python"
G=${CUDA_VISIBLE_DEVICES:-0}
export PATH=$S/bin:$PATH CUDA_TK=$FC LD_LIBRARY_PATH=$FC/lib64:${LD_LIBRARY_PATH:-} GPU_LEASE_UUID=${G%%,*}
{ nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv -i "${GPU_LEASE_UUID:-0}"
  ldd "$FC/bin/flock-circuit"; sha256sum "$FC/bin/flock-circuit"; } > "$OUT/host.txt" 2>&1
RESEARCH_RUN_DIR=$OUT/pt PT_DEVICE=${PT_DEVICE:-gpu} STEPS=prove PT_BINARY=$FC/bin/flock-circuit PT_UNITS= FLOCK_WORK=$S/fw \
  PT_STATEMENTS="$(ls -d "$S"/statements/*/ | sed 's|/$||' | tr '\n' ' ')" WARM=${WARM:-0} RUNS=${RUNS:-1} VCPUS=${VCPUS:-4} \
  bash benchmarks/served_units/prove_time.sh
rc=$?
printf 'name\texit\twall_s\tpeak_mib\n' > "$OUT/summary.tsv"
for d in "$S"/statements/*/; do
  nm=$(basename "$d"); o=$OUT/pt/out/${PT_DEVICE:-gpu}/$nm; k=$OUT/kept/$nm; mkdir -p "$k"
  cp -r "$d/sessions" "$k/" 2>/dev/null; for f in pub.bin partition.json tables; do [ -e "$d/$f" ] && cp -r "$d/$f" "$k/"; done
  x=$(cat "$o/exit" 2>/dev/null || echo none); echo "$x" > "$k/exit"
  w=$(python3 -c 'import sys; print(round(float(open(sys.argv[2]).read()) - float(open(sys.argv[1]).read()), 1))' "$o/start" "$o/done" 2>/dev/null)
  m=$(cut -f2 "$o/smi.tsv" 2>/dev/null | cut -d, -f2 | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1)
  printf '%s\t%s\t%s\t%s\n' "$nm" "$x" "${w:-none}" "${m:-none}" >> "$OUT/summary.tsv"
  [ "$x" = 0 ] || rc=1
done
cat "$OUT/summary.tsv"
exit $rc
