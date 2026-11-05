#!/usr/bin/env bash
# The uniform-datapath probes on one datacenter GPU: the uniform datapath (probes/uniform.cuh) and the IMAD forms of
# duo's ALU work (probes/imad_forms.cuh), each family's counts pinned by its own table. From a pushed commit, on a 1x
# H200 or 1x B200:
#
#   research run --on <node> --project verity --source <tree> --cwd source --custody-ttl 12h \
#     --timeout 30m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/uni.sh sm90|sm100
#
# Outside the lease: build and SASS-gate both binaries (gate-build.txt; a refusal stops the run). Under the lease:
# probe.py run (probes/: rows.jsonl, gate.json, times.jsonl) with nvidia-smi sampled every 200 ms (env/smi.csv), then
# uniform.py analyze (uniform.json, uniform.txt). Exit: 0 done, 1 build or gate refused, 2 wrong device or no nvcc.
# DC_BUILD_ONLY=1 stops after the gate (any machine with CUDA 13.0). UNI_TABLES picks the tables (default "uniform imad";
# "issue" is the issue-saturated stream, the honest step beside 256 FFMA; "async" is the uniform chain beside a saturated
# async tensor op (wgmma on sm90, tcgen05.mma on sm100): its rows.jsonl carries the K-sweep for an offline slope fit;
# "ldtm" alone is sm100_ldtm.cu, the tcgen05.ld.32x32b.x128 issue price, analyzed by ldtm.py into ldtm.json and
# ldtm.txt; "a2u" is sm100_a2u.cu, A2u with every SMSP's issue saturated beside tcgen05.mma and a uniform chain,
# analyzed by a2u.py into a2u.json and a2u.txt).
set -uo pipefail
FAM=${1:?usage: uni.sh sm90|sm100}
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "uni.sh: no nvcc (CUDA 13.0 has sm_90a and sm_100a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
case "$FAM" in
  sm90) ARCH=sm_90a; CC=9 ;;
  sm100) ARCH=sm_100a; CC=10 ;;
  *) echo "uni.sh: family sm90 or sm100, not $FAM" >&2; exit 2 ;;
esac

if [ "${DC_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/probes"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  if [ "${DC_BUILD_ONLY:-}" != 1 ]; then
    nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = "$CC" ] || { echo "uni.sh: $FAM needs compute capability $CC.x, the device reports '$cap'" >&2; exit 2; }
  fi
  srcs=()
  for t in ${UNI_TABLES:-uniform imad}; do srcs+=("$HERE/probes/${FAM}_$t.cu"); done
  python3 "$PROBE" build --arch "$ARCH" --out "$WORK" "${srcs[@]}" \
    >"$WORK/built" 2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
  mapfile -t bins <"$WORK/built"
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "uni.sh: gate-build PASS"
  [ $rc -eq 0 ] || { grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2; exit 1; }
  [ "${DC_BUILD_ONLY:-}" = 1 ] && exit 0

  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 25 --)
  [ ${#lease[@]} -eq 0 ] && echo "uni.sh: no gpu-lease on this node; running on its one GPU directly" >&2
  "${lease[@]}" env DC_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}" "$FAM"
  exit $?
fi

smi_pid=
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
smi_pid=$!
trap '[ -n "$smi_pid" ] && kill "$smi_pid" 2>/dev/null' EXIT
mapfile -t bins <"$WORK/built"
python3 "$PROBE" run "${bins[@]}" --out "$OUT/probes" --reps "${REPS:-5}" >"$OUT/probes/run.txt" 2>&1
rc=$?
[ $rc -eq 0 ] || { tail -5 "$OUT/probes/run.txt" >&2; exit 1; }
case " ${UNI_TABLES:-} " in
  *" a2u "*) python3 "$HERE/a2u.py" "$OUT" --json "$OUT/a2u.json" >"$OUT/a2u.txt" 2>&1 || true ;;
esac
if [ "${UNI_TABLES:-}" = ldtm ]; then
  python3 "$HERE/ldtm.py" analyze "$OUT" --json "$OUT/ldtm.json" >"$OUT/ldtm.txt" 2>&1 || true
else
  python3 "$HERE/uniform.py" analyze "$OUT" --json "$OUT/uniform.json" >"$OUT/uniform.txt" 2>&1 || true
fi
echo "uni.sh: done"
