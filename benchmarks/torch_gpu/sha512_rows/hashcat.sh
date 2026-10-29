#!/usr/bin/env bash
# hashcat's SHA-512 kernels against sha512x.cuh's default compression on one leased GPU, in one lease at one locked SM
# clock. hashcat (HC_TAG, default v7.1.2) is built from its source in scratch. Its benchmark (`hashcat -b`: the mask
# ?a?a?a?a?a?a?a, a 7-byte candidate, and for -m 1710 the self-test salt's 12 bytes, so one SHA-512 block a candidate) runs
# -m 1700 and -m 1710 each with its optimized kernel (-O, what `hashcat -b` runs by default) and its pure one, at -w 3
# (the benchmark's default), and -m 1700 -O at -w 4. Ours: commit_gpu's raw rate (`commit_gpu sha`, sha_rate: each thread
# chains full compressions of a variable block) for commit_gpu's default build (a0r0l) and sx0000, sha512x.cuh's default
# force-included (ruler.sh's builds), each checked bit for bit against reference.py in the lease before it is timed; and
# hashcat_forms.cu, our compression in hashcat's shapes (its header), its first digests checked against hashlib, and the
# honest chain and -O's shape at each CTA size in SHAPES, in whole waves of the CTAs an SM holds (commit_gpu's sha launches
# 8 CTAs of 256 an SM, 3 resident, a partial last wave).
# cwd = the source tree's root. Outside the lease: the builds and reference.py's cases. The lease: the SM clock locked at
# LOCK_MHZ (default 1980) and reset on exit, nvidia-smi every 200 ms, the checks, one warm-up a hashcat setting (its kernel
# build, Accel/Loops/Thr/Vec and the mask), then ROUNDS rounds with every arm interleaved. After it: the SASS pipe mix of
# hashcat's cached kernels (cubins from its cuLink) and of ours (sass_count.py), then hashcat_summary.py: summary.json and
# verdict.txt's VERDICT lines.
#
#   OUT=DIR bash benchmarks/torch_gpu/sha512_rows/hashcat.sh
#
# Needs a CUDA GPU (an H200, or an RTX PRO 6000), CUDA's nvcc, `gpu-lease` on PATH, `sudo -n nvidia-smi` for the clock
# lock and network access for hashcat's source.
# ROUNDS (3), LOCK_MHZ (1980), LEASE_MIN (25), HC_TAG (v7.1.2), JOBS (16), SHAPES ("128 256 384 512 640 768 1024"),
# PY (python3; it needs numpy: on a node without it, PY="uv run -q --no-project --with numpy python").
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export D=benchmarks/pouw/commit_gpu X=benchmarks/torch_gpu/sha512_rows PY=${PY:-python3} PYTHONPATH=.
W=$(mktemp -d /dev/shm/hashcat24.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 90 | 100 | 120) SM=${CC}a ;; *) SM=$CC ;; esac
export SM LOCK=${LOCK_MHZ:-1980} ROUNDS=${ROUNDS:-3} LEASE_MIN=${LEASE_MIN:-25} HC_TAG=${HC_TAG:-v7.1.2}
export SHAPES=${SHAPES:-128 256 384 512 640 768 1024}
echo "compute_cap $CC sm_$SM" > "$O/arch.txt"
nvcc --version > "$O/nvcc.txt" 2>&1

build() {
  local t=$1 flags="-DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1" sx
  if [ "$t" != a0r0l ]; then
    sx=$($PY -c "import sys; sys.path.insert(0, '$X'); import engine; print(*engine.variant_flags('${t#sx}'))") ||
      { echo "build $t rc flags" >> "$O/rc.txt"; return; }
    flags="$flags -include $X/as_commit_gpu.cuh $sx"
  fi
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v $flags -o "$W/commit_gpu_$t" $D/commit_gpu.cu \
    -lcublasLt -lnvidia-ml > "$O/build-$t.log" 2>&1
  echo "build $t rc $?" >> "$O/rc.txt"
}
hashcat_build() {
  git clone -q --depth 1 --branch "$HC_TAG" https://github.com/hashcat/hashcat.git "$W/hashcat" > "$O/hashcat-build.log" 2>&1 &&
    git -C "$W/hashcat" rev-parse HEAD > "$O/hashcat-commit.txt" &&
    make -C "$W/hashcat" -j "${JOBS:-16}" hashcat modules/module_00000.so modules/module_01700.so modules/module_01710.so \
      >> "$O/hashcat-build.log" 2>&1
  echo "build hashcat rc $?" >> "$O/rc.txt"
}
# hashcat dlopens libnvrtc.so or libnvrtc.so.1 alone: a link to the toolkit's own, whatever its version suffix
mkdir -p "$W/lib"
nvrtc=$(find /usr/local/cuda/lib64/ /usr/local/cuda/targets/*/lib/ -maxdepth 1 -name 'libnvrtc.so*' 2>/dev/null | sort | head -1)
[ -n "$nvrtc" ] && ln -sf "$nvrtc" "$W/lib/libnvrtc.so"
echo "nvrtc ${nvrtc:-none}" >> "$O/rc.txt"
export LD_LIBRARY_PATH=$W/lib:$LD_LIBRARY_PATH
build a0r0l &
build sx0000 &
nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -o "$W/hashcat_forms" $X/hashcat_forms.cu -lnvidia-ml \
  > "$O/build-forms.log" 2>&1 && echo "build forms rc 0" >> "$O/rc.txt" || echo "build forms rc $?" >> "$O/rc.txt" &
hashcat_build &
$PY $D/reference.py prepare "$W/cases" quick > "$O/prepare.jsonl" 2> "$O/prepare.err"
echo "prepare rc $?" >> "$O/rc.txt"
wait
cat "$O/rc.txt"
[ -x "$W/hashcat/hashcat" ] || { echo "no hashcat binary"; tail -30 "$O/hashcat-build.log"; exit 1; }

export HC="$W/hashcat/hashcat" HOME="$W/home" CFGS="1700-O 1700-pure 1710-O 1710-pure 1700-O-w4"
mkdir -p "$HOME"
hcargs() {   # the hashcat flags of setting $1: MODE-KERNEL[-wN]
  local m=${1%%-*} rest=${1#*-} w=3
  case $rest in *-w4) w=4; rest=${rest%-w4} ;; esac
  printf -- '-b -m %s -w %s %s --backend-ignore-opencl --backend-ignore-hip' "$m" "$w" "$([ "$rest" = O ] && echo -O)"
}
export -f hcargs
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  set -u
  U=$GPU_LEASE_UUID
  ph() { local n=$1 s rc; shift; s=$(date +%s.%N); "$@"; rc=$?; echo "$n $s $(date +%s.%N) $rc" >> "$O/phases.txt"; return $rc; }
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$U" \
    > "$O/gpu.txt" 2>&1
  nvidia-smi -i "$U" -q -d CLOCK,PERFORMANCE > "$O/clock-before.txt" 2>&1
  nvidia-smi -i "$U" -q -d SUPPORTED_CLOCKS 2>&1 | head -40 > "$O/supported-clocks.txt"
  # the clock under load, from hashcat_forms (NVML after each launch): the least of its forms medians
  probe() { "$W/hashcat_forms" time 5 2>/dev/null | python3 -c "import sys, json; print(int(min(json.loads(l)[\"mhz\"] for l in sys.stdin)))"; }
  sudo -n nvidia-smi -i "$U" -lgc "$LOCK,$LOCK" > "$O/lock.txt" 2>&1; echo "lock rc $?" >> "$O/rc.txt"
  trap "sudo -n nvidia-smi -i $U -rgc >> $O/lock.txt 2>&1; nvidia-smi -i $U -q -d CLOCK > $O/clock-after.txt 2>&1" EXIT
  # a lock this card resolves below LOCK (an H200 puts -lgc 1980,1980 at 1830 under load): unlocked, if the
  # boost holds LOCK under load, else the lock as it resolves. Within 1% is held: on the RTX PRO 6000 NVML reads the lock
  # 0.4% low (-lgc 2085,2085 at 2077).
  locked=$(probe) free=n/a mode=locked
  if [ "${locked:-0}" -lt $((LOCK * 99 / 100)) ]; then
    sudo -n nvidia-smi -i "$U" -rgc >> "$O/lock.txt" 2>&1
    free=$(probe)
    if [ "${free:-0}" -ge "$LOCK" ]; then mode=boost; else
      sudo -n nvidia-smi -i "$U" -lgc "$LOCK,$LOCK" >> "$O/lock.txt" 2>&1
    fi
  fi
  echo "clock mode $mode: under load $locked MHz locked at $LOCK,$LOCK, $free MHz unlocked" >> "$O/rc.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -lms 200 -i "$U" > "$O/smi.csv" 2>&1 &
  smi=$!
  start=$SECONDS TIMED=""
  for t in a0r0l sx0000; do
    [ -x "$W/commit_gpu_$t" ] || continue
    "$W/commit_gpu_$t" selftest >> "$O/selftest.jsonl"; s=$?
    $PY $D/reference.py check "$W/commit_gpu_$t" "$W/cases" "$t" > "$O/check-$t.jsonl" 2> "$O/check-$t.err"; c=$?
    echo "selftest $t rc $s; check $t rc $c ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
    [ $s = 0 ] && [ $c = 0 ] && TIMED="$TIMED $t"
  done
  if [ -x "$W/hashcat_forms" ]; then "$W/hashcat_forms" check > "$O/forms-check.jsonl" 2>&1; echo "forms check rc $?" >> "$O/rc.txt"; fi
  echo "timed:$TIMED" >> "$O/rc.txt"
  cd "$W/hashcat" || exit 1
  ph hashcat-I "$HC" -I --backend-ignore-opencl > "$O/hashcat-I.txt" 2>&1
  for cfg in $CFGS; do
    # shellcheck disable=SC2046
    ph "warm-$cfg" "$HC" $(hcargs "$cfg") > "$O/hc-warm-$cfg.txt" 2>&1
    echo "warm $cfg rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
    if ! grep -q "Speed.#" "$O/hc-warm-$cfg.txt"; then   # refused (a driver or toolkit warning): once more with --force
      touch "$O/force-$cfg"
      # shellcheck disable=SC2046
      ph "warm-$cfg-force" "$HC" $(hcargs "$cfg") --force > "$O/hc-warm-$cfg-force.txt" 2>&1
      echo "warm $cfg --force rc $?" >> "$O/rc.txt"
    fi
  done
  for r in $(seq "$ROUNDS"); do
    for t in $TIMED; do ph "sha-$t-$r" "$W/commit_gpu_$t" sha 20 >> "$O/sha-$t.jsonl" 2>> "$O/bench.err"; done
    [ -x "$W/hashcat_forms" ] && ph "forms-$r" "$W/hashcat_forms" time 20 >> "$O/forms.jsonl" 2>> "$O/bench.err"
    [ -x "$W/hashcat_forms" ] && ph "shapes-$r" "$W/hashcat_forms" shapes 20 $SHAPES >> "$O/shapes.jsonl" 2>> "$O/bench.err"
    for cfg in $CFGS; do
      # shellcheck disable=SC2046
      ph "hc-$cfg-$r" "$HC" $(hcargs "$cfg") $([ -e "$O/force-$cfg" ] && echo --force) --machine-readable \
        > "$O/hc-$cfg-$r.txt" 2>&1
    done
    echo "round $r done ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"

# hashcat's kernel cache: cuLink's cubins (WITH_CUBIN), the code the device ran; a PTX cache file is assembled by ptxas
find "$W" -name 'm017*.kernel' > "$O/hashcat-kernels.txt"
while read -r k; do
  b=$(basename "$k" .kernel)
  file "$k" >> "$O/hashcat-kernels-file.txt"
  if cuobjdump -sass "$k" > "$W/sass-$b.txt" 2> "$O/sass-$b.err" && grep -q 'Function :' "$W/sass-$b.txt"; then
    cp "$k" "$O/$b.cubin"
  else
    ptxas -arch=sm_$SM -O3 -o "$W/$b.cubin" "$k" >> "$O/sass-$b.err" 2>&1 && cuobjdump -sass "$W/$b.cubin" > "$W/sass-$b.txt" &&
      echo "$b: PTX, assembled by ptxas -arch=sm_$SM -O3" >> "$O/hashcat-kernels-file.txt"
  fi
  gzip -c "$W/sass-$b.txt" > "$O/sass-$b.txt.gz"
done < "$O/hashcat-kernels.txt"
for t in a0r0l sx0000; do
  [ -x "$W/commit_gpu_$t" ] || continue
  cuobjdump -sass "$W/commit_gpu_$t" > "$W/sass-$t.txt" 2>&1
  $PY $X/sass_count.py "$W/commit_gpu_$t" sha_rate --trip 4 --sass "$W/sass-$t.txt" > "$O/sass-$t.json" 2>> "$O/sass.err"
  awk '/Function : /{p = /sha_rate/} p' "$W/sass-$t.txt" | gzip -c > "$O/sass-sha_rate-$t.txt.gz"
done
[ -x "$W/hashcat_forms" ] && cuobjdump -sass "$W/hashcat_forms" | gzip -c > "$O/sass-forms.txt.gz"
$PY $X/hashcat_summary.py "$O" | tee "$O/verdict.txt"
cat "$O/rc.txt"
