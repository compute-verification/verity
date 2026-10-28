#!/usr/bin/env bash
# ovl.cu on one leased GPU (cwd = the source tree): the build and stage_cost/prep.py's constants outside the lease; under one
# lease, at each M, stage.cu's gate and then ovl over KS. An M whose gate fails is not timed (the run exits 5). Clocks are
# read, never set. Results in $OUT/*.jsonl; `python3 benchmarks/pouw/stage_ovl/report.py $OUT` renders them.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/stage_ovl/run.sh
# env: MS (512 1024 2048), KS (1 2 4 8 16 32), ARMS (ovl.cu's default), REPS (5), DRAW_US (300), PLAIN_RB (8192),
# MAXMIN (25), CUDA_DEVICE_MAX_CONNECTIONS (32: a hardware queue per chunk), FR_ROT (0: stage.cu's bits, the commitment's
# rotations on the FMA pipe), PARTS (thru gate ovl: shathru's SHA-512 throughput, the gate, the grid; ovl and eovl need gate;
# beside: ovl.cu's beside at each M:K of BESIDE, 2048:4 2048:16 512:32; cgcheck: commit_gpu's case set (CHECK, full) on
# its small path, the small path with the top fused into the leaf kernel and the next block prefetched (fused), and its
# default path (CG_CHECK, small fused default), against verity.ml.commitments).
# The cg arms (commit_gpu's committer): CG_RS (64), the tile heights; ovl runs with ARMS at the first and with CG_ARMS
# (plain chunked chunkedrr2 chunkedrr4 cgrr2 cgrr4 cgonly nohashrr4) over CG_KS (8 16) at each further one. Every such
# run dumps chunk 0's six commitments, which `reference.py verify` holds to verity.ml.commitments after the lease.
# cgdiag: the committer's breakdown at each M (DIAG_R, 128; DIAG_KS, 8; DIAG_ARMS, chunked cgrr2 cgonly liteconly litecrr2):
# ovl, then each timing-only build of DIAGS (noop nos nomask notree all: commit_gpu's -DCOMMIT_GPU_DIAG_*, the same
# compressions without the operand's loads, the salt records' loads, M_key y's lookups, the CTA levels' global memory;
# all four) and ovl with CG_SKIP=salts and CG_SKIP=top, into diag-NAME.jsonl (its cg roots differ by construction).
# eovl: the edited arms (ovl.cu's E:, ER:, eg.cu: EGEMM's edited GEMM with windowed duo receipts) at each M and each
# receipt stride S of EG_SS (16 32 64 128), X committed at tile height EG_R (128): ovl over EG_KS (1 8) with EG_ARMS into
# eovl-M-sS.jsonl. Each run dumps chunk 0's last call of each linear, which eref.py holds to EGEMM's reference
# (segment.PouwSegCheck_v2) and `reference.py verify` holds to verity.ml.commitments after the lease, with the X dumps.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
D=benchmarks/pouw/stage_ovl
G=benchmarks/pouw/commit_gpu
export B; B=$(mktemp -d /tmp/stage-ovl.XXXXXX); trap 'rm -rf "$B"' EXIT
export MS=${MS:-512 1024 2048} KS=${KS:-1 2 4 8 16 32} REPS=${REPS:-5} PARTS=${PARTS:-thru gate ovl}
export BESIDE=${BESIDE:-2048:4 2048:16 512:32}
export CG_RS=${CG_RS:-64} CG_KS=${CG_KS:-8 16} CG_ARMS=${CG_ARMS:-plain chunked chunkedrr2 chunkedrr4 cgrr2 cgrr4 cgonly nohashrr4}
export DIAGS=${DIAGS:-noop nos nomask notree all} DIAG_R=${DIAG_R:-128} DIAG_KS=${DIAG_KS:-8}
export DIAG_ARMS=${DIAG_ARMS:-chunked cgrr2 cgonly liteconly litecrr2}
export EG_SS=${EG_SS:-16 32 64 128} EG_R=${EG_R:-128} EG_KS=${EG_KS:-1 8}
export EG_ARMS=${EG_ARMS:-plain chunked chunkedrr2 Q:chunked Q:chunkedrr2 EN:chunked E:chunked E:cgprend-fp ER:cgprend-fp \
  ER:cgpre-fp ER:cgprerr2-fp ERL:cgpre-fp ERL:cgprerr2-fp E:cgprendeq-fp ER:cgprendeq-fp ER:cgpreeq-fp}
export PYTHONPATH=.
NV=(nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -DFR_ROT="${FR_ROT:-0}")
LIBS=(-lcuda -lcublasLt -lnvidia-ml -ldl)
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
if [[ " $PARTS " == *" cgcheck "* ]]; then
  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/commit_gpu" $G/commit_gpu.cu -lcublasLt -lnvidia-ml \
    > "$O/build-commit_gpu.log" 2>&1 &
  python3 $G/reference.py prepare "$B/cases" "${CHECK:-full}" > "$O/prepare.jsonl" 2> "$O/prepare.err" &
fi
if [[ " $PARTS " == *" cgdiag "* ]]; then
  for v in $DIAGS; do
    case $v in
      all) f="-DCOMMIT_GPU_DIAG_NOOP -DCOMMIT_GPU_DIAG_NOS -DCOMMIT_GPU_DIAG_NOMASK -DCOMMIT_GPU_DIAG_NOTREE" ;;
      *) f="-DCOMMIT_GPU_DIAG_${v^^}" ;;
    esac
    # shellcheck disable=SC2086
    "${NV[@]}" $f -Xptxas -v -c -o "$B/ovl-$v.o" $D/ovl.cu > "$O/build-$v.log" 2>&1 &
  done
fi
# ovl.cu and eg.cu (its edited arms: commit_gpu's constants for the receipts' rows in their own unit) side by side
"${NV[@]}" -Xptxas -v -c -o "$B/eg.o" $D/eg.cu > "$O/build-eg.log" 2>&1 &
egp=$!
"${NV[@]}" -Xptxas -v -c -o "$B/ovl.o" $D/ovl.cu > "$O/build.log" 2>&1
o=$?
wait $egp
e=$?
cat "$O/build-eg.log" >> "$O/build.log"
[ $o = 0 ] && [ $e = 0 ] && "${NV[@]}" -o "$B/ovl" "$B/ovl.o" "$B/eg.o" "${LIBS[@]}" >> "$O/build.log" 2>&1 ||
  { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
sha256sum "$B/ovl" | cut -c1-64 > "$O/ovl.sha256"
python3 benchmarks/pouw/stage_cost/prep.py "$B/c" --rows "${MS// /,}" > "$O/prep.log" 2>&1 || { tail -20 "$O/prep.log"; exit 6; }
# the cg arms' CGP1 files: stage t's operand (M x SK[t] int8, stage.cu's SK) as tile leaves of R rows, its own domain, one Ks
SK=(4096 6144 256 4096 4096 14336)
RS=$(printf '%s\n' $CG_RS $([[ " $PARTS " == *" eovl "* ]] && echo "$EG_R") | sort -un)
for M in $MS; do
  for R in $RS; do
    for t in 0 1 2 3 4 5; do
      python3 $G/reference.py params "$B/c/cg-$M-$R-$t.bin" $(((M + R - 1) / R * (SK[t] / 32))) $((256 * R)) --ks-seed 7 &
    done
  done
done
# the edited arms' receipts' CGP1 files: linear t's receipts (EN[t] x EK[t] weights) at stride S as 2 KiB rows (a window of
# a tile each), tree R-NAME, one Ks of their own; S as ovl.cu's es[t] (the largest multiple of 4 at most S dividing EK / 32)
EN=(6144 0 0 4096 28672 4096) EK=(4096 0 0 4096 4096 14336) SN=(qkv qk pv o gate_up down)
if [[ " $PARTS " == *" eovl "* ]]; then
  for M in $MS; do
    for S in $EG_SS; do
      for t in 0 3 4 5; do
        s=$S
        while ((s > 4 && (EK[t] / 32) % s)); do s=$((s - 4)); done
        python3 $G/reference.py params "$B/c/eg-$M-$S-$t.bin" $((M / 128 * (EN[t] / 128) * (EK[t] / (32 * s)))) 16384 --ks-seed 11 \
          --tree "R-${SN[t]}" &
      done
    done
  done
fi
wait
if [[ " $PARTS " == *" cgdiag "* ]]; then
  for v in $DIAGS; do
    [ -s "$B/ovl-$v.o" ] && "${NV[@]}" -o "$B/ovl-$v" "$B/ovl-$v.o" "$B/eg.o" "${LIBS[@]}" >> "$O/build-$v.log" 2>&1
  done
fi
[[ " $PARTS " == *" cgcheck "* ]] && { [ -x "$B/commit_gpu" ] || { tail -20 "$O/build-commit_gpu.log"; exit 4; }; }
mkdir -p "$B/dump" "$B/edump"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  gf=0
  if [[ " $PARTS " == *" cgcheck "* ]]; then
    for p in ${CG_CHECK:-small fused default}; do
      COMMIT_GPU_SMALL=$([ "$p" = default ] && echo 0 || echo 1) COMMIT_GPU_FUSE=$([ "$p" = fused ] && echo 1 || echo 0) \
        COMMIT_GPU_PREFETCH=$([ "$p" = fused ] && echo 1 || echo 0) \
        python3 benchmarks/pouw/commit_gpu/reference.py check "$B/commit_gpu" "$B/cases" "$p" > "$O/check-$p.jsonl" 2> "$O/check-$p.err"; c=$?
      echo "check $p rc $c" >> "$O/rc.txt"; [ $c = 0 ] || gf=7
    done
  fi
  if [[ " $PARTS " == *" thru "* ]]; then
    "$B/ovl" shathru > "$O/shathru.jsonl" 2>> "$O/errors.txt"; echo "shathru rc $?" >> "$O/rc.txt"
  fi
  if [[ " $PARTS " == *" beside "* ]]; then
    for mk in $BESIDE; do
      "$B/ovl" beside "${mk%%:*}" "${mk##*:}" r3 >> "$O/beside.jsonl" 2>> "$O/errors.txt"; echo "beside $mk rc $?" >> "$O/rc.txt"
    done
  fi
  [[ " $PARTS " == *" gate "* ]] || exit $gf
  for M in $MS; do
    "$B/ovl" gate "$B/c" "$M" >> "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
    echo "gate $M rc $g" >> "$O/rc.txt"
    if [ "$g" -ne 0 ]; then gf=5; continue; fi
    if [[ " $PARTS " == *" cgdiag "* ]]; then
      for v in base $DIAGS nosalts notop; do
        bin="$B/ovl-$v" skip=""
        case $v in base) bin="$B/ovl" ;; nosalts) bin="$B/ovl" skip=salts ;; notop) bin="$B/ovl" skip=top ;; esac
        [ -x "$bin" ] || { echo "diag $v not built" >> "$O/rc.txt"; continue; }
        CG_R=$DIAG_R CG_SKIP=$skip ARMS="$DIAG_ARMS" "$bin" ovl "$B/c" "$M" $DIAG_KS "r$REPS" > "$O/diag-$M-$v.jsonl" 2>> "$O/errors.txt"
        echo "diag $M $v rc $?" >> "$O/rc.txt"
      done
    fi
    if [[ " $PARTS " == *" eovl "* ]]; then
      for S in $EG_SS; do
        EG_S=$S CG_R=$EG_R CG_DUMP="$B/dump" EG_DUMP="$B/edump" ARMS="$EG_ARMS" "$B/ovl" ovl "$B/c" "$M" $EG_KS "r$REPS" \
          > "$O/eovl-$M-s$S.jsonl" 2>> "$O/errors.txt"
        r=$?
        echo "eovl $M S=$S rc $r" >> "$O/rc.txt"
        [ $r = 0 ] || gf=10
        q
      done
    fi
    [[ " $PARTS " == *" ovl "* ]] || continue
    first=1
    for R in $CG_RS; do
      if [ $first = 1 ]; then
        CG_R=$R CG_DUMP="$B/dump" "$B/ovl" ovl "$B/c" "$M" $KS "r$REPS" >> "$O/ovl.jsonl" 2>> "$O/errors.txt"; r=$?
      else
        CG_R=$R CG_DUMP="$B/dump" ARMS="$CG_ARMS" "$B/ovl" ovl "$B/c" "$M" $CG_KS "r$REPS" >> "$O/ovl-r$R.jsonl" 2>> "$O/errors.txt"; r=$?
      fi
      first=0
      echo "ovl $M R=$R rc $r" >> "$O/rc.txt"
      q
    done
  done
  exit "$gf"
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
if compgen -G "$B/edump/*" > /dev/null; then
  python3 $D/eref.py "$B"/edump/* > "$O/eref.jsonl" 2> "$O/eref.err"; e=$?
  echo "eref rc $e" >> "$O/rc.txt"
  [ $e = 0 ] || [ $r != 0 ] || r=9
fi
if compgen -G "$B/dump/*" > /dev/null || compgen -G "$B/edump/*" > /dev/null; then
  python3 $G/reference.py verify $(compgen -G "$B/dump/*") $(compgen -G "$B/edump/*") > "$O/verify.jsonl" 2> "$O/verify.err"; v=$?
  echo "verify rc $v" >> "$O/rc.txt"
  [ $v = 0 ] || [ $r != 0 ] || r=8
fi
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl "$O/verify.jsonl" "$O/eref.jsonl"; do [ -e "$f" ] && tail -1 "$f"; done
grep -v "^device" "$O/errors.txt" | tail -5 || true
exit "$r"
