#!/usr/bin/env bash
# Pearl-C's sm_90a follow-up session on one H100 or H200, from a build's ship as h200.sh takes it (cwd = the source tree's root):
#
#   tar -C BUILD_RUN/out/ship -cf /tmp/ship.tar .
#   research run --on vy-dc-h200 --project verity --campaign pouw --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=LANE --send /tmp/ship.tar -- bash benchmarks/pouw/pearl_c_sm90/h200_profile.sh
#
# Under one lease of at most LEASE_MIN minutes (default 20): run.py's check at the fixture's shape (no diagnostic timing), and
# only if it passed, each step whose switch is set:
#   TIMING  N:K ... (default 2624:6144)                glm53_timing.py at those shapes (TIMING_M: its --m), SCHEDULES
#                                                      (default serial) its pearlc arms, interleaved (GRAPHS=1: --graphs),
#                                                      into $OUT/timing[-graphs]
#   STAMPS  1 (default 0)                              glm53_profile.py stamps (every in-domain shape at m 16 and 64, serial
#                                                      and deferred-a; STAMPS_SHAPES N:K ... and STAMPS_M M ... instead),
#                                                      into $OUT/stamps.json
#   HOST    1 (default 0)                              glm53_profile.py host (a call's host time and its cProfile, at m 16
#                                                      and 64, SCHEDULES' arms; GRAPHS=1: --graphs too), into $OUT/host[-graphs].json
#   LANES   SCHEDULE ... (default unset)               glm53_profile.py run under nsys with graphs' nodes traced, a schedule
#                                                      at a time (12 calls at m 16 and 64 at LANES_SHAPES, default 512:6144
#                                                      4096:6144; GRAPHS=1: --graphs), into $OUT/lanes/SCHEDULE
#   PASSES  SCHEDULE (default unset)                   glm53_passes.py capture --schedule SCHEDULE (every in-domain shape;
#                                                      GRAPHS=1: --graphs), in scratch
#   TILES   N:K ... (default 512:6144 3072:6144 6144:1536)  glm53_passes.py capture --shapes N:K, a shape at a time, in scratch
#   PROFILE 1 (default)                                glm53_profile.py run under nsys (the expert shapes at m 16 and 2,048), into $OUT/profile
#   GROUPED 1 (default 0; the ship's cubin must have the grouped kernels)  grouped_check.py (its graphed arms too), its passes
#                                                      in scratch
#   GPROF   1 (default 0; as GROUPED)                  glm53_profile.py grouped --graphs (host and events, cProfile) into
#                                                      $OUT/gprof/host.json, then its --nvtx calls under nsys into $OUT/gprof/prof
#   GROUPED_PLACES PLACE ... (default base)            GROUPED's check and GPROF's nsys calls once per grouped.py PLACE
#                                                      (PEARLC_GROUPED_PLACE; base none), into grouped[-PLACE] and gprof[-PLACE]
# Then, the lease released: PASSES' replay (glm53_passes.py verify --fast) and full check of FULL_PASSES (default decode; full:
# every tile and every served y word; their JSON and manifests into $OUT/passes), both under nice with FULL_JOBS jobs (default
# 8 of the VM's 16 CPUs, which a GPU session beside them shares), each TILES shape's passes' scalar replay (glm53_passes.py verify without
# --fast: every drawn tile on the reference chain; their JSON and manifests into $OUT/tiles/NxK), the profile's SQLite export
# and report, GPROF's lanes report, and the grouped passes' replay (--fast).  VERITY_BENCH_VENV and
# CUDA as h200.sh.  LANES' exports and lanes reports (glm53_profile.py lanes) run after the lease too, under nice.  Every
# step's exit code is in $OUT/rc.txt; exits non-zero if any step failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/pouw/pearl_c_sm90
CUDA=${CUDA:-/usr/local/cuda-13.0}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw${PYTHONPATH:+:$PYTHONPATH}
TIMING=${TIMING-2624:6144} TILES=${TILES-512:6144 3072:6144 6144:1536} PROFILE=${PROFILE:-1} GROUPED=${GROUPED:-0}
SCHEDULES=${SCHEDULES:-serial} TIMING_M=${TIMING_M:-} GRAPHS=${GRAPHS:-} STAMPS=${STAMPS:-0} PASSES=${PASSES:-}
STAMPS_SHAPES=${STAMPS_SHAPES:-} STAMPS_M=${STAMPS_M:-}
HOST=${HOST:-0} LANES=${LANES:-} LANES_SHAPES=${LANES_SHAPES:-512:6144 4096:6144} GPROF=${GPROF:-0}
GROUPED_PLACES=${GROUPED_PLACES:-base}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export O D PY S TIMING TILES PROFILE GROUPED SCHEDULES TIMING_M GRAPHS STAMPS STAMPS_SHAPES STAMPS_M PASSES HOST LANES
export LANES_SHAPES GPROF GROUPED_PLACES
: > "$O/rc.txt"
mkdir -p "$S/ship" && tar -C "$S/ship" -xf "$IN/ship.tar" && (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/ship/pearl_c_sm90.cubin" "$D/" && rm -rf "$D/check" && cp -r "$S/ship/check" "$D/check"
cp "$S/ship/source.txt" "$S/ship/toolchain.txt" "$S/ship/sass.json" "$S/ship/MANIFEST.sha256" "$O/"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
nsys --version > "$O/nsys.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  PEARLC_EXPECT_UUID=$GPU_LEASE_UUID SHAPES="" "$PY" "$D/run.py" "$D/check/check.json" > "$O/run.jsonl" 2> "$O/run.err"
  rc=$?; echo "run.py rc $rc" >> "$O/rc.txt"
  [ "$rc" = 0 ] || exit "$rc"
  if [ -n "$TIMING" ]; then
    t=timing${GRAPHS:+-graphs}
    # shellcheck disable=SC2086
    "$PY" "$D/glm53_timing.py" --out "$O/$t" --shapes $TIMING --schedule $SCHEDULES ${TIMING_M:+--m $TIMING_M} ${GRAPHS:+--graphs} \
      > "$O/$t.log" 2>&1
    echo "$t rc $?" >> "$O/rc.txt"
  fi
  if [ "$STAMPS" = 1 ]; then
    # shellcheck disable=SC2086
    "$PY" "$D/glm53_profile.py" stamps --out "$O/stamps.json" ${STAMPS_SHAPES:+--shapes $STAMPS_SHAPES} \
      ${STAMPS_M:+--m $STAMPS_M} > "$O/stamps.log" 2>&1
    echo "stamps rc $?" >> "$O/rc.txt"
  fi
  if [ "$HOST" = 1 ]; then
    for g in "" ${GRAPHS:+-graphs}; do
      # shellcheck disable=SC2086
      "$PY" "$D/glm53_profile.py" host --out "$O/host$g.json" --schedule $SCHEDULES ${g:+--graphs} > "$O/host$g.log" 2>&1
      echo "host$g rc $?" >> "$O/rc.txt"
    done
  fi
  for sc in $LANES; do
    mkdir -p "$O/lanes/$sc"
    # shellcheck disable=SC2086
    nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite true -o "$O/lanes/$sc/prof" \
      "$PY" "$D/glm53_profile.py" run --schedule "$sc" ${GRAPHS:+--graphs} --calls 12 --shapes $LANES_SHAPES --m 16 64 \
      > "$O/lanes/$sc/run.log" 2>&1
    echo "lanes $sc rc $?" >> "$O/rc.txt"
  done
  if [ -n "$PASSES" ]; then
    "$PY" "$D/glm53_passes.py" capture --out "$S/passes" --schedule "$PASSES" ${GRAPHS:+--graphs} > "$O/capture.log" 2>&1
    echo "capture $PASSES rc $?" >> "$O/rc.txt"
  fi
  for sh in $TILES; do
    "$PY" "$D/glm53_passes.py" capture --out "$S/tiles/${sh/:/x}" --shapes "$sh" > "$O/tiles-${sh/:/x}.log" 2>&1
    echo "tiles ${sh/:/x} rc $?" >> "$O/rc.txt"
  done
  if [ "$PROFILE" = 1 ]; then
    mkdir -p "$O/profile"
    nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite true -o "$O/profile/prof" "$PY" "$D/glm53_profile.py" run > "$O/profile/run.log" 2>&1
    echo "profile rc $?" >> "$O/rc.txt"
  fi
  for pl in $GROUPED_PLACES; do
    x=${pl#base}; x=${x:+-$x}
    if [ "$GROUPED" = 1 ]; then
      PEARLC_GROUPED_PLACE=$pl "$PY" "$D/grouped_check.py" --out "$O/grouped$x" --passes "$S/grouped$x" > "$O/grouped$x.log" 2>&1
      echo "grouped$x rc $?" >> "$O/rc.txt"
    fi
    if [ "$GPROF" = 1 ]; then
      mkdir -p "$O/gprof$x"
      if [ -z "$x" ]; then
        "$PY" "$D/glm53_profile.py" grouped --graphs --out "$O/gprof/host.json" > "$O/gprof/host.log" 2>&1
        echo "gprof host rc $?" >> "$O/rc.txt"
      fi
      PEARLC_GROUPED_PLACE=$pl nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite true \
        -o "$O/gprof$x/prof" "$PY" "$D/glm53_profile.py" grouped --graphs --nvtx --calls 12 > "$O/gprof$x/run.log" 2>&1
      echo "gprof$x nsys rc $?" >> "$O/rc.txt"
    fi
  done'
echo "lease rc $?" >> "$O/rc.txt"
if [ -f "$S/passes/capture.json" ]; then
  nice -n 19 "$PY" "$D/glm53_passes.py" verify --out "$S/passes" --fast --jobs "${FULL_JOBS:-8}" > "$O/verify.log" 2>&1
  echo "verify rc $?" >> "$O/rc.txt"
  # shellcheck disable=SC2086
  nice -n 19 "$PY" "$D/glm53_passes.py" full --out "$S/passes" --jobs "${FULL_JOBS:-8}" --passes ${FULL_PASSES:-decode} > "$O/full.log" 2>&1
  echo "full rc $?" >> "$O/rc.txt"
  mkdir -p "$O/passes" && cp "$S"/passes/*.json "$O/passes/"
  for m in "$S"/passes/*/manifest.json; do
    p="$O/passes/$(basename "$(dirname "$m")")"; mkdir -p "$p" && cp "$m" "$p/"
  done
fi
for c in "$S"/tiles/*/capture.json; do
  [ -f "$c" ] || continue
  t=$(dirname "$c"); sh=$(basename "$t")
  "$PY" "$D/glm53_passes.py" verify --out "$t" --jobs "$(nproc)" > "$O/tiles-verify-$sh.log" 2>&1
  echo "tiles verify $sh rc $?" >> "$O/rc.txt"
  mkdir -p "$O/tiles/$sh"
  cp "$t"/*.json "$O/tiles/$sh/"
  for m in "$t"/*/manifest.json; do
    p="$O/tiles/$sh/$(basename "$(dirname "$m")")"; mkdir -p "$p" && cp "$m" "$p/"
  done
done
if [ -f "$O/profile/prof.nsys-rep" ]; then
  nsys export -t sqlite --force-overwrite true -o "$S/prof.sqlite" "$O/profile/prof.nsys-rep" > "$O/profile/export.log" 2>&1 &&
    "$PY" "$D/glm53_profile.py" report "$S/prof.sqlite" --out "$O/profile/profile.json" > "$O/profile/report.txt" 2>&1
  echo "profile report rc $?" >> "$O/rc.txt"
fi
for r in "$O"/lanes/*/prof.nsys-rep; do
  [ -f "$r" ] || continue
  l=$(dirname "$r"); sc=$(basename "$l")
  nice -n 19 nsys export -t sqlite --force-overwrite true -o "$S/lanes-$sc.sqlite" "$r" > "$l/export.log" 2>&1 &&
    nice -n 19 "$PY" "$D/glm53_profile.py" lanes "$S/lanes-$sc.sqlite" --out "$l/lanes.json" > "$l/lanes.txt" 2>&1
  echo "lanes report $sc rc $?" >> "$O/rc.txt"
done
for r in "$O"/gprof*/prof.nsys-rep; do
  [ -f "$r" ] || continue
  g=$(dirname "$r"); x=$(basename "$g")
  nice -n 19 nsys export -t sqlite --force-overwrite true -o "$S/$x.sqlite" "$r" > "$g/export.log" 2>&1 &&
    nice -n 19 "$PY" "$D/glm53_profile.py" lanes "$S/$x.sqlite" --out "$g/lanes.json" > "$g/lanes.txt" 2>&1
  echo "$x report rc $?" >> "$O/rc.txt"
done
for c in "$S"/grouped*/*/capture.json; do
  [ -f "$c" ] || continue
  t=$(dirname "$c"); sh=$(basename "$t"); x=$(basename "$(dirname "$t")")
  "$PY" "$D/glm53_passes.py" verify --out "$t" --fast --jobs "$(nproc)" > "$O/$x-verify-$sh.log" 2>&1
  echo "$x verify $sh rc $?" >> "$O/rc.txt"
  cp "$t/verify.json" "$O/$x/$sh/verify.json" 2>/dev/null
done
cat "$O/rc.txt"
tail -3 "$O/run.err" 2>/dev/null
grep -E "slowdown|no stock" "$O"/timing*.log 2>/dev/null
tail -n 40 "$O/stamps.log" "$O"/host*.log "$O/capture.log" "$O/verify.log" "$O/full.log" 2>/dev/null
tail -n 8 "$O"/tiles-*.log "$O"/grouped*.log 2>/dev/null
head -60 "$O/profile/report.txt" 2>/dev/null
grep -hE "period|^  [a-z]" "$O"/lanes/*/lanes.txt "$O"/gprof*/lanes.txt 2>/dev/null
tail -n 40 "$O/gprof/host.log" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
