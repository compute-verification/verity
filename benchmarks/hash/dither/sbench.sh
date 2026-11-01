#!/usr/bin/env bash
# Dither's served standard number (sbench.py) on one leased GPU for at most SBENCH_MAX_MIN (40) minutes: libh1c.so, libh1f.so
# and libh1o.so built for sm_$SBENCH_ARCH first (the dither arms'; 120a, else 100a on the B200 or 90a on the H200, every
# library below and FlashInfer's JIT built for it).  Extra arguments go to sbench.py (e.g. --arms exact,dither --gate --ops decode
# --t 64); SBENCH_SCRIPT runs another script with the same libraries and environment (h1ogate.py: the units kernels at
# kernel level).  DITHER_NSYS=1 runs it under nsys, recording the kernels inside the gate's CUDA profiler range (each graph's
# kernels as nodes), then the dither worker's kernels inside the gate's NVTX range (kernels.py over the SQLite export).
# DITHER_NSYS=prof captures the prof op's range instead (the gate unprofiled): steps.py's and stepprof.py's step by part.
# An exact-fused or -xl arm runs only after exactlin's kgate.py passes in the same lease (SBENCH_KGATE=0|1 overrides).
# DITHER_NCU=REGEX runs it under ncu instead (--clock-control none): the first DITHER_NCU_COUNT (16) launches of kernels
# matching REGEX, their throughput, occupancy, scheduler, warp-state and instruction sections, as ncu.ncu-rep and ncu.csv.
# Inside the lease everything runs on the leased GPU's NUMA node (numabind.py: that node's CPUs among the run's, and its
# memory), or unbound with DITHER_NUMA=0 or without numactl or a node; numa.json, the `numa:` line and summary.json's "numa"
# say which.
# On an RTX PRO 6000 under a GPU lease (SBENCH_ARCH=100a on a B200, 90a on an H200):
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/sbench.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
# the run's scratch, under SBENCH_SCRATCH_ROOT (/workspace/pouw, else, where that can't be made, as on the CPU nodes,
# the system's temporary directory)
SR=${SBENCH_SCRATCH_ROOT:-/workspace/pouw}
if ! { mkdir -p "$SR" && S=$(mktemp -d "$SR/tmp-dither-sbench.XXXXXX"); } 2>/dev/null; then
  [ -n "${SBENCH_SCRATCH_ROOT:-}" ] && { echo "sbench: NO_SCRATCH under $SBENCH_SCRATCH_ROOT"; exit 2; }
  S=$(mktemp -d -t tmp-dither-sbench.XXXXXX) || { echo "sbench: NO_SCRATCH"; exit 2; }
fi
trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA/bin:$PATH
ARCH=${SBENCH_ARCH:-120a}
case $ARCH in
  120a) FI_ARCH=12.0f ;;
  100a) FI_ARCH=10.0a ;;
  90a) FI_ARCH=9.0a ;;
  *) echo "sbench: SBENCH_ARCH $ARCH is not 120a, 100a or 90a"; exit 2 ;;
esac
GEN="arch=compute_$ARCH,code=sm_$ARCH"
echo "sbench: sm_$ARCH" | tee "$OUT/arch.txt"
# SBENCH_SHIP_OUT=1 builds the libraries below and the coins' Lean verifier, writes them to out/sbench-ship.tar and exits;
# SBENCH_SHIP=TAR (such a tar, copied to the host) takes them in place of building, so a host that may not build builds
# nothing.  A ship is refused (exit 17) unless its key is this run's (what the builds read: the CUDA
# sources here, sha512_rows' sha512x.cuh and the coins' Lean packages' trees; the arms that pick builds; SBENCH_ARCH and the flags), its
# MANIFEST.sha256 holds and lake build --no-build then finds the coins' verifier built, so one ship serves every commit
# that leaves those alone.  Neither covers --bw-passes.
COINS_PKG=verity/core/protocols/pouw/lean/check
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
SHIP_LIBS="HB_H1U HB_H1U_KSPLIT HB_H1U_KRONLY HB_H1U_KC HB_H1U_KC_B DITHER_H1C_KC"
# the coins' package and every path package its manifest lists (the inherited ones too), from the tree's root
COINS_PKGS=$(python3 -c 'import json, os, sys; p = sys.argv[1]; print(p, *(os.path.normpath(os.path.join(p, x["dir"]))
  for x in json.load(open(os.path.join(p, "lake-manifest.json")))["packages"] if x["type"] == "path"))' "$COINS_PKG")
ship_key() {
  git ls-tree -r HEAD -- benchmarks/hash/dither benchmarks/torch_gpu/sha512_rows/sha512x.cuh | grep -E '\.(cu|cuh)$'
  for p in $COINS_PKGS; do echo "lean $p $(git rev-parse "HEAD:$p")"; done
  echo "arms $(for t in dither-a-kh dither-a-hbk dither-a-hbr dither-a-hbc; do case " $* " in *$t*) printf '%s ' $t ;; esac; done)"
  echo "CUDA=$CUDA ARCH=$ARCH HB_KC_FLAGS=${HB_KC_FLAGS:-} HB_KC_H1C_FLAGS=${HB_KC_H1C_FLAGS:-}${HB_KC_FLAGS_B:+ HB_KC_FLAGS_B=$HB_KC_FLAGS_B}"
}
if [ -n "${SBENCH_SHIP:-}${SBENCH_SHIP_OUT:-}" ]; then
  case " $* " in *" --bw-passes"*) echo "sbench: SHIP_STALE: a ship doesn't cover --bw-passes"; exit 17 ;; esac
  ship_key "$@" > "$S/key"
fi
if [ -n "${SBENCH_SHIP:-}" ]; then
  tar -xf "$SBENCH_SHIP" -C "$S" || { echo "sbench: SHIP_STALE: can't unpack $SBENCH_SHIP"; exit 17; }
  cmp -s "$S/key" "$S/ship/key" \
    || { echo "sbench: SHIP_STALE: its key isn't this run's (< this run, > the ship):"; diff "$S/key" "$S/ship/key" | head; exit 17; }
  (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256) || { echo "sbench: SHIP_STALE: MANIFEST.sha256 doesn't hold"; exit 17; }
  cp "$S"/ship/*.so "$S"/ && cp "$S"/ship/*.log "$S"/ship/*.txt "$OUT"/ 2>/dev/null
  while read -r name f; do export "$name=$S/$f"; done < "$S/ship/libs"
  (cd "$COINS_PKG" && mkdir -p .lake && flock .lake/build.lock tar -xf "$S/ship/lean.tar" -C "$ROOT" \
    && lake build --no-build pouw-check-difftest) > "$OUT/lean-coins-build.log" 2>&1 \
    || { echo "sbench: SHIP_STALE: lake build --no-build finds the coins' verifier unbuilt"; tail -5 "$OUT/lean-coins-build.log"
         exit 17; }
  echo "sbench: shipped ($(tail -1 "$S/key"))"
else
for lib in h1c h1f h1o; do
  nvcc -O3 -std=c++17 -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
    -o "$S/lib$lib.so" "benchmarks/hash/dither/$lib.cu" > "$OUT/build-$lib.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL $lib"; grep -v "^ptxas info" "$OUT/build-$lib.log" | tail -30; exit 11; }
done
grep -B1 -A2 "h1f_k\|h1r_units" "$OUT/build-h1f.log" | grep -i "registers\|spill" | head -6
grep -B1 -A2 "h1o_units" "$OUT/build-h1o.log" | grep -i "registers\|spill" | head -8
grep -B1 -A2 "h1c_a_" "$OUT/build-h1c.log" | grep -i "registers\|spill" | head -6
# dither-a-kh's side unit: h1u.cu's kh_unit_ws, built here (HB_H1U)
case " $* " in *dither-a-kh*)
  nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
    -o "$S/h1u.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-h1u.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL h1u"; grep -v "^ptxas info" "$OUT/build-h1u.log" | tail -30; exit 11; }
  grep -B1 -A2 "h1u_ws\|h1s_noise" "$OUT/build-h1u.log" | grep -i "registers\|spill" | head -6
  export HB_H1U=$S/h1u.so ;;
esac
# dither-a-hbk's: the K split's kh_one (H1U_KSPLIT = 4, H1K_AUTO), as job.sh's ka variant (HB_H1U_KSPLIT)
case " $* " in *dither-a-hbk*)
  nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
    -DH1U_KSPLIT=4 -DH1K_AUTO=1 -o "$S/h1u_ka.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-h1u_ka.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL h1u_ka"; grep -v "^ptxas info" "$OUT/build-h1u_ka.log" | tail -30; exit 11; }
  grep -B1 -A2 "h1k" "$OUT/build-h1u_ka.log" | grep -i "registers\|spill" | head -6
  export HB_H1U_KSPLIT=$S/h1u_ka.so ;;
esac
# dither-a-hbr's: the same K split built receipts only (H1K_RONLY), as job.sh's kr variant (HB_H1U_KRONLY)
case " $* " in *dither-a-hbr*)
  nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
    -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 -o "$S/h1u_kr.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-h1u_kr.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL h1u_kr"; grep -v "^ptxas info" "$OUT/build-h1u_kr.log" | tail -30; exit 11; }
  grep -B1 -A2 "h1k" "$OUT/build-h1u_kr.log" | grep -i "registers\|spill" | head -6
  export HB_H1U_KRONLY=$S/h1u_kr.so ;;
esac
# dither-a-hbc's: that build with HB_KC_FLAGS too (h1u.cu's knobs, space-separated), as a job.sh kr* variant (HB_H1U_KC);
# J10-R's units on part of the SMs (sbench.py's ARM-smqQoD) take -DH1U_JCTA=1 -DH1U_WARPS=4 in it, after any other
# H1U_WARPS (the last -D holds)
case " $* " in *dither-a-hbc*)
  read -r -a kc <<< "${HB_KC_FLAGS:-}"
  nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
    -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kc[@]}" -o "$S/h1u_kc.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-h1u_kc.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL h1u_kc"; grep -v "^ptxas info" "$OUT/build-h1u_kc.log" | tail -30; exit 11; }
  echo "sbench: h1u_kc.so with ${HB_KC_FLAGS:-no flags}" | tee "$OUT/kc-flags.txt"
  grep -B1 -A2 "h1k\|h1u" "$OUT/build-h1u_kc.log" | grep -i "registers\|spill" | head -8
  export HB_H1U_KC=$S/h1u_kc.so
  # and, with HB_KC_FLAGS_B, a second such build for the ARM-kcb arms (HB_H1U_KC_B)
  read -r -a kcb <<< "${HB_KC_FLAGS_B:-}"
  if [ ${#kcb[@]} -gt 0 ]; then
    nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared \
      -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kcb[@]}" -o "$S/h1u_kcb.so" benchmarks/hash/dither/h1u.cu > "$OUT/build-h1u_kcb.log" 2>&1 \
      || { echo "sbench: NVCC_FAIL h1u_kcb"; grep -v "^ptxas info" "$OUT/build-h1u_kcb.log" | tail -30; exit 11; }
    echo "sbench: h1u_kcb.so with ${HB_KC_FLAGS_B}" | tee -a "$OUT/kc-flags.txt"
    grep -B1 -A2 "h1um\|h1jr_rows" "$OUT/build-h1u_kcb.log" | grep -i "registers\|spill" | head -8
    export HB_H1U_KC_B=$S/h1u_kcb.so
  fi
  # and, with HB_KC_H1C_FLAGS (h1c.cu's knobs), its own libh1c.so: its A, z, y and receipt tables (DITHER_H1C_KC)
  read -r -a kh <<< "${HB_KC_H1C_FLAGS:-}"
  if [ ${#kh[@]} -gt 0 ]; then
    nvcc -O3 -std=c++17 -gencode "$GEN" -Xptxas -v -Xcompiler -fPIC -shared "${kh[@]}" \
      -o "$S/libh1c_kc.so" benchmarks/hash/dither/h1c.cu > "$OUT/build-h1c_kc.log" 2>&1 \
      || { echo "sbench: NVCC_FAIL h1c_kc"; grep -v "^ptxas info" "$OUT/build-h1c_kc.log" | tail -30; exit 11; }
    echo "sbench: libh1c_kc.so with ${HB_KC_H1C_FLAGS}" | tee -a "$OUT/kc-flags.txt"
    export DITHER_H1C_KC=$S/libh1c_kc.so
  fi ;;
esac
# --bw-passes (dbw.py): PoUB's prover libraries, as remote_session/run.sh builds them (BW_LIB, BW_KV)
case " $* " in *" --bw-passes"*)
  nvcc -O3 -gencode "$GEN" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v \
    -o "$S/session.so" benchmarks/remote_session/session.cu > "$OUT/build-session.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL session"; grep -v "^ptxas info" "$OUT/build-session.log" | tail -30; exit 11; }
  nvcc -O3 -gencode "$GEN" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DKV_FIELD=1024 \
    -DKV_MASK=1 -o "$S/kv.so" benchmarks/word_codec/kv.cu > "$OUT/build-kv.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL kv"; grep -v "^ptxas info" "$OUT/build-kv.log" | tail -30; exit 11; }
  export BW_LIB=$S/session.so BW_KV=$S/kv.so ;;
esac
fi
if [ "${SBENCH_SHIP_OUT:-}" = 1 ]; then
  (cd "$COINS_PKG" && mkdir -p .lake && flock .lake/build.lock lake build pouw-check-difftest) > "$OUT/lean-coins-build.log" 2>&1 \
    || { echo "sbench: LEAN_BUILD_FAIL"; tail -20 "$OUT/lean-coins-build.log"; exit 13; }
  mkdir -p "$S/ship"
  for name in $SHIP_LIBS; do [ -n "${!name:-}" ] && echo "$name $(basename "${!name}")"; done > "$S/ship/libs"
  cp "$S"/*.so "$OUT"/build-*.log "$S/ship/" && { [ ! -f "$OUT/kc-flags.txt" ] || cp "$OUT/kc-flags.txt" "$S/ship/"; }
  cp "$S/key" "$S/ship/key"
  tar -cf "$S/ship/lean.tar" $(for p in $COINS_PKGS; do [ ! -d "$p/.lake/build" ] || echo "$p/.lake/build"; done) \
    || { echo "sbench: SHIP_FAIL lean.tar"; exit 17; }
  (cd "$S/ship" && sha256sum -- * > "$S/MANIFEST.sha256" && mv "$S/MANIFEST.sha256" .)
  tar -cf "$OUT/sbench-ship.tar" -C "$S" ship
  echo "sbench: ship $OUT/sbench-ship.tar ($(du -h "$OUT/sbench-ship.tar" | cut -f1)), key:"; cat "$S/key"
  exit 0
fi
[ -n "${BUILD_ONLY:-}" ] && { echo "sbench: built (BUILD_ONLY)"; exit 0; }
# the dither windows' coins are the Lean verifier of record's (pouw.verifier.coins): built before the lease, not inside it,
# under a lock, since runs of one commit share its source tree and concurrent lake builds there corrupt each other's .c/.o
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
(cd verity/core/protocols/pouw/lean/check && mkdir -p .lake && flock .lake/build.lock lake build pouw-check-difftest) \
  > "$OUT/lean-coins-build.log" 2>&1 \
  || { echo "sbench: LEAN_BUILD_FAIL"; tail -20 "$OUT/lean-coins-build.log"; exit 13; }
# and with --bw-passes PoUB's verifier of record, under the lock lean_verifier.build takes (BW_LEAN_EXE);
# BW_PY runs the audit's side (verifier and prover), by default remote_session/run.sh's torch venv
if [ -n "${BW_LIB:-}" ]; then
  (mkdir -p verity/core/difftest/lean/.lake && cd verity/core/protocols/remote_bandwidth/lean \
    && flock "$ROOT/verity/core/difftest/lean/.lake/harness-build.lock" lake build remote-audit-verifier) \
    > "$OUT/lean-bw-build.log" 2>&1 || { echo "sbench: LEAN_BUILD_FAIL bw"; tail -20 "$OUT/lean-bw-build.log"; exit 13; }
  export BW_LEAN_EXE=$ROOT/verity/core/protocols/remote_bandwidth/lean/.lake/build/bin/remote-audit-verifier
  export BW_PY=${BW_PY:-/workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python}
  [ -x "$BW_PY" ] || { echo "sbench: no BW_PY at $BW_PY"; exit 3; }
fi
# coins off the node (sbench.py's "Coins"): every port the arms' windows connect to (POUW_CHALLENGER, DITHER_CHALLENGERS'
# {arm: {"session": ID, "address": HOST:PORT, "secret_file": F, "commitment": HEX, "rounds": R}}), each its own session
# from the challenger's gate first (`coins.py open`: reserved in a dev period, opened on its stand-in root, its secret in F on
# this node, HEX the root of the coin commitment its challenger signed, which its hello must answer, R its --rounds when
# not 16384), waited for with ss -ltn
# before the lease (never a probe connect: a challenger serves the first connection whose hello it accepts); none: in-process
# (not certified)
if [ -n "${POUW_CHALLENGER:-}${POUW_CHALLENGER_SESSION:-}" ] && { [ -z "${POUW_CHALLENGER:-}" ] || [ -z "${POUW_CHALLENGER_SESSION:-}" ]; }; then
  echo "sbench: NO_CHALLENGER: a challenger session is POUW_CHALLENGER and POUW_CHALLENGER_SESSION together"; exit 15
fi
ats=$(python3 -c 'import json, os; m = json.loads(os.environ.get("DITHER_CHALLENGERS") or "{}"); ok = isinstance(m, dict) and all(isinstance(v, dict) and {"address", "session"} <= set(v) <= {"address", "commitment", "rounds", "secret_file", "session"} and all(isinstance(x, str) and x for k, x in v.items() if k != "rounds") and type(v.get("rounds", 1)) is int and v.get("rounds", 1) > 0 for v in m.values()); print(*(v["address"] for v in m.values())) if ok else exit(1)') \
  || { echo "sbench: NO_CHALLENGER: DITHER_CHALLENGERS is not {arm: {\"session\": ID, \"address\": HOST:PORT[, \"secret_file\": F][, \"commitment\": HEX][, \"rounds\": R]}}"; exit 15; }
ats=$(echo ${POUW_CHALLENGER:-} $ats)
for at in $ats; do
  for _ in $(seq $((${CHALLENGER_WAIT_S:-900} * 20))); do ss -Hltn "sport = :${at##*:}" | grep -q . && break; sleep 0.05; done
  ss -Hltn "sport = :${at##*:}" | grep -q . || { echo "sbench: NO_CHALLENGER at $at"; exit 15; }
done
echo "sbench: coins ${ats:-in-process (not certified)}" | tee "$OUT/coins.txt"
export DITHER_H1C=$S/libh1c.so DITHER_H1F=$S/libh1f.so DITHER_H1O=$S/libh1o.so VLLM_DISABLE_COMPILE_CACHE=1
# SBENCH_PICK pins Inductor's autotune picks (picks.py); the node's shared Inductor cache would hand back another run's
# timed .best_config, so a pinned run compiles into its own
[ -n "${SBENCH_PICK:-}" ] && export TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache
# the credit gate at its pinned head, for the dither-1s-credit arm
CREDIT_GATE_SHA=${CREDIT_GATE_SHA:-b25b34d96e}
git show "$CREDIT_GATE_SHA:integrations/vllm/verity_vllm/protocols/pouw/credit_gate.py" > "$S/credit_gate.py" 2>/dev/null \
  && export DITHER_CREDIT_GATE=$S/credit_gate.py || echo "sbench: no credit_gate.py at $CREDIT_GATE_SHA"
# H1V3 (with the wrap correction and the peel, z = A B^T): dither_one's gate runs its h1tile.reference.  By default
# this source's own (PouwH1Tile_v4, ungated); H1V3_SHA names another commit, from this repository, else (the node's lacks
# the commit) the archive `git archive --format=tar.gz $H1V3_SHA verity > h1v3-$H1V3_SHA.tar.gz` makes, in
# $RESEARCH_RUN_DIR/inputs
if [ -n "${H1V3_SHA:-}" ]; then
  mkdir -p "$S/h1v3"
  if git cat-file -e "$H1V3_SHA^{commit}" 2>/dev/null; then
    git archive "$H1V3_SHA" verity | tar -x -C "$S/h1v3"
  elif [ -f "$D/inputs/h1v3-$H1V3_SHA.tar.gz" ]; then
    tar -xzf "$D/inputs/h1v3-$H1V3_SHA.tar.gz" -C "$S/h1v3"
  fi
  [ -f "$S/h1v3/verity/core/protocols/pouw/circuit/h1tile.py" ] || { echo "sbench: H1V3_NO_TREE at $H1V3_SHA"; exit 14; }
  export DITHER_H1V3_TREE=$S/h1v3 DITHER_H1V3_SHA=$H1V3_SHA
else
  export DITHER_H1V3_TREE=$ROOT DITHER_H1V3_SHA=${RESEARCH_SOURCE_SHA:-source}
fi
# J10R's Definition (h1tile.j10r_reference) for DITHER_SO_UNITS=j10r's gate (j10r_so.py): J10R_SHA's verity/, as H1V3_SHA's
# (the archive: j10r-$J10R_SHA.tar.gz)
if [ -n "${J10R_SHA:-}" ]; then
  mkdir -p "$S/j10r"
  if git cat-file -e "$J10R_SHA^{commit}" 2>/dev/null; then
    git archive "$J10R_SHA" verity | tar -x -C "$S/j10r"
  elif [ -f "$D/inputs/j10r-$J10R_SHA.tar.gz" ]; then
    tar -xzf "$D/inputs/j10r-$J10R_SHA.tar.gz" -C "$S/j10r"
  fi
  [ -f "$S/j10r/verity/core/protocols/pouw/circuit/h1tile.py" ] || { echo "sbench: J10R_NO_TREE at $J10R_SHA"; exit 14; }
  export DITHER_J10R_TREE=$S/j10r DITHER_J10R_SHA=$J10R_SHA
fi
# the per-call challenger round's measured round trips (paced samples in percall-wait-ns.txt, sent with the run) for the dither-1s-percall-rtt arm
[ -f "$D/inputs/percall-wait-ns.txt" ] && export DITHER_PERCALL_WAITS=$D/inputs/percall-wait-ns.txt
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PATH=$(dirname "$PY"):$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=$FI_ARCH
# the wrap correction's pieces against H1V3 on the CPU, before the lease
"$PY" "$ROOT/benchmarks/hash/dither/wrapcheck.py" | tee "$OUT/wrapcheck.json"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "sbench: WRAPCHECK_FAIL"; exit 12; }
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ] || [ "${DITHER_NSYS:-}" = prof ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
elif [ -n "${DITHER_NCU:-}" ]; then
  NSYS=("$CUDA/bin/ncu" --clock-control none -k "regex:$DITHER_NCU" --launch-count "${DITHER_NCU_COUNT:-16}"
        --section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats --section WarpStateStats
        --section ComputeWorkloadAnalysis --section InstructionStats -o "$OUT/ncu" -f)
fi
# an exact-fused or -xl arm is timed only after exactlin's kgate.py passes on this GPU, in the same lease (OUT/kgate)
case " $* " in *exact-fused*|*-xl[\ ,-]*) KGATE=1 ;; *) KGATE=0 ;; esac
export KGATE=${SBENCH_KGATE:-$KGATE} KGATE_PY=$PY KGATE_SCRIPT=$ROOT/benchmarks/hash/exactlin/kgate.py
# SBENCH_LEASE_AT (a UTC time) holds the lease request until then, the builds done, so a timed lease (GPU_LEASE_TIMED=1)
# starts inside its clean window; SBENCH_ON (e.g. 4,5,6,7) is the GPUs it may take
if [ -n "${SBENCH_LEASE_AT:-}" ]; then
  at=$(date -u -d "$SBENCH_LEASE_AT" +%s) || { echo "sbench: SBENCH_LEASE_AT $SBENCH_LEASE_AT is not a time"; exit 2; }
  now=$(date -u +%s)
  [ "$at" -gt "$now" ] && { echo "sbench: holding the lease request until $SBENCH_LEASE_AT"; sleep $((at - now)); }
fi
if [ -n "${SBENCH_ON:-}" ]; then
  gpu-lease() { command gpu-lease "$1" --on "$SBENCH_ON" "${@:2}"; }
fi
echo "sbench: asking for one GPU at $(date -u +%H:%M:%SZ)${SBENCH_ON:+ on $SBENCH_ON}${GPU_LEASE_TIMED:+ (timed)}"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -l 2 \
  > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
# SBENCH_MPS=1: in the lease, after the kgate, an MPS control daemon of the run's own for the leased GPU (its pipe and
# logs under the scratch, so no other process finds it), every process of the run its client, which names the GPU as
# device 0 within the server's set; quit when the run ends (out/mps-*.log)
export SBENCH_MPS=${SBENCH_MPS:-} MPS_DIR=$S/mps
MAX_MIN=${SBENCH_MAX_MIN:-40}
# SBENCH_PREEMPTIBLE=1: the lease may be taken back by a non-preemptible one (which ends the run unfinished)
PRE=(); [ "${SBENCH_PREEMPTIBLE:-}" = 1 ] && PRE=(--preemptible)
# SBENCH_TIMED=1: a timed lease (gpu-lease --timed), the one a booked window grants, the node kept quiet while it's held
[ "${SBENCH_TIMED:-}" = 1 ] && PRE+=(--timed)
# SBENCH_MEM_GB=G: the host memory the lease reserves (gpu-lease --mem-gb), else gpu-lease's default
[ -n "${SBENCH_MEM_GB:-}" ] && PRE+=(--mem-gb "$SBENCH_MEM_GB")
gpu-lease 1 --wait "${PRE[@]}" --max-min "$MAX_MIN" -- "$PY" "$ROOT/benchmarks/hash/dither/numabind.py" "$OUT/numa.json" \
  bash -c '
  if [ "$KGATE" = 1 ]; then
    timeout 900 "$KGATE_PY" "$KGATE_SCRIPT" --out "$0/kgate" --shapes all --models llama-3.1-8b,qwen3-8b --no-stock \
      > "$0/kgate.log" 2>&1
    g=$?
    tail -1 "$0/kgate.log"
    if [ $g -ne 0 ]; then echo "sbench: kgate rc $g, nothing timed"; exit 4; fi
  fi
  if [ "$SBENCH_MPS" = 1 ]; then
    mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
    export CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPS_DIR/log
    CUDA_VISIBLE_DEVICES=${GPU_LEASE_UUID:?} nvidia-cuda-mps-control -d || { echo "sbench: MPS_FAIL"; exit 16; }
    export CUDA_VISIBLE_DEVICES=0
    echo "sbench: MPS on $GPU_LEASE_UUID"
  fi
  [ "$SBENCH_MPS" = 1 ] || exec timeout $((60 * '"$MAX_MIN"' - 50 - SECONDS)) "$@"
  timeout $((60 * '"$MAX_MIN"' - 50 - SECONDS)) "$@"; rc=$?
  echo quit | nvidia-cuda-mps-control
  for f in "$MPS_DIR"/log/*.log; do cp "$f" "$0/mps-$(basename "$f")"; done
  exit $rc' "$OUT" "${NSYS[@]}" "$PY" \
  "$ROOT/benchmarks/hash/dither/${SBENCH_SCRIPT:-sbench.py}" --out "$OUT" --scratch "$S" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "sbench: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" stats --report cuda_gpu_kern_sum --format csv -o "$OUT/gate" "$S/gate.nsys-rep" > "$OUT/nsys-stats.log" 2>&1 \
    || echo "sbench: nsys stats rc $?"
  "$PY" "$ROOT/benchmarks/hash/dither/kernels.py" "$S/gate.sqlite" | tee "$OUT/kernels.json"
  # the prof op's ranges: each arm's kernels a decode step (exactlin/steps.py); the report kept when under 200 MB
  "$PY" "$ROOT/benchmarks/hash/exactlin/steps.py" "$S/gate.sqlite" > "$OUT/steps.json" || echo "sbench: steps.py rc $?"
  "$PY" "$ROOT/benchmarks/hash/dither/stepprof.py" "$S/gate.sqlite" > "$OUT/stepprof.json" \
    && "$PY" "$ROOT/benchmarks/hash/dither/stepprof.py" --table "$OUT/stepprof.json" | tee "$OUT/stepprof.txt" \
    || echo "sbench: stepprof.py rc $?"
  [ "$(stat -c %s "$S/gate.nsys-rep")" -lt 209715200 ] && cp "$S/gate.nsys-rep" "$OUT/"
fi
if [ -f "$OUT/ncu.ncu-rep" ]; then
  "$CUDA/bin/ncu" --import "$OUT/ncu.ncu-rep" --csv --page details > "$OUT/ncu.csv" 2> "$OUT/ncu-import.log" \
    || echo "sbench: ncu import rc $?"
fi
exit "$rc"
