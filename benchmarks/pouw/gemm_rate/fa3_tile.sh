#!/usr/bin/env bash
# Build the pinned FA3's hdim-64 forward as torch extensions (fa3_tile_build.py), one a variant, for attn_knobs.py's
# --fa3-so arms, under `research run` (cwd: the root of a source tree; no GPU):
#
#   research run --on MACHINE --project verity --source . --cwd source --timeout 60m \
#     [--env VARIANTS="base:0 m128:128"] [--env VENV=DIR] -- bash benchmarks/pouw/gemm_rate/fa3_tile.sh
#
# A variant is NAME:M[:EMU[:DEG[:LAZY]]], M the hdim-64 causal tile's kBlockM (0: the pinned 192), EMU a NUM/DEN share of
# the score columns' exp2 on the FMA pipe (0: none), DEG its polynomial's degree, and LAZY 1 for a kept running max or
# one of --lazy-max's modes (checked, thresh, held, held-redo) (fa3_tile_build.py's --exp2-emu, --exp2-deg, --lazy-max).
# KFA3_CACHE (kept outside the run dir) holds vllm-flash-attn at 506341a1 with cutlass, a tree a variant with
# apply_fa3_tap.py applied (built with VERITY_TAP=0), and its build: KFA3_CACHE/build/NAME/kfa3_NAME.so. VENV is a Python
# with torch (attn_knobs.sh's). $OUT gets each build's log, build_info.json and sass-NAME.txt (each kernel's MUFU.EX2,
# FFMA, FMNMX, SHFL, LOP3 and VOTE counts), and rc.txt.
set -uo pipefail
O=${OUT:?}
TREE=$PWD
C=${KFA3_CACHE:-/workspace/cache/kernels-kfa3}
VENV=${VENV:-$C/venv}
export CUDA_HOME=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH
PIN=506341a143fcabd4bb79052a7605ada727d6b3f5
VARIANTS=${VARIANTS:-"base:0 m128:128"}
JOBS=${JOBS:-6}
FA=$C/vllm-flash-attn
TAP=$TREE/integrations/vllm/verity_vllm/acquire/fa3_tap_src
: > "$O/rc.txt"
mkdir -p "$C/trees" "$C/build"
if [ ! -d "$FA/.git" ]; then
  git clone -q https://github.com/vllm-project/flash-attention.git "$FA" > "$O/clone.log" 2>&1
fi
(cd "$FA" && git checkout -q $PIN && git submodule update --init --depth 1 csrc/cutlass) >> "$O/clone.log" 2>&1
echo "checkout rc $?" >> "$O/rc.txt"
[ "$(cd "$FA" && git rev-parse HEAD)" = "$PIN" ]
echo "pin rc $?" >> "$O/rc.txt"
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$C/bin" UV_NO_MODIFY_PATH=1 sh > "$O/uv-install.log" 2>&1
  export PATH=$C/bin:$PATH
fi
"$VENV/bin/python" -c "import ninja" 2> /dev/null || uv pip install -q --python "$VENV/bin/python" ninja > "$O/ninja.log" 2>&1
"$VENV/bin/python" -c "import ninja"
echo "ninja rc $?" >> "$O/rc.txt"
nvcc --version > "$O/nvcc.txt" 2>&1
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt" "$O/clone.log"; exit 1; }

pids=()
for v in $VARIANTS; do
  IFS=: read -r name m emu deg lazy <<< "$v"
  t=$C/trees/$name
  rm -rf "$t" && mkdir -p "$t/csrc"
  cp -r "$FA/hopper" "$t/hopper" && cp -r "$FA/csrc/common" "$t/csrc/common" && ln -s "$FA/csrc/cutlass" "$t/csrc/cutlass"
  python3 "$TAP/apply_fa3_tap.py" "$t/hopper" "$TAP" > "$O/apply-$name.log" 2>&1
  tm=(); [ "${m:-0}" != 0 ] && tm=(--tile-m "$m")
  [ -n "${emu:-}" ] && [ "$emu" != 0 ] && tm+=(--exp2-emu "$emu" --exp2-deg "${deg:-3}")
  case "${lazy:-0}" in 0) ;; 1) tm+=(--lazy-max) ;; *) tm+=(--lazy-max "$lazy") ;; esac
  ( "$VENV/bin/python" "$TREE/benchmarks/pouw/gemm_rate/fa3_tile_build.py" --tree "$t" --name "kfa3_$name" \
      --build-dir "$C/build/$name" "${tm[@]}" --jobs "$JOBS" --cuda "$CUDA_HOME" > "$O/build-$name.log" 2>&1
    echo "build $name rc $?" >> "$O/rc.txt"
    cp "$C/build/$name/build_info.json" "$O/build_info-$name.json" 2> /dev/null
    cuobjdump -sass "$C/build/$name/kfa3_$name.so" 2> /dev/null \
      | awk '/Function :/ { f = $3 } /MUFU\.EX2/ { e[f]++ } /FFMA/ { a[f]++ } /FMNMX/ { x[f]++ } /SHFL/ { h[f]++ } /LOP3/ { l[f]++ }
             /VOTE/ { o[f]++ } END { for (k in e) print k, "MUFU.EX2", e[k], "FFMA", a[k], "FMNMX", x[k]+0, "SHFL", h[k]+0, "LOP3", l[k]+0, "VOTE", o[k]+0 }' \
      | sort > "$O/sass-$name.txt" ) &
  pids+=($!)
done
wait "${pids[@]}"
research keep "$C" --owner @kernels --expires 2026-11-11 --priority normal --why "KFA3's FA3 tile builds and the served vLLM venv" \
  > "$O/keep.log" 2>&1
for f in "$O"/build-*.log; do echo "== $f"; grep -a -m3 -i -E "error|BUILD-OK" "$f"; grep -a -E "registers|smem" "$f" | tail -4; done
for f in "$O"/sass-*.txt; do echo "== $f"; cat "$f"; done
cat "$O/rc.txt"
! grep -qv ' rc 0$' "$O/rc.txt"
