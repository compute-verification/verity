#!/usr/bin/env bash
# Serve Llama 3.2 1B INT8 through the vLLM adapter from a clone of this tree, on a machine of your own, and seal what it served:
# the environment and the taps (pod_bootstrap.sh), the public base from Hugging Face with no token and the INT8 pin by the
# tree's recipe, the workload file by the store's public fetch with no key, `predict row`, the traced Build, `serve-openai`
# committing every value, one request (tools/served_run/serve_load.py), and the server stopped by its pid so it seals its window.
#
#   bash adapters/vllm/verity_vllm/ops/serve_fresh.sh [--work DIR] [--gpu N] [--dry-run]
#
# Everything it writes is under DIR (default ./.verity-work): WORK, VENV, HF_HOME, FA2, NT, RT, HOME and every cache, whatever
# the caller's environment holds; the install's uv and pip caches are the bootstrap's, under DIR and removed after the install.
# --gpu N sets CUDA_VISIBLE_DEVICES=N (default: the caller's, else the first Hopper card). --dry-run prints the paths and the
# steps and does nothing. Before building anything it refuses, in one line each, a machine with no Hopper card (the row is
# h200's: compute capability 9.0), no nvcc, no uv, or under DISK_GB (default 35) free at DIR. Where the toolkit lacks the
# CUDA math headers, the bootstrap takes them from the environment's NVIDIA wheels (CPATH). The last line is `SERVE-FRESH SERVED AND SEALED ...`
# with the window's root, or `SERVE-FRESH FAILED <step>: ...`; logs are in DIR/logs.
set -uo pipefail
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
W=$PWD/.verity-work; GPU=""; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --work) W=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "usage: serve_fresh.sh [--work DIR] [--gpu N] [--dry-run]" >&2; exit 2 ;;
  esac
done
W=$(realpath -m "$W")
ROW=llama32-1b__int8__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager
MODEL=verity-int8/Llama-3.2-1B-INT8; REV=15b0976d400d796ee589e7737d03a919526bf800
export WORK=$W VENV=$W/venv312 HF_HOME=$W/hf FA2=$W/cp/fa2 NT=$W/cp/norm_tap RT=$W/cp/router_tap HOME=$W/home
export XDG_CACHE_HOME=$W/home/.cache XDG_DATA_HOME=$W/home/.local/share UV_PYTHON_INSTALL_DIR=$W/uv-python TMPDIR=$W/tmp
export TORCH_EXTENSIONS_DIR=$W/cache/torch_extensions TRITON_CACHE_DIR=$W/cache/triton VLLM_CACHE_ROOT=$W/cache/vllm
export CUDA_CACHE_PATH=$W/cache/nv NATIVE_COLLECT_BUILD=$W/cp/nc_build FA2_TAP_HDIMS=64 FA3_TAP_HDIMS=64   # the row's head size
unset UV_CACHE_DIR PIP_CACHE_DIR
PY=$VENV/bin/python
# every workspace member's import root: the parent of each package its wheel names (the root pyproject.toml's members)
toml_list() {
  awk -v s="[$2]" -v k="$3" '/^\[/ { in_s = ($0 == s) } in_s && $0 ~ "^" k " *=" { on = 1; sub(/^[^=]*=/, "") }
    on { print; if (/\]/) on = 0 }' "$1" | tr -d '[]",'
}
ROOTS=$TREE
for m in $(toml_list "$TREE/pyproject.toml" tool.uv.workspace members); do
  for p in $(toml_list "$TREE/$m/pyproject.toml" tool.hatch.build.targets.wheel packages 2>/dev/null); do
    r=$TREE/$m/$p; r=${r%/*}
    case ":$ROOTS:" in *":$r:"*) ;; *) ROOTS=$ROOTS:$r ;; esac
  done
done
export PY PY312=$PY PYTHONPATH=$ROOTS
HSO=$FA2/build/fa3_matReqG/verity_fa3_matReqG.so; NSO=$NT/build/verity_norm_tap.so
WL=$TREE/adapters/vllm/workloads/$ROW.json; LOGS=$W/logs; PORT=${PORT:-$(( 18200 + RANDOM % 700 ))}
fail() { stop_group; echo "SERVE-FRESH FAILED $1: $2"; exit 1; }
refuse() { echo "serve_fresh: $1" >&2; echo "SERVE-FRESH FAILED check: $1"; exit 3; }

if [ -z "${CUDA_HOME:-}" ]; then for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU
if [ "$DRY" = 1 ]; then
  echo "serve_fresh --dry-run: tree $TREE"
  for v in WORK VENV HF_HOME FA2 NT RT HOME XDG_CACHE_HOME XDG_DATA_HOME UV_PYTHON_INSTALL_DIR TMPDIR TORCH_EXTENSIONS_DIR \
           TRITON_CACHE_DIR VLLM_CACHE_ROOT CUDA_CACHE_PATH NATIVE_COLLECT_BUILD; do echo "path $v=${!v}"; done
  echo "pythonpath $PYTHONPATH"
  echo "path uv and pip caches: $W/.cache/{uv,pip} (the bootstrap's; removed after the install)"
  echo "checks: a card of compute capability 9.0, nvcc under CUDA_HOME ($CUDA_HOME), uv, ${DISK_GB:-35} GB free at $W"
  [ -f "$CUDA_HOME/include/cusparse.h" ] || echo "headers: $CUDA_HOME/include lacks cusparse.h; the bootstrap takes the CUDA" \
    "math headers torch reads from the environment's NVIDIA wheels (CPATH)"
  echo "build: the FA2 and FA3 taps for head size 64 only (the row's), with MAX_JOBS=${MAX_JOBS:-$(nproc)} (the CPUs this" \
    "shell may run on); building the taps is the longest step and its time falls with those CPUs, so give it all you can"
fi

if [ "$DRY" = 0 ]; then
  command -v nvidia-smi >/dev/null || refuse "no nvidia-smi: this needs an NVIDIA Hopper card (H100 or H200) and its driver"
  cards=$(nvidia-smi --query-gpu=index,compute_cap,name --format=csv,noheader 2>/dev/null | tr -d ' ')
  want=${CUDA_VISIBLE_DEVICES:-$(echo "$cards" | awk -F, '$2=="9.0"{print $1; exit}')}; want=${want%%,*}
  echo "$cards" | awk -F, -v i="$want" '$1==i && $2=="9.0"{f=1} END{exit !f}' \
    || refuse "no Hopper card (compute capability 9.0)${want:+ at index $want}: the row is h200's; run on an H100 or H200 (--gpu N picks the card)"
  export CUDA_VISIBLE_DEVICES=$want
  [ -x "$CUDA_HOME/bin/nvcc" ] || refuse "no nvcc under CUDA_HOME=$CUDA_HOME: install the CUDA 12.9 toolkit or set CUDA_HOME to it"
  command -v uv >/dev/null || refuse "no uv: install it (https://docs.astral.sh/uv/getting-started/installation/) and rerun"
  mkdir -p "$W" || refuse "cannot make $W: pass --work DIR where you can write"
  free=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
  [ "${free:-0}" -ge "${DISK_GB:-35}" ] || refuse "${free}G free at $W, under the ${DISK_GB:-35}G this needs: free room there or pass --work DIR on a larger disk"
  mkdir -p "$HOME" "$TMPDIR" "$LOGS"
fi
# Each child runs in a process group of its own and is waited on in the background, so a signal to this script or a failure
# stops it and everything it started (the compilers, the server and its engine) rather than leaving them on the card. A server
# signalled mid-startup can take minutes to act on TERM, so the script polls, a second apart, for the whole group to be gone
# (STOP_GRACE_S polls, default 60), KILLs what is left, and only then prints its last line.
CHILD=""; STEP=start
in_group() {   # a process of group $1 that is not a zombie
  local f s a
  for f in /proc/[0-9]*/stat; do
    { read -r s < "$f"; } 2> /dev/null || continue   # a process can exit between the glob and the read
    read -r -a a <<< "${s##*) }"
    [ "${a[2]:-}" = "$1" ] && [ "${a[0]:-}" != Z ] && return 0
  done
  return 1
}
stop_group() {
  [ -n "$CHILD" ] || return 0
  local g=$CHILD i; CHILD=""
  kill -TERM -- "-$g" 2> /dev/null
  for ((i = 0; i < ${STOP_GRACE_S:-60}; i++)); do in_group "$g" || return 0; sleep 1; done
  echo "serve_fresh: process group $g outlived TERM by ${STOP_GRACE_S:-60} poll(s); KILL" >&2
  kill -KILL -- "-$g" 2> /dev/null
  while in_group "$g"; do sleep 1; done
}
trap stop_group EXIT
trap 'stop_group; echo "SERVE-FRESH FAILED $STEP: stopped by a signal"; exit 143' TERM INT HUP
# step NAME CMD...: runs CMD (in adapters/vllm) into $LOGS/NAME.log, or prints it under --dry-run
step() {
  local name=$1 rc; shift
  if [ "$DRY" = 1 ]; then echo "step $name: $*"; return 0; fi
  local t=$(date +%s)
  echo "serve_fresh: $name ..."; STEP=$name
  (cd "$TREE/adapters/vllm" && exec setsid -w "$@") > "$LOGS/$name.log" 2>&1 &
  CHILD=$!; wait "$CHILD"; rc=$?; stop_group
  [ "$rc" = 0 ] || fail "$name" "exit $rc (log $LOGS/$name.log: $(tail -1 "$LOGS/$name.log" | cut -c1-200))"
  echo "serve_fresh: $name done in $(( $(date +%s) - t )) s"
}
step bootstrap bash verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B_INT8 --out "$W/bootstrap"
step workload "$PY" -m research data fetch-fixtures --repo "$TREE" --json "$WL"
if [ "$DRY" = 0 ]; then
  want=$(awk -v n="$ROW.json" '$2==n{print $1}' "$TREE/adapters/vllm/workloads/SHA256SUMS")
  [ "$(sha256sum < "$WL" | cut -c1-64)" = "$want" ] || fail workload "$WL is not the sha256 SHA256SUMS lists ($want)"
fi
step predict "$VENV/bin/verity-vllm" predict row --row "$ROW" --out "$W/predicted"
step build "$VENV/bin/verity-vllm" row run "$ROW" LLAMA32_1B_INT8 "$MODEL" "$REV" --sweep-dir "$W/llama" \
  --research-run-dir "$W/llama-kept" --stages build --config-run 1 --replay-k 460 --hidden-so "$HSO" --norm-tap-so "$NSO"

SERVE=("$VENV/bin/verity-vllm" serve-openai --model "$MODEL" --revision "$REV" --out "$W/serve" --port "$PORT"
  --precision verity_int8 --run-id "fresh-$(date -u +%Y%m%dT%H%M%SZ)" --scheme rows-sha512 --seal --max-num-batched-tokens 512
  --window-steps 256 --required-manifest "$W/predicted/manifest.json" --hidden-so "$HSO" --norm-tap-so "$NSO" --tap-cap-mb 2048
  --admit release-order --window-requests 1 --max-model-len 512 --window-gb 4 --disk-floor-gb 5 --seal-hold-gb 8
  --seal-warm-gb 0 --copy-records-to "$W/records")
LOAD=("$PY" adapters/vllm/tools/served_run/serve_load.py --port "$PORT" --model "$MODEL" --varied 1 --rate 1000 --max-tokens 4
  --prompt-range 16:24 --seed 1 --out "$W/load.jsonl")
if [ "$DRY" = 1 ]; then
  echo "step serve: ${SERVE[*]}"; echo "step request: ${LOAD[*]} --release UUID"
  echo "step stop: SIGINT to the server's pid, which seals window 0 before it exits; then the seal read from $W/serve/seals.jsonl"
  exit 0
fi
echo "serve_fresh: serve on port $PORT ..."
STEP=serve
(cd "$TREE/adapters/vllm" && exec setsid -w "${SERVE[@]}") > "$LOGS/serve.log" 2>&1 &
SERVER=$!; CHILD=$SERVER
for _ in $(seq 1 300); do
  kill -0 "$SERVER" 2>/dev/null || fail serve "the server exited before it listened (log $LOGS/serve.log)"
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { kill -INT "$SERVER"; fail serve "not listening after 25 min (log $LOGS/serve.log)"; }
STEP=request
(cd "$TREE" && "${LOAD[@]}" --release "$(cat /proc/sys/kernel/random/uuid)") > "$LOGS/request.log" 2>&1
lrc=$?
STEP=stop
kill -INT "$SERVER"
wait "$SERVER"; src=$?; stop_group
[ "$lrc" = 0 ] || fail request "exit $lrc (log $LOGS/request.log)"
[ "$src" = 0 ] || fail stop "the server exited $src (log $LOGS/serve.log)"
seal=$(grep -m1 '"window": 0[,}]' "$W/serve/seals.jsonl" 2>/dev/null)
root=$(printf '%s' "$seal" | sed -n 's/.*"run_root": "\([0-9a-f]*\)".*/\1/p')
case $seal in *'"error"'*) root="" ;; esac
[ -n "$root" ] || fail seal "window 0 is not sealed ($W/serve/seals.jsonl, $LOGS/serve.log)"
echo "SERVE-FRESH SERVED AND SEALED window 0, root $root"
