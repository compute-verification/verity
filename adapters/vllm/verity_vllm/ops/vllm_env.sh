#!/usr/bin/env bash
# A vLLM build of `engine.stock.BUILDS` other than the pin, in an environment of its own beside the pin's: Python 3.12, the
# build's sha256-pinned wheel, the torch it names and FlashInfer's compiled kernels, which this build's wheel leaves to a
# step of its own and without which vLLM turns FlashInfer off.  The pin's environment (pod_bootstrap.sh's $WORK/venv312)
# is never read or written.
#
#   VENV=$(bash adapters/vllm/verity_vllm/ops/vllm_env.sh) || exit    # the last stdout line is the path
#
# Idempotent: a directory that holds the build, with the freeze.txt an install writes last, is left as it is, and one that
# holds anything else is refused.  An install that fails removes what it made.  Fails with a named class
# (VLLM_ENV_FAIL_<step>) and exits 3.
# Env: SERVE_VLLM (the build, by short commit: 52aeb5806; no VLLM_ name, which vLLM's own check of its environment would
# report as unknown), WORK (/workspace), VENV ($WORK/venv-vllm-$SERVE_VLLM), NEED_GB (the free space the install asks of
# VENV's filesystem: 30), OUT (where pip.log and freeze.txt are copied; none).
set -uo pipefail
BUILD=${SERVE_VLLM:-52aeb5806}
case "$BUILD" in
  52aeb5806)   # v0.34.0rc1: its torch is 2.14.0, which has no cu129 wheel
    COMMIT=52aeb580658571613f8ab28a075a06fa30279beb
    W=vllm-0.34.0rc1-cp38-abi3-manylinux_2_28_x86_64.whl
    WSHA=3c816985a80f66cd333788ed107a25bd296e1ac3b81b601d4b9423eab567a042
    CU=cu130 ;;
  d9105ea80) echo "VLLM_ENV_FAIL_BUILD: d9105ea80 is the pin, whose environment is pod_bootstrap.sh's" >&2; exit 3 ;;
  *) echo "VLLM_ENV_FAIL_BUILD: $BUILD is no build of engine.stock.BUILDS that this script installs (52aeb5806)" >&2; exit 3 ;;
esac
WORK=${WORK:-/workspace}; VENV=${VENV:-$WORK/venv-vllm-$BUILD}; PY=$VENV/bin/python
say() { echo "[vllm-env] $(date -u +%FT%TZ) $*" >&2; }
fail() { echo "VLLM_ENV_FAIL_$1: ${*:2}" >&2; exit 3; }
# the build an environment holds, read from the file setuptools-scm wrote into its vLLM: a tagged build's __version__
# carries no commit, and __commit_id__ names every build
held() { sed -n "s/^__commit_id__ = .*['\"]g\([0-9a-f]*\)['\"].*/\1/p" "$VENV"/lib/python3.*/site-packages/vllm/_version.py 2>/dev/null | head -n 1; }

if [ -e "$VENV" ]; then
  h=$(held); [ "$h" = "$BUILD" ] || fail OTHER "$VENV exists and does not hold vLLM $BUILD (it holds: ${h:-no vLLM}); name another VENV"
  [ -f "$VENV/freeze.txt" ] || fail OTHER "$VENV holds vLLM $BUILD without freeze.txt, so its install did not finish; name another VENV"
  say "$VENV holds vLLM $BUILD"
else
  command -v uv >/dev/null || fail UV "no uv on PATH"
  parent=$(dirname "$VENV"); mkdir -p "$parent" || fail DIR "cannot make $parent"
  free=$(df -BG --output=avail "$parent" | tail -n 1 | tr -dc 0-9)
  [ "$free" -ge "${NEED_GB:-30}" ] || fail DISK "$parent has $free GB free and the install asks for ${NEED_GB:-30} (NEED_GB)"
  # the wheel and uv's cache on VENV's own filesystem, so that uv links its files instead of copying them, removed at exit
  tmp=$(mktemp -d "$parent/.vllm-env.XXXXXX") || fail DIR "cannot make a directory under $parent"
  undo() { rm -rf "$tmp"; [ "${ok:-0}" = 1 ] || rm -rf "$VENV"; }
  trap undo EXIT
  curl -fsSLo "$tmp/$W" "https://wheels.vllm.ai/$COMMIT/$W" || fail WHEEL "vLLM wheel download"
  echo "$WSHA  $tmp/$W" | sha256sum -c >/dev/null || fail WHEEL_SHA "vLLM wheel sha256 != $WSHA"
  say "installing vLLM $BUILD and torch $CU into $VENV ($free GB free)"
  uv venv "$VENV" --python 3.12 >/dev/null || fail VENV "uv venv --python 3.12 failed"
  UV_CACHE_DIR=$tmp/uv UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} uv pip install --python "$PY" \
    --extra-index-url "https://download.pytorch.org/whl/$CU" --index-strategy unsafe-best-match \
    "$tmp/$W" "numpy>=1.26" safetensors huggingface_hub zstandard pytest ninja >"$tmp/pip.log" 2>&1 \
    || { tail -n 5 "$tmp/pip.log" >&2; [ -z "${OUT:-}" ] || cp "$tmp/pip.log" "$OUT/"; fail PIP "see the lines above"; }
  h=$(held); [ "$h" = "$BUILD" ] || fail IMPORT "the installed vLLM is not $BUILD (it is: ${h:-none})"
  # the package imports (its own command needs a device even to print its version, and a node may have none)
  "$PY" -m vllm._version >"$tmp/import.txt" 2>&1 || { tail -n 5 "$tmp/import.txt" >&2; fail IMPORT "the installed vLLM does not import"; }
  # FlashInfer's own step, the one vLLM's image runs: the kernels of every GPU architecture, by uv into this environment
  UV_CACHE_DIR=$tmp/uv UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} FLASHINFER_DISABLE_VERSION_CHECK=1 \
    "$PY" -m flashinfer download-kernels >>"$tmp/pip.log" 2>&1 \
    || { tail -n 8 "$tmp/pip.log" >&2; [ -z "${OUT:-}" ] || cp "$tmp/pip.log" "$OUT/"; fail KERNELS "flashinfer download-kernels: see the lines above"; }
  ls -d "$VENV"/lib/python3.*/site-packages/flashinfer_cubin >/dev/null 2>&1 \
    || fail KERNELS "flashinfer download-kernels left no flashinfer_cubin"
  [ "$(held)" = "$BUILD" ] || fail KERNELS "FlashInfer's step changed the installed vLLM"
  UV_CACHE_DIR=$tmp/uv uv pip freeze --python "$PY" >"$VENV/freeze.txt" || fail FREEZE "uv pip freeze"
  [ -z "${OUT:-}" ] || cp "$tmp/pip.log" "$VENV/freeze.txt" "$OUT/"
  ok=1
  say "installed: $(du -sh "$VENV" | cut -f1), $(wc -l <"$VENV/freeze.txt") distributions ($VENV/freeze.txt)"
fi
echo "$VENV"
