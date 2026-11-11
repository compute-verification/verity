#!/usr/bin/env bash
# A11's one-stage audit (a11.py --load-values) of stage.py's saved directory, on a fresh pod, from a clone of the tree
# (`research run --cwd clone`): the Rust, Lean and uv toolchains if missing, then hidden_zk's builds (flock-circuit,
# flock-verify, the venv, kept under SP_CACHE), the sampled-proofs verifier, and the audit.
#   SP_MODE=build bash benchmarks/spot_proved/run_audit.sh                       # the builds only
#   SP_MODE=audit SP_VALUES=DIR SP_ARGS="--per-kind 2 --tamper none" bash benchmarks/spot_proved/run_audit.sh
#   (SP_VALUES may be a .tar.gz of one such directory, e.g. sent with --send: "$RESEARCH_RUN_DIR/inputs/values.tar.gz")
#   SP_MODE=audit SP_SYNTH=LABEL[,LABEL] SP_ARGS=... bash ...  # synth.py's random inputs: a timing run, not served values
# Audits sharing a pod share its sessions and Lean verifies through a11's slots: SP_ARGS="... --lock-dir /workspace/spot-locks
# --slots session=1,lean=4".
set -uo pipefail
for t in rustup elan uv; do
  case $t in
    rustup) flock /tmp/boot-rustup.lock sh -c '"$HOME/.cargo/bin/cargo" --version >/dev/null 2>&1 || curl --proto =https -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1' ;;
    elan) flock /tmp/boot-elan.lock sh -c '[ -x "$HOME/.elan/bin/lake" ] || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null 2>&1' ;;
    uv) flock /tmp/boot-uv.lock sh -c 'command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1' ;;
  esac
done
export PATH="$HOME/.elan/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export HZ_CONFIGS=spot HZ_MEM_GB=${SP_MEM_GB:-400} HZ_CACHE=${SP_CACHE:-/workspace/spot-cache}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/protocols/spot/verifier/lean && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
[ "${SP_MODE:?}" = build ] && { log "spot: builds done"; exit 0; }
if [ -n "${SP_SYNTH:-}" ]; then
  "$PY" benchmarks/spot_proved/synth.py --out "$W/synth" --labels "$SP_SYNTH" >> "$OUT/a11.log" 2>&1 || exit 2
  SP_VALUES=$W/synth
fi
S=${SP_VALUES:?}
case $S in
  *.tar.gz) mkdir -p "$W/values" && tar -xzf "$S" -C "$W/values" || exit 2
            S=$(dirname "$(find "$W/values" -name values.json | head -1)") ;;
esac
log "spot: audit of $S ($(du -sh "$S" | cut -f1))"
# shellcheck disable=SC2086
"$PY" benchmarks/spot_proved/audit.py --bin "$BIN" --lean "$LEAN" --load-values "$S" --kinds "$S/kinds.json" --out "$W/a11" \
  ${SP_ARGS:-} 2>&1 | tee -a "$OUT/a11.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a11"
(cd "$W/a11" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a11/")
log "spot: audit done, rc $rc"
exit "$rc"
