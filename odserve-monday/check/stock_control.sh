#!/usr/bin/env bash
# A stock control for NET-190's serving cost, on the served node: served_verdicts.sh's serve step alone, at a served run's
# engine and load but without the network certifier. Its settings are remote's own file (REMOTE, the load counts, private
# on the node) with WINDOW_STEPS and MAX_TOKENS where the served run set them; its load is that file's LOAD, never a front.
# No H1 (H1_GATE empty: no window reserved, no lane period), no PoUW and no piece. Its aggregate (`served_overhead.py
# metrics`, the form remote's out/serve/overhead.json takes) stays in its serve dir's private/stock-overhead.json, and
# odserve_check.sh then checks the served run RUN with that aggregate as STOCK, its lines in out/odserve.txt.
#   research run --on vy-dc-h200-2 --project verity --source TREE --cwd clone --timeout 60m --send stock_control.sh \
#     --send odserve_check.sh --send odserve_check.py --send odserve_cost.py --env RUN=RID --env PRED=DIR --env ROW=DIR \
#     [--env WINDOW_STEPS=16 --env MAX_TOKENS=4] -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/stock_control.sh"'
set -uo pipefail
OUT=$RESEARCH_RUN_DIR/out IN=$RESEARCH_RUN_DIR/inputs
mkdir -p "$OUT"
say() { echo "$*" | tee -a "$OUT/control.txt"; }
RUN=${RUN:?the served run whose serving cost the control prices, or later (the aggregate alone, checked by check_stock.sh)}
REMOTE=${REMOTE:-/workspace/research/vllm-served-verdicts/settings-remote/settings.env}
[ -f "${PRED:-}/manifest.json" ] || { say "PREFLIGHT: PRED=${PRED:-} holds no manifest.json, which serve-openai's --required-manifest reads"; exit 3; }
[ -d "${ROW:-}" ] || { say "PREFLIGHT: ROW=${ROW:-} is not the node's row dir"; exit 3; }
[ -r "$REMOTE" ] || { say "PREFLIGHT: REMOTE=$REMOTE, remote's settings file, isn't readable"; exit 3; }
command -v gpu-lease > /dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
S=$(mktemp -d "${TMPDIR:-/tmp}/stock-control.XXXXXX")
chmod 700 "$S"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
{ echo ". $REMOTE"
  [ -z "${WINDOW_STEPS:-}" ] || echo "WINDOW_STEPS=$WINDOW_STEPS"
  [ -z "${MAX_TOKENS:-}" ] || echo "MAX_TOKENS=$MAX_TOKENS"
  echo "[ \"\$LOAD\" != front ] || LOAD=varied"; } > "$S/settings.env"
export PRED ROW PRODUCT=onsite H1_GATE= POUW_CMD= WIPE_CMD= POUS_CMD= BANDWIDTH_CMD= CERTIFIER_CMD= CERTIFIER_CHAIN_CMD= \
  CERTIFIER_PERIOD_CMD= CERTIFIER_FRONT= V3C_CLASS= SERVE_ARGS=
export SETTINGS=$S/settings.env D=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}/${RESEARCH_RUN_ID:-local}
say "TREE $(git rev-parse HEAD) on $(hostname -s): the stock control of $RUN (remote's settings${WINDOW_STEPS:+, WINDOW_STEPS $WINDOW_STEPS}${MAX_TOKENS:+, MAX_TOKENS $MAX_TOKENS}; no certifier, no H1); serve dir $D"
t0=$SECONDS
OUT=$OUT/serve GPU_LEASE_WHO=odserve-stock gpu-lease 1 --wait --max-min 40 --mem-gb 128 -- bash benchmarks/one_stage/served_verdicts.sh serve
rc=$?
say "CONTROL: served_verdicts.sh serve exit $rc after $((SECONDS - t0)) s; its lines:"
sed 's/^/  /' "$OUT/serve/verdicts.txt" 2>/dev/null | cut -c1-900 | tee -a "$OUT/control.txt"
rm -rf "$S"
[ "$rc" = 0 ] || { say "CONTROL FAIL: no aggregate without a serve"; exit 1; }
export PYTHONPATH=$PWD:$PWD/integrations/vllm:$PWD/tools/research/src
(umask 077; "${VENV:-/workspace/cp/venv312}/bin/python" benchmarks/one_stage/served_overhead.py metrics "$D" \
  > "$D/private/stock-overhead.json" 2> "$D/private/stock-overhead.err") \
  || { say "CONTROL FAIL: served_overhead.py metrics exit $? (the serve dir's private/stock-overhead.err)"; exit 1; }
[ "$RUN" != later ] || { say "CONTROL: its aggregate in $D/private/stock-overhead.json, for a check job's STOCK once the served run ends"; exit 0; }
say "CONTROL: its aggregate in the serve dir's private/stock-overhead.json; the check of $RUN against it follows"
RUN=$RUN STOCK=$(realpath --relative-to="$IN" "$D/private/stock-overhead.json") bash "$IN/odserve_check.sh"
