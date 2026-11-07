#!/usr/bin/env bash
# NET-190's checks of a served run's serve dir, on the node that served it, in a clone of the run's tree, once its serve
# step is done (its SERVE stopped line):
#   research run --on NODE --project verity --source TREE --cwd clone --send odserve_check.sh --send odserve_check.py \
#     --send odserve_cost.py [--send STOCK.json] --env RUN=RID [--env STOCK=STOCK.json] -- \
#     bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/odserve_check.sh"'
# RUN is the served run's id (its serve dir SERVE_ROOT/RUN, SERVE_ROOT default /workspace/research/vllm-served-verdicts);
# STOCK names an input, the remote run's public out/serve/overhead.json. odserve_check.py holds the network certifier's
# records and every served request to the front's kept fetch, odserve_cost.py the serving cost to STOCK. out/ gets their
# public lines alone (verdicts, the link-windows' counts, ratios); every file they read or write stays in the serve dir's
# private/odserve, 0700.
set -uo pipefail
RUN=${RUN:?the served run whose serve dir to check}
D=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}/$RUN
IN=$RESEARCH_RUN_DIR/inputs OUT=$RESEARCH_RUN_DIR/out P=$D/private/odserve
mkdir -p "$OUT"
say() { printf '%s\n' "$*" | tee -a "$OUT/odserve.txt"; }
say "TREE $(git rev-parse HEAD) on $(hostname -s), the serve dir of $RUN"
[ -d "$D/private/certifier" ] || { say "REFUSED: $D has no private/certifier: no front kept a fetch there"; exit 3; }
(umask 077; rm -rf "$P" && mkdir -p "$P") || exit 3
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > "$P/uv.log" 2>&1
uv sync -q --all-packages --extra torch-cpu > "$P/sync.log" 2>&1 || { say "REFUSED: uv sync exit $? (the serve dir's private/odserve/sync.log)"; exit 4; }
PY=$PWD/.venv/bin/python
say "CERTIFIER-FRONT session: $("$PY" benchmarks/one_stage/served_verdicts.py front-line "$D/private/certifier-front/session.json" 2>&1 | cut -c1-600)"
(umask 077; "$PY" "$IN/odserve_check.py" "$D" "$P" > "$P/check.line" 2> "$P/check.err")
crc=$?
say "CHECK (exit $crc): $(tail -1 "$P/check.line" | cut -c1-900)"
tail -1 "$P/check.line" > "$OUT/check.json"
stock=; [ -n "${STOCK:-}" ] && stock=$IN/$STOCK
(umask 077; "$PY" "$IN/odserve_cost.py" "$D" "$P" "$stock" > "$P/cost.line" 2> "$P/cost.err")
say "COST (exit $?): $(tail -1 "$P/cost.line" | cut -c1-1200)"
tail -1 "$P/cost.line" > "$OUT/cost.json"
rm -rf "$P/fetch"
exit "$crc"
