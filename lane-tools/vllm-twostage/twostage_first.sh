#!/usr/bin/env bash
# vllm-twostage (VLLM-250): a served window's first stage on the auditor's challenger, from a node that is
# vy-challenger-1's gate peer and holds lane vllm's gate key (vy-nebius-cpu-3), under research run with --cwd source from
# a tree holding C (#8045) and P1 (#8066). two_stage.py first registers the class's commit.json and samples it on one
# withheld session the gate opens on the window's run root (gate:research@204.12.178.84, --peer this node, the
# deployment's challenger-key record); its stage1.json and the session's evidence (the auditor's signed ledger, the
# host's pins log, the session's files) land in OUT/stage1/; check_stage1.py then holds the coin evidence to
# coins.coins_reason (OUT/check.txt).
#   research run --on vy-nebius-cpu-3 --project verity --source TREE --cwd source --timeout 30m \
#     --send twostage_first.sh --send check_stage1.py --send main.txt \
#     (--send commit.json --send interior.json | --send make_standin.py) \
#     --env PEER=vy-nebius-cpu-3 -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/twostage_first.sh'
# main.txt: the verifier's own `git rev-list --first-parent origin/main`, made where the launch is. With commit.json
# sent, the class is the served one (its check and directory from commit.json) and interior.json, [[name, schema], ...],
# is the replay side's interior values; without it, make_standin.py's stand-in of class G at V=128 on a fresh run root.
# [REPLAY=rate:1/2] [K=1] [GATE=research@204.12.178.84] [SSH_KEY=~/.ssh/vllm_lane_ed25519]
set -euo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; I=$D/inputs; mkdir -p "$OUT"
PEER=${PEER:?set PEER, this node's name in the gate's /etc/vy/peers} GATE=${GATE:-research@204.12.178.84}
KEY=${SSH_KEY:-$HOME/.ssh/vllm_lane_ed25519} REPLAY=${REPLAY:-rate:1/2} K=${K:-1}
[ -f "$I/main.txt" ] || { echo "twostage-first: --send main.txt, the verifier's git rev-list --first-parent origin/main"; exit 2; }
uv sync -q --all-packages --extra torch-cpu
PY="uv run --no-sync python"
if [ -f "$I/commit.json" ]; then
  [ -f "$I/interior.json" ] || { echo "twostage-first: a served class needs the replay side's interior values (--send interior.json)"; exit 2; }
  class=$D/boundary/$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("w%s/p%s-o%s" % (d["window"], d["prompt"], d["outputs"]))' "$I/commit.json")
  mkdir -p "$class"; cp "$I/commit.json" "$class/"; cp "$I/interior.json" "$OUT/interior.json"
else
  serve="gate-rehearsal-$(date -u +%Y%m%dT%H%M%SZ)-$(od -An -N4 -tx4 /dev/urandom | tr -d ' ')"
  class=$($PY "$I/make_standin.py" "$OUT/standin" "$serve" | tee "$OUT/standin.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["class"])')
  echo '[["x","hm96-sha512/row/v2"],["out","hm96-sha512/row/v2"]]' > "$OUT/interior.json"
fi
check=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["check"])' "$class/commit.json")
echo "FIRST: $(date -u +%T) class=$class check=$check replay=$REPLAY k=$K peer=$PEER"
$PY benchmarks/one_stage/two_stage.py first --boundary "$class" --check "$check" --replay "$REPLAY" --k "$K" \
  --interior "$OUT/interior.json" --challenger "gate:$GATE" --peer "$PEER" \
  --challenger-key benchmarks/pouw/tests/fixtures/challenger-key.json --ssh-key "$KEY" --main "$I/main.txt" \
  --out "$OUT/stage1"
echo "FIRST-DONE: $(date -u +%T)"
rc=0; $PY "$I/check_stage1.py" "$OUT/stage1/$(basename "$class")/stage1.json" > "$OUT/check.txt" 2>&1 || rc=$?
tail -n 3 "$OUT/check.txt"
exit $rc
