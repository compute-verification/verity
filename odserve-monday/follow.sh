#!/usr/bin/env bash
# Monday's follow-on on FOLLOW_NODE (env.sh's, vy-nebius-cpu-12 unless set), once send.sh printed "REPROOF SEND: PASS":
# follow/follow_launch.sh at the run's tree, which is the serve's (SERVE_REF = PIN_REF = REF: no FOLLOW_ANY_TREE), with
# MEM_GB 645 (a reservation, waited for up to MEM_WAIT, default 3 h), the four caps from settings-monday.env (CAPS: audit
# 13h, chain 4h, period 1h, composite 12h, together its --timeout of 108000 s), its custody key for 24 h, no re-proof copy
# to cpu-3 (REPROOF 0) and follow/shim/research ahead on PATH (the clone's venv synced before reproof_follow.sh). It
# removes infra's memory hold on the node (place-hold.sh's) as the job launches, and prints the run's launch.
#   [FOLLOW_NODE=vy-nebius-cpu-11] [MEM_WAIT=S] [DRY=1] bash follow.sh RID
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh"
RID=${1:?the served run id}
[[ $RID =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $RID is not a store run id"; exit 3; }
follow_ip "$FOLLOW_NODE" > /dev/null
tree_ok
export CAPS=$H/settings/$SETTINGS_NAME
grep -q '^V3C_AUDIT_TIMEOUT=13h CERTIFIER_TIMEOUT=4h CERTIFIER_PERIOD_TIMEOUT=1h COMPOSITE_TIMEOUT=12h$' "$CAPS" \
  || { echo "REFUSED: $CAPS doesn't set Monday's caps (13h, 4h, 1h, 12h)"; exit 3; }
grep -q '^PRODUCT=remote-network-no-pouw H1_TIE= V3C_CLASS=8,32,4,4,16 CLASSES=8,32,4,4,16 V3C_LAW=subset:1$' "$CAPS" \
  || { echo "REFUSED: $CAPS isn't Monday's settings"; exit 3; }
echo "CAPS $CAPS $(sha256sum < "$CAPS" | cut -c1-16)"
export SERVE_REF=$REF PIN_REF=$REF MEM_GB=${MEM_GB:-645} MEM_WAIT=${MEM_WAIT:-10800} REPROOF=0 MAIN_REPO
export CERTIFIER_PIN=$CERTIFIER_GATE_PIN REAL_RESEARCH=$RESEARCH
cd "$TREE"
PATH=$H/follow/shim:$PATH exec bash "$H/follow/follow_launch.sh" "$TREE" "$FOLLOW_NODE" "$RID"
