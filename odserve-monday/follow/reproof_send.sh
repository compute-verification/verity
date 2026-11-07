#!/usr/bin/env bash
# The prover-side copy of a deferral's serve dir D from the served node to the V3c follow-on's node, after the served run's
# on-node part: the files D/private/deferred/inputs.txt lists, and the serve dir's re-proof inputs (onsite_record.py's
# REPROOF, reproof_files.py) it omits, listed in D/private/reproof-extra.txt and hashed in D/private/reproof-extra.sha256,
# each at its path under D. With ROWS=1 (the default) it sends the Commit's sealed row dir of the window as well,
# windows/w<K>/ less build_request_*/ (step 3's compile writes those after the close): the step_rows trees and row tables an
# audit-time completion of a member's --registered file reads, listed in D/private/rows.txt and hashed in
# D/private/rows.sha256, which reproof_follow.sh checks before the audit opens its session. ROWS=0 sends none.
# Witness: it prints PASS or REFUSED alone, no name, count or size, and writes nothing to out/.
#   research run --on vy-dc-h200-2 --project verity --source TREE --cwd clone --timeout 2h --env D=SERVE_DIR \
#     --env NAME=SERVED_RUN_ID --env COPY_TO=research@vy-nebius-cpu-11 (i) or research@vy-nebius-cpu-12 (ii) [--env COPY_KEY=PATH] \
#     --send reproof_send.sh --send reproof_files.py -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/reproof_send.sh"'
# COPY_DEST, when given, replaces COPY_TO:NAME/ (a local dir in a dry run).
set -uo pipefail
IN=${RESEARCH_RUN_DIR:-.}/inputs
: "${D:?the serve dir}" "${NAME:?the served run id}"
[ -n "${COPY_DEST:-}" ] || : "${COPY_TO:?the follow-on node as USER@HOST, its PP_PEER: research@vy-nebius-cpu-11 for (i), research@vy-nebius-cpu-12 for (ii)}"
COPY_KEY=${COPY_KEY:-$HOME/.ssh/vy-h200-2-to-prover}
dest=${COPY_DEST:-$COPY_TO:$NAME/}
refuse() { echo "REPROOF SEND: REFUSED: $*"; exit 3; }
P=$D/private
[ -f "$P/deferred/inputs.txt" ] && [ -f "$P/deferred/deferred.env" ] || refuse "D holds no deferral (private/deferred/inputs.txt, deferred.env)"
read -r K VTAG < <(set +u; . "$P/deferred/deferred.env"; echo "$K" "${V3C_CLASS//,/-}")
[ -n "$K" ] && [ -n "$VTAG" ] || refuse "deferred.env names no window or no V3c class"
umask 077
python3 "$IN/reproof_files.py" "$PWD" "$D" "$K" "$VTAG" > "$P/reproof-files.txt" || refuse "reproof_files.py failed"
LC_ALL=C comm -23 <(LC_ALL=C sort -u "$P/reproof-files.txt") <(LC_ALL=C sort -u "$P/deferred/inputs.txt") > "$P/reproof-extra.txt"
(cd "$D" && tr '\n' '\0' < private/reproof-extra.txt | xargs -0 -r sha256sum) > "$P/reproof-extra.sha256" \
  || refuse "hashing the extra files failed"
rm -f "$P/rows.txt" "$P/rows.sha256"
if [ "${ROWS:-1}" = 1 ]; then
  [ -d "$D/windows/w$K" ] || refuse "D holds no row dir windows/w$K"
  (cd "$D" && find "windows/w$K" -type f ! -path "windows/w$K/build_request_*" | LC_ALL=C sort) > "$P/rows.txt" \
    && [ -s "$P/rows.txt" ] || refuse "the row dir windows/w$K holds no file"
  (cd "$D" && tr '\n' '\0' < private/rows.txt | xargs -0 -r sha256sum) > "$P/rows.sha256" || refuse "hashing the row dir failed"
fi
rsh=(ssh -i "$COPY_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30)
[ -n "${COPY_DEST:-}" ] || [ -f "$COPY_KEY" ] || refuse "no copy key $COPY_KEY"
{ cat "$P/deferred/inputs.txt" "$P/reproof-extra.txt"; printf '%s\n' private/deferred/inputs.txt private/reproof-extra.txt \
    private/reproof-extra.sha256
  [ ! -f "$P/rows.sha256" ] || { cat "$P/rows.txt"; printf '%s\n' private/rows.txt private/rows.sha256; }
} | LC_ALL=C sort -u > "$P/reproof-send.txt"
(cd "$D" && rsync -a --files-from=private/reproof-send.txt -e "${rsh[*]}" ./ "$dest") > "$P/reproof-send.log" 2>&1 \
  || refuse "rsync to the follow-on's node failed (D/private/reproof-send.log)"
echo "REPROOF SEND: PASS: the deferral's inputs and the serve dir's re-proof inputs it omits copied to the follow-on's node," \
     "each at its path under D, the extras hashed in D/private/reproof-extra.sha256$([ ! -f "$P/rows.sha256" ] \
     || echo ", and the window's sealed row dir, hashed in D/private/rows.sha256")"
