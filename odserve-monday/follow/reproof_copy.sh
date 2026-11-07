#!/usr/bin/env bash
# REPROOF_CMD of the V3c follow-on (`served_verdicts.sh v3c` runs it after V3c's audit as `bash -c CMD _ D K VW` in TREE):
# the extras reproof_send.sh sent held to their sums, then the serve dir's re-proof inputs (reproof_files.py over D, the
# follow-on's openings, draw and coins among them, less the audit's composed circuits, audit/window/m<k>/circuit.txt: public,
# composed again from the window commit's circuit.json, and at 4,32,4 about 550 GB, more than the destination holds) hashed
# in D/private/reproof-files.sha256 and copied with it to
# REPROOF_DEST in D's layout over REPROOF_KEY's write-only route (a local directory in a dry run), and the copy's listing
# written to REPROOF_LISTING once rsync has sent every file. Its last line is the follow-on's REPROOF line; it exits
# nonzero, failing the follow-on, where the copy fails. The copy, its listing and their bytes stay off out/.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=${1:?D} K=${2:?K} VW=${3:?VW}
: "${REPROOF_DEST:?the destination of rsync}" "${REPROOF_LISTING:?where the listing goes}"
VTAG=${VW##*/v3c-w"$K"-}
P=$D/private
umask 077
fail() { echo "REPROOF COPY: FAIL ($1)"; exit 3; }
if [ -f "$P/reproof-extra.sha256" ]; then
  (cd "$D" && sha256sum --check --strict --quiet private/reproof-extra.sha256) > "$P/reproof-extra-check.log" 2>&1 \
    || fail "the extras unlike the served node's sums: D/private/reproof-extra-check.log"
fi
python3 "$HERE/reproof_files.py" "$PWD" "$D" "$K" "$VTAG" "${H1_TIE:+-tie}" | grep -vxF -e private/reproof-files.sha256 \
  -e private/reproof-listing.txt -e private/reproof.log | grep -vxE "v3c-w$K-$VTAG/audit/window/m[0-9]+/circuit\.txt" \
  > "$P/reproof-files.txt" || fail "reproof_files.py"
[ -s "$P/reproof-files.txt" ] || fail "reproof_files.py named no file"
(cd "$D" && tr '\n' '\0' < private/reproof-files.txt | xargs -0 -r sha256sum) > "$P/reproof-files.sha256" || fail "hashing"
REPROOF_KEY=${REPROOF_KEY:-$HOME/.ssh/reproof_copy_ed25519}
case $REPROOF_DEST in
  *:*) rsh=(-e "ssh -i $REPROOF_KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30") ;;
  *) rsh=(); mkdir -p "$REPROOF_DEST" ;;
esac
(cd "$D" && { cat private/reproof-files.txt; echo private/reproof-files.sha256; } \
   | rsync -a --files-from=- "${rsh[@]}" ./ "$REPROOF_DEST") > "$P/reproof-copy.log" 2>&1 \
  || fail "rsync; D/private/reproof-copy.log"
{ cat "$P/reproof-files.txt"; echo private/reproof-files.sha256; } | LC_ALL=C sort -u > "$REPROOF_LISTING"
echo "REPROOF COPY: PASS: $(grep -c . "$REPROOF_LISTING") files, the serve dir's re-proof inputs and their sums, copied in" \
     "D's layout; the listing is off out/"
