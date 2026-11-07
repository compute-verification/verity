#!/usr/bin/env bash
# After served_verdicts.sh, before the run's outputs are kept: move what out/ mustn't publish into the serve dir's
# private/withheld/ (same relative paths), then list what out/ still holds by key name into a private file on the node.
# Usage: scrub-d3.sh OUT SERVE_DIR SCAN_PY
set -uo pipefail
O=${1:?OUT} D=${2:?SERVE_DIR} SCAN=${3:?SCAN_PY}
W=$D/private/withheld
(umask 077; mkdir -p "$W")
moved=0
withhold() {  # REL: a file or directory under OUT
  local rel=$1
  [ -e "$O/$rel" ] || return 0
  (umask 077; mkdir -p "$(dirname "$W/$rel")")
  if [ -e "$W/$rel" ]; then rm -rf "$W/$rel.prev"; mv "$W/$rel" "$W/$rel.prev"; fi
  mv "$O/$rel" "$W/$rel" && { echo "WITHHELD $rel"; moved=$((moved + 1)); }
}
withhold pieces/certifier/chain
withhold pieces/pouw/ledger.jsonl
for f in $(cd "$O" && find . -type f \( -name 'requests.jsonl' -o -name 'prompts*' -o -name 'answers.json' -o -path '*/scan/*' \
           -o -path '*/endpoint/*' -o -name 'leak-guard.json' \) | sed 's|^\./||'); do
  withhold "$f"
done
find "$O" -type d -empty -delete 2>/dev/null
echo "SCRUB: $moved withheld into the serve dir's private/withheld/"
python3 "$SCAN" "$O" > "$D/private/scrub-scan.txt" 2>&1
echo "SCRUB: key scan of out/ in the serve dir's private/scrub-scan.txt ($(grep -c . "$D/private/scrub-scan.txt") lines)"
