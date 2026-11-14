#!/bin/bash
# pub.sh SLUG BODY.md: publish BODY.md as a note (note.sh) with every `@NOW@` in it replaced by the clock's reading at
# this moment (`date -u +%H:%MZ`), so that no hour is written before the clock is read. Keeps the published text beside
# the body as BODY.md.pub.
set -uo pipefail
now=$(date -u +%H:%MZ)
sed "s/@NOW@/$now/g" "$2" > "$2.pub"
grep -q '@NOW@' "$2" || echo "pub.sh: the body names no @NOW@: check every hour in it against $now"
bash /tmp/carrier/scripts/note.sh "$1" "$2.pub"
