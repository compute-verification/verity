#!/bin/bash
# backup.sh: pack the irreplaceable part of /workspace/pouw into this research run's directory, so the run's custody publishes it
# to the evidence store (the node holds no store credential of its own; `research run --on vy-nebius-2` mints one per run).
#
#   research run --on vy-nebius-2 --project verity --campaign pous-infra -- bash /workspace/pouw/infra/bin/backup.sh
#
# One deterministic .tar.zst per unit: each job directory under fill-out/ and approved-weights/, every other second-level
# directory, and one tarball of the loose files.  Names sorted, mtimes, owners and pax times zeroed, so an unchanged unit is the
# same bytes every hour: custody uploads only the units that changed (one HEAD each for the rest), and on disk an unchanged unit
# is a hard link to the previous backup's copy.  Custody walks every run file, so tens of thousands of small files made the
# 08:11Z backup's publish take over an hour; a few hundred tarballs don't.
# Left out, because they can be rebuilt: Python environments (env/, venv/, .venv/, site-packages/), caches (.cache/, *-cache/,
# *_cache/, cache/, such as a job's uv-cache), __pycache__, Lean
# build dirs (.lake/), synced source trees (trees/, and the src/ copies fill jobs make), the check toolchain, a fill job's
# running copy.  The run writes backup/MANIFEST.tsv (unit, file, sha256 and bytes of the whole tarball, chunks) and
# backup/summary.json.  Restore a unit: cat tars/NAME.tar.zst.* | zstd -d | tar -x
# It packs nothing while a timed window runs or waits (the fill runner's status.txt): it waits before each unit, and after
# BACKUP_QUIET_MAX_S (default 3600) of windows it stops with exit 75, leaving the rest to the next hour's backup.
set -euo pipefail
: "${RESEARCH_RUN_DIR:?run this through research run}"
SRC=${BACKUP_SRC:-/workspace/pouw}
LAST=${BACKUP_LAST:-/workspace/pouw/infra/last-backup}
DEST=$RESEARCH_RUN_DIR/backup
mkdir -p "$DEST/tars"
EXCL=(--exclude=env --exclude=venv --exclude=.venv --exclude=site-packages --exclude=__pycache__ --exclude=.cache --exclude=.lake
      --exclude='*.tmp' --exclude=src --exclude=src-*)
TAR=(tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime
     "${EXCL[@]}")
PREV=$(readlink -f "$LAST" 2>/dev/null || true)
STATUS=${BACKUP_STATUS:-/workspace/pouw/fill/status.txt}
QUIET_MAX_S=${BACKUP_QUIET_MAX_S:-3600}
waited=0
quiet() {
  while head -1 "$STATUS" 2> /dev/null | grep -qE "timed True|window waiting True"; do
    if [ "$waited" -ge "$QUIET_MAX_S" ]; then
      echo "backup.sh: timed windows held it for ${waited}s; stopping, the next hour's backup takes the rest" >&2
      exit 75
    fi
    sleep 10
    waited=$((waited + 10))
  done
}
quiet
cd "$SRC"
# units.txt: the unit directories; loose.txt: every kept file outside them (so each kept file is in exactly one tarball)
python3 - "$DEST" <<'EOF'
import os, re, sys
dest = sys.argv[1]
skip_dir = re.compile(r"^(env|venv|\.venv|site-packages|__pycache__|\.cache|.*[-_]cache|cache|\.lake|src|src-.*)$")
skip_top = {"./trees", "./infra/toolchain", "./fill/running"}
units, loose = [], []
for root, dirs, files in os.walk("."):
    dirs[:] = sorted(d for d in dirs if not skip_dir.match(d) and os.path.join(root, d) not in skip_top)
    depth = 0 if root == "." else root.count("/")
    grouped = root.split("/")[1:2] in (["fill-out"], ["approved-weights"])
    unit_depth = 2 if grouped else 1               # a unit is ./<g>/<group>/<job> under those, else ./<top>/<dir>
    if depth == unit_depth:
        units.extend(os.path.join(root, d) for d in dirs)
        dirs[:] = []
    loose.extend(os.path.join(root, f) for f in sorted(files) if not f.endswith(".tmp"))
open(f"{dest}/units.txt", "w").write("".join(u + "\n" for u in sorted(units)))
open(f"{dest}/loose.txt", "w").write("".join(f + "\n" for f in loose))
EOF
printf 'unit\tfile\tsha256\tbytes\tchunks\n' > "$DEST/MANIFEST.tsv"
pack() {   # pack NAME LABEL TAR-ARGS...: one deterministic tarball in 60 MiB chunks (NAME.tar.zst.0000, ...), hard-linked to the
           # previous backup's chunks when the whole tarball is identical.  Every object stays a single PUT: multipart uploads of
           # larger files stalled from node 2 (connections left in CLOSE-WAIT at 0 B/s)
  local name=$1 label=$2; shift 2
  local tmp="$DEST/tars/.$name.tmp" try
  quiet
  # a unit a running job rewrites can change or lose a file mid-read (tar exits non-zero): retry it, and after three misses
  # leave it to the next hour's backup (skipped.txt) rather than lose every unit after it
  for try in 1 2 3; do
    if "${TAR[@]}" -cf - "$@" 2> /dev/null | zstd -q -3 -T1 > "$tmp"; then break; fi
    if [ "$try" = 3 ]; then rm -f "$tmp"; echo "$label" >> "$DEST/skipped.txt"; return 0; fi
  done
  local sum size prev n
  sum=$(sha256sum "$tmp" | cut -d' ' -f1)
  size=$(stat -c %s "$tmp")
  prev=$([ -n "$PREV" ] && [ -f "$PREV/MANIFEST.tsv" ] && awk -F'\t' -v n="tars/$name.tar.zst" '$2 == n {print $3}' "$PREV/MANIFEST.tsv" | head -1 || true)
  if [ -n "$prev" ] && [ "$prev" = "$sum" ] && ls "$PREV/tars/$name.tar.zst."* > /dev/null 2>&1; then
    for c in "$PREV/tars/$name.tar.zst."*; do ln -f "$c" "$DEST/tars/$(basename "$c")"; done
    rm -f "$tmp"
  else
    split -b 60M -d -a 4 "$tmp" "$DEST/tars/$name.tar.zst."
    rm -f "$tmp"
  fi
  n=$(ls "$DEST/tars/$name.tar.zst."* | wc -l)
  printf '%s\ttars/%s.tar.zst\t%s\t%s\t%s\n' "$label" "$name" "$sum" "$size" "$n" >> "$DEST/MANIFEST.tsv"
}
# units over BACKUP_MAX_UNIT_MB (default 1024) go to large.txt instead: one custody upload of many GB stalls (a dropped
# connection has no resume), so each large unit is backed up by a run of its own (backup_unit.sh), once it stops changing
MAX_KB=$((${BACKUP_MAX_UNIT_MB:-1024} * 1024))
: > "$DEST/large.txt"
while read -r u; do
  if [ "$(du -sk "$u" | cut -f1)" -gt "$MAX_KB" ]; then echo "$u" >> "$DEST/large.txt"; continue; fi
  pack "$(echo "${u#./}" | tr '/' '_')" "$u" "$u"
done < "$DEST/units.txt"
[ -s "$DEST/loose.txt" ] && pack loose "(loose files)" --no-recursion -T "$DEST/loose.txt"
units=$(($(wc -l < "$DEST/MANIFEST.tsv") - 1))
bytes=$(awk -F'\t' 'NR > 1 {s += $4} END {print s + 0}' "$DEST/MANIFEST.tsv")
printf '{"schema": "pouw-backup/v2", "source": "%s", "units": %d, "bytes": %d, "large_units_left_out": %d, "skipped_units": %d, "linked_to": "%s", "at": "%s"}\n' \
  "$SRC" "$units" "$bytes" "$(grep -c . "$DEST/large.txt" || true)" "$(cat "$DEST/skipped.txt" 2> /dev/null | grep -c . || true)" "$PREV" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$DEST/summary.json"
ln -sfn "$DEST" "$LAST"
cat "$DEST/summary.json"
