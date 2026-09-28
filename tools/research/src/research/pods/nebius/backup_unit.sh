#!/bin/bash
# backup_unit.sh UNIT: back up one large unit of /workspace/pouw (a directory backup.sh left out, over 1 GiB) as its own research run,
# so its custody upload stands alone: a dropped connection restarts only this unit.  UNIT is relative to /workspace/pouw
# (./fill-out/fp4-recheck/mxf4-s20261003).  Records the unit's tarball sha256 in /workspace/pouw/infra/large-units.tsv, which
# the infra lane reads to launch each unit once it has stopped changing, and again only if its content changes.  The tarball is
# stored in 60 MiB chunks (NAME.tar.zst.0000, ...); restore: cat NAME.tar.zst.* | zstd -d | tar -x
#
#   research run --on vy-nebius-2 --project verity --campaign pous-infra -- bash /workspace/pouw/infra/bin/backup_unit.sh ./fill-out/...
set -euo pipefail
: "${RESEARCH_RUN_DIR:?run this through research run}"
U=${1:?usage: backup_unit.sh ./path/under/workspace/pouw}
SRC=${BACKUP_SRC:-/workspace/pouw}
cd "$SRC"
[ -d "$U" ] || { echo "no unit $U" >&2; exit 2; }
DEST=$RESEARCH_RUN_DIR/backup
mkdir -p "$DEST"
name=$(echo "${U#./}" | tr '/' '_')
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
  --exclude=env --exclude=venv --exclude=.venv --exclude=__pycache__ --exclude=src --exclude='*.tmp' -cf - "$U" | zstd -q -3 -T1 > "$DEST/$name.tar.zst"
sum=$(sha256sum "$DEST/$name.tar.zst" | cut -d' ' -f1)
size=$(stat -c %s "$DEST/$name.tar.zst")
split -b 60M -d -a 4 "$DEST/$name.tar.zst" "$DEST/$name.tar.zst." && rm "$DEST/$name.tar.zst"   # single-PUT objects (see backup.sh)
printf '%s\t%s\t%s\t%s\t%s\n' "$U" "$sum" "$size" "${RESEARCH_RUN_ID:-?}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee "$DEST/unit.tsv" >> "$SRC/infra/large-units.tsv"
