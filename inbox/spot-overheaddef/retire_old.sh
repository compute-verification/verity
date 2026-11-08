#!/usr/bin/env bash
# Retire OVERHEADDEF's superseded captures on vy-spot-glm53 through retention (their metadata is in run r20261108-103046-91f7).
set -u
BY=https://cursor.com/agents/bc-5609d8d8-985e-50b9-9216-15dfd5276546
REF="cursor/verity-proofs-84b5@be445bbd70 inbox/OVERHEADDEF.md:324 (own data: retention rm --approved-by @proofs); proofs coordinator 10:24Z Sun: whole-block replays 72b6 then c343 on node A"
WHY="superseded (capture.py before 275521d357); its metadata is kept in run r20261108-103046-91f7 (run record art:df3e37c1b09e3d1508b9fd52ce29f585f5499db8420962f3f5630473ff3d8c16); deleted to give the c343 whole-block replay (r20261108-102636-bea5) its 40 GB"
PATHS=(/workspace/spot-capture/r20261108-023627-9d0e /workspace/spot-capture/r20261108-031739-9bd2 /workspace/spot-capture/r20261108-031214-95e8)
df -BG --output=avail /workspace | tail -1
for p in "${PATHS[@]}"; do echo "$p: $(find "$p" -type f | wc -l) files, $(find "$p" -type f -links +1 | wc -l) with other links"; done
NOW=$(date -u +%FT%TZ)
for p in "${PATHS[@]}"; do research keep "$p" --owner @proofs --expires "$NOW" --priority low --why "$WHY" --by "$BY" || exit 3; done
research retention rm "${PATHS[@]}" --approved-by @proofs --ref "$REF" --by "$BY" || { echo "dry run refused"; exit 4; }
research retention rm "${PATHS[@]}" --approved-by @proofs --ref "$REF" --by "$BY" --apply
echo "rm rc=$?"
for p in "${PATHS[@]}"; do [ -e "$p" ] && echo "STILL THERE $p" || echo "gone $p"; done
df -BG --output=avail /workspace | tail -1
