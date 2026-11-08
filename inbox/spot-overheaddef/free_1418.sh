#!/usr/bin/env bash
# Retire 1418's capture (its opened-layer tensors) on vy-spot-glm53 through retention (proofs coordinator 14:16Z); its metadata
# stays in run r20261108-140822-e566, whose out/ holds hard links of it.
set -u
BY=https://cursor.com/agents/bc-5609d8d8-985e-50b9-9216-15dfd5276546
REF=note:20261030T2240Z-report-coordinator-gen4b
P=/workspace/spot-capture/r20261108-102600-1418
E=/workspace/research/runs/r20261108-140822-e566/out/r20261108-102600-1418
date -u +%T; df -BG --output=avail /workspace | tail -1; du -sh "$P"
echo "e566 metadata files before: $(find "$E" -type f | wc -l)"
research keep "$P" --owner @proofs --expires 2026-11-08 --priority low \
  --regenerate "rerun r20261108-102600-1418's job (inbox/spot-overheaddef/launch_block_72b6.sh on the inbox branch)" \
  --why "1418's opened-layer tensors: its verdict is recorded and labelled, its metadata is preserved in r20261108-140822-e566, and rerunning 1418 regenerates them; frees disk for e6e3, ebb0 and HASHCUT" \
  --by "$BY" || exit 4
research retention rm "$P" --approved-by @proofs --ref "$REF" --by "$BY" || { echo "dry run refused"; exit 5; }
research retention rm "$P" --approved-by @proofs --ref "$REF" --by "$BY" --apply
echo "rm rc=$?"
[ -e "$P" ] && echo "STILL THERE $P" || echo "gone $P"
echo "e566 metadata files after: $(find "$E" -type f | wc -l)"
date -u +%T; df -BG --output=avail /workspace | tail -1
