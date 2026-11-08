#!/usr/bin/env bash
# Verify ebf8's capture against 1ccd's custody, then retire both hard-linked copies through retention (proofs coordinator 14:09Z).
set -u
BY=https://cursor.com/agents/bc-5609d8d8-985e-50b9-9216-15dfd5276546
ART=art:0482d2bb756f3c54cf6c266777e1212922b5f1b4dcccf8e60c2b21a5fb25d4f1
REF=note:20261030T2240Z-report-coordinator-gen4b
PATHS=(/workspace/spot-capture/r20261108-092918-ebf8 /workspace/research/runs/r20261108-100548-1ccd/out/capture)
date -u +%T; df -BG --output=avail /workspace | tail -1
ls /workspace/spot-capture; du -sh /workspace/spot-capture/* 2>/dev/null
for p in "${PATHS[@]}"; do research keep show "$p" 2>&1 | head -3; done
/workspace/venv312/bin/python "$RESEARCH_RUN_DIR/inputs/verify_ebf8.py" || { echo "verification failed: nothing deleted"; exit 3; }
date -u +%T
for p in "${PATHS[@]}"; do
  research keep "$p" --owner @proofs --expires 2026-11-08 --priority low --art "$ART" --why 'redundant with 1ccd export; frees disk for e6e3' --by "$BY" || exit 4
done
research retention rm "${PATHS[@]}" --approved-by @proofs --ref "$REF" --by "$BY" || { echo "dry run refused"; exit 5; }
research retention rm "${PATHS[@]}" --approved-by @proofs --ref "$REF" --by "$BY" --apply
echo "rm rc=$?"
for p in "${PATHS[@]}"; do [ -e "$p" ] && echo "STILL THERE $p" || echo "gone $p"; done
date -u +%T; df -BG --output=avail /workspace | tail -1
