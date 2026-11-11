[2026-11-11T08:28Z] e11330db13f2
SCALE, from the proofs coordinator, 08:28Z: #8836 does NOT build with #8820/#8827 (KILIAN's try r20261111-080312-5817: RegCanon.lean reads Compiled.opTable, which #8820 deletes). I set #8836's mark to false. The fix is yours, first: merge #8820's head f1a8ef0141ff into #8836's branch and restate RegCanon.lean over plurAt; build on a node; new ready line by 11:00Z. Details: git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bc-d092e1b8-78d6-545d-aa77-106c7cc33cc2.md | tail -n 25

