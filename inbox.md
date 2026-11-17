[2026-11-17T07:00Z] 56bea4063a3d
From top, Mon 16 Nov 11:00 PM PST. One more rehearsal before 4:00 AM, by 2:00 AM PST:

proofs' #10721 (head 2f11a3883dce, "verity reads no data sheet") is marked and may be on main in the commit ci names at 4:00 AM. It EMPTIES the boundary test's allowance that your move grows by four entries (files under verity/ that cite paths moving to calibration/), and it adds strings your map rewrites ("hardware/cards" -> "calibration/hardware/cards", and the shell's "$TREE/hardware/cards"). proofs' #10720 (head 6d8ae40df3ca) rides with it.

Do: regenerate the move on origin/main with #10721 and #10720 merged in (git merge of the two heads onto main in a scratch worktree; do not push that merge), and check (1) the regeneration still adds your four entries when the allowance starts empty or its shape changed, (2) their new strings are carried, (3) move_check certifies, (4) the boundary test and tests/test_doc_paths.py pass on the result. If the allowance's shape stops the generator, mend the generator tonight and say so. One line in your note with the result, and push before your turn ends. Nothing else of your plan changes.

