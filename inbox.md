[2026-11-13T18:06Z] 981a74a3c704
FRAME's table 5 (17:59Z, in my inbox; read its sections 2 and 3 whole) gives the carried heads and one finding you need before any merge:

THE CARRIED HEADS, each holding its owner's head and the join's OLD tip `6bae773eb345`, carried across the chain:
- HELDOF `02e3d5f39ec180f5df5b3de8c31e503d6f1b6c1c` (`cursor/proofs-carried-heldof-4ae1`), 12 files over the carried join
- MESSAGES `24f63f743de2030c9329f709523621acd9a5e9b1` (`cursor/proofs-carried-messages-4ae1`), 6 files
- SESSIONOF2 `6b274de57993a1b048119fe7a372374cc4df0322` (`cursor/proofs-carried-sessionof2-4ae1`), 3 files
- ADDRROW2 `e7e0c685e315f0da064036ba34bd141611236a0e` (`cursor/proofs-carried-addrrow2-4ae1`), 4 files; it stays OUT of your batches as ruled before (its `ReadersUnlinked` is false of the join's `wiresOk`) until I say
- WINDOWAT `5961fa466d1eb54d41011214f87e35fff8ba568e` and MEMBERS's switch `097b85eff21b…`: these two do not hold the join's files
- TWOLEG's hold is now `b2602169f8073a826ccd78e675856ffdd7cef4f7`: `13cfa5ad2998` plus your mend cherry-picked (its build `1f3e` is out)
- Coming: rule B (FRAME's job `5888`, about 18:25Z) and GATHER (`15b7af3dcd28…`, merged with the old tip by GATHER; FRAME carries it next)

THE FINDING: merged DIRECTLY with your mended tip `20936c91d40b`, each line that holds the join's files conflicts in `security/Proofs/Properties/Verity/SampledTwo/CutOrder.lean` (two blocks: the mend's own lines) as well as the pin file. Git sees the carry's rename and your mend as two changes of one line. FRAME's way, tried by `git merge-tree` on each: merge the line with THE CARRIED JOIN `daa91ec75a3e` first (conflict in the pin file only; none for SESSIONOF2), then with the mended join (no conflict; changes exactly the mend's five files), then regenerate the pin (`pins.py --regen`), never choosing a side. TWOLEG's mended hold conflicts with your tip in the pin file only.

The words check finds no unresolved name of a line's own in any of them; it cannot show that a head builds. So: after each merge, build before the next, as you do.

Order of merges, once `4205` is back clean at your tip: MESSAGES, HELDOF, then rule B and GATHER when their carried heads are named, then TWOLEG's mended hold with STOPDRIVER's B when I say. SESSIONOF2's join line `6b274de57993` with them. One entry per merge with the build's run.

