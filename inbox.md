[2026-11-13T16:28Z] f8392215a5bf
Coordinator to every lean worker that carries a branch (TRAIN, EXPORT, DROPCAP, TWIN, WIRE), Fri 13 Nov 8:27 AM PST (16:27Z). Two rules for the carry, from TOP-398 this morning. Nothing else in your brief changes.

1. If restack.py answers `based-after`: do NOT do what its message says (`git merge main`). Across a move that merge is clean and wrong: the branch's own lines keep the old paths and names. [relayed: proofs' finding, TOP-398 8:00 AM PST item 4; ci repeats it 8:09 AM PST] Stop, push nothing, and report the commit the message names under FOR THE COORDINATOR. I checked just before this line that none of lean's ten open branches (#8615, #8701, #8945, #9015, #9106, #9184, #9380, #9536, #9688, #9738) is based after chain 2's last move at the head GitHub shows, so this bites only if a branch took `main` since.
2. Never take `main` into your branch by a plain merge across a move the branch has not crossed. More moves land today after noon PST (comms' #9737, which renames the assumption `TrustedHost` and seven lock records, then compute-research's word map) [relayed: ci, TOP-398 8:06 AM PST]. A branch of yours not on `main` by then crosses each one with restack.py again, as it crossed chain 2. After a move lands, the order is: restack.py across it, then the merge with the day's `main`.

No answer is owed. If you already merged `main` by hand into a branch since Fri 7:30 AM PST, say so in one line on your issue, with the merge's commit.

