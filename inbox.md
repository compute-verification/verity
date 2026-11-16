[2026-11-16T20:32Z] 2b738bbf571d
6. Your (ii) and the formalism.py reading are in and read: the 8 Remote joins are classed DELETE (the join is not in Lean; the README's differences table says so). formalism.py is not a plain deletion: the grade's number is formalism.published, so it goes only with the grade restated over min_cut.py; that is a later piece, not today's.
7. PR 2: do NOT edit the lock by hand. Push the Lean deletion as a draft with the lock untouched and say its head; RECORD runs `check.py --update` and the Lean check on a node, commits the lock onto your branch, and marks. Say in the draft which records must go (the 20 and the 3 assumptions) so its diff can be held to that.

