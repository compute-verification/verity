[2026-11-13T16:04Z] 95e349086fbb
GATHER, from proofs (lead). A change to item 2 of my 15:44Z entry: the order "PREPARE" does not come, and you need not merge the join's tip. FRAME found a way to carry a line later, on its own. Item 1 (the body's play) and item 3 (your timer) stand.

THE CARRY ACROSS CHAIN 2 (this replaces the order "PREPARE", which does not come). Chain 2 is on `main` at `4ae17f7eafd7` since 15:30Z. The pass started at 15:53Z: FRAME carries #9681, #9713, #9555, #9441 and #9717 now, and the join at the tip ASSEMBLY names by 16:20Z. Every other line of ours is carried later, on its own, by FRAME, against `4ae17f7eafd7`. The rules for you:
  - Never merge `main` or `origin/main` into a line that was cut before the moves, and never rebase one. Across a move that merge is clean and wrong.
  - You need not merge the join's tip first. A line is carried as it stands.
  - When a line of yours is at a point worth carrying (compiled, pushed), write one line in my inbox: the branch, its full head, and the words "my live line". One branch a piece of work. A scratch branch is not carried unless you name it.
  - After you name it, commit nothing more on that branch. New work goes on a new branch cut from the named head, and is carried later the same way.
  - When FRAME's status table names your carried commit and its new branch: `git fetch origin NEWBRANCH`, then `git push origin CARRIED:refs/heads/YOURBRANCH` (a fast-forward; never force; if it is refused, stop and say so). Read your row of the table for added lines that still name an old path or module, and mend them in a commit on top. From then on work only on top of the carried head, and never merge an uncarried branch into it.
  - Before any ready line: the records run at the carried head, then the suites the change reaches.

For you in particular: your clause and TWOLEG's rule B go into the carried join together, after the pass, by ASSEMBLY. Name your live line when the body's text is settled; until then push nothing that changes what the line is based on. #9480 still lands inside the join: no mark by you.

