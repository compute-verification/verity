[2026-11-13T16:03Z] 2def54e1b847
ADDRROW2, from proofs (lead). Your entry 1 of 15:46Z is read whole and is on PRF-490 since 16:00Z, with a correction of one count of mine at 16:01Z. Your two NEEDS:

1. THE CHOICE-BUILT FINDER: a third case, not empty. I re-read `StrictCR.lean` 51 to 97 and the lock's `trust.base` line on `main`: as you say. My ruling, put to soundness with a "unless you say no": each lemma that concludes the bare form and feeds `collOf` is restated to give its pair as a function of the two openings, as `LinkFinder.outOf` and `forkPick` do, and the knowledge finders output that pair. It moves records of two deployed guarantees, so it goes to Daniel from me today with your part 2.
2. THE TWO DUO STATEMENTS: outside the ruling, as you read them. I told soundness so.

PART 2, by 18:30Z as ordered, with these added, because it is what Daniel will read:
  a. Every lemma that concludes `Merkle.Collision H` (or another bare form) and is handed to `collOf`, `onePick`, `conflictPick` or `TreeRewind.binds`: its name, file and line, and, by reading its proof, whether the pair is built from the two openings. Say "built", "not built" with what it does instead, or "not read". If one is not built from the play, that is a `FOR DANIEL` heading at once.
  b. For each, the restated conclusion: the function that gives the pair, and what it reads.
  c. `Leaf.bind` (the field of the leaf commitment, read by all 50): whether it can give its pair too (the two leaves' own inputs), and what changes if it does.
  d. What the restatement moves: which definitions change, which of the 50 records move, the two deployed ones named, and whether any listed statement's text changes or only what it reads.
  e. Its size: files, lines, how many pull requests, and what must be compiled to know it holds.
  f. One paragraph in plain words for Daniel: what the premise says today, what its words in the lock say, why nothing is false, and what the restatement buys.
  Nothing is written or compiled for this; it is a sizing.

YOUR BRANCHES. FRAME saw `cursor/proofs-addrrow2-unlinked-scratch-681e` committed to today and it is not among the lines I named. Say in part 2 which of your four branches are live.

THE CARRY ACROSS CHAIN 2 (this replaces the order "PREPARE", which does not come). Chain 2 is on `main` at `4ae17f7eafd7` since 15:30Z. The pass started at 15:53Z: FRAME carries #9681, #9713, #9555, #9441 and #9717 now, and the join at the tip ASSEMBLY names by 16:20Z. Every other line of ours is carried later, on its own, by FRAME, against `4ae17f7eafd7`. The rules for you:
  - Never merge `main` or `origin/main` into a line that was cut before the moves, and never rebase one. Across a move that merge is clean and wrong.
  - You need not merge the join's tip first. A line is carried as it stands.
  - When a line of yours is at a point worth carrying (compiled, pushed), write one line in my inbox: the branch, its full head, and the words "my live line". One branch a piece of work. A scratch branch is not carried unless you name it.
  - After you name it, commit nothing more on that branch. New work goes on a new branch cut from the named head, and is carried later the same way.
  - When FRAME's status table names your carried commit and its new branch: `git fetch origin NEWBRANCH`, then `git push origin CARRIED:refs/heads/YOURBRANCH` (a fast-forward; never force; if it is refused, stop and say so). Read your row of the table for added lines that still name an old path or module, and mend them in a commit on top. From then on work only on top of the carried head, and never merge an uncarried branch into it.
  - Before any ready line: the records run at the carried head, then the suites the change reaches.

