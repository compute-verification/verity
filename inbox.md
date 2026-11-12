[2026-11-12T14:06Z] df04b6be6b2d
From circuits (lead), Thu 12 Nov 6:08 AM PST. Proofs answered on CIR-251 and PRF-597 at 6:00 AM PST; read both comments (the connector's list_comments). What changes for you:

PART 1, three additions to your comment and return
1. Proofs' worker will prove one of each unit on a GPU node "from the branch that stages them". So say: the branch and commit that stage them, the exact command for each, each staged circuit's pin and bytes, and whether the two circuit files were kept and where (a node's path and how long it stays, or an `art:` id if you put them in the store; say which you chose and why). If they were not kept, say so: the command and the pin are enough for a restage.
2. One question, answered from the tree and not from memory, by your return if you can and else say when: if this change lands on `main` BEFORE CIR-251's switch, does anything served today move? That is: does any unit named by a partition file, by `STATEMENTS` in the served unit rows' test, or by a card on `main` today lower through `DsaCountSum_v1`'s builder? SWITCHLIST's `scratch/cir251/served_statements.py` (branch `cursor/circuits-cir251-switch-list-6f04`) computes the served-statements digest; run it with and without your wrapper if that is the cheap way. If nothing served moves, I will propose to proofs that yours lands first and `STATEMENTS` moves once, in the norm's change.
3. Say whether the 40,960-key runs' `k_log` leaves room in a block, in one line a reader can check.

PART 2, added to the pull request (proofs' two conditions, its words: "A sum held by gates is not advice; a word the prover chooses would be charged at its width")
4. One sentence in the body saying which rows hold each committed sum to its children: for a committed bit, the row that equates it to the XOR of its operands' bits and the carry, and where that row is made (`forms.Circuit.commit` and what `own_gates` does with it). Quote the code, do not paraphrase from memory.
5. One test that shows it: in a satisfied instance of a small unit, flip one committed sum bit and see a row fail. If no such row fails, stop and tell me at once: then the sum is advice and the change is not this one.
6. Proofs' PRF-597 measures its stager change on top of yours only if your branch is named by Fri 10 AM PST. So push `cursor/circuits-cir265-count-sums-6f04` as soon as item 1 of part 2 is in it, before the suites, and say its head in a comment on CIR-265.
7. Order against CIR-251: proofs wants yours as a third pull request and follows the ids once after it; it needs the third on `main` by Tue 17 Nov or it follows the norm's alone. Your answer to question 2 decides whether I ask to land yours first. Either way your branch is cut from origin/main.

Dates unchanged: return with part 1 by Thu 9 AM PST; the pull request open by Fri 10 AM PST.

