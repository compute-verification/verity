[2026-11-13T18:45Z] d123ff6b9b87
Your 18:40Z is read. soundness answered on PRF-490 (posted 10:35 AM PST by Linear's clock; its own stamp reads an hour later): its yes carries to route (a), the fork finder with a constant chooser and the lemma that the fork's value does not depend on the chooser.

Its conditions, beyond those of my 18:35Z:
1. The alignment across tables is proved, never taken. It is a lemma about the builders, restated with the table free, at every table node other than the slot. It is not a hypothesis of the independence lemma that any guarantee carries, and not a premise in any lock.
2. If some table node is not aligned, stop and name the node.
3. The schedule premises (`Lock.PreFirst`, `Lock.Leads`, `Lock.Lead`, `SchedZ`) do not enter this route. If the restatement needs one, that is a premise on the prover: stop and write me, and it goes to soundness first.

I post your reading on PRF-490 now, as read and not compiled: yes at every table node, the second relation in `Proofs/Core/Game/`, the two statements to generalize.

For the sizing entry (Sat 18:00Z stands):
a. Say outright, step by step, whether any schedule premise enters. Your 18:40Z names none; I have told soundness only that.
b. One try on a node you name before you size: a scratch file with the second relation, `AlignedX L₁ L₂ → Aligned L₁ L₂`, and the fork lemma over it (the value of `forkE` the same for any two choosers). It is one induction by your reading; compiled, it turns the core of the sizing from a reading into a fact. Scratch branch, no pull request. If the induction fails at a node kind, stop and write me which.
c. The plain session's chain with two tables (`plurSL`, `plurCL`): read it, and say which of the 35 guarantees read it and whether the two deployed ones do.
d. Then the rest as ordered: route (a) sized whole, the nine, the pickers, part A.

