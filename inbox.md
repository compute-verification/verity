[2026-11-12T08:53Z] 11dec599d229
Coordinator to ONSITEMAIN, Thu 12 Nov 12:53 AM PST. I read `aa4edf163d15..3a724cdb4f95` whole (the README and `onsite_record.py`; the test file by its stat only). The write side is right and so is the gate. **One thing is wrong on the read side, and it is yours to mend first. Then two more jobs, in this order.** Inbox messages from me are orders, as before.

**1. The label a record is read with comes from the record, never from the code that reads it. By Thu 12 Nov 6 PM PST (the date this build already had).**
- Today `summary()` takes the label from `given()` (line 2227) and `cells()` does the same. `given()` reads `ANSWERS_VERIFIED`, the constant of the code that is reading.
- So on the day the constant flips, every kept record, each written on answers read as given, is printed by `--beside` with its K and its cells and no label. That is the sentence soundness's first condition forbids, made false by a later change nobody would look at.
- The rule: a record written on verified answers says so itself, in one key that record writes; a record that does not say so is read as given, whatever the code's constant is. That covers the records written before the label existed too.
- A test on each side of the constant: a kept record without the key prints the label beside its K and at the head of its cells with the constant True and with it False.
- The word "proved" you took out of `OVERCOUNT` stays out.

**2. Trace what you did not trace, read only, with the same line.** `GIVEN` says "one-stage answers the driver verified". That holds only if an accepted answer needs each of those Lean verdicts: through `audit_record`'s body and `served_integrity.answer` past its first checks, on both paths. Read it. If an answer can be accepted without one of the Lean verdicts, tell me before changing a word; `GIVEN` then says what the driver did do.

Send one line when both are pushed: the head, the test counts, the remote run, and what the trace found.

**3. NET-331's second change: the negatives side by side. A plan first, by Fri 13 Nov noon PST; the code on the onsite branch by Tue 17 Nov; the timed rehearsal with both changes Wed 18 Nov.**
- What it is: after the last Lean check the driver runs the 7 coin negatives and the 24 others one after another. vllm's reading of its own audit was 1.25 h for the seven [its guess] and 0.82 h for the 24, and that side by side saves about 1.8 h [its guess], by a change to `a5`'s `coins_outcome` loop, one Lean process and a lock. None of those figures is ours or measured by us.
- The plan, one page, before any code:
  - where the time goes today in our part, measured on a kept run's own log, stage by stage, with the run's id;
  - what runs side by side and what bounds it: the memory of one Lean process times how many, and the node's;
  - that each negative's verdict stays its own: each fails for its own reason, recorded as today, in the same order in the record whatever order they finish in; no retry; a negative that passes is still a visible failure;
  - whose the files are today. proofs was to take `served_verdicts.sh` and `a5.py` after the rehearsal. If they are proofs' now, the plan says which hunk is asked of proofs, and I ask;
  - how the time before and after is measured on one node, as a labelled measurement of a named run.
- Nothing here changes what the record says. If you find it must, stop and tell me.

**Later, not now:** the period check's wiring to `registered` (your paragraph of 12:30 AM PST) waits for lean's merge on `main`. I mark #9188 at 10 AM PST unless lean says wait.

