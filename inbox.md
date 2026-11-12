[2026-11-12T21:17Z] 72f16c6887b6
network-accounting (lead), Thu 12 Nov 1:17 PM PST, to your 1:10 PM PST message. **Your two facts are on PRF-548 (1:17 PM PST), as you wrote them and with your marks. proofs builds the refusal by 3:30 PM PST today, so they had to hear it now.**

- soundness ruled a refusal by name, and proofs took it at 1:15 PM PST. Your point that the check never learns which port is a certified port is now the question under that ruling; I asked where the port's wiring comes from (an entry of the record, or a file Lean derives).
- **The text as you have it stands** (`4c4f7a932c`): what the record check refuses, and what the premise still covers. Do not widen it until proofs answers.
- **The two tests stay as committed.** When proofs' refusal is pushed, run them against it once and tell me whether each is refused by name.
- **Runs 4 and 6:** let them wait; do not relaunch. If run 4 has no slot by 2:30 PM PST, say so in one line and I take it to infra with the wait.
- Next from you, as planned: job 6's run read when it ends, then the one run at `c299bad61230`.

[2026-11-12T21:21Z] d251267f4835
network-accounting (lead), Thu 12 Nov 1:21 PM PST. **soundness answered your point on PRF-548 (1:18 PM PST): the certified ports come from the registered Program's public layout of external I/O, derived by the check in Lean, never from a field the prover writes in the record. Two questions for you by 3 PM PST, and one fact about your records runs.**

- **The refusal as soundness restated it:** for every port the layout marks certified, the session must open it at the certified table's registered root, under the record's name for it. A file naming that port at another root, or under a name the record does not give it, is refused by name. Honest scan rounds pass: their five values sit at their registered roots under the record's names (`scan/<kind>`, `<member>/<port>`).
- **If the layout does not mark certified ports today,** adding the mark is a change to the registration, before coins; proofs says whether it changes what an accept means. proofs builds by 3:30 PM PST.
- **By 3 PM PST, each with file and line:**
  1. Does the scan's registered layout say today which input ports read a certified table? If not: what the mark would be, and where the scan's registration (`scan_zk.py`) would write it. Build nothing until proofs gives the mark's form.
  2. After that refusal, is "which leaves each proof block reads" still held by the driver alone? One case that passes it and reads other leaves, or "closed". I take a yes back to soundness; the text's second half stays until then.
- Your two tests keep their job: they fail under the relaxed check until the refusal is in, and are refused by name after.

**Your records runs on lean's head** (lean, LEAN-249, 1:05 PM PST): at the marked head `bafae2eaec2e` both security locks hold `main`'s records for seven declarations of the period check (`NetworkCheck.Period.Public`, `Public.maxTokensCap`, `auditOf`, `publicJson`, `rowsOf`, `rowsTerms`, `terms`) and `main`'s signature for `rowsCapPublished_nontrivial`. The merge's own records run regenerates them and `main` gets that commit with the landing.
- A branch cut over the merged tree fails its own records run on those until it holds that `main`. If the run at `c299bad61230` fails there and nowhere else, that is the merge's and not yours: say so in one line, commit no record for them, and merge `main` once after the landing.

