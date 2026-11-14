[2026-11-14T04:03Z] 4250cf5ebf02
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 8:03 PM PST. proofs settled condition 2's design on NET-436 at 8:00 PM PST. Nothing of it is built or run yet. One part is ours, and it goes in your Mon 16 Nov noon PST line.

**What proofs builds**
- One-stage refuses, under `R7-output-refused`, an output port read from a record the verifier's verdict log names refused.
- The log is a top-level `verdicts` parameter, `{digest: bool}`, as `Pool.verdicts` is today.
- **A record the log does not name passes**, as in the pool.
- After parsing, the window-history form `{"port": P}` and the explicit `{record: D, port: P}` are the same read, so one-stage cannot tell them apart. Refusing the explicit form for a carried start stays the period check's.
- `Registered` gains one conjunct; `acceptMeans`'s lock record moves and its signature does not.
- Its pull request with a ready note by Mon 16 Nov 6 PM PST, likely sooner.

**What is ours**
- **No caller on `main` passes `verdicts`.** Until ours does, a start is carried past a refused link-window as today.
- **An unnamed record passes, so the check is only as good as the log our caller hands.** A caller that hands none, or leaves out the earlier link-window's record, gets no refusal. That hold cannot be one-stage's. It is ours.

**For your Monday noon line, three answers, each read and not built:**
1. **Which of our programs hands `verdicts`,** and at which call: the scan's driver when a later link-window's record is checked, the period step, or both. Name the function.
2. **What the log names.** My default: every record a start is read from, with the verdict the verifier itself gave it, and never a verdict taken from the prover's files. Say where that verdict comes from in the driver today and whether the driver has it at that point.
3. **What holds that the log was handed and names the earlier record.** My default: the period check refuses a later link-window whose earlier record's accept it does not itself hold, so that condition 1's one log and this are the same refusal, and one-stage's check is the second line. Say whether the period check, as it reads its case today, can see each link-window's verdict, or reads the answers as given.

The two-link-window run does not wait on proofs' change. If the run can hand the log once proofs' branch is up, say what it costs to add that as a second case: a first link-window refused, the second's start refused by one-stage.

Unchanged: Mon 16 Nov noon PST (leaf 0's shape, one Program, what the run needs on a node, where the explicit form's refusal goes), Wed 18 Nov noon PST the run's result, Fri 20 Nov noon PST the branch.

Standing rules, unchanged.

