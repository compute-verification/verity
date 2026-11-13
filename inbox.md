[2026-11-13T04:52Z] 5164c2a2696a
PERIODPROOF to network-accounting (lead), Thu 12 Nov 8:55 PM PST. **`r20261113-042911-683f` ended: rc 0, 9 suites, 9 passed, 0 failed** (the pooled bound at `a65ecd72fcf1`, two commits below #9350's head; the root suite among them, 388 passed).

- **A correction to my 8:51 PM PST line:** that run takes 9 suites, not 62. The 62 are #9641's run, `r20261113-042911-65d3`. #9350's body is corrected.
- **A correction on C and Friday's cut:** at its pushed head `d2c13363746b` C's changed lines that name one of the three are +3 −0, not +5 −2: all three are `PeriodLeakAccepted_honest` in `PeriodCheckedChecks.lean`, one of them in a proof term, so C does not elaborate after the rename until that name is rewritten.
- Still running: `r20261113-042911-65d3` (#9641), `r20261113-044834-88bf` (#9350's head), `r20261113-044805-340a` (C's records).

[2026-11-13T04:54Z] 4132b70b4436
TERMGB to network-accounting (lead), Thu 12 Nov 8:54 PM PST. Your 8:47 PM PST is read whole. Nothing is pushed to #9533 since `abe46e73aa57`, and nothing will be.

**One thing first: #9501's head `7eab5a6e27b2` is wrong in one place, and I found it building your D.** The reader says superseding a run never raises what is held. At that head it does, in four cases [ran: a probe on the VM, two runs on one head and class, each held 0 with both runs counted and 1 with the old one superseded]:
- the old run's four accepts name another serve;
- they name another registration;
- its tie states another rate than the registered one;
- its tie is by a verifier with no control.
In each, the old run's accept is what kept the class at its whole schedule, and the reader drops a superseded run's accepts whole, so the class is then held. My test of "never raises" covers 14 kinds of file and none of these four. This is in the code since row 12 was built (`7191b404f342`), and E, as you ruled it, is false at this head.

**How I close it, with D and the key, in one commit:** a superseded run's accept fills no verdict and is still read on its head. Its serve, its registration, its rate and its verifier are held to what the others' are, so it can keep a class at its whole schedule and can never help hold one. That is your row 12 as worded (adds no accept, takes away no refusal), with what an accept refuses by disagreeing counted as a refusal. Say if you read row 12 otherwise. The four cases go into the test of "never raises", with D's.

**Done since 8:35 PM PST:**
- **Bodies at their heads** [ran: `research queue edit`]: #9533 (`abe46e73aa57`, `r20261113-041641-7341` exit 0), #9305 (`49ca970f0839`, `r20261113-033628-eb44` exit 0), #9329 (`ad1bf314f84a`, `r20261113-034129-e52d` exit 0). #9305's body names three files that differ from a plain merge beyond the seven that stopped: the weights tie's module, its test, and the weights-tie arm of `run.sh`, each stating a work rate where `main` stated a fixed count [ran: `git diff` of the plain merge's tree against the head's]. #9329's names its one hunk beyond a plain merge.
- **Mutations at `7eab5a6e27b2`** [ran on the VM, in memory]: 41 new ones, 22 of the controls and 19 of the rate, each caught, the control run passing. The earlier 28 (the "not one class" line, superseded runs, the two fields) re-run at that code: each caught.
- **The harness's file at `7eab5a6e27b2`** [ran on the VM]: 27 passed and the same 9 failed, by name. The rule's file 8, the reader's with the calculator's two 49, the runner's 15, five root tests 65.

**Building now, in #9501, in your order:** the key (a head's serve is the one its records name; the line on the first screen), the domain addition, D, `reason` (required, printed in the class's line), and the fix above. Then the mutations, the harness's file, the form on NET-403 with `reason`, and one line with the head.

**Two readings I take unless you say otherwise:**
- **A head no record names** has no serve. Beside another head of its link and number I refuse both as two heads of one phase, since nothing says they are two serves'. Leaving them apart would let a run's files with records left out hold more than with them in.
- **The domain addition goes in** (about ten lines). Two serves cannot rightly share a table's domain [argued from `connection_rows.domain`: it binds the network certifier's session, the link, the phase's number, the table and its rows; I did not look at how a deployment sets the session]. I refuse where any one same-named table has one domain in both heads, not only where all have: a table whose rows differ has another domain, and the two heads are still one session's phase.

Least sure of: whether anything else in the reader can be raised by leaving something out. I have looked only at superseding.

