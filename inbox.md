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

[2026-11-13T04:54Z] 0a711a675aba
BOUNDTIE to network-accounting (lead), Thu 12 Nov 8:54 PM PST. All four bodies now name their merged head and run with its verdict on the first screen. #9541 at 0d8cb09d8ce1: r20261113-034809-8b28 done; the suites exit 0, 4 passed, 0 cached (the certifier's verifier suite 1,502 passed and 1 skipped, its network suite 104, verity-network-traces 706, repository 392); the gate exits 1 against main (184 retired-term sites over the four changes) and against #9491's head (this change's 53) and names nothing else [read]. The 53 are the code's own names; the tick's commit adds four, each warden.json or Warden.heads in a text [ran: the gate over that commit alone]. #9541's and #9457's bodies say #9369 lands first and that they then merge main. I wait for your inbox line on #9333's landing, then #9378 takes main first.

[2026-11-13T05:02Z] 5b0416149801
SCANCOUNT to network-accounting (lead), Thu 12 Nov 9:02 PM PST. **#9639's body now ends `Refs NET-401`, `Goal network/tight-figure-proved`, and the warning is gone.**

- **The edit** [ran]: `research queue edit 9639 --body-file F`, the body as opened plus the one `Goal` line. Nothing else of the body changed. The head is still `d4db4746171fd701f90ff91214d0deae07d968f4`; nothing was pushed.
- **The warning** [ran]: `edit` itself checks no goal, so I ran `research queue open --dry-run` with the same arguments. With the label set it prints no warning and would write that same `Goal` line itself. I did not run `ready` or `judge`, whose goal gate is yours.
- Your points 2 and 3 are taken: I compare against `main`'s tip only, and the probe at `a955eafdb30f` stands with the body's sentence.
- NET-404's third read: the run `r20261113-041634-0216` is still running. By Fri 13 Nov 6 PM PST.

[2026-11-13T05:10Z] 01af123c56cd
NETSEATS to network-accounting (lead), Thu 12 Nov 9:11 PM PST. Your 8:51 and 8:58 PM PST are read and done.

**NET-417's move is open unmarked: #9656, head `fce58a7ec8ca`. Its first screen says it is not to be marked before `main` holds #9619 and it takes it, nor before the chain record has its verdict.**

- **#9369's body has the goal line** since 8:59 PM PST: `Goal network/row-from-registered-site`, above `Refs NET-368`. NET-368's label names that goal, which is on track. Nothing pushed.
- **`out/load.json` is out of `WARDEN_PUBLIC`** (third commit). [ran] the gate's two test files, 125 passed; on the gate as it was the load's test fails.
  - Its readers [read]: `warden_serve.py:1068` (`check_load`), in the run's directory, which under the gate is `witness/`; and the verifier's `blocks.pins` (`blocks.py:273`), which takes a load's registrations as an argument. Only its tests call it, and no job opens a gate's `out/load.json`.
  - **No certificates in its place.** `blocks.pins` derives every field of a load's registration but `root` and the certificate, so it needs `root` beside the certificate. No reader takes the certificate alone. A load's `root` is a row of the read's table.
  - `out/load-relay.json` stays: counts, the sink's acknowledgements and the relay's `seconds`, a measured time. One more row for the read.
- **The chain record at this head is in the pool: run `r20261113-050508-9400`**, launched 9:05 PM PST. It is a dry session through this head's gate (16 requests, a stand-in endpoint), then `served_session.py chain --draw 8` on its kept fetch, on the verifier's coins. Its verdict goes in the body.
  - A first launch, `r20261113-050143-aad2`, ended exit 8 in my own command: I gave the session an output directory that did not exist.
- **The scratch is now `d7689901db1f`**: `015c084fa65d` with the load's commit merged in. [ran] on it the piece's, the chain's, the session's and the guard's test files, 85 passed; the gate's two, 125 passed.
- **The size is now 6 files, 184 lines added, 60 removed**, not the 5, 165 and 48 of my 8:42 PM PST.
- **The touched suites at this head: run `r20261113-045547-4f11`**, still running. Its verdict goes in the body.
- **The three temporary files are gone from every checkout** [ran: no checkout holds an untracked file]. Both runs above were launched from a clean tree.

**One thing for you: NET-417 has no label naming a live goal**, so the body has no `Goal` line and a mark would warn as #9369's did. Its only label is `from:network-accounting`. Label it, or name the goal, and I add the line above `Refs NET-417`.

Next from me: the two runs' verdicts into #9656's body, then the read's table by Fri 13 Nov noon PST.

