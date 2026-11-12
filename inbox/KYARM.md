# KYARM's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 17:52Z: your brief: HASH-124, the overhead arm commits y as the served call does, checked off a card

**Who you are.** KYARM, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work, Fable 5.2 plans it). You take
HASH-124 from KMOVE (bc-36640c07-bf19-56ed-9958-139cfa2d230f), who wrote the wide row digest (#9225, on
`main`) and the reading this issue rests on, and who is on other work.

**Why now.** Daniel today: start what follows a landing before the landing. HASH-124 needs lean's merge,
which is not on `main` yet, so it is built on that merge's branch and marked the hour the merge lands.

**The task (read HASH-124 whole first; its "Done when" is the contract).** The overhead arm
(`benchmarks/pouw/pearl_c_sm90/overhead_arm.py`) commits no y at any shape: its launch, gate and
transcript hold A's root, the tile leaves, the flags and y's bytes; `retain` (`:427`) passes no `y=`. The
served call commits y. compute-accounting ruled (CMP-380, Thu 12 Nov 13:10Z) that the overhead of record
must time y's commitment and asked the arm's code of kernels. **Yours, off a card, are the issue's first
three done-conditions:**
1. the arm commits y on the device as the served call does (y's rows widened to FP32 words by a launch,
   their row digests, the tree); its buffers are made in `prepare`; `launch` allocates nothing and calls no
   torch, so a graph replay times it;
2. y's tree is in the arm's gate, in its poisoned buffers and in its transcript (`retain` with `y=`);
   `verify_run.py` accepts the arm's own transcript with y's rows opened, and a control that writes nothing
   into y's tree is rejected;
3. off a card: the arm's y root is `audit.commit_rows`'s at the four linears' shapes and at
   (16, 128,256, 2,048), on the stand-in and the host twin.
**Not yours now:** any card or lease (the H200 slot is Tue 17 Nov); the cubin's pin in `overhead_arm.json`
(it moves with the row of record, not before); any mark; any label.

**The base.** Cut `cursor/kernels-overhead-arm-y-` and four hex from
`origin/cursor/lean-remote-merge-onmain-fc8d` (lean's #9375: `main` with the remote branch merged, as one
commit; it holds `served_y.py`, `verify_run.py`'s y check and #9225). **Never from
`cursor/lean-remote-merge-landing-fc8d` or `remote`:** the queue refuses any head that reaches them
(`git merge-base --is-ancestor origin/cursor/lean-remote-merge-landing-fc8d HEAD` must fail). #9375's
head moves a few times until lean marks it (about 20:00Z today); merge it again before you report a head.
Push only to your own branch, each push in a command of its own (`git -C <checkout> push -u origin HEAD`).

**Pointers.**
- `~/.research/notes/lanes/kernels/20261112T1251Z-report-hash111-overhead-point.md` (after
  `research notes sync --path lanes/kernels`): every line that leaves y out, and what y's device
  commitment needs from its caller. Its line numbers are on older heads: find each again on yours.
- KMOVE's report beside it (`20261110T0806Z-report-kmove.md`: the 12:30Z return and the design of the
  wide path); the card run `r20261112-113022-98b1` (the wide kernels and y's tree equal to the references
  on an H200); CMP-380's comments of Thu 12 Nov 13:05Z and 13:10Z (benchmarks' and compute-accounting's
  answers). `benchmarks/pouw` is compute-accounting's directory: the pull request is theirs to read.
- The skills: `team`, `kernel-engineering` (what a timed run must pass; recording attempts), `testing`
  (suites, their inputs, CPU nodes), `friction`, `smell-check`.

**How.** Heavy work runs with `research run` on a CPU node you name (`--on NAME`, not `auto` blind, not
`vy-nebius-cpu-7` blind), never on your VM. Claim the family before you build
(`research notes claim … --lane kernels --hypothesis "…"`, the kernel-engineering skill). Every claim you
write says how it is known: read, run (the run's id), derived, or guess. Report on HASH-124 with
`research msg linear comment HASH-124 'TEXT' --worker KYARM` (at most 100 words a comment), never
through the Linear connector's `save_` tools. Your note is
`~/.research/notes/lanes/kernels/<stamp>-report-kyarm.md`, a CHECKPOINT line at its top every 30
minutes, synced.

**Return by Fri 13 Nov 02:00Z** (or earlier, at a decision that is not yours), with: the design in ten
lines (where y's buffers are made, what `launch` gains, what the gate, the poison and the transcript
gain, what the control is); what is built and pushed (the head); what a CPU node has shown, by run id;
what is not shown; and whether "the three conditions hold off a card by Sat 14 Nov 18:00Z" stands, or
the date that does. End your turn done or blocked, never on a plan. I resume you.
