# KTILE's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 20:56Z: your brief: two pull requests the worker KP2 left written and unopened: the forced consumer-tile gate (stacked on #9258, HASH-120) and `session.sh`'s record of its generator (HASH-127)

**Who you are.** KTILE, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You go on from the worker KP2
(bc-f82d00ff-a2e7-5ef7-9110-7e2482422662), whose handoff is
`~/.research/notes/lanes/kernels/kp2-hash120/HANDOFF.md` (after `research notes sync --path
lanes/kernels`): read sections 1, 2, 5 and 7 first.

**Background.** #9258 (`cursor/kernels-poub-leaf-in-readers-4c1e` at `b7623f9bfa`, marked ready today)
adds a second way to run a shard's PoUB pass in `benchmarks/remote_session` (the "leaf pass"). In it
a consumer warp takes a tile itself (`help_leaf_tile`) only where its CTA is on an SM before any
reading CTA has started. KP2 wrote two commits on the pre-move twin
(`cursor/kernels-poub-leaf-consumer-tiles-node-4c1e`: `87eef0022c`, which lets `rs_leaf_pass` take
`ctas` -1 so that no receipt grid is launched and every tile is a consumer's own, with the gate
`consumer_tiles`; and `5dacbb547b`, one docstring sentence in `pass_mode`). Its card run is done:
F, `r20261112-202411-4a54`, on eight RTX PRO 6000 over 700 GiB: nine checks of nine, every root
today's, no reading CTA started (read by the lead from the record).

**Piece 1 (HASH-120): the follow-up pull request, opened unmarked.**
- A branch `cursor/kernels-poub-leaf-consumer-tiles-` and four hex, cut from #9258's head
  `b7623f9bfa` (or from `main` if #9258 has landed when you start: `git merge-base --is-ancestor`),
  with the two commits cherry-picked from the twin. The twin sits on the pre-move base: if a
  cherry-pick stops, the cause is the layout move's paths (one include path, comments, imports);
  close it so that the commit's own change is line for line the twin's, and show that it is
  (the added and removed lines of each commit, twin against yours).
- **Run, on a CPU node you name** (`--on NAME`; not `auto` blind, not `vy-nebius-cpu-7` blind): the
  sm_120a build of your head with `run.sh`'s flags as #9258's body describes (KP2's draft marks
  this "TO RUN"), with the SASS gate `kp2-hash120/pr/sass_pr.py --gate` on it; that the four pass
  kernels are the same instructions as #9258's build; `benchmarks/remote_session`'s suite and the
  repository's suite at your head.
- **The body** from `kp2-hash120/pr-followup/body-draft.md`, with the TO RUN lines filled from your
  runs and F as its card run (say that F ran the twin's `87eef0022c`, and why that is this
  change). `Lane kernels`, `Refs HASH-120`, `Goal kernels/poub-pass-near-plain-read`. Open with
  `research queue open` (the team skill's landing section). **Unmarked: the lead reads and
  marks.** It stacks on #9258: say so in the body's first lines.

**Piece 2 (HASH-127): `session.sh`'s pull request, opened unmarked.**
- `cursor/kernels-p2-stage-weights-sha-4c1e` at `ee58fefe5b` (one file,
  `benchmarks/torch_gpu/p2_stage/session.sh`: `inputs.sha256` records the tree's
  `served_weights.py` that generates the stream, not a sent copy). Check it still merges onto
  `main`; merge `main` if it does not. Run the suites it touches at its head (KP2's run is in
  `pr-session/body.md`; rerun if the head moves). Open it with the body from
  `pr-session/body.md`, `Lane kernels`, `Refs HASH-127`, `Goal kernels/code-at-overhaul-bar`.
- One worker comment on HASH-127 naming the pull request and what memory-accounting should read
  (its sessions run the script); I ring it.

**Not yours:** any card or lease; #9258's branch (push nothing to it: a new head needs a new
mark, and T and F would have to be said again); marking anything; the driver's frozen files.

Report on HASH-120 and HASH-127 with `research msg linear comment ISSUE 'TEXT' --worker KTILE` (at
most 100 words each), never through the Linear connector's `save_` tools. Note:
`~/.research/notes/lanes/kernels/<stamp>-report-ktile.md`, a CHECKPOINT line every 30 minutes,
synced. If GitHub's API refuses for its rate limit, wait ten minutes and try once more, then say so.

**Return by Fri 13 Nov 01:00Z** with the two pull requests' numbers and heads, each run's id and
verdict, the line-for-line check of the cherry-picks, and what is not shown.
