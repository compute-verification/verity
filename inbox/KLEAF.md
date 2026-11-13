# KLEAF's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Fri 13 Nov 00:30Z: your brief: the wide level-0 leaf's measurement of record (HASH-119), taken over from KMOVE. Tonight, with no card: the merged tree, the Lean verifier's answer on a full-size wide proof, and the H200 job dry-run

**Who you are.** KLEAF, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule: Opus 5.5 does the work). You go on from the worker KMOVE
(bc-36640c07…, on the lead's model, so not resumed). Its note is
`~/.research/notes/lanes/kernels/20261110T0806Z-report-kmove.md` (after `research notes sync --path
lanes/kernels`); **read these sections whole before anything else**, newest first: "HASH-119, Thu 12 Nov
17:17Z: THE SCREEN RAN" (line 108), "HASH-119: at 15:30Z, as it now stands" (413), "HASH-119, Wed 11 Nov
21:25Z: what accepts a wide session" (772), "what Friday's measured pair needs" (1021), "what would make
it faster, and what the H200 run needs" (1141). Its job files are `kmove-jobs/hash119/` beside the note
(`thu/wide-launch.sh`, `thu/make.sh`, `thu/prebuild-wide.sh`, `thu/read-screen.py`, `thu/salt-bytes.py`,
`widesess.sh`).

**The goal** (`kernels/wide-leaf-measured`, due Tue 17 Nov 6 PM PST; it is kernels' part of proofs'
`proofs/spot-bind-term`, and Daniel approved the wide leaf on Wed 11 Nov): Flock's prover commits its
first layer in leaves up to 1,028 symbols wide for tables of 2^32 to 2^35 bits, and **a statement of the
largest size is proved both ways in one run on an H200, each kind's proof accepted by the Lean verifier,
and measured beside today's**: seconds a statement, the salts' bytes, memory on the card and on the
host, proof bytes, the Lean verifier's seconds.

**Where it stands** (each line: by whom):
- The option in the measuring script is on `main`: #9077, merged Thu 11:01Z.
- **The prover's side is #9079** (`cursor/kernels-wide-leaf-7c1e` at `c4bdf9e1ec4a`), kernels' own,
  open and unmarked; proofs reads and merges it after its own verifier change.
- **proofs' verifier for the wide version is #9108** (`cursor/proofs-wideleaf-masked-782c` at
  `16757e170166`, open, unmarked, last moved Thu 13:31Z; "the verifier and its analysis in one").
  On Thu 17:28Z I asked proofs on HASH-119 whether #9108's tip accepts a full-size wide session;
  no answer yet. **Piece C below answers it by running it.**
- **Thursday's screen** (`r20261112-155330-a835`, node A card 6, tree `288196b2d11c`, the head block
  `SparseMlaHeadBlock_v2_NVIS_64_FIRST_false`, pin `4b31e7caba011398`, m = 35, one prover, four
  alternating pairs of 121 sessions): the wide leaf at 0.79 to 0.84 of today's seconds, the proof
  3.66 times the bytes. Every session accepted by the prover binary's own verifier **and by nothing
  else**: main's Lean verifier refuses the wide form, so it is a screen and no result. The salts'
  bytes there are arithmetic on the parameters, not read from the prover.
- benchmarks' rule (HASH-98): a row of record is `main`'s build. So the run of record needs #9079
  and #9108 on `main`. A run before that, on the two merged, can be "accepted by the Lean verifier"
  and is still no row of record.

**Tonight, on CPU nodes only. Name each node (`--on NAME`, after looking at its check line and load;
not `auto` blind).**

**A. The merged tree.** `origin/main` with #9079's head and #9108's tip merged in, on a branch of your
own (`cursor/kernels-wide-record-` and four hex), pushed. Say whether it merges clean; a conflict in
proofs' files is not yours to resolve: say which file and stop that piece. The suites the two changes
touch, at that head. KMOVE's whole `verity-flock` with its slow tests did not end inside 90 minutes and
has no verdict at the screen's head: give it the time or the shards it needs and get one.

**B. The prover's binary**, built from that tree on a CPU node with CUDA as `thu/prebuild-wide.sh` builds
it (the gpu feature, sm_90), its sha256 recorded: the card's machine builds nothing.

**C. Does the Lean verifier at that tree accept a full-size wide proof?** One wide session and one of
today's at the head block's size, made on processors (KMOVE's 21:25Z section and `widesess.sh` show how
its CPU sessions were made, and that the verifier then refused a wide one "only because no program was
given with it": so each kept proof goes with its Program word, as KSTEADY's kept proofs do:
`ksteady-jobs/h115/` and the nine arts on HASH-115 are the model). Then the Lean verifier of the merged
tree on each, by the runner on `main` (`benchmarks/prove/lean_verify.sh`; KHSHARE's
`20261111T2030Z-khshare-prover-walk/launch-lean.sh` shows the launch and the 48 GB it needs; **there its
rule is "the verifier is main's, file for file": for the wide form the verifier is #9108's, so say in
every line which tree verified**). **Report the verdict's own words for each.** If the wide one is
refused: the reason's words, which check, and stop: that goes to proofs through me.

**D. The H200 job, written and dry-run, nothing launched on a card.** From Thursday's
`thu/wide-launch.sh pair` (eight steps, four pairs, one prover, one lease), plus what the record needs
and the screen did not have:
- one kept proof of each kind, put in the store as its step ends, each with its Lean run sent to a CPU
  node at once (KHSHARE's `fri-kept.sh` is the model);
- **the salts' bytes read from the prover's own output**, not computed from the parameters;
- the ten checks of a steady row on every step (`verity_bench.steady` on `main`), each row put as an
  art, unlabelled;
- a start check (the card's lease free, no compute process on it, exits otherwise), one lease, the
  cores named, the prebuilt binary held to its sha256, an absolute end;
- a dry run on stand-ins until every line holds; the files' sha256 beside your note.

**E. In your return:** what a run on the merged tree would show and what only `main`'s build can; the
minutes it needs (Thursday's took 71 of its lease); and whether it would fit **node A, card 2, cores
24-33, Fri 13 Nov 17:30Z to 19:45Z** (kernels released those hours on Thu; read
`~/.research/notes/calendars/node-a.md` for whether a lane has taken them since; book nothing: I do).

**Not yours:** any card or lease tonight; a mark; proofs' branch (read only; push nothing to it);
#9079's branch unless `main` no longer merges with it (then merge `main` into it, the suites again,
and say so). Nothing of the tree cites a run id, a date or a lane's name.

Report on HASH-119 with `research msg linear comment HASH-119 'TEXT' --worker KLEAF` (at most 100 words
a comment), never through the Linear connector's `save_` tools. Note:
`~/.research/notes/lanes/kernels/<stamp>-report-kleaf.md`, a CHECKPOINT line every 30 minutes, synced;
scripts and outputs beside it. **Commit and push before you end a turn: a resume starts you on a new
machine.** If a run would wait or build more than 40 minutes, end your turn naming it; I resume you at
its end.

**Return by Fri 13 Nov 11:30Z** with: the merged tree's head and whether it merged clean; each suite's
verdict; the binary's sha256; **the Lean verifier's verdict on the wide proof and on today's, in its own
words**; the job's files, their sha256 and its one launch line; and E.

## Fri 13 Nov 00:56Z: YOU RUN ON THE LEAD'S OWN MACHINE, not on a VM of your own (my launch left the placement out). Six rules follow from it. Go on with the work

By me just now: your tmux session `keep`, your worktrees `/tmp/kl/rec` and `/tmp/kl/v`, and your
files under `~/.research/notes/lanes/kernels/kleaf-jobs/` are on my machine, and my own notes
sync at 00:55Z committed twenty of your job files under my name (harmless; they are yours). You
are not to be moved: what you have built stays. But the checkout, the notes clone, the tmux
server and the processors are shared with me, so:

1. **`/workspace` (`/agent/repos/verity`) is my checkout, on the coordinator branch. Never
   switch its branch, never commit, stash, reset or clean in it, never run a tool there that
   writes the tree.** Work only in your own worktrees under `/tmp/kl/`, as you do. `git -C
   /workspace fetch` and `worktree add` are fine.
2. **The notes clone `~/.research/notes` is shared.** Sync only your own paths:
   `research notes sync --path lanes/kernels/kleaf-jobs` and `--path` your own note's file; never
   `--path lanes/kernels` whole, and never a reset or a checkout in that clone. If a sync says
   another process holds the lock, wait a minute and try again. I do the same from now.
3. **tmux:** your session is `keep`. Never touch `kernels-lead` (it holds the secrets) or
   `kernels-marks`, and never `kill-server`.
4. **Nothing heavy on this machine:** no build, no suite, no Lean, no prover session here.
   Every one of those is a `research run` on a CPU node you name. Scripts that read, diff and
   write text are fine.
5. **This machine carries a coordinator's rights.** They are not yours: mark nothing, review
   nothing in the queue, push to no branch but your own two, write no calendar row, cancel no
   run you did not launch.
6. A resume does NOT give you a new machine: you come back here, and `/tmp/kl` is still there
   unless I say otherwise. Push before you end a turn all the same.

If any of the six has already been crossed, say so in your next checkpoint, plainly; nothing
is lost by saying it.

## Fri 13 Nov 01:52Z: your return is read: the Lean verifier accepts the wide proof, and the row is BOOKED: card 2, cores 24-33, Fri 17:30Z to 19:45Z, STANDS. One thing to mend in the job before it (the quiet-cards list), then nothing until my go

Read by me: your return and your note whole to section C; the wide verdict's record itself
(`r20261113-010259-ea93`, cpu-12, tree `4824eb289e5b`: `lean-verify.json`, `accepted` true,
`verity/flock-circuit+wide-leaf`, the label); `launch.sh` whole. **Answering in ninety minutes,
by a run, the question proofs had not answered in a day; taking #9108's tip with #9079 for the
verifier's tree when the three-way merge conflicted in proofs' files, and saying which tree
verified in every line; and writing down at 01:00Z, unasked, which of the six rules you had
crossed before you could read them: all three are how this work should be done.** Nothing was
harmed here: `/workspace` is clean and on its branch.

Done by me since: **proofs rung on HASH-119 (01:58Z)** with the verdict, the two conflicting
files (theirs to merge `main` into #9108), the one racing test of `verity-flock` (theirs), and
the question when #9108 is marked. **The calendar's row is on the remote** (notes `c9951e3556`):
kernels, card 2, cores 24-33, 2026-11-13T17:30Z to 19:45Z, STANDS, naming HASH-119, as your
launcher reads it.

**1. Mend before Friday: `QUIET_CARDS=5,7` is card 6's list, not card 2's.** Card 2 sits on the
socket of cores 0-47 with cards 0, 1 and 3. In your hours kernels' own untimed run is on card 3
(lease 17:00Z to 17:45Z at the latest, then its verifier on cores 34-43 to 18:27Z) and circuits'
captures are on cards 0 and 1 (to 19:45Z). **This run asks nobody for quiet** (the row says
so), so a busy neighbour must be RECORDED beside each step and must neither stop the run nor
drop a step's row. Read what `QUIET_CARDS` does in `steady.sh` and the reader (names a step,
voids its row, refuses), set the list to card 2's own socket (0, 1, 3), and make the outcome
"named beside the step, the row kept, the table saying which steps had which neighbour". Dry-run
again as a research run on a CPU node; new `job.sha256`. Say what you changed.

**2. The start may move to 17:45Z.** I have offered memory-accounting card 2 from 16:35Z to
17:45Z (HASH-112, its word by Fri 14:00Z). Say whether the job fits from 17:45Z by its own
arithmetic (`ROW_OPEN`, `NEED_MIN`, the end at 19:44Z), and what it drops first if the lease
runs short. I tell you in the go which start it is, and change the row myself.

**3. On this machine:** the launcher's append of RULES' node A table to
`~/.research/machines.toml` is fine (the file does not exist here yet). The reader runs in your
tmux session `keep`; it is light. Your syncs stay on your own paths, as since 01:00Z.

**4. Friday's words, for every report of this run:** timed, **not of record** (not `main`'s
build: the prover is `main` with #9079, the verifier #9108's with #9079); "accepted by the Lean
verifier of #9108" with the label's own words; each step with its neighbours. Your launcher's
header calls it "the measurement of record": it is not; say so where a reader will meet it.

**5. The go is mine, in this file, at about 17:20Z on Friday.** Then: the launcher's own checks,
the launch, the reader, the first pair's rows and the two kept proofs' Lean runs told to me at
once (I ring proofs), one worker comment on HASH-119 at the end (at most 100 words), your
return. A failed gate or a refused start: nothing relaunched, tell me with the line.

Return when 1 and 2 are done, by Fri 13 Nov 06:00Z.
