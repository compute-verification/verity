# KWALK's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 21:08Z: your brief: Friday's window on node A's card 6 for the head block with the level walk (HASH-112), from KHSHARE's handoff. Tonight: read it, rehearse it, read five Lean verdicts. No card before the go

**Who you are.** KWALK, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You run a prepared job that
the worker KHSHARE (bc-0242b8c1-6216-52c0-bd2b-285078e870aa) designed, ran on three cards today and
wrote down for someone who was not there.

**What it is.** Flock's prover builds the deepest part of the head block's witness on the host; the
"level walk" builds it on the GPU, bit for bit the same (HASH-112, goal `kernels/witness-on-the-card`).
Today it was timed on three H200s at once: four provers 2.097 → 1.592 GPU-seconds a statement, two
provers 2.182 → 1.587, one prover 2.629 → 2.440, two provers with proofs' keyed salts 1.660 → 1.163
(22 rows, each passing its ten checks). **Friday is the repeat on a quiet socket: card 6 alone,
cores 72-81, Fri 13 Nov 15:45Z to 19:45Z, nothing timed before 16:15Z** (proofs holds card 7 and
that socket's other cores until then). proofs prices its cost model from these rows for Daniel's
top goal, so the window has one try.

**The handoff is the brief:** `~/.research/notes/lanes/kernels/20261111T2030Z-khshare-prover-walk/FRIDAY-HANDOFF.md`
(after `research notes sync --path lanes/kernels`; 609 lines, sections 0 to 11). Read it whole,
section 0 (the rules that bind the window) and section 11 (what is not verified) twice. Use
`WORKER=KWALK` and `INBOX=KWALK` where it says `<YOUR-NAME>`. Where this file and the handoff
differ, this file wins; where a newer section here and an older one differ, the newer.

**Tonight, with no card and no lease:**
1. **The rehearsal of section 10**, on stand-ins and CPU nodes as it says, until it holds on
   your VM: `fri-rehearse.sh` and the launcher's cases, with `fri.sha256` checked first (37
   files). Change no job file. If a pinned file must change for the rehearsal to hold, stop and
   say which and why: the pins are what Thursday's runs and Friday's go rest on.
2. **Section 11's list, turned into what you will watch in the window's first ten minutes:** the
   node script has run on stand-ins only since its two fixes (the fit look judges memory only;
   four provers get two warm sessions); the launcher's three look refusals are unmet; the
   reader's own put of a kept proof has only run dry. For each: what you expect to see in the
   log, and what you do if you do not.
3. **The five Lean verdicts of Thursday's kept proofs** still waiting on `vy-nebius-cpu-12`
   (`r20261112-191526-4a95`, `…-191607-530e`, `…-192137-6432`, `…-193046-30e0`, `…-194304-84f2`;
   the sixth, `…-190939-3668`, is accepted): read them with `bash lean-verdict.sh RUN…` (exit 1 on
   a refusal, 3 while any is not in). When all are in, or at your return, one worker comment on
   HASH-112 with the six verdicts (`research msg linear comment HASH-112 --file F.md --worker
   KWALK`). **A refusal: post it and end your turn at once.** Send no Lean run again.
4. **Questions.** Anything in the handoff you cannot follow or that its files contradict: list
   it in your return. KHSHARE can be asked once more through the lead.

**Return by Fri 13 Nov 04:00Z** with: the rehearsal's verdict and what it ran; the watch list of
item 2; the Lean verdicts so far; your questions. **If the rehearsal has not held by then, say so
plainly**: the lead then gives the window back to KHSHARE. You are resumed at about 15:25Z on
Friday for the go.

**Not yours:** any card, lease or run on node A before the go; cards 2 and 3 on Friday (another
worker's rows); the walk's branch (it stays at `f9fc32ff3fc4`); any label; any pull request.

## Thu 12 Nov 21:33Z: the rehearsal held, so the window is yours. Your six questions answered; one look at node A tonight, now

Read: your return and your note whole (the watch list is right, and B2's reading of the fit's
rule against the handoff's "memory only" is a real catch); the handoff's sections 1, 2, 8 and 9;
`launch-fri.sh`'s look (lines 339-343) and `FIT` (293-301); `launch-lean.sh`'s tree rule;
`lean-runs.tsv`; the calendar's Friday rows; the CPU nodes' load at 21:30Z, by me.

**1. vllm and 15:44Z: I do not ask vllm for a minute.** Its row is its own until 15:45Z (the run
of record for Daniel's goal, due Fri 17:00Z); ours opens at 15:45Z. As you wrote it: from 15:44Z
launch the same line the first minute the look reads `LEASE 6 free` with no `APP` entry; wait,
kill nothing. Nothing is timed before 16:15Z, and the gate and the fit need about 17 minutes, so
a launch by 15:55Z loses nothing. **Past 15:50Z still held: tell me with the `LEASE 6` entry and
I ring vllm.** vllm writes its hours in its row by Fri 02:00Z; I read it and say in the go when
its job stops.

**2. The Lean node: I name it in the go (about 15:20Z), from a look at the nodes then.** Tonight
at 21:30Z: cpu-12 has six in its check line (benchmarks' one, then our five); cpu-13 and cpu-6
have none and over 540 GiB free. If the go names a node, put `LEAN_NODE=<it>` in front of the
reader's line and `NODE=<it>` in front of `lean-verdict.sh` for Friday's two; if it names none,
cpu-12 as written. Thursday's five: none sent again; read them first thing at the go.

**3. A fit that is NOT-FIT for a reason other than memory.** Two cases, and only the first is
yours to act on:
- **(a) The look was cut at its 900 s cap and nothing else is wrong**: every session that ended
  was accepted, refused 0, no prover out of device memory, `unit_walk [True]`, no line saying the
  walk is off, the peak under 140,000 MiB, and the `NOT held:` text names the cap alone. **Then
  you have my word now for one relaunch with `FIT=none`:** write the FIT line into your note
  first; cancel the run by its id; wait until the look (`DRY=1 LOOK=1`) reads card 6's lease free
  and no scratch; launch the same line with `FIT=none` in front, once; start the reader on the
  new id; tell me in your next return with both ids. (Thursday's three four-prover pairs ran
  with `FIT=none` and two warm sessions and peaked at 138,110 MiB of 143,771; the dry run's case
  H2 covers the line.) A second failure of any kind: no third launch.
- **(b) Anything else** (a refused session, a nonzero exit, the walk off, no row, a prover out
  of memory, or a line you cannot place): **relaunch nothing.** The two- and one-prover steps
  run on. Tell me at once with the FIT line and end your turn; I answer within minutes. Those
  reasons say the prover is wrong on this card, and four provers without the look would time it
  wrong.

**4. The Lean verifier's commit: as built, no pin, no file changed.** The runner's tree is
`main`'s head at each put, and `lean-runs.tsv` records the verifier's four directories' git
trees (`verifier_trees`). Thursday's six were launched from four commits of `main` (`0582899c`,
`4a901744`, `ab3f1d53`, `5f0c008a`) and all six read
`20e03d443f4b,259b9cc157ec,7d23c577923a,17ca71762f0c`: one verifier. **In the kept proofs'
comment say both `main` commits and whether the two `verifier_trees` are equal**; if they differ,
say which directory moved and tell me. Nothing more.

**5. The first pair's comment: HASH-112 only.** Tell me at once with the two rows' arts; I read
them and ring proofs on HASH-123 myself, as on Thursday.

**6. The look at node A: run it once tonight, now, before 01:00Z** (`DRY=1 LOOK=1 AT="2026-11-13
15:44" CARD=6 SRC=… bash launch-fri.sh rows`; read-only, launches nothing; the calendar's rule 4
is that every run looks before it starts). vllm's rehearsal holds every card until 01:00Z, so
tonight's look reads a HELD card: that is the point. It shows the transport, and one of the
refusals never met, 18 hours ahead instead of 20 minutes. Write down exactly what it prints:
- the `LEASE 6 …` entry in the form the launcher's filter expects, and whose it is;
- the `APP` entries: **card 6's uuid only?** If they name other cards' uuids, or there is no
  `LEASE 6` entry while the card is plainly held, the look misreads: **tell me tonight**. The
  launcher is pinned and KHSHARE wrote it; I ask KHSHARE for the fix and you rehearse again.
- its exit code and its `REFUSED` line, if any.
Not between 01:05Z and 02:55Z (memory-accounting's timed passes) and not after 06:15Z (vllm's
runs of record) until the go. Once after the go, as the handoff has it.

Then return, with the look's lines. After that you idle until the go at about 15:20Z.
