# For KHSHARE (HASH-100), from the kernels lead

## Tue 10 Nov 05:45Z: runs for a later window start themselves on the node

Timers may not wake the lead (INF-666), so no run's start may depend on an agent being awake when a window opens.

- **Launch a later window's run ahead, with a start guard.** The run's command first waits for the window's opening
  time and does nothing else until then: one sleeping shell pinned to core 95, nothing staged, no lease, no lock.
  Compute the wait from an absolute UTC time, never from "in N hours".
- **Then, in this order:** refuse and exit if the time left before the window closes is less than the run needs;
  take `flock /dev/shm/kernels-node-a.lock`; check the time left again; take `gpu_lease.sh 1 --wait --max-min N` with
  N no longer than the time left; stage on `/dev/shm`; run; remove what was staged, on failure too (a trap).
- **Custody:** give `research run` a `--custody-ttl` longer than the wait plus the run.
- **Windows (UTC):** Tue 10:30 to 16:00. Wed 04:30 to 14:00: open at 04:30, not 03:00, unless the lead writes here
  that proofs released the first 90 minutes.
- **Say what is queued:** comment each queued run's id, its opening time and what it runs on your issue, so the lead
  can cancel it (`research cancel`) if proofs or infra withdraws a window.

## Tue 10 Nov 05:45Z: so launch the two-size run yourself, guarded, when it is ready

This replaces "launch nothing on the GPU node in this turn" and "I resume you at about 10:25Z". When the job is ready
and its no-GPU checks pass, launch it in this turn with the start guard above: opens Tue 10:30Z, closes 16:00Z, at most
45 minutes of one card, cores 64 to 95, `GPU_LEASE_WHO=kernels-khshare`. Then end your turn naming the run id. proofs
may still say no on HASH-100 or INF-659 before 10:00Z; the lead cancels the run if so. If the job is not ready, say what
is missing and launch nothing.

## Tue 10 Nov 06:46Z: three changes before you queue the run (or cancel and requeue it if you already did)

1. **Open at 13:15Z, not 10:30Z.** KSTEADY has three runs queued in the same window, opening 10:30Z, 11:30Z and 12:15Z
   (HASH-98, 06:40Z), and they come first. Yours takes the kernels lock at 13:15Z, waits behind theirs if one still
   holds it, and must end by 15:55Z.
2. **proofs' condition for a run that starts itself** (INF-659, 05:52Z; read it): at its start, after the kernels lock
   and before anything is staged, check that no compute process of another user or lane is on the cards it will use
   (`nvidia-smi --query-compute-apps`) and that the node's lease is free. If either fails, exit non-zero, stage nothing,
   and do not retry later on its own. Tonight a vllm run took all eight cards inside kernels' window, so this check is
   not a formality.
3. **proofs asks for one more measurement, the one it needs most** (HASH-100, 06:29Z; read both of its comments there):
   whether a block's seconds follow its cells. No run has timed one type at two block sizes. `fp8-coord` at K = 2048
   (blocks of 2^23 bits) and K = 6144 (2^24) are both staged in proofs' run r20261110-054930-e53e. Add them to your run
   if the whole of it still fits 75 minutes of the card: each at steady state as your three sizes are, same prover,
   same cores. If it does not fit, queue it as a second run after the first and say so. Report seconds a block against
   cells a block for both, beside the fixed and marginal split at 256, 512 and 1,024 blocks.

proofs has no objection to the run (HASH-100, 05:00Z) and now reads the lever as a cut in cells, which is circuits' and
proofs' work, not the prover's speed. Your run's job is the two measurements above.

## Tue 10 Nov 06:59Z: open at 14:15Z, not 13:15Z

proofs has cores 48 to 95 on node A from 12:00 to 14:15Z (and one card from 13:50Z), which a run of yours at 13:15Z
would share. Kernels' Tuesday hours are now on INF-659 (the lead's table, 07:02Z): yours are **Tue 14:15 to 15:55Z**,
GPU 7 else 6, cores 64 to 95. vllm has all eight cards from 16:00Z, so the lease must end by 15:55Z and the run must
refuse to start if less time is left than it needs. Check at your start only the cards you will use (7 and 6) and
their lease, as proofs' condition says. If the whole run with `fp8-coord` needs more than 95 minutes, run `fp8-coord`
first (proofs needs it most) and move the three sizes of `moe-w13` to Wednesday (04:30 to 12:00Z). If you already
queued a run for 13:15Z, cancel it (`research cancel`) and queue again.

## Tue 10 Nov 07:31Z: two rules from infra, for every worker (INF-666, INF-671)

1. **Every resume replaces your VM, so unpushed work is gone.** Before you end a turn, push everything you would need
   again: commits to your branch, your note and any job scripts beside it in the notes (`research notes sync --path
   lanes/kernels`, last line `; pushed`), results in the store. Keep nothing only in `/tmp`, your home directory or an
   uncommitted file. Set no timer of your own: a worker with a timer pending gets doubled when it is resumed. When you
   wait, end your turn naming the run; the lead resumes you.
2. **Never list command lines, even filtered:** no `ps ... args`, `ps -ef`, `ps aux`, `pgrep -af` or `pgrep -a`. The
   exec daemon's arguments hold tokens. To find a process use `pgrep -f PATTERN` (pids only), and kill by pid.

## Tue 10 Nov 07:54Z: STOP. Daniel pushed this goal back; push what you have and end your turn

Daniel, Mon 11:44 PM PST (top on TOP-394, 07:50Z): a fast check and the repository's new layout come first, every other
goal is pushed back, and every lead stops each worker on a pushed-back goal that is more than 2 hours from its result.
Yours is HASH-100, which serves goal 6, and its next result is hours away. So:

- **Start nothing.** Do NOT queue the node A run. HASH-100 keeps its answer from the stored profile; the two-size run waits.
- **The lead cancels your runs itself, now:** r20261110-072924-87f6 and r20261110-074607-5c5d (the two CPU runs). Do not relaunch them and do not launch others: the pools' slots
  are for landing tonight.
- **Push everything a later worker needs** (job scripts, staging notes, the launcher) beside your note in the notes
  repository, write one final CHECKPOINT that says exactly where a later worker picks up and what is stored where, sync,
  and end your turn with a three-line report. Take no more than 15 minutes over it.
- Nothing you did is lost: the work resumes when Daniel lifts the hold, from your note.

## Tue 10 Nov 16:46Z: HASH-100 is done and closed; nothing more is yours

Your node A run (r20261110-160454-6084) and its comment are read and checked against `out/fit.md`; the lead closed
HASH-100 at 16:45Z. This section replaces the 07:54Z STOP, which the 15:35Z brief had already lifted: you were right
to take the brief. Card 5 with cores 24-47 is KSTEADY's from 16:50Z. Launch nothing; if you are ever resumed, read
your note's first line and wait for a new section here.

## Tue 10 Nov 23:58Z: a new issue, HASH-109: where a fused expert block's steady second goes

Daniel asked every lead (TOP-394's thread, 23:52Z) to use what sits idle and look ahead. benchmarks' command now
prices 91.4% of Flock's proving from KSTEADY's steady rows, and on the new census the two fused expert types are
most of it: 1.4848 and 1.4420 GPU-seconds a statement at one prover, the card busy under half the time. Nobody has
profiled them. Read HASH-109's brief; this section is how.

- **What:** the two fused types of benchmarks' rows 3 and 5 (specs and pins in proofs' comment on HASH-98, 18:44Z;
  stage at full from `d7ab9059c3a5f94093759753d23b06a4872026c0`, as KSTEADY did: its `ksteady-jobs/` show how), at 1,
  2 and 4 provers, on **12 cores, one card's share of the node**. A steady statement's wall time split: the host's
  witness build, GPU kernels by HASH-100's classes, copies, the waits, idle card time; the prover's own phases
  beside it; how the split moves from 1 to 4 provers. Then the three largest levers, each with the most it could
  save, marked inferred, and what would have to change.
- **Where:** node A, **card 3, cores 0-11, Wed 00:15Z to 03:30Z** (the calendar's row, pushed 23:57Z). The same start
  check under the kernels lock, one lease on card 3, an end time before 03:30Z. Cards 0 to 2 are not yours. KSTEADY
  times on cards 4 to 7 meanwhile: say what ran beside you, and keep half your processes without nsys as before.
- **Checks a timed number needs here:** every session accepted, no CPU throttling (the cgroup's counters before and
  after: Monday's prover run on a rented pod was throttled and nobody saw), the card clear at start and end, the
  verifier named (Rust's in loopback, not Lean).
- **Staging first on a CPU node** if it saves the card's time, as you did for HASH-100; a run that waits on cpu-3
  or cpu-6 goes to `vy-nebius-cpu-12` instead (`inbox/RULES.md`, "Runs").
- **Report on HASH-109 by Wed 04:00Z:** the run ids with their verdict lines, the table, the three levers, in at
  most 150 words for proofs and benchmarks; exports and summary in the store; the note synced. No change to the
  prover, no pull request, no label on any row.

## Wed 11 Nov 01:00Z: one line more for your report: 12 cores against KSTEADY's 24

infra (INF-659, 00:53Z): node A's CPU quota is 81.6 cores, 10.2 for each card, and cannot be raised. KSTEADY's
steady rows for these two types were timed with 24 cores for the card, busy under half the time, so they are bound
by the host, and your runs on 12 cores are the first reading of what a card's own share gives. Nothing to launch
for it. **In the report:** GPU-seconds a statement at 1, 2 and 4 provers on your 12 cores beside KSTEADY's
labelled rows 3 and 5 at 24 cores (the same pins; its figures are on HASH-98), the card's busy fraction beside
each, and one sentence on whether the host's cores bound the rate. Say 12 cores, not "a card's share": the share
is 10.2. (The lead's `inbox.py` test message of 00:33Z, token `KT-2e2964`, is in `inbox/RULES.md`: one line in
your next HASH-109 comment says whether its text reached you.)

## Wed 11 Nov 02:45Z: HASH-109 is read and closed. Two things next: an answer for proofs tonight, then HASH-112

Your result is relayed to proofs and benchmarks (HASH-98, 02:30Z). The lead read two of your runs' records; the
figures are yours. proofs answered at 01:49Z: lever 2, the witness on the card, is kernels'; lever 1, the salts, it
puts to Daniel on Wednesday at 16:00Z. Read `inbox/RULES.md` ("Agents, today") first.

**1. For proofs, on HASH-98 by 05:00Z (it needs it by Wed 14:00Z).** From what you already have, no new GPU run:
- the salt bytes a statement draws on row 3's and row 5's types (leaves, bytes a leaf, by level if it differs),
  and how many `getrandom` calls that is;
- what your 1.46 to 0.84 assumes: salts at no cost. Then the same bound with the salts expanded by SHA-512 from one
  key a session, on the host (state the rate you assume and where it comes from, measured or looked up) and on the
  card (from HASH-100's hashing rates), and with a cheaper expander if one is already in the tree (ChaCha20 is:
  `verity.ml.crypto`). Each marked inferred, with the arithmetic.
- At most 120 words on HASH-98, the detail in your note.

**2. HASH-112 is yours** (read its brief): the witness of the two fused types built on the card, bit for bit the
host builder's. Tonight only its first item, **the design**, due Thu 12 Nov, with a first draft in your note before
you end this turn:
- where C-Flock's prover builds these units' witness today, by file and line (`verity/ml/flock/live/`; the move
  tonight renames it to `verity/protocols/flock/prover/`: read `main` as it is, and say the commit), what
  `FC_HOST_WITNESS` selects, and why these two types take the host path;
- what a device evaluator must compute for them, its inputs and outputs, where the 8.6 GB a statement comes from,
  and what stays on the host;
- how it is held to the host builder on every word (a CPU twin in the suite, the card's check), and the first
  failing word reported;
- **the list of what must change in the prover outside the kernel**: that part is proofs', and proofs asked to be
  told on the issue;
- what could be built and checked on node 2's cards or a CPU node before an H200 is needed, and how many H200
  hours the rest takes.
- No code tonight, no GPU. Card 3 is not yours any more (the calendar).

**The move lands tonight** (about 03:45Z or later) and renames the prover's directory. Branch nothing before it.
The agent cap falls to 100 when it lands: leave your note so that a fresh worker could go on from it, and end your
turn when the draft and the answer for proofs are posted.

## Wed 11 Nov 02:16Z: both are read. Next, on the lead's resume after the move: the Rust twin

- **Read by the lead:** your answer for proofs (HASH-98, 02:02Z) and the design's draft (HASH-112, 02:13Z); two
  of its lines checked on `main` (`gpu_circuit.rs:308,386`, `flock-circuit.rs:4328`). The lead asked proofs
  (HASH-112) who writes items 1 to 8, with the default that you do and proofs reviews; its answer is due Thu 12 Nov
  18:00Z.
- **When you are resumed** (after the move is on `main`, and when the agent count allows): your proposed first
  piece, as you wrote it: `unit_ab_levels` beside `unit_ab_into`, equal on every word in the crate's tests, and
  the small command that prints a staged circuit's level schedule. A branch from the moved `main`, no CUDA, on a
  CPU node. **Its first result is the two units' level counts and rows a level**, on HASH-112: that decides
  between the level walk and the native seeds.
- Touch none of items 1 to 8 until proofs has answered or its date has passed.
- `FC_DEV_PREFETCH=1`: the lead asked proofs whether the rows of record should have it on. Do not change KSTEADY's
  rows.

## Wed 11 Nov 02:21Z: proofs says start now. Tonight: the Rust twin and the two units' level counts

proofs (HASH-112, 02:20Z): you write items 1 to 8 on your one branch with the kernel and the Rust twin, and proofs
reviews; Mon 16 Nov for the first type on a card stands. This replaces "touch none of items 1 to 8" and "branch
nothing before the move" in the two sections above.

- **Tonight's piece, and only it:** `unit_ab_levels` beside `unit_ab_into`, equal on every word in the crate's
  tests, and the command that prints a staged circuit's level schedule. One branch from `origin/main` as it is
  (`cursor/kernels-witness-card-<hex>`), no CUDA, CPU nodes only (`inbox/RULES.md`, "Runs":
  `--on vy-nebius-cpu-12` for a quick run). **Post on HASH-112 the two units' level counts and rows a level**,
  and which evaluator form they point to. Then end your turn, note pushed.
- **The move lands tonight** (not before about 03:45Z) and renames `verity/ml/flock/live`. Your branch crosses it
  once, by the recipe top posts (HASH-107 carries kernels' branches); say on HASH-112 that it is a pre-move
  branch. Open no pull request before it has crossed.
- **proofs' terms for review:** name the branch on HASH-112 when the third witness mode builds, and again when
  the self-test's third arm passes; its reader answers within a working day. The pull request is marked by
  proofs, on the lead's word that the read-back compare passed on a card.
- **Item 9 is ruled for now: the ZK server's refusal stays.** So a figure with the card's witness is a figure
  without the firewall, is labelled so, and replaces no served row. SERVEDCARD says by Wed 22:00Z what the
  firewall would have to read back, and whether `FC_DEV_PREFETCH` was set in the steady rows.
- Items 1 to 8 themselves start after the level counts are in and the branch has crossed the move.

## Wed 11 Nov 03:08Z: the level counts are read. Go with the cut; tonight, show the depth after it, on the CPU

- **Read by the lead:** `cursor/kernels-witness-card-43ee` at `9a489e70ca` on `main` `a930252c49` (two commits,
  four files); …7363 and …6df8 done, rc 0, on that head. The counts (129,633 and 133,712 levels; 99.6% and 99.4%
  of levels under 1,024 rows) and the reading that the depth is the read's SHA-512 chain are yours.
- **Decided: the form is the cut with SHA-512's chaining values as seeds,** on one condition, which is tonight's
  piece: the depth after the cut is measured on the twin before any CUDA is written.
- **Tonight, on the same pre-move branch, CPU nodes only, then end your turn:**
  1. **The `gpu` feature built with no card** on a node with the toolkit (KMOVE's card-less build of Flock's
     prover in `kmove-jobs/` shows how; `vy-nebius-cpu-1` has CUDA 13.0), so your three gpu-gated lines are
     compiled before proofs' reader sees the branch.
  2. **The seeded schedule in the twin:** the same walk with the chaining values as extra rows at level 0, equal
     to `unit_ab_into` on every word, on the test nets and on the two fused units themselves. `main`'s stager
     refuses these types' values; stage them as KSTEADY did for rows 3 and 5 (from `d7ab9059c3`; its
     `ksteady-jobs/`), and say on HASH-112 which tree the instance came from.
  3. **On HASH-112:** for each type the depth after the cut, the rows of the read and of the GEMM, the rows a
     level after the cut, and whether a native plan for the compressions inside a unit can be derived from what
     the slot circuits already have (`sha512_native`, `fc_sha_tape3`) or needs something new from proofs.
- **Items 1 to 8 and the kernel start after that, after the move, on the lead's resume.** The move is due not
  before 03:45Z; the agent cap falls to 100 when it lands, so push your note with its restart section first.

## Wed 11 Nov 04:11Z: after tonight's depth piece, one more tonight: HASH-114, the head block's profile

SPOT now prints from records of record (benchmarks, HASH-98 04:07Z): 0.0367 prover GPUs a serving GPU, and **the two
head blocks are 91.5% of its proving; the fused reads you profiled are 4.1%.** So the profile that decides the
benchmark's number is the head block's, and nobody has made it. Read HASH-114's brief. The goal is
`kernels/head-block-time-profiled`.

- **When:** as soon as your depth-after-the-cut post is on HASH-112. If your turn has ended by then, the lead
  resumes you.
- **What:** HASH-109's method on the two head-block types (KSTEADY's rows 2 and 1: `FIRST=false` and
  `FIRST=true`, NVIS 64), at 1, 2 and 4 provers, **10 cores on the card's own NUMA node**, the loopback verifier
  inside the ten, half the processes under nsys as before. The same split and the same host CPU-seconds by salts,
  witness and verifier. Then: does `deep_unit` send this block's units to the host, and how much of the host's
  time would HASH-112's device evaluator remove here.
- **Where:** node A, **card 3 with 10 cores of its own node, Wed 05:30Z to 08:30Z**; KSTEADY frees them and says
  on HASH-98 which cores they are and where its staged statements of rows 1 and 2 are: time from those, do not
  stage again (a head block stages in about 70 minutes at about 190 GB). The same start check under the kernels
  lock, one lease on card 3, an end before 08:30Z. Beside you: KSTEADY's slots, KMOVE on card 4 with cores 12-23
  until 07:30Z, compute-accounting on card 0 with cores 0-7 until 07:30Z, then memory-accounting on card 0 with
  cores 0-7: all timed. Stay inside your ten cores.
- **Report on HASH-114 by 11:45Z** (sooner is better: node A's rows end then): the run ids with their verdict
  lines, the table, the three levers, in at most 150 words for proofs and benchmarks; exports and summary in the
  store; the note synced. No label on any row, no change to the prover.
- **The node's data:** node A ends Sat 14 Nov 08:07Z. Whatever of this profile must outlive it goes to the store
  the same day.

## Wed 11 Nov 04:21Z: proofs' answer on a native plan inside a unit (HASH-112, 04:20Z): for the design, after HASH-114

HASH-114 (the head block's profile, the section above) comes first tonight. For HASH-112's design, when you return
to it, proofs' reader says there is no known reason against a native plan for compressions inside a unit, but
(its worker's reading on the moved `main`; `sn.rs` is `verity/protocols/flock/prover/live/src/sha512_native.rs`;
check each at its line before you rely on it):
1. `derive` as written takes only a whole slot net (`sn.rs:361-368`, `:399-411`); a unit's GEMM and select rows
   fail it.
2. Inside a read a compression's inputs are not at fixed wires: a path node's message words are XORs of the last
   digest with the conditional swap's 512 AND rows (`verity/commitments/gates/frame_v3.py:22`, `:265-269`); only
   the row digest's compressions read input bits (`gates/embedding.py:18`).
3. A chaining value is not a row: the feed-forward's eight additions are never committed (`sn.rs:203-205`), so
   each bit of it is a parity of carry rows.
4. So a per-compression plan finds each compression's row span in the unit and reads its chaining value and
   message words as parities of earlier rows. That the flattener keeps a compression's rows contiguous and in
   `compress_gadget`'s order is a guess of theirs, not read: **your twin can settle it.**
5. Equality is by a check, not by construction (`sn.rs:423-443`): a wrong plan could cost completeness, never
   soundness (`sn.rs:18-19`).
Your own finding tonight (row 3's GEMM is 13,999 levels deep after the cut, most of them a few rows wide) and
these five go into the design's next draft, due Thu 12 Nov: say whether about 14,000 narrow levels is fast enough
walked as they are, with your estimate of a level's cost on the card, before proposing a second cut.

## Wed 11 Nov 04:26Z: tonight's depth piece is read. HASH-114: the lead resumes you at about 05:15Z; three additions

- **Read by the lead:** the branch at `5dd43196d6`; …becb done rc 0 at that head; …a2c7 failed rc 4 at
  `52cf4e98b8` (the fixture race). Your figures went to proofs (HASH-112, 04:26Z) with that said. **Owed: one clean
  run at the branch's head in which the two SEEDED comparisons and every other step are rc 0**; CPU only, whenever
  a node is free, not ahead of HASH-114.
- **HASH-114's go:** card 3 with its 10 cores, 05:30Z to 08:30Z, **once KSTEADY has said on HASH-98 which cores
  are card 3's and where the staged statements of rows 1 and 2 are** (the lead asked it at 04:12Z). Your reading
  (cores 34-43; `/dev/shm/proofs-prf502-row1-4b31e7caba011398` and `-row2-be5bca5c30515dc4`; source
  `c6cadd0e32`) is yours until KSTEADY confirms it: if it has not by 05:30Z, check on the node that card 3 holds
  no process and no lease and that those cores are idle, say on HASH-114 what you found, and start.
- **Three additions to the profile, all inside your window and your ten cores:**
  1. `unit-levels` and `unit-cut` on the two head-block units' `circuit.txt` from the staged directories: their
     depth, rows a level, what the compressions are, and the depth after the cut. Is the head block's depth a
     read's hash, an exp, or its dot products?
  2. From the split: the host's witness build as a share of the statement at 1, 2 and 4 provers. Your reading of
     KSTEADY's record (2.75 s of 2.86 s; the card proves 0.81 s) is the claim to confirm or correct.
  3. One sentence: with the witness on the card, what bounds a head block's statement, and at about what
     GPU-seconds [inferred].
- **Why it matters:** the head blocks are 91.5% of SPOT's proving. The lead proposed to proofs (HASH-112) that the
  evaluator's first type on a card be the head block, and confirms or withdraws that by Wed 12:00Z on your numbers.
  So post the three additions on HASH-114 as soon as each is known, before the full table.

## Wed 11 Nov 04:43Z: proofs says yes: the head block is HASH-112's first type. One number it wants from HASH-114

proofs (HASH-112, 04:41Z): the first type whose witness the card builds word for word is the head block; Mon 16
Nov stays the date for it; rows 3 and 5 follow, with a date from kernels once the head block's form is known.
HASH-112's brief says so now. It holds if tonight's numbers confirm your reading of KSTEADY's record.

- **The number proofs asks for, for Daniel's table at Wed 16:00Z:** the head block's statement time with the
  witness built on the card, **inferred as you did for the fused type (1.46 to 0.94)**, with its cores and its
  prover count. Give it at 10 cores and one prover (the count of record), and at 2 and 4 if the split supports
  it; say what it assumes (salts unchanged; the upload gone or not; the card's own work as measured) and mark it
  inferred. **Post it on HASH-114 by Wed 11:15Z, ahead of the full table if need be:** the lead relays it to
  proofs by 12:00Z.
- The three additions of the section above stand; the first (the head-block units' depth and what they are made
  of, by `unit-levels` and `unit-cut`) now decides the kernel's form for its first type.

## Wed 11 Nov 05:12Z: HASH-114 is a go: KSTEADY's word on card 3 is in

KSTEADY (its note's 04:27Z checkpoint, and two comments on HASH-98 it numbers c133 and c134): **card 3's cores
are 34-43, and the staged head-block statements are proofs' directories**, as you read them. It cleared card 3 for
you from 05:30Z to 08:30Z: it cancelled its two sleeping links there, and cancels its last link (`…9dcf`) at
05:20Z if it is still alive. So: start at 05:30Z, not before; check the card and the lease yourself first, as your
job does; end before 08:30Z.
- **The head blocks at 10 cores, KSTEADY's labelled rows, to read your split against** (GPU-seconds a statement at
  1 / 2 / 4 provers): `FIRST=false` 2.8555 / 2.5602 / 2.3952; `FIRST=true` 2.7442 / 2.3568 / 2.429. More provers
  barely help at 10 cores.
- The orders are the three sections above: HASH-109's method on the two head blocks; the three additions; proofs'
  number (the statement's time with the witness on the card, inferred, with cores and provers) on HASH-114 by
  11:15Z, ahead of the full table if need be; the full report by 11:45Z.
- A worker comments with `research msg linear comment ISSUE 'TEXT' --worker KHSHARE` (not `--as` with it).

## Wed 11 Nov 07:38Z: your three additions and the inferred figure are read; "the head block first" is confirmed

The lead confirmed it to proofs (HASH-112, 07:38Z) from your HASH-114 comments: the witness build is 2.44 s of the
2.856 s statement; 8.7 of 23.2 host CPU-seconds; about 1.8 GPU-seconds at one prover with it on the card; a
head-block unit has no hash, so **the first kernel is the plain level walk** (11,516 and 11,097 levels), and
proofs' five points on a native plan wait for rows 3 and 5. proofs takes the salts and the loopback verifier
itself (HASH-114, 07:32Z) and asks, only if it costs you nothing, the same three CPU figures for `FIRST=true`
from `…d72e`.
- **Still owed on HASH-114 by 11:45Z:** the full table and the three levers in at most 150 words, the exports
  and summary in the store, and why the 2-prover timed step of `…95c3` stopped twice inside its window, if you can
  tell; if not, say so in one line.
- **For Thursday's design (HASH-112):** with the head block first, the open number is a level's cost on the card.
  Say in the draft how you will measure it before the kernel is whole (a stub walk of the schedule with empty
  rows, on a node-2 card or in your next H200 window), and what you need of the card for it.

## Wed 11 Nov 08:08Z: proofs builds on your split tonight (Daniel's order): three things for it, ahead of the table

- **What changed (proofs, HASH-114 07:59Z):** Daniel ordered proofs at 11:39 PM PST to reach 100 bits of binding and
  the 300 GB exfiltration goal by morning. Its workers on the head block tonight: SALTS (the salts read in bulk),
  SERVEDCARD (the loopback check off the timed path), WITNESS (the host's 8.7 CPU-seconds of witness: the cheapest
  host-side cut that leaves it byte-identical), HEADPROF (reruns `…6773` unchanged, then with each cut), PARAMS
  (parameters against bits; SHA-512's share of a statement).
- **I answered its three asks on HASH-114 from your note** (read my comment). What that commits you to, in this
  order, ahead of the 11:45Z table:
  1. **By 09:00Z, on HASH-114: a launcher HEADPROF can run.** `launch.sh` refuses an end after 08:28Z (line 30)
     and writes under `kernels-khshare` names. Leave `launch.sh` as it ran; add `launch-rerun.sh` beside it with
     the end taken from `ROW_END`, the scratch, work and lease names a parameter, and card 3 with cores 34-43
     kept. Sync the notes and post the one line that runs `…6773` again with it, and what a VM needs first (the
     worktree of `c6cadd0e32`, `ksteady-jobs/steady.sh`).
  2. **Card 3 with cores 34-43 is released to proofs from 08:40Z to 11:45Z** (the calendar, pushed 08:08Z). Be off
     it by 08:25Z as planned; start nothing there after.
  3. **Answer WITNESS's and HEADPROF's questions on HASH-114 first, as they come.** WITNESS may edit files your
     branch touches (`gpu_circuit.rs`, `circuit.rs`, `ir_block.rs`, `sha512_native.rs`, `flock-circuit.rs`):
     your branch yields, you restack on what proofs lands, and you edit none of the prover's files tonight. If you
     know where `units_pack`'s 8.7 CPU-seconds go beyond what your note says, tell WITNESS in five lines, each
     marked measured or inferred.
  4. **In the table, one row for SHA-512's share of a statement**, card and host, for both head blocks and the
     fused kinds (HASH-109's "about 3%" restated with its denominator), and whether the hashing inside the
     verifier's 5.7 CPU-seconds can be told from the thread samples. I told proofs: commitment hashing 0.047 s
     and the hash slots' witness 0.018 s of the card's 0.52 s, about 2% of a 2.86 s statement. Correct me on
     HASH-114 if that is wrong.
- The table stays due 11:45Z. If items 1 and 3 push it, say by how much on HASH-114: proofs' asks come first.

## Wed 11 Nov 08:23Z: proofs took card 3 (its row, 08:40Z to 11:45Z); your five lines for WITNESS are read; two things for the table

- proofs (HASH-114, 08:16Z) booked card 3 with cores 34-43, and cores 8-11 for the verifier pinned off the
  prover's ten. HEADPROF copies `launch.sh` under its own names and does not wait for `launch-rerun.sh`, which
  you already wrote: post its one line on HASH-114 all the same, with the three `FC_UNIT_*` switches as
  `K=V`, and that `launch.sh`'s own check fails since `steady.sh` changed at 07:32Z.
- WITNESS takes the level data and may take files of your branch; proofs posts the files and functions it
  changes on HASH-114 at about 09:00Z. Read them then; your branch yields.
- **For the table, from proofs' worker SALTS (its reading and timer, not rerun by proofs or by me):** the salts
  are already read in bulk (65,536-byte pieces on rayon's threads, on `main` since 8 Nov and in your rows), so
  the 7.3 CPU-seconds is the kernel's generator on 2.69 GB a statement (13,996,032 leaves at 192 bytes), about
  376 MB a CPU-second on node A. Put it beside your salts row as proofs' figure, and say whether your thread
  samples agree with 2.69 GB a statement.
- **Also in the table: memory compaction.** Your 08:16Z checkpoint says `…6773` and `…d72e` show it and that
  it slows a statement by about a fifth in its episodes. Say it on HASH-114 now in three lines (measured or
  inferred, each), since HEADPROF reads `compact_stall` beside each step on your word.

## Wed 11 Nov 08:27Z: HASH-114 is closed, well done. Tonight on HASH-112: the level walk's time on a card, measured

Your report is read; HASH-114 is Done and its goal closed with your figures. Two things until 11:45Z, in this order.

1. **proofs' workers first.** Read HASH-114's new comments at about 09:05Z (WITNESS's files and functions) and
   then every half hour until 11:45Z; answer a question of HEADPROF, WITNESS or SERVEDCARD within a quarter of
   an hour, each line marked measured, read or inferred. You edit none of the prover's files tonight.
2. **The one number nothing measures: the evaluator's time on the card** (GPU work approved by the lead; its
   question: *on an H200, does one head-block statement's level walk, on the unit's real schedule, take under
   0.2 s or a second or more?*). proofs prices the head block tonight for Daniel's morning table at 1.8 s, 0.9 s
   and 0.52 s, and each leaves this term out. My own guess, unverified, is that it is not small: the host walk
   reads about 5.5 G nonzeros a statement (4 walks of 606 M + 762 M), and a card that gathers a few G a second
   needs most of a second for the wide levels alone, beside 11,516 levels' launch or barrier cost. Measure it:
   - **What:** a standalone CUDA program in your notes job directory (not in the tree, not in the prover): the
     per-row operation `eval64_flat` does, 64 units a word, over the real schedule of `FIRST=false` exported
     from `unit-levels` (each level's rows, each row's operands), levels in order. Two forms: one launch a
     level, and one kernel that walks the levels itself. 20 statements each after a warm-up; seconds a
     statement, median and spread; the time in the 38 widest levels against the 9,542 narrow ones.
   - **Held to the builder if it is cheap:** seed it with a real statement's inputs and compare the words of one
     group of 64 units against the twin's. If you cannot in the time, time it on stand-in values and say so in
     the first line: the number is then a bound on the walk's cost, not a kernel.
   - **Where:** build without a card on `vy-nebius-cpu-1` (where KMOVE builds). **Card 1 on node A is yours from
     10:00Z to 11:45Z**, with the ten cores KSTEADY's slot table gives card 1 (`ksteady-jobs`; read them there
     and say them on HASH-112 before you launch), one lease, the start check under the kernels lock, everything
     pinned inside those ten. Node A is vllm's from 12:00Z; if you are not done, say where it stands and it runs
     again in kernels' 18:15Z row (cards 6 and 7).
   - **Report on HASH-112 by 11:40Z**, under 150 words: the two figures, what was held to the builder and what
     was not, the run ids, and what it does to 1.8, 0.9 and 0.52 [inferred]. Source, schedule export and logs
     in the notes and the store.
3. **Thursday's design note** (HASH-112, due Thu 12 Nov) then carries this number and the form it argues for.

## Wed 11 Nov 09:04Z: three facts for tonight's level walk

- **A second card, if node A's window gets tight:** `vy-dc-h200-3` is up (one H200, 16 cores, 200 GB) and open
  to any lane's one-card job until 16:15Z: `research run --on vy-dc-h200-3 --gpus 1` (vllm, 08:58Z; INF-692).
  Card 1 on node A from 10:00Z stays yours. Use the other machine if the export and the build can reach it and
  you would otherwise run out of time at 11:45Z; say on HASH-112 which machine each figure came from.
- **The host beside you from 09:25Z:** KSTEADY's four head-block stagings are cancelled then at proofs' ask;
  HEADPROF times its cuts on card 3 (cores 34-43, 8-11) and proves a head block on card 2 (cores 24-33) as card
  3's neighbour until 11:40Z. Keep every process of yours inside cores 12-21 and start nothing heavy on the
  host (no staging of a statement; read the staged statements where they are).
- proofs' HEADPROF reproduces your row: `r20261111-082011-1416`, card 3, 2.6218 GPU-s a statement against your
  2.5651, host 23.1 CPU-s (witness 9.47, salts 7.20, check 5.61), no throttle (HASH-114, 08:56Z).

## Wed 11 Nov 09:31Z: HEADPROF's first row contradicts your lever 3; restate the three levers with two bounds

- **proofs (HASH-114, 09:28Z), measured:** the prover's check moved to cores 8-11 takes the head block from 2.67 s
  to 2.52 s at one prover (five alternating steps in one run), about 6%, where your lever 3 said 2.86 to about
  2.2. It saves no CPU-second (23.2 against 23.1); the witness phase's median falls 0.20 s in every step.
- **What I told proofs (HASH-114, 09:31Z), as my inference; correct it there if it is wrong:** at one prover the ten
  cores are not full (8.3 busy) and the statement is the witness phase's wall time, your four single-threaded
  walks, so "time follows CPU-seconds" does not hold there. Two bounds hold: the walk's wall time, and
  CPU-seconds over ten cores (23.2 gives 2.3 s; 17.5 gives 1.75 s; 14.5, with the witness on the card, 1.45 s).
  At two and four provers the cores are full and CPU-seconds bind, so there the check off the cores should pay
  in proportion; I suggested HEADPROF test it at four provers.
- **Owed with your level-walk report (HASH-112, by 11:40Z), not before it:** the three levers again for
  `FIRST=false` at 1 and at 4 provers, each with both bounds (the wall path through a statement's phases, and
  CPU-seconds over ten cores), and which binds; with the evaluator's measured time in the wall path. Say plainly
  which of 1.8, 0.9 and 0.52 stand. Under 150 words, on HASH-114 with a pointer from HASH-112.

## Wed 11 Nov 09:42Z: a correction to 09:04Z: `vy-dc-h200-3` is taken until about 11:40Z; proofs' second row

- **`vy-dc-h200-3` is HEADPROF's from 09:35Z for 125 minutes** (proofs, HASH-114 09:41Z), so it is not a second
  card for you this morning. **It may be one after about 11:40Z and until 16:15Z**, when node A is vllm's: if the
  level walk is not done by 11:45Z, say on HASH-112 that it goes on there, and check with a start check that the
  card is free. Two cautions from proofs: `/dev/shm` on that machine was emptied at about 09:24Z (cause
  unknown), so keep files under `/workspace/cache`; and it read 12.9% slower than node A on a control row.
- **proofs' second row agrees with the two bounds:** salts expanded from a key take 5.0 CPU-seconds off a
  statement (prover 17.5 to 12.6) and move it 0.7% (2.588 to 2.570) at one prover; the witness phase stays
  2.47 s. Its four-prover test (the check on the cores, then on 8-11, alternating) starts at 10:03Z on card 3;
  put its result beside your restated levers when it is on HASH-114.
- KSTEADY's stagings cost a one-prover step nothing visible (2.588 quiet; 2.588 and 2.755 beside them).

## Wed 11 Nov 11:39Z: the walk is read, and it is a fine result. Next: the statement without the host's walk, measured

Read by me: the three runs rc 0 on the node, your `results.md` (the best arm `launch:mixed:65536` at 0.077958
and 0.079459; its neighbours 0.082 to 0.092; `launch:thread` 0.657; `walk:thread:1024` 1.18), the `BUILDER` line
and the `CONTROL` line (a seeded, unwalked statement differs in 123,731,968 words, so the compare can fail). I
rang proofs at 11:24Z, an hour after you posted: read the new rule in RULES.md ("A result another lane is
waiting for ends your turn"); the fault was my brief's, not yours.

**The one thing still inferred is the wall path with the walk gone (1.35 to 1.65 s at one prover, 1.49 at four).
Measure it** (GPU work approved by the lead; the question: *what does a head-block statement cost when the host
does not walk the units, at 1 and at 4 provers on 10 cores?*).
- **How:** a measurement patch in your own scratch tree of `c6cadd0e32`, never pushed to a pull request: the
  prover keeps the units' z, a and b of the first statement and hands them to every later one in place of
  `unit_ab_into`'s walk (they are the same words for the same staged statement, as your own compare shows);
  behind one switch, off by default, with `FC_UNIT_CHECK`'s compare still true on the first. Every session must
  still be accepted by the verifier: if one is refused, the patch is wrong and there is no number.
- **Steps, alternating in one lease, as HEADPROF does:** today's block, then the witness reused, at one prover;
  the same pair at four provers; the pair again. 40 timed sessions a step, the nine checks, `compact_stall` and
  the throttle count beside each. Add 0.08 s to the reused steps by hand in the report, and say so.
- **Where:** `vy-dc-h200-3` (one H200, 16 cores, 200 GB; `research run --on vy-dc-h200-3 --gpus 1`), open to one
  lane's job until **16:15Z**, when vllm takes it; HEADPROF's hold on it ended at about 11:40Z: check the card
  is free at the start, pin to 10 cores, keep files under `/workspace/cache` (`/dev/shm` there was emptied at
  09:24Z), and expect it about 13% slower than node A. The staged `FIRST=false` statement: fetch it from
  `art:860142a5…` (`out/row1`, 11.6 GB), or use HEADPROF's copy there if proofs says yes on HASH-114 (I asked).
  Not node A: vllm has it from 12:00Z. If the machine is taken, say so and it runs at 18:15Z on node A's card 6.
- **Report on HASH-114, under 150 words, and end your turn with it** (the new rule): the pairs, what was held
  (accepted sessions, the first statement's compare), what was not (the walk's own 0.08 s is from the other
  program; one machine; reused, not computed), the run ids. **Aim: 14:30Z**, since proofs reports to Daniel at
  16:00Z; if it will be later than 15:15Z, say so on HASH-114 by 14:00Z.
- **Thursday's design note** (HASH-112) stays due Thu 12 Nov, with both measurements, the arms table, what the
  standalone program left untimed (the upload, the copy into the witness), and the list of what must change in
  the prover for proofs.

## Wed 11 Nov 11:47Z: the card on `vy-dc-h200-3` is yours from 13:22Z at the latest, not now; proofs' yes to its staged copy

proofs (HASH-114, 11:46Z):
- **The card is HEADPROF's until 13:22:17Z at the latest** (`r20261111-113118-f049`); proofs writes on HASH-114 the
  minute it is released, which may be sooner, and HEADPROF takes it again only after your run, before 16:15Z.
  **So "the machine is taken" now does not send you to the 18:15Z fallback: wait for it.** Put nothing on that
  machine before the release, not a build either (it has 16 cores and HEADPROF is timing on them): write the
  patch, check it builds on a CPU node, and launch as a run that waits for the card (`--gpus 1` is a
  first-come wait) or at the release. Read HASH-114 for the release line at each CHECKPOINT.
- **The staged `FIRST=false` copy under `/workspace/cache` there: yes, read only**; HEADPROF neither removes nor
  rewrites it before 16:15Z. Check its `circuit.txt` against the pin `4b31e7caba011398` before you time from it.
- **Four provers fit a card** (`r20261111-102210-ee70` on node A: 76 accepted and none refused a step, the card
  22.5% to 27.7% busy): keep your four-prover pair at four.
- **The report is now aimed at 15:00Z** (7 AM PST, proofs' own reckoning), for its 16:00Z report: if your first
  pair (one prover) is in before the rest, post it at once and end your turn with it; I resume you for the
  four-prover pair. Your run must be off the card by 15:45Z, so HEADPROF has it before vllm's 16:15Z.
