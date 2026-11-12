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

## Wed 11 Nov 12:09Z: `vy-dc-h200-3`: the card at about 12:40Z; one and two provers there; the four-prover pair goes to node A at 18:15Z

proofs (HASH-114, 12:03Z); read its comment for the exact words.
- **The card comes sooner, at about 12:40Z** (an estimate; the release line follows on HASH-114).
- **The staged copy:** `/workspace/cache/proofs-headprof/staged/proofs-prf502-row1-4b31e7caba011398` (five files,
  12 GB, `pin.txt` beginning `4b31e7caba011398`). Copy it to your own scratch under `/workspace/cache` and
  write nothing into it.
- **Four provers do not fit that machine's HOST memory** (196 GiB; about 70 GiB a prover with its verifier; three
  already failed for HEADPROF). **So there: the pair at one prover, then the pair at two, and the two pairs
  again.** Ten cores (say which), the check inside the ten as in kernels' rows; proofs notes that with the check
  on four cores of its own a two-prover row there is bound by the check.
- **The four-prover pair runs on node A's card 6 with cores 72-81, 18:15Z to 20:15Z** (kernels' row; the calendar
  says so from now). Prepare it (`launch-rerun.sh`'s kind of launcher, the statement from proofs' `/dev/shm`
  copy on node A if it is still there, else from the store), launch nothing there before my go; I resume you
  at 18:10Z. It may share that slot with a short check of KP2's; I say the order then.
- **Unchanged:** post the one-prover pair on HASH-114 as soon as it is in and end your turn with it; off
  `vy-dc-h200-3` by 15:45Z.
- **One more question from proofs, for after the run, by Thu 12 Nov 18:00Z, by reading** (HASH-114, 12:07Z, its
  worker PARAMS; a what-if that is Daniel's to rule on): a wide level-0 leaf at a head block's table (m = 35),
  1,028 lanes (16,448 bytes) and 524,288 leaves. Does the card's level-0 leaf kernel take 1,028 lanes as it is,
  and what would `t.encoding_commitment` and `t.ligerito` read there, if it can be said without a run? Answer on
  HASH-114 with file and line, each sentence marked read or inferred, and end your turn with it.

## Wed 11 Nov 12:40Z: `vy-dc-h200-3`'s card is released since 12:35:03Z: it is yours now

proofs (HASH-114, 12:38Z): HEADPROF's run ended by itself at 12:35:03Z; the card holds 0 MiB and nothing of
proofs' is alive there; it starts nothing until your one- and two-prover runs are done. Its staged copy stays
where it is for you. **Launch when your patch has built; check the card yourself at the start as always.**
**When your last run there has ended, say so on HASH-114 in one line ("the card on `vy-dc-h200-3` is free")
in the same comment as your figures, or before them if they take longer to write**: proofs waits on that line
to take the card back before vllm's 16:15Z.

## Wed 11 Nov 13:14Z: your first pair is read and with proofs; for when I resume you (about 13:48Z)

Read: the pair (3.849 s against 2.583 s, 0.671; 43 of 43 accepted each), and that you ended your turn with it.
I told proofs: the card is free at about 13:46Z (I read the run at 13:48Z and post the "free" line myself);
yes to HEADPROF's copy onto the disk meanwhile (CPUs 12-15); and your `nvidia-smi` at 12:34:56Z, owned.
**When I resume you, the report on HASH-114 (under 150 words) also says:**
- **which phase holds the reused statement**: its witness falls from 3.77 s to 0.28 s but the median
  `prove_total_s` rises from 0.69 s to 1.81 s. Say what the prover waits on there (the pads of `zk_prepare`,
  `ZKPRE.rep0_s`, is my guess), from the run's own lines, measured or inferred;
- the two-prover pair and both repeats, with their spread, accepted and refused, and the nine checks if
  KSTEADY's `after.py` can be run on these steps (else which of the nine you can show);
- that your scratch is removed and nothing of yours is left on the machine.
**For 18:15Z on node A's card 6:** the four-prover pair as prepared. I offered proofs one more row there, every
cut of proofs' together with `FC_UNIT_REUSE=1`, if it names its every-cut build's commit by 17:30Z: say in your
report whether your patch applies on proofs' HEADPROF source (`a5922158dc64` on `c6cadd0e32`, by `git apply
--check` only) so that the answer is known before then.

## Wed 11 Nov 13:24Z: proofs took the offer (HASH-114, 13:22Z); the order on node A's card 6 at 18:15Z

- **Proofs names its every-cut build's commit on HASH-114 by 17:30Z**, for you to time at four provers with
  `FC_UNIT_REUSE=1`. HEADPROF also applies your switch on that build itself, as a row of its own on
  `vy-dc-h200-3`; yours is the four-prover row on node A, the one that turns the inferred 0.77 s into a
  measurement.
- **The slot's order: KP2 first, 18:15Z to 18:32Z** (#8662's H200 run), **then you, 18:32Z to 20:10Z**, cores
  72-81. Yours in this order: (1) the four-prover pair as prepared, with its repeat; (2) in what is left, the
  row at proofs' commit, reuse on against reuse off, alternating, at four provers. If the commit is not named
  by 17:30Z, or your patch does not apply on it, run only (1) and say so.
- Build what you can off node A before the slot, so its minutes go to timed steps. Launch nothing on node A
  before my go at about 18:10Z.

## Wed 11 Nov 13:42Z: proofs named its every-cut commit (HASH-114, 13:36Z), for the second row at 18:15Z

- **The commit: `aa23d1312dce7121b6385aa403b120eec406db5a` on `cursor/proofs-headprof-2b18`**: your
  `unit-reuse-on-c6cadd0e32.patch` applied unchanged on proofs' every-cut tree `7b12b8825b39` (read by me: one
  file, `gpu_circuit.rs`, +54 -2; the commit is on that branch). Proofs has not built it; measurement only, no
  pull request.
- **The row's switches: `FC_SALTS_KEYED=all FC_UNIT_REUSE=1`.** Its other half on node A is the same build with
  `FC_SALTS_KEYED=all` and `FC_UNIT_REUSE` unset, alternating, at four provers, after your own pair.
- **Say what the row is in your report, in proofs' words:** the timed sessions prove one statement again and
  again, so every session after the first skips the walk; it is the price with no walk at all, a floor, and
  not a served figure.
- HEADPROF runs the same build on `vy-dc-h200-3` once that card is free (about 40 minutes); yours is the
  four-prover row on node A's 10 cores.
- Your `nvidia-smi` at 12:34:56Z: proofs read that step and it is not odd (3.1174 s a statement, between
  3.1277 and 3.0234). Closed.

## Wed 11 Nov 13:49Z: your run ended 13:43:59Z, rc 0; the card is proofs' now. Report from the record; nothing on that machine

- I posted the "free" line on HASH-114 at 13:49Z; HEADPROF takes the card for about an hour of timed rows.
  **From now until proofs writes that HEADPROF is off `vy-dc-h200-3`, run nothing there: no ssh, no read, no
  `rm`.** One `nvidia-smi` of yours fell inside its step this morning; none of anything this time. Read your
  steps from the run's record (`research fetch r20261111-123832-934d` and its `out/`). If a figure is only in a
  file on the machine, say which and leave it. Your scratch stays where it is: say its path and size from your
  launcher, not from a look, and I tell you when to remove it.
- **The report on HASH-114, under 150 words, as the 13:14Z section says:** which phase holds the reused
  statement (measured or inferred); the two-prover pair and both repeats with their spread; accepted and
  refused; the nine checks. For the nine: KSTEADY's module is open as #8979 (`verity_bench.steady` at
  `a4a9eba2d67c`, a function of a run's own files, no GPU); run it on your steps if their files fit, else say
  which checks you can show. **End your turn with that comment.**
- I resume you after it for the 18:15Z preparation (proofs' commit `aa23d1312dce`, the section above), the
  PARAMS answer and Thursday's note.

## Wed 11 Nov 14:01Z: your report is read and rung to proofs; until 18:10Z, four things, in this order

Read by me: your comment (13:56Z) and your return. Reading the record from the store and not through `fetch`
was right. Your comment named no handle, so it rang nobody: I rang proofs (14:00Z) with the reading: the
measured ratio stands in place of my inferred 1.35 to 1.65 s; the salts read ahead are the next bound; the
repeat's two-prover rows are not steady, so the two-prover ratio rests on the first pair. **Start a comment
for another lane with its handle (`@proofs ...`) so it rings.** The goal on the Goals page has the ratio.

Until I resume you with the go (about 18:10Z), in this order:
1. **The salts, read not inferred:** where rep 0 takes its salts in the prover's code, and whether a prover
   with the walk gone waits there (`SALTSAHEAD`). One paragraph in your note, file and line; it goes into
   Thursday's design as the lever after the walk.
2. **Why `level` fails on the repeat's two-prover rows** (thirds 1.65 and 1.34 against 1.25): from the record
   only. Say what the thirds show (a slow first third, a slow last, one long proof) and whether the first
   pair's rows were near the line. If it cannot be told from the record, say so; do not guess.
3. **Tonight on node A's card 6 (18:32Z to 20:10Z, cores 72-81), prepared off node A:** your own four-prover
   pair with its repeat, then proofs' commit `aa23d1312dce` (`FC_SALTS_KEYED=all`, reuse on against off).
   **Every row must carry what the nine checks read, `flat` included** (the rate fields are `server_rate.py`'s
   at #8679's head `26b5ddf7cb6e`): say in your CHECKPOINT how the rows get them, or which they will lack and
   why. Build both trees before the slot where a build needs no card, and say where; what must build on node
   A is the first minutes of your slot, counted in your plan. The plan fits 98 minutes with its checks, or
   you say what is cut.
4. **Then the PARAMS answer** (the 1,028-lane level-0 leaf question, by Thu 18:00Z, by reading code) and
   **Thursday's note**. The PARAMS answer ends your turn when it is posted, with `@proofs` leading it.
- Also: merge `origin/main` (`72e4d46d9209` or later) into your own agent branch before Thu 07:59Z and say
  the commit in your CHECKPOINT.
- **End your turn by 17:55Z at the latest with a CHECKPOINT**, whatever is unfinished, so I can resume you
  with the go. Nothing on `vy-dc-h200-3` until I write that proofs is off it; nothing on node A before my go.

## Wed 11 Nov 14:42Z: your four items are read and rung to proofs; tonight's plan stands with three changes; now Thursday's note

Read by me: your 14:15Z and 14:38Z comments, your return, and both builds rc 0 (`…69e6`, `…a1e2`). Finding
the burst in your own figures and correcting them unasked is the work I want. I rang proofs at 14:41Z with
all of it (your two comments had rung nobody), corrected my own 14:00Z and the goal's text, and asked proofs
by 17:30Z whether its staged head-block statement is still on node A, where, and whether you may read it.

**Tonight's plan stands** (yours 18:32Z to about 19:18Z, proofs' row to about 19:55Z), with three changes:
1. **Windows of whole bursts.** Choose each step's timed session count so the window holds whole bursts (a
   multiple of 16 at four provers, if provers file in fours there: look at the first step and say), and
   report every row both ways, labelled: the window's rate and the rate over whole bursts; `level` by thirds
   as the module has it and by whole bursts. If 48 sessions a step also buys proofs' row its fourth step,
   take it; say which you chose and why.
2. **From 18:15Z, while KP2's run has the card** (a pass-or-fail check, not timed): you may stage on node A,
   on cores 80-81 only, no GPU: the builds from the store, and the statement (proofs' copy if proofs says
   where, else the 12 GB from the store), under one directory of yours that you remove at the end. So your
   own minutes from 18:32Z are timed steps.
3. **The gate step is cut: say so in the report**, with what stands in for it (every step's accepted and
   refused; the gate ran for your tree on `vy-dc-h200-3` only, and not by kernels for proofs' tree), and what
   the "own" step is, in one clause.

**Until 17:55Z: Thursday's design note** (HASH-112), which now has what it lacked: the measured ratio with
its two readings, the lever after the walk (the salts read ahead, with your file and line), the kernel's form
for the head block, and what tonight's rows will add. Draft it in your note; no ring. **End your turn by
17:55Z with a CHECKPOINT** so I can resume you with the go; earlier if the draft is done.
- Still: nothing on `vy-dc-h200-3` until I write that proofs is off it; nothing on node A before my go.

## Wed 11 Nov 15:03Z: a correction of mine (no action for you now)

At 14:03Z I told you to lead a comment with the other lane's handle "so it rings". That was wrong, and you
were right: a worker's comment rings no one whatever it leads with (RULES.md, "A result another lane is
waiting for ends your turn"). Keep doing what you did at 14:39Z: say in your return which comments need my
ring, and I ring them. Leading with the handle is still right, so the reader knows whom it is for.

## Wed 11 Nov 15:15Z: the design draft is read whole; piece 1 runs TONIGHT after your two rows (card 6 is yours to 20:50Z)

Read by me, every section. It is the design I wanted: each figure marked, the lever after the walk found and
cited, and what is open said as open. The session count's reasoning (56 gives 48 gaps, three whole bursts of
16, one in each third) is right, and refusing a count whose window is not whole bursts is the right guard.
Two things I did with it: I told proofs before its 16:00Z table of the two small host-code changes (level 0
read ahead whenever a next session will come; a ring of page-locked buffers), as read and not run; and I
asked proofs whether node A's four provers share the card through MPS. The whole note goes to proofs on
Thursday with tonight's rows in.

**Piece 1 tonight.** Card 6 and cores 72-81 were unbooked after 20:15Z, so I extended kernels' row to 20:50Z
(the calendar has it, pushed 15:15Z). After proofs' row, in the same lease:
1. **Four copies at once** of the standalone walk's three best arms (`launch:mixed:65536`, `hybrid:mixed:64:64`,
   `walk:mixed:32:32`), four processes on the card, each held to the host's words as before, with the 0xA5
   poison and the no-walk control: seconds a statement for each copy and arm, and the card's peak memory.
2. **One copy beside provers**, only in a form that cannot take the provers' memory: by your own figures four
   provers leave about 11 thousand MiB and the schedule with one walk needs about that, so beside four it may
   not fit. Choose beside two or three provers (say which and why), or cut it; an out-of-memory there must
   void only that extra step, never a row. Nothing of it is quoted as a prover's rate.
3. **Order and cuts:** your pair and its repeat first, proofs' row second, piece 1 last. If the rows run late,
   piece 1's part 2 is cut first, then proofs' fourth step, then part 1. Off the card by 20:45Z, the lease
   released, your staging removed; say the minute in your report.

**Before the go, from records only:** node A's glibc, if any earlier run's record of yours holds it (an
`ldd --version`, a loader error, a binary built there): if it is below 2.39, plan the two builds inside the
slot now and say what that costs proofs' row. And prepare piece 1's launcher, dry-run, beside the others.
**End your turn with a CHECKPOINT when both are done**; I resume you with the go at about 18:10Z. Nothing on
node A and no GPU before it; nothing on `vy-dc-h200-3` until I write that proofs is off it.

## Wed 11 Nov 15:56Z: proofs answered (HASH-114, 15:54Z): the staged statement, no MPS, HEADPROF off `vy-dc-h200-3`, and a question for Thursday's design

Read by me in proofs' comment; its looks are HEADPROF's.
1. **Remove your scratch on `vy-dc-h200-3` NOW, before 16:10Z.** HEADPROF is off that machine since 15:42Z
   and touched nothing of yours; vllm has it from 16:15Z. One `rm -rf /workspace/cache/kernels-khshare` and a
   look that it is gone, nothing else there, and nothing under `/workspace/cache/proofs-headprof` (proofs'
   20 GB). Say the minute in your CHECKPOINT. **If it is past 16:10Z when you read this, do not go there:**
   leave it and say so, and I ask vllm for a minute.
2. **The staged head-block statement on node A is there and you may read it, read-only:**
   `/dev/shm/proofs-prf502-row1-4b31e7caba011398` (`circuit.txt` 12,191,652,715 bytes, `inst.bin`, `pub.bin`,
   `stage.json`, `pin.txt` beginning `4b31e7caba011398`; owner root, mode 644; a look at 15:32:51Z, not
   hashed then). It is proofs' SERVEDCARD's staging: **never write to it or remove it.** Copy it as `fused.sh`
   copies `STAGED`, which refuses a copy whose SHA-512 is not its `pin.txt`. So no 12 GB fetch tonight
   unless that copy fails.
3. **No MPS:** the four provers on node A are plain processes, each its own `flock-circuit prove` on the
   lease's card; at 15:33:33Z no `nvidia-cuda-mps` process ran, `/tmp/nvidia-mps` did not exist, and every
   card's compute mode was `Default`. So tonight's four-at-once walk is four plain processes too, as you
   planned; say the card's compute mode in the run's record.
4. **For the note:** proofs corrected its own no-walk row the same way (whole bursts): 1.862 to 1.865 s at one
   prover, 0.54 of today's in that run, and 1.31 to 1.37 s at two (0.44 to 0.49), with keyed salts on every
   tree and #8891's cuts, six cores, a floor. Its run shows your salts pattern too (one session in four,
   rep 0 at 2.0 to 2.4 s against 0.4 to 0.6 s); cause not established by proofs.
5. **A question proofs put to Thursday's design (no hour):** the head block cut into a scores unit and values
   units (#8917 and #8765, on `main` now) costs, at two provers on six cores, 1.35 to 1.39, 1.09 to 1.12 and
   1.02 to 1.06 s at G = 128, 64 and 32 with the values unit walked on the card as today, and 1.05, 1.03 and
   1.00 s with `FC_HOST_UNITS=1`, the follow-up build and salts `all`. **Should a values unit go to the host
   by default (`deep_unit`, `HOST_UNIT_LEVELS` in `gpu_circuit.rs`)?** Proofs changes nothing there before
   the design says. So the design gains a section: the cut's two unit kinds read on `main` (their depth and
   width by the twin, no GPU), which of them the level walk serves and at what launch count, and your answer
   to that question with what it rests on. If the cut replaces the head block as the type of record, say what
   of the design carries over and what does not. After tonight's rows; it is Thursday's work.

## Wed 11 Nov 16:08Z: a question for your next return (no action before tonight's rows): can the head block's kernel be timed on node A by Fri 13 Nov?

Node A's lease lapses Sat 14 Nov 08:07Z unless proofs has it renewed to Mon 16 Nov, and infra decides the
week's GPU plan by Fri 02:00Z. The head block at four provers needs about 280 GiB of host memory and 10 full
cores, which `vy-dc-h200-3` (16 threads, 200 GiB) has not: after Saturday only one and two provers fit. I
told infra and top (INF-682, 16:08Z) that I choose by **Thu 12 Nov 18:00Z** between two ways:
1. **pulled forward:** the walk in the prover behind the new mode, as a measurement branch (not merged, said
   as such), held to `pack_units` word for word, and timed at four provers on node A on **Fri 13 Nov**, in
   hours I book; proofs' review and the merge follow next week;
2. **as planned:** the kernel Mon 16 Nov, its rows at one and two provers on `vy-dc-h200-3`, the four-prover
   figure waiting for the next large host.
**In your return after tonight's rows, say which of the two you would take and why:** what of your ten items
way 1 needs at the least (the mode, the buffer, the build ahead; what can stay a stand-in), the hours of
card for its gates and its rows, what tonight's four-at-once result changes, and what could make Friday
slip. Do not start it before I say; tonight's rows and Thursday's design come first.

## Wed 11 Nov 16:10Z: tonight's plan is read and stands, with one change: build both trees during KP2's run. Until 17:55Z: the design's section on the cut

Read by me: your CHECKPOINT, the plan's table and cuts, `r20261111-153122-8ca7` rc 0. The scratch on
`vy-dc-h200-3` is gone (15:58:11Z, 550 MB): I told proofs. The cuts enforced by the job, three provers for
(b) with the copy loaded on an empty card first, and saying what (a) does not measure: all as I would want.
Say in the report that (b)'s provers are keyed and reused, the busiest case, and not today's block.

**One change: yes, build at `stage`, both trees, from 18:15Z, on cores 78-81** (two a tree, niced, no GPU),
while KP2's run has the card on cores 72-77. KP2's is a pass-or-fail check, not timed; I move its pin to
72-77. A tree not built when KP2 ends at about 18:32Z finishes on all ten cores before your first step. So
the builds cost your rows at most a few minutes, and piece 1 keeps its margin. If `stage` shows the
prebuilt binaries load after all, it builds nothing. Put it in the launcher, dry-run.

**Until 17:55Z, Thursday's design, the section proofs asked for** (my 15:56Z section, item 5): the head
block cut into a scores unit and values units (#8917 and #8765 on `main`). Read the Definitions, and by the
host-side twin on a CPU run (no GPU) give each unit kind's rows, levels and width at G = 128, 64 and 32;
which of them the level walk serves and at what launch count; and your answer to proofs' question (should
a values unit go to the host by default, `deep_unit` and `HOST_UNIT_LEVELS`), with what it rests on. If the
cut replaces the head block as the type of record, say what of the design carries over. Then, as far as
you get, my 16:10Z question on Friday (way 1 or way 2), which may wait for tonight's rows.
**End your turn by 17:55Z with a CHECKPOINT.** Nothing on node A and no GPU before my go; `vy-dc-h200-3` is
vllm's now: nothing there at all.

## Wed 11 Nov 17:18Z: your 17:15Z CHECKPOINT is read; the cut's answer and the build-key finding are with proofs. Wait for the go

Read by me: your return, the cut's table, the answer to proofs' question, Friday's view, the stage build's
two runs. **You were right and I was wrong on the cut's figures:** 1.35 to 1.39 and the rest are ratios of
the pair over the head block, not seconds; I misread proofs' comment and wrote "s" in your 15:56Z section
and in my note (corrected there). Reading the seconds in HEADPROF's own records before answering is what
kept the answer sound.
- **Rung to proofs by me (HASH-114, 17:18Z):** the units by the twin, your answer (host at G = 128 as the
  code is; at 64 and 32 only once the salts are cheap; the rule on the unit's rows), the two numbers as
  choices and not fits, the memory correction, and **your finding in the build's key** (`sha512.cuh` from
  the kernels' directory is compiled in and not keyed; `70-class-sweep.sh:53-54`, `cuda_sha512_patch.py:22`,
  both read by me on `main`): the file is proofs', so the fix is proofs'.
- **Friday:** your view is taken (way 1 as a try with a stop, if tonight's piece 1 allows). I decide by Thu
  18:00Z, after tonight's report.
- **Tonight:** nothing changes. I resume you with the go at about 18:10Z; `stage` and `build` start at
  18:15Z on cores 78-81, KP2 has the card and cores 72-77 until about 18:32Z.

## Wed 11 Nov 18:01Z: GO. One change: KP2 does not run first. You have card 6 and all ten cores (72-81) from 18:15Z

KP2's pull request no longer merges onto `main` (a deletion of vllm's landed at about 18:00Z), so its H200
run moves to the end of the slot, after you. **So from 18:15Z the card and cores 72-81 are yours alone:**
- **Start `stage` and `build` at 18:15Z on all ten cores** (no need to keep to 78-81), and your first timed
  step as soon as both trees are built and the card's start check passes; do not wait for 18:32Z.
- **The plan is otherwise yours as written:** your row (own, B, A, B), proofs' keyed row (own, B, A, B),
  piece 1 (a) four at once, (b) one copy beside three keyed, reused provers; the cuts in the order the job
  enforces; proofs' statement read from `/dev/shm/proofs-prf502-row1-4b31e7caba011398`, never written or
  removed; windows of whole bursts, both readings a row; the card's compute mode in the record.
- **Release the lease as early as the plan allows, and no later than 20:42Z. The minute it is released, end
  your turn with that minute first in your return** ("card 6 free at HH:MMZ"), then the figures: KP2 takes
  the card after you, to 21:15Z at the latest (the calendar has it). Remove your staging directory after
  KP2's run, not before, if removing it would load the host while KP2 runs: say what is left and where.
- **The report** (on HASH-114, for my ring, with `@proofs` leading): your row's ratio at four provers both
  ways, proofs' row in proofs' words (no walk at all, a floor, not a served figure), piece 1's seconds a
  statement for each copy and arm with the card's peak memory, the gate step cut and what stands in for it,
  what the "own" step is, and what is not verified. Then your view on Friday (way 1 or way 2).
Nothing on `vy-dc-h200-3`. If the start check finds another lane's process on card 6, launch nothing and end
your turn saying what it found.

## Wed 11 Nov 20:00Z: tonight's rows are read and rung; card 6 went to KP2 at 19:58Z. Next: clean up now, the note final, and the kernel in the prover at full speed

Read by me: your return and your two comments. Eight steps, every session accepted, nothing void, piece 1
whole, the lease back 50 minutes early, and your statement of what the rows cannot say (compaction in all
eight windows; one step in each row failing `flat` by its letter): a clean night's work. **Rung to proofs
(HASH-114, 19:58Z)** with the rows, the walk's 0.137 s beside provers, the memory stop, and my two inferred
figures replaced (1.5 s became a measured floor of about 1.47 s; 0.77 s became 1.03 s). The goal on the
Goals page has them.
1. **Clean your 8.7 GB on node A now**, while KP2's run has the card: it is a pass-or-fail check, not timed,
   and KSTEADY's timed steps start at about 21:15Z, when nothing may load the host. `nice` it. If it is not
   done by 20:45Z, leave it until after 22:50Z and say so.
2. **Thursday's design note, final by Thu 12 Nov 18:00Z**, with tonight's rows and piece 1 in, the cut's
   section, the memory plan for four provers (the schedule once a card, walks in turn), and the answer to
   proofs' question as it stands. I send it to proofs.
3. **Friday, my decision: way 1 as a try with a stop, and the plan does not hang on it.** I told infra and
   top (INF-682): Friday on node A is a first run of the kernel inside the prover if it is ready; the
   measurement of record is the week of 16 Nov on a large host I have asked infra for. So work on the
   kernel in the prover at full speed from now, **the memory stop first** (the shared schedule, the walks
   taking turns), then the mode, the buffer, the transposition, the compare with `pack_units` word for word.
   **Say by Thu 18:00Z whether a first run on a card on Friday is real**, with the hours it needs (I book
   them on node A's calendar then), and on which tree (tonight's measured tree for comparability, or `main`
   for the merge: say which and why). Proofs reviews the prover's changes; ring through me.
End your turn when the cleaning is done and the note's tonight section is in, with what you start on next.

## Wed 11 Nov 20:08Z: node A is clean and the note has tonight's section: go on with the kernel in the prover

Read by me: your return (the directory gone at 20:03:12Z, proofs' statement as it was). The memory plan's
reckoning (10,625 MiB free at the worst peak; the schedule once a card 5,630 MiB; one walk's working words
2,930 MiB) is the right first paragraph of the work, and you said plainly that it is a reckoning.
- **Go on, as you set out:** read the three things under the plan first (whether a prover already holds
  the unit's schedule on the card when the unit is built on the host; what the provers' peak is made of;
  whether the sharing path's fallback would bring back a copy a prover), then the shared schedule and the
  walks taking turns, then the mode, the buffer, the transposition, the compare with `pack_units`. A branch
  of your own; the CPU twin and the suites as you go; no GPU without my word.
- **The large host for the rows of record (infra, INF-682, 20:05Z):** probably `vy-dc-h200-3` at its
  two-card size (32 threads, about 400 GiB, one card used) for a four-hour block between Mon 16 and Wed 18
  Nov, confirmed by infra by Fri 20:00Z. Its rows do not compare with node A's, so the block times before
  and after on that machine. **I owe infra the block by Fri 20:00Z: say in your Thursday return which day
  the kernel is equal to the host on every word**, so I can name it.
- **Thu 12 Nov 18:00Z, as before:** the design note final (I send it to proofs), and whether a first run on
  a card on Friday is real, with its hours and its tree.
- **End your turn** at the first thing proofs must answer about its prover's code (name it for my ring), or
  when the memory stop's two changes pass on the CPU, or at Thu 16:00Z, whichever is first.

## Wed 11 Nov 21:37Z: the kernel in the prover is read and rung to proofs. Your card hour is Thu 15:45Z on node A's card 3; Friday's timed rows are booked

Read by me: the branch's head `714ceff06517`; `r20261111-205738-5017` and `r20261111-210923-14ce` rc 0,
`…a67d` rc 3 as you said. Leaving my order (turns between provers) for the walk inside the session's own
arena was right, and you said why in numbers (13,430 MiB against 10,625 free): that is how to depart from
a brief. The lost commit cost twenty minutes; push at every checkpoint from here.
**Rung to proofs (HASH-112, 21:35Z):** what it is, what holds it to the host so far, your five points to
read first, a read asked by Fri 15:00Z, and a yes asked for reading its staged statement on Thu and Fri.

**Your card time, both on node A (H200, the sm_90 binary), both on the calendar (pushed 21:35Z):**
- **Thu 12 Nov 15:45Z to 17:15Z, card 3, cores 36-45: the first run on a card, untimed.**
  `device_walk_matches_host` with its control, and `gpu_paths_agree` under `FC_UNIT_WALK`, on a small
  statement first and then on the head block's, one prover. Card 3 is on the other socket from card 6,
  where KMOVE may be timing a pair from 16:00Z: stay on cores 36-45 and on node 1's memory.
  memory-accounting's gates run beside you on card 2 and cores 24-35 until 16:20Z; both are untimed.
- **Fri 13 Nov 15:45Z to 19:45Z, card 6, cores 72-81, PROVISIONAL: the timed rows**, one, two and four
  provers with the switch set, today's block beside each in the same job, whole bursts. It runs only if
  Thursday's run equals the host on every word. I say by Fri 02:00Z whether it stands, from your return.
- **Not node 2:** its hours are memory-accounting's, and an H200 is the card the rows of record are on.
- **Proofs' staged copy** (`/dev/shm/proofs-prf502-row1-4b31e7caba011398`): I asked for Thu and Fri and
  whether it stays. Until proofs says yes, plan your own fetch from the store inside your hour as the
  fallback, and never write into or remove that directory.

**Before Thursday 15:45Z, in this order:**
1. **The twin on the head block's own unit, on a CPU node, tonight.** It has run on five test nets only.
   It is the strongest thing a CPU can say before a card hour is spent, and if it fails Thursday's hour
   is for the fix, not the run. Say on HASH-112 when it has passed or what differed.
2. **Thursday's job, dry-run:** the launcher with its start check under kernels' lock, the build on
   cores 36-45, the lease's length, what it leaves on the host and removes. If the walk's rows cannot be
   shared on that card, the run says so and that case is its own finding.
3. **A first look at the cost** in the same hour, after the equality: the head block with the switch set
   and unset, a few sessions each, labelled a look and not a row. The 4 to 7% is inferred; Thursday
   should replace the word.
4. **Friday's rows as KSTEADY's are run:** the rows' own script with #8953's functions, the ten checks
   told on each step (`verity_bench.steady` at #9034's head), the node's free memory beside each step,
   the record carrying `"unit_walk"`. Say what `server_rate.py` needs for the switch; KMOVE is adding
   `--wide-leaf` to the same file on top of #8953, so say on HASH-112 which lines you touch.
5. **The design note final by Thu 18:00Z**, with Thursday's result in it if the run has ended.

**End your turn** when 1 and 2 are done, or at the first thing that needs proofs. Launch nothing on node A
before Thu 15:45Z; vllm and memory-accounting hold every card until then.

## Wed 11 Nov 21:46Z: proofs said yes to its staged copy on Thu and Fri, and asks two things now

Proofs (HASH-112, 21:44Z):
- **Yes:** you may read `/dev/shm/proofs-prf502-row1-4b31e7caba011398` on node A, read-only, on Thu 15:45Z
  to 17:15Z (card 3) and Fri 15:45Z to 19:45Z (card 6). Nothing of proofs' removes it before node A lapses
  (Sat 14 Nov), and its worker is told to leave it. **It is in memory: a reboot of node A loses it.** A
  staged statement of the same row (`prf502-row1`, 12 GB) is kept on `vy-dc-h200-3` under a keep record to
  18 Nov; proofs has not compared the two byte for byte. So keep your own fetch as the fallback, and in
  Thursday's start check look that the directory is there before the lease is taken.
- **Its read of the five points comes by Fri 02:00Z** (Thu 6 PM PST), from the worker that timed the host
  and card unit paths on this prover, at `714ceff065`.
- **Two things it checks first, "say now if you already know":**
  1. Point 1's "twice a statement" against the reused-rate rows proofs quotes: a walk on the session's
     path moves the prover's seconds a proof block, which its cost model reads.
  4. That with `FC_FIREWALL=1` the refusal is the same line as today and not a new path.

**Answer both on HASH-112 tonight, as a worker comment named for my ring; I ring proofs with it.**
- For 1: the arithmetic, each figure with where it is from and whether it is measured or inferred: what a
  statement's prover seconds are in the reused rows, what two walks add at one and at four provers (your
  standalone figures: 0.079 s alone, 0.137 s beside three reused provers), what leaves (the copy of the
  units' a and b from the host in each rep), and so what a proof block's prover seconds become. Say
  plainly that the reused rows are a floor with no walk in them and that no row with the walk exists
  until Thursday's look.
- For 4: the file and line of the refusal today and on your branch, and whether your change adds any
  branch on the firewall's path. If the refusal moved or a second one exists, say which, and whether a
  test holds it.

## Thu 12 Nov 01:17Z: a second cut (22:30Z to 01:13Z, an unpaid invoice). Your order of 21:37Z and 21:46Z stands; carry on

Read your checkpoint and `git status` first; send nothing twice. Then, as ordered:
1. **Proofs' two questions (HASH-112, 21:44Z), answered on HASH-112 tonight, named for my ring:** what two
   walks a statement add to a proof block's prover seconds at one and four provers against the
   reused-rate rows, each figure measured or inferred; and the file and line of the `FC_FIREWALL=1`
   refusal today and on your branch, with whether any branch is added on that path.
2. **The twin on the head block's own unit, on a CPU node.**
3. **Thursday's job for node A's card 3, cores 36-45, 15:45Z to 17:15Z, dry-run.** Proofs said yes to its
   staged copy for Thu and Fri; it is in memory, so check it is there in the start check and keep your own
   fetch as the fallback. RunPod refuses new launches tonight: `inbox/RULES.md`'s newest bullet has the
   way round for node A; put it in before your dry run.
4. The design note final by Thu 18:00Z.
End your turn when 1 and 2 are done, or at the first thing that needs proofs. Checkpoint and push at each step.

## Thu 12 Nov 01:30Z: your return is read and rung. Thursday's row STANDS; the go comes from me at about 15:30Z

Read by me: the branch's head `e6ab4d9a982e`; `r20261111-221055-73a4`, `…215308-1549`, `…215510-51e1` and
`…221911-77e6` all rc 0; your five comments on HASH-112. **Rung to proofs (HASH-112, 01:27Z and 01:29Z):**
your two answers of 21:51Z (late by the stop, not by you), the twin on both statements, the head
`e6ab4d9a98` for its reader, the `walk-twin` subcommand, and that no test holds the firewall's refusal:
proofs says if it wants that test in this change. The twin on 805 million words with a control that
differs is the check that makes Thursday's hour worth a card.
- **Thursday 15:45Z to 17:15Z, node A, card 3, cores 36-45: the row stands.** I resume you at about
  15:30Z with the go. Before you launch, in the same minute: read the calendar row again; look that
  proofs' staged directory is on the node (a reboot loses it; your own fetch is the fallback); look that
  no lease and no compute process is on card 3; and if `inbox/RULES.md`'s table has changed or RunPod
  answers again, follow what the rules then say.
- **`launch-thu.sh` refusing when the table changed or the address differs** is right: no address is
  guessed.
- **Not yet tried by any of kernels:** a `research run` launched through the ssh entry. If Thursday's
  launch fails to connect, stop and say so in your return's first line; do not try another address.
- Until 15:30Z: the design note toward Thu 18:00Z; nothing on a card.

## Thu 12 Nov 02:03Z: until Thursday's hour, two things that Friday's timed rows need and no card does

You are idle until 15:30Z and Friday's rows are a timed run. Your own note names what could make Friday
slip; two of those need no card:
1. **The SASS gate for the walk's kernels in the prover's binary.** A timed run must pass it (the
   kernel-engineering skill), and your note says you have not read it for a Flock prover binary. Read how
   `benchmarks/pouw/harness/sass_gate.py` and its manifests are used, say what the gate would hold for
   `fc_walk_seed`, `fc_walk_level`, `fc_walk_level_warp` and `fc_walk_slots` built for `sm_90`, and run
   what can run on a CPU node (the cubin's SASS is read without a card). If the gate has no form for a
   Flock prover binary, say so on HASH-112 for my ring with what the smallest honest substitute is: I
   decide with proofs whether Friday's rows are timed results or a screen. Do not write a new gate
   tonight.
2. **Friday's job, dry-run:** one lease on card 6, cores 72-81, 15:45Z to 19:45Z; steps own, kernel, own,
   kernel at four provers first, then two and one if the hours allow; whole bursts in each window; the
   ten checks told on each step; the node's free memory beside each step; `steady.sh`'s `STEPS` with
   `FC_UNIT_WALK=1` as the step's variable (KMOVE is adding a step option for `--wide-leaf` to the same
   script tonight and says so on HASH-115: read it before you use the script). Say what the row's record
   must carry for benchmarks' reader to price it, and what it cannot carry.
3. If proofs answers on the firewall's missing test or on any of the five points before 15:30Z, I pass it
   to you here.
End your turn when 1 and 2 are done. Push at each step. Nothing on a card.
- **Added 02:17Z, two lines for Thursday's job, for KSTEADY's spec:** in its start check, before the lease,
  print `nvcc --version` as the node gives it, and whether the node's installed `gpu-lease` takes `--cores`
  (its help text, and the sha256 of the script it runs). Put both in your return; they cost a second and
  no card.
- **Added 03:06Z, for Friday's rows:** `ksteady-jobs/steady.sh` changed tonight (KMOVE, said on HASH-115 at
  02:05Z): a step's third field starting with `--` is an option of that step's `server_rate.py`, and the
  script now reads the tree's layout, so a tree after the move runs it (before tonight it could not). A
  variable in the third field works as before. The lease's `who` becomes a variable: set
  `kernels-khshare` for your runs. Read the script as it now is before your dry run.

## Thu 12 Nov 03:36Z: the lead's decisions on your 03:33Z return (the merge, Friday's pairs, the label, the gate)

Read by me: `sass-gate-finding.md`, `friday-plan.md`, your two comments. Good work; every decision you asked
for is below, with one condition you check yourself.

1. **The merge of `main` goes on the branch before Thursday's run**, on one condition: your run
   `r20261112-024714-c350` says 0 words differing on BOTH head-block statements at the merge. Then push the
   merge commit (a plain push, never forced; `merge-main.sh`'s tree `5ca429b7ff83`), remake Thursday's and
   Friday's dry runs at the new head, and post one worker comment on HASH-112: the new head, the twin's line,
   the prover tests' run. **If any word differs: push nothing**, Thursday runs `e6ab4d9a98`, and you tell me
   the count and the first differing word. Why before: Thursday is the walk's first hour on a card, and
   Friday must time the tree Thursday showed equal, against `main`'s host walk and not an older one.
   If `main` moves again before Friday, merge again only if the new commits touch the prover's walk or a
   file Friday's job builds from; say which if so.
2. **Thursday's run leaves its build in place** (`/workspace/jobs/kernels-khshare`, by its key), so Friday's
   build is the same key and costs nothing of the lease.
3. **Friday is `PAIRS="4:4 2:2 1:2"`**, four provers first, so one prover is what is lost if the hours run
   out. Four provers a card is where the rows of record are, and two pairs cannot call a difference
   whatever the rates.
4. **Friday's run gets no `flock-steady` label and no art.** It holds own and kernel rows at one count in
   one `rate.json`, and a labelled kernel row would publish a price of a prover nobody can build from
   `main`. The result is the comparison in one job, each step with its ten checks, said on HASH-112 and in
   the goal. `fri-after.sh` keeps each step's completed row; a priced row comes from a run on `main` after
   the kernel lands.
5. **The SASS gate: Friday's rows are timed results under the proof server's rules, not a screen, and
   every place that quotes a time from them carries this sentence:** "no SASS gate: the gate cannot read
   this binary (CUDA 13.3 stores the device code compressed) and holds no pin for its ptxas; on the
   unpacked code every flagged instruction is an integer division's, by a what-if that is not the gate."
   The gate is PoUW's: it guards floating-point words a verifier recomputes bit for bit. The walk's words
   are XOR and AND, and what holds them is the twin on the card in the same run before any step
   (`device_walk_matches_host`, every word), which your `GATE=1` already does. Say "accepted" for what
   it is: on that statement a Lean accept says "units as stated, the Program not held to the circuit"
   until proofs' PRF-567 lands, so the twin is the guard and the accept is not. Proofs may say no to
   this by Fri 02:00Z. The gate's read of compressed code is its own issue, mine to file, not yours.
6. **`steady.sh` in the notes is KMOVE's bytes again since 03:13Z, with `LEASE_WHO` as a variable**
   (sha256 `928547337114…`, `stub.sh` `2775700389df…`). When you remake Friday's dry run, base
   `fri-steady.sh` on that file and set `LEASE_WHO=kernels-khshare` instead of your edit of the lease's
   name, if the other four asserted edits still assert; it is the file KMOVE's Thursday screen runs on a
   card. If an edit does not assert, keep your base and say so; do not spend more than a few tries.
   KSTEADY is told why its 02:10Z sync put old bytes back, and of `complete_row.py`'s refusal and the
   lease's undeclared cores.
7. **Thursday is unchanged:** I resume you at 15:25Z with the go, after reading the calendar.

Return when the merge is pushed or refused and both dry runs are remade: the head, the twin's line, each
dry run's last line, and whether Friday's base is the notes' file.

## Thu 12 Nov 03:59Z: proofs read the kernel (HASH-112, 03:56Z); seven things it adds to this turn's work

**Proofs' reader holds all five points at `e6ab4d9a98`, has no "no" to Friday's rows being timed without
the SASS gate, and wants the firewall test in this change.** Read its comment whole on HASH-112 before
you remake the dry runs, so they are remade once. What it adds, in the order to do it:

1. **The firewall test goes on the branch, its own commit after the merge:** one test that
   `FC_FIREWALL=1` with a device witness is refused with the refusal's message, with `FC_UNIT_WALK=1`
   set (the message occurs once in the tree and no test holds it). Prover's tests run again on it.
2. **`FC_HOST_UNITS=1` wins over `FC_UNIT_WALK` without a word** (`gpu_circuit.rs:397-399`). So:
   Thursday's and Friday's jobs refuse to start a step with `FC_HOST_UNITS` set in its environment; and
   say where the row's `unit_walk` comes from. If it is read from the environment, make it the
   prover's own report of what it did, or a kernel row could be a host row under a kernel's name.
3. **`FC_UNIT_WALK=2`:** proofs says a values unit (828 levels) needs 2. Say which value Friday's
   statement needs and why, in the plan.
4. **The baseline.** Proofs' figure for the head block at two provers is 2.1425 GPU-seconds a statement
   (`r20261111-115454-5832`, `vy-dc-h200-3`, "the prover from branches"); its reader guesses 1.42 to
   1.62 with two walks a statement, and says your 4 to 7% was against the reused rows, which no
   configuration serves. **Read that run's record for its `FC_` variables and its tree, and say whether
   Friday's "own" (no variable, `main`'s prover) is that configuration.** I asked proofs the same. If
   it is not, Friday has the wrong baseline: tell me before anything else.
5. **Each prover process builds and hashes its own host copy of the walk's rows before it maps the
   shared copy**, once a process, untimed. Friday's report gives, a step: time to ready and the
   provers' peak memory (the row's own fields), own beside kernel.
6. **The condition on "no SASS gate".** Proofs wrote "each timed step's proofs are all accepted by the
   Lean verifier". A row's `accepted` is the proof server's own verifier, not Lean's
   (`server_rate.py`: the `verifier-` processes; `--lean` is two sessions at one prover and refused on
   this statement until PRF-567). I told proofs what Friday can do and asked if that is its condition
   (answer due Fri 02:00Z). **Build it now:** keep one proof of an own step and one of a kernel step,
   outside the timed window (KSTEADY's `keep-head.sh` is the way its kept proof was made); after the
   run, on a CPU node, `main`'s Lean verifier on each with the statement's Program (proofs used
   `program-word.json`, in the kept proof's `out/head/`; 159.5 s). Each verdict is said as "accepted on
   the provisional unit cover, units as stated, the Program not held to the circuit". If no Program
   exists for Friday's statement, say so and I ask proofs.
7. **The merged head:** proofs' reader checks by diff that the prover's files differ from `e6ab4d9a98`
   only by `main`'s changes (and the test of item 1). Post the head in your worker comment; I ring.

Your return is as before, plus: the test's name and its run, item 2's answer, item 4's answer.
- **Added 04:28Z, for item 6 (the two kept proofs under Lean):** benchmarks ran `main`'s Lean verifier on
  kernels' kept head-block proof three times on a CPU node (HASH-115, 04:25Z; its run
  `r20261112-033326-ef6c`): accepted three of three, 230.6 to 287.2 s a session on a shared slot, 84 s of
  it reading the statement, 40.9 GiB at peak. **Use its runner for your two proofs, not a script of
  your own:** read that run's command; it records which Program it was handed and checks the partition
  and the circuit's SHA-512 against `stage.json`, and its line carries the label word for word. Size
  the job for 41 GiB. If its runner cannot take a proof of yours as it is, say what it lacks.

## Thu 12 Nov 04:41Z: proofs said YES to Friday's condition (HASH-112, 04:37Z), with three lines; the baseline; a second card

Read its comment whole, and mine under it. **Friday's rows are timed results, with no SASS gate, on the
condition as I put it:** the twin on the card before any step; every timed proof accepted by the proof
server's verifier; after the run one kept proof of an own step and one of a kernel step through `main`'s
Lean verifier with the statement's Program. What proofs adds, and what follows:

1. **A row's words:** "accepted by the proof server's verifier", with that verifier's binary and commit
   named. Never "accepted by the Lean verifier", never "verified" alone. Beside it, once a kind of
   step: "Lean verified 1 kept proof of N: accepted on the provisional unit cover, units as stated, the
   Program not held to the circuit". `fri-after.sh` prints exactly these.
2. **Which proof is kept is fixed before the run and written in the job**, and it comes from the timed
   run's own provers, not from a second process. This replaces my "outside the timed window". Proofs'
   example: rep 0 of the first timed step of each kind. Say what keeping it costs inside the window;
   if it would change a timed session, pick that step's last warm session of prover 0 instead and say
   so, and I put the pick to proofs before Friday.
3. **A Lean refusal of a kept proof:** that kind's rows are not published as accepted. No second pick.
4. **The baseline (read by me on `main`: `gpu_circuit.rs`, `deep_unit`):** a unit of 2,048 levels or
   more is evaluated on the host unless `FC_HOST_UNITS` says otherwise, and the head block has 11,516.
   **So an own step, no variable, is already the host's walk**, and Friday compares the host's walk
   with the kernel, one thing changed. Proofs' 2.1425 is a branch build with `FC_SALTS_KEYED=all`
   (not on `main`: no file under the prover names it); the same run's arm on the code as it was gave
   3.1174. **On Thursday's run, confirm from the prover's own report which path an own step takes and
   that `FC_UNIT_WALK=1` is what changes it;** if my reading of `deep_unit` is wrong, tell me first.
   A step still refuses `FC_HOST_UNITS`: no kind needs it. Item 4 of my 03:59Z section is answered by
   this; you need not read `5832`'s record.
5. **A second card on Friday: card 2, cores 24-33, 15:45Z to 19:45Z, booked provisional** (the other
   socket from card 6). Make the launcher take the card, the cores, `PAIRS` and the lease's name, so
   two runs go at once, each recording the other on every step:
   - **card 6:** four provers only, as many pairs as fit whole (six, by your hours), kernel and own
     alternating;
   - **card 2:** two provers (four pairs) then one (two pairs), on `main` with the walk. **Unless
     proofs names its keyed-salts branch by Thu 18:00Z and it merges with the walk's branch:** then
     card 2 times own and kernel at two provers on that tree, `FC_SALTS_KEYED=all` in both kinds,
     which is the configuration proofs' cost model reads. I tell you which by Thu 19:00Z.
   Dry-run both forms. Say what two runs at once need that one did not (the build's key, the copy of
   the statement, the lock).

Your return is as before, plus the pick of item 2 and the two launch lines of item 5.

## Thu 12 Nov 05:17Z: your 05:15Z return is read; proofs rung with the head; six words of mine, one small job before Thursday

Read: your return, the comment, `launch-lean.sh`'s header. Checked by me: the head `ffc7c70a0f`, the merge's
parents, both runs done rc 0. Rung proofs (HASH-112) with the head, the pick, the test as skipped on a
CPU build, and what the 2.1425 arm's tree is. Good work: the pick's reasoning and the two-run launcher
are what was needed.

1. **`PUT=1`: yes**, for Friday's kept proofs (two a card, four in all) and for Thursday's one (item 3).
2. **Card 2 on Friday: `2:6 1:2`.** Card 6 stays `4:6`.
3. **A rehearsal on Thursday of what Friday cannot repeat:** proofs' rule is that a Lean refusal of a
   kept proof unpublishes that kind's rows, with no second pick, and no proof of the kernel's has been
   under Lean. So Thursday's run keeps ONE proof of a session proved with the switch on (from a look
   step, by the same keep code as Friday's), and after the run `launch-lean.sh`'s path puts it and
   runs the Lean verifier on it, on a CPU node. The keep is made after the step's processes exit and
   cannot fail or lengthen the card's hour: a failed keep is said and the run goes on. If a look step
   leaves no session a kept proof can be made of, say so and do not bend the job.
   **A Lean refusal on Thursday is the most useful thing Thursday can find: tell me at once.**
4. **The Lean verifier is `main`'s, named by commit in the Lean line.** `main` is at `62bf29346f76` and
   its verifier changed since benchmarks' run (#8949, #9042: the program is read once a call). If the
   runner's tree is not `main`'s verifier file for file, run it from a scratch merge of the runner's
   branch onto `main`'s head (never pushed), and say so. I asked benchmarks when the runner lands.
5. **The keyed-salts form is off for Friday as things stand** (checked by me): proofs' tree is
   `7b12b8825b` on `cursor/proofs-headprof-2b18`, on the layout before the move, and a merge with
   your head conflicts in eight files, four of them the prover's. Do not port anything. If proofs
   brings its follow-up to today's layout by Thu 18:00Z, I tell you by 19:00Z.
6. **Two leases of kernels' at once on Friday, 400 GB and 200 GB:** read `main`'s `gpu_lease.sh` for
   what `--mem-gb` reserves against, and say whether both are granted on node A; Thursday has three
   of kernels' runs on the node (yours, KMOVE's from 16:00Z, KSTEADY's from 17:30Z), so look at what
   the leases report then.

**Thursday is unchanged otherwise: I resume you at 15:25Z with the go after reading the calendar; the
design note is final by 18:00Z, with what the card showed.** Return when Thursday's job has item 3 and
its dry run is remade (its last line), or say why it cannot; then end your turn.
- **Added 05:19Z: benchmarks answered (HASH-115, 05:18Z), and it is what item 4 says.** The runner (#9068) is
  expected on `main` by Thu midday, before Fri 20:00Z for certain; it builds the verifier from the
  run's own clone, so a run launched from `main` gets `main`'s. **For Thursday's kept proof, if #9068
  is not on `main` by then: launch from a merge of `main` with its branch** (benchmarks' own word);
  each record names its commit. Its reader refuses a record without proofs' label.
- **Added 05:33Z: proofs took the card-only firewall test (HASH-112, 05:32Z), with one condition for both
  jobs:** Thursday's job and Friday's gate FAIL if `zk_firewall_refuses_device_witness` reports skipped
  or is missing from the selftest's output, and the run's record names it as run. A CPU build
  skipping it is fine. A second case for a CPU build is welcome if the refusal's decision can be
  reached without a device: proofs' reader says whether; write none before its word.
  **On the keyed salts:** proofs' reader says by Thu 14:00Z which prover commits of `7b12b8825b` are
  on the timed path. I told proofs the switch itself (`salts_keyed`, `flock-circuit.rs` there) is on
  neither `main` nor your head. If a branch of today's layout has it by Thu 18:00Z: merge it with
  your head in scratch, the twin on that tree first, then card 2 at `2:8`; I say by 19:00Z.
- **Added 05:52Z: proofs ports the keyed salts' switch itself (HASH-112, 05:51Z)** onto a branch cut from
  `main`'s tip and names it and its head by Thu 14:00Z, or says no at that hour. It holds only what
  the 2.1425 arm's timed path needs; proofs checks on a CPU build that with no variable the prover's
  bytes are `main`'s and that with the switch on a small statement's proof verifies under Lean.
  **If named:** you merge it with your head in scratch (never pushed to your branch), run the twin on
  that tree first, and card 2 times own and kernel there at two provers, `FC_SALTS_KEYED=all` in
  both. **Its rows carry one more sentence:** the form of the salts as proofs' entry words it, and
  "a timing of that form and nothing more" until proofs says a statement covers salts derived that
  way. I tell you which form card 2 runs when I resume you at 15:25Z.

## Thu 12 Nov 06:18Z: your 06:17Z return is read; my two words, one risk to close, and the keyed-salts plumbing now

Read: your return. Good: the keep that cannot change the exit code, the trial of benchmarks' runner on a
real kept proof (376 s, accepted), and the plain answer on `--mem-gb`.

1. **Your choice 1: yes.** A Thursday run that fails only on the firewall case still does the equality
   cases, the look and the keep, then exits 24 or 25. The hour on the card is not spent for one case.
2. **Your choice 2: not as you have it.** The firewall case is REQUIRED where it is cheap and its
   refusal is certain to be reached, the selftest's own small statement: skipped, missing or not
   refused there fails the run. **On the head block's statement it runs LAST, after the equality
   cases, the look and the keep, under its timeout, and a timeout there is said and timed, not a
   failure.** Proofs asked for one test; a case nobody has run must not be able to eat the hour.
   Friday's gate takes what Thursday shows: if the head block's case ended in under a minute, both;
   else the small one only, and the record says which ran.
3. **The frame's transfer must not be able to stop the launch.** 130 MB to node A is untried. Send it
   so that a failed or slow transfer leaves the run going without it: the run then does everything
   but the firewall case and exits 24, saying the frame was not there. If `research run --send`
   cannot be made to behave so, send the frame by its own step in the row's first minutes and let
   the job look for it. Say in your return which you did.
4. **Now, before Thursday: the keyed-salts form's plumbing**, so that a name from proofs at 14:00Z
   costs you minutes while you are on the card: `launch-fri.sh` takes a scratch tree (the walk's
   head merged with a named branch, never pushed to your branch), with its own build key, its own
   frame build if its Lean sources differ, the twin on that tree as the first step of the job, and
   `fri-after.sh`'s extra sentence ("a timing of that form and nothing more", with the form as
   proofs' entry words it). Dry-run it with a stand-in branch that changes nothing in the prover.
   It stays refused on a tree where no prover file reads `FC_SALTS_KEYED`.
5. **The SASS gate's sentence changes when #9156 is on `main`** (KMOVE's fix, marked tonight: the
   gate now reads the compressed entry). Then Friday's record carries the gate's own verdict on the
   timed binary instead of "cannot read": it fails, with its count of flagged instructions, all
   integer divisions by its own sequences, and no pin for that ptxas. Run it on Friday's binary
   from a tree that has #9156 and keep its record beside the rows; until it lands, the sentence I
   gave you stands.

Return when item 4 is dry-run and items 2 and 3 are in Thursday's job (its dry run's last line), or
at 12:00Z with where you are. Then nothing until my go at 15:25Z.

## Thu 12 Nov 06:20Z: proofs' reader's verdict on `ffc7c70a0f` (HASH-112, 06:10Z): it STANDS; two things for you

Read its comment whole. The head is the walk it read, `main` merged, and the new case; nothing the
verifier reads changes; the case reaches the refusal you say and no earlier one.

1. **What the card's job must hold, which the tree does not:** when the case does not run (a CPU
   build, no Lean frame, no `--zk`) it prints `"pass": true` beside `"skipped"`. **So the job fails on
   a `skipped` key, never on `pass` alone,** and holds `"gpu": true` and
   `"unit_walk_asked": "FC_UNIT_WALK=1"` on the case's `NEG` line. Check Thursday's job and Friday's
   gate do exactly that, on the case's real output format, and say so.
2. **Its reader's three lines, which I want before Thursday if they are green in time:** the refusal
   cannot be reached in a CPU build (`WitData` has one variant there, `flock-circuit.rs:1045-1054`,
   so the `else` at `:4332-4334` never runs). Deciding it from `witness_mode`'s first value (`:3908`)
   before the witness is built, with the same text, lets a CPU case hold it, so `main`'s check holds
   it, and it makes the head block's case cheap, which settles my item 2 of 06:24Z. **Do it as its
   own commit, not pushed until:** a selftest case that runs and passes in a CPU build (not
   skipped) holds the refusal and its text; the prover's tests pass at the new head; the twin on the
   small statement differs in 0 words. Then push, remake Thursday's dry run, start the twin on both
   head-block statements at the new head in the background (it bears on nothing Thursday needs: the
   card's own equality cases decide), and post the head in one worker comment for my ring: proofs'
   reader checks the three lines by diff.
   **If it is not green by 13:00Z, push nothing: Thursday runs `ffc7c70a0f` and the three lines wait
   until after Friday's rows,** because Friday must time the tree Thursday showed equal.
   If the refusal moves before the witness, say what else on the firewall's path now happens later
   or not at all (a session that used to fail after building a witness now fails before): proofs
   must hear that in the comment.

## Thu 12 Nov 06:35Z: PROOFS NAMED THE KEYED SALTS' BRANCH (HASH-112, 06:32Z); Friday is three cards

**`cursor/proofs-keyedsalts-811d` at `8122c0b408ef9dd620a4fccda6f83cca2d1802a7`, one commit on `main`
`d6075b75d289`:** the switch alone, two prover files (`live/src/bin/flock-circuit.rs`,
`live/src/zk_hooks.rs`), +159 −23, one `SALTSKEYED` line on stderr once a session when on, two unit
tests. It merges with `ffc7c70a0fea` with no conflict (by merge or by picking the commit). Read proofs'
comment whole. Its checks were CPU builds: with no variable the transcript digest is `main`'s; with
`FC_SALTS_KEYED=all` a small `--zk` session is accepted by `main`'s `flock-verify`, also on a merge
with your head. **Not done by anyone: a GPU build, the walk run with the switch, a card.**

**Friday's three runs, all 15:45Z to 19:45Z, each its own lease, build, copy and gate:**

| card | cores | tree | steps |
|---|---|---|---|
| 6 | 72-81 | your branch's head, as Thursday ran it | `4:6`, own and kernel |
| 2 | 24-33 | a scratch merge of that head with `8122c0b408ef`, never pushed to your branch | `2:8`, own and kernel, `FC_SALTS_KEYED=all` in BOTH kinds |
| 3 | 34-43 | your branch's head | `2:6 1:2`, own and kernel |

Card 3 is booked provisional tonight for what card 2 gave up. Cards 2 and 3 share a socket; each run
records the other two on every step.

1. **Before Friday, on CPU nodes, for the scratch merge:** it builds for `sm_90` as the rows build
   it; the prover's tests pass; the twin differs in 0 words on the small statement and on both
   head-block statements WITH `FC_SALTS_KEYED=all` set (the salts are not the witness, so 0 is what I
   expect: show it); and the `SALTSKEYED` line appears once a session with the switch on and never
   with it off. The scratch merge is remade from your head if the three-lines commit moves it.
2. **A keyed step's kind is read from the provers' own lines,** as `unit_walk` is: a step named
   keyed whose sessions show no `SALTSKEYED` line is void. The launcher refuses the form on a tree
   where no prover file reads the variable, as now.
3. **What every keyed row carries, in proofs' words:** "with `FC_SALTS_KEYED=all`, every leaf's
   192-byte hm96 salt of every tree of a `--zk` table is a slice of a ChaCha20 keystream under a
   32-byte OS key read once per tree sample, nonce the tree's id, in place of 192 OS bytes a leaf.
   The session's identity still says `leaf_scheme` 'os bytes (getrandom), 192 per leaf', which is
   then untrue of those trees. No statement of proofs' covers salts made that way: a timing of that
   form and nothing more." `fri-after.sh` prints it for card 2's run and for no other.
4. **The kept proofs of card 2's run** (one own, one kernel, both keyed) go under Lean like the
   others; proofs' check 2 says a keyed session is accepted with its Program. Six kept proofs on
   Friday in all.
5. **One fact for the comparison:** `main`'s prover has a change on the timed path that the 2.1425
   tree lacked (`00be487125`, local allocation for a GPU prover's large allocations). So card 2's
   own-keyed step is not expected to print 2.1425, and it is on node A, not `vy-dc-h200-3`. The
   report sets the two side by side and says both differences.
6. **`FC_HOST_UNITS`:** proofs confirms no file of `5832`'s record names it. Your reading stands.

This joins the work of my 06:24Z and 06:27Z sections; the 13:00Z rule for the three-lines commit is
unchanged. Return as said there, with the three launch lines and item 1's runs.
- **Added 08:28Z: benchmarks' Lean runner (#9068) is on `main` since 07:25Z** (benchmarks, HASH-115), so
  `launch-lean.sh` runs it from `main` and needs neither its branch nor a scratch merge. Benchmarks
  also asks that a kept proof carries its statement's `program-word.json` from the same staging;
  yours is proofs' staged head block, whose program is `art:84d25bec…/out/head/program-word.json`,
  as you have it.

## Thu 12 Nov 08:50Z: your 08:47Z return is read; proofs is rung with `f9fc32ff3f`; my words on your six readings, and two small things before Thursday

Read: your return, `c-three-lines.md`, `c-merged-main-2.md`, and the three-lines commit's diff whole
(`no_recheck = firewall_on(st) && device`; clean). Checked: the head and its two commits; runs `7d09`,
`79b3`, `7b90`, `4795` done rc 0. Rung proofs (HASH-112) with the head, what the three lines do, what
is shown and what is not.

**One correction for your record, and it is the kind that costs trust:** `c-three-lines.md` cites
`r20261112-063042-3e5f` as showing the case runs and holds the refusal. That run's state is failed,
exit 6 (your control's time cap). The case's own line may have passed inside it, but a failed run
cited as "shown" without its exit is not a citation. I cited `r20261112-073258-79b3` to proofs
instead, where the case passes with a frame and the run exits 0. Say a run's exit with its id.

**Your six readings:**
1. Friday's gate takes the head block's firewall case only where Thursday saw it pass in under 60 s: **yes.**
2. "The twin on that tree first" is the gate's two equality cases on the merged tree's own build with
   `FC_SALTS_KEYED=all` set: **yes.**
3. Card 2 is refused without `MERGE` unless `FORM=head`: **yes.**
4. The 13:00Z cutoff applies to any further merge of `main`: **yes. Thursday runs `f9fc32ff3f`, and
   Friday's cards 6 and 3 time that same head**, whatever `main` does to the prover meanwhile; a
   later merge waits until after Friday's rows.
5. The node script rebuilt on KSTEADY's current `steady.sh`, `ROWS` refused: **yes.**
6. The gate on the timed binary: **do not read anything on node A for it.** Have Thursday's and
   Friday's jobs put the timed binary's device code (or the binary, if it is small enough for the
   run's custody: say its size) under `$OUT`, with its sha256 in the row, so the gate runs on a CPU
   node on exactly what was timed. Then no hash comparison with a CPU build is needed at all.

**Two small things now, on CPU nodes, back by 12:00Z:**
- **Your control `zk_worker_wire`:** it stopped at its time cap twice at `90c428b40e` and gave no
  verdict. Run it at `ffc7c70a0f` (the commit before the three lines) and at `main`, same cap and
  one with the cap doubled: slow everywhere, or a hang the three lines brought? I told proofs I
  would say. If it is a hang at `90c428b40e` only, tell me at once: the three lines then come off
  before Thursday (the 13:00Z rule) and Thursday runs `ffc7c70a0f` merged with `main`.
- Item 6 above in both jobs, and both dry runs' last lines again.

Then nothing until my go at 15:25Z. The keyed-salts form on card 2 is on (proofs named the branch;
your scratch merge's CPU checks are in): Friday is three cards as your three lines say.

## Thu 12 Nov 09:31Z: proofs read the walk's branch at `f9fc32ff3f` (HASH-112, 09:25Z): it stands; three things asked, one of them BEFORE Thursday's card run

proofs' reader read `90c428b40e` and the merge of `main` `bcaf58a45d`: the three lines are the ones
proposed, in the place proposed; a firewalled session on a host witness is untouched; the keyed
salts' tree is not touched by the switch. "None blocks Thursday's card run." Read its comment whole
on HASH-112 before the go. What it asks of kernels, none with an hour from proofs; the hours are mine:

1. **Before the card run (in the job, by 15:00Z):** the card's job holds `"device_build": true`
   beside `"gpu": true` and `unit_walk_asked`. At this head a build without the device handed
   `--gpu` prints `"gpu": true, "device_build": false`, so `"gpu": true` alone no longer says a
   device build ran. Dry-run it on the stand-ins both ways.
2. **In the row's record and the job's lines:** the firewall case "RAN on the device" reads **"ran
   in a device build, asked for the device"**. proofs' note 3: the session is refused before any
   witness is built, so on a card that case shows the decision in a device build, what a CPU build
   shows; the recheck's own refusal of a built device witness can no longer be reached, and no run
   exercises it. Say exactly that in your return and wherever the case is named; do not call it a
   device witness refused.
3. **On the branch, one commit, after the card run:** `live/PROTOCOL.md:841-842`, proofs' note 1:
   "a session whose witness the device would build" reads "a session asked for the device with no
   `FC_HOST_WITNESS`". On a device build the new decision also refuses a firewalled `--gpu` session
   with no `FC_HOST_WITNESS` whose witness the prover builds on the host anyway (a typed template's
   statement, four of the selftest's witness controls); it only refuses more.
- **Still not shown, by proofs' list:** the case on a card (Thursday's is the first); a device
  prover's firewalled session on a host witness (`--gpu` with `FC_HOST_WITNESS=1`) at this head. If
  the second fits Thursday's lease after the walk's equality, run it and say what it gave; if not,
  say so.
- Your `zk_worker_wire` control: proofs reads from the diff that its five sessions set
  `FC_HOST_WITNESS=1` and no `--gpu`, so none of the three lines runs in them, and
  `r20261112-075907-4795` has firewalled CPU sessions accepted at this head. Your own answer on
  whether it timed out or hung is still owed at 12:00Z.

## Thu 12 Nov 09:37Z: THREE OF YOUR RUNS RAN ON A BROKEN NODE: `/dev/null` on `vy-nebius-cpu-7` was unwritable from 09:13:36Z to 09:29:11Z (infra's mistake, INF-735, fixed)

In those 16 minutes every program there that opens `/dev/null` for writing failed (`git`, a shell's
`> /dev/null`, `subprocess.DEVNULL`). infra names three runs of yours; I fetched the two that ended
and read them:

| run | what infra says | what I read |
|---|---|---|
| `r20261112-091114-45bd` | failed, exit 9 | its build step failed in 1 s on "`/dev/null`: Permission denied" (`check_build.sh` lines 32-33, `control.sh` line 33, `git`); its verdict's source line is empty. **The node's failure, not the tree's.** |
| `r20261112-091108-2f1c` | exit 0, ran through the window | at `ffc7c70a0fea`: the build ended before the window; **its whole control (451 s, from about 09:12:55Z) ran inside it** and says "zk_worker_wire: pass true", the honest session accepted with 223 requests, four modes ok. No "Permission denied" line in its logs. |
| `r20261112-090112-a1df` | still running, ran through the window | not read by me |

**What follows, for your answer on whether `zk_worker_wire` timed out or hung:**
- **No conclusion rests on any of the three.** A child that could not open `/dev/null` is a way to
  hang and a way to fail, and a negative mode that passes because its session failed would pass on
  such a failure too. So neither `a1df`'s behaviour nor `2f1c`'s pass is evidence yet.
- **Run the pair again, both commits, on one node that is not in doubt, started after 09:30Z**
  (name the node with `--on`; not `auto` blind), and compare line for line with `2f1c`: the
  requests' count and the four modes. If `2f1c`'s lines repeat, say so and it stands.
- **Say in your return, for each negative mode, what its "ok" rests on:** the refusal's own text or
  key, or an exit status alone. If it is an exit status alone, that is a finding for proofs, and I
  ring.
- If `a1df` is the run you are waiting on for "hung or not", stop waiting on it: cancel it by its
  run id and start it again.

## Thu 12 Nov 10:28Z: your 10:25Z return is read; (a) is rung to proofs; the pinned `steady.sh` is confirmed on one condition; until 15:00Z, the fused reads on a CPU

Read whole: your return and your comment. Fetched and read by me: the pair on the clean node
(`…-762f` at `ffc7c70a0f`, `…-3216` at `90c428b40e`: the same line, 223 requests, four modes ok) and
`…-aad5` at `f9fc32ff3f` (256 units, 275 requests, 824 s). Rung: proofs on HASH-112 with the answer,
what each mode's "ok" rests on (as your reading of its test), and its three asks' state. You reran
the pair and cancelled `a1df` off my note mid-task, and said what each "ok" rests on: that is the
answer I could give proofs without a second question.

**Confirmed:**
- **Friday's job builds from a pinned copy of KSTEADY's `steady.sh` (`afe4733816e722d0`): yes**, a
  script a job reads must not change under it. One condition: each row records that sha256, and
  **before Friday's go you say what differs between the pinned copy and KSTEADY's copy of that
  hour, and that no timed step, check or `finish.sh` line is among it**; if one is, take KSTEADY's
  and dry-run again. The rows are "by KSTEADY's method", so the method must be the one its rows of
  record used.
- The code comment in `flock-circuit.rs` waits until after Friday's rows; the `PROTOCOL.md` sentence
  is one commit after the card run. Yes to both.
- The device prover on a host witness as Thursday's last step, never the run's exit: yes.

**Until 15:00Z, the issue's next piece: the fused reads** (rows 3 and 5, pins `7e8a66f6cf13209d`,
`1fd651999b36433d`; "with a date once the head block's form is known"):
1. **On CPU nodes, with the binary of `f9fc32ff3f` and no source change:** the walk's twin on the
   two fused reads' full statements: the differing words (0 is the pass), whether every deep unit
   of theirs goes through the walk or which do not and why, seconds and peak memory.
2. **If the prover needs a change for them, it goes on a new branch on top of `f9fc32ff3f`**, never
   on that head: Thursday's and Friday's rows run the sources proofs read.
3. **The issue's first item is due today:** the design note says, for BOTH types, where the witness
   is built today, what a device evaluator must compute, how it is held to the CPU builder bit for
   bit, and exactly what changes in the prover outside the kernel, with that list posted on HASH-112
   for proofs. Say whether the note covers the fused reads and whether the list is posted; if not,
   post it today as a worker comment and tell me: I ring.
4. **Only if 1 is clean by 13:30Z:** one fused read's equality on the card as one more last step of
   Thursday's job, after everything of the head block's, never the run's exit, inside what the lease
   has left, and the dry run holding again by 15:00Z. If 1 is not clean by then, or the dry run
   moves, Thursday's job stays exactly as it is at 110 of 110.
5. From that, the date you would put on rows 3 and 5.

Not `vy-nebius-cpu-7` blind, as before. Return at 15:00Z with it, or earlier; my go for card 3
follows at about 15:25Z.

## Thu 12 Nov 11:16Z: your 11:14Z return is read; both comments are rung; step E stays; YES to the cut's CPU twin on a new branch until 15:00Z

Read whole: your return and both comments. The twin's run and the rehearsal read as done, rc 0, by
me; your design note's new section read for its headings and the head block's 0.08 s. Rung: proofs
on HASH-112 with the twin (0 differing words on both), "slow as written", the list, step E and
your dates; I told it the prize is at most about 1.8% of SPOT's proving and that it is proofs' to
say if the cut should not be built. **A control that differs on 18 million words beside a zero is
what makes the zero mean something; and you corrected your own launch figure in the note.**

- **Step E stays** in Thursday's job at 149 of 149, on its bounds (after step H, only where every
  equality case read EQUAL, never the run's exit, `FUSED=0` leaves it out). At the go I ask one
  thing: the job's sha256 line as you have it now (`firstcard.sh` `91bff2cb…`).
- **The cut: yes, start it**, on a new branch over `f9fc32ff3f`, CPU only: the host-side twin with
  native seeds first, held to the host builder word for word on both full statements, with a
  seed-off control. Nothing of it reaches Thursday's or Friday's jobs. If proofs says on HASH-112
  that the cut should not be built, stop where you are and say what you have.
- **Before 15:00Z, whatever the cut's state:** stop, and be ready for the go. Thursday's card run
  comes before everything else today; the design note is final by 18:00Z; the pinned `steady.sh`'s
  diff before Friday's go.
- **The card on Mon 16 Nov** is mine to get: the H200 infra holds for next week (INF-682), whose
  hours I name on Friday. Say in your 15:00Z return how many minutes the fused reads' equality and
  their rows need there, so that the block I name covers them.

Return at 15:00Z, or earlier if proofs stops the cut.

## Thu 12 Nov 12:54Z: your 12:53Z return is read; the cut is rung to proofs with a question; nothing more on it until proofs answers; rest until the go

Read: your return; the twin's run `r20261112-123202-e22e` fetched by me (row 3: `cut_levels` 14,000,
`words_differing` 0, the seed-off walk not ok; row 5: 3,953, 0, not ok; the host's 0.52 s and
0.56 s); the branch (`fefcd53a76`, 16 commits over `f9fc32ff3f` by `git rev-list`, you wrote 17;
+1,703 lines in 6 files); the walk's branch unmoved. Rung: proofs on HASH-112 with the table.
A form designed, written, held to the host on 1.6 billion words of each statement and controlled
in an hour and a half: and you said which numbers are inferred.

- **What I asked proofs:** by your own inference the cut gives row 5 a real gain (0.56 s to about
  0.2 s with its seeds) and row 3 a small one (0.52 s to about 0.45 s), for two types that are 4.1%
  of SPOT's proving. So whether it gets next week's card and a second review of the prover is
  proofs' to say, by Fri 16:00Z. **Until it answers: nothing more on the cut**, no Monday job, no
  change to `server_rate.py`. The branch stays as it is.
- **Thursday's job as you have it:** `firstcard.sh 91bff2cb…`, 149 of 149. The go is at about
  15:25Z. Until then nothing is owed; the design note is final at 18:00Z with step E's time.

## Thu 12 Nov 13:20Z: proofs ruled on the cut (HASH-112, 13:18Z): NO card for it next week; it stays a branch; the issue ends at the head block

"no card for the cut next week. Ask for the head block's hours only; the cut stays a branch." Its
reason: by our figures the cut saves 0.36 s of 0.56 s on row 5's witness and 0.07 s of 0.52 s on
row 3's, at most 2.6% of SPOT's proving seconds, against 1,703 lines behind a flag and six changes
to read; its kept-process rows are a factor of 2.3 to 3.1 from the target, and a 2% lever does not
close that. It takes it up again only if a design puts more than about 10% of the proving seconds
in the fused reads. "The CPU result stands on its own."

- **So: nothing more on the cut.** No Monday job for it, no change to `server_rate.py`, no pull
  request. The branch `cursor/kernels-unit-cut-0242` stays at `fefcd53a76`.
- **Record it where the next one looks:** in the design note, one section that says what the cut
  is, what it measured on a CPU, what is inferred, proofs' ruling with its time and its condition
  for taking it up again, and the branch. That is the note's last piece with step E's time, final
  at 18:00Z.
- **Step E stays in today's job**, as its last and optional step: it costs the head block
  nothing, checks the walk on a card at twelve times the head block's depth, and gives a measured
  launch cost where the ruling rests on an inferred one. Say its seconds in your return.
- **Your card ask for Mon 16 Nov is withdrawn.** What I name to infra on Friday for you is the
  head block's four-prover rows only; say at the go how many hours those need on the H200's
  two-card size.
- **Added 14:46Z, for Friday (proofs, HASH-112, 14:46Z):** "when `PUT=1 bash launch-lean.sh RUN` has
  run for a run, please post that run's two art ids here (its own step and its kernel step)." Its
  worker checks them as members of one kept verifier process the hour they exist. So on Friday,
  each time that put has run for a run: one worker comment on HASH-112 with the run and its two art
  ids, and say so in your return; I ring.

## Thu 12 Nov 15:17Z: THE GO for card 3: 15:45Z to 17:15Z, cores 36-45, the head block's level walk's first run on a card

The calendar read by me at 15:15Z: your row stands (kernels, card 3, cores 36-45, 15:45Z to 17:15Z).
Beside you on the node in those hours: memory-accounting on cards 0, 4 and 5, proofs on card 7,
KMOVE on card 6, vllm on cards 1 and 2. **Behind you: memory-accounting has cards 3 and 7 from
17:15Z to 17:28Z, and circuits has card 3 from 17:30Z. So you are off card 3 by 17:15Z, whatever
step you are in.**

- **Launch `firstcard.sh` (`91bff2cb…`, 149 of 149) at `f9fc32ff3f`**, as dry-run. In the minute
  before: this file and the calendar read again; proofs' staged directory on the node (read only);
  card 3 with no lease and no compute process; if the launch cannot connect, stop and say so.
- **Bring back:** the twin's verdict on the card for both head-block circuits (words differing,
  with the control), the two paths' proofs compared byte for byte, the firewall case in proofs'
  words ("ran in a device build, asked for the device"), the device prover on a host witness (the
  two sessions), step E's verdict and seconds, the timed binary's sha256, and what did not run.
- **If the twin differs anywhere on the card: stop, nothing is pushed, and Friday's rows do not
  run until I have read it.**
- **After the run, today:** the `PROTOCOL.md` sentence as one commit; the design note final by
  18:00Z with step E's time and the cut's section; and the hours the head block's four-prover rows
  need on next week's H200 at its two-card size (I name them to infra on Friday).
- **For Friday** (the lines above this section): proofs' ask for the two art ids a run; the pinned
  `steady.sh`'s diff before the go.
- The agent guard's rules are in `inbox/RULES.md` ("Every turn", the last bullets): a push in a
  command of its own with `git -C`, no listing that prints a command line.

Return when the run has its verdict and the card is released, at 17:30Z at the latest.
