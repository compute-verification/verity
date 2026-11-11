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
