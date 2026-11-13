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

## Thu 12 Nov 19:07Z: KMOVE's handoff for you is written: read it before you go further

It is at the top of `~/.research/notes/lanes/kernels/20261110T0806Z-report-kmove.md` (after
`research notes sync --path lanes/kernels`): ten points with every line number (at #9375's
`238ad6621c27`; the arm's lines are four further down from `e209a56847d4`) and the run table.
What matters most for your design, as KMOVE gives it (relayed; check each in the source):
1. **The wide path in `run.py`:** the bounds and table `:347-351`, `rs_refusal` `:405`; the header
   states `rs_heads_w` are made in `Pipeline.__init__` `:648-652`, only when `sha` is `fast`, a
   `segdig_*` buffer exists and k/256 > 128; the launches are in `row_digests` `:760-782` and need
   scratch.
2. **`served_y.py`:** `_pipe` `:94-100` gets the states through its `segdig_b` key; `_tree`
   `:102-125` allocates tree, digests and scratch with torch on every call (the arm's `launch` may
   not: its buffers are made in `prepare`). y's rows are n words, so the segment count is n/256.
3. **Widths:** gate_up's n = 57,344 is 224 segments, so its y tree needs the wide launches even at
   the four linears' shapes. The cubin pinned in `overhead_arm.json` (`cdf0e443…`) has no wide
   kernel; `5a6d7461…` has (built at `main` today with both gates passing).
4. **`main` changed the arm:** #9247 renamed `arm.Gate` to `arm.GateResult` and typed its methods;
   #9375 holds it from `e209a56847d4`.
5. **The stand-in dry run and the host twin**, the tests that hold y's tree
   (`test_pearl_c_hash_wide.py:152` is the form to copy), the job files under
   `kmove-jobs/hash111/` with the launch form, and five traps (node A's Python 3.11; the lint
   counts the word "cap"; build tests skip without `VERITY_POUW_NVCC`; the device sources include
   `verity/definitions/kernels/sha512.cuh`; the guard refuses a ref passed through a shell
   variable).
Not covered by any test, KMOVE says: `DeviceTrees` on sm_90's runner past 128 segments.
Also: `main` moved `benchmarks/specs/` to `benchmarks/harness/specs/` today (#9189, a certified
move); #9375 takes it at its next merge.

## Thu 12 Nov 20:08Z: lean's #9375 is MARKED at `bafae2eaec2e5d6e04569b75c31d422b8c015d8c` (20:05Z) and does not move again until it is on `main`

lean (LEAN-249, 20:06Z): "From now until it is on `main` nothing is pushed to it; a fix is a new
pull request." So the head of `cursor/lean-remote-merge-onmain-fc8d` to build on, search and test
at is `bafae2eaec2e`: merge it once now and say that sha in what you report. Its candidate's check
is the first Lean build and the first suites' verdict on that tree, so it may still come back red;
if lean says so on LEAN-249 I tell you here. When it is on `main`, your base is `main`.

## Thu 12 Nov 21:14Z: your 21:12Z return is read: the three conditions off a card, in four hours. Next: the readers get the change today, and its first run on a card is FRIDAY on node A, not Tuesday

Read: your return whole; the branch at `c5840ecef3` (it holds the marked head `bafae2eaec2e` and
does not reach the landing branch; 7 files, +377, -41 against the marked head, by me); the two
adapter files' diff whole (`served_y.py`: `tree_launches`, `chunk_rows`, `pipe`; `ends.py`:
`_retained`'s `y_tree`); `verify_run.py:443-444` for your note on the one-pass key. **The
control you built is the right one: with `y=` alone the dump would rebuild y's tree from a
poisoned y and the no-write run would pass. And you covered the gap KMOVE named
(`DeviceTrees` past 128 segments).**

**1. The readers, today.** The change touches two lanes' files: `benchmarks/pouw` and
`benchmarks/harness` are compute-accounting's, and `adapters/vllm/.../pearl_c/served_y.py` and
`ends.py` are vllm's served code.
- **Open the pull request now, unmarked, stacked on #9375** (lean, TOP-402: "Stack on it as on
  any open pull request"; `research queue open` with its base, per the team skill; if the queue
  cannot open a stacked one, say so and post the compare of `bafae2eaec2e...c5840ecef3` on
  CMP-380 instead, and open it the hour #9375 is on `main`). `Lane kernels`, `Refs HASH-124`,
  `Goal kernels/row-digest-takes-head-rows`.
- **Its body says, in its first lines:** what changes in the served adapter and that the served
  call's launches and bytes are the same (show it: the recording stand-in's launch list of a
  served call before and after, event for event; `y_tree` defaults to None); what was not run
  (anything on a card); the three runs.
- One worker comment on CMP-380 (compute-accounting and benchmarks read there) and one on
  HASH-124, each at most 100 words, naming the pull request and what each lane should read; tell
  me and I ring both and vllm.
- **Your note on `verify_run.py`'s one-pass key** (`replay_key` over the transcript's root, the
  records' root and the salt; y's root not in it): put it in the body under "Seen and not
  changed", as you gave it to me, with the line. It is compute-accounting's to weigh.

**2. Friday, node A: the arm's first run on a card, untimed.** I booked, PROVISIONAL: **card 3,
cores 34-43, Fri 13 Nov 17:00Z to 18:30Z** (`~/.research/notes/calendars/node-a.md`). It brings
the part of condition 2 that needs a card forward by four days:
- the arm at the four linears' shapes and at (16, 128,256, 2,048): its gate with y's tree in
  it, every output filled with 0xA5 first, the no-write control refused; `widen_a` held to its
  numpy twin on the card's bytes; the device branch of `_retained(y_tree=)`; then `verify_run.py`
  on the run's own transcript with y's rows opened, and the poisoned-y control rejected at its y
  openings. Untimed: no figure of it is a result.
- **The cubin:** the prebuilt sm_90a one with the wide kernels (sha256 `5a6d7461f5f01bb2…`, from
  `r20261112-084338-5963`; KMOVE rebuilt the same bytes at `main` today), checked by sha256
  before any launch; nothing built on the node. The pin in `overhead_arm.json` does not move:
  the screen passes the cubin by path, and says so.
- **How a job runs there:** KMOVE's card run of today is the model (`r20261112-113022-98b1`; its
  launcher and traps under `~/.research/notes/lanes/kernels/kmove-jobs/hash111/` and at the top
  of `20261110T0806Z-report-kmove.md`: node A's Python 3.11 cannot import `verity`, so the
  record's step and `verify_run.py` run afterwards on a CPU node from the run's published
  files; a tree the node has not seen takes 5.5 minutes to ship, so send early and sleep until
  the row). One lease of at most 30 minutes; a start check that exits if card 3 holds a lease or
  a compute process; kill only by PID.
- **Tonight:** the job written beside your note, dry-run on the stand-ins until every line
  holds, its files' sha256 and its one launch line in your return. **I say by Fri 12:00Z whether
  the row stands, from that; the go is at about 16:50Z on Friday.** Nothing on any card before.

**3. Tuesday's timed rows** (the 2a screen, the development row, the row of record on
`vy-dc-h200-3`): after Friday's run. Say in your return what Friday's run will already have
shown of them and what it leaves.

Return when the pull request is open and Friday's job is dry-run, by Fri 13 Nov 06:00Z.

## Thu 12 Nov 22:17Z: your 22:14Z return is read; the three lanes are rung. Two things before the row stands: the job under node A's own Python stack (on cpu-8, tonight), and an ACCEPT that drew tiles

Read by me: your return and the note's "Return (Thu 22:14Z)"; #9489 (open at `c5840ecef31f`, base
lean's branch, +377, -41 in 7; its body); the dry run `r20261112-215253-cefb` (rc 0, tree
`c5840ecef31f`, its verdict's last lines: `AFTER ok`, `DRY ok`); `arm_card.py`'s `expect`,
`_verify`, `poisoned_y` and `verify_part`; `card.sh` lines 40 to 74. **Running `run.py`'s
identity check yourself, once you saw `Kernel.load` skips it, is the kind of catch this run is
for.** Rung at 22:17Z: compute-accounting and benchmarks on CMP-380, vllm on HASH-124; their
word by Sat 02:00Z; I mark once #9375 is on `main` and Friday's run has held.

**1. Node A's Python is checkable tonight without node A.** `card.sh` takes its Python from
`benchmarks/harness/env.sh`'s `VENV`, which on node A is `/workspace/venv312` (KQUANT's card run
of today, `r20261112-195025-eb35`, its `job.log` line 1; KMOVE's failure was the node's bare
Python 3.11, not this venv). **`vy-nebius-cpu-8` holds a copy of that stack at
`/workspace/cp/venv312`** (python 3.12.14, torch 2.13.0+cu129, numpy 2.3.5, vLLM
0.28.1rc1.dev472; KGATE ran its no-card preflight there tonight, `r20261112-212951-ec22`; how it
pointed its job at that venv is in `lanes/kernels/20261112T2116Z-report-kgate.md` and its
`kgate.sh`).
- Run there, with `VENV=/workspace/cp/venv312` and no other Python on the path: `card.sh`'s own
  preflight as the card runs it (`CUDA_VISIBLE_DEVICES=` and all), then the stand-in dry at the
  nine shapes under that Python. One `research run --on vy-nebius-cpu-8`; if its pool is full,
  wait for it rather than take another node: the point is that venv.
- **If it imports and the dry holds, the import is no longer a guess**, only `env.sh`'s paths on
  node A are, and the row stands. **If it does not import: that is tonight's work**: say what is
  missing (`verity`, `verity_config`, `verity_vllm`, a dependency), and make the job carry it
  from the tree (`PYTHONPATH`, as the suites do) without installing anything on node A.
  Dry-run again; new `SHA256SUMS`.
- Nothing on node A itself tonight: no look, no probe.

**2. The honest ACCEPT must have drawn tiles.** On the stand-in every honest line reads
`clear_work's tiles (0 drawn, …`. On the card, `honest_accepted` passes on `rc == 0` and
`ACCEPT` alone. **Add to `verify_part`'s expectations on the card: the honest ACCEPT line's
drawn count is over 0** (parse it from the line, or from the verdict's JSON if it has the count;
the default `--draws` stays, as the harness's own record command has it), and record the count
a shape. An accept that replayed no sampled tile would say nothing about the GEMM. Say in your
return what the count will be at each shape, if `verify_run.py`'s default says.

**3. Friday, as you wrote it, with these words:**
- **Launch at 16:45Z, the window's first minute**, so the ship has 15 minutes.
- Operands made inside the lease: fine for an untimed run; leave it. The graph replay reported
  and not gated: fine; its result goes in your return.
- **Card 3 until about 16:22Z is KAUDIT's** (HASH-98's row 1, three chained timed runs; a
  rebuild adds 7 minutes). Your job's wait to 17:10Z covers an overrun; kill nothing.
- The go is mine at about 16:42Z, in this file. No STANDS in the calendar's row, no launch (your
  launcher already refuses).

Return when the run on cpu-8 has ended and `expect` has its count, by Fri 13 Nov 04:00Z, with
the new sha256 of each changed file and the one launch line.

## Thu 12 Nov 22:29Z: COMPUTE-ACCOUNTING SAID YES to #9489 (CMP-380, 22:27Z), with one blocker for Tuesday that is now yours: the wide cubin as an art, gated, and the pin moved. After tonight's two things; nothing here changes Friday

Read compute-accounting's comment on CMP-380 whole (the newest of its lead's): it read #9489 at
`c5840ecef3` file by file, reran your test file (62 passed) and says "yes, no change is needed
to merge it". **Your note on the one-pass key is closed: "class (c), not a gap", no pull
request**; its reasons are there. I answered at 22:29Z that the blocker is ours.

**Order: tonight's two things first (the section above; back by Fri 04:00Z). Then Friday's
run. The piece below is due Mon 16 Nov 18:00Z and may start whenever you have a waiting hour.**

**1. The blocker for Tue 17 Nov: `overhead_arm.json` pins a cubin without the wide kernels**
(`art:378f025e…`, sha256 `cdf0e443…`, 4 Nov), so the arm refuses gate_up at prepare and the
slot would be lost. compute-accounting's steps, which are the piece:
- a cubin built at a head that has `hash_wide.cuh`: **it exists, sha256 `5a6d7461f5f01bb2…`,
  8,037,440 bytes** (KMOVE's `r20261112-175053-2b44` on cpu-1: `build.sh` for sm_90a, both gates
  passing, a second build the same bytes; and `r20261112-084338-5963`'s, which Friday's job
  takes). Say which head `main` builds it from today, and build it once more from `main` if
  `main` has moved under its sources;
- **the run's SASS gate on it** (`verity_bench/overhead_records.py:201`, `cubin_why`: read what
  it asks of a cubin of record and meet each line);
- **put as an art** (the kind and the label the pinned one carries: `research data show
  art:378f025e…` is the model), its sha256 and bytes checked after the put;
- **a pull request that moves the pin** (`cubin.art`, `sha256`, `bytes`; nothing of `window`),
  with whatever test holds the pin, on #9489's branch or stacked on it, unmarked, `Lane
  kernels`, `Refs HASH-124`. **Friday's card run is its evidence that the wide kernels hold on
  a card**: say so in its body with the run's id, and what is still unshown (nothing timed; C's
  and L's points).
- Tell me when it is open; I ring compute-accounting, who reads it by Mon 18:00Z.

**2. The docstring** (`overhead_arm.py:14-16`, "as the served call does"): take
compute-accounting's sentence: the arm makes y's tree serially on the timed stream, where the
served call overlaps it on its retain stream, and widens by `widen_a` where the served call
uses torch's copy; the bytes are the same, the launches are not; so the figure with y is an
upper bound on y's cost against an overlapped served schedule. **One commit on #9489, AFTER
Friday's run** (the job is pinned to `c5840ecef3`; do not move the head before it), with the
suite it touches run again.

**3. For Tuesday's plan, from the same comment:** the first minutes of the slot on
`vy-dc-h200-3` are one untimed shape per linear (the gate, the dump, ACCEPT, the control's
REJECT) before anything is timed, whatever Friday showed on node A; and the γ of that retime
is the first figure without `WITHOUT_Y`, not comparable with earlier ones. Put both in your
plan for the slot.

## Thu 12 Nov 22:57Z: THE ROW STANDS (card 3, Fri 17:00Z to 18:30Z). Finding that the card node has no `lake` saved the run. Now, with eighteen idle hours: the wide cubin's art and the pin's pull request (the 22:35Z section's piece 1), tonight

Read by me: your return and the note's "Return (Thu 22:53Z)"; `r20261112-223410-5382` (cpu-8,
rc 0 at `c5840ecef31f`, its preflight line: Python 3.12.14 at `/workspace/cp/venv312`, the
Lean executable `9d344082…` answering coins, `DRY ok`); `SHA256SUMS` against the six files
(all OK, the sums you gave); `lean-exe.sh` and `card.sh` lines 68 to 73. **By me on the two
nodes: the executable needs glibc 2.17 and only libc's own libraries; node A has glibc 2.35,
`/workspace/venv312`, and no `lake`.** So your blocker was real, and the executable will start
there. Cutting PATH to find it is exactly what the dry run was for.

**The calendar's row says STANDS** (notes `71d64e227c`; the launcher's own check reads it). The
go is still mine, at about 16:42Z on Friday, in this file: I look then that KAUDIT's chain on
card 3 has ended. **If one of its timed steps is still running at 16:45Z, do not send the tree
beside it: wait for its end, inside your launcher's window (to 17:05Z); past that, tell me.**

**Tonight: piece 1 of the 22:35Z section, in full, without waiting for Friday.**
- the cubin built once more from `main`'s head today (say the head; it must be `5a6d7461…`
  again, or say what differs and stop), the run's SASS gate on it (`overhead_records.cubin_why`:
  each line it asks of a cubin of record), **put as an art**, sha256 and bytes checked after;
- the pin's move (`cubin.art`, `sha256`, `bytes`; nothing of `window`) with the test that holds
  it, on a branch of its own cut from #9489's head; the suites it touches;
- **the pull request opened unmarked, its base #9489's branch** (so that its diff is the pin
  alone), `Lane kernels`, `Refs HASH-124`. First lines: the pin moves to the cubin with the wide
  row digests so that gate_up's y tree can be made; **not marked before Friday's card run has
  held, whose id goes here then**; what is shown (the build twice the same bytes, both gates,
  the art) and what is not (anything on a card with it until Friday; nothing timed).
- **#9489's own head stays `c5840ecef3` until Friday's run has ended** (the launcher wants the
  pushed head). The docstring's sentence goes on after.
- Friday's job is not touched again: it takes the cubin by path, as dry-run.
- One worker comment on CMP-380 with the pull request and the art (at most 80 words); tell me,
  and I ring compute-accounting.

Return by Fri 13 Nov 06:00Z.

## Fri 13 Nov 06:53Z: a small piece BEFORE Friday's card run, and it touches nothing of that run: HASH-132, the H200 kernel's bytes across the weekend's move. Your `cubin-main.sh` is most of it. Return by 12:00Z

**Friday's card run stands as it is:** card 3, the go from me at about 16:42Z, launched from a
clean worktree at `c5840ecef3`, your job files as they are in the notes. Nothing below changes
a job file, #9489 or #9550. One thing to know for the go: compute-accounting said (HASH-131,
06:45Z) that the red team broke the H100 conjecture on the model (PRF-612) and that the mend is
on the verifier's side, so the kernel's sources should not change; if that changes, I say so
in the go.

**What happened tonight.** Daniel agreed the architecture page; its moves start now. Three of
them rewrite an include line inside our kernel sources, because `pearl_c/hash_h3.cuh:18`,
`pearl_c/sha512.cuh:9`, `served_zk/hm96_rows.cu:18` and `commit_gpu/commit_gpu.cuh:47`
include `../../../verity/definitions/kernels/{row_commit,sha512}.cuh` [read by me on `main`]:
- **the weekend's move (top, Sat 14 or Sun 15 Nov):** `verity/definitions/kernels/` goes to
  `verity/circuits/kernels/` (`tools/move/definitions_kept_map.toml` on top's branch
  `cursor/top-moves-c-b396` at `ec36d28b1dc5`; the map is not on `main`);
- Monday's move of the two headers to `verity/commitments/` (proofs with circuits; no PR yet);
- Wednesday's move of PoUW's package (compute-accounting; no branch yet).

By reading, an include path does not reach the binary (you built the cubin from two
directories and got the same bytes). **Nobody has built it across a move, and
compute-accounting's timed slot on Tue 17 Nov builds `pearl_c_sm90` from `main` and holds it
to the stored binary.** HASH-132 is that check for each of the three moves; this brief is the
first.

**Do, on a CPU node with CUDA 13.0's nvcc (where `cubin-main.sh` ran), outside an infra
window, in a worktree and a run of their own:**
1. **Before:** `origin/main` at a commit you name: `pearl_c_sm90`'s cubin by `build.sh`, its
   sha256. It must be `5a6d7461f5f01bb209dcd40f144eff9f1edf3b289561e9b97c3836ca1bb19571`
   (8,037,440 bytes, `art:20a1fa98…`). If it is not, stop and tell me: that is a finding
   about `main`, not about the move.
2. **After:** the same tree with the weekend's move applied, in a scratch copy inside the
   job. Read top's branch first: which maps the move runs and in what order (its body or
   TOP-398 if the branch does not say); if the branch's tree is already moved, build that
   tree. The same toolkit, the same flags, the same `build.sh`. **Its sha256 against 1's.**
3. **The diff of our sources across the move:** which lines of the four files (and of any
   other `.cu`, `.cuh`, `build.sh` or `fixture.py` of ours) the move rewrites. If the map
   leaves kernel sources alone, the includes point at a directory that is gone: then the
   build fails, and that is the finding.
4. **The other binaries whose sources it rewrites:** `pearl_c_sm120`'s cubin, `served_zk`'s
   row hashing, `commit_gpu`. For each: is it pinned or named by a run of record; built
   before and after with the same bytes, or not built and why.
5. The opt-in build tests and the machine-code check's tests on the moved tree, as your job
   runs them.
6. One comment on HASH-132 (at most 100 words) with the two sha256 and the verdict. **Tell
   me: I say it to top on HASH-131 before it marks the move** (promised by Sat 14 Nov 8 AM
   PST).

Your note: a section in your report note, a CHECKPOINT line each 30 minutes, synced by its
own path. Return by Fri 13 Nov 12:00Z. I resume you for Monday's move when its PR exists.

## Fri 13 Nov 07:10Z: the weekend's move is pulled into TONIGHT, with more in it. Finish the build you are on; then the same check on chain 2's staged head, which I give you at about 11:30Z. The answer is owed before 13:00Z

Daniel asked for the repository in its structure when he wakes. top (TOP-415, 07:05Z): every
path-only move is on `main` by Fri 15:00Z, in two chains marked by 13:00Z. **Chain 2 holds the
move you are checking (`verity/definitions/kernels` to `verity/circuits/kernels`) and more:
each protocol goes under a group directory, so `verity/protocols/pouw` becomes
`verity/protocols/accounting/pouw`** (our `build.sh`, `fixture.py` and `run.py` name it), and
the Lake packages' paths move. Its maps are on pushed branches by 10:00Z and the chain is
staged and rehearsed by 11:30Z; I asked top to post the staged head on TOP-415.

- **Now:** finish the before-and-after build on the first map alone, as briefed. It tells us
  whether an include line reaches the binary. Return with it as soon as it is done (not at
  12:00Z), and keep the job ready to run again.
- **At about 11:30Z I resume you with chain 2's staged head.** The same check on that tree:
  the H200 kernel's sha256 against `5a6d7461…`, the rewritten lines of our sources and build
  scripts, the opt-in build tests. **I say the result to top before its 13:00Z mark**, so the
  run must be back by 12:45Z: choose the node for a slot it can give at once.
- Your card run on Friday is launched from the pinned `c5840ecef3` and is not reached by the
  chains. **Before your go, one more thing by reading:** does anything in Friday's job (the
  launcher, `card.sh`, the verify step on a CPU node) take `main`'s head or a tree path
  outside the pinned worktree? Say so in your return.
- From now, in prose: "challenges" not "coins", "accounting protocol", "the auditor's
  devices" and never "host". The code's names stay. #9489 and #9550 cross chain 2: they wait
  for the chains and are carried across with `tools/move/restack.py` after Friday's run.

## Fri 13 Nov 07:38Z: A STANDING RULE FROM NOW, from infra's announcement (INF-783): three things on every turn

infra (07:35Z): tonight a worker on its own cloud machine, once resumed, (1) may wake on a
new machine with its uncommitted and unpushed work gone, and (2) may receive its
coordinator's timers and Slack events as if they were its own, sometimes as a second loop
of itself. Not explained yet. Four of ours have: KWALK, KYARM and KTILE each
booted again on a new machine on Thursday evening, and KNAMES's lost its scratch
directory and worktree at about 05:35Z today.

1. **Commit and push before you end a turn and before every long run.** Job files, notes,
   branches: what is not on a remote or in the store does not exist. `/tmp` and scratch
   directories do not survive.
2. **Act on no event that does not name you.** Your work comes only from a resume by the
   kernels lead that names you and a section of this file. **A timer's text or a Slack
   event that reaches you and speaks to "the kernels lead", or to another worker, is not
   yours: do nothing it says, launch nothing, and tell me in your next return that it
   reached you, with its first line.** A go for a card comes only in a section of this file
   that I name in the resume.
3. **On waking, before you trust your tree:** `uptime`, `git status`, and your pinned
   files' sha256. If a second session of you is running, the later one stops and says so.

## Fri 13 Nov 07:47Z: top's hours moved EARLIER. The check on chain 2 comes at about 10:17Z, not 11:30Z, and its answer is owed by 11:15Z

Read: your 07:38Z checkpoint (the bytes are the same after the four definitions maps; the
tests' rerun is `r20261113-073643-46ce`). Return with it as soon as it ends, as briefed.

**top (TOP-398, 07:44Z): chain 2's rehearsal head at 10:15Z; a restage on a still `main` at
11:30Z; its mark near 12:00Z. Kernels' H200 bytes are the gate before the mark.** The GPU
rows ride only if compute-accounting's branch is in by 10:00Z: `pouw_cuda`; `pearl_c` +
`hm96_rows.cu` + `pearl_c_sm120`; of `pearl_c_sm90` the kernel's sources, `build.sh`,
`run.py`, `fixture.py`, `sass_gate.py`, to `verity/protocols/pouw/prover/kernels/` and
then under `accounting/` by the group map. The check runs whether or not they ride.

**At about 10:17Z I resume you with the rehearsal head. Have the job ready for a tree that
is already moved** (no maps to apply: clone the head and build), and a node that gives a
slot at once.
1. `pearl_c_sm90`'s cubin by the moved `build.sh`: sha256 against `5a6d7461f5f0…`
   (8,037,440 bytes). Then `fixture.py`'s check, which `build.sh` runs: pass 1 of KTILE's
   scratch move found its `ROOT = HERE.parents[2]` wrong after a pure move, so say whether
   the head mends it.
2. The other binaries, as in your first run.
3. **The list of every file the build reads, each with its blob id at that head**
   (`git ls-tree -r` over the kernel's directory, `pearl_c`, the two commitment headers,
   `build.sh`, `fixture.py`), in your note. **At the restage (11:30Z) I compare that list
   with the restaged head's: the same blobs at the same paths means the same bytes, and no
   rebuild.** If one differs, you rebuild.
4. The opt-in build tests and the machine-code check's tests on the moved tree.
5. **Return by 11:15Z** with the sha256, the verdict, the list's path. One comment on
   HASH-132 after the restage's verdict, not before.

## Fri 13 Nov 08:34Z: your return is read (the four maps change no binary). For 10:17Z: the Pearl-C directories do NOT move tonight; the check still runs

By my call on TOP-415: `pearl_c`, `pearl_c_sm120` and `pearl_c_sm90`'s files stay in
`benchmarks/pouw/` tonight (the served loader takes `run.py`, `fixture.py`, the cubin and
`grouped.py` from one directory, and `fixture.py` cannot live under `verity/` as written).
`pouw_cuda` rides; `hm96_rows.cu` may. **So the rehearsal head rewrites lines in our
sources where they are** (the definitions maps' includes; the group map's paths in
`build.sh`, `fixture.py`, `run.py`; Flock's `sha512.cuh` under `hm96_rows.cu`).
`moved-head.sh` as you wrote it fits: it finds each source by name.

Two additions to the job:
- Print `verity.__file__` and the adapter's before the tests (the trap your first run hit
  is now posted to the other lanes; your job is the example of the guard).
- **`fixture.py`'s and `run.py`'s sha256 at the rehearsal head beside `main`'s**, and
  `kernel.py`'s `source_digest` of the pair before and after: the manifest's source digest
  changes at the chain with the same cubin, and vllm is told to expect it.

The resume comes with the head, its branch and the hour, which may slip (the rehearsal's
base is the commit ci names for the Lean batch). Return by one hour after the resume.

## Fri 13 Nov 08:56Z: the hour moved again: the rehearsal head is due 10:40Z to 11:10Z, and the cubin's sha256 is owed 35 minutes after it

top (TOP-398, 08:51Z): chain 1 is on `main` (`e4da505fe488`); chain 2's rehearsal head
comes 75 minutes after lean's whole head, so 10:40Z to 11:10Z; its base is a candidate
commit (`5b2c1e735b92`), not `main`, so the head is on a stager's branch that I name in the
resume. **Kernels' bytes are owed 35 minutes after the head.** Your dry run of
`moved-head.sh` on `main` took at most 17 minutes, so it fits if nothing waits:
- **The cubin's sha256 first:** the moment `pearl_c_sm90`'s build ends, write it to your
  note and sync it, before the other binaries and the tests. I read it there and post it.
- A node that gives a slot at once; if cpu-1 is busy, any CPU node with the same toolkit.
- The restage is now conditional (only if the stager's scratch merge with `main` finds
  something), so `build-reads.txt` matters as before: I compare it by git myself.
The served Definitions' map is out of chain 2; the kept and merge maps ride, and lake if
it certifies. The Pearl-C directories do not move (my section above).

## Fri 13 Nov 09:20Z: GO NOW, a first pass of `moved-head.sh` on top's trial tree, which has the group map in it. Return as soon as the cubin's sha256 and the tests are in

top (TOP-398, 09:19Z): chain 2's head is now expected 11:15Z to 11:45Z (the stager
rehearses its whole stack first), and **our bytes are owed 35 minutes after that head**.
Your job has never run on a tree with the group map in it. So we rehearse too, twice:

1. **Now: `origin/cursor/top-moves-chain2-trial-f2f4` at
   `dd0c5b48fe5ee7c47c6e80bb3306e074c88726fc`** (fetched by me: on chain 1's head; ten maps,
   the group map last; `verity/protocols/{accounting,proofs,policy,compose,interface}`;
   our kernel directories still in `benchmarks/pouw/`, with `hash_h3.cuh:18` including
   `verity/circuits/kernels/row_commit.cuh` and `hm96_rows.cu:17` including
   `verity/protocols/proofs/flock/prover/cuda/sha512.cuh`). top: PoUW's two rows are not
   in it, the served Definitions' map did run there (it is out of the real chain), the
   difftest map did not. **It answers one question: does the job run with the group map
   in, and does the H200 kernel build to `5a6d7461f5f0…` there.**
2. **At about 10:15Z: the stager's rehearsal tree** (the real step list on the paused
   `main` `53bb345aa7e9`, with PoUW's two rows; without lean's stages). I resume you with
   it when top posts it.
3. **11:15Z to 11:45Z: the real head.** The gate. The cubin's sha256 first, at once.

**Do now, on 1:** `moved-head.sh` as you wrote it, on a CPU node with the toolkit, from a
pushed ref (the branch above is on origin). In order: the cubin's sha256 (write it to your
note and sync at once), `fixture.py`'s check by the whole `build.sh`, the other four
binaries, `build-reads.txt`, `run.py`'s and `fixture.py`'s sha256 and `source_digest`
beside `main`'s, `verity.__file__` and the adapter's printed, the tests. **Say every fault
of the job itself that the new layout shows, and mend the job for the next pass.** Commit
and push the job's changes. Return as soon as it is done; I expect about half an hour.

## Fri 13 Nov 09:53Z: for after Friday's card run, when #9489 and #9550 are carried across chain 2: vllm's refactor batch touches `ends.py` and `served_y.py`

vllm (HASH-124, 09:52Z): its refactor's next batch changes `pearl_c/ends.py` (an alias,
six annotations, `cast(Path, cur.retain)` at the five reads in `PassEnds._retained`, the
function #9489 gives `y_tree=None`) and `pearl_c/served_y.py` (docstrings, an
annotation); typing and wording only; not before chain 2 is on `main`. Its own conditions:
its head merges with #9489's with no stop in either file, your two test files pass on the
merge, mypy finds nothing its shrunk list no longer allows. **I said go (HASH-124).**
When you carry #9489 with `tools/move/restack.py` after the card run: if the batch is on
`main`, merge it in, keep both, rerun the two test files and mypy on the result, and say
so on HASH-124. Not now.

## Fri 13 Nov 10:46Z: PASS 2, NOW, on a tree with the machine-code check MOVED: compute-research's `cursor/compute-move-chain2-scratch5-a526` at `37304917bcb8`. Return by 11:10Z whatever is in

Read: your return of 09:38Z (the trial tree: same bytes, the job runs, the digest step
mended). The stager's own rehearsal tree is not pushed yet, and the real head is due
11:15Z to 11:45Z. **One layout change your job has not met: compute-research's map moves
`benchmarks/pouw/harness/sass_gate.py` and its pins to `verity/protocols/pouw/prover/`.**
It rides chain 2, and `build.sh` runs both gates.

**The tree** [fetched and read by me]: `origin/cursor/compute-move-chain2-scratch5-a526`
at `37304917bcb8a326b0d8012bd97b7a2f19272ede`, on chain 1's head, twelve maps applied
(definitions kept, merge and lake; difftest; compose; hardware; `sass_gate_map` last),
**not the group map**. There: the gate is `verity/protocols/pouw/prover/sass_gate.py`
with `sass_pins.json` beside it; `pearl_c_sm90/sass_gate.py` loads it from
`parents[3] / "verity" / "protocols" / "pouw" / "prover"`; `build.sh:32` runs
`$ROOT/verity/protocols/pouw/prover/sass_gate.py`. Our kernel directories are in
`benchmarks/pouw/`.

**Do:** `moved-head.sh` on it, the cubin's sha256 first and synced at once. What this pass
is for:
1. Does the whole `build.sh` pass with the gate at its new place, both gates' verdicts
   as before (448 flagged, 0 offending)?
2. **Does your job still find the machine-code check's tests?** They may have moved with
   the gate. If the job names them by path, make it find them by name, as it finds the
   sources.
3. The mended digest step, inside a job for the first time: `source_digest` of both
   kernels.
4. The gate's `pins_sha256` on this tree, beside `main`'s `75e6a039…` and the trial's
   `c56dc407…`.
**Return by 11:10Z at the latest, with whatever is in: the real head must find you idle.**
If the run is still going at 11:08Z, return with its id and what it has printed. Commit
and push the job's changes before the run and before you end.

## Fri 13 Nov 11:38Z: GO, PASS 3: chain 2's after-maps tree, `origin/cursor/top-moves-chain2-42e2` at `a282b4223b0dd48b8780459ceefd1847040555df`. The cubin's sha256 first; return the moment the run ends

Read: your pass 2 (the gate moved: same bytes; twelve minutes; the job mended).

**The stager posted its tree after the last map (TOP-398, 11:32Z)** [fetched and read by
me]: lean's whole head, the kept stack, **all 17 maps in the real order with the group map
last**, every hand step. There: `verity/protocols/accounting/pouw/prover/sass_gate.py`,
`…/prover/kernels/pouw_cuda/`; our Pearl-C directories in `benchmarks/pouw/`. Not yet in
it: vllm's listing commit, the pins' regeneration, the fixtures, a repin. **The pull
request's head follows at 11:55Z to 12:25Z, and top's mark waits on our bytes.**

**This pass is the one whose result I post.** At the PR's head I compare
`build-reads.txt`'s blobs with that head's by git; you rerun only if one differs.

**Do:** `moved-head.sh` on `a282b4223b0d`, from the pushed ref, on a node that gives a
slot at once.
1. `pearl_c_sm90.cubin`'s sha256 against `5a6d7461f5f0…`: **to your note and synced the
   moment it is printed.**
2. The whole `build.sh` (both gates' verdicts, `fixture.py`'s check), the other four
   binaries, the tests with `__file__` printed.
3. **`build-reads.txt` with every file's blob id, saved in the notes as
   `build-reads-chain2-a282b4223b.txt` and synced: I read it from there.**
4. Both kernels' `source_digest` and the gate's `pins_sha256` on this tree, beside
   `main`'s (`053b8e50fe8a`).
Return the moment the run ends, with the run's id, the sha256, the verdicts and the
list's path. No HASH-132 comment yet: after my post on TOP-398. Commit and push before
the run and before you end.

## Fri 13 Nov 16:25Z: GO, early. Card 3 is free now; your launch for 17:00Z as rehearsed, from the clean worktree at `c5840ecef3`

This section is your go for the card run (HASH-124); my resume names it. The 07:38Z rule
holds. I give it sixteen minutes before the hour I said, so that the boot is not rushed.

**Card 3 now** [KAUDIT's return, and my `gpu-lease status` at 16:24Z]: free since 16:21Z;
KAUDIT's three runs of row 1 are done and labelled. **The calendar** (notes `c40bb63098`):
card 3 with cores 34-43 is kernels' to 18:30Z, yours from 17:00Z; memory-accounting has a
fallback row on it from 18:30Z. Card 6 is KWALK's (timing since 16:15Z), card 2 is
memory-accounting's from 16:35Z.

**Do, as the 22:57Z section of Thursday and your job's own steps have it:**
1. On your new machine: the job files from the notes (`kyarm-jobs/card/`),
   `sha256sum -c SHA256SUMS`, the clean worktree at `c5840ecef3` (#9489's pushed head),
   the stored cubin `5a6d7461f5f0…` by its sha256.
2. **Before the launch, one thing I asked for this morning: your dry launch prints, on
   node A, where `verity` and the adapter are imported from** (`verity.__file__`,
   `verity_vllm.__file__`), since node A's `/workspace/venv312` is not read and the
   clone's packages come first on `PYTHONPATH` (`env.sh:27`). They must be the shipped
   clone's. If they are not, stop and tell me.
3. The launch as rehearsed: your line, your variables, nothing changed. One lease of at
   most 30 minutes on card 3, inside the launcher's window. Untimed.
4. **Chain 2 is on `main` since 15:30Z** (`4ae17f7eafd7`). Your reading says the job
   reads nothing of `main`'s head. If the launch refuses, or a `research` call of the
   verify step on the CPU node fails, tell me at once with its line; do not relaunch on
   the card without my word.
5. **After the card:** `verify_run.py` on the run's own transcript on a CPU node; then
   the verdict: the gate with y's tree in it at the four linears' shapes and the head's,
   the 0xA5 fill and the no-write control, `widen_a` against its twin, the reference
   verifier's ACCEPT; one worker comment on HASH-124; tell me, and I ring
   compute-accounting on CMP-380.
6. Then, not before: #9489 and #9550 across chain 2 with `tools/move/restack.py`, with
   vllm's typing batch merged if it is on `main` (the 09:54Z section).
CHECKPOINT each 15 minutes while the run is live; commit and push before the launch and
before you end. Return with the verdict, or at once on a refusal.

## Fri 13 Nov 19:23Z: for the carry of #9489 and #9550 (read with the 09:53Z section): the Pearl-C split, #9799, is open and lands first

- **#9799** (`cursor/kernels-pearlc-split-9a4b` at `95eec6302d30`, KTILE's, HASH-133) is approved by compute-accounting
  as owner and is marked once vllm has read it. It is on `main` by Mon 16 Nov 12 PM PST or after Tuesday's run.
- **It stops against #9489 and #9550 in one file** [my `git merge-tree` of the three heads, 19:25Z]:
  `benchmarks/pouw/tests/test_pearl_c_sm90_overhead_arm.py`. The split renames
  `test_the_development_build_is_build_sh_s_cubin_and_gates_alone` and drops `CUBIN_ONLY` from `build_dev`
  (`overhead_arm.py`, which merges by itself). `build.sh` now always ends at its gates; a ship is `ship.sh`.
  It also changes `Kernel` (no `fixture` field; `Kernel(run, dev, cubin_sha256)`) and `call_keys` (no fixture
  argument), which your tests may construct.
- **So:** carry #9489 across chain 2 with `restack.py` as planned, then merge the split's branch head into it and
  mend that one test, so that your head is `main` with the split and the arm. Say in the body that it lands after
  #9799. Tuesday's timed rows need both on `main` by Mon 16 Nov 12 PM PST.
- **Tuesday's job is ours** (compute-accounting, HASH-133, 19:21Z: nothing of theirs runs `build.sh` that day).
  `CUBIN_ONLY=1` in your kit is harmless after the split, and a ship comes from `ship.sh`.
- **A rule from infra, in force from Sunday (INF-798):** a push is refused if it adds a commit whose author or
  committer is not your checkout's own configured identity. Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`. A job
  on a node that commits (`restack.py`) names your identity ahead of its one command:
  `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)"
  GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
  Before each push: `git log --format='%an <%ae> | %cn <%ce>' origin/main..HEAD | sort -u`, and say what it printed.
- A PR body carries a line `Goal <id>` naming a live goal (`kernels/overhead-counts-output-commitment` for #9489),
  or the mark warns.
