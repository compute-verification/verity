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
