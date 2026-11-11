# Inbox: KP2 (HASH-106)

## Tue 10 Nov 16:06Z: both of your readings are confirmed. What happens at 20:00Z, and what does not

- **Confirmed:** the text lines of `pous.py` name the store's `Makefile`, which is what builds the library; and the
  `BENCHMARKS_IMPORTERS` entry for `pous.py` stays, with a reason that names the store alone.
- **#8525 is being closed for #8662** (asked of top on TOP-394; said on HASH-99). Push nothing to #8525's branch.
- **memory-accounting has your paragraph** (relayed on HASH-106 with their handle). If they ask for a change, it is a
  new commit on your branch: #8662 is unmarked, so that is allowed. Mark nothing.
- **At your 20:00Z wake:** read this file and `inbox/RULES.md` again first. Card 4, cores 12-23, at most an hour,
  only if KMOVE's last run on card 4 has ended (your start check says so; if the card is in use, wait and try again
  until 20:40Z, then say so on HASH-106 and end your turn). Then the verdict lines on #8662's body and HASH-106.
- **The kernels lock:** today's `steady.sh` releases it after the start check and the lease's grant (its line 104);
  only the old `launch.sh` wrapped a whole run in it. Yours taking it around the check alone is right.
- After the smoke: the whole `verity-vllm` suite at your head in one named run, and its verdict on HASH-106.

## Tue 10 Nov 16:22Z: memory-accounting read #8662 and said yes to `load.py` as it is at 475737d29d (HASH-106, 16:21Z)

Both choices are settled: the stream's names stay in the class (the store's `store.py` already holds the same table;
making the two one is theirs, after it lands), and no test of theirs is needed before it lands. Their map of the P2
store (Wed 11 Nov 6 PM PST at the earliest) takes `load.py` with `gpu.py`: whichever lands second carries the other.
"Mark it when the layout is on `main` and your H200 run is read."

So what is left, in order, from your 20:00Z wake:
1. the H200 run at old `main` and at your head (card 4, cores 12-23), verdict lines on #8662 and HASH-106;
2. the layout will be on `main` by then (top posts the base commit and the two commands on TOP-394, about 19:15Z):
   `git merge BASE` into your branch, then `tools/move/restack.py`; say the new head on HASH-106;
3. the whole `verity-vllm` suite, `p2_stage`, `benchmarks/pous` and the boundary tests at the new head, one named run;
4. then stop and say so: **the lead marks, not you**, after reading the H200 run's lines. Change nothing in
   `load.py` that memory-accounting did not read without saying so on HASH-106.

## Tue 10 Nov 17:02Z: #8662 is marked only after vllm's delete (VLLM-290) is on `main`; your 20:00Z steps stand

- **vllm read #8662 at 475737d29d: yes, no change asked** (HASH-99, 16:46Z; its worker ran `git grep` at your head and
  at `main` and read the diff; it ran no suite).
- **vllm's delete of the in-process remote session** (VLLM-290; due on `main` Wed 11 Nov 20:00Z) meets #8662 in seven
  files [vllm's `git merge-tree` of its `2cb82d67c2` with your head, not run by the lead]: deleted there and edited
  by you, `engine/int8_credited_env.py`, `remote_bandwidth/in_place.py`, `tests/engine/int8_store_gpu.py`; edited by
  both, `engine/build.py`, `remote_bandwidth/remote.py`, `tests/test_boundaries.py`, `public_boundary.toml`.
- **The lead agreed the delete lands first** (HASH-99, 17:01Z): with it on `main`, the by-name load of
  `int8_credited_env` in `build.credited_read`, its test and its two boundary rows have nothing left to load.
- **Tonight is unchanged:** steps 1 to 3 of the 16:22Z section (the H200 run at old `main` and at your head, the
  restack over the layout, the suites). Do not merge vllm's branch and push nothing to it.
- **Step 4 changes.** After step 3, say the head and the verdicts on HASH-106 and end your turn. The lead resumes you
  when the delete is on `main`: merge `main`, drop what the delete leaves nothing for, rerun step 3's suites, and say
  on HASH-106 which files the H200 run loaded that this merge changed. If none, tonight's run stands; if any, the
  lead books a second hour on a card. Then the lead marks.
- **If the delete is not on `main` by Thu 12 Nov 02:00Z,** the lead marks #8662 as it stands at step 3's head
  (rule 5's date for the staging decode is Thu 12 Nov), and vllm's delete takes the hunks out.

## Tue 10 Nov 17:50Z: the layout does not land today, so there is no restack; one more file for your later merge

- top, TOP-394 17:39Z: nobody waits for the layout (`inbox/RULES.md`, "The queue, today", rewritten at 17:50Z).
  **Step 2 of the 16:22Z section is void:** no `git merge BASE`, no `restack.py`. Your branch stays on `main`'s
  paths. Tonight at 20:00Z: the H200 run at old `main` and at your head, then step 3's suites at your head, the
  verdicts on HASH-106, and end your turn. Everything in the 17:02Z section stands (vllm's delete lands first).
- **#8669 is marked** (the READMEs' Interface sections). Its table in `benchmarks/torch_gpu/README.md` says
  `integrations/vllm/verity_vllm/protocols/pous.py` loads `p2_stage/stage.py` and `p2_stage/layout.py`, which your
  change makes false. When you merge `main` after vllm's delete, if #8669 is on it, correct those rows in the same
  commit (what still loads `p2_stage`, and the new module's home), and say so on HASH-106.

## Tue 10 Nov 22:32Z: your suites run is read; you wait for vllm's delete, and the lead resumes you

r20261110-201110-336d passed at 475737d29d (`verity-vllm` 14,478, `benchmarks/pous` 218, `examples/h200` 169,
`p2_stage` 66, the repository's boundary tests 56; read by the lead, said on HASH-106). With your H200 run at both
heads, nothing is left at this head. HASH-106 is In Review. When vllm's delete (VLLM-290) is on `main` the lead
resumes you for the merge, as the 17:02Z and 17:50Z sections say. If the agent watch starts you fresh before that,
read this file first and do nothing but confirm you are idle.

## Wed 11 Nov 00:25Z: for your resume, two things the move and Daniel's word on `archive/` change in #8662

Nothing to do now. When the lead resumes you (vllm's delete on `main`, and the move, which lands first):
- **`examples/h200/pyproject.toml`:** the move puts the demo in `archive/`, and Daniel (00:10Z) has `archive/`
  deleted by top's worker. Your hunk there drops; say so on HASH-106.
- **`benchmarks/torch_gpu/p2_stage/README.md:20`** names `archive/examples/h200`'s `P2Store`. KSLICE's pull request
  (HASH-104) rewrites such lines in kernels' other files and leaves this file to you: rewrite that line in your
  merge commit so it names no path under `archive/`.

## Wed 11 Nov 02:27Z: a second issue for you, HASH-113: two measurements of PoUB's pass, on Thursday

#8662 still waits for vllm's delete and the move, as above. Meanwhile memory-accounting asked kernels (HASH-113,
02:24Z; read it) for two measurements on `vy-nebius-2`, the eight-card node, and the lead said yes for you. The
goal is `kernels/poub-pass-measured`. Read `inbox/RULES.md` ("Agents, today", "Runs", and the correction about
the lead's section times) first.

**What is asked. Measurements only: no kernel is built and no file of memory-accounting's is changed.**
1. **Where a pass's extra time goes.** A pass (`benchmarks/remote_session/session.cu`, `rs_pass`) reads the
   store and computes a receipt for every 4 KiB block; a bare read is the same read with no receipts. A pass took
   1.0607 × the bare read on 15 Oct and 1.0553 × on 23 Oct (memory-accounting's worker's reading of
   `r20261015-203743-d101` and `r20261023-123709-339d`; the lead checked neither). Wanted: both under the same
   profiler, the pass's extra time split by kernel and by wait, with the two ratios measured again beside it.
2. **The host segment inside a pass**, for each pair of `--host-ctas` and `--host-consumers`
   (`benchmarks/remote_session/device.py:1418-1419`; defaults 16 and 8): its rate a card and for eight, with the
   bare host read and the pinned copy over PCIe measured in the same session (the records have 211.4, 337.3 and
   447.9 GB/s).

**Tonight, no GPU, then end your turn:**
- read `session.cu`, `device.py`, `node.py` and the two records' command lines; find how a bare read and a
  pass were timed (`timing.stated`);
- **the grid of the two arguments and the profiling method, posted on HASH-113** (due Thu 12 Nov 16:00Z; tonight
  if you can), with how long each takes on the node, so memory-accounting can place them in its window;
- the job scripts, beside your note, written to `research run`'s rules (the writing-runs skill), stub-tested with
  no card; the checks each timed number needs here (every card clear at start and end, the lease, no throttling,
  what ran beside it, the commit);
- ask memory-accounting on HASH-113, in one comment, for whatever only it knows (the commit to measure, the store
  on the node, the two runs' exact arguments if the records lack them).

**Thursday:** memory-accounting holds `vy-nebius-2` Thu 12 Nov 18:00Z to 24:00Z for served sessions that take
under an hour in all, and gives you the rest. Run only in the hours it names on HASH-113, under gpu-lease, as the
kernel-engineering skill says for node 2. The lead resumes you for it. **The two tables on HASH-113 by Fri 13 Nov
18:00Z**, each number with what it rests on.

This does not change #8662: when the lead resumes you for the merge (vllm's delete on `main`, after the move), that
comes first.

## Wed 11 Nov 02:34Z: memory-accounting's answer (HASH-113, 02:33Z): 2.5 hours on Thursday, and the commit

- **Your hours on `vy-nebius-2`: Thu 12 Nov 21:00Z to 23:30Z**, shorter from the start if its three served
  sessions run over; it confirms at Thu 20:30Z. The node stops at 24:00Z. Nothing is served while you measure.
  **So size everything to 2.5 hours, and order it by what it decides:** the pass against the bare read first (it
  is about 3.3 points of the share), the two-argument sweep second (about 1 to 2 points), each able to stop early
  with what it has.
- **The commit: `5444b0d5fdbb34415e26221c83b4e4b48075a60b`** (on `main` since Wed 01:21Z, before the move). It
  names a later one by Thu 16:00Z if its sessions move. It has not compared `session.cu` there with October's:
  do that yourself (`git diff` of the file between the two records' source commits and this one) and say on
  HASH-113 what changed, since the old ratios may not be this kernel's.
- **The two command lines:** in each record's `job.json` (`research inspect r20261015-203743-d101`,
  `research inspect r20261023-123709-339d`); its worker GAMMA06 posts them by Thu 02:00Z. Read the records
  yourself tonight rather than wait.
- **How a job gets on that node:** only through a slot (`tools/verity/check/slot.py`, as
  `benchmarks/remote_session/gate_run.sh` does); inside its window the queue is off the node; infra is told
  (INF-660). If your job is refused for the lane's name, say so on HASH-113 and its worker launches yours as
  written. Write the jobs so that either can launch them.

## Wed 11 Nov 03:01Z: memory-accounting adds two lines to the profile, and offers the 64-row pass (HASH-113, 03:00Z)

- **Two lines of their own in the pass's split:** the sketch's tensor-core time, and the receipt hash's time,
  each apart from the leaves and the tree. Why: Daniel's audit wants PoUB's memory line at 2^-100, which takes 64
  sketch rows where the session runs 16 (MEM-253): four times the tiles a block, and three compressions a
  receipt where there is one, by its worker's reading of `reader.cu`. So those two lines predict a 64-row pass's
  cost. Plan the profile so they can be read off it (kernel names, NVTX ranges or a split run), and say in your
  method how.
- **Its offer:** memory-accounting writes and times the 64-row pass itself (MEM-258, its worker starts Wed). The
  lead's answer, **to put in your grid comment on HASH-113** (due Thu 12 Nov 16:00Z): memory-accounting writes
  it; kernels reviews it against the kernel-engineering skill's bar (the SASS gate, 0xA5 poisoning and a no-write
  control, the reference verifier on the run's own transcript) once its branch builds, and asks it to name the
  branch on HASH-113. The review is yours, after Thursday's measurements; it is not a reason to change them.

## Wed 11 Nov 03:16Z: the review of the 64-row kernel is settled, with a date

memory-accounting (HASH-113, 03:15Z) writes the 64-row kernel itself, starting now (MEM-258, its worker ROWS64); its
branch and one-card check are due on MEM-258 by Thu 04:00Z (Wed 8 PM PST). If they are there by then, **you post
what you find on MEM-258 by Thu 12 Nov 20:00Z**, before your own hours on the node (the lead said so on HASH-113 at 03:16Z; your grid comment need
not repeat it). Read it against the kernel-engineering skill's bar: what holds the kernel to its Definition on
every word, the SASS gate, 0xA5 poisoning and a no-write control, the reference verifier on the run's own
transcript, and whether its one-card check would catch a wrong receipt. Findings as a list, each with its line;
no change to its branch. The lead resumes you for it on Thursday morning; the two measurements stay first if the
hours collide.

## Wed 11 Nov 03:46Z: read; memory-accounting's first answers to your five (HASH-113, 03:45Z), for your next resume

Your grid, method, jobs and questions are read; the lead rang memory-accounting for them (a worker's comment rings
no one). Nothing is asked of you until Thursday morning. Its answers so far:
- **The commit: `5444b0d5fdbb`, confirmed.** It must be one from before the move: node 2 keeps its old scripts
  this week (INF-683). So your jobs run that tree's old paths, not the moved ones.
- **1. The store: yes**, 700 GiB on the cards and 8 GiB on the host, synthetic.
- **4. The slot: `/workspace/research/locks/gpu/slots`.** Whether `--by kernels` is admitted it does not know:
  try it, and if refused its worker launches your line with `BY=memory-accounting`.
- **2, 3 and 5 by Wed 22:00Z** from its worker: the 15 Oct prover's run and pinning, the timing slice and whether
  you may enter it, the torch environment, `gpu-lease` and `nsys` (it reads `nsys` off the node Wed evening).
- **A warning on your default pinning:** `run.sh`'s whole-node line at that commit is `GPUS=8 TIMED=1 GIB=700
  ROLE=prover PROVER_SLICE=<the timing slice> PROVER_CPUS=128-191`, and on cores 48-95 beside other sessions' jobs
  a prover saw roots up to 35 ms late. Your default of `taskset` to 48-87 with no slice may not repeat October's
  pinning: make the job take the slice and `128-191` as arguments, and use them when the slice is named.
- **Thursday:** the lead resumes you in the morning for the review of the 64-row kernel (MEM-258; findings by
  20:00Z) and before 20:30Z for the runs (21:00Z to 23:30Z).

## Wed 11 Nov 08:22Z: a small pull request tonight, apart from HASH-113: `slot.py` says "not started" where a machine lists no check slots

Your Thursday work is unchanged. This is one small fix top asked kernels to send (the workflow thread, 08:17Z).

- **What happened:** at 07:35Z the lead sent four quick tiers with `research queue ready N --on vy-nebius-cpu-12
  --by kernels`. A CPU node has pools and no check slots: its `/workspace/research/locks/slots` exists and lists
  none. Each run ended after 0 s with `slot: /workspace/research/locks/slots lists no slots`, exit 1, and
  `ready` printed "the quick tier failed ... Push a fix, or run it again", though no test ran
  (`r20261111-073606-33fe`).
- **The change, and only this:** in `tools/verity/check/slot.py` on `origin/main`, the one exit at line 896,
  `sys.exit(f"slot: {SLOTS} lists no slots")`. It becomes a "not started": a message on stderr and `return
  NOT_STARTED` (75), as the branch a few lines above it does for a `CHECK_SLOTS` file the machine lacks. The
  message says how to fix it: the file lists no check slots so nothing ran, the machine's pools by name
  (`lane_pools`, as the `only=trains` exit below names them) with `research run --on MACHINE --pool NAME`, and
  that `research queue ready N --on auto` places a quick tier itself. `queue.py` already reads exit 75 as "never
  ran" (`SLOT_NOT_STARTED`), so it needs no change.
- **Keep to that one exit** (top): its #8835 (head `b27d46450b4a`, a land verdict, waiting) changes `slot.py`
  and `tools/verity/check/tests/test_slot.py` too. Touch no other line of `slot.py`; not `check.py:590` or
  `test_placement.py`, which say the same words about another path; not `queue.py`, not `queue status`'s hint.
  Merge #8835's head onto your branch locally (`git merge --no-commit`, then abort) and say whether it conflicts;
  if it does, place your lines and your test so that it does not.
- **A test** in `tools/verity/check/tests/test_slot.py`: a slots file that lists none gives exit 75, not 1, and a
  message naming the pools beside it; no clock, no sleep, no subprocess of the real machine. If an existing test
  holds the old exit, change it and say so.
- **Before the pull request:** the suite `verity-check` at your head, as a run (`research run --on auto --cwd
  source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py
  verity-check --quick'`), rc 0 read from its record. Nothing runs on your VM. Then read what `ready` prints for
  a tier that exits 75 (`queue.py`, `NotStarted`, `READY_NOT_STARTED`): if that text would mislead here (it
  says a slot was waited for), do not change it; tell the lead.
- **The pull request:** branch `cursor/kernels-slot-empty-<hex>` off `origin/main`; `research queue open --head
  BRANCH --sha SHA --title T --body-file F --issue CI-489 --by kernels` (top: ci is told on CI-489 when it is
  marked; it needs no owner's review). The body opens with `Lane kernels`, says what changed and what you ran in
  under 150 words. No run id, issue id, date or name in code, comments or the test. **Do not mark it**: the lead
  reads the diff and the run and marks.
- **Due:** the pull request open by 09:45Z. Report: its number and head, the diff's lines, the run id and
  verdict, #8835's merge result, and anything you chose not to do.

## Wed 11 Nov 08:30Z: for Thursday's report on HASH-113: memory-accounting asks whether kernels takes the two changes

Not for tonight. memory-accounting (HASH-113, 08:29Z) wants one line from kernels by **Fri 13 Nov 20:00Z**: does
kernels take the two changes your tables size, both, one or neither, with the date a first change could be in
review. Its default is to take both itself. The two changes: a pass about 1.02 times as long as a bare read
(October's records say 1.055 to 1.061), and the host's memory read inside a pass at 337 GB/s or more (they say
211). By its reading both together take the served store from about 0.147 to about 0.10, its goal's new target.
- **So your report on the two tables ends with a section for the lead**, for each change: whether the tables show
  it is there to make (or the extra time is in the hash, or the host read is the memory's limit); what it would
  touch (`benchmarks/remote_session/session.cu`, the host's read path); its size from your numbers, measured or
  inferred; and what you would need to have a first change in review, with a date.
- The lead answers memory-accounting from that section. Nothing in Thursday's grid or hours changes.
