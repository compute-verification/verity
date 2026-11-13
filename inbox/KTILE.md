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

## Thu 12 Nov 21:57Z: both pull requests are read; #9506 is MARKED. Next: HASH-127's other two items, a verdict each (`attn_bf16`; `s8_linear`'s row-tables build)

Read by me: your return and both worker comments; **#9506's own diff over #9258's head whole
(two files, +44, -14: `ctas == -1` launches no reading grid; the gate's `consumer_tiles`;
`GpuBackend` refuses a negative `leaf_ctas`)**; your two runs on cpu-3 (both exit 0 at tree
`bc1fd81bb85c`; the SASS gate's last lines; "2 suites: 2 passed, 0 failed"); #9483's diff whole.
The patch-id check and going to cpu-3 when cpu-1 was full were right.
- **#9506: marked ready at `bc1fd81bb85c` at 21:55:33Z, stacked on #9258** (the queue lands a
  stack's base first). The preflight collected four suites; you ran two. I checked that the
  other two (`tools/verity/graph`, `poub/device/live_verifier`) name none of the changed
  symbols; the candidate's check runs them. **If #9258's head moves I resume you to cut it
  again.**
- **#9483: not marked yet.** I rang memory-accounting on HASH-127 (21:58Z): its word by Fri
  18:00Z, else I mark. HASH-127 now has its goal label.

**Your next piece (HASH-127, the other two items; due Fri 20 Nov; no card, no GPU node):
a verdict each, with its evidence: KEPT with what loads it, or DELETED in a pull request of
its own.**

**A. `benchmarks/torch_gpu/attn_bf16/`** (11 files on `main` `5f0c008aa040`).
**B. The row-tables build in `benchmarks/torch_gpu/s8_linear/build.sh`** (the `row_table/` step:
the header's lines on it, the `if [ -x "$PY" ]` block, what `run.sh` seeds from it, and what the
step alone makes `ship_stage` and `ship_source` take of `adapters/vllm/verity_vllm`). It was
built for `s8_linear/commit.py`, which #9260 deleted.

For each, on `main`:
1. **Every loader in the tree**, each with its file and line: imports by name; loads by path
   (`sys.path`, `runpy`, `importlib`, a file opened or a cubin loaded by name); shell steps
   (`run.sh`, `build.sh`, `ship.sh`); tests and their `[tool.verity.tests] inputs`; sidecars;
   the lists that name it (`BENCHMARKS_IMPORTERS`, `tools/verity/check/public_boundary.toml`,
   the READMEs' Interface sections). `benchmarks/pouw/gemm_rate` is said to time `s8_linear`
   and `attn_bf16` (KSLICE's `20261110T1530Z-report-kslice-leftovers.md`, line 105): say exactly
   which files, functions and kernels it reaches, and which files of A no loader reaches.
   For B: does anything `run.sh` still runs (`bench.py`, `fuse.py`, `tune.py`, `lt.py`,
   `run.py`) load the commitment kernels (`row_tables_device`) or read the seeded
   `~/.cache/verity/row_table`?
2. **The store's runs of it since its last change** (the method and table shape of KSLICE's
   note, section "last change, last run, on its current text?").
3. **The verdict**, one of:
   - **KEPT**: what loads it, by name. Nothing changes in the tree, unless a README's Interface
     section says otherwise than you found (then that line, in a pull request).
   - **DELETED**, whole or the part no loader reaches: one pull request per item, on `main`,
     unmarked, `Lane kernels`, `Refs HASH-127`. It also rewrites every line that names what goes
     (READMEs, docstrings, the lists above; a list only shrinks). The suites the change touches,
     on a CPU node. **For B**, `build.sh` run before and after on a CPU node with CUDA 13 (cpu-1
     or cpu-3): `s8_wgmma.cubin` the same bytes, the SASS gate passing, and the manifests
     differing by the `row_table` entries alone.
   - **If the only loader is another lane's code: delete nothing; name the lane and the line**,
     and I ask it.
4. One worker comment on HASH-127 with both verdicts (at most 150 words; the tables in your
   note), and tell me what to ring.

Nothing of this is timed, and nothing runs on node A or `vy-nebius-2`.

Return by Fri 13 Nov 09:00Z with the two verdicts posted and any pull request opened.

## Thu 12 Nov 22:48Z: both verdicts read; #9525 is MARKED. Next: A's pull request, opened unmarked now, so that benchmarks reads a diff and not a question

Read by me: your return and the note whole; #9525's diff whole (5 files, +13, -28);
`r20261112-220554-d7cb` (exit 0 at tree `10c3daffdf23`; `manifest.diff`: `ptxas.txt` and the three
`row_table` entries and nothing else; "4 suites: 4 passed, 0 failed"). By me besides: #8671 and
its crossed branch touch nothing under `torch_gpu`, KGATE's job does not use `s8_linear`'s ship,
and `run.sh:31` was the ship's `row_table`'s only reader on `main`. **Saying that the manifests
also differ by ptxas's compile-time lines, where my brief said "the `row_table` entries alone",
is how a return should correct its brief.**
- **#9525: marked ready at `10c3daffdf23` at 22:48:02Z** (the preflight collected the four
  suites you ran, and no other).

**A: I take your proposal: `attn_bf16/` goes whole.** It is kernels' own, nothing serves with
it, nothing has run since 28 Oct, and its one reader was deleted as dead on 11 Nov. What is not
ours is benchmarks' test, so (the friction skill: ask the owner and prepare the change
meanwhile) **the pull request is opened now, unmarked, and is not marked without benchmarks'
yes on its three files.**

The pull request, on `main`, `Lane kernels`, `Refs HASH-127`:
1. **Delete `benchmarks/torch_gpu/attn_bf16/`** (11 files).
2. **benchmarks' three files, the smallest change that leaves the shared gate's test whole:**
   `benchmarks/pouw/tests/test_mma_gate.py` (its `attn_bf16` case: `GATES`, `REAL_ARCH`,
   `attn_bf16()`), `benchmarks/pouw/pyproject.toml:165` (the declared input),
   `tests/fixtures/mma_gate/verdicts.json` (its pinned verdicts). **Say in the body, for
   benchmarks, exactly what the shared gate (`mma_gate`) is still held to after the case goes**
   (which gates and which architectures remain), and what, if anything, only this case
   covered (an atom form, an architecture, a rule of the gate). If it was the only case of
   something, say so in the body's first lines: that is what benchmarks decides on.
3. **Every line that names it** (your list): `benchmarks/README.md:144`,
   `benchmarks/torch_gpu/README.md:22,49,73`, `ship.sh:2`,
   `benchmarks/pouw/harness/sass_gate.py:1745-1747` (the comment, with its stale `units/`),
   `benchmarks/pouw/harness/README.md:19`, the two lint allowlists' entries, and
   `tools/move/layout_one_map.toml:781-782` if a test holds the map's rows to files that exist
   (if the map is a record of a finished move that no test reads against the tree, leave it
   and say so).
4. The suites the change touches, on a CPU node, at the pushed head.
5. **The body's first lines:** "Not marked without benchmarks' yes"; what goes and why (no
   loader, no run since 28 Oct, the reader deleted on 11 Nov); what changes in benchmarks'
   files; how to bring it back (`git checkout <the commit before>^ -- benchmarks/torch_gpu/attn_bf16`).
   No run ids of results in any file of the tree.
6. One worker comment on HASH-127 with the pull request (at most 100 words). **Tell me, and
   I ring benchmarks once, with the number.**

If the test's case turns out to be the only thing holding a rule of the shared gate, or
anything else says the directory is not dead (a reader you missed, a doc that promises it):
**stop, delete nothing, and return with that.**

Return by Fri 13 Nov 02:00Z with the pull request open, or with why not.

## Thu 12 Nov 23:00Z: #9525 has its land verdict (ci's reviewer, 22:59Z), with one line of bookkeeping for a follow-up. After A's pull request, not instead of it

ci's reviewer: "#9525 land verdict at `10c3daffdf23` (s8_linear, j10r_sm90 and tools/move
suites pass on the merge with main `417526501974`; lints same). Bookkeeping, not holding:
`tools/move/layout_one_map.toml:788` still lists `benchmarks/torch_gpu/s8_linear/build.sh:
integrations/vllm`, a reference `build.sh` no longer makes; drop it in a follow-up."

**The follow-up, one small pull request of its own** (on `main` once #9525 is on it, else
stacked on #9525; unmarked; `Lane kernels`, `Refs HASH-127`):
- drop that row;
- **and, in the same block of the map (lines 780 to 792 on `main`), every other row of
  `benchmarks/torch_gpu` whose file or whose reference is gone**: by me, the rows of
  `s8_linear/realw.py` and `units/README.md` name files #9260 deleted; say for
  `s8_linear/run.sh: integrations/vllm` whether `run.sh` still makes that reference after
  #9525. Find them with the map's own scan (`layout.py --map … --dry-run`, or whatever the
  reviewer's `tools/move` suite holds the rows to) and not by eye; say which check judges a row.
- **`attn_bf16`'s two rows (781 and 782) go with A's pull request**, not this one.
- Only rows of kernels' own files. A stale row of another lane's file: list it in the body
  for its owner, change nothing.
- The `tools/move` suite and the repository's, on a CPU node.

Tell me when it is open; it changes no code, so I read it and mark it.

## Fri 13 Nov 00:07Z: both pull requests read. #9552 is MARKED; benchmarks said YES to #9542 within three minutes of the ring. #9542 is marked after #9552 lands and you have rebased it: nothing to do until I resume you

Read by me: your return; #9552's diff over #9525's head (the eight rows; I checked the four
`examples/h200` files, the three gone files and `run.sh:25` on `main`); #9542's change to
benchmarks' files whole. **Judging the rows by `layout.py`'s scan on the tree before the move,
when you saw that today's dry run calls every row stale, and adding the direct `HFMA2.MMA`
test for the one thing the case alone held, are both the right kind of care.**
- **#9552: marked ready at `c3f7b9357889` at 00:04:36Z.** The queue's word at the mark: a pull
  request based on another's branch is not seen until that one lands, so I retargeted it to
  `main` (the head is the same; the queue orders it after #9525). **For your next ones: base
  `main`, and say in the body what it contains.**
- **#9542: benchmarks, HASH-127, 00:08Z: "Yes to #9542 ... Mark it."** Not marked yet: it and
  #9552 delete neighbouring rows of the map, and a stack of three (#9525, #9552, #9542) is
  past the queue's two. **When #9552 is on `main` I resume you:** merge `main` into
  `cursor/kernels-attn-bf16-delete-aa75`, keep both deletions, the touched suites again at the
  new head, push, tell me, and I mark it.

## Fri 13 Nov 02:14Z: #9552 IS ON `main`, with #9525, #9506, #9258 and #9299. Now #9542 takes `main`: the map's conflict closed with both deletions, the suites again, and tell me; I mark it

By me: `main` is `49d12bdc366a`; `git merge-base --is-ancestor` says yes for #9552's
`c3f7b9357889`, #9525's `10c3daffdf23`, #9506's `bc1fd81bb85c`, #9258's `b7623f9bfa91` and #9299's
`8289acf569ce`. **Both of yours from Thursday evening and KP2's two are landed.** `git merge-tree`
of `main` with #9542's head `1d3b431248ad`: one conflict, `tools/move/layout_one_map.toml`, as you
said; the two lint allowlists merge by themselves.

**Do:**
1. Merge `origin/main` into `cursor/kernels-attn-bf16-delete-aa75` (a merge, no rebase, no
   force-push). In the map keep both deletions: #9552's eight rows stay gone and `attn_bf16`'s
   two go. Nothing else of the map changes.
2. Look once more that nothing new on `main` names `attn_bf16` (a grep at the merged head; `main`
   moved by hundreds of commits since you cut the branch).
3. The suites the change touches, at the merged head, on a CPU node you name after looking at
   its slots (not one inside an infra window: cpu-2 and cpu-13 are tonight).
4. Push; the pull request's body says the new head and the run. One worker comment on HASH-127
   (at most 60 words). **Tell me the head: I mark it** (benchmarks' yes is on HASH-127, 00:08Z).

Return by Fri 13 Nov 04:30Z.
