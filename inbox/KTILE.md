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

## Fri 13 Nov 02:40Z: AFTER #9542's merge is pushed and told: HASH-129, infra's question about 127 October backups on `vy-nebius-2` (544 GiB). A comparison by reading, a list, a proposal; kernels answers infra by Fri 16:00Z

infra asked kernels on INF-767 (02:36Z; read its two comments there, and HASH-129's text, which
is this brief's): may it remove the `backup/` directories of the 127 hourly `backup.sh` runs on
`vy-nebius-2` (`r20260930-091913-2c58` to `r20261007-141537-9d6e` under
`/workspace/research/runs`)? The disk is at 95% and proofs builds Lean there tonight. I
answered that the yes or no comes by Fri 16:00Z from your comparison.

**What I read on the node** (nothing changed there): `/workspace/pouw/infra/bin/backup.sh` (one
deterministic tarball a unit; an unchanged unit is a hard link to the previous backup's copy;
`backup/MANIFEST.tsv`: unit, file, sha256, bytes, chunks; `backup/summary.json`); the newest
backup: 854 units, 21.8 GB, 41 large units left out, 700 of them `fill-out` jobs; the live
`/workspace/pouw` (552 GiB, not touched by a yes) has no `gpu3-fp8` at its top level, which the
newest backup holds (23 units). infra: each run has a custody record, but the remote does not
cover every file.

**The work, all of it by reading. On the node: no write, no delete, no hashing of the tars, no
decompressing more than a listing needs, everything niced; proofs' builds have the node.**
1. **The union.** From the 127 `MANIFEST.tsv` files: every unit path, each version of it
   (sha256, bytes, the first and last run that hold it), and its newest version.
2. **Place each unit's newest version in one of three lists:**
   - **in the store:** a tarball of that sha256 is preserved in some run's custody (the store's
     own record of each run, as you read attempts for HASH-127; say how you know);
   - **live:** the same path exists under `/workspace/pouw` today (say whether its newest file
     is older or newer than the backup);
   - **only in a backup:** neither. For each: bytes, the last run that holds it, its top
     directory, and what it is (the tarball's listing of names, the head of it).
   **Units that an older backup holds and the newest does not** (deleted before 7 Oct) go
   through the same three lists.
3. **A search:** does anything on `main`, in `fixtures/artifacts.json`, or in the notes name one
   of the 127 runs or a path inside its `backup/`?
4. **Your proposal for the third list**, unit by unit or by directory: preserve (how many
   bytes; a small `research run --on vy-nebius-2` whose `$OUT` hard-links those tarballs so
   that custody uploads them is the way, since the node holds no store credential: write its
   line, do not launch it) or release, with the reason in a line (what it was, what replaced
   it, why nobody needs it).
5. One worker comment on HASH-129 (at most 100 words) with the three counts and bytes, and your
   note with the lists. **Tell me; the yes or no to infra is mine.**

Return by Fri 13 Nov 09:00Z.

## Fri 13 Nov 03:01Z: BEFORE HASH-129, and small: our three kernel build gates name the SASS gate by its repository path, so that compute-research's move of it follows them (HASH-127, one more item)

**Why.** compute-research moves `benchmarks/pouw/harness/sass_gate.py` and `sass_pins.json` to
`verity/protocols/pouw/prover/` on Fri 13 or Mon 16 Nov (the page for Daniel, TOP-415; the map
is on `main`: `tools/move/sass_gate_map.toml`, read its header). The mover rewrites each path
that names the file. Three of ours build the path from parts, so the mover leaves them and they
break when the file is gone [read by me on `main` `8c11a1d5ab`]:

- `benchmarks/pouw/pearl_c_sm90/sass_gate.py:20`: `parents[1] / "harness" / "sass_gate.py"`
- `benchmarks/pouw/pearl_c_sm120/sass_gate.py:19`: the same
- `benchmarks/torch_gpu/s8_linear/sass_gate.py:29`: `parents[2] / "pouw" / "harness" / "sass_gate.py"`

`benchmarks/pouw/tests/test_mma_gate.py` loads all three, so the move's own check would go red on
them; this takes that red away before it happens. I told compute-research and vllm on TOP-415
(03:0xZ) and promised the PR's number tonight.

**Do, on a new branch cut from `origin/main`, in a PR of its own (not in #9542):**
1. In each of the three, the path as one literal from the repository's root, the form
   `benchmarks/pouw/gpu_kit/sass_gate.py:50` has (`… / "benchmarks/pouw/harness/sass_gate.py"`).
   Nothing else in those files; no `.cu`, `.cuh`, cubin or pin changes.
2. **Show that it does what it is for:** in a scratch copy, `tools/move/layout.py --map
   tools/move/sass_gate_map.toml --dry-run --report FILE` (read its `--help` first; nothing of
   the dry run is committed). The report must list the three lines as rewritten to
   `verity/protocols/pouw/prover/sass_gate.py`. If it does not, stop and tell me what the mover
   wants instead.
3. **Every other reader that reaches the gate without naming that literal path**, in the whole
   tree (`"harness" / "sass_gate.py"`, `harness` on `sys.path` with `import sass_gate`, a
   `cd` into `harness/`): a list with file, line and owner. Known: `adapters/vllm/tests/engine/
   int8_attention_sass.py:50` (vllm's, told); `benchmarks/torch_gpu/attn_bf16/sass_gate.py`
   (gone with #9542). **Edit none outside kernels' three; the list goes in the PR's body.**
4. Whether any digest or pin covers one of the three files' bytes (I found none beyond the
   suites' declared inputs: `benchmarks/pouw/pyproject.toml:165`, `s8_linear/pyproject.toml`).
   If one does, say which and what it re-pins; do not re-pin without telling me.
5. The suites the change touches (`benchmarks/pouw`, `benchmarks/torch_gpu/s8_linear`), on a CPU
   node outside an infra window. Open the PR unmarked, body in the repository's form (what
   changed, why, the dry run's three lines, the list of 3, the run). One worker comment on
   HASH-127 (at most 60 words). **Tell me the PR's number and head: I read it and mark it.**

Then HASH-129 as briefed above; its return moves to Fri 13 Nov 10:00Z.

Return with this item by Fri 13 Nov 05:00Z.

## Fri 13 Nov 03:04Z: #9542's merge is read and right, but `main` moved under it (the remote merge, #9375, landed): one new conflict, one README row. FIRST of three: #9542 takes `main` once more, with the `stays` line; I mark on your word

**By me, on your head `fd4f0353ba20`:** its parents are `1d3b431248` and `49d12bdc36`; against
`49d12bdc36` it is 11 deletions and 11 edits, and the map loses `attn_bf16`'s two rows and
nothing else. Right as told. **But `main` is now `4a7291116e11`, 24 commits later, with #9375 in it,
and `git merge-tree` of it with your head gives one conflict: `benchmarks/pouw/harness/README.md`,**
the `sass_gate.py` row. Both sides edited that one line:
- `main` (`ca2e80f1ed`): "them," became "them (a div body that folds a constant numerator in,
  for that constant alone: `numerators`),";
- yours: `attn_bf16/` leaves the list of gates ("`pearl_c_sm120/`, torch_gpu's `s8_linear/`").

`benchmarks/pouw/pyproject.toml` and `tests/lint/allowlists/repo_retired_terms.json` merge by
themselves. Outside its own directory `attn_bf16` is named on `main` on the same 20 lines, in
the same files, as at `49d12bdc36` [my grep], so #9375 brought no new name of it.

**Your order of work now (three items; the two briefs below this one stand):**

**1. #9542, now.**
   a. Merge `origin/main` into `cursor/kernels-attn-bf16-delete-aa75` (a merge, no rebase, no
      force-push). In the README row keep both edits. Fetch first: if `main` moved again, take
      what it is then and tell me which commit.
   b. **Yes to your question:** `benchmarks/torch_gpu/attn_bf16/` leaves `"stays"` in
      `tools/move/layout_maps.json`, in this PR, as `d2a20a7bf7` did for `benchmarks/judge.py`.
      A list that names a directory the same change deletes is stale the moment it lands.
   c. The grep again at the merged head; nothing else may name `attn_bf16` but the lint
      allowlists' own history, if any (say what is left and why).
   d. Push, and **launch the suites at once** (`--changed` against the `main` you merged, a CPU
      node outside an infra window). The PR's body: the new head, the second merge, the run.
   e. Do not wait idle for the run: go on with item 2. **When the run ends green, tell me in
      your return: I mark #9542 then.** If it is red, stop and tell me what failed.

**2. The three kernel build gates by repository path** (the 03:0xZ brief above), on a branch
   cut from the same `main`, while #9542's suites run.

**3. HASH-129** (the 02:40Z brief). **The standing rule "nothing on `vy-nebius-2`" has one
   exception, for this item only: reading there, niced, with no write, no delete and no
   hashing of the tarballs,** as its brief says. Anything else on that node is still not yours.

Return with items 1 and 2 by Fri 13 Nov 05:00Z (sooner if #9542's run is green before the
gates' PR is open: then return at once with the head and come back for the rest). HASH-129
by Fri 13 Nov 10:00Z.

## Fri 13 Nov 03:08Z: ONE CHANGE to item 2 (the gates' PR): vllm's line goes in it too. Four lines, not three

vllm (lead) on TOP-415, 03:07Z: "Put our one line in kernels' PR tonight, with its three: you
have my yes as the file's lead, and no read by us is needed."

- **The fourth line:** `adapters/vllm/tests/engine/int8_attention_sass.py:50`,
  `HARNESS = Path(__file__).resolve().parents[4] / "benchmarks" / "pouw" / "harness" / "sass_gate.py"`.
  It becomes the repository path as one literal, as `gpu_kit/sass_gate.py:50` writes it. That
  line and nothing else in vllm's tree.
- vllm, read by it on `main` `4a7291116e1`: the adapter's other two mentions
  (`adapters/vllm/pyproject.toml` lines 131 and 152) already name the repository path. Check
  that the dry run rewrites them, and say so; change neither.
- **The dry run must now show four lines rewritten.** The suites: the adapter's too
  (`verity-vllm`), since its tree changes.
- The PR's body says whose yes the fourth line has (vllm's lead, TOP-415, 03:07Z), in its own
  line, and that the move's map is not edited.
- If you have already opened the PR with three, add the fourth as one more commit; no force-push.

Everything else in the 03:0xZ brief stands, the list of other part-built readers included.

## Fri 13 Nov 03:16Z: ITEM 2 IS REPLACED. I was wrong: the mover follows the four loaders. The gates' PR is now ONE LINE, `benchmarks/hash/exactlin/ksweep.py:51`. Do not change the three kernel gates or vllm's line

**What happened.** compute-research ran the mover with the gate's map and read each line
(TOP-415, 03:09Z): a path built from parts under `parents[n]` is an anchor the tool follows.
I had read the map's header and not run the tool. **Run by me since** [`tools/move/layout.py
--map tools/move/sass_gate_map.toml --no-lock` in a scratch tree at `main` `4a7291116e11`, each
line read after it]: `pearl_c_sm90/sass_gate.py:20`, `pearl_c_sm120/sass_gate.py:19` and
`torch_gpu/s8_linear/sass_gate.py:29` come out as `parents[3] / "verity" / "protocols" / "pouw" /
"prover" / "sass_gate.py"`, and `adapters/vllm/tests/engine/int8_attention_sass.py:50` the same
under `parents[4]`. **The two sections above this one that ask for those four lines (03:0xZ
and the "ONE CHANGE" one) are withdrawn.** I have told vllm that its tree is not touched.

**The one line of ours the tool does leave** (compute-research's finding, seen in my run too):
`benchmarks/hash/exactlin/ksweep.py:51`,
`SASS_GATE = Path(__file__).resolve().parents[2] / "pouw/harness/sass_gate.py"`. It came to
`main` with #9375. After the move it names a file that is gone.

**Item 2 now, on a branch cut from `origin/main`, a PR of its own:**
1. That line becomes `Path(__file__).resolve().parents[3] / "benchmarks/pouw/harness/sass_gate.py"`
   (the form `gpu_kit/sass_gate.py:50` has, which the tool rewrites). **Nothing else in the
   file:** #8671 edits `ksweep.py` elsewhere and is being merged with `main` by another worker
   tonight.
2. **Show it:** the mover's dry run with the gate's map and `--report FILE` lists `ksweep.py`
   with one more path than on `main` (on `main` its entry is `{"paths": 1}`); or run the move
   itself in a scratch tree, read line 51 after it, and throw the tree away. Say which you did
   and what the line read.
3. The suite the change touches (`benchmarks/hash/exactlin`), on a CPU node outside an infra
   window. Opened unmarked; one worker comment on HASH-127 (at most 50 words). **Tell me the
   number and the head: I mark it and ring compute-research,** who carries the line by hand
   in the move if ours is not on `main` by Fri 13 Nov 18:00Z.
4. **If you already pushed or opened anything for the four loaders:** do not force-push and do
   not close anything. Tell me the branch or the number in your return and leave it unmarked;
   the one-line PR is a new branch. If you had only edited locally, drop the edits.
5. **Not ours, told on TOP-415, change neither:** `benchmarks/pouw/settings.py:24`
   (compute-accounting's: `STANDALONE` lists `"harness/sass_gate.py"`) and
   `tools/verity/check/public_boundary.toml:138` (ci's: a brace row naming the two files).

Items 1 (#9542) and 3 (HASH-129) stand as they are. The returns' times stand.

## Fri 13 Nov 04:00Z: both are read and MARKED (#9603 at `8801fc5b44e9`, #9542 at `d6fc43e9bdb8`). Yes to `job.sh:94`, as one line and half an hour at most. Then HASH-129

**By me:** #9542: the two merges and the `stays` commit; against `4a7291116e11` 11 deletions
and 12 edits; the README row holds both edits; `attn_bf16` is in no file at the head; it merges
with `main` `8ef93f01963c`. I took "(not marked without benchmarks' yes)" out of its title,
since the title becomes the queue's commit on `main`. #9603: the one line, `parents[3]` is the
root, it merges with `main` and with #8671's branch as it is now (`28f124b29546`). **Both marked
at 03:5xZ; the preflights refused nothing.** Finding `job.sh:94`, and running the real move
before my stop reached you, is the work I should have asked for in the first brief.

**1. `benchmarks/torch_gpu/j10r_sm90/job.sh:94`, now, in a PR of its own.** It is ours, no test
runs it (it is a card's job), and a break there is found on a card, which is the dearest place
to find it.
- The line: `h=$S/s8/benchmarks/pouw/harness`. The scratch tree mirrors the repository, so
  after the move the gate's copy must sit where `s8_linear/sass_gate.py` then looks for it
  (`$S/s8/verity/protocols/pouw/prover/`).
- **Write it so that the mover follows it**, in one line if you can: for example the
  directory taken from the gate's repository path as one literal
  (`h=$S/s8/$(dirname benchmarks/pouw/harness/sass_gate.py)`). Whatever form you choose,
  **show it as for #9603**: the real move in a scratch copy, then lines 94 and 99 read after
  it, and say whether the copy lands where the moved `s8_linear/sass_gate.py` loads it from.
- Say in the body what stays true and unshown: the job takes `sass_gate.py` from LIBS, so a
  run whose LIBS holds files from before the move with a tree from after it (or the other way
  round) still mismatches; that is the run's to pin, not this line's.
- `bash -n` on the script; the directory's suite and whatever `suites.py --changed` picks, on
  a CPU node. Opened unmarked; one comment on HASH-127 (at most 40 words); tell me the number.
- **Half an hour. If no one-line form is followed by the mover, stop:** tell me, and I give
  compute-research the line as a known leftover of its move.

**2. The unopened branch `cursor/kernels-sass-gate-literal-path-30cb`:** delete it on the
remote (`git push origin --delete`); it was never opened and nothing reads it. Say so.

**3. HASH-129,** as the 02:40Z brief has it, with the exception for reading on
`vy-nebius-2`. Return by Fri 13 Nov 10:30Z.

Return with item 1 by Fri 13 Nov 04:45Z, then go on to HASH-129 when I resume you.

## Fri 13 Nov 04:05Z: `job.sh:94` IS NOT NEEDED: compute-research's move already takes that line by hand. If your PR is not open yet, drop it and go straight on to HASH-129

compute-research (TOP-415, 04:03Z): its staged move has a second, hand commit of three places the
tool does not reach, and `job.sh`'s scratch directory is one of them
(`h=$S/s8/verity/protocols/pouw/prover`; `c5032c44805e`; its suites passed on that tree, 63 of
63 [relayed]). That hand commit stays in the move whatever lands tonight. A PR of ours is
optional, and it would only make them drop a line from a commit that already works.

- **Not opened yet:** stop, open nothing, throw the local edit away, and **go straight on to
  HASH-129 without returning to me first** (the 02:40Z brief; reading on `vy-nebius-2` only,
  niced; return by Fri 13 Nov 10:30Z). In that return, one line saying where you stopped.
- **Already opened:** finish it as briefed (the scratch move's two lines, the suite), then go
  straight on to HASH-129 too, and give its number in a comment on HASH-127 (at most 30
  words) so that I can read and mark it while you work. Either way is fine for the move.
- The stray branch `cursor/kernels-sass-gate-literal-path-30cb` is still to be deleted.

## Fri 13 Nov 04:43Z: HASH-129 is read: the lists, the search and the proposal are what infra's question needed. One thing before my yes: launch your preserving run for the two units, and show that the store holds their bytes

**Read by me:** your note `20261113T0440Z-report-ktile-hash129.md` whole; CMP-71 and MEM-61
(both Done on 21 Oct: infra's ask, compute-accounting's approval, "put anything under fill-out
that isn't in the store yet into it … after that, infra removes the rest"). Finding that
"PRESERVED" on those backups meant the record and not the tarballs, by the per-chunk check
against the records, is the finding of the piece; infra hears it from me.

**My decision:** yes to your proposal. `fill-out/*`, `gpu3-fp8` and the three markers are
released: their owners decided on 21 Oct. `exhaustion.rpC5JB/rows` and
`tmp-noise-sweep.KI0k3j/ship-b96af447` are kernels' own scratch that no decision names:
510 MB is cheap against a deletion that cannot be undone, so they are preserved first.

**Do:**
1. **Launch the line your note has** (the hard links into `$OUT`, the manifest rows beside
   them), niced, on `vy-nebius-2`'s research pool. It may write in its own run directory
   only. If the pool gives no slot in 15 minutes, say so; do not move it to another node (the
   tarballs are only there).
2. **Show the bytes are in the store, file by file:** the run's record lists each of the 9
   chunks as on the remote, with its bytes, and each one's sha256 equals the chunk's sha256
   in its backup's `MANIFEST.tsv`; the total is 509,613,229 B. "Preserved" as a word from
   `research data preserved` is not the check, as you found.
3. One comment on HASH-129 (at most 60 words): the run's id and that check. **Tell me: my
   yes to infra for all 127 follows your return.**

**#9628:** read. It is right, and it is not marked: compute-research's hand commit already
holds that line, and a landing of ours between its staging and its move's landing would stop
its move. It stays open and unmarked; do nothing to it.

Return by Fri 13 Nov 05:45Z.

## Fri 13 Nov 05:40Z: infra removed the 127 on our yes (855 GiB free). NEXT, and the last of this: HASH-130, the 31 large-unit backups, the same comparison by reading; kernels' word to infra by Mon 16 Nov

**What happened:** my yes went to infra on INF-767 at 04:5xZ with your counts, your finding
about "PRESERVED" and the 31 runs named as not covered. infra (05:39Z): the 127 `backup/`
directories are gone (`research retention rm`, 127 deleted, none refused; each run's record
and logs stay); it takes `backup.sh` and `backup_unit.sh` off `main` in its next clean-up.
**For the 31 large-unit runs, 150 GiB on the node: "your comparison by Mon 16 Nov 6 PM PST is
soon enough, and I remove them on your word here."**

**HASH-130 is that comparison** (read its text: it is this brief). You have the scripts, so
do it now rather than carry it:
1. The 31 `backup_unit.sh` runs (1 and 7 Oct; your note's "Outside the question"): each
   unit placed in one of the three lists, as for HASH-129: in the store whole (by sha256,
   file by file, not by the word "preserved"); live under `/workspace/pouw` (and whether
   newer); only in a backup.
2. For each unit only in a backup: its bytes, the head of its listing, and whether the
   owners' two arts hold its files by path and size (as you matched `fill-out`), or which
   decision released it (CMP-71, MEM-61, `left-behind.tsv`, `MANIFEST-large.tsv`).
3. The search: `main`, `fixtures/artifacts.json`, the notes, for any of the 31 runs.
4. **Your proposal, unit by unit: release (whose decision), or preserve.** Preserving here
   may be tens of GB: **do not launch a preserving run; give its line, its bytes and why
   the unit is worth them, and I decide.**
5. On the node: reading only, niced, through pool slots as before; no write outside a run's
   own directory, no delete, no hashing of whole tarballs beyond what a listing needs. If
   proofs' Lean runs hold the pool, wait your turn; nothing here is urgent.
6. Your note (a new one or a section of the HASH-129 note), the per-unit table, one comment
   on HASH-130 (at most 100 words). **Tell me; the word to infra is mine.**

Return by Fri 13 Nov 09:00Z.

## Fri 13 Nov 06:10Z: HASH-130 is read and closed: my yes for the 31 went to infra with your finding about the store's missing blobs. NEXT: HASH-131, the plan for moving GPU code beside what runs it. By reading; no code changes

**Since your return:** infra has the word (INF-767): all 31 may go; `die6` released on the
hash lane's decision of 21 Oct; the 395 absent blobs named to infra as its finding. Using
`research data verify` instead of `fetch --list`'s `remote: true`, and saying that 27
tarballs cannot be decompressed without their first chunk, turned a guess ("likely nothing to
keep") into a fact. #9542 and #9603 are on `main` (both merged 05:31Z).

**HASH-131** (read its text: it is this brief, and it says what is settled). Daniel read the
architecture page tonight. Its section 7 has a row of kernels': GPU code is in 61 directories,
49 under `benchmarks/` (my count of Thursday evening), and a served run loads PoUW's kernel
from `benchmarks/`, which the page forbids. I promised on it **a plan with days by Fri 13 Nov
6 PM PST**. You know these directories from HASH-104 and HASH-127. The work:

1. **The table.** A row for every directory on `main` that holds GPU sources. State the rule
   that counts a directory (`.cu`, `.cuh`, `.ptx`, a cubin or its build script; say what
   you do with Python that drives a GPU, such as the adapter's `engine/int8_*.py` and Triton
   kernels) and compare your count with 61 and 49; if they differ, say by which directories.
   For each row: what it is in a line; its owner (`research msg owners PATH`); **what loads it
   and how, by file and line** (an import, a path built in code, a shell step, a job script,
   a suite's inputs); whether a served run, a protocol's prover, a commitment, a dashboard
   number, a test only, or nothing reads it; **where it goes at the end state**
   (`verity/protocols/<p>/prover/`, `verity/protocols/pouw/prover/kernels/`,
   `verity/commitments/`, the adapter, `benchmarks/kernels/`), or that it is deleted, with
   the reason.
2. **For each row that moves: what follows a path.** A binary's digest or pin
   (`overhead_arm.json`, `sass_pins.json`, a `MANIFEST.sha256`, an `art:`), a build script's
   embedded path (does the cubin's sha256 change when its directory does? HASH-102's four
   conditions), a suite's declared inputs, a list in `tests/test_boundaries.py`, a row of
   `tools/verity/check/public_boundary.toml`, a move map. And what must hold after the move.
3. **What is already moving, by whom, so the table shows it and does not plan it twice:**
   memory-accounting's two provers (its #9061, Fri 13 Nov); the machine-code check
   (compute-research, Fri after noon PST); PoUW's package (compute-accounting with vllm and
   kernels, not before Tue 17 Nov); the served kernels' circuits (top, vllm, circuits).
   Read each one's PR or map and say which rows of the table it takes.
4. **The plan, on one page:** the moves in order, each with its owner, what it waits on and
   a day you would propose; PoUW's first; what is deleted instead of moved (it joins the cut
   of our three benchmark directories, Fri 20 Nov); what cannot be dated and why. **Mark
   each line as read, run, or guessed.**
5. A note, the table as TSV beside it, one comment on HASH-131 (at most 100 words). **No
   branch, no PR, no file in the tree.** The days are agreed by me with vllm and
   compute-accounting on Friday; your plan is what I bring.

Return by Fri 13 Nov 14:00Z. If the table is done earlier, return then.

## Fri 13 Nov 07:15Z: THE PLAN'S ROW 4 IS TONIGHT. PoUW's served GPU directories move into its package in top's chain 2, by a map compute-accounting's worker writes by 09:30Z. Your table is what checks it. Two passes: a scratch run now, then the real branch

**What happened since your table.** Daniel agreed the architecture page and asked for the
repository in its structure when he wakes (TOP-415, 07:05Z): every path-only move is on
`main` by Fri 15:00Z, in two chains marked by 13:00Z. Daniel also regrouped the protocols:
`verity/protocols/pouw` becomes `verity/protocols/accounting/pouw` (the proofs under
`protocols/proofs/`, the policy under `protocols/policy/`). compute-accounting (07:20Z): **its
path-only moves join chain 2 tonight: `pearl_c`, `pearl_c_sm90`, `pearl_c_sm120`,
`served_zk`'s `hm96_rows.cu` and `filler` into the package's `prover/kernels/`; `pouw_cuda`
out of the adapter with vllm's yes; `pearl_c4`, dither's `h1u.cu` and `h1sha.cuh` and
`served_h1` into `verity/protocols/experimental/`.** Its worker pushes a branch with those
rows by 09:30Z and rings me with the head; a row that needs a code edit is left out and
listed. I said yes for kernels on TOP-415 (07:2xZ) and promised two lists.

**Pass 1, now (the list is owed by 09:30Z; return with it as soon as it is done):**
1. In a scratch copy of `origin/main` (say which commit), a map of your own with those rows
   (copy the format of `tools/move/sass_gate_map.toml`; destinations under
   `verity/protocols/pouw/prover/kernels/`, since the group move is another map's). Nothing
   of it is committed anywhere.
2. Run the mover for real on that copy and list **every reader it leaves behind**: a path
   built in a form it does not follow (as `ksweep.py:51` and `job.sh:94` were), a setting's
   default, a spec's `kernel =`, a `contents/kernels/*.toml` field, an include across
   directories, a build script that names a sibling, a suite's declared inputs, a row of
   `tests/test_boundaries.py` or of the public export, a pin. File, line, what it reads
   after the move, whose it is. Your table's `path_sensitive` and `followers` columns say
   where to look; this pass says what the tool in fact does.
3. `python3 tools/verity/check/move_check.py` on the scratch range, if it runs there: does it
   certify, and what does it refuse.
4. What you would tell compute-accounting's worker in ten lines: the rows that are safe as
   pure moves, the rows that need a hand edit, and the order.

**Pass 2, when I resume you with the branch's head (about 09:30Z; the list is owed by
11:00Z):** the same on the real branch, and its rows against your table: anything of the
table's that the map misses, anything the map takes that the table has staying.

**Not yours tonight:** building anything (another worker builds the H200 kernel before and
after and compares bytes), a PR, a file in the tree. From now, in prose: "challenges" not
"coins", "accounting protocol", "the auditor's devices" and never "host".

## Fri 13 Nov 07:33Z: LATER, after both passes on tonight's map and not before: one more backup, `r20260930-081105-b32f` (HASH-130 reopened for it)

infra removed the 31 on our word (965 GiB free) and took your finding as INF-790. It left one
`backup/`: the first hourly run, `r20260930-081105-b32f`, 1.5 GiB, raw files with no manifest,
which our yes did not name. Your HASH-129 note says all 84,073 files are on the remote; if
that was by `fetch --list`, INF-790 says it cannot be trusted. I ran `research data verify` on
its record (`art:623eb8c6d156988da1679bf9202ba13007776732a6772591d8d16313befecdac`) and stopped
it at 420 s with no verdict. I told infra: yes or no by Mon 16 Nov 6 PM PST.

When tonight's passes are done: run that check to its end (in a session that outlives a
command, on your machine: it is a store read), list what the store lacks, and for those
files say whether the live `/workspace/pouw` has them (reading, niced, as before) and what
they are. Then your proposal. One comment on HASH-130. No hurry: by Sat 14 Nov 18:00Z.

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

## Fri 13 Nov 07:43Z: for pass 2, two additions: chain 2's kept rows, and both of `pouw_cuda`'s loader lines

Nothing to do until I resume you with a branch's head. When I do, read this first.

**compute-accounting took pass 1's list as its rule for chain 2 (TOP-415, 07:40Z).** Kept:
`pouw_cuda`; then `pearl_c` + `hm96_rows.cu` + `pearl_c_sm120` as one group with the hand
edits; then of `pearl_c_sm90` only the kernel's sources, `build.sh`, `run.py`,
`fixture.py` and `sass_gate.py`. Out: `filler`, `served_h1` with dither's two files,
`pearl_c4`, and any experimental Python file a served PoUW file still imports. Its worker
runs the touched suites on the moved tree and posts its head on TOP-415.

**`pouw_cuda` has two loader lines, and vllm's worker writes the row's second commit
(a pushed branch by 09:30Z):**
- `adapters/vllm/verity_vllm/protocols/pouw/device.py:33`: the mover leaves it.
- `adapters/vllm/verity_vllm/commit/row_tables_device.py:48`: the mover rewrites it to
  `parents[4] / "verity" / … / "pouw_cuda"`, which names the files in a checkout only
  (your note's line 86; my own run of the row alone on `main` `9182900fb94e` shows the
  same). I had not carried it into my post; vllm found the line by reading.
- The mend, vllm's and your note's: both take the directory from the package,
  `Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"` with
  `from verity.protocols.pouw import prover`.

**Pass 2 checks, beside the rows against your list:** both lines read from the package at
the posted head; no `parents[N]` join to `pouw_cuda` is left anywhere
(`git grep -n pouw_cuda` for every computed path); and `row_tables_device.built`'s host
build compiles from the moved directory.

## Fri 13 Nov 08:34Z: the Pearl-C group is OUT of chain 2 by my call; pass 2 for `pouw_cuda` is done by me. YOUR NEXT PIECE: the design of the Pearl-C kernel directories' move as a pull request of its own. Return by 13:30Z

**What happened (read TOP-415 from 08:29Z whole first).** compute-accounting's worker MAPROWS
staged PoUW's rows on chain 1: maps `cursor/pouw-move-rows-chain1-1de3` at `6f02807a4d66`,
the moved tree `cursor/pouw-move-staged-1de3` at `a61749354433`. `pouw_cuda`, receipts and
Dither pass with path edits. The Pearl-C group does not: `fixture.py` of both kernels reads
the verifier's tests' vectors through `sys.path` and cannot live under `verity/`. MAPROWS
proposed leaving `fixture.py` and `sass_gate.py` in `benchmarks/` (its step 7).

**What your pass 1 and my relay of it both missed, read by me on `main` `b3902993caea`:**
- **The served loader takes a kernel directory as a unit.**
  `adapters/vllm/verity_vllm/protocols/pouw/pearl_c/kernel.py:54-55` loads `run.py` and
  `fixture.py` from the directory the `kernel` setting names and takes the cubin there;
  the served arm calls `fixture.keys_of` (`pearl_c/__init__.py:456`; see also
  `host.py:147`'s `unit_block`); `source_digest` is the hash of those two files' hashes.
  So `fixture.py` can neither stay behind nor move as it is.
- **`grouped.py` and `debit.py` belong with `run.py`:** the adapter's
  `pearl_c/grouped.py:294` loads `grouped.py` beside `run.py`; `debit.py:24-25` imports
  `run` from its own directory. The list I posted ("only the kernel's sources, `build.sh`,
  `run.py`, `fixture.py`, `sass_gate.py`") split them.
- **The method's gap: a directory's move is checked against what loads the directory, not
  only against the paths inside the files that move.**

**My call (TOP-415, about 08:34Z): `pearl_c`, `pearl_c_sm120` and `pearl_c_sm90`'s files stay
out of chain 2, whole. They move in a pull request of ours with the code change.** I owe
top the GPU plan by Sat 02:00Z; this design is its largest row.

**Do (reading and design; push nothing to another lane's branch; a scratch branch of ours
on `main` after the chains is for later, not now):**
1. **Every loader of each of the three directories as a unit.** Who takes which file by
   name from a kernel directory: the adapter (`kernel.py`, `grouped.py`, `host.py`), the
   specs' `kernel =`, `PEARLC_KERNEL`, the setup scripts, `bench_arm.py`, `overhead_arm.py`,
   `debit.py`, the tests. A TSV: file, loaded by (file:line), how (import, by file, glob,
   run), served code or benchmarks.
2. **`fixture.py`, both kernels:** which functions served code calls and what each needs
   to import; which only the build's check needs (the verifier's tests' vectors). **Propose
   the split:** the served part's name and place, the check's place. For each option, what
   it does to `source_digest`, and what names a source digest today (manifests, the gate's
   record, any pin in the tree: `git grep`).
3. **Which files travel with `run.py`** and which stay, one reason a file.
4. **`build.sh` runs `fixture.py` as its check:** where that check goes so that nothing
   under `verity/` runs a `benchmarks/` file.
5. **`sass_gate.py` of each kernel** once compute-research's map has moved the harness's.
6. **The readers' hand edits** as a checklist: MAPROWS's step 8 and your pass 1 list,
   merged, with the suites that must pass (the whole suites, not `--quick` alone), KYARM's
   bytes check and a served smoke on an H200.
7. **Owners and order:** which change is kernels' and which is vllm's (`kernel.py`), the
   commits in order, what each depends on. No estimate in days: say what must be true first.

**Output:** `lanes/kernels/<stamp>-report-ktile-pearlc-move-design.md`, two pages at most,
the TSV beside it, and a draft of the plan's row (what moves, to where, in which pull
request, after what). CHECKPOINT lines each 30 minutes; commit and push before you end.
**Return by 13:30Z.** HASH-130's backup record (by Sat 18:00Z) stands after this.

## Fri 13 Nov 08:41Z: for your design's part 2, vllm's answer (TOP-415, 08:40Z): read it at your next checkpoint

vllm takes the call and designs the split of `fixture.py` with us. Its input:
- **The source digest is compared with nothing stored** [its `git grep` for `source_sha256`
  and `source_digest` on `main` `b3902993caea`]: `pearl_c/kernel.py:57` computes it,
  `ends.py:342` writes it into the manifest, `__init__.py:386` prints it; no test or check
  of the adapter pins a value. **Check that once yourself, and look outside the adapter
  too** (the harness's records, `overhead_arm.json`, the gate's record, the evidence store's
  labels), since your option table needs it.
- **Its wish for the design: the digest covers exactly the files the served loader
  executes**, so that a change to a test vector's reader no longer renames a served
  kernel. Serving reads `keys_of` from `fixture.py` and everything of `run.py`.
- It answers our plan the evening it is posted (Fri 6 PM PST).
So one option in your table is: the served part of `fixture.py` in a file of its own beside
`run.py`, and the digest over `run.py`, that file and `grouped.py` where it is loaded.
Say what that does to a reader who compares manifests across the change.

## Fri 13 Nov 08:57Z: YOUR DESIGN IS ACCEPTED AS PROPOSED, and you write both pull requests, after chain 2 is on `main`. NOW, while it lands: the backup `r20260930-081105-b32f` (the 07:33Z section). Return by 12:30Z

**The design** (read whole by me; five claims and the registry checked on `main`
`257b6c5e2df5`; posted on TOP-415 as a proposal). **compute-accounting, the owner of the
loader's files, `benchmarks/pouw/` and the prover (TOP-415, 09:00Z): "Accepted as proposed
… Your worker writes both pull requests, since it holds the design and the 79 loaders; we
review and mark them as owners."** vllm takes it too and answers the plan the evening it
is posted. MAPROWS drops its Pearl-C work; its step 8 is your cross-check only.

**compute-accounting's conditions, which are the pull requests' acceptance:**
- after chain 2 **and** compute-research's map of the machine-code check are on `main`;
  each carried with `tools/move/restack.py` if anything moves under it;
- **the split changes no path and no cubin;** its body says the source digest changes and
  why, with each loaded file's hash in the manifest (vllm's wish);
- the move covers the three loaders step 8 lacks (`glm53_passes.py:75`,
  `glm53_timing.py:256`, `overhead_arm.py:241`) and your whole table;
- the H200 bytes check (KYARM's job) runs on the move's head;
- **nothing lands within a day before compute-accounting's timed H200 run (Tue 17 Nov).**

**I resume you for the split when chain 2 is on `main`** (top's target is Fri 15:00Z; it
may slip). Do not start a branch before that: chain 2 rewrites `fixture.py`'s imports and
moves the prover's package under `accounting/`.

**NOW, in the hours between:** the 07:33Z section above, the backup
`r20260930-081105-b32f` (HASH-130, reopened for it): the store's copy verified to
completion, file by file, and the same comparison by reading as for the 31; your word on
whether infra may remove it; a comment on HASH-130 of at most 100 words. A CHECKPOINT each
30 minutes; commit and push before you end. **Return by 12:30Z.** If it cannot finish by
then, return with what is verified and what is left, and do not hold the split for it.
