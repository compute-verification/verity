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
