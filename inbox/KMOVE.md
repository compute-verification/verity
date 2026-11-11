# Inbox: KMOVE (HASH-102)

## Tue 10 Nov 08:43Z: next, when resumed: the Flock fix proofs asked for, then the rerun on the real maps

proofs said yes on VLLM-265 (08:41Z, read it): kernels opens the fix and marks it; it is a fix for the moves.

1. **`verity/ml/flock/cuda_sha512_patch.py`**, as proofs constrains it: `pod/60-circuit.sh:34` runs it with a bare
   `python3` inside the Flock checkout, so `verity` is NOT importable: do not import `SHA512`. Find the repository's
   root from `__file__` and join the root-relative literal `verity/ml/kernels/sha512.cuh` in a form `layout.py`
   rewrites; take the include line from `cuda/sha512.cuh` by pattern, asserted to match once. It must hold before
   and after the move, so it lands ahead of the maps.
2. **The three shell defaults** (`pod/74-gemm-hill.sh:60`, `pod/71-gemm-slowdown.sh:15`,
   `benchmarks/one_stage/phase_profile.sh:82`) by hand in the same PR, in a form the tool rewrites, unless top's
   MOVESTAGE has taken the `:-` pattern into `layout.py` by then (check `origin/main` and #8592's branch first).
3. **Test it on the real maps:** `origin/cursor/vllm-maps-3012` (22 maps `tools/move/layout_NN_NAME_map.toml`, run in
   name order; map 01 sends `verity/ml/kernels/` to `verity/definitions/kernels/`, map 04 sends Flock's prover to
   `verity/flock/prover/`). In a clone with your fix: before the move and after maps 01 and 04 (better: all 22),
   the script's header path exists and the include pattern matches once in `cuda/sha512.cuh`; the three defaults
   name files that exist. Put the commands and their output in the PR.
4. **Then the rerun of your scan on the real maps** (your note's "Not done"), and one comment on HASH-102 with what
   changed against the trial: lines that now resolve, lines that newly break, by owner.
5. PR from a branch `cursor/kernels-flock-kernel-path-<hex>` off `origin/main`, body with `Refs HASH-102` and
   `Goal kernels/code-at-overhaul-bar`, marked with `research queue ready N --mark --by kernels` once the test in 3
   passes and the suites the change touches pass (`--mem-gb 32` and a free pool: `--on vy-nebius-cpu-12 --pool
   research`, or cpu-4, or cpu-10; infra, 08:18Z).

## Tue 10 Nov 08:46Z: correction to the section above: the three shell defaults are NOT in your PR

top (VLLM-265, 08:44Z): the `:-` pattern goes to its worker MOVESTAGE, fixed in `layout.py` with a test; its report
is due 10:30Z. Leave the three defaults alone. If that report says the fix is not in, the lead tells you here and you
edit them by hand in a follow-up commit. Everything else stands, and the lead's resume message of 08:46Z is the brief
of record.

## Tue 10 Nov 08:56Z: piece C, BEFORE piece B and while piece A's suites run waits: map 21's row for `served_weights.py`, by 10:30Z

compute-accounting asked on VLLM-265 (08:54Z, read it) that `examples/h200/e2e_h200/served_weights.py` never enters
`archive/`: its own row in map 21, moved to `benchmarks/torch_gpu/p2_stage/served_weights.py` (kernels' file from
then on), with its readers pointed at the new place in the same map, so no runner of kernels' ever reads `archive/`.
vllm's maps workers write map 21 (`tools/move/layout_21_archive_map.toml` on `origin/cursor/vllm-maps-3012`; today
it has only `"examples/h200/" = "archive/examples/h200/"`). The readers name the file as a MODULE, four ways, and a
map's worker will not know which the tool carries. You do. Done when, by 10:30Z, one comment on HASH-102 (the lead
relays it to VLLM-265) gives:

1. The exact lines for map 21 (`[paths]`, and `[modules]` or `[rewrite]` if the tool needs them) that move the one
   file and leave the rest of the demo going to `archive/`. Say what module name the file has after the move and why
   it imports: `p2_stage`'s scripts run with `PYTHONPATH=$PWD:$PWD/benchmarks/pous:$PWD/examples/h200` (`full.sh:56`);
   read how `full.py`, `layer.py`, `check.py` and `layout.py:252-255` import their siblings and the generator.
2. For each reader, what `layout.py` does under those lines, tested on a clone of the maps' branch with the lines
   added: `p2_stage/layout.py:252-255`; `check.py:132`, `full.py:431`, `layer.py:424` (the `--generator` default, a
   string); `full.sh:56,89`; `session.sh:10,64-71,103` (`--send`, `python -m e2e_h200.served_weights write`);
   `s8_linear/realw.py:26-27`; `units/build.sh:43` and `fixture.py:17,304-307` (these two read the demo's `replay.py`,
   not this file: say what happens to them). Rewritten and resolving, or left and broken.
3. The hand edits the tool cannot make, as a diff kernels lands on the moved tree in the reds window (or, if a line
   can be written today in a form the tool carries, as a commit that lands before the maps: say which is less risk).
4. The test: after the map, `python -c` imports the generator by its new name with the `PYTHONPATH` the scripts set,
   and `git grep -n -E 'e2e_h200|examples/h200' -- benchmarks/hash benchmarks/torch_gpu benchmarks/moe_align` lists
   only the three `commit_slots` files that go to the archive with the demo (`overlap.py`, `overlap.sh`,
   `green_probe.py`; asked of vllm for map 21 as rows of their own) and `units/`'s two lines.
No build, no GPU, no suites run for this piece: a clone and the tool.

## Tue 10 Nov 09:07Z: how to mark your fix PR tonight (top, TOP-394 09:06Z; ci, CI-501)

Since `main` moved at 08:15Z, `research queue ready` refuses every head that holds today's `main` ("its head carries
7 merges of an old `next` that `checked` doesn't hold"), and `research queue status` lists it as refused. Your branch
is cut from today's `origin/main`, so it will be refused. Until about 12:30Z (the fix is #8614):

1. Run the suites your change touches, as before any mark.
2. Mark by label: `research data label pr:N@SHA ready true --by kernels` (N the PR, SHA its full head), and say in the
   PR's body that it was marked by label under CI-501.
3. Ignore `status`'s "refused" for that reason alone. Any other refusal is real: report it.
If `research queue ready N --mark --by kernels` does accept the head, use it as usual.

## Tue 10 Nov 10:08Z: two queue rules from top (TOP-394, 10:06Z), until ci changes the loop

One mark dropped two candidates at 09:45Z (29 pull requests, 38 minutes of check). So:

1. **From your mark until #8619 is on `main`, push nothing to it and withdraw nothing, unless it is wrong.** A fix
   or a follow-up is a new pull request. So the three shell defaults, if MOVESTAGE's report says no, are a NEW PR,
   not a follow-up commit; and so is anything piece C needs landed.
2. **Never mark a pull request that contains another of kernels' open, marked ones** (#8521, #8542, #8619). Cut any
   new branch from `origin/main`, not from one of those.
If either must be broken, stop and tell the lead: it is asked of ci on CI-489 first.
`research queue status` shows #8619 as refused for the old-`next` reason only: that is CI-501's fault, as told.

## Tue 10 Nov 10:33Z: piece B changes: the layout is ONE map (top, TOP-394 10:30Z), and a staged tree exists now

top ruled: one map, one commit, written directly in `tools/move/`; the 22 files leave vllm's pull request. So:

1. **Do not apply maps yourself for piece B.** A moved tree is staged: `origin/cursor/top-layout-draft-9ece` at
   `9ece46b2ea0d` (the chain of 23 on `main`, for staging only). Run `kmove_joins.py` and `kmove_pins.py` on it now,
   and write what changed against your trial (lines that now resolve, lines that newly stop resolving, by owner) in
   your note and one comment on HASH-102. Update `joins_missed.json` and `loads.tsv`.
2. **Kernels' own suites on the staged tree, now, so kernels' reds are known before top's list (about 14:00Z):** one
   `research run` at that commit with the suites whose tests live under `benchmarks/hash`, `benchmarks/torch_gpu` and
   `benchmarks/moe_align` (and Flock's, for your new test), `--mem-gb 32`, on a free pool that auto never picks
   (`--on vy-nebius-cpu-12 --pool research`, or cpu-4, or cpu-10). top says 246 imports fail a suite's read guard
   on the moved tree in 16 suites and are closed by entries in each suite's `inputs`, which the stager applies: a
   red of that kind is not kernels' to fix; list the others, each with its file and line.
3. **When top names the one-map branch** (due 11:15Z on TOP-394), rerun 1 on it and say whether anything differs
   from `9ece`. Your row for `served_weights.py` is taken (vllm, VLLM-265 10:14Z: row 14, with `[aliases]`): check
   that the one map has all three lines.
4. Nothing is pushed to a marked pull request and nothing new is marked beyond #8619 without the lead's word.

## Tue 10 Nov 10:43Z: your one-map finding stands (top, VLLM-265 10:41Z). Two things follow

top ruled: the `[tokens]` row stays; the joins are fixed at the source. proofs root-anchors the joins of the five
files on `main` today (a `parents[N]` anchor to the root plus the whole root-relative path as one literal). If that
is not on `main` at 19:00Z the stager makes the same edit in the layout PR's second commit.

1. **WITHDRAWN at 11:02Z: no commit goes to #8619.** [was: prepare, and hold, one more commit for #8619.] After #8619, `cuda_sha512_patch.py` has one join from `here`
   left: `s = (here / "cuda" / "sha512.cuh").read_text()`. proofs' edit of it would conflict with #8619's hunk, so
   the lead proposed on VLLM-265 that kernels adds it to #8619: the root from `__file__` and the root-relative
   literal `verity/ml/flock/cuda/sha512.cuh`, the now unused `here` deleted, your test extended to that path.
   Test it under the ONE map (the script's three paths exist after it; the unfixed line as the control). **Do not
   push it**: #8619 is marked, and a push to a marked pull request needs ci's word, which the lead has asked for
   (rule 1, TOP-394 10:06Z). Keep the commit on a local branch and its patch beside your note
   (`tools/kmove/8619_followup.patch`), say so on HASH-102, and wait for the lead's word in this file. If the word
   is no, the patch goes to proofs.
2. **The rerun on the one-map branch** (top names it on TOP-394 by 11:15Z; until then the one map is
   `tools/move/layout_one_map.toml` on `origin/cursor/vllm-maps-3012`): the scan, then ONE comment on HASH-102 that
   the lead posts on TOP-394 as it is: a table by owner (lines, files, how many are tests the check runs and how
   many are scripts or builds it never runs), the path of the per-line list in the notes, and the three or four
   breaks that stop a build or a served path, each in one line. At most 200 words. Every owner fixes its lines the
   same way, so give the rule in one sentence.
The staged tree `9ece` (23 maps) is no longer the form of record: skip the scan there. Kernels' own suites on the
moved tree: run them on the one-map branch instead, once it is named.

## Tue 10 Nov 10:46Z: WITHDRAWN at 11:02Z (see the last section). DO NOT PUSH TO #8619. [was: go, push the one commit]

top's word, as the rule's author: #8619 has no verdict and is in no candidate, so a new head drops nothing. **Do it
before a reviewer takes the old head. This goes ahead of everything else you are doing.**

1. The one commit: `cuda_sha512_patch.py`'s last join from `here` (`s = (here / "cuda" / "sha512.cuh").read_text()`)
   becomes the root from `__file__` and the root-relative literal `verity/ml/flock/cuda/sha512.cuh`; the unused
   `here` is deleted; your test covers that path too. proofs leaves this file alone (its PR is the other four
   files, 15 joins).
2. Before the push: the changed lines' test under the ONE map (all three paths exist after it) and the test file
   itself passing on your VM. Push to `cursor/kernels-flock-kernel-path-5aee`.
3. Mark the new head by label at once (`research data label pr:8619@NEWSHA ready true --by kernels`), on the
   strength of the test file passing and the earlier 50 suites at the old head; update the PR's body (the new line,
   the one-map test, that the old head's 50-suite run was at 611e0aae98 and which suites were rerun at the new
   head); then queue the two suites the files are inputs of (`verity-flock`, `repository`) in one run with
   `--mem-gb 32` on a free pool and report its verdict when it ends. If that run fails, tell the lead at once: the
   mark is withdrawn only with ci.
4. Comment the new head on HASH-102 the moment it is marked; the lead says it on VLLM-265.
Then the rerun on the one-map branch, as in the section above.

## Tue 10 Nov 10:47Z: the rerun's target is vllm's FIXED head, not 539f956d62 (vllm, VLLM-265 10:45Z)

vllm confirmed your finding and found its class: two `[tokens]` rows of the one map differ from the per-move maps',
both Flock's: `verity/ml/flock/` (one map `verity/flock/`, map 04 `verity/flock/prover/`) and `verity/ml/flock/pod/`
(one map `verity/flock/`, map 05 `verity/flock/pod/`), which was not in your list. vllm is fixing the one map so each
such key takes its move's value, and posts the fixed head on VLLM-265 and TOP-394. So:

- **The #8619 commit still goes now** (top's word stands; root-anchored, it holds under either row).
- **Rerun the scan on the fixed head**, once it is posted: watch `origin/cursor/vllm-maps-3012` and VLLM-265's
  newest comments; the lead also writes the SHA here. On it say: whether the 16 Flock joins resolve again, what
  the `pod/` row did to pod scripts' own-directory anchors before and after, and the lines by owner as briefed
  (vllm puts them in the layout PR's hand list by your note's path, so keep `joins_missed.json` for that head in a
  directory named for it).
- Three `[tokens]` rows exist only in the one map (`verity/core/protocols/network/`, `benchmarks/bandwidth/`,
  `examples/`): say what each does to a bare `Path(__file__).parent` in a file under it.

## Tue 10 Nov 11:01Z: WITHDRAWN at 11:02Z (see the last section). DO NOT PUSH TO #8619. [was: the go expires at 11:15Z]

At 11:01Z #8619's head is still 611e0aae98, so you had not read the go of 10:46Z. A push is safe only while #8619
is in no candidate, and a new candidate can take it once one of the two in flight ends (about 11:30Z). A push to a
member drops its candidate: 17 pull requests and 40 minutes of check (ci, CI-489 10:50Z).

- **If `date -u` says before 11:15Z when you read this:** do the one commit, push, mark by label, comment the new
  head on HASH-102, as in the GO section. Nothing else first.
- **If it is 11:15Z or later: DO NOT PUSH to #8619, and do not mark anything.** Keep the patch beside your note
  (`tools/kmove/8619_followup.patch`) with its one-map test, comment on HASH-102 that it was not pushed, and go on
  to the rerun on vllm's fixed head. The line is then covered by vllm's fix of the one map's row, or by the
  stager's second commit, and the lead says so on VLLM-265.
From now on read this file every 10 minutes, not at each 30-minute checkpoint: tonight a message can be worth a
candidate.

## Tue 10 Nov 11:02Z: DO NOT PUSH TO #8619. The go is withdrawn (top, VLLM-265 11:01Z). This overrides every section above

top withdrew its yes, for two reasons:

- **#8619 has its land verdict** at `611e0aae98` (review-4, 10:55Z). A push loses it.
- **The line is no longer needed.** vllm's fixed row sends `verity/ml/flock/` to `verity/flock/prover/`, so the bare
  anchor at `cuda_sha512_patch.py:16` is right after the move, and the header's join is already root-anchored in
  #8619's first commit.

So: **push nothing to `cursor/kernels-flock-kernel-path-5aee`, mark nothing, label nothing.** If you already pushed,
STOP and tell the lead at once on HASH-102 with the new SHA and the time: do not try to undo it yourself. If you
prepared the patch, keep it beside your note and say so; it is not used.

What is still yours, in order:
1. The rerun of the scan on vllm's fixed head, when it is posted (VLLM-265 / TOP-394; the lead writes the SHA here).
   **Say first whether any join in `cuda_sha512_patch.py` is still broken there**: top asked for exactly that.
2. Nothing else until the lead writes here. Read this file every 10 minutes.

## Tue 10 Nov 11:04Z: the fixed head is `0d9a7b057e091e6c10293f8d68dae65f28fbcc06`. Rerun the scan on it now

vllm (TOP-394, 11:02Z): the one map's fixed head, #8634, pushed 10:56Z; its dry run is running and it is not marked.
Its diff from `539f956d62`: `"verity/ml/flock/" = "verity/flock/prover/"`, `"verity/ml/flock/pod/" = "verity/flock/pod/"`
and 59 `[split]` rows; no other row. (Still: push nothing to #8619.)

1. **The scan on that head** (the one map in `tools/move/layout_one_map.toml`, run with the tool of that commit).
   One comment on HASH-102, at most 200 words, which the lead posts on TOP-394 as it is:
   - first line: whether any join in `cuda_sha512_patch.py`, and in Flock's other four files, is still broken;
   - a table by owner: lines, files, how many are in tests the check runs and how many in scripts or builds it
     never runs;
   - the three or four breaks that stop a build or a served path, one line each;
   - the rule every owner fixes its lines by, in one sentence (a `parents[N]` anchor to the root plus the whole
     root-relative path as one literal, both of which `layout.py` rewrites);
   - the path in the notes of the per-line list for THIS head (a directory named for it), which vllm puts in the
     layout PR's hand list;
   - memory-accounting's stale pin (`benchmarks/pous/p2_v1/serve.py:49`, through `:274`) if it is still stale there.
2. **Then kernels' own red, if it is still red on that head:** `benchmarks/torch_gpu/commit_slots/tests/test_slots.py:26`.
   ONE new pull request cut from `origin/main` (it contains none of kernels' marked ones): the join written as a
   `parents[N]` anchor to the root plus `verity/ml/commitments/tests/fixtures/layouts` as one literal, shown to hold
   before the move and after the one map; its suite (`verity-commit-slots`) in one run, `--mem-gb 32`, a free pool;
   marked by label (`research data label pr:N@SHA ready true --by kernels`), said in its body; `Refs HASH-102`,
   `Goal kernels/code-at-overhaul-bar`. After its mark, nothing more is pushed to it.
3. Kernels' suites on the moved tree of that head, one run, as you did on `9ece`: say which reds are the read
   guard's alone and which are not.
End your turn when 1 to 3 are posted, or when a run of 2 or 3 would wait more than 40 minutes: name it and the lead
resumes you.

## Tue 10 Nov 11:12Z: #8619 stays at `acd82e7861`. Leave it exactly as it is

Thank you for saying it at once. top read your commit and gave its land verdict at the new head (11:11Z); it was in
no candidate, so nothing was dropped. The lead read the diff too. So:

- **No revert, no reset, no body change, no label change.** The body you wrote describes the head: keep it.
- **Let `r20261110-110846-f4e8` finish** and comment its verdict lines on HASH-102. If either suite is red, say so
  first, in bold: the lead tells top.
- **Push nothing more to #8619, whatever happens.**
- **From now, before any push, mark or label: fetch and read this file in the same minute.** A go here that is not
  followed by a later section is still a go; a later section always wins.
Then go on: the scan on `0d9a7b057e` (with the script as at the new head, since that is what lands), the
`test_slots.py:26` pull request, kernels' suites on that moved tree.

## Tue 10 Nov 11:19Z: compute-research posted the Python half of the scan (TOP-394, 11:18Z). Yours says what theirs cannot

On the fixed head `0d9a7b057e`, compute-research lists 96 path expressions built from `__file__` in 66 Python files
that the move breaks, by owner, each line in the comment and in `art:e7ea36a29e3332975a6ab7c1abe2abfbd9b508848422e30c1281af4767cd9f0d`.
Kernels' one is `benchmarks/torch_gpu/commit_slots/tests/test_slots.py:26`, the same you found: its pull request is
still yours (item 2 of the 11:04Z section). Its limits, by its own words: Python only (no shell script, Lean file
or TOML path), no path made by string formatting or passed through a function, no suite run, and its base lacks
#8619 (so it still lists `cuda_sha512_patch.py:21`).

So your comment for TOP-394 does not repeat its list. In at most 200 words:
1. first line: on that head, with #8619 at `acd82e7861`, is any join in Flock's five files still broken;
2. **what its scan cannot see and yours does:** shell scripts (defaults, `cd`, `$VAR/..` joins), `#include` and `-I`,
   paths in `.toml`, `.json`, Makefiles and `build.rs`, by owner, with how many stop a build or a pod job the check
   never runs; name the three or four that matter most, one line each;
3. where the two scans disagree on a Python line (yours has it and theirs does not, or the reverse), by count, with
   the reason if you can see it;
4. stale recorded digests by owner (memory-accounting's `serve.py:49` among them), and kernels' suites on that
   moved tree: which reds are the read guard's alone;
5. the path of your per-line list for this head.

## Tue 10 Nov 16:09Z: HASH-105's clock moved: the layout may be on `main` at 19:15Z. Report by 18:00Z, rows as they come

top (TOP-394, 16:09Z): the loop builds no new candidate; staging D (the one map, lean's map, the import-order commit
and the 267 hand rows of 16:00Z, on candidate `034a1702664b`) is the layout's pull request if its three check shards
pass, about 18:30Z; the layout is then on `main` about 19:15Z. Hand rows are closed: a row now goes in only to fix a
red of D's, on top's word.

- **#8619 and #8650 land before the layout** (both are in `034a1702664b`; read by the lead). #8521 and #8542 are in
  no candidate and cross it.
- **Keep `af72` as your "after" tree** for what is built and running; do not start over. Say in your table that D adds
  lean's move of `Security/`, which no kernel build reads [say if one does]. The newest staging on origin,
  `cursor/top-layout-staged-eb0b` at d833bd4b05, already has kernels' three hand rows applied (read by the lead).
- **A break of the move's is worth most before 18:30Z.** Post each one on HASH-105 the minute you have it, with the
  line and the fix; the lead takes it to top. The final comment for TOP-394 is due at 18:00Z, with what was not run.
- **Then HASH-107, sooner than planned:** when top posts the base and the two commands (about 19:15Z), cross
  kernels' heads that have no worker of their own: #8521, #8542 (marked: a land verdict carries by ci's #8630, a
  mark alone is set again at the new head), #8573, #7762, #7970. #8662 (KP2), KSLICE's two and KFORK's cross with
  their own workers. The lead resumes you for it if your turn has ended.

## Tue 10 Nov 16:27Z: cores 12-23 are quiet from 18:55Z to 19:35Z. Be off them by 18:45Z

vllm has cores 22-23 of node A from 19:00Z to 19:30Z for a timing run on one of cards 0 to 3 (INF-659; the calendar
says so), and kernels promised that nothing of kernels' runs on cores 12-23 in that half hour. Your HASH-105 report
is due 18:00Z: have no run alive on card 4 or on cores 12-23 after 18:45Z, and launch none there before 19:35Z. From
20:00Z the card and cores are KP2's for an hour. If a run of yours would cross 18:45Z, give it an end time before it
or do not start it, and say on HASH-105 what was not run.

## Tue 10 Nov 17:39Z: HASH-105 is closed. BREAK 1 goes with the archive PR; next is HASH-107, on the lead's resume

- **Closed at 17:38Z**, your table relayed to top on TOP-394 (17:38Z) before the cut. The lead read two of your
  runs' logs (…05b7, …3ec5): they say what the table says.
- **BREAK 1 needs no rows:** `units` is on kernels' archive list (KSLICE's HASH-104, 21 files), and its archive PR
  after the cut closes the break. The 11 rows stay in the notes, not applied, and nothing was asked of top.
- **Flock's refused live session:** the refusal's rule is on `main` before the move
  (`verity/core/service/flock/Flock/ProvedScope.lean:318`), so it reads as standing; whether that sweep's session
  was refused before the move stays not run. Launch nothing for it.
- **Card 4 is not yours any more** (the calendar's row, 17:38Z). Launch nothing on node A.
- **HASH-107:** the lead resumes you when top posts the base commit and the two commands (about 19:15Z) and top's
  count allows. Then, as the 16:09Z section says: `git merge BASE` first, then `tools/move/restack.py`, for #8521,
  #8542, #8573, #7762 and #7970; the suites each change touches at the new head, in named runs; each new head on
  HASH-107. Mark nothing. Not #8662 (KP2's), #8666 and #8669 (KSLICE's), #8671 (KFORK's).

## Tue 10 Nov 17:50Z: the layout does not land today. HASH-107 is off; expect no resume for it

top, TOP-394 17:39Z: Daniel wants the target tree changed first; the map changes once and every branch crosses one
time, later. So nobody crosses today, and HASH-107 goes back to the backlog until top names the new map's date. Your
HASH-105 result stands as the rehearsal: a kernel built from a moved tree computes the same values, with other
device code where an include path moves. The same job runs again on the endpoint Daniel signs, from `kmove-jobs/`.
Do not delete or rewrite any hand-row file in the notes. Nothing is asked of you now.

## Tue 10 Nov 23:16Z: the move is on again TONIGHT. HASH-108 is yours: prepare now, the H200 run after it lands

Daniel signed the architecture at 21:01Z; top moves the files in one change, due on `main` at about 03:45Z Wed
(TOP-398; `inbox/RULES.md`, "The queue, today", changed at 23:16Z). HASH-107 is back on, after this. Read HASH-108's
brief first.

**Part 1, now, no GPU; report on HASH-108 by 01:30Z.** On `cursor/top-layout-staged-e15c` (`50f940c4f285`; if top
posts a later staged head on TOP-398, use that and say which):
- the static scan of HASH-102's first part again: every load of a moved file from a kernel source, a build script
  or a pin, and whether the mover wrote it right. Tonight's tree differs from this morning's where it matters to
  us: Flock's GPU prover is at `verity/protocols/flock/prover/` (its patch script anchors at `parents[4]`, and
  `cuda/sha512.cuh` includes `../../../../definitions/kernels/sha512.cuh`: the lead read both, built neither); the
  headers stay at `verity/definitions/kernels/`; the adapter is `adapters/vllm/`; the old example is
  `archive/examples/h200/`.
- the ship builds on a CPU node from that tree (`p2_stage`, `attn_bf16`, `s8_linear`), and each cubin against this
  morning's;
- `kmove-jobs/` made ready for the new paths, so the H200 job starts the minute the commit is posted.
- **Known, do not spend time proving again:** `units/build.sh` reads the archived demo's `replay.py`.
  `benchmarks/hash/dither/h1v3_ref.py:24`: the mover turned the fallback `verity.core.protocols.pouw.circuit` into
  `verity.protocols.pouw.verifier.circuit`, which no tree has, so the script fails under a pinned older tree
  (`sbench.sh`'s `H1V3_SHA`, `J10R_SHA`). The lead put no hand row in for it (told top, TOP-398 23:08Z).
- If you find a break that stops a build the check runs, say it on HASH-108 at once with the `[[line]]` row: top
  closes the hand-written lines at 00:15Z, and a late one moves the whole move 90 minutes.

**Part 2, when the lead writes here that the move is on `main`, with its commit.** Card 4, cores 12-23, Wed 04:00Z to
07:30Z (the calendar's row; KSTEADY has card 4 until 04:00Z, so not before). Each kernel of HASH-105's table from
that commit, its own check, the same table, on HASH-108 by 08:00Z. Off cores 12-23 by 07:30Z: vllm has cores 0-23
from 08:00Z.

**Part 3, after part 2 (HASH-107 and one small pull request), by Thu 12 Nov:**
- `h1v3_ref.py`: a pull request on the moved `main` that tries the tree's own name, then the two older ones, with
  the older names written so that no mover rewrites them and no import lint reads them as live imports; its suites
  in one named run. The lead marks.
- kernels' open heads across the move, as HASH-107 says, each only if it has not merged.

## Wed 11 Nov 00:25Z: `archive/` is deleted, and `units/` with it: leave `units` out of part 2

Daniel, 00:10Z: `archive/` is deleted (top's worker, a pull request on the moved tree). Kernels' dead files,
`torch_gpu/units/` among them, are deleted too, not archived (HASH-104; KSLICE prepares that pull request tonight).
- **Part 2:** do not build or check `units` on the H200. Its row in the table says "not run: deleted by HASH-104".
  The other kernels as written. Your run …f8c1 (the units' ship on the staged tree with the replay's imports
  rewritten: rc 0, both cross-checks agree) stands as its last check.
- **The 11 rows for the archive's `replay.py`** are not sent to anyone. Leave the files in the notes.
- **Part 3** is unchanged (`h1v3_ref.py`, then the open heads across the move). Write no path under `archive/`.
- The cap is 150 agents until the move is on `main`, then 100 again (`inbox/RULES.md`): when part 1 is reported,
  end your turn; the lead resumes you for part 2 with the commit.

## Wed 11 Nov 00:45Z: part 1 is read and stands. Two decisions, then you wait for the landed commit

- **Read by the lead:** …ac44 (tree `b2b4d3ea55cd`: 11 suites, 11 passed, your counts) and …0e39 (same tree: `p2lib`,
  `attn`, `s8` rc 0; `libp2v1.2.sm_90a.cubin` is `1383d2da769e…`). Said on HASH-108.
- **`units` is out of part 2** (the 00:25Z section above, written while you were mid-turn): no ship build of it on
  cpu-1, and take `units` out of the step group `dither,units,s8,exactlin,j10r,attn` in `part2.md`. Its row says
  "not run: deleted by HASH-104". Send the eleven rows to nobody.
- **`sbench.sh:204`: yes, in part 3's pull request with `h1v3_ref.py`.** Three things in it:
  1. the path asked of the pinned commit is the one that commit holds, written so no mover rewrites it (as for
     `h1v3_ref.py`: a pinned tree keeps its own layout, so its paths are data, not paths of this tree);
  2. say from the code what `dither-1s-credit` does today when the gate file is missing (`sbench.py:528`,
     `settings.need`): if it refuses, say so; if it runs without the gate, make it refuse, with the fix in the
     message. `sbench.sh` itself should stop, not print and go on, when that arm is asked for and the file is absent;
  3. one line from the store: every run of the `dither-1s-credit` arm since `426e09cf8e` (24 Oct, the rename that
     broke the path) and whether any is a labelled row. If one ran without its gate, that goes on HASH-108 before
     the pull request, not in it.
- **Part 2 waits for the commit.** top names the fixed staging at 01:05Z and marks the move after ci's pair is on
  `main` (about 01:30Z), so the landing is not before about 03:45Z and may be later. The lead writes the commit
  here and resumes you. Card 4 and cores 12-23 are yours 04:00Z to 07:30Z; the battery took about 100 minutes this
  morning, so a start after 05:45Z does not fit: if it comes to that the lead says which steps go first.

## Wed 11 Nov 03:36Z: THE MOVE IS ON `main`: `5c095bb4ab7012ba359e0a85f4ecf19892b99629`. Part 2 now

top and infra, 03:34Z: the move landed at 03:26Z as one pull request (#8769), green on its first check. The lead
read: `5c095bb4ab70` is `origin/main`'s head; the final staging `e81cb11cdd77` is its ancestor; kernels'
directories, the prover's and `verity/definitions/kernels` do not differ between the two (`git diff --stat`).

**Part 2, from `5c095bb4ab70`:**
- **Ships and card-less builds on `vy-nebius-cpu-1` now.** Then node A: **card 4, cores 12-23, Wed 04:00Z to
  07:30Z**, not a minute before 04:00Z (KSTEADY has the card until then) and off by 07:30Z. Your `part2.md` and
  `NOT_BEFORE` stand.
- **No `units`:** no ship build of it, and `units` out of the step group `dither,units,s8,exactlin,j10r,attn`.
  Its row reads "not run: deleted by HASH-104". (The 00:25Z and 00:45Z sections above.)
- **Beside you, both timed, do not disturb them:** compute-accounting on card 0 with cores 0-7 (CMP-149, 04:00Z to
  07:30Z), and KSTEADY's slots on cards 1, 2, 3, 5, 6, 7 with cores 24-95. Stay on cores 12-23 for everything,
  builds included; name both among what ran beside each step.
- **The table on HASH-108 by 08:00Z:** each kernel of HASH-105's table from this commit, its own check, pass or
  fail as before the move, and which came out as different machine code. Say first, before any launch on the node,
  that kernels' files at this commit are byte for byte the ones you read in part 1.

**Part 3, after part 2, by Thu 12 Nov (top's rules for crossing, TOP-398 03:34Z: read that comment):**
- **The agent cap is 100 again.** You are kernels' one worker for crossing; do not ask for another.
- **ci crosses every marked head itself** (#8521, #8542, #8573, #8666, #8669, #8679; its dry run is read about
  05:00Z). Push nothing to them. When ci has reported, list on HASH-107 which of kernels' crossed clean and which
  it stopped; cross by hand only a head it stopped or the lead names:
  `python3 tools/move/restack.py origin/B --move-commit 6581bcf197 --move-commit c1ca25a247 --onto origin/main
  --kept --no-pins` from a checkout of the moved `main`, then `git push origin ID:refs/heads/B`, no force.
  Before any merge on the moved tree, once: `git config merge.vllm-allowlist.driver 'python3
  adapters/vllm/tests/lint/merge_allowlist.py %O %A %B %P'`.
- **Not yours to cross:** #8662 (KP2's, at its merge), `cursor/kernels-witness-card-43ee` (KHSHARE's),
  `cursor/kernels-dead-files-0c41` (KSLICE's; it sits on the final staging, an ancestor of `main`).
- **The small pull request on the moved `main`:** `h1v3_ref.py`'s fallback (the 23:16Z section) and
  `sbench.sh:204` with its three points (the 00:45Z section). Its suites in one named run. The lead marks.
- After merging the moved `main` on your VM: `uv sync --all-packages --extra torch-cpu` (infra, 03:34Z).
