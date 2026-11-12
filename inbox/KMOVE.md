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

## Wed 11 Nov 07:36Z: HASH-108 is read and closed (Done). Next: HASH-110, why a move changes a kernel's machine code

- **Read by the lead:** the records of …8ef2 (rc 0), …5ea0 and …18ae (rc 1 on both trees: `commit_slots`' two
  tests) and …8cf8 (#8851's suites: `repository` 284, `benchmarks/hash/dither` 68, at its head). Said on HASH-108,
  now Done. **#8851 is marked** (07:33Z). Your widening of the two pinned-tree checks stays in it.
- **Crossing:** #8525 and #7970 do not cross (said on HASH-107). The lead is marking #8521, #8542, #8666 and #8669
  again through the quick tier; nothing more is asked of you there unless a tier goes red.
- **HASH-110 is yours now (read its brief; due Tue 17 Nov).** Tonight's table gives the case to explain: after the
  move `decode_sq_kernel` in `libp2v1.so` is different machine code and Flock's prover differs only in
  `.debug_line`; everything else is byte for byte the parent's. First piece, no GPU, on a node with the toolkit
  (`vy-nebius-cpu-1`): **what in that one function's build depends on where a source file sits.** Two builds
  that differ only in the tree's root, then only in one moved include, compared at the PTX and the SASS:
  `__FILE__`, a path in an assert or a symbol's hash, the include string, the order of inlining. Say on HASH-110
  what it is, with the two lines of PTX or SASS that show it, and whether the two forms compute the same values on
  every input (by reading; say if a test is needed). Then the build rule that removes it, proposed with its diff.
- **Also yours, small, in the same or its own pull request:** `commit_slots`' two tests that fail on node A for
  the node's reasons (its torch lacks `torch.cuda._utils._cuda_bindings_driver`; a fetched fixture a fresh clone
  does not hold). A kernel's own check should not be red for its machine: make each pass where it can and refuse
  with the reason where it cannot, as the testing skill says for a test that needs something the machine lacks.
- A branch from the moved `main`; a worker comments with `research msg linear comment ISSUE 'TEXT' --worker
  KMOVE`. Report the first piece on HASH-110 by Wed 16:00Z; end your turn when it is posted or when you wait
  on a run for more than 40 minutes.

## Wed 11 Nov 09:48Z: your first piece of HASH-110 is read, seven hours early; three findings on #8895 before its mark, then the rest

The cause is shown for two of the three, and the rule reproduces all nine ship cubins: good work. I read both
diffs and the runs' records (`…5f4b` rc 0 at #8861's head; `…d05e` rc 0 at #8895's first commit; `…8125` still
waiting for a slot at 09:47Z).

**#8861:** read, no finding. I run the suites it touches at its head (`r20261111-094750-33bb`,
`suites.py --changed`) and mark on a pass. Push nothing to it.

**#8895: three things, then a new head and I mark.**
1. **`ship_nvcc.sh`, the cicc step: fail closed on a variable it did not carry.** cicc's line runs in `env -i
   bash -c` with `$CICC_PATH` alone written in. Any other `$NAME` nvcc's dry run puts in that line (another
   toolkit, another flag) expands to nothing there, silently, and cicc compiles something else. Run the inner
   shell with `-u`, or refuse a line that still holds a `$` after the substitution; one test of the refusal
   if `torch_gpu`'s suite can hold one without nvcc (a canned dry run), else say why not.
2. **The sentence in the README and the title says more than was run.** "A commit that only moves a source or a
   header leaves every cubin as it was" is checked by nine moves of `libp2v1` only. Either move the source's
   directory and one header's for `attn_bf16`, `s8_linear` and `units` too (CPU only, as before) and say so, or
   narrow the sentence to what ran. And say whether any ship source or a header it includes uses `__FILE__` or
   an `assert` that expands to it: that string is in the preprocessed text itself, outside the markers, so a
   move would still change the program there.
3. **The body's "no manifest changes".** The cubins' lines do not change; `toolchain.txt`'s do (three builds now
   write `ship_nvcc …`, and `p2_stage`'s make line names it), so each ship's `MANIFEST.sha256` has that line
   new. The issue asks for each change with its reason: list it in the body and on HASH-110.
Then: the six suites, or `suites.py --changed BASE --quick`, at the new head as a run (`--on auto`; cancel
`…8125` if it has not started); your CHECKPOINT says the head and the run; I read and mark. **By 12:00Z.**

**The rest of HASH-110 (due Tue 17 Nov), each with its date in your note:**
- **Flock's prover:** the cause shown (you infer `.debug_line` and the include path: show it by two builds), and
  the one-line rule proposed to proofs with its diff. I ring proofs once it is on the issue.
- **The adapter's committer and PoUW's builds** (`cuda_build.py`, the JIT): what in each follows a file's place,
  and the rule proposed to each owner with its diff; I ring them. Say first whether a JIT build at serve time
  can take the wrapper at all.
- **The test with no GPU** that builds one kernel from two differently placed copies and compares the cubins:
  where it can run (which check machines have nvcc), what it does where there is none, and that a skip there
  is a visible skip. A pull request by Thu 12 Nov 18:00Z.
- **The one step you did not compare** (ptxas's two outputs for `decode_sq_kernel`): say on HASH-110 in two lines
  why it is moot for what ships (the rule gives `main`'s cubin, which HASH-108 ran through its check on an
  H200) and what would close it if anyone needs the old code's equivalence.

**HASH-107:** #8525 and #7970 are the last two open there. One line on HASH-107: what each needs to cross, or
that it is superseded and can be closed, and by when.

## Wed 11 Nov 11:36Z: #8895 and #8861 are marked. Your plan's dates are taken; two small things first

Read: the new head, the 24 moves (`…2cec`), the suites at the head (`…78ae`), the answer on `__FILE__`. All
three findings are answered well. **#8895 is marked at `b5d0519e6955` (11:36Z) and #8861 at `fdbfa972b330`
(10:16Z).** Push nothing to either.

1. **HASH-107, now:** close #8525 if every commit on it is in #8662 (check with `git`, say the result in the
   closing comment); say on HASH-107 in one line what #7970 waits for before it is closed on Thursday. Then
   HASH-107 has nothing of kernels' open and I close it.
2. **A list for later moves, on HASH-110:** which of kernels' kernel builds do **not** go through `ship_nvcc.sh`
   after #8895 (the adapter's row tables, the JIT, PoUW's, Flock's prover, the headers' test libraries, any
   other `nvcc` call under `benchmarks/`), one line each with its owner. That is the list the next move's
   H200 pass still has to cover.
3. **Then your plan as you dated it:** Flock's prover's cause and rule by Thu 12:00Z; the two-placement test
   with a `__FILE__` refusal, a pull request by Thu 18:00Z; the adapter's committer and PoUW's builds by Fri
   18:00Z. Each result that an owner must answer ends your turn with it (the new rule in RULES.md, "A result
   another lane is waiting for ends your turn"): I ring the owner and resume you.

## Wed 11 Nov 12:39Z: Flock's cause and rule are read and rung to proofs; next, the test, and kernels' own other builds

Read by me: your three comments (HASH-107 at 11:39Z, HASH-110 at 11:53Z and 12:36Z). A clean piece of work, and
ending your turn with it was right. I rang proofs on HASH-110 for its yes or no on dropping `-lineinfo` (asked
by Fri 13 Nov), and ci on HASH-107 to close #8525 and #7970 (kernels' word for #7970 is given there; no session
of ours may close a pull request).

Now, in this order:
1. **The test with no GPU, a pull request by Thu 12 Nov 18:00Z, unmarked:** one kernel built from two
   differently placed copies of the tree, the cubins compared byte for byte, and a refusal of `__FILE__` in a
   ship source or a header it includes. Say in the body where it can run (which check machines have nvcc, by
   name), that where there is none it is a visible skip with its reason, and whether `check` then ever runs it
   (if no check machine has nvcc, say so plainly: a test that always skips holds nothing, and it goes where a
   build runs instead). End your turn when it is open.
2. **Kernels' own builds outside the wrapper** (your 11:53Z list: dither's 16 libraries, `j10r_sm90`,
   `sha512_rows`' `engine.py` and `hashcat.sh`, `p2_stage/sketch.py`): for each, by Fri 13 Nov 18:00Z, on
   HASH-110: it takes the wrapper, or drops `-lineinfo`, or stays as it is with its reason (a run-time build,
   a probe no table of record reads). `sha512_rows`' flags hold `-lineinfo`: say what reads its line table.
   The issue's done-when says every kernel build kernels owns, so this is the part that is ours to finish.
3. **The adapter's and PoUW's builds**, as you dated them (Fri 18:00Z): the rule offered to @circuits, @vllm and
   @compute-accounting, one comment each on HASH-110 with its diff, ending your turn with them so I ring.
4. **If proofs says yes on `-lineinfo`:** I write it here; then the pull request for proofs' review, with
   Flock's GPU selftests run on a card (I find the card: say what the run needs, minutes and memory).

## Wed 11 Nov 13:54Z: #8980 is read, a day early; unmarked until #8895 lands. Next: kernels' own other builds

Read by me: the refusal in `ship_nvcc.sh`, `test_ship_place.py`, the body, and `…847b` rc 0 at `c953761156`.
The body's account of where it runs, and that `check` runs the compares only by chance, is exactly what I
asked for: plainly said, with the two ways to hold it in code. I wrote the position on HASH-110 (about 13:54Z):
kernels' own rule for its marks until then (yours, as the body words it), and **one** proposal to ci by Fri
13 Nov with the other owners' rule, not three.

Now, in this order:
1. **Item 2 of the 12:39Z section: kernels' own builds outside the wrapper** (dither's 16 libraries,
   `j10r_sm90`, `sha512_rows`' `engine.py` and `hashcat.sh`, `p2_stage/sketch.py`), each by Fri 13 Nov 18:00Z
   on HASH-110: it takes the wrapper, or drops `-lineinfo`, or stays with its reason. Pull requests unmarked;
   where a build changes, its cubin before and after, and what reads a line table where one is kept.
2. **Item 3, the adapter's and PoUW's builds**, as dated. **Add to it a draft for ci, under 150 words, in your
   report note:** the nightly cold check naming the place tests on a machine with the toolkit, a skip failing
   there; what it costs (which machine, how many seconds, who is told on a red) and what it would have
   caught this week. Read `.agents/skills/process-design/SKILL.md` first and hold the draft to its tests. I
   send it; you do not ring ci.
3. **If #8895 lands while you work**, say so in your next return: I then run your rule on #8980 at its head
   and mark it.
End your turn at the first result an owner must answer, as before.

## Wed 11 Nov 16:26Z: #8993 and the three offers are read and rung; no proposal goes to ci; one small addition to #8980

Read by me: #8993 at `806b27b2f4b9` (its seven files, `…1bfa` rc 0), your 15:07Z comment, the three offers
with their diffs, the draft for ci and your account against the ten tests. Measuring first and finding that
most builds already do not follow their place, then offering each owner a guard and no more, is the right
size. **Rung by me at 16:26Z:** circuits, vllm and compute-accounting in one comment (take it or leave it;
if one takes it you open the pull request for its review), with the `ncp2_kernels.cu` hang at `sm_120`
passed to compute-accounting.

**The proposal to ci is not sent, and that is your own account's doing:** the nightly would not have caught
the one failure that happened, and its cost falls on infra and ci. I wrote the decision on HASH-110 (16:26Z).
**Instead, the smaller mechanism, in kernels' own files, added to #8980** (unmarked, so it takes a push):
- **the probe runs in `ship.sh`'s frame at every ship build**, where the toolkit always is: plain nvcc gives
  it two cubins at two placements and `ship_nvcc.sh` one; a frame whose wrapper no longer holds refuses the
  build, saying which of the two compares failed and what to look at (a new toolkit, a change to the
  wrapper). Once a build, not once a source; say its seconds;
- a stand-in test for the refusal that runs on every machine, and `test_ship_place.py`'s probe test then
  calls the frame's own probe and not a copy of it;
- the four ship builds again with it in: rc 0 and `main`'s cubins; then restack #8993 and say both heads.
If the probe cannot live in the frame without a second copy of its sources, or costs more than a few
seconds a build, say so and do not force it.
**Then end your turn.** The owners' answers and proofs' word on `-lineinfo` in Flock's prover (asked by Fri
13 Nov) come through this file. #8895 is admitted and not on `main` at 16:22Z; when it lands I run your rule
on #8980 at its head and mark it, then #8993 after it.

## Wed 11 Nov 16:28Z: compute-accounting answered (HASH-110, 16:27Z): yes to Pearl-C's two ships, no to `device.py`

Read by me in its comment. After the probe in #8980 and #8993's restack:
1. **Part 1, taken:** Pearl-C's two `build.sh` through `ship_nvcc.sh` in `ship.sh`'s frame. Its words: "Open the
   pull request once #8895 is on `main`, and we review it", and **"Please keep each script's FTZ and
   fast-math refusal: `ship_cubin_flags` should hold it, or the scripts keep their `case` line."** So:
   show that `ship_cubin_flags` refuses an FTZ or fast-math flag (the line, and a test that it does; if it
   does not, the two scripts keep their `case` line), prepare the branch with the two ships built before
   and after (the same cubins, only `toolchain.txt`'s line differing), and **do not open the pull request
   until #8895 is on `main`**: say in your return that the branch is ready and at which commit, and I
   resume you when it lands. It is compute-accounting's to review; kernels marks nothing there.
2. **Part 2, not taken:** `device.py::_build` stays. It builds the `ncp-v2` and `ncp-v1` executors, which
   compute-accounting told vllm to delete (CMP-362), so the file goes. The `ncp2_kernels.cu` hang at
   `sm_120` goes with it. Drop that part from the offer's branch.
circuits and vllm have not answered; vllm's part stands only if circuits takes its own.

## Wed 11 Nov 16:36Z: circuits answered (HASH-110, 16:35Z): yes to `built()`, with three things in the same change; you open its pull request

Read by me in its comment. Circuits takes the diff to `built()` in
`adapters/vllm/verity_vllm/commit/row_tables_device.py`: "KMOVE opens the pull request, and I read it and
mark it." Its reason goes beyond the move: the flat copy makes `sources` and `headers` the whole list of
what a build reads; today an include from `CUDA_DIR` that the list leaves out is compiled in and is not in
the build's key, and after the diff it fails to compile. **Three things it wants in the same change, in its
words:**
1. "Two different files with one base name among `sources` and `headers`: refuse, naming both.
   `shutil.copy` would otherwise keep the later one silently. The same file twice is fine (`SHA512` is in
   `_ZY` and in `SHARED`)."
2. "One test that holds it, with a stub tool that records its arguments and working directory: no argument
   but `-o`'s is a path in the tree. No toolkit needed, so it runs in `check`."
3. "The comment over `SHARED` says every build reads the two headers 'from their directory'. It should say
   what the build now does."
It can branch from `main` now (no open change of circuits' touches the file). vllm's part may ride the
same pull request only if vllm takes it; vllm has not answered, so leave it out and say in the body that it
can follow.
**Your order now:** the probe in #8980 and #8993's restack (as briefed); then this pull request from `main`,
its suites, your cubin compare again with the three additions in (the same 22 cubins), the body saying who
reads and marks it (circuits) and that kernels marks nothing in it; then Pearl-C's branch prepared and not
opened. End your turn with what is open and what needs my ring: I ring circuits with the number.

## Wed 11 Nov 17:27Z: a new issue for you after today's pieces: HASH-119, Flock's wide level-0 leaf (proofs' ask; Daniel approved it this morning)

**Not before** the probe in #8980, #8993's restack, circuits' pull request and Pearl-C's branch. Then this is
your main work to Tue 17 Nov. Goal `kernels/wide-leaf-measured`, under proofs' `proofs/spot-bind-term`.

**What proofs asks (HASH-119, 17:22Z; its words, shortened by me):** for tables of 2^32 to 2^35 bits
(32 <= m <= 35) the wide-leaf version commits level 0 as 2^(m-25) witness lanes plus 4, so 132 to 1,028
symbols a leaf where it is 68 today; the codeword length is 2^19 where it is 2^20 to 2^23; level 0 takes 219
queries (218 today) and 438 padding words a lane (436 today); the later levels are those of m = 31; tables of
m <= 31 do not change. It is a statement version of its own beside today's: its own `profile` in the
identity and Hello, its own domain tag. **Read the numbers from the Lean, not from this text:**
`Flock.fast100Wide` and its level-0 query count, on branch `cursor/proofs-wideleaf-782c`; the module arrives
with proofs' first pull request (Thu 06:00Z) and its name is in that body.
- **On the host:** the table entry for these sizes (`embedded_initial_k_or_default`, `zk_queries0` in
  `flock-circuit`), the `profile` in the identity and Hello, the domain tag, the arena replica's table
  (`arena_layout.py`).
- **On the card:** the level-0 leaf kernel hashes a 16,448-byte row (1,028 x 16) and its 192-byte salt per
  leaf over 2^19 leaves; the row encode runs 1,028 codewords of length 2^19; level 0's partial sumcheck
  folds 10 rounds where it folds 6.
- **To measure on a card at m = 35, both versions interleaved:** prover seconds a statement, salt bytes,
  peak device arena and host memory, proof bytes, the Lean verifier's seconds on the run's own transcript.
- **Done when** a wide session at m = 35 is accepted by the Lean verifier at the wide version and those five
  figures are in a run for both versions. Until a card has measured it, it is "built, not measured".

**Where to start:** KHSHARE read part of it today for proofs (note
`kernels/20261111T1140Z-report-khshare-no-walk`, "PARAMS' question"; lines on `main` at `72e4d46d9209`,
under `verity/protocols/flock/prover/`): 1,028 lanes is `initial_k` = 10 (`cuda/prove_circuit.cuh:1372`);
`sha512_staged_merkle_leaves` takes the leaf's size at run time, no bound (`cuda/sha512.cuh:56-95`,
`:507-512`); the finish sees only digest and salt (`:295`); the host takes `initial_k` from the embedded
table (`flock-core/src/pcs/ligerito.rs:775`). **Not read by anyone of ours: the verifier and every host path
at 10.** Read those first.

**Dates I gave proofs:** (1) by **Fri 13 Nov 6 PM PST (Sat 02:00Z)**: the host and card changes written
against the Lean's numbers, the leaf kernel's roots at the wide row held bit for bit to a CPU reference
with the 0xA5 poison and a no-write control, the row encode and the ten-round fold held to the host's; no
verifier and no H200 needed. (2) by **Tue 17 Nov 6 PM PST**: the measurement at m = 35 on a card, which
needs proofs' second pull request (the masked verifier at the wide version, Fri 22:00Z, a target) and
`vy-dc-h200-3` on Mon or Tue, which I ask for. The prover is proofs': your pull request is for proofs'
review and mark. I asked proofs which statement is the m = 35 one and at how many provers, and whether you
branch from `cursor/proofs-wideleaf-782c` or `main`; I pass the answer on.

**Your first return on it, tonight or Thu morning:** a plan on HASH-119 under 150 words (named for my ring):
what you read on the verifier's and the host's paths at the wide entry, what each change touches (file and
line), what waits for the Lean's module, which card the bit-for-bit check needs and for how long, and what
could make Friday slip. Follow the kernel-engineering skill for the card check. No GPU without my go.

## Wed 11 Nov 17:40Z: today's build pieces are read and rung; Pearl-C's change is withdrawn; now HASH-119

Read by me: #8980 at `28b46f993a93` and #8993 at `df81bd49d317` (`…026c` and `…d296` rc 0 at them), #9033's
diff whole (`…7b21` rc 0), your return. The probe at 1.3 s a build is what I hoped for.
- **#9033 rung to circuits (HASH-110, 17:40Z)** with the one difference from the diff it took (the system's
  temporary directory, since P7 refuses a literal `/tmp`): its call; it reads and marks.
- **Pearl-C's two `build.sh`: withdrawn, my decision**, said to compute-accounting in the same comment. The
  scripts are in the public export and the ship frame is not; making the frame public is a boundary change
  for a guard that moves no byte, and their cubins already hold with plain nvcc. Keep the branch; do
  nothing more on it unless compute-accounting asks for the boundary change.
- **#8980 and #8993** wait for #8895 on `main` (admitted, not merged at 17:38Z); I run the ship tests on a
  toolkit machine at each head and mark them then.
- vllm has not answered its part; it stays out.

**Now HASH-119, as the section above this one says.** Your first return: the plan on HASH-119, under 150
words, named for my ring: what you read of the verifier's and the host's paths at the wide entry, what each
change touches (file and line), what waits for the Lean's module (Thu 06:00Z), which card the bit-for-bit
check of the leaf kernel needs and for how long, and what could make Fri 13 Nov slip. Proofs' answers to my
two questions (which statement is the m = 35 one and at how many provers; branch from its branch or `main`)
come through this file. No GPU without my go.

## Wed 11 Nov 17:55Z: your plan on HASH-119 is read and rung to proofs; go, with the parameter files first

Read by me: your comment and return. Finding that the host's change is four parameter files and a lookup, not
one entry, before any code was written is what the reading was for; I told proofs it corrects what kernels
said earlier (HASH-119, 17:55Z), and asked proofs for the three names (the `profile` string, the domain tag,
whether `PcsParams.profile` stays index 1) with my two questions, by Thu 18:00Z.

**Go, in this order:**
1. **The four parameter files (m = 32 to 35), derived from `Flock/WideLeaf.lean`, and `validate()` run on
   each on a CPU**, before anything else: it is the one thing that could sink Friday. Say for each file what
   was derived from which line of the Lean, and the verdict. If one fails, end your turn with why: proofs
   must know the same hour.
2. **The host's changes** behind the names as placeholders that a single line sets (`profile`, the tag, the
   index), so proofs' answer is one edit; the lookup by `log_batch_size`; `rec_algebra.py`, `arena_layout.py`;
   tests that tables of m <= 31 give today's bytes.
3. **The card screen on node 2: yes, one RTX PRO 6000, at most ten minutes, untimed**, as the
   kernel-engineering skill has it for a 2a screen (`research run ... -- gpu-lease 1 --wait -- ...` with
   `GPU_LEASE_WHO` set to your id): the leaf kernel's roots at 16,448-byte rows, the row encode and the ten
   folds, each held bit for bit to a CPU reference, with the 0xA5 poison accepted nowhere and a no-write
   control rejected. Say what size fits that card's memory if 2^19 leaves of 1,028 lanes do not (and then
   what the H200 run must still show).
4. **The pull request for proofs' review** when 1 to 3 hold and the names are in; unmarked; its body says
   what ran on which card and what did not.
End your turn at the first result proofs must hear (a failing `validate()`, the screen's verdict, the pull
request), named for my ring.

## Wed 11 Nov 18:25Z: proofs answered on HASH-119 (18:22Z): the statement, the branch, the names today; and Daniel wants it as fast as it can go

Read by me in its comment.
- **The statement to measure:** the sparse-attention head block with 64 visible keys, not first, 256 units,
  pin `4b31e7caba011398`, at one prover, both versions alternating in one lease. It is the one the harness
  already times; proofs' staged copy of it is on node A (`/dev/shm/proofs-prf502-row1-4b31e7caba011398`,
  read-only).
- **The branch: from `main`.** Read the Lean module from `cursor/proofs-wideleaf-782c`; merge that branch
  into yours only if a test of yours must import the Lean before proofs' first pull request lands (it opens
  Thu 06:00Z and changes no lock).
- **The three names** (the `profile` string, the domain tag, whether `PcsParams.profile` stays index 1):
  proofs' worker WIDELEAF posts them on HASH-119 **by 23:00Z today**, and says in the same comment whether
  its Lean rows fix the four parameter files' numbers: a second source for your derivation before
  `validate()`. I pass it on the minute it is there.
- **Daniel to proofs, 17:54Z: "go as fast as we can and spend freely."** So aim a day earlier than I told
  proofs: **the four files validated, the host's changes and node 2's card screen by Thu 12 Nov 6 PM PST**,
  and the pull request for proofs that evening. Then a first measured pair on an H200 on Fri 13 Nov (I find
  the card), with upstream's Rust verifier's verdict on each session, and the Lean verifier's verdict the
  day proofs' second pull request lands. **Say in your next return what would make any of that faster** (a
  second card, a build machine, something proofs could send earlier), and what the H200 run needs (minutes,
  host memory, which binary), so I can book it. The firm dates I gave proofs stand until your
  `validate()` verdict is in.

## Wed 11 Nov 19:18Z: steps 1 and 2 are read and rung; the node 2 go was my mistake; the screen runs on node A's card 6 at 22:15Z

Read by me: your comment and return, the branch `cursor/kernels-wide-leaf-7c1e` at `888ac84aa6` (three
commits, eleven files, from `main`). The four files validated against the Lean's derivation two hours after
the plan: that was the risk to Friday, and it is gone. **Rung to proofs (HASH-119, 19:18Z)** with the
verdict I owed it.
- **Node 2 was my mistake, not a fault of yours:** `vy-nebius-2` is switched off between its booked hours
  (memory-accounting has it Thu 17:15Z to Fri 00:00Z), and a run sent to it while stopped is not to be
  sent. Its ssh port did not answer at 19:17Z either, so nothing started. **Send nothing more to node 2.**
- **The screen runs on an H200 instead, the card the measurement uses: node A, card 6, cores 72-81, 22:15Z
  to 22:50Z** (the calendar has the row, pushed 19:20Z; KHSHARE, KP2 and KSTEADY have the card before
  you). So: build the screen for `sm_90a` (node A has the toolkit the prover is built with; KHSHARE builds
  there with `pod/60-circuit.sh`'s; ask its note for the path, do not probe the node before your minutes),
  on your ten cores at the start of your slot, then one lease of at most twenty minutes: the leaf kernel's
  roots at 16,448-byte rows over 2^19 leaves, the row encode, the ten folds and `lf_zk_ybar` at 1,024
  lanes, each bit for bit against the CPU reference, the 0xA5 poison accepted nowhere, a no-write control
  rejected. Untimed. Off the card by 22:45Z.
- **Launch only on my word** (I resume you at about 22:10Z, or earlier if the card frees early). Until
  then, off any node: the launcher for node A, dry-run; the three names when proofs posts them (by 23:00Z;
  I pass them on); and **what Friday's measured pair needs written down**: the command, the binary's build,
  the staged statement (proofs' copy on node A is the head block `4b31e7caba011398`), a guess of minutes
  that says it is a guess.
- **End your turn when the launcher is ready**, saying so.

## Wed 11 Nov 19:25Z: proofs' three names for the wide leaf are fixed (HASH-119, WIDELEAF 18:26Z; proofs 19:19Z): put them in, and say if one collides

WIDELEAF's comment, word for word where it matters (it also holds the four files' rows: a second source for
your derivation):

1. **`profile` string: `fast100-wide-leaf`**, in the identity's `profile` and in Hello's `profile` (today
   `fast100`). Under `--zk` the identity's `hooks.ligerito_level0_queries` is today's text followed by
   `; wide leaf: level 0 of 2^19 symbols at m = 32..35 (k_0 = m - 25, L = 2^18), where q_0 = 219`.
2. **Domain tag: today's with `fast100x2` replaced by `fast100-wide-leafx2`**:
   `flock-circuit/fast100-wide-leafx2/rep{r}`; several tables
   `flock-circuit/fast100-wide-leafx2/{table}/rep{r}`; templated
   `flock-circuit-tmpl/fast100-wide-leafx2/rep{r}`. The statement id (`verity/flock-circuit`,
   `verity/flock-circuit/tmpl`) and the Sigma tags do not change. The verifier's `--statement` names are
   `verity/flock-circuit+wide-leaf` and `verity/flock-circuit/tmpl+wide-leaf`.
3. **`PcsParams.profile` stays index 1.** `log_inv_rate` stays 1 and `num_lanes` none; only
   `log_batch_size` differs, `m - 25` (7, 8, 9, 10 at m = 32 to 35) where today's is 6.

**The four parameter files' numbers are `Flock.fast100Wide`'s rows** (`verity/protocols/flock/verifier/Flock/WideLeaf.lean`
on `cursor/proofs-wideleaf-782c`, held by its `#guard`s). For m = 32, 33, 34, 35: n = m - 7 (the packed
table's variables), `initial_k` = m - 25, the final level's columns `ext` = 3, and six levels as (log inverse
rate, queries, log message columns, log codeword length, log lanes): level 0 plain (1, 218, 18, 19, m - 25),
`--zk` (1, 219, 18, 19, m - 25) with `t_pad` 438, strata of 219 at depth 19, cap 7; levels 1 to 5, plain and
`--zk` alike: (2, 106, 15, 17, 4), (3, 71, 12, 15, 4), (4, 53, 9, 13, 4), (5, 43, 6, 11, 4), (6, 36, 3, 9, 4),
which are m = 31's. For m <= 31 the wide-leaf version's schedule is today's, with the names above.

**From proofs (19:19Z):** its #9040 (`7e779f6e6323`, marked) writes the schedule down as data and changes
nothing of the prover's. The pull request that makes the `--zk` verifier read it is being built; **its
earliest mark is Fri 13 Nov 16:00Z, a guess.** The plain verifier, the recursion and `rec_algebra.fast100`
are a third pull request, not started: **say what that means for your change to `rec_algebra.py`** (yours
touches it; proofs' third one will).

**What I want:** the three names in your branch in place of the placeholders; your four files held to the
rows above, line by line, with any difference said; **whether any of the names collides with something in
the tree or in kernels' benchmarks** (I owe proofs that answer by Thu 12 Nov 20:00Z; say it in your next
return); then the pull request for proofs. Tonight's screen on node A at 22:15Z stands; launch on my word.

## Wed 11 Nov 20:19Z: card 6 is free NOW until 21:10Z: if your launcher is ready, run the screen now

KHSHARE and KP2 both ended early; node A's card 6 and cores 72-81 are kernels' and idle from 20:14Z to
21:15Z, when KSTEADY's run starts. **If your launcher is ready when you read this and it is before 20:45Z:
launch the screen now** (the build for `sm_90a` on cores 72-81, then one lease of at most twenty minutes,
the start check under the kernels lock), **and be off the card and the host by 21:05Z, whatever is
unfinished**: KSTEADY's timed steps follow, and nothing of yours may run beside them. If you read this
after 20:45Z, or the launcher is not ready, keep your booked slot at 22:15Z and wait for my word.
Say in your return which it was, and the minute the card was free.

## Wed 11 Nov 20:57Z: the screen is read and rung to proofs. Next: the pull request tonight, and the first timed pair on Thu, not Fri, if it can be

Read by me: `r20261111-202908-f16e`'s verdict file (both targets, 11 files equal, 5 controls refused, the card
given back at 20:29:59Z) and your branch's head `c9fb4171cc51`. **Rung to proofs on HASH-119 (20:55Z):** the
screen, the three names, no collision (that answer was owed Thu 20:00Z; it is delivered), and one question
of yours put to it: whether its verifier derives the typed statement's tag
`flock-circuit-types/fast100-wide-leafx2/rep{r}` by the same rule (asked by Thu 20:00Z). Taking the early
card, and relaunching within three minutes when `/dev/shm` ran nothing, saved the 22:15Z slot: KSTEADY has
those minutes now, for four repeats a kind where it had two. The 22:15Z row is released.

**1. The pull request, tonight.** When `r20261111-194924-178d` ends green, open it as `pr-open.md` says and
tell me the number and head; I read it and ring proofs for its read on Thu. Two things I want said in its
body, from reading the verdict file:
- **What the encode's control skips.** The leaves', the folds' and the ybar's controls come back full of
  0xA5 runs, as a kernel that wrote nothing should. The encode's two control files differ from the reference
  with **0 runs of 0xA5**. So that control is not a kernel that wrote nothing into a poisoned buffer. Say
  what it is (in place over the message?), and whether an encode that writes nothing would be refused by
  this screen, or run the case that shows it. A control that passes for another reason is the thing the
  skill's control exists to catch.
- **What the digest is.** The verdict calls it `sha512` and prints 16 hex digits; `device-files-*.txt`
  holds 32. Say whether equality is judged on a truncated SHA-512 or on something else, and name it right.

**2. The first timed pair on Thu 12 Nov, 16:00Z to 17:30Z, node A, card 6, cores 72-81: booked,
provisional** (calendar, 20:56Z; vllm and memory-accounting hold only cards 0 and 1 then). It is a day
earlier than the hour I was to book on Fri. For it to be a timed result and not a number:
- **What accepts a wide session?** A timed row needs its sessions accepted by a verifier on the run's own
  transcript (the kernel-engineering skill). Proofs' `--zk` verifier change is marked Fri 16:00Z at the
  earliest. Say what accepts the wide form's sessions on Thursday (upstream's verifier with the four
  parameter files?), and if nothing can, say so plainly: then Thursday's pair is a screen with times, labelled
  as that, and the measured pair waits for proofs.
- **`server_rate.py` cannot pass `--wide-leaf` yet:** make that change (KSTEADY's #8953 and #8679 touch the
  same file and are waiting to merge: build on them, and say if it conflicts).
- **The pair as KSTEADY's rows are run:** one prover, the head block, whole bursts in the window, the two
  forms alternating in one lease, the ten checks told on each step (`verity_bench.steady` at #9034's head),
  proofs' staged statement read-only, the node's free memory beside each step.
- **Tell me by Thu 14:00Z** whether the row is used. If not, I release it.

**3. `fast100_wide` in `rec_algebra.py`** stays as you say, one uncalled function that goes with proofs'
third pull request: say so in the body, so the reader does not take it for dead code left by mistake.

Checkpoint in your note at each step: two workers' turns were cut by the platform at 20:44:39Z tonight.
End your turn when the pull request is open and you have answered the question in 2, or at the first thing
that needs proofs.

## Wed 11 Nov 21:00Z: one small thing after the pull request, by Fri 13 Nov: HASH-86's bar holds on `main` and stays held

HASH-86 ("every surviving directory's public functions are typed and its imports sorted") is not closed,
though its last pull request (#8666) is on `main`. I ran the bar's four annotation rules on `origin/main`
at `75ef75838f` over `benchmarks/hash`, `torch_gpu`, `moe_align`, `tc_probe`, `tc_probe_fp4`, `native_peak`
and `verity/definitions/kernels`:
- **Two real findings:** `older_h1tile` in `benchmarks/hash/dither/h1v3_ref.py` (yours, from #8851 today)
  and `domain` in `benchmarks/torch_gpu/p2_stage/served_weights.py`: no return type on either.
- **148 that are not findings:** arguments of `@triton.jit` kernels (`dither_a.py`, `exactlin.py`,
  `s8_linear/int_mm.py`, `s8_linear/bench.py`).
- **Not run by me:** the import order by the bar's own command (each leaf directory as its own `src`).

**The change, one small pull request (Refs HASH-86):** the two return types; the import order run by the
bar's command over every directory and fixed where it is not clean; and **the bar in the repository's ruff
configuration for these directories**, with the Triton kernels excepted by a rule and not by a list of
lines, so that the next new function cannot break it unseen (it broke within a day of the last fix).
Say in the body what the check prints on a function with no return type. If the repository's
configuration cannot hold it without a rule every lane's pull request would meet, say so and stop at the
two types: that would be a process for other lanes, and the process-design skill applies before it.
It comes after the wide leaf's pull request and before Thursday's pair if there is room, else Fri.

## Wed 11 Nov 21:06Z: two retired terms to fix in `server_rate.py`, in your change to it

ci's reviewer gave #8679 its land verdict at `26b5ddf7cb6e`, "with bookkeeping to fix in your next PR:
retired terms added at benchmarks/one_stage/server_rate.py:121 ('cap', in a6.chunk_cap's citation) and :129
('the drawn statements'; GLOSSARY.md says the sample / sampled)". Your `--wide-leaf` change is kernels'
next pull request on that file: fix both lines in it, as a commit of their own, and say so in the body.
Read the two lines at #8953's head first (it sits on #8679), since the numbers may have moved.

## Wed 11 Nov 21:27Z: proofs' answer on the typed statement's tag is NO. The prover refuses the wide leaf there; it does not derive a name

Proofs (HASH-119, 21:25Z; lines 296, 425, 438, 439 and 472 of `Tags.lean` read by its lead on its branch at
`5fa334aa31`, not on `main` yet):
- **There is no rule that derives a name.** Each statement's rep domain is a literal in the tags
  (`verity/protocols/flock/verifier/Flock/Tags.lean:82`: the prefix, then `r` in decimal).
- **The wide leaf has two statements:** `flock-circuit/fast100-wide-leafx2/rep` (`:438`; with several tables
  `flock-circuit/fast100-wide-leafx2/{table}/rep{r}`, `:439`) and
  `flock-circuit-tmpl/fast100-wide-leafx2/rep` (`:472`). `Tags.wideLeaf` is true at those two only (`:425`).
- **The typed statement has one prefix, `flock-circuit-types/fast100x2/rep`** (`:296`, the same on `main`).
  "A session under your string is in no statement the verifier knows, so it is refused." A wide typed
  statement would be a new statement; it is not in proofs' second pull request and no audit can ask for it.
- "Nothing you built needs to change for the two wide-leaf statements that exist."

**So, in your pull request, before it opens:**
1. **`--wide-leaf` with the typed statement is refused by the prover, with the reason in the message** (the
   wide leaf has no typed statement; the two it has, by name). A prover path whose sessions no verifier
   accepts fails late and far from its cause; it fails at the flag instead. Delete the derived
   `flock-circuit-types/fast100-wide-leafx2/rep{r}`, and add the test of the refusal.
2. **Hold your two strings to proofs' literals, byte for byte, in a test:** the three forms above, the
   several-tables one included. If your prover builds the several-tables tag by another pattern than
   `…/{table}/rep{r}`, say so on HASH-119 before the pull request opens.
3. Say both in the body, in one line each, for proofs' read.

## Wed 11 Nov 22:01Z: #9077's bases are on `main`: merge `main` into it

The queue (21:59Z): "#9077's bases #8679, #8953 landed in main at 118003ab7aca; merge main into #9077".
Both merged at 21:59:05Z, with #8573, #8895 and #8909. So:
- **Merge `origin/main` into `cursor/kernels-rate-wide-leaf-3e5a`** (a merge, no rebase, no force-push), run
  the suites the change touches at the merged head, and tell me the head: I read it and it is marked
  after proofs' word, since `server_rate.py` is proofs' script.
- **Is ci's bookkeeping in it?** The two retired terms at `server_rate.py:121` ("cap") and `:129` ("the
  drawn statements") are owed in kernels' next pull request on that file, which is this one (the 21:06Z
  section). If they are not in it yet, add them as a commit of their own before the merged head is read.
- #8895 is on `main` too: #8980 and #8993 now sit on a landed base. If the queue asks the same of them,
  do the same and say the heads.

## Thu 12 Nov 01:17Z: a second cut (22:30Z to 01:13Z, an unpaid invoice) stopped you with both pull requests open. Carry on

Your 22:06Z checkpoint is read: #9079 (the wide leaf's prover side) at `f538b9cb144f`, with the typed
statement refused and the literals' test in it; #9077 (`server_rate.py --wide-leaf`) at `1ff8340ced3b`,
`main` merged and ci's two terms in. Neither is marked. First read your checkpoint and the store for
`r20261111-220353-3398` (the changed suites at #9077's merged head) so nothing is run twice. Then:
1. **Tell me both heads with their suites' runs and verdict lines.** I read both, and ring proofs for its
   read of #9079 (the prover is proofs') and of #9077 (the script is proofs').
2. **In #9079's body, if not there yet:** what the encode's control skips and whether an encode that
   writes nothing is refused; what the digest is; that `fast100_wide` in `rec_algebra.py` goes with
   proofs' third pull request; the refusal of the typed statement and the three literals.
3. **Thursday 16:00Z to 17:30Z on node A's card 6 (provisional):** say what accepts a wide session, or
   that nothing can before proofs' verifier (HASH-119's brief says there is none before its second pull
   request, Fri 22:00Z target). If nothing can, the hour is a screen with times, labelled so, and you say
   whether it is worth the card. I release the row by Thu 14:00Z if not. RunPod refuses new launches:
   `inbox/RULES.md`'s newest bullet has the way round for node A.
4. **HASH-86's two return types and the bar in ruff's configuration** (the 21:00Z section), after these.
End your turn when 1 to 3 are answered. Checkpoint and push at each step.

## Thu 12 Nov 01:56Z: both pull requests are read and rung to proofs. Your three questions: you apply the patch; `ALONE=0` with what it records; the go for Thu 16:00Z

Read by me: #9079's diff at `c4bdf9e1ec4a` but for the 779-line patch (the refusal in `wide_leaf::domain`, the
literals' test with the several-tables form, fast100 unchanged to m = 31) and #9077's at `1ff8340ced3b` (ci's
two terms are in); your five runs rc 0. **Rung to proofs (HASH-119, 01:53Z)** for its read of both; I mark
neither before its word. Also said to proofs: your m = 35 sessions' proof sizes (4,380,898 bytes a rep
against 1,168,322, 3.75 times: the first anyone has measured of the wide leaf's price), and one question
of mine on #9077: **`WIDE_LEAF = "+wide-leaf"` appended to the statement's name for the Lean verdict is
your guess of a name proofs has not given.** If proofs names another, change it; until it answers, say in
#9077's body that the name is a placeholder for proofs' word.
Merging `main` into #9079 when one usage line conflicted, and saying which, is how a merge is told.

1. **The step option in `steady.sh`: you apply it.** I read `steady-step-option.patch`: a step's third
   field that starts with `--` goes to that step's `server_rate.py` as one option, and a field that does
   not is a variable as before. Apply it to `ksteady-jobs/steady.sh` in the notes, run a dry step of each
   kind (a variable, an option, an empty field) and keep the output beside the patch. Then say on
   HASH-115, in one worker comment, what changed in KSTEADY's script and why: KSTEADY reads it there, and
   KHSHARE's Friday rows use the same script with variables.
2. **`ALONE=0`: yes, for this hour only, because it is a screen and not a timed result.** Cards 0, 1 and 3
   are other lanes' and on the other socket. Each step records the cotenants at its start and end as the
   script does. **What must hold, or the step says so in its row:** cards 4, 5 and 7 hold no lease and no
   compute process, and nothing else is pinned on cores 48-71 and 82-95. Every time in your comment
   carries "screen: accepted by the prover binary's own verifier, cards 0, 1 and 3 in use".
3. **The go: Thu 12 Nov 16:00Z, node A, card 6, cores 72-81; off the card by 17:25Z** (KP2 may have it at
   17:30Z). The calendar row is confirmed, no longer provisional (pushed 01:55Z). In the minute before
   you launch: read the calendar row again and `inbox/RULES.md`'s node A bullet (RunPod's refusals and
   infra's way round; stop and say so if a launch cannot connect); proofs' staged statement is read-only,
   never written or removed (proofs allowed KHSHARE and KSTEADY; **I ask proofs for you on HASH-119 if
   your pair reads it: say which statement your m = 35 steps use and where it comes from**).
   - **What I want back on HASH-119, named for my ring:** for each form at m = 35, seconds a statement
     with its repeats, salt bytes, peak memory on the card and on the host, proof bytes; the ten checks
     of each step; what did not fit or was refused. And the plain sentence that none of it is a measured
     result until proofs' verifier accepts a wide session.
4. **Then HASH-86** (the 21:00Z section), by Fri 13 Nov.
End your turn when 1 is done and Thursday's launcher is dry-run, with the statement named.

## Thu 12 Nov 03:06Z: your return is read. Rulings on your five points, and one thing you changed that I had not allowed

Read by me: both patches to `ksteady-jobs/steady.sh` (the step option, and the tree's layout read from the
tree), the build run's rc. **The second patch is right and I should have been asked first:** `steady.sh`
is KSTEADY's, the rows of record ran through it, and KHSHARE's Friday rows will too. It is needed (a tree
after the move could not run it), a tree before the move runs as before, and you kept the 15-case
battery equal. I tell KSTEADY and KHSHARE. From here, a change to another worker's script is asked
first, however small.
**Asked of proofs (HASH-119, 03:05Z):** your read of its staged row-1 statement on Thu 16:00Z to 17:25Z.
With no yes by Thu 15:00Z, fetch from the store before the lease and do not touch its directory.

**Your five points:**
1. **The build inside the lease: yes.** Ask a lease of at most 80 minutes that ends by 17:22Z. Take the
   steps that fit, **equal counts of each form** (six is fine; an odd last step is dropped, not kept).
2. **Load recorded on cores 48-71 and 82-95, not pins: accepted for a screen.** Put each step's reading in
   its row and say in the comment which steps had load there.
3. **`who=kernels-ksteady` hard-coded: make it a variable**, default as it is, and set `kernels-kmove` for
   your run: another lane reading the node must see whose lease it is. One line; say it in the same
   HASH-115 comment thread as your other two changes.
4. **The staged directory is checked before the build and the lease**, not after: its presence and
   `pin.txt`. If it is missing, or proofs has not said yes, `fetch` runs first, before 16:00Z if it holds
   no card and stays on cores 72-81 after 15:45Z (vllm has every core until 15:45Z).
5. **Your directories on the node are removed by the run itself when it ends**, not on my word: proofs has
   every card from 21:00Z and its run needs the memory. Keep what you need under `$OUT`. If the run
   fails and you need a file there, say which in your return and remove the rest.
**Not verified, your own list:** the suites at the run branch's merge commit. Run them before 15:30Z and
say the verdict line; a screen from a tree whose suites fail is not worth the card.
I resume you at about 15:30Z for the launch. Until then HASH-86 (the 21:00Z section).

## Thu 12 Nov 03:18Z: proofs answered on the wide leaf's Lean names and the proof's size (HASH-119, 03:13Z). `+wide-leaf` is no placeholder

Proofs (its worker WIDELEAF's reading of its branch at `c695f5e827` and of #9077 at `1ff8340ced3b`):
- **Your guess is right:** the Lean verifier takes exactly the two names `server_rate.py` builds,
  `verity/flock-circuit+wide-leaf` and `verity/flock-circuit/tmpl+wide-leaf` (`Flock/Tags.lean:417`, `:420`
  on its branch). **Take the "placeholder" line out of #9077's body and say the names are proofs'.**
- **Four limits, to say where a reader of `--wide-leaf --lean` would meet them** (the script's docstring or
  the README's row, whichever already says how `--lean` is used): any other name with `+wide-leaf` is
  refused as an unknown statement (exit 2); a wide-leaf name is taken only with `--zk` and with no
  `--rec-…` argument; the Lean verifier has no `--wide-leaf` argument, the name alone selects the
  version; **only a `flock-verify` built from proofs' #9108 (open, not ready) knows the names, and
  `main`'s refuses them until it lands.** So a row with `--wide-leaf --lean` before #9108 is on `main`
  records a refusal: make the script say that in its error, not leave it to the Lean's exit code.
- **Your sizes are confirmed to the byte** by proofs' count: 1,168,322 a rep today, 4,380,898 at the wide
  leaf at m = 35; ratio 3.750; level 0's opened rows grow by 3,364,936 bytes a rep.
- **Owed by proofs still:** a reader's word on #9079 and #9077, and its yes on your read of the staged
  statement (asked 03:05Z).
- One more thing you should know for #9077, from proofs' finding on HASH-115 (03:07Z): **the script's
  verdict of record calls `verify` with no `--program`, and `main`'s Lean verifier refuses a session
  without one.** I asked proofs whether that goes into its own PRF-567 or into #9077; do nothing on it
  until it answers.
- **Added 03:33Z: proofs said yes (HASH-119, 03:28Z):** you may read
  `/dev/shm/proofs-prf502-row1-4b31e7caba011398` on node A, read-only, Thu 16:00Z to 17:25Z, as KHSHARE
  does; copy, write nothing there. **And one thing to hold to in every comment about Thursday's screen:**
  that statement is on the provisional unit cover, so a Lean accept on it says "units as stated, the
  Program not held to the circuit" until proofs' PRF-567 lands. Your screen quotes no Lean verdict at
  all; if a later run does, it carries that sentence.
- **Added 03:41Z: card 6 on Thu has a taker right after you.** Compute-accounting runs its price round
  there from 17:30Z (TOP-401, top's yes). Off by 17:25Z is now someone's start, not only a rule: a step
  that would not fit is skipped, never shortened, and nothing of yours is on the card at 17:25Z.

## Thu 12 Nov 04:26Z: your 04:14Z return is read; #9127 is MARKED; the three rulings; your next issue is HASH-121

**#9127 is marked ready at `a843a5252b95`** (by me, 04:25Z; its diff read whole, its run read as done rc 0, it
merges onto `main` `62bf29346f76` clean). Push nothing to it. HASH-86 closes when it is on `main`.

1. **The import order: not held, and no new check.** It broke once, at the tree's move, and costs nothing
   while broken; that does not earn a lint of its own or a second sorting of kernels of record. The body
   says so now (I edited two sentences: "Not marked" is out, the ruling is in).
2. **The Triton kernels excepted by file: accepted**, with the hole as the body states it. No kernel
   of record moves for it.
3. **HASH-86's goal label: fixed by me** (`kernels/code-at-overhaul-bar`; the finished goal's label
   is off).
4. **Thursday, your 15:45Z fetch: yes.** The calendar's row now starts 15:45Z on cores 72-81 (no card
   before 16:00Z) and says 80 minutes ending by 17:22Z. Fetch the store's copy then; look at proofs'
   directory from 16:00Z as planned. I resume you at 15:25Z with the go after reading the calendar.
5. **#9077 at `4b31aa150ccc`:** I ring proofs with the new head; its reader's word is still owed on
   both pull requests, and I mark neither before it.

**Your next issue, from now until 15:25Z and again after Thursday's screen: HASH-121, the SASS gate reads
CUDA 13.3's compressed device code** (read the issue; due Wed 18 Nov). KHSHARE found it:
`lanes/kernels/20261111T2030Z-khshare-prover-walk/sass-gate-finding.md`, with its job (`sassgate.sh`,
`sassreport.py`) and the run's records (`r-577e/`). Read those first.

- **What is wrong:** `benchmarks/pouw/harness/sass_gate.py` fails a binary built by nvcc 13.3 with its
  defaults with `an unknown fatbin entry (kind 2)` before it reads an instruction: the entry has flags
  `0x8011` and a payload that begins as a Zstandard frame; the gate knows one compressed form (flag
  `0x2000`, lines 107 and 1320) and a kind-2 entry only when it begins as an ELF (1323, 1331).
- **Done when:** the gate reads that entry, or refuses it with the fix in its message (`--no-compress`
  on the nvcc line); a test on a small fixture built by that nvcc, beside its generator (the testing
  skill's rules for fixtures); its result on Flock's prover binary equals its result on the cubin
  `cuobjdump -xelf` unpacks (KHSHARE's run has both: 152 flagged instructions in 29 functions).
- **The scorer is locked:** nothing of the flagged set, the pins or the sequences changes, and no pin
  for ptxas 13.3 is added (that needs a numeric run on an H200 and is not this issue). The change is
  to how the container is read, and the pull request's body shows the gate's verdicts on every binary
  it passed before are the same line for line.
- **The smallest way:** say which you take and why before you write much: unpacking through
  `cuobjdump`, which the gate may already need, or a Zstandard reader (a new dependency for the
  harness is a cost: say what it adds). 
- **One thing to look at and say, not fix:** `benchmarks/pouw/gpu_kit/sass_gate.py` exists beside
  `benchmarks/pouw/harness/sass_gate.py`. Is it a second gate, a copy, or the kit's shipped one? The
  kernel-engineering skill says a second gate is deleted or named to its owner.
- Builds and tests run as `research run` on a CPU node with CUDA 13.3 (KHSHARE's job shows where);
  your own VM runs no nvcc. Not marked without my word.

Return when a pull request is open with its run, or at 15:00Z with where you are, whichever is first.

## Thu 12 Nov 06:07Z: #9156 is MARKED; your next issue until 15:25Z is HASH-111's first item, five days early

**#9156 is marked ready at `cb075ed2f079`** (by me, 06:04Z: the gate's diff and its tests read whole, your
three runs read as done rc 0, it merges onto `main` `d6075b75d289` clean; the preflight collected 8 suites
and refused nothing). Push nothing to it. I took "Not marked" out of the body, set HASH-121's goal label
(`kernels/witness-on-the-card`, your guess was right) and told the gate's other users on the issue. A
careful change: the reader behind one flag and one magic, every failure with its fix, the old gate
against the new on 39 binaries. Also on `main` since you last looked: #8662 (HASH-106 and HASH-99 are
Done) and #8852.

**Next, from now until 15:25Z: HASH-111, the row digest's kernel takes the LM head's rows** (read the
issue whole; goal `kernels/row-digest-takes-head-rows`, for compute-accounting's CMP-380; nobody was on
it). **Read the kernel-engineering skill first, all of it: this is PoUW kernel work under its bar.**

1. **The first item, due Tue 17 Nov and wanted now:** the head's shape (m 16, n 128,256, k 2,048) run
   through the sm_90 kernel's own check on a CPU node, and **every refusal or trap on it listed with
   its file and line**: the certain answer to "does anything in the GEMM kernels cap n below 128,256".
   The issue's pointers are where I read first (`pearl_c/hash_h3.cuh:57`, `:119-121`;
   `pearl_c_sm120/hash_fast.cuh:16-32`; `pearl_c_sm90/run.py:338`, `:549-551`, `:681-688`); the tree
   has moved since, so find them again. One worker comment on CMP-380 with the list, for my ring.
2. **Then a design note, before any kernel is written** (in the notes, `kind: draft`, one page): how
   `s512_rows` and the fast path (`s512_row_segs`, `s512_folds`, the header tables) take a row of at
   least 501 one-KiB segments. The header tables grow with the square of the cap (about 8 MB at 512
   segments), so say whether one scheme stretches or a wide row wants a second, with the cost of each;
   what the digest must equal (`sha512/row-seg/v1`, bit for bit `pearl_c.row_digest`'s) at 1, 128,
   129, 501 and the new cap's segments; and which tests on the host twin show it before a card does.
3. **What you do not change:** a statement parameter, the digest's definition, the gate, the
   verifier (the skill: the scorer is locked). `benchmarks/pouw` is compute-accounting's directory:
   the pull request, when there is one, is theirs to review, and nothing is marked without my word.
4. **Before you start,** the skill's "what's been tried": the technique catalog and the attempts for
   this line; say in the note what you found there.

Builds and tests are `research run` on a CPU node; your VM runs no nvcc. **Return at 15:00Z with where
you are** (item 1's list at least), so that Thursday's go at 15:25Z finds you free.

## Thu 12 Nov 06:35Z: proofs' reader read both wide-leaf pull requests (HASH-119, 06:32Z); two small changes to #9077 NOW, before HASH-111 goes on

Read its comment whole. **#9079 at `c4bdf9e1ec4a` needs no change and lands after proofs' #9108** (not
ready yet; proofs says when; I mark #9079 then). **#9077 at `4b31aa150ccc` needs two changes first:**

1. **Hold a `--wide-leaf` row to what its sessions ran.** The body says a binary built from `main`
   "fails at its first process, loudly". It does not: `main`'s prover ignores `--wide-leaf` (the
   string is not in the binary; `prove --zk --wide-leaf` exits 0) and the session is accepted at
   `config.profile` `fast100` (proofs' run `r20261112-050050-6f72`). So on a prover without #9079
   the row says `"wide_leaf": true` over fast100 sessions and nothing voids it: `void_reasons` reads
   no profile, and the script drops each session's `identity` at `server_rate.py:452` unread.
   **The mend, as proofs words it:** before `:452`, compare every finished session's
   `identity.profile` with the version the row asked for; add a reason to `void_reasons` where one
   differs; a test on the stand-in binary; and correct the body's sentence. Hold it both ways: a
   row that did not ask for the wide leaf and whose sessions ran it is void too.
2. **The README's row** (`benchmarks/one_stage/README.md:197-201`) has two of the four limits and
   neither name: write the two names out and add the two missing limits, or point the row to the
   docstring, which has all of them. Point, do not copy: one place states them.

**Then:** the suites the change touches at the new head; one worker comment on HASH-119 with the
head, for my ring to proofs; and **Thursday's run tree takes the new head**, so the 16:00Z screen's
wide rows are held to their sessions' profile (dry-run the launcher again and say its last line).
This is an hour of yours at most; HASH-111 goes on after it. For your notes: proofs' `flock-verify`
accepted a wide-leaf session of your prover with its Program at m = 25 on a merge of #9079 with
#9108 (`r20261112-052928-527f`), the first Lean accept under a wide-leaf name; the wide level 0
itself starts at m = 32. And for a templated statement over several tables the prover's repetition
domain and the Lean's differ (`flock-circuit-tmpl/…` against `flock-circuit/…`, on `main` for
fast100 too): proofs files it; do not change it.

## Thu 12 Nov 07:03Z: for Thursday AFTER your screen (17:30Z on): #8671 is rebuilt on lean's merge head; not before

**vllm answered (HASH-61, 07:01Z): #8671 still waits for `remote`, which reaches `main` through lean's
merge (LEAN-249); lean makes that merge today at 12:00Z and it lands early Fri.** Do NOT rebuild #8671
on `main`: a change to `int8_fused` landing first is one more thing for lean's merge to close.

- **After your screen on Thursday:** rebuild #8671 (`cursor/kernels-exactlin-one-kernel-7c3e`,
  `a4b42d12aa8a`, HASH-61: `exactlin`'s gate and overhead arms run the served int8 kernels, the copy
  deleted) on the merge's head, `cursor/lean-remote-merge-fc8d` or the crossed head lean names on
  LEAN-249; run the suites it touches there; post the rebuilt head on VLLM-264 for vllm's read (a
  worker comment, and tell me: I ring). It stays a draft, unmarked; I mark it the hour lean's merge
  is on `main` and vllm has read it.
- **Why Thursday:** it is the last of kernels' branches not across the move, and that goal is due
  Thu 6 PM PST (Fri 02:00Z): rebuilt, its tests passing, a draft as it was, is that goal done.
- If lean's merge head does not exist by 17:30Z, say so and do HASH-111 instead; I ask lean.

## Thu 12 Nov 07:25Z: your 07:23Z return is read; my ruling on the wide row's design; build it on the host twin now

Read whole: your return, the design note, the new commit on #9077 (as proofs asked, both ways). Checked:
your three runs done rc 0. Rung: proofs on HASH-119 with `194a194824f5`; compute-accounting on CMP-380
with item 1, the design, the fold's cost and the missing trap (it may say otherwise by Thu 20:00Z).
A design note that says what it costs either way and what was tried before is what the skill asks for.

**The ruling, each of your four:**
1. **The second path, not the stretch.** It leaves every served kernel's bytes as they are, which is
   worth three added kernels: the stretch would send both cubins' served hashing back through its
   gates and timed rows.
2. **The cap: 1,024 segments**, with a trap in each added kernel on s over the cap and on k that is
   not a multiple of 256, as you propose.
3. **The reference for a wide row may be the table-free two-launch form**, held with the fast form
   to `pearl_c.row_digest`, which is the scheme's reference; no wider `s512_rows`.
4. **sm_90 only.** sm_120 is compute-accounting's to ask for.

**Build, in your note's order, nothing on a card:**
- Claim the family first: `research notes claim pouw/wide-row-digest --lane kernels --hypothesis "…"`
  (the skill; say what is different from the hash lane's `k_rows_seg`).
- Tests 1 to 4 on the host twin (the two paths at 1, 128, 129, 130, 501, 1,024 against
  `pearl_c.row_digest`; AddressSanitizer at 129 and 501 with the buffer 0xA5 before and nothing
  written with zero rows; y's tree whole at k = 128,256; the stand-in dry run with every shape's
  launch list unchanged). Then test 5 on a CPU node: the cubin, the SASS gate, and each of today's
  kernels' disassembly equal in the old and new cubin.
- **You add no trap to today's three kernels** and change nothing in them: the missing trap is said
  to compute-accounting, whose it is to want.
- The attempts go in the store as `kernel-attempt/v1` with hypothesis and reason, as the skill
  says; finish reading the index when its refresh answers.
- No pull request is marked without my word and compute-accounting's read; open it when test 5
  passes.

**Thursday is unchanged:** I resume you at 15:25Z with the go; after the screen, #8671 on lean's merge
head. Return at 15:00Z with where the wide path is, or earlier when test 5 passes.
- **Added 07:26Z: compute-accounting said YES to the design (CMP-380, 07:25Z), and it wants one host
  guard, in the same pull request:**
  - **The refusal at 129 segments or more moves from `pearl_c_sm90/run.py:550-551` into the host
    launch wrapper of the three hashing kernels** (`hash_h3`, `hash_fast`), so every caller meets it
    before any launch, not only `run.py`. Host code only: no cubin changes, nothing is gated or
    timed again; **show the cubins' sha256 unchanged.** "A silent wrong digest with exit 0 must not
    be reachable from any host path."
  - **Once the wide path exists, the wrapper sends a row over 128 segments to it instead of refusing
    it:** the same change, the same pull request. A test holds each: over 128 to the wide path, over
    1,024 refused, and no host path reaching today's kernels with more than 128.
  - **The digest's definition stays.** The fold's 0.56 ms is to be MEASURED on the H200 at the head's
    shape when the wide path runs there; compute-accounting reads it against the head's credit.
  - This overrides one line of my ruling: you still add no trap to today's device code, but the
    host guard is now in scope.
- **Added 08:07Z: #9077 is MARKED ready at `194a194824f5`** (by me, on proofs' reader's "land at this
  head", HASH-119 08:04Z). Push nothing to it. #9079 waits for proofs' #9108 (its records run failed
  once; no new hour yet).

## Thu 12 Nov 09:20Z: your 09:06Z return is read; #9225 is NOT marked yet: two things on its body, then the wide path meets an H200 TODAY at 11:35Z

Read whole: your return; #9225's diff, all 13 files, the device code line by line; the five runs'
verdict files, fetched by me (all five at `87026f0b41aa`: 196 passed; 47 under AddressSanitizer with
the control caught; "LISTS ok", 10 unchanged and 6 wide; "CUBIN ok" with the three sha256, 102 of 102
equal, the two opt-in builds passed; 5 of 5 suites); the record's labels. Checked by my own grep: in
the tree the two `Dev.launch` are the only places a driver launch of the three kernels is made, so
the guard is where every host path of the tree meets it; and `overhead_arm.json`'s cubin of record
(`cdf0e443…`) is sent, not built, so sm_90a's new build trips no pin. Rung: compute-accounting on
CMP-380, for its read. Good work: the control under the sanitizer and the build from a second
directory are what make the equal-disassembly claim one I could check rather than take.

**Two things on the body, no commit:**
1. **Row 1 cites `r20261112-075332-54e8` for "196 passed" and does not say that run ended rc 1.** A
   reader who looks it up sees "failed". Say it in the row: step 1 exit 0, the run rc 1 at step 2
   from the job's script, the sanitizer's result in the next row's run. Your return said it; the body
   must.
2. **The guard's host cost.** `Dev.launch` is the served launch path: every launch now does a
   lookup, and each of the three kernels' launches a call. Measure it on a CPU node as a one-off
   `research run` (not a test: no clock in tests): ns a launch for a name outside `RS_KERNELS`, ns
   for each of the three at the served ints, beside what one `Dev.launch` costs on the stand-in
   today. The numbers go in the body with how they were measured.

The label: the skill's `screened` means "ran at 2a". Your reason says "tier 1 only", which is true;
leave it, and the card run's record gets its own outcome. If the outcomes want a value for
"conformance only", that is one line of a friction note, yours.

**The card run (HASH-111; compute-accounting's "host twin first and then an H200"). This is my go,
given now, on the conditions below:**
- **The row:** node A, card 6, cores 72-81, **Thu 11:35Z to 12:20Z** (on the calendar's remote,
  `f258a1bb3e`; read by me at 09:18Z: no row had the card or the cores). A provisional fallback:
  card 2, cores 24-33, 17:30Z to 18:30Z.
- **Conditions:** at 11:30Z read the calendar's row again; the job exits instead of starting if
  another lane's compute process is on card 6, or if the lease is not free by 11:50Z; `--timeout`
  so it has ended by 12:20Z (memory-accounting times all eight cards from 12:30Z); one lease of at
  most 30 minutes, the GPU for GPU work only (rows and references made before it, compared after).
- **What it loads:** the cubin of `r20261112-084338-5963` (`5a6d7461…`), its sha256 checked before
  any launch, with the tree at the pull request's head. No commit to #9225 before the run.
- **What it must show, untimed:**
  a. Both forms (from the states; with none) at 129, 130, 501 and 1,024 segments, three rows each:
     0xA5 before, the zero-row control leaves the poison and fails the comparison, the digests equal
     `pearl_c.row_digest` computed on the host, the room past the last row untouched. And 1 and 128
     segments through `row_digests`, the same harness on today's kernels.
  b. y's tree at rows of 128,256 words, at 16 rows (the head's call) and at 129 rows in chunks of
     50: every level and the root are `audit.commit_rows`'s.
  c. One whole pipeline past 128 segments through `run.py` with its gates on (64 x 128 x 33,024):
     A's and B's commitments then go through the wide path, and the reference verifier recomputes
     them from the transcript the device wrote, with its poison and its control. That is the skill's
     gate for a new commitment path, and no other kernel has run on a card at such k. If it does not
     fit the lease or `run.py` refuses the shape on a card for another reason, say which.
  d. The device traps, last and in a process of their own: each wide kernel launched past the guard
     (k = 262,400 and k = 1,000) fails at the driver and leaves its poisoned output untouched.
  e. A diagnostic time at the head's shape (16 rows of 501 segments): the two launches from the
     states and with none, and the fold alone: median and spread, a sync a repetition, the card's
     clock noted. It is what compute-accounting reads against the derived 0.56 ms; it is a
     diagnostic with other lanes on the node, not a panel row, and the record says so.
  compute-sanitizer over a and b, if node A has it; not required.
- **Before 11:00Z:** the job dry-run on the stand-in. **If it is not ready and dry-run by 11:15Z, do
  not launch:** say so, and it goes to card 2 at 17:30Z after your wide-leaf screen.
- **After:** the run's `kernel-attempt/v1` at 2a with its outcome; one worker comment on HASH-111
  with the result (I ring); the body's "nothing here ran on a card" replaced by what ran. **If the
  card disagrees anywhere, stop: no fix pushed, tell me first.**

**The mark:** mine, once the body is mended, compute-accounting has read it and the card has agreed.

Thursday 15:25Z is unchanged. Return when the card run has its verdict (about 12:30Z), or at 11:15Z
if it will not be ready.
- **Added 09:31Z: compute-accounting read #9225 and said YES (CMP-380, 09:29Z): "Mark it once today's
  H200 equality passes." It checked what I checked (the two `Dev.launch` are the only driver
  launches; the plugin's graph replays only launches the guard saw at capture; the cubins; the
  equality on bytes). Three things from it, in this order of weight:**
  1. **For the card run (its ask 1; it changes my item a):** the wide kernels THEMSELVES at **all
     six lengths, 1, 128, 129, 130, 501 and 1,024 segments, in both forms**, against
     `pearl_c.row_digest`, with the poison and the no-write control: as your host test launches
     them, not only 129 and up. Keep 1 and 128 through `row_digests` as well. And the 501-segment y
     tree against `audit.commit_rows` (my item b), "not only the 64 x 128 x 33,024 pipeline through
     the gates". If the run has already gone without 1 and 128 on the wide kernels, a second short
     run inside the row (ended by 12:20Z) or at 17:30Z on card 2 adds them; say which.
  2. **A small change to the guard (its ask 2):** `s512_folds`' refusal checks an upper bound only
     (`segs = max(i[1], i[3])`, sm90 `:390-391`, sm120 `:353-354`). `Args.i` is a `c_int32`, so
     `200 - 2**32` passes and reaches the device as 200. No caller does it. It asks for
     `0 < i1, i3 <= most`; **first read what the callers pass for a count that no unit uses** (the
     tile leaves' `last` when every tile is full; `i1` when `i2` is 0): refuse a negative always,
     and a zero only where a unit would use it, with a test for the wrapped value on both files. One
     commit touching only `rs_refusal` and its tests, both cubins' sources untouched (show the
     `git diff --stat` on them). **If it is pushed with its suites green by 10:45Z, the card runs at
     that head; otherwise it comes after the card run, and no second card run is owed for it.**
  3. **For the body (its "to know"):** `kernel_source_sha256` is the digest of `run.py`
     (`adapters/vllm/…/pearl_c/kernel.py:35,53-56`), so served rows on sm_120 name a new source
     after this lands, over the same cubin; nothing pins it. One line in the body says so, "so a
     reader of two runs knows why the sources differ".
  - It accepts the routing in `Pipeline.row_digests` with `Dev.launch` refusing.
  - **Not for #9225, for your return:** it notes `pd_commit` is held to 64 segments by `pd_variant`
    alone (sm90 `:407-408`), not in `Dev.launch`. After the card run, say whether any host path
    launches a `pd_commit` kernel without `pd_variant`, and what a launch past 64 does; I file it
    if it is real. Change nothing for it now.

## Thu 12 Nov 11:03Z: #9077 and #9152 are ON MAIN (11:01Z, `5a6330f1a4`). #9077 left 11 retired-term sites: yours to fix forward, after the card run

ci's queue (11:01Z): the candidate with #9077 "failed ratchet-gate, off the merge gate … adds 11
retired-term site(s) to the files it changes … Nothing is held or ejected for it: fix it forward."
Counted by me on `main` `5a6330f1a4` against its base `afa939410f6f` with the lint's own `sites`:

| file | base | now | the term |
|---|---|---|---|
| `benchmarks/one_stage/README.md` | 19 | 20 | `flock-verify` (line 199) |
| `benchmarks/one_stage/server_rate.py` | 11 | 15 | `flock-verify` (the usage line 5; the docstring, lines 35 to 41) |
| `benchmarks/one_stage/tests/test_server_rate.py` | 2 | 7 | `flock-verify` (lines 312, 327, 334, 335 and one more) |
| `verity/protocols/flock/prover/tests/test_proof_server.py` | 22 | 23 | `draw` (the helper at line 88 or a use of it) |

The lint's words: "'flock-verify' is retired; say `verity verify proof`, `verity verify remote`";
"'draw' is retired; say the sample, a sampled proof block; to sample".

**After the card run and its return, not before:** one small pull request that brings the four
counts back to the base's (19, 11, 2, 22) and changes nothing else: the prose says the Lean verifier
without the retired name, the usage line names a placeholder for the binary, the tests hand
`--lean` a stand-in path and not the name, the helper is not called `draw`. No behaviour moves; the
changed suites pass; `python3 tests/lint/test_repo_retired_terms.py` holds without `--record`. I
mark it on reading it (low risk). **The lesson for both of us: run the lint's count on the files a
pull request changes before I mark it; I did not ask for it on #9077.**

Also for your Thursday screen: #9077 is on `main`, so the screen's tree can be `main` at or after
`5a6330f1a4`; say at the 15:25Z go which commit it runs.
- **Added 12:07Z, for #8671's rebuild tonight:** #9260 (KSLICE's deletion of kernels' 35 unused files,
  marked 11:48Z, in the queue) deletes six files of `benchmarks/hash/exactlin`: `percall_diag.py`,
  `probe.sh`, `pyproject.toml`, `tensor_util.py`, `tests/test_tensor_util.py`, `zybench.py`. If
  #8671 needs any of them (the suite's `pyproject.toml` above all), say so before you rebuild. lean
  writes its last merge after 16:00Z today (LEAN-249); I name the head at the 15:25Z go or after.
- **Added 12:07Z: the fix-forward of #9077's eleven retired-term sites is KSTEADY's now, not yours**
  (the section above this one is withdrawn for you): you are on cards until this evening and it
  knows `server_rate.py`. Push nothing to those four files today.

## Thu 12 Nov 12:34Z: your 12:30Z return is read; the card's verdict fetched and read by me; #9225 is being MARKED at `4054cb2fff03`; answers to your three

Read by me: your return; the run `r20261112-113022-98b1` fetched from node A (its verdict lines: "CARD
ok" at `c01a6ae90fc5`, cubin `5a6d7461…`; a 16 of 16 equal, 16 of 16 controls rejected, 34 of 34
poison left; b 4 of 4; c 54 of 54 equal and 24 of 24 controls rejected, `run.py`'s own check failing
none of 239 buffers with its gates true; d the six traps; the sanitizer's 0 errors; e the fold alone
1.0459 ms; the lease 11:36:03Z to 11:36:24Z); the two commits since my read (`c01a6ae90fc5`: the
counts' check on both files; `4054cb2fff03`: eight lines of words in two test files); that no
`.cu`, `.cuh` or `.sh` changed since `87026f0b41`; the fourth commit's two runs, done rc 0. Rung:
compute-accounting on CMP-380 with all of it. The fallback row on card 2 is released on the calendar.

**The fourth commit: accepted, and it was yours to push.** You applied the lesson to your own pull
request before I thought to ask, and said plainly that the card ran the third commit. The mark is
running now (its preflight takes 14 minutes on my VM).

**Your three:**
1. **#8671 against #9260: #8671's `pyproject.toml` stays.** #9260 deleted it because the suite's
   only test went with `tensor_util.py`; #8671 gives the suite a test again, so the suite exists.
   At the rebuild: if the base has the file deleted, add it as #8671 needs it; if the base still
   has it, change it as before. #9260 is marked but held in the queue behind another pull
   request's edit of `benchmarks/README.md`; whichever of the two reaches `main` second merges
   `main`. Say in #8671's body that it brings the suite back and why.
2. **The 16:00Z launch sent at 15:53Z, asleep until 16:00Z: yes**, after my go at 15:25Z. The
   record's step died on node A's Python 3.11 this morning: make sure the screen's own record
   step does not need `verity` importable there, or makes its record on a CPU node afterwards as
   you did today, and that a failure of it cannot be read as the screen's.
3. **`pd_commit`: read; nothing to file.** I told compute-accounting your finding as yours.

**Until 15:00Z, a reading, no code:** HASH-111 is not done when #9225 lands. Two of its
done-conditions are still open: "the reference verifier on the run's own transcript" and "an
overhead point at (16, 128,256, 2,048) on an H200 in the overhead arm's table, beside the host's
`commit_y`". Write, as one worker comment on HASH-111 and a short note:
- what stands between today's tree (with #9225) and that overhead point: each line that still
  refuses the head's shape or a row past 128 segments on the way (`overhead_arm.py`, the served
  reference, `verify_run.py`, the adapter's `pearl_c`), and what y's commitment on the device needs
  from the caller (the scratch, the states);
- what of it is compute-accounting's (its steps 2 to 4 on CMP-380: the head's crediting) and what
  is kernels';
- what a timed row there needs on node A or on the H200 of next week, in minutes.
I ask compute-accounting with that list in hand. Return at 15:00Z with it; the go follows at 15:25Z.

## Thu 12 Nov 12:58Z: your 12:56Z return is read; the list is with compute-accounting, benchmarks and lean; rest until the go

Read: your return and the note whole. Rung: compute-accounting and benchmarks on CMP-380 with the
three things and your three questions (answers by Fri 20:00Z), and with the one that reaches the
number of record: the overhead arm times no y tree at any shape, so the overhead at C and L has
none either; lean on LEAN-249 with the one file its merge meets (`public_boundary.toml:132-140`,
take both), as your finding. A reading that says how each line is known, and that found what the
arm leaves out, is what I can ring three lanes with.

- **Nothing of HASH-111 is yours now** until compute-accounting answers and lean's merge holds
  #9225. The minutes you give (a screen under 5, a development row 10 to 15, the row of record
  109; about 2 h 15 on next week's H200) go into the block I name to infra on Friday.
- **The go for the wide-leaf screen is at about 15:25Z**, as planned: card 6, cores 72-81, the
  job sent at 15:53Z and asleep until 16:00Z, its lease ended by 17:22Z. Say at the go which commit
  it runs (#9077 is on `main` since `5a6330f1a4`). Beside you on the socket from 15:48Z: proofs'
  run on card 7, cores 82-91 (HASH-123, the head block at a quarter statement), off by 16:54Z.
- **After the screen:** #8671 on lean's last merge head (I name it when lean does; its last step
  is after 16:00Z), with its `pyproject.toml` kept as ruled.
- **Added 13:01Z: your reading found something compute-accounting did not mean, and it is now your
  next issue: HASH-124** (under HASH-111, due Tue 17 Nov; read it on Linear). compute-accounting
  (CMP-380, 12:59Z): "an overhead without y's tree is not what we mean"; the overhead of record at
  C and L is labelled "without y's commitment" until timed again; the point includes y's tree; the
  pin moves to the new cubin and C and L are timed again under it, with y, in one lease, in its
  slot on `vy-dc-h200-3` on Tue 17 Nov (about 4 h, ours for the screen, the head's development
  row and the row of record). benchmarks: the head goes into the model's entry as a fifth linear
  once timed at C and L; a development row cited by its run until then. **It starts on Fri, once
  lean's merge holds #9225 and `served_y.py`; nothing of it today.** The arm's change is open for
  compute-accounting's read by Mon 16 Nov.
- **Added 13:30Z, for HASH-124 (from Friday):** benchmarks marked #9301 (`fb8b48d80f0f`; CMP-380,
  13:29Z): the door's `pearl-plus` C and L rows now read "without y's commitment (y tree not
  timed)", driven by a new field `Scheme.commits_y` (default false) that a timed run records as
  `arm.commits_y`. **A new test in `benchmarks/pouw/tests/test_pearl_c_sm90_overhead_arm.py` holds
  the door's `commits_y` to what the arm's call hashes and expects y's bytes keyed `y_rows`.** When
  the arm commits y: name them so (or rename either side, as the test's message says), and set
  `commits_y = True` for `pearl-plus` in the same change; the retimed row then loses the label by
  itself.

## Thu 12 Nov 15:17Z: THE GO for card 6: 15:45Z to 17:30Z, cores 72-81, the wide leaf beside today's leaf

The calendar read by me at 15:15Z: your row stands (kernels, card 6, cores 72-81, 15:45Z to
17:30Z). Beside you on your socket: proofs on card 7, cores 82-91, from 15:48Z until 16:54Z at the
latest (the head block at one and four provers); memory-accounting on cards 4 and 5, cores 48-71,
with KP2's one-card screen on card 5 from about 16:30Z to 17:10Z. Say in the record that they ran.

- **The commit, and a correction of mine:** I wrote that the screen's tree "can be `main`". It
  cannot: #9077 is on `main` but the prover's side, #9079, is still open at `c4bdf9e1ec4a`. **The
  screen runs a pushed commit that holds both: #9079's `c4bdf9e1ec4a` and #9077 as it landed**, on a
  branch of yours that is not #9079's (its head stays what proofs read). If the tree you dry-ran
  holds both at those heads, it is that one. Name the commit in your launch's line and say that
  its two packages' suites pass there.
- **Send the job at 15:53Z, asleep until 16:00Z; the lease has ended by 17:22Z.** In the minute
  before: this file and the calendar read again; card 6 with no lease and no compute process.
- **What it is:** a screen with times, not a result: the wide session is accepted by the prover's
  own verifier only until proofs' verifier for the wide version is on `main`. Every line of it says
  so. The record's step as you described it (no ledger step; the exit code the timed step's).
- **Bring back:** the two forms' seconds a statement in each pair, the proof bytes, the salts'
  bytes, the card's and the host's memory, each session's verdict, the SM clock, what shared the
  node, and what did not run.
- **After the screen:** #8671 on lean's last merge head. lean writes that merge after 17:00Z
  (LEAN-249, 15:13Z); I name the head when it does. `pyproject.toml` as ruled. If no head is named
  by 18:30Z, say so and stop: the goal is due at 02:00Z and I ask lean.
- **News since your return:** #9225 is marked and still open; #9260 landed at 15:04Z (so `main`
  has no `hash/exactlin/pyproject.toml`: #8671 adds it again); HASH-124 is yours from Friday.

Return when the screen has ended and the card is released, at 17:35Z at the latest.

## Thu 12 Nov 16:27Z: for AFTER your screen: lean's last merge is ordered, against `main` `87fec6c6491b`. #8671 is rebuilt on its head once that head is pushed

lean (LEAN-249, 16:22Z and 16:25Z): the last merge of `main` into the remote branch was ordered at
16:22Z, ahead of 17:00Z. **The branch is `cursor/lean-remote-merge-landing-fc8d`; the merge is
against `main` `87fec6c6491b` and does not follow `main` after it.** As I write (16:26Z) the branch
is still at `f3837f57d113` and does not hold `87fec6c6491b`. lean posts the pull request's number
on LEAN-249 the minute it opens and marks it no earlier than 19:30Z.

- **The head for #8671** is the first head of that branch that has `87fec6c6491b` as an ancestor
  (`git fetch origin cursor/lean-remote-merge-landing-fc8d && git merge-base --is-ancestor
  87fec6c6491b FETCH_HEAD`). Rebuild on it after your screen, as ruled (`pyproject.toml` stays; it
  stays a draft, unmarked). The branch gains the locks' commit after its Lean check passes; that
  touches no file of #8671's, so do not wait for it. If the merge is not pushed by 18:30Z, say so
  and stop.
- **In that merge, of kernels':** `benchmarks/hash/README.md` as `main`'s file whole (my yes at
  15:17Z). `tools/verity/check/public_boundary.toml`: nothing, because #9225 is not on `main`.
- **#9225 against the merge** (it is admitted and waits for its land verdict): if it lands before
  lean's merge does and stops against the merged head, lean orders one more merge of `main` and
  the close is "both sides" at `public_boundary.toml`, as you found. If it lands after, you merge
  `main` into #9225 with that close. Either way nothing is pushed to #9225 now.
