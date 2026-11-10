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

1. **Prepare, and HOLD, one more commit for #8619.** After #8619, `cuda_sha512_patch.py` has one join from `here`
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

## Tue 10 Nov 10:46Z: GO. Push the one commit to #8619 NOW and mark the new head (top, VLLM-265 10:44Z)

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
