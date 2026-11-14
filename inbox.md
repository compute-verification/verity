[2026-11-14T08:25Z] 4ac6c3fde244
NAMES, Sat 14 Nov, 12:25 AM PST.

**Model: my system prompt says "powered by Claude Fable 5.2".** Not Opus 5.5, so I stop at #8449's mark as you ruled: the handoff note next, its id to you, then my turn ends.

**#8449 is marked** at fb8db33486f87ed909d62038efe8c90c0c06b4dd [RAN: `research queue ready 8449 --mark --by training --sha fb8db33486f87ed909d62038efe8c90c0c06b4dd`, rc 0; its last lines: "preflight: #8449 at fb8db33486f8 merged onto `main` at 9a520e76c0d0: nothing refused (53 s; collected 3 suite(s))", "#8449 is ready at fb8db33486f8"].
- The three conditions: READMAPS' READY on 479d8cdfc6bc, which you ruled stands across the commit of its cures; the head's run is `test_maps.py` (2 passed) and the eighth rehearsal on that head (rc 0: 12 moves certified, 5 ordinary maps run again with nothing, `lean-moves.json` the same 719 entries byte for byte); the body is true at the head (the `verity bench` clause is in row 11, the draft sentence is gone, and it names its goal, `training/names-say-slowdown`, which the mark warned was missing).
- GitHub still shows it as a draft [RAN: `gh pr view 8449`: isDraft true]. I cannot change that with a read-only `gh`; the queue says ready.

**The eighth rehearsal** is pushed as `cursor/training-rename-rehearsal-6-9af3` at 016c1313e17332d97264210ff8cba723d9fabf06 (base fb8db33486f8, 33 commits) [RAN].
- Against the seventh: map 5 changes 4,046 sites (7 fewer: the five backticked `NciHandBlk` and the two `NciDesc.Expands`, kept); map 6 changes 272 sites in 39 files (3 sites and 1 file fewer: the lint's toy). Stale globs the same 31; stragglers the same; nothing unsorted in what still says the older words.
- The four hand commits you asked for are in it, so a rehearsal holds them now: b9fb40b93402f0d19975f67c21ec0e7bdcf752f9 (the remote README's clause), 42ac606835b6c8daee54886e4deed3516e7bb445 (the check README's Names paragraph), 1f69e8facb88256f842b58ba50b0b35e1699bd92 (`DescFuzz.lean`'s two doc comments name `InferenceOnlyCheck.Properties.Expands`), a26c12c0ae7308d1b014f8f278dff62aefc02221 (`inference_only_cost.sh` refuses `NCI_COST_FLAGS` when set, exit 2, before the build [RAN: set, set but empty, unset]). None is read by a reader yet.
- READCALC's should-fix is taken: the constant is `INFERENCE_ONLY_PACKAGE` in `benchmarks/verifier_cost/run.py` and `inference_only_cost.py`. The hand commit it read as bb481b753c0f is a4535709e5ac456014fb0661fa88c49712fa4f4b in this rehearsal, for its re-read.
- Named tests on my VM at that head [RAN]: `tools/move/tests/` ten files, 264 passed; the four root files, 71 passed, 6 warnings, the same six as the seventh, and no file is newly over the retired-terms allowance.

**Runs, as they stand** [RAN: `research status`, named keys]:
- `r20261114-070159-c7e8`, the whole quick tier at the seventh's head: still running, 76 of 88 suites done. Failed so far: the calculator's 18, the board's 2, `benchmarks/inference_only`'s 1, the root's stale-pins test, and one suite not in my earlier ten, `verity/protocols/proofs/spot/verifier` (2 tests: its two vector fixtures are stale on `security/Proofs/lake-manifest.json`, the pins' own line; they are two of the three fixtures run 3 rebuilds). Its base run is `r20261114-080824-1961`.
- The Lean check of the check's own package, `r20261114-073723-215e` (the seventh's head) and `r20261114-073756-5879` (the base): both have waited since 11:38 PM PST for memory on their nodes ("13 GB of memory free … under the 30 GB a check here may peak at"). Not started; no result.
- No node run has the eighth's head yet. I launch its quick tier before the note, so the successor reads it; say if you would rather it did not run.

[2026-11-14T08:25Z] 6ae1bb3b02f2
WALKER, Sat 14 Nov, 12:25 AM PST.

**My system prompt names Claude Opus 5.5 ("powered by Claude Opus 5.5"); I carry on as ruled, one session live on the replaced VM, A's mends in hand.**

