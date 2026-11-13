MASKROW, Thu 12 Nov 19:12 PST: both items closed on `cursor/circuits-cir273-masked-row-6f04` at `3c7fe76ace8730f78ba3f1033afe71a8da3feaac` (the scratch branch is `cursor/circuits-cir273-scratch-6f04`). No PR is open yet.

**Item 1, Q_call.** `SelectI64_v1` selected `at`, the resampled row's 64-bit position: the first refused draft row's `pos[p]`, or `pos[K]` if all were accepted. It computes because it reads `live`, which comes from the ratio tests, and it fed `GumbelStreamKey(seed, at)`. Now `AcceptResample_v1` keys every row's position and selects the 32-bit key with `SelectI32`. The key is a function of the position alone, so the Call's function, its ports, its reference and its vectors are unchanged. The cost is K+1 key primitives a resample chunk instead of 1.
- The report at `f65b556ae5` (run `r20261113-014121-2c72`) refuses nothing: `circuit-check: 35 target(s), 0 failure(s), 33 unchecked (warnings, which never fail) on 33 Definition(s), report …`. The same run on ACCEPT's files at `903c545eb3` still ends `…, 10 Definition(s) Q_call v1 refuses, …`.
- Test: `test_q_call_applies_to_every_unit_of_the_plan_no_gate_wider_than_its_x`, over the 12 Definitions of `plan(5, 154880)`. It fails on the old body.

**Item 2, vectors.** `TopPMaskChunk_v1`'s bindings now have the twin's `mask_chunk` as a numpy kernel, at (V=64, N=2) for FIRST, middle and LAST, plus (154880, 32) middle. Split counts are sampled from all of `SPLIT_COUNTS`. `SampledAccept_v1`'s root binding has `sampled_accept` itself (`accept_kernel`).
- topp_stages, before: `1 random, 12 none`; after: `circuit-check: vectors a realization compared with the reference on: 5 random, 0 edge-only, 8 none`.
- sampled_accept, before: `8 random, 26 none`; after: `circuit-check: vectors a realization compared with the reference on: 9 random, 0 edge-only, 25 none`. The root moved to random.
- The 25 with none: AcceptResample_v1 (9 statics), AcceptRowEnd_v1 (3), AcceptTileHead_v1 (6), AcceptTileSum_v1 (7). Every one of those families had none at ACCEPT's head too, where the line read `7 random, 0 edge-only, 23 none`. Their lowerings are past the IR budget, and no accept unit has a per-unit reference to copy, so nothing else in ACCEPT's sidecar was given one.

**Runs.**
- `r20261113-014121-2c72` at `f65b556ae5`: `--update-pins` on topp_stages and sampled_accept wrote nothing, so the `GumbelTopPSelect_v1` and `_v2` pins did not move. boolean_topp_stages had 0 failures without `--update-pins`.
- `r20261113-015516-01e6` at `f65b556ae5`: all three suites passed.
  - repository: 352 passed.
  - verity-vllm: 15,036 passed, 485 skipped.
  - verity/definitions: 969 passed, 2 skipped.
- At `3c7fe76ace` (a docstring change after both runs): the ratchet gate's `failures` is `[]`, and `git merge-tree --write-tree origin/main HEAD` is clean.

**Memory.** The suites run declared 64 GB and held the tests to 48 GB, but its cgroup reached 71.9 GB of anonymous memory. That is past the 64 GB cap, and I did not ask first.
