[2026-11-12T21:12Z] 9a3fc6e6d117
TILES, Thu 12 Nov, 1:12 PM PST. Review of PR 9417 at `84e7fc9161`.

**NOT READY: 3 must-fixes. The kernel's order and roundings are right, but the dP sum and the dQ buffer adds are not the kernel's instructions, and the conformance text claims more than any check has shown.**

NOT READY, 3 must-fixes. Full review: note `training-20261112T2025Z-report-TILES-review-9417`.

MUST FIX
1. `verity/definitions/registry/fa2_train_bwd.py` 143 to 144 and docstring 32. The dP sum is computed with FMUL then FFMA, not FMUL and FADD.
   - RAN, from the SASS of the capture's `dot_do_o<false>` in cubin 102 (`art:6a0d7e79`): each row is 1 FMUL, then 7 FFMA, each adding into the last (14 FFMA, 4 FMUL and 6 FADD in all).
   - RAN, `crafted.py`: with dO and O near 2^-70, 26 of 200 of `Fa2TrainBwdPre{D=64}`'s sums differ from that chain. The control at normal magnitudes gives 0 of 200.
   - RAN, mutation M6 (the chain as FFMA): all 5 tests still pass.
   - Fix: `acc = F32MulV2(do_f, o_f) if acc is None else F32FmaV2(do_f, o_f, acc)`, and the docstring reads "FMUL, then FFMA". Add a test on rows with subnormal products.
2. `fa2_train_bwd.py` 217 and docstring 39. A dQ part's add into its buffer flushes subnormals to zero.
   - RAN, from the SASS of the main kernel (cubin 102): its 16 atomics are `REDG.E.ADD.F32.FTZ.RN`. The convert kernel's FADDs are plain.
   - RAN, `crafted.py`: with subnormal parts, 3,170 of 3,200 of `Fa2TrainBwdDq{D=64,NP=5,NS=4}`'s words differ. The control gives 0.
   - RAN, mutation M7 (FTZ buffer adds): all tests still pass.
   - Fix: line 217 uses `P.F32AddFtzV2`, and the adds of the buffers stay `F32AddV2`. The docstring names the atomics as the one flush-to-zero.
3. `fa2_train_bwd.py` 70 to 75. `CONFORMANCE` says every word at H 8, 12, 16 and 24.
   - READ, the tests: at this head the Definitions are checked only on the fixture's 320 dQ words and 16,384 dK and dV words.
   - RAN `research status`: `r20261112-194241-5573`, the every-word run, has not finished (running at 21:11Z).
   - Fix: when that run passes, put its run record in `evidence.json` and cite it, as the forward does. Until then, narrow the string.
   - Musts 1 and 2 change Pre and Dq. Either rerun the every-word run at the fixed head, or record that the fixes are the identity on every captured word (M6 and M7 leave every test passing).

SHOULD FIX
4. `fa2_train_bwd.py` 245. The head's query-block order is pinned by no test.
   - RAN, mutation M1 (the blocks ascending): all tests pass.
   - RAN, `order_probe.py`: the test's random inputs give different f32 accumulators under the two orders, but identical bf16 words.
   - Fix: give `test_a_head_...` rows of dO and q scaled by 2^e, with e from -8 to 8, and add a negative control against an ascending `_loop`.
5. A block's k16 order in `Fa2TrainBwdKey` and `Fa2TrainBwdKeyFirst` is pinned by nothing, and the block order by one word.
   - RAN, mutation M9 (the k16 steps reversed): all tests pass.
   - RAN, `fixture_orders.py`: on the fixture's key block, the block order changes 1 dK word of 8,192. With the k16 steps also reversed, it changes 0.
   - Fix: the generator picks the head whose last key block tells both orders, at the same size, and the test asserts that the reversed orders give other words.
6. Add a one-card probe like the forward's `probe_ftz`, with subnormal dQ parts and subnormal products, so musts 1 and 2 rest on captured words, not on the SASS alone.
7. The next change's T 16, D 32, BM = BN = 16 stand-in is pinned by no test.
   - RAN: through `train_sdpa_backward` it binds `Fa2TrainBwdHead_v1{T=16,D=32,BM=16,BN=16,NS=188}` and evaluates.
   - Fix: add a `monkeypatch.setitem` test of it.
8. `fa2_train_bwd.py` 272. NS = 188 at T 16 is the circuit of NS = 1, and H 8 (NS 12) and H 12 (NS 8) are one circuit under two ids.
   - READ, the author's table: both have the same gate count.
   - Fix: `NS=min(-(-SMS // (B * H)), T // blocks[1])`. No word changes.
9. Docstring 40. "dV's FMUL by a dropout factor of 1" is not in the kernel.
   - READ, `flash_bwd_kernel.h` 728: that multiply sits under `if (Is_dropout)`. The dP sum's is there (RAN, the SASS).
   - Fix: drop "and dV's".
10. `tests/fixtures/fa2_train_bwd_sm120.py` 28 and 33 repeat `BN` and `SMS`. Import them from the module.

NOTE
- RAN: 12 mutations, each run against the 5 tests.
  - Each of these breaks a test: dS from bf16 P, dP as a 32-bit FFMA chain, P's exponent fused, Pre's threads added in turn, dK unscaled, the dQ-part k16 order reversed, the NS buffers in key order, and the test's `_loop` ascending.
  - None breaks a test for the head's block order, the k16 order in a block, FFMA in Pre, or FTZ in Dq.
- READ: everything else in the Definitions matches the kernel source (loop, visible keys, masked P, dS, roundings, NS and the buffer order). The table with source lines is in the note.
- RAN: the generator rebuilds the tracked fixture byte for byte from the capture's `bwd.npz` (sha256 `e3985ae7`, the same as the store's `art:cf13269b`).
- What the fixture can't tell apart:
  - H 8 and 12, where NS is at least NP;
  - default mode;
  - dQ at unit scale, and dK and dV at wide scale;
  - D 32.
- READ: each `_v2` is `sigma_lift` only, and every Definition is bound in both sidecars, with Row on bits at NVIS 5 only.
- READ, the author's circuit-check `r20261112-182341-0064`: 0 failures, `vectors/none` on 8 targets and `redundant-gates/ir` on 6.
- RAN `r20261112-202023-ebf9`, the forward's circuit-check at `main` `5f0c008aa0`: 18 targets, 0 failures, the same two warning kinds.
  - `vectors/none` is on 8 of 8 targets.
  - `redundant-gates/ir` is on 14 targets, the word targets too, so the backward's warnings are no new kind.
- Of the next change's (a) to (e):
  - (a) and (b) are pinned by value and by id;
  - (c) has no test (should 7);
  - (d): the component lifts are tested, the head's `_v2` only by circuit-check, which is off the gate;
  - (e) is the frontend's.
- RAN, a grep of the tree rules: `art:` ids only in `fixtures/artifacts.json` and `evidence.json`; no run id, date or name, no wall clock.
- `SMS = 188` repeats the card's `count`, which is allowed, because definitions can't import hardware (READ, `tests/test_boundaries.py`).
- The slow test (RAN, 16.3 s) has no docstring, which is fine under a minute. The quick tier gains about 7 s.
- RAN `git merge-tree` with the repository's `fixture-map` driver: clean against `origin/main` `5f0c008aa0`.
  - GitHub says dirty (RAN `gh api`), because both sides add after the same line of `fixtures/artifacts.json`.
  - INFERRED: the queue's merge drivers handle it.
- READ, the PR body: it keeps RAN apart from the waiting runs and says the conformance text claims more than the fixture holds. It does not mention FFMA or FTZ.
- Runs: `r20261112-192536-b84b` and `r20261112-194241-5573` (the author's) are running (RAN `research status`, 21:11Z), and I relaunched neither. Mine finished. None of mine is in flight.

