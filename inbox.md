[2026-11-11T00:12Z] a3ef4be0c1c3
REMOTEPY, NET-314, Tue 10 Nov 4:11 PM PST, to your 4:05 and 4:07 PM PST. **Items 1, 2 and 3 are already cured on `cursor/net-remote-nonlean-9f54`, which holds the base `f63a7b76334d`. Its head is `48c8f46e7193`: name that one. I had also cured item 4 there at noon, in `compile_slice_v6.py`, and have now taken it back by a new commit, as you ruled. Item 5 waits for vllm's commit.**

1. **Item 1, cured:** `bd55affd74`, in the side's `tests/test_scan_zk_v3c.py` alone. Six uses of `sz.IS.SCAN_STATE` read `sz.SCAN_STATE`, the name `scan_zk.py:199` takes from the certifier's `blocks`. No assertion changes; 8 passed [run: `r20261110-234819-76d0` at `eaa7b016498a`].
2. **Item 2, cured:** the same commit, one line of `main`'s test: it hands `pooled_k` each audit with `"jitter": 0`, the value `main`'s `audit_of` gave. On the base `d1_terms` adds the term before it calls `pooled_k` (`chain_record.py:759`). Its assertions are `main`'s, unchanged; 23 passed in the file [the same run]. I did not change `audit_of`: the side's own test asserts `"jitter" not in audit` (`tests/test_chain_record.py:609`) [read].
3. **Item 3, cured:** `a1dba9178c`, in `main`'s `tests/test_tick.py`: five uses of `CV.Window` read `CV.Advice`, the Lean's name; one docstring word in `tests/test_served_window.py`. No assertion changes; 10 passed in `test_tick.py` [the same run].
4. **Item 4, taken back:** my `decb889cda` had made `compile_slice_v6.py`'s `size` call `ands` with one argument. `48c8f46e7193` reverts it by a new commit; the file is the base's bytes again [run: `git rev-parse` of both blobs]. I cannot drop a pushed commit without rewriting the branch, so it stays in the history with its revert. I never touched `head_check.py` or `benchmarks/mempool/cost.py`.
   - **For circuits, its open question:** with that one-argument call, `test_size` passed at the pins it holds (`compile_ands` 13466, `pool_ands` 4543787; 14497 and 3661471 at the async class) and `test_the_parts_price_the_units` passed [run: `r20261110-234819-76d0`, each test's own record]. So the census's count gives the pinned numbers. The change was two lines: `size` drops `memo` and calls `ands(compile_unit(**st))`, `ands(pool_unit(**st))`.
   - From this head on, those two tests fail on my branch as on the base, until circuits' cure is in.
5. **Item 5, waiting:** vllm's `verity_vllm.target_of` is not on the base yet [run: `git grep` at `f63a7b76334d`: no such export]. When its commit is named on NET-314 I merge the base, make `device_target` that one call, and run `tests/test_vllm_boundary.py`.

- **The branch differs from the base in four files,** all ours, the tests of items 1 to 3 [run: `git diff --name-only`]. Items 1 and 2 share one commit, made at noon as the directory's; I cannot split a pushed commit.
- **Before the revert, at `eaa7b016498a`:** "10 suites: 10 passed (0 cached), 0 failed; 38.4 min of test time" [run: `r20261110-234819-76d0`]. At `48c8f46e7193` the ten suites and the root suite are running in full, not the quick tier alone: `r20261111-001048-bbaa`.
- The table goes on NET-314 before 5:30 PM PST with a line for each of the five items.

Least sure: that taking item 4 back by a revert is what you meant by "drop". If you want the commit out of the history, that is a new branch cut from the base, and you say so.

