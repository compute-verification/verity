[2026-11-11T00:05Z] 2162eb5c8784
Coordinator to REMOTEPY, NET-314, Tue 10 Nov 4:05 PM PST. **lean has listed 15 failing tests of ours on the merged branch (LEAN-249, 3:53 PM PST). Their cures are yours, as one branch that changes only our files, named to me by Wed 11 Nov 9 AM PST. In your table tonight say, for each of the five items, whether `cursor/net-remote-nonlean-9f54` already cures it.**

The run is `r20261110-222347-d728`, every suite's quick tier at `bbaf05b89b12`, read by lean's worker MAINMERGE (its 3:44 PM PST comment on LEAN-249 has every line). None is cured at `97876e5336e1` by its reading.

1. `benchmarks/network_traces/tests/test_scan_zk_v3c.py`: 8 errors at setup, line 40: `ingress_scan` has no `SCAN_STATE`. The test is the side's. It reads `sz.IS.SCAN_STATE` at six lines; the base's `scan_zk.py:195` takes `IS` from `main`'s module and `scan_zk.py:199` takes `SCAN_STATE` from `verity.core.protocols.certifier.blocks`.
2. `benchmarks/network_traces/tests/test_chain_record.py`: 1 failed, `KeyError: 'jitter'` at `chain_record.py:725` (`pooled_k`). `main`'s `audit_of` returns `"jitter": 0`; the base's (lines 688 to 690, closed in `c2420ee157`) returns none.
3. `benchmarks/network_e2e/encodings/tests/test_tick.py`: 3 failed, `enc_code_v1` has no `Window`. `code_v1.py` was closed as `main`'s file under the side's name `Advice` (`91a62a4d46`); the test is `main`'s and reads `CV.Window`.
4. `verity/core/protocols/certifier/tests/test_network_certifier_compile_slice_v6.py`: 2 failed, `ands()` called with two arguments at `compile_slice_v6.py:1099`. The base's `head_check.ands` is the side's `ands(fn)`; `main`'s is `ands(fn, memo=None)`. circuits' `benchmarks/mempool/cost.py:299` makes the same call.
5. `benchmarks/prove_path/prove_path.py:1365` imports `vllm.envs` (`tests/test_vllm_boundary.py`).

**My rulings.**
- Item 4: `head_check.ands` takes `main`'s form back, `ands(fn, memo=None)`, so both callers work and the side's one-argument calls still do. If the side's function counts differently from `main`'s, stop on this item and tell me what differs.
- Item 5: vllm answered on NET-314 (3:35 PM PST). Its worker adds `verity_vllm.target_of(model=None, async_scheduling=False) -> dict` to the merged branch by Wed 8 AM PST and names the commit there. Our side is then the one line. If vllm says by 8 AM that the merged branch cannot take it, tell me and leave item 5 out of the branch.
- Items 1 to 3: you choose which of the two files changes. No assertion is weakened or deleted to pass, and a name keeps the meaning the rest of the base gives it.

**The branch.** Cut from the base's head, `cursor/lean-remote-merge-fc8d` at `f63a7b76334d` now. The base moves tonight when the layout's move comes in, and lean names the new head on LEAN-249: cut from that one if it is named before you start. Each cure a commit of its own. At its head run the quick tier of the three suites and `tests/test_vllm_boundary.py`. You push only your own branch; lean merges it. I name it on LEAN-249.

No new worker may be started tonight (Daniel's cap), so this is yours alone.

