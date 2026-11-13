[2026-11-13T09:08Z] 05a61e93fba0
DGRAD, Fri 13 Nov 1:08 AM PST (TRN-156: `per_row`, q4muon-full held, one route fix, the before-and-after runs)

**q4muon-full is held. Grow's step 164 showed one more route difference, B's gradient arranged as a transposed product because the rebuilt B was a transposed view; it's fixed. The before-and-after runs restart in chunks, because both whole-library runs hit the check slot's one-hour limit.**

1. **`per_row`, as ruled** (commit e6211d4432): a row whose phase is in `gates.OPTIMIZER_PHASES` stays whole. The docstring says why, and a toy test has an `other`/`optimizer` row with a dimension of rows × ctx kept whole beside a `fwd` row of the same shapes that is divided.
- **Your 5** (commit 09a8646593): `shape` there is the start's, with `d_model`, `d_ff` and `vocab`. So `tokens_by_size` refuses an entry whose record has more than one row a forward and a width of rows × ctx, naming the width. It is tested; at one row a forward it divides nothing and so refuses nothing.
- The PR body gets the one line on what this doesn't fix.

2. **q4muon-full held** (RAN r20261113-081616-25d9 at dcff1a93f6, 32:13 wall, 18.9 GB): `equal, 1 expected difference, 1 named line of the record's program alone`; 880/880 uses, 59/61 products, norm expected, the printed loss named (`record_only` count 16). Its entry is in `of_record.toml`.

3. **Grow, counted to step 164** (RAN r20261113-081745-be6f on the scratch branch, `STEPS = 165` with a grow entry; 29:41 wall, 3.5 GB):
- It matches 896/896 uses and 68/73 products; the four differences are each adapter's B gradient. The record computes it as `[out,1024]·[1024,84]`, while the route computed `[84,1024]·[1024,out]`: same MACs and counts (128 + 64 = 192 and 320 + 384 = 704), the product transposed.
- Cause (RAN locally, a dispatch-mode probe): `dgrad.factor` handed the step `v.T`, a transposed view, and autograd arranges a weight's gradient by its layout.
- The rank-84 scratch run matched only because it ran on the prototype, whose B was a contiguous parameter. The four LoRA entries are affected the same way on the route as it was, so they recount.
- Fix (commit 6e15d4ba1a): B is `v.T.contiguous()`, with a comment saying why. A copy isn't counted, and the dgrad tests pass.
- Recount: r20261113-090724-d211. The entry still waits on #9531's `step` field.

4. **The suites** (RAN r20261113-075801-a29c at b6b00e0fa0): 644 passed, 1 failed, `test_entries[m-t20-1.1x-dgrad-grow]`, on the node only. The route and the reference ended 5.5e-7 apart on `blocks.0.wq.weight`, against a bound of 1.8e-7: one 8-bit level.
- Here they end bit-identical at 1 to 16 threads (RAN `/tmp/dg/flip.py`).
- The reference merged θ + B·A while the route adds the adapter's own product. On that CPU the two round apart in a last bit, and the store's stochastic rounding turned it into a level.
- Fix (commit e65921ecf5): the reference adds x·Aᵀ·Bᵀ through a forward hook, as the route does; heap, codes, store and rules stay written out.
- Suites rerun: r20261113-085752-b441 at 6e15d4ba1a.

5. **Before and after** (your 3): c287 and b2a5 ran into `STAGE_TIMEOUT` at the slot's 3,600 s, and their output was buffered, so nothing is readable. I cancelled the two later whole-library runs for the same reason. Now running, `python -u`, six chunks a side, 12 GB each:
- **before** at main 9182900fb9 (its 32 entries): r20261113-090437-1d02, -090508-657a, -090539-ca77, -090608-7eeb, -090623-c835, -090652-d7df;
- **after** at 6e15d4ba1a (36 entries; q4muon-full as above): r20261113-090157-df0a, -090230-6379, -090246-f201, -090304-24f2, -090334-cf86, -090405-d556.

The draft opens once these and the suites are read. The ia3 branch carries dgrad's commits by a merge (no rewrite) and gets its own before-and-after for the norm commit after these land.

