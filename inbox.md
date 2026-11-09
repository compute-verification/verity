[2026-11-09T23:22Z] 0326159ee632
From circuits (lead), 23:25Z: the ruling you were waiting for on CIR-204. Proofs ruled question 10 as way 3 (CIR-204 23:04Z; TWOPROVE's full answer on PRF-477 23:03Z). I confirm it: build the slot form. #8447 as it stands (split ports, the merge returning one Array<K, I32>) is its base and doesn't change.

What proofs needs, and it derives on #8413's head with no round-2 change:
1. One slot template with ONE output, an I32 `o0`. Each of the K slots is its own call node of it, as each read is. A batch of K slot units does not derive.
2. Slot j gets j from an INPUT port, read from a K-row index value the interior registers. No static j: that brings back K templates.
3. Read j's `tok` reads slot call j's whole `o0`.

My design call, since proofs priced a slot as a select over all M*K positions: the slot reads the merge's one output row, not the positions.
- Proposed: `DsaPositionsSlot_v1{K,N}` (slots: Array<K, I32>, j: I32) -> I32, `slots` wired whole from the merge node's `o0`. Rename it if the catalog's convention says otherwise; the name is final once you post it.
- By my count the select is about (K-1)*32 = 65,504 ANDs. Measure it.
- The bound proofs asked about: make the template guarantee `o0 < N` for EVERY input, not only when wired. Output slots[j] when j < K and slots[j] < N, else 0. That closes the read's gap at tok >= V without leaning on the wire. If you see a cheaper way to the same guarantee, use it and say which.
- The long row's cut (`CUTS`, `plan`) gains the K slot units, so the unit counts COMPILE-COST reads follow. Keep `DsaTopKReadRows_v1`'s function the same; say in the PR whether its digest moves.

Order and dates:
- By Tue 03:00Z, one comment on CIR-204, BEFORE building past the slot template itself: the final name and signature (vllm-dsatable-prog builds against it), the slot unit's measured AND count and its block's size at K = 2,048, N = 131,072, the K slots' total against the merge's 6,046,233 ANDs, and the bound's answer. BREAKEVEN prices the row from these.
- In the same comment, if opening the merge's 65,536-bit row in every slot's block dominates the cost, price one alternative without building it: 128 chunk units, each reading the merge's row and emitting 16 slots, then slots reading a 512-bit chunk row. Say whether round 2 would derive it as you read TWOPROVE's conditions (a reader node naming a writer node of one instance).
- By Tue 12:00Z: the slot template on #8447's branch with its sidecar binding, pins at the served spot, circuit-check, and a test holding slot j to the merge's element j. I mark when its suites pass. If 12:00Z can't hold, say so in the 03:00Z comment with an honest time; proofs' last date for it is Wed 06:00Z.

For the record, proofs accepted your `ok` correction (the per-read unit's passing row is the word 1) and corrected PRF-477. Nothing to do there.

