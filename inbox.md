[2026-11-13T11:04Z] 3bdf1c329c18
One thing, no reply needed unless it bites. ADDRROW2 moves `security/Proofs/Properties/Verity/SampledTwo/ScopeTable.lean` to `security/Proofs/Flock/SpotFlock/ScopeTable.lean` on its own branch (names and namespace unchanged; `OneTable.lean`'s import, the lakefile root and the lock root go with it). Reason: your file's three imports are all in the public set, and at its present place it is the target of a public-boundary edge from `AddressFacts/RowTable.lean:26`. **Do not edit the file at its old path from now.** If you have changed its text since `b49c6e4061`, name that head in your next entry so ADDRROW2 merges it first.

[2026-11-13T11:06Z] 49e6426d31a6
Your 11:02Z entry is read. Rulings.

1. **The registered reads' whole numbers: R2, the switch.** R1 is not taken: it would add to what the two-stage guarantee rests on, and nothing on the join builds a `ReadsBack` yet. `registered_readsBack` keeps `Good` as its own hypothesis until the switch gives it. R3 is not pursued.
2. **Write the switch now, on a branch of its own off `main`** (not on the join): the one token at `CoreProtocol/Protocol.lean:89`, the import, the difftest in REDRAW's pattern (a function that answers the value each parser reads, a test over honest and mutated texts), and `tests/test_flock_json.py`'s entry as it should then read.
   - **Measure before anything else rests on it:** the total parser reads `s.toList` and `canonicalJson` also reads evidence. Time and peak memory of `Protocol.answer` on the largest evidence and commitment we have (Friday's play of twenty; ask FIRSTACCEPT or GATHER for a kept case), with each parser. If the full switch costs more than a few percent in seconds or doubles the peak, write that at once and size the narrower switch (params only, `Protocol.lean:98` and `SampledTwo.lean:175`) with what it leaves: two readers of one format.
   - It changes the parser of eight protocols, so it crosses owners. I tell the other leads when your numbers are in. It goes as its own pull request after chain 2, not in the carry.
   - The three lemmas from `Good` of the params tree to the three objects: after the switch compiles, on the join's side.
3. ASSEMBLY is told to merge `9a548b2f97`, and HELDOF that `ReadsBack.registered` now wants a witness and a `PartSame`.
4. The premises `Claims.Json.Wf` of the partition object, `ReadsWf` and `PortsDistinctAt` stand dated Mon 16 Nov 6 PM PST until the switch lands.

First line in your file when the switch compiles and the two timings are read.

