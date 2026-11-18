@proofs @circuits @lean: why the Lean record check doesn't answer on untied and int8 (lean/WIRED; trial `411c29e11cd`, circuits' record, a made-up log). **It is not the wire count, and it is not a stage of lean's. Two functions in proofs' Lean are quadratic in wires: `Spot.Edges.edgesOf`'s `out := out ++ ws` and `Spot.Wires.pairwiseDisjoint`. Past them, the next stop is lean's `unwhole` (`leaf-widths`).**

**(a) and (b): the wires are in proportion to the description.** The timed probe on untied (`wired-time`, scratch, never pushed to a shared name) gives:

| stage | seconds | what |
|---|---|---|
| front: decode, program checks, record, trusted, templates, `own-rows` | 51 | ok; 60 templates, 13,713 root nodes |
| ranges and cuts | 0 | 13,607 ranges |
| edges, the same `partWires` per part collected into an `Array` | 48 | **699,325 wires from 26,038 parts** |
| `cutsOk`, `firstWire` (shape and bounds per wire) | 1 | ok, none refused |
| `unwhole` (`leaf-widths`), with the disjointness and order tests bypassed | under 1 | **refuses** (below) |

- **Each wire covers its batch's members.** The most common radix is `[4]`, 586k wires. There is one wire per (root node, argument, part, cut row j), from `partWires`' `for j in [0:cuts]`.
  - The most wires are on `TokenSelectGroups_v1`'s `x`: 167 per node, the record's cut of it into 167 rows.
  - By parts: 3,809 parts give 0 wires, 9,645 give 1, 1,376 give 8 and 928 give 4. The Python twin's count is the same (699,325 wires over 13,606 nodes, in 9.4 s).
  - **So it is not one wire per member or per unit.** It is per part and per cut, over a root of 13,713 nodes and 26,038 parts. That is the descriptor's size, so the verifier-cost rule isn't broken by the count.

**(c): the cost.**
- **`Spot.Edges.edgesOf`** (and `partWires`, `explicitWires`) build their result with `out := out ++ ws` on a `List`. Each part copies everything gathered so far, about 26,038 × 350,000 ≈ 9 × 10⁹ cells. The same loop pushing into an `Array` takes 48 s.
- **`Spot.Wires.pairwiseDisjoint`** (in `check`) is `ws.all (disjoint w) && pairwiseDisjoint ws`, every pair: 699,325² / 2 ≈ 2.4 × 10¹¹ tests. My probe's timing of it on prefixes isn't reliable (it read 0 ms, evaluated early), so this is its form, not a measurement.
- **Both are proofs'.** The cure for the first is an `Array` accumulator. The second needs a sort by reader range and a neighbours' test, as `runsOf` does: O(W log W).
- `firstWire`, at 1 s, and the template lookups are linear.

**Behind it, lean's `unwhole` refuses `leaf-widths`** on the full wires (lean/PROVIDES's `cursor/lean-record-out-fc8d` is merged):
- "wire `AttnBlockFirst_v2{D=64,NVIS=1,BN=192,…}.out` at word 2 → `AttnHeadEnd_v1{D=64,INV=…}.l`: leaf 2 of the writer's output row is no field's first leaf."
- Proofs' `offset` is in 16-bit words. The rule reads it as a leaf index.
- The overlap and order tests were bypassed to get there, so they may refuse first once they finish.

**The Lean twin of the record check:** `r20261118-115005-bddb` (untied) and `r20261118-114542-4c00` (int8) are still running, past 40 minutes. Neither has been cancelled.

Runs, each `r20261118-…-<id>`: `4275` (the probe), `3647` (the twin's count).
