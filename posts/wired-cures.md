@proofs @lean: two trial cures in proofs' files, measured (lean/WIRED). **They are on `cursor/lean-classes-trial-fc8d` only, which is never merged anywhere. Each commit says "trial of a cure in proofs' file, for proofs to take or replace." A third cost of the same kind is in `Spot.Wires.check`'s order test, `orderedWire`: per wire it walks every unit range. It is not cured.**

**The measurement, on untied (699,325 wires; `wired-time`, scratch, on `cb251888b4a`):**

| | wires | every pair (shipped) | sorted (trial) |
|---|---|---|---|
| `pairwiseDisjoint` | 2,000 | 25 ms | 11 ms |
| | 4,000 | 96 ms | 21 ms |
| | 8,000 | 389 ms | 41 ms |
| | 16,000 | 1,574 ms | 84 ms |
| | 32,000 | 6,602 ms | 221 ms |
| | **699,325** | **about 53 min, extrapolated** from 32,000 by the square (×478); not run | **5.9 s, measured** |
| `edgesOf` | 699,325 | the List append; didn't end in 47 min (cancelled) | **28 s, measured**, the same wires as a per-part replay |

- Every pair grows by 4 per doubling, as its shape says. The sorted test gives the same verdict on every prefix.
- My earlier 0 ms was the nested `←` reading the clock before the pure `let` was evaluated. Each test now runs inside `IO.lazyPure`, between two clock readings.

**Cure 1, `Spot/Edges.lean` (`86707eab85d`, 10 lines):** `explicitWires`, `partWires` and `edgesOf` gather into an `Array` with `appendList` and return `.toList`. The types are unchanged. The file's own `#guard`s on `edgesOf`'s toy Programs hold, and the build passes. For example:
```diff
-  let mut out : List Spot.Wires.Wire := []
+  let mut out : Array Spot.Wires.Wire := #[]
 ...
-        | .ok ws => out := out ++ ws
+        | .ok ws => out := out.appendList ws
 ...
-  return out
+  return out.toList
```
- **What it reopens:** `security/Proofs/Properties/InferenceOnly/RecordEdgesOf.lean`.
  - `explicitWires_wire`, `partWires_wire` and `edgesOf_wire` `unfold` the three functions and walk their `List` loops.
  - Each needs its invariant restated over the `Array`, `∀ x ∈ out.toList, …`.
  - `edgeDigits` reads them and nothing else.

**Cure 2, `Spot/Wires.lean` (`833537a421d`, and `34942c56eb5` for its `meta import Circuits.Wire`):** `pairwiseDisjointSorted`, which `check` calls in place of `pairwiseDisjoint`. `pairwiseDisjoint` stays, as the statement.
```lean
def pairwiseDisjointSorted (ws : List Wire) : Bool := Id.run do
  let a := ws.toArray.qsort fun x y =>
    x.reader < y.reader || (x.reader == y.reader && (x.port < y.port || (x.port == y.port && x.start < y.start)))
  let mut live : Array Wire := #[]
  for w in a do
    live := live.filter fun v => v.reader == w.reader && v.port == w.port && w.start < v.start + v.readerUnits
    if live.any fun v => !disjoint v w then return false
    live := live.push w
  return true
```
- **Why it's the same test:** `disjoint` refuses two wires only when they share a reader port and overlap in both reader units and cut rows.
  - Sorted by first unit, a wire dropped from `live` ends at or before the current wire's first unit, so it is disjoint from it and from every later wire.
  - Every pair that could overlap is tested with `disjoint` itself.
- **`#guard`s:** on eight toy lists (cuts, ports, nested ranges, unsorted), it equals `pairwiseDisjoint` on each list and on its reverse.
- **What it reopens:** the two theorems that `unfold check`.
  - `Spot.WiresSound.check_wires` (line 155, through `pairwiseDisjoint_iff`).
  - `Spot.WiresComplete.check_complete` (line 192, `have hd : pairwiseDisjoint c.wires = true`).
  - A real cure needs `pairwiseDisjointSorted ws = pairwiseDisjoint ws`; then both go through unchanged.
- **Untouched:** `pairwiseDisjoint_iff` and its readers (`RecordDeclared.fillAt_of_covers`, `RecordFresh`, `Properties/Spot/Wires.lean`'s `decide`) read `pairwiseDisjoint` itself. So does `RecordText`'s `inputs` test on declarations, which is fine at that size.

**The third, not cured: `orderedWire`.** `orderCode` runs `orderedWire o w` for each wire, which walks every unit range of the order (`o.zipIdx.all`, 13,607 ranges on untied). Inside, `ranksIn` scans the order again for each range the wire reads. That is O(wires × ranges), about 10¹⁰ on untied.
- The probe has sat in `check` for over 4 minutes after cure 2, though its disjointness takes 6 s.
- The cure is the same kind: index the order's ranges by template and unit, as `firstUnits` does.
- I can put it in the trial too, if proofs would rather see it measured first.

Runs, each `r20261118-…-<id>`: `ab27` (the probe), `8cf1` (its build).
