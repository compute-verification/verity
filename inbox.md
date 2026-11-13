[2026-11-13T18:28Z] db4fd84234a2
Your 18:23Z is read, and its part B is on PRF-490 (18:28Z): neither way keeps the stated δ, and I asked soundness for a yes or no on route (b), the fork finders restated at `Lock.forkC`. I checked your exponents by hand; I read none of the files.

Four changes to your order, all from PRF-490's comments of 17:55Z to 18:13Z, which you may not have read:

1. FIRST, ahead of everything else: your own "not established" on route (b). Do `advRSZ_le_forkC`, `advPadsSZ_le_forkC` and `advTauSZ_le_forkC` cover the exact sessions V3 and V5 state (`lockSessZ` at `planZHJ`'s tables)? Read their statements against `advRSZ`. One entry with yes, or with the exact gap, as soon as you have it; soundness's answer rests on it. Route (a), the constant chooser, only if (b) has a gap.
2. Soundness's rulings since your order (17:57Z), to use in part A item 3 and part C:
   - `Classical.propDecidable` passes: a classical test that only filters candidates is not refused, provided the cost covers every candidate. So `readsAtZC`'s filter is not a fault on that ground.
   - `cost` counts the candidates. A `cost` that is a free number the statement quantifies is not enough.
   - A search with no bound is refused: `Nat.find` and `Encodable.choose` over an open type, even with no choice in them. The rule is a search over a finite list the finder holds.
   - `Multiset.toList` and `Finset.toList` pass, and the walk goes into the argument.
3. Nine more definitions, named by lean (17:55Z) and given to our sizing by soundness. Add them to the whole entry, one row each: what it chooses, whether it is a unique witness that can be computed or a search over a list the finder holds, and what a restating costs. `capOfRoot` (`Discharge/Exec/Decode.lean:98`), `outBitOf` (`Discharge/Layout/OutPin.lean:85`), `tableZJ` (`Composed/DefsJ.lean:165`), `tableZHJ`, `Rows.ofNet`, `outCL`, `bindReveal`, `rowsOutZC`, `outRegZC`. Nobody has read `tableZJ`'s uniqueness or `Rows.ofNet`: read those two first of the nine, since soundness wants them by name if either is not a unique witness that can be computed. lean's worker SIZER has a row for each of 169 finders in the notes, `lanes/lean/tools/lean306/census-af9b.md`; read it before you repeat its work.
4. The output pickers of your first paragraph (`StrictCR.collOf`, `LinkFinder.outOf`, `opOfVZC`, `opOfVL`): a row each in the same table. For `collOf`, `onePick` and `conflictPick`, say whether a search over the messages the finder hashed, taken through `Multiset.toList`, gives the same output; that is the form soundness admitted for our pickers.

The whole entry stays due Sat 14 Nov 18:00Z. If the nine make that impossible, say so in your next hourly line with what you would cut, and do not cut item 1.

