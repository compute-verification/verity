[2026-11-13T17:56Z] cf65fd08e54a
**Coordinator to SEARCH**, Fri 13 Nov 9:55 AM PST (17:55Z). **A change to your task before your first report: four finders, not five. `HashedOpenings.finder` is out of your pull request until soundness rules on it. SIZER has read all five; start from its reading and check it.**

- **Why:** `HashedOpenings.finder`'s slots `k, k'` range over `Q`, and the statement of `Pouw.SecurityProofs.MonitoringPeriodBindingPerProver` sets `Q = ℕ` (`security/Definitions/Accounting/Guarantees/PeriodBinding.lean:53`). No finite list holds its candidates, and a bound changes that guarantee's statement. It is with soundness and compute-accounting on PRF-490. Do not touch it. In your first report say only whether you read it the same way.
- **SIZER's reading of the other four** is its comment of 9:46 AM PST on LEAN-306: section 5 has the search, the lemmas that unfold each and the callers; section 3 has the listed guarantees that read each. In short:
  - `outOf`: `List.finRange K` and `find?`; `outOf_collision` breaks; no listed reader.
  - `outOfAt`: `[Fintype P]` and a `find?` over `List.finRange K ×ˢ univ.toList`; its four callers infer the instance at `Fin (L.nG x)`; `outOfAt_collision` breaks; read by `Flock.SecurityProofs.SpotFirstHitTwo` through `finderXCls` and `finderICls`.
  - `Reads.finder`: `Claims.findColl` over `drawnHashed`, which stays as it is; `hfind` in `readUnitDrawnCR` breaks; read by `Accounting.SecurityProofs.TableReadBinding` and its `_nontrivial`.
  - `Receipts.finder`: the receipts sorted, then `findSome?` and `findColl`; `fork_le_finder` breaks; read by `Accounting.SecurityProofs.Remote.ReceiptsBindingOfCR`.
- **These are read, not compiled.** Where your reading differs, yours goes in the report with the line.
- **`findColl`'s lemmas** (`findColl_isSome`, `findColl_spec`) sit under `security/Proofs/Flock/Claims/`, with a copy in Pous [SIZER]. If you need them below Flock, say in the first report where they would go and what else moves. Do not write a third copy.
- **Cost:** leave each finder's `cost` as it is. soundness is asked on PRF-490 whether a constant charge meets its rule; if `cost` must count candidates, that is a statement change and comes later.
- The done-condition is as written, for the four. The issue's text now says so.

