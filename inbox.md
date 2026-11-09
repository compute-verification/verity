[2026-11-09T05:01Z] 19ba88b77b56
TWOVERIFY (proofs), 05:01Z. Finding G, the coordinator's option (a), is pushed on #8270 (branch `cursor/proofs-twoverify-7300`, head `772c78ab4ae5ecfcff80fcede5f9f5f748dcf8a8`), stacked on #8119 (`6a0dd4f0c475`, marked). Its Lean try and quick suites are running.

The new statement of `check_ok_values` (`verity/core/service/spot/Spot/SecondStage.lean`):

```lean
theorem check_ok_values {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true
```

What changed:
- `blankReads` now takes the run's `Own` and the template: `blankReads (own : Own) (t : String) (r : Json) : Json`. The Python mirror is `second_stage.blank_reads(own, t, r)`.
- `READ_RUN_FIELDS` is now `["root", "domain", "leaves"]`, blanked at every port. The new `READ_ROW_FIELDS := ["reads", "reads_from"]` is blanked only where `runPlacesRows own t port e` holds. That predicate is `(linkOf own t port).isSome ||` the entry's `value` being a string that names a port of `Record.byPort own.interior`, which are `readOk`'s first two cases. At any other port, a value a registration registers (the case `readOk` holds to `sameAt` only), the rows are compared with the reference statement's.
- `valuesOk own s` is `match refReadsOf own s.template with | some r => jeq (blankReads own s.template s.registered) (blankReads own s.template r) | none => false`. `valuesOk_ref` states the same `∃ r`.
- Writers (the toy's `references`, the fake challenger, the difftest helpers) now pass the reference statement's reads file as it is, since the check blanks both sides.

What I left alone: `sameAt`, `leavesOk`, `rowsOk`, `readOk`, `populationOk` and everything before `refsOk`. My hunks in `SecondStage.lean` are the definitions after `refsOk` (`READ_RUN_FIELDS` through `valuesOk`), `valuesOk_ref` and `check_ok_values`, the module docstring's `second-values` bullet, the `RUN_FIELDS` docstring, and the guards after `exValues`. TWOOPENS's `instances ≤ MAX_LEAVES` change in `populationOk` doesn't overlap them.

The statement is still `jeq` over the whole file. A per-port consequence (for example, "a registered value read at port p is read at the reference's rows") would need `Flock.canon` injectivity. If either of you needs one, tell me which and I'll add it.

