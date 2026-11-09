[2026-11-09T00:31Z] da04701803eb
Coordinator 00:36Z: take (a). Change PooledLeakCarried in place. Its unwitnessed entry was never on main or checked, so regenerating it at the final statement is not a re-key of a landed entry; say so in the PR body. Note the reset rule is now PUBLIC (genesis at a first run, and after the verifier refused w-L), plus at most one prover-chosen reset a week. So #Cs is about 1,009 a week, not 2^m with a free bit per run. Charge exactly that.

[2026-11-09T00:36Z] 3e33e388ab77
From PRTRIAGE (bc-3d0ffbf3-da22-5402-aa59-eeee7c6060f6), 00:37Z, to MEMLEMMA and CARRYLEAN.

The coordinator reversed the reset bit at 00:06Z. cursor/proofs-poolreset-3e1c @ b962cedc69 (PR #8220) has the replacement, and Pool and poolOpen compile there. Its Lean try passed: r20261109-002259-7e74, Properties/Spot with --siblings, and the axiom probe found nothing beyond the standard three.

- **`Registration.Pool`** is `{pin, pout, genesis, refused, reset, spent}`, the last three `Bool`s. All of them are verifier params, fixed before the window's coins (`OneStage.poolOf`).
  - `refused`: the verifier refused the window L windows of record back, at a fixed public L. MEMLEMMA: the verifier holds that verdict here, as a Pool field, and doesn't read it from the log.
  - `reset`: serving's own reset for an engine crash. The record's window names it (`R6-pool-reset`).
  - `spent`: an earlier window of the week already reset.
- **`Pool.chains p log program`** is `match log.back? with | some e => !p.reset && !p.refused && jeq (getD e "program") (.str program) | none => false`.
- **`poolOpen` is back to the if-shape**: `if p.chains v.log (hexOf sha512) then` the previous record's `pout` entry, `else` genesis's root. The `else if !p.reset then none` branch is gone, so `poolOpen_of_chains` should be rfl again.
- **R6** gains `PoolResetOnce v := (match v.pool with | some p => !(p.reset && p.spent) | none => true) = true` (`R6-pool-reset-week`): at most one chosen reset a week. `PoolChained` and `R7` are textually unchanged.
- **CARRYLEAN, for `Cs`:** the pattern is the public rule (a first window, a boundary, a refusal at lag L) plus at most one chosen reset a week. For a week of n windows that is at most n + 1 patterns a week, log2 1009 = 9.98 bits a week for 600 s windows, instead of 2^m. On the event that every run accepts, `refused` is false everywhere, so there the public rule is only the first window and boundaries.

