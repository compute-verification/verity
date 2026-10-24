import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Data.Fintype.Pi

/-!
# The run tree of a prover's wipes (trusted)

The abstract run tree the memory cap's lifetime pool (the memory-cap restatement's step 7, "One delta") is a lemma
over: over a prover's whole life, every wipe is `k` timed rounds, each challenging a block drawn uniformly from the
wiped store `B` by the verifier's fresh coins (the deployed challenger's index stream is a PRF of its live seed,
`prf/sha-512`, which the uniform draws idealize). A wipe's *un-held set* `U ⊆ B` is a label on the tree, fixed by
everything before the wipe's coins, so its *un-held fraction* `x = |U| / |B|` lies in `[0, 1]`; nothing here ties `U`
to what the prover holds.

* `WipePlay k`: one prover strategy as a finite run tree. `wipe B U pass next` is a wipe over the store `B` whose
  un-held set `U` and pass `pass` the history fixes before its `k` draws `ω : Fin k → B`; a wipe that passes continues
  with `next ω`, one that fails ends the run, caught. `coin` is any other verifier coin, and `stop` ends the run with
  every wipe passed. A tree is finite, of any depth, so it covers any number of wipes, each choosing `U` adaptively.
* `prEv E P a`: the probability that every wipe of `P` passes and the run's un-held total `a + Σ_t x_t` satisfies `E`.
* `UnheldFails`: at every wipe, a draw that challenges a block of `U` fails it, the pool lemmas' hypothesis
  (`Accounting.SecurityProofs.Actual.WipePool`). Every tree meets it once each `U` is empty, so it is no physical
  premise; the pool's physical premise is memory-accounting's restatement on the challenger's verdicts (MEM-55).
-/

namespace Accounting

open scoped ENNReal

/-- **One prover strategy over wipes of `k` rounds**, as a finite run tree. `wipe B U pass next` is a wipe over the
store `B`, its un-held set `U` and whether it passes at each draw `ω : Fin k → B` fixed by the history before its
coins; a passing wipe continues with `next ω`. `coin` is any other verifier coin; `stop` ends the run. -/
inductive WipePlay (k : ℕ) : Type 1 where
  | stop : WipePlay k
  | coin (C : Type) [Fintype C] [Nonempty C] (next : C → WipePlay k) : WipePlay k
  | wipe (B : Type) [Fintype B] [Nonempty B] (U : Finset B) (pass : (Fin k → B) → Bool)
      (next : (Fin k → B) → WipePlay k) : WipePlay k

namespace WipePlay

variable {k : ℕ}

/-- **The un-held fraction** `x = |U| / |B|` of a wipe over the store `B` with un-held set `U`. -/
noncomputable def unheld {B : Type} [Fintype B] (U : Finset B) : ℝ := (U.card : ℝ) / Fintype.card B

open Classical in
/-- **The probability that every wipe passes with the un-held total in `E`**: the probability over the verifier's
uniform coins that the run reaches `stop`, every wipe passed, with `E (a + Σ_t x_t)`, the fractions summed over the
wipes played. -/
noncomputable def prEv (E : ℝ → Prop) : WipePlay k → ℝ → ℝ≥0∞
  | .stop, a => if E a then 1 else 0
  | @WipePlay.coin _ C iC _ next, a => ∑ c : C, (@Fintype.card C iC : ℝ≥0∞)⁻¹ * prEv E (next c) a
  | @WipePlay.wipe _ B _ _ U pass next, a => ∑ ω : Fin k → B, (Fintype.card (Fin k → B) : ℝ≥0∞)⁻¹ *
      if pass ω = true then prEv E (next ω) (a + unheld U) else 0

/-- **Every wipe fails at a draw that challenges an un-held block**: at every wipe, a draw `ω` that passes
challenges no block of `U`. -/
def UnheldFails : WipePlay k → Prop
  | .stop => True
  | @WipePlay.coin _ _ _ _ next => ∀ c, (next c).UnheldFails
  | @WipePlay.wipe _ _ _ _ U pass next => (∀ ω, pass ω = true → ∀ i, ω i ∉ U) ∧ ∀ ω, (next ω).UnheldFails

end WipePlay

end Accounting
