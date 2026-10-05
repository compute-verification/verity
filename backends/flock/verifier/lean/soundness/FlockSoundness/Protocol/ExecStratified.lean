import FlockSoundness.Protocol.ExecDraw
import FlockSoundness.Protocol.Audit.Stratified
import FlockSoundness.Protocol.E2E

/-!
# The executable stratified draw, and the e2e theorems at its law

`Flock.Draw.stratified` is the verifier's executable stratified draw: `Flock.Draw.subset` on each stratum's own stream,
local indices mapped to the stratum's units (`Stratum.unit`), and the union sorted, or `none` if a stream runs out.
`ExecStrata σ k S` says the executable strata `S` are the model's: the same `k`, the same sizes, and each stratum's
units run over `σ`'s stratum.

* `stratified_exec_escape_le`: over independent uniform streams, the executable draw misses `B` with probability at most
  `Law.stratified`'s escape, the product over strata of `subset_exec_escape_le` (`ExecDraw.lean`).
* `execStratified`: the executable draw as a law, a draw that runs out drawing every unit. `execStratified_escape_le`: it
  misses each set with at most `Law.stratified`'s escape. `execStratified_miss_le`: its count curve is at most
  `Law.stratified`'s, which `stratified_miss_eq_greedy` computes.
* `flock_e2e_drawn_exec`, `flock_e2e_count_exec`: `flock_e2e_drawn` and `flock_e2e_count` (`E2E.lean`) at that law, the
  count curve bounded by `Law.stratified`'s `miss`.

The streams are uniform by assumption A3 (`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`). The live draw,
`Flock.Draw.drawOS`, extends a stream that runs out and reruns, rather than drawing every unit; `ExecDrawOS.lean` proves the
same bounds for it.

`stratumK` is the count law's `k` by `rfl`: `countRule_eq_draw` (`Audit/FlockWork.lean`).
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- The executable strata `S` describe `σ`'s: stratum `s` draws `k s` of its units, has `σ`'s stratum's size, and its
`j`-th unit, for `j` below that size, runs over `σ`'s stratum `s`. -/
def ExecStrata (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum) : Prop :=
  ∀ s, (S s).k = k s ∧ (S s).size = (stratum σ s).card ∧
    (range (S s).size).image (S s).unit = (stratum σ s).map Fin.valEmbedding

end Law

namespace Partition

open Model Accounting Extract LinkFinder

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

end Partition

end FlockSoundness.Audit
