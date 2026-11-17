module

public import Proofs.Flock.Soundness.Audit.Influence
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fintype.BigOperators
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Game

/-! Exfiltration, at most what is delivered. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n)
variable {L : SamplingDesign n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
  {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
  {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
  {a : Fin C.N → Bool} {εks εbind δin : ℝ≥0∞} {D : Finset (Fin C.N)}

theorem audit_exfiltration_delivered {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (hks : (P.analysis X ext).KnowledgeSound εks) (hlink : (P.analysis X ext).BindSound εbind)
    (hin : AnchorsSound X a δin) (hD : D ⊆ P.committed) (hb : Fin n → ℕ) (K : Finset (Fin n) → Finset (Fin C.N))
    (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D) (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u)
    {δ : ℝ≥0∞} {H k : ℕ} (hH : IsHarmBound L (fun u => (hb u : ℝ)) δ H) (hk : L.miss (k + 1) < δ)
    (σ : M → Strategy (audit L Reg session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (X (reg (σ m)) (cont (σ m)))) = m)
        (audit L Reg session) (σ m)) ≤
      ((min (2 ^ H * ∑ j ∈ range (k + 1), n.choose j) (2 ^ D.card) : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + εbind + δin) := by
  classical
  refine (P.audit_exfiltration_influenceSet hks hlink hin δ σ g).trans ?_
  have hcount := (P.card_influenceSet_le_harm (a := a) hD hb K hK hsep hw hH).trans
    (Nat.mul_le_mul_left _ (card_admissible_le hk))
  have hall : (P.influenceSet D L.missProb a δ).card ≤ 2 ^ D.card :=
    (card_le_univ _).trans (le_of_eq (by rw [Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]))
  have hmin := le_min hcount hall
  gcongr

end Partition

end FlockSoundness.Audit
