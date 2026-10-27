import Definitions.Accounting.Guarantees.TS
import Proofs.Pouw.RelativeSecurity

/-!
# (I8) from its pieces

`Accounting.SecurityProofs.TS.ProofsSoundOfPinned : Accounting.Guarantees.TS.ProofsSoundOfPinned`: the event `E` of
`ProofsSound` is "some round's binding fails or some proof is unsound", whose probability the union bound
(`prob_exists_le`) puts at `S·(δ_bind + n_π·δ_ZK)`. Off it, on an accepted path, every proof of every round is accepted
(`Of`), so each opens its round's leaves to the committed values (not `Moved`) and its opening satisfies its relation
(not `Unsound`): the committed values satisfy every relation, the ideal game accepts (`Of`), they are right (`Pins`),
and the fixed words follow (`RightFixed`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- Off binding failures and unsound proofs, an accepted path's committed values satisfy every proof's relation. -/
theorem rel_committed {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (x : Paths Ω S)
    (hp : ∀ w, pf.Proved x w) (hM : ∀ w, ¬ pf.Moved x w) (hU : ∀ w i, ¬ pf.Unsound x w i) (w : Fin S)
    (i : Fin nπ) : pf.rel x w i (pf.committed x w i) := by
  have heq : pf.opened x w i = pf.committed x w i := by
    by_contra hne
    exact hM w ⟨hp w, i, hne⟩
  rw [← heq]
  by_contra hr
  exact hU w i ⟨hp w i, hr⟩

/-- (I8) from binding at `δ_bind`, proof soundness at `δ_ZK`, the Definitions' pinning and E1(c). -/
theorem ProofsSoundOfPinned : Accounting.Guarantees.TS.ProofsSoundOfPinned := by
  intro K T L U V _ _ _ _ Ω _ S nπ P pf Nw δbind δzk hOf hB hZ hPin hRF
  refine ⟨fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)),
    pf.Moved x w ∨ ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i, ?_, ?_⟩
  · have hw : ∀ w, prob (fun x => pf.Moved x w ∨ ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i) ≤
        δbind + nπ * δzk := fun w => by
      refine (prob_or_le _ _).trans (add_le_add (hB w) ((prob_exists_le _ _).trans ?_))
      calc ∑ i : Fin nπ, prob (fun x => pf.Unsound x w i) ≤ ∑ _i : Fin nπ, δzk :=
            Finset.sum_le_sum fun i _ => hZ w i
        _ = nπ * δzk := by simp
    calc prob _ ≤ ∑ w : Fin S, prob (fun x => pf.Moved x w ∨
          ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + nπ * δzk) := Finset.sum_le_sum fun w _ => hw w
      _ = S * (δbind + nπ * δzk) := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · intro x hx hE
    have hM : ∀ w, ¬ pf.Moved x w := fun w h => hE ⟨w, Finset.mem_univ _, Or.inl h⟩
    have hU : ∀ w i, ¬ pf.Unsound x w i := fun w i h =>
      hE ⟨w, Finset.mem_univ _, Or.inr ⟨i, Finset.mem_univ _, h⟩⟩
    have hrel := rel_committed pf x (hOf.1 x hx) hM hU
    exact ⟨hOf.2 x hrel, hRF x hx (hPin x _ hrel)⟩

end Accounting.SecurityProofs.TS
