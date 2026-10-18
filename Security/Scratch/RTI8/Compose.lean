import Proofs.Accounting.TS.Drawn
import Models.Accounting.Assumptions.Compose

/-!
# RT-I8 scratch: what Compose's `CircuitCredit` needs from sampled (I8)

Scratch for the red team of #2466 (never lands). Run from `Security/Proofs` with `lake env lean`.

`CircuitCredit` (compose-stmt, d469bc568c) takes #2466's pieces with joint soundness `δ_ks` and two accounting premises
with the same antecedent, `RightFixedDrawn` and `RightFoldedDrawn`. `ProofsSoundOfDrawn` takes per-draw `SoundAt` and
concludes `ProofsSound`, which exports no drawn-right fact, so the fold bound can't be read off it. The restated crypto
half (`drawn_right_off`: joint soundness, the drawn-right event exported) gives both bounds and the service's acceptance
off one event of probability `δ_bind + δ_ks`, at `CircuitCredit`'s own types.
-/

namespace RTI8.ComposeCheck

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

theorem drawn_right_off {K T L U A B : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype A] [Fintype B]
    {S s : ℕ} {J V : Type} (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (δbind δks : ℝ)
    (hOf1 : ∀ x, P.accepts x → ∀ w i, pf.accepted x w (pf.drawnAt x w i))
    (hB : ∀ w, prob (fun x => ∃ i, pf.Moved x w (pf.drawnAt x w i)) ≤ δbind)
    (hZ : ∀ w, prob (fun x => ∃ i, pf.Unsound x w (pf.drawnAt x w i)) ≤ δks) (hPin : pf.Pins) :
    ∃ E : Paths (A × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, P.accepts x → ¬ E x → ∀ w i, pf.right w (pf.drawnAt x w i) (pf.committed x w (pf.drawnAt x w i)) := by
  refine ⟨fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)), (∃ i, pf.Moved x w (pf.drawnAt x w i)) ∨
    (∃ i, pf.Unsound x w (pf.drawnAt x w i)), ?_, ?_⟩
  · calc prob _ ≤ ∑ w : Fin S, prob (fun x => (∃ i, pf.Moved x w (pf.drawnAt x w i)) ∨
          ∃ i, pf.Unsound x w (pf.drawnAt x w i)) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + δks) :=
          Finset.sum_le_sum fun w _ => (prob_or_le _ _).trans (add_le_add (hB w) (hZ w))
      _ = S * (δbind + δks) := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · intro x hx hE
    exact Accounting.SecurityProofs.TS.DrawnLayer.right_committed pf x (hOf1 x hx)
      (fun w i h => hE ⟨w, Finset.mem_univ _, Or.inl ⟨i, h⟩⟩) (fun w i h => hE ⟨w, Finset.mem_univ _, Or.inr ⟨i, h⟩⟩)
      hPin

/-- **At `CircuitCredit`'s types and premises** (one round on the call sites' coins, the service's permutation and the
proofs' coins `Ξ`; the service's game at share `1` per unit of skipped work; `pf.draw = drawOf hK`; `Of`, binding at
`δ_bind`, joint soundness at `δ_ks`, `Pins`, `RightFixedDrawn`, `RightFoldedDrawn`): one event of probability
`δ_bind + δ_ks` off which an accepted window is accepted by the service's game, its fixed words cover all but the
skipped work, and its folded words all but the whole work. -/
theorem compose_sampled {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C]
    [Fintype Ξ] (Cd : Declared ι) {Np Kd : ℕ} (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.calls)
    (stg : Fin Np → ℕ) (wt : Fin Np → ℝ)
    (Pl : Play (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer (Paths C Cd.calls) (Equiv.Perm (Fin Np) × Ξ) 1 Kd (Fin Np) V)
    (folded : Paths (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (Nw δbind δks : ℝ)
    (_hgame : Pl.game = serviceGame (Ξ := Ξ) stg W wt 1 1 Kd) (_hdraw : pf.draw = drawOf hK) (hOf : pf.Of Pl)
    (hB : ∀ w, prob (fun x => ∃ i, pf.Moved x w (pf.drawnAt x w i)) ≤ δbind)
    (hZ : ∀ w, prob (fun x => ∃ i, pf.Unsound x w (pf.drawnAt x w i)) ≤ δks) (hPin : pf.Pins)
    (hRF : RightFixedDrawn Pl pf Nw)
    (hRFo : RightFoldedDrawn Pl pf Nw (fun x => W.share stg wt (coinAt x 0).1 .whole) folded) :
    ∃ E, prob E ≤ δbind + δks ∧ ∀ x, Pl.accepts x → ¬ E x →
      Pl.game.accepts x ∧ (1 - Pl.game.deficiency x / ((1 : ℕ) : ℝ)) * Nw ≤ (Pl.run x).fixed ∧
        (1 - W.share stg wt (coinAt x 0).1 .whole / ((1 : ℕ) : ℝ)) * Nw ≤ folded x := by
  obtain ⟨E, hE, hoff⟩ := drawn_right_off Pl pf δbind δks hOf.1 hB hZ hPin
  exact ⟨E, by simpa using hE, fun x hx hx' =>
    ⟨hOf.2 x (hoff x hx hx'), hRF x hx (hoff x hx hx'), hRFo x hx (hoff x hx hx')⟩⟩

end RTI8.ComposeCheck

#print axioms RTI8.ComposeCheck.compose_sampled
