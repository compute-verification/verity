import Definitions.Pouw.Guarantees.DitherInstance
import Proofs.Pouw.RelativeSecurity

/-!
# The headline on the cards from the headline on the model

* `workSoundAt_of_workSound`: `WorkSound` gives `WorkSoundAt c pb` on every card table: an instance of its quantifiers
  at the counted and timed verdict, which accepts no more than the service's verdict, with the import bound from the
  certifier's counts. The rounds' work fits the period from the deadline: the certifier's clock bounds each round's
  work to `Δ` of wall time, G2 bounds its use by `ρ_r Δ`, and `k Δ ≤ T`.
* `workSoundH200_of_workSound`, `workSoundSm120_of_workSound`: the two instances.
* `h200_floors`, `sm120_floors`: θ's floors with the product form are met by some open parameter on each card, so
  `WorkSoundAt`'s hypotheses on `x` are not empty.
-/

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance

open scoped NNRat

/-- `WorkSound` gives the headline on every card table. -/
theorem workSoundAt_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) (c : Verity.Accel.Card)
    (pb : PubBest c) : WorkSoundAt c pb := by
  intro x _ _ D P L G e _ k verdict Pd μ Hmm Hsha Ho η δ nmax macs Ein Eout κS κx Δ t cin cout arrive
    hdual hmm hsha ho hH hpm hps hη hδ hδ1 hn hm hEi hEo hkΔ g1 g2 p1 p2 sak sha svc sees clock ext
  have hβ : 0 ≤ nmax * Hmm / macs := div_nonneg (mul_nonneg hn hmm.le) hm.le
  have hK : 0 ≤ Ein + Eout := add_nonneg hEi hEo
  have svc' : ServiceSound G P k (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) κS := fun A =>
    le_trans (prob_mono fun ox hox => ⟨hox.1.1.1, hox.2⟩) (svc A)
  have imp : ImportWithinCertifier G P k (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) (Ein + Eout) :=
    fun A o x hv => (sees A o x).trans (hv.1.2.1.trans (le_add_of_nonneg_right hEo))
  refine ⟨h G P k _ Pd μ (pubCost e μ pb.mm) (pubCost e μ pb.sha) Hmm Hsha Ho η δ (nmax * Hmm / macs) (Ein + Eout)
    κS κx hdual hmm hsha ho hH hpm hps hη hδ hδ1 hβ hK g1 g2 p1 p2 sak sha svc' imp ext, ?_⟩
  intro A o x hv r
  have hrate : 0 ≤ G.rate r := (g2.1 r).le
  have hw : ∀ w, G.use (A.work w) (A.workIn o x w) r ≤ G.rate r * Δ := fun w => by
    have h1 := ((g2.2 (A.work w) (A.workIn o x w)).2 r).2
    have h2 : G.wall (A.work w) (A.workIn o x w) ≤ Δ := by
      have := clock A o x w
      have := hv.2.2 w
      linarith
    exact h1.trans (mul_le_mul_of_nonneg_left h2 hrate)
  show _ ≤ G.rate r * Pd.T
  calc ∑ w, G.use (A.work w) (A.workIn o x w) r ≤ ∑ _w : Fin k, G.rate r * Δ :=
        Finset.sum_le_sum fun w _ => hw w
    _ = G.rate r * (k * Δ) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
    _ ≤ G.rate r * Pd.T := mul_le_mul_of_nonneg_left hkΔ hrate

theorem workSoundH200_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundH200 :=
  workSoundAt_of_workSound h _ _

theorem workSoundSm120_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundSm120 :=
  workSoundAt_of_workSound h _ _

/-- An open parameter at θ's floors: every dot, dense and sparse, the constant zero (in product form), `sync = 1`,
`b1 = 1/8`, `madHi = 2`, no backlog. -/
def floorOpen : Verity.Accel.Open where
  slack := 0
  b1 := 1 / 8
  sync := 1
  madHi := 2
  dot _ _ _ _ := fun _ _ _ => 0
  dotSp _ _ _ _ := fun _ _ _ => 0

theorem floorOpen_productForm : Verity.Assumptions.AccelProductForm floorOpen :=
  ⟨fun _ _ _ _ => ⟨fun _ _ _ _ => 0, fun _ _ _ _ _ => rfl⟩, fun _ _ _ _ => ⟨fun _ _ _ _ => 0, fun _ _ _ _ _ => rfl⟩⟩

/-- The H200's floors with the product form are met. -/
theorem h200_floors :
    ∃ x, Verity.Accel.H200.card.floors x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨floorOpen, ⟨le_rfl, le_rfl, le_rfl⟩, floorOpen_productForm⟩

/-- The RTX PRO 6000's floors with the product form are met. -/
theorem sm120_floors :
    ∃ x, Verity.Accel.Sm120.card.floors x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨floorOpen, le_rfl, floorOpen_productForm⟩

end Pouw.SecurityProofs.DitherInstance
