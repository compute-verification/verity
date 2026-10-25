import Properties.Pous.Obligation
import Definitions.Pous.Guarantees.Obligation
import Proofs.Pous.Period
import Proofs.Pous.P2.Rounds
import Proofs.Pous.Thm1.Theorem1ii

/-!
# Proofs of PoUS as an obligation: tag freshness, the snapshot reduction, and the per-window guarantee

For an injective snapshot `tag`, reading a function on tags at the snapshot and off it is a bijection (`splitEquiv`),
so the store's oracle with any other coins is, as a uniform sample, the other tags' key pairs beside the snapshot's
primitive and those coins (`oracleEquiv`). An event bounded at every value of the other tags' keys is then bounded over
the whole oracle (`pr_snap_le`, and `pr_snap_le_real` for `Pouw.pr`). The window's bounds are that at each value: P2's
graded bound (`PousP2Rounds.graded`) for a payload and responder fixed by those keys, its chain over the rounds under the
committed-content conjecture, and `PeriodHeld` at the snapshot for the deployed check. A block's offset absorbs its
content by translating the offset's element by the difference of the contents' elements.
-/

namespace Pous.Mutable.SecurityProofs

open Pous Pous.Mutable
open scoped ENNReal

variable {N : ℕ} {T : Type}

open Classical in
/-- For an injective snapshot, reading a function on tags at the snapshot and off it is a bijection. -/
noncomputable def splitEquiv {K : Type} (tag : Fin N → T) (htag : Function.Injective tag) :
    (T → K) ≃ (Fin N → K) × (Rest tag → K) where
  toFun f := (fun j => f (tag j), fun t => f t.1)
  invFun y t := if h : ∃ j, tag j = t then y.1 h.choose else y.2 ⟨t, fun j hj => h ⟨j, hj⟩⟩
  left_inv f := by
    funext t
    by_cases h : ∃ j, tag j = t
    · simp only [dif_pos h]
      exact congrArg f h.choose_spec
    · simp only [dif_neg h]
  right_inv y := by
    refine Prod.ext (funext fun j => ?_) (funext fun t => ?_)
    · have h : ∃ j', tag j' = tag j := ⟨j, rfl⟩
      simp only [dif_pos h]
      exact congrArg y.1 (htag h.choose_spec)
    · have h : ¬ ∃ j, tag j = t.1 := fun ⟨j, hj⟩ => t.2 j hj
      simp only [dif_neg h]

/-- The store's oracle beside other coins `Z`, as the other tags' key pairs beside the snapshot's primitive and `Z`. -/
noncomputable def oracleEquiv {w p : ℕ} [NeZero p] (tag : Fin N → T) (htag : Function.Injective tag) (Z : Type) :
    Oracle w p T × Z ≃ RestKeys tag w p × ((ringRO N w p).Ω × Z) where
  toFun y := (rest tag y.1, (snap tag y.1, y.2))
  invFun x := ((x.2.1.1, (splitEquiv tag htag).symm (x.2.1.2, x.1)), x.2.2)
  left_inv y := by
    obtain ⟨⟨φ, f⟩, z⟩ := y
    have h := (splitEquiv tag htag).symm_apply_apply f
    change ((φ, (splitEquiv tag htag).symm (splitEquiv tag htag f)), z) = ((φ, f), z)
    rw [h]
  right_inv x := by
    obtain ⟨e, ω, z⟩ := x
    have h := (splitEquiv tag htag).apply_symm_apply (ω.2, e)
    have h1 : (fun j => (splitEquiv tag htag).symm (ω.2, e) (tag j)) = ω.2 := congrArg Prod.fst h
    have h2 : (fun t : Rest tag => (splitEquiv tag htag).symm (ω.2, e) t.1) = e := congrArg Prod.snd h
    change ((fun t : Rest tag => (splitEquiv tag htag).symm (ω.2, e) t.1),
      ((ω.1, fun j => (splitEquiv tag htag).symm (ω.2, e) (tag j)), z)) = (e, (ω, z))
    rw [h1, h2]

theorem pr_false_eq_zero {α : Type} [Fintype α] : pr (fun _ : α => False) = 0 := by
  classical
  rw [pr_def]
  simp

/-- **An event of the snapshot bounded at every value of the other tags' keys is bounded over the store's oracle.** -/
theorem pr_snap_le {w p : ℕ} [NeZero p] [Fintype T] [DecidableEq T] {Z : Type} [Fintype Z] [Nonempty Z]
    (tag : Fin N → T) (htag : Function.Injective tag) (F : RestKeys tag w p → (ringRO N w p).Ω × Z → Prop)
    (c : ℝ≥0∞) (h : ∀ e, pr (F e) ≤ c) :
    pr (fun y : Oracle w p T × Z => F (rest tag y.1) (snap tag y.1, y.2)) ≤ c := by
  haveI : Nonempty (Fin p) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne p)⟩⟩
  calc pr (fun y : Oracle w p T × Z => F (rest tag y.1) (snap tag y.1, y.2))
      = pr (fun x : RestKeys tag w p × ((ringRO N w p).Ω × Z) => F x.1 x.2) :=
        PousThm1ii.pr_equiv (oracleEquiv tag htag Z) (fun x => F x.1 x.2)
    _ ≤ pr (fun _ : RestKeys tag w p => False) + c := PousThm1ii.pr_prod_le F (fun _ => False) c fun e _ => h e
    _ = c := by rw [pr_false_eq_zero, zero_add]

/-- `pr_snap_le` for `Pouw.pr`. -/
theorem pr_snap_le_real {w p : ℕ} [NeZero p] [Fintype T] [DecidableEq T] {Z : Type} [Fintype Z]
    (tag : Fin N → T) (htag : Function.Injective tag) (F : RestKeys tag w p → (ringRO N w p).Ω × Z → Prop)
    (c : ℝ) (hc : 0 ≤ c) (h : ∀ e, Pouw.pr (F e) ≤ c) :
    Pouw.pr (fun y : Oracle w p T × Z => F (rest tag y.1) (snap tag y.1, y.2)) ≤ c :=
  calc Pouw.pr (fun y : Oracle w p T × Z => F (rest tag y.1) (snap tag y.1, y.2))
      = Pouw.pr (fun x : RestKeys tag w p × ((ringRO N w p).Ω × Z) => F x.1 x.2) :=
        Pouw.pr_equiv (oracleEquiv tag htag Z) (fun x => F x.1 x.2)
    _ ≤ c := Pouw.pr_prod_le (fun x => F x.1 x.2) c hc fun e => h e

end Pous.Mutable.SecurityProofs

namespace Pous.SecurityProofs

open Pous Pous.Mutable
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **Tag freshness** (`Pous.Guarantees.SnapshotFresh`). -/
theorem SnapshotFresh : Pous.Guarantees.SnapshotFresh := by
  intro N T K _ _ _ tag htag E
  exact PousThm1ii.pr_equiv (Pous.Mutable.SecurityProofs.splitEquiv tag htag) (fun y => E y.1 y.2)

/-- **A block's offset absorbs its content** (`Pous.Guarantees.P2ContentAbsorbs`). -/
theorem P2ContentAbsorbs : Pous.Guarantees.P2ContentAbsorbs := by
  intro p w _ φ K v v₀
  refine ⟨φ.1.symm.trans ((Equiv.addRight (eltD φ v - eltD φ v₀)).trans φ.1), fun τ => ?_⟩
  change encBlock φ K τ v = encBlock φ K (φ.1 (φ.1.symm τ + (eltD φ v - eltD φ v₀))) v₀
  unfold encBlock
  rw [Equiv.symm_apply_apply,
    show eltD φ v₀ + (φ.1.symm τ + (eltD φ v - eltD φ v₀)) = eltD φ v + φ.1.symm τ by ring]

/-- **An audit window of the mutable store, from P2's per-round floor** (`Pous.Guarantees.ObligationOfRoundSecure`). -/
theorem ObligationOfRoundSecure : Pous.Guarantees.ObligationOfRoundSecure := by
  intro p ε₀ D Q L h N _ h1 h2 T _ _ tag htag Sz k W A₁ A₂
  haveI : Nonempty (Fin N) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne N)⟩⟩
  exact Pous.Mutable.SecurityProofs.pr_snap_le tag htag
    (fun e (x : Outcome (p2ring N 16448 (2 ^ 16448 - 21065) 16432) k) =>
      varPassesOn _ k k (W e) (A₁ e) (A₂ e) x) _
    fun e => PousP2Rounds.graded p ε₀ D Q L h N h1 h2 Sz k (W e) (A₁ e) (A₂ e)

/-- **An audit window of the mutable store with content committed before its key coins**
(`Pous.Guarantees.ObligationOfCommitted`). -/
theorem ObligationOfCommitted : Pous.Guarantees.ObligationOfCommitted := by
  intro p ε₀ D Q L h N _ h1 h2 T _ _ tag htag π W hW Sz k A₁ A₂
  haveI : Nonempty (Fin N) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne N)⟩⟩
  refine Pous.Mutable.SecurityProofs.pr_snap_le tag htag
    (fun e (x : Outcome (p2ring N 16448 (2 ^ 16448 - 21065) 16432) k) =>
      varPassesOn _ k k (W e x.1) (A₁ e) (A₂ e) x) _ fun e => ?_
  obtain ⟨bad, hbad, hstep⟩ := h N h1 h2 (π e) (W e) (hW e) Sz k (A₁ e) (A₂ e)
  classical
  have hc := PousP2Rounds.pr_chain (fun j x => varPassesOn _ k j (W e x.1) (A₁ e) (A₂ e) x ∧ ¬ bad x)
    (fun j => Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) (1 - p) k
    (fun j x hx => ⟨PousP2Rounds.varPassesOn_of_succ _ _ _ _ x hx.1, hx.2⟩) (fun j hj hlow => hstep j hj hlow) k le_rfl
  exact (PousP2Rounds.pr_le_and_not_add _ bad).trans (add_le_add hc hbad)

end Pous.SecurityProofs

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-window obligation guarantee** (`Pous.Period.ObligationHeld`): `PeriodHeld` at the snapshot, at every
value of the other tags' keys, averaged over them. -/
theorem obligationHeld : ObligationHeld := by
  intro hd hp hH D Q L hP ι α ΩW ΩQ T _ _ _ _ tag htag d r P round S hdev hS o
  exact Pous.Mutable.SecurityProofs.pr_snap_le_real tag htag
    (fun e x => P.Accepts (o.1, (o.2, e), x) ∧ ¬ P.Break (o.1, (o.2, e), x) ∧
      (r.held (o.1, (o.2, e), x) : ℝ) < S * (r.win .capacity).len) _
    (add_nonneg δ_nonneg ENNReal.toReal_nonneg)
    fun e => periodHeld hd hp hH D Q L hP ι α ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065)) d r P round S
      hdev hS (o.1, (o.2, e))

end Pous.Period.SecurityProofs
