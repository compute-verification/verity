import Proofs.Accounting.Headline.Luck

/-!
# RT-HEADLINE r4 scratch (never lands): `OperandHMu` against `CubeOperands` and against (H-μ) at the real view

On coins `Unit × Finset (Fin 2)` (CubeOperands' coin type, two cube bits):

* (F) `G4`: the cheater sees bit 0, skips (share 1) iff it is set, and guesses bit 1 as `true`; its deficient words are
  right iff both bits are set. Its real view is bit 0 and the skipped value bit 1 (`R4`), with point mass `½`. It meets
  `Pinned`, `OperandHMu ¾` and `ReceiptLuckAt ½`, so (I7) at `7/8`, but no `CubeOperands` at any ring: its share reads a
  bit. r3's premise excluded this cheater; r4's covers it.
* (H) `Gd`: the cheater sees both bits, skips iff bit 0 is set and always commits the wrong value. (H-μ) at its real
  view (`Rreal`, both bits) fails, since the view pins the skipped value; `OperandHMu ¾` holds through the coarser view
  `R4`. The free view makes the premise weaker than (H-μ) at the real view, here for a cheater that is never right.
-/

namespace RTHeadlineR4

open Accounting.TS Accounting.Assumptions.TS Accounting.SecurityProofs.TS Accounting.HMu Pouw.RelativeSecurity

abbrev Ω4 : Type := Unit × Finset (Fin 2)

theorem card4 : Fintype.card Ω4 = 4 := by
  first
  | decide
  | rfl
  | simp [Fintype.card_prod, Fintype.card_finset]

theorem prob4_eq {p : Ω4 → Prop} [DecidablePred p] {k : ℕ} (hk : (Finset.univ.filter p).card = k) :
    prob p = (k : ℝ) / 4 := by
  rw [prob_eq_card, hk, card4]
  norm_num

theorem prob_false' {Ω : Type} [Fintype Ω] (p : Ω → Prop) (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  classical
  unfold prob
  rw [Finset.filter_false_of_mem (fun ω _ => h ω)]
  simp

theorem prob_nonneg' {Ω : Type} [Fintype Ω] (p : Ω → Prop) : 0 ≤ prob p := by
  classical
  unfold prob
  positivity

/-! ## (F) a cheater whose share reads a cube bit, never better than ½ -/

/-- Its real view: bit 0 decides the share; the skipped value is bit 1; it commits `true` for it. -/
noncomputable def R4 : SkipRound Ω4 Bool Bool Bool where
  view ω := decide ((0 : Fin 2) ∈ ω.2)
  skipped ω := decide ((1 : Fin 2) ∈ ω.2)
  d o := if o then 1 else 0
  committed _ := true
  honest _ x := x

theorem r4_pointMass : R4.PointMass (3 / 4) := by
  intro o ho t
  cases o with
  | false => simp [R4] at ho
  | true =>
    have e2 : prob (fun ω : Ω4 => R4.view ω = true) = ((2 : ℕ) : ℝ) / 4 :=
      prob4_eq (by first | decide | decide +kernel)
    have e1 : prob (fun ω : Ω4 => R4.view ω = true ∧ R4.skipped ω = t) = ((1 : ℕ) : ℝ) / 4 :=
      prob4_eq (by cases t <;> first | decide | decide +kernel)
    first
    | (rw [e1, e2]; norm_num)
    | linarith [e1, e2]

noncomputable def G4 : ReceiptGame Ω4 1 :=
  .round (fun ω => if (0 : Fin 2) ∈ ω.2 then 1 else 0) (fun ω => (0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2)
    (fun ω => (0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) (fun ω => (0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2)
    (fun _ => .done)

theorem g4_pinned : G4.Pinned := And.intro (fun _ h => h) (fun _ => trivial)

theorem g4_receipt_luck : ReceiptLuckAt (1 / 2) G4 := by
  refine And.intro ?_ (fun _ => trivial)
  show prob (fun ω : Ω4 => ((0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) ∧ ¬ ((0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) ∧
      0 < (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0)) ≤
    1 / 2 * prob (fun ω : Ω4 => ¬ ((0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) ∧
      0 < (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0))
  have h0 : prob (fun ω : Ω4 => ((0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) ∧
      ¬ ((0 : Fin 2) ∈ ω.2 ∧ (1 : Fin 2) ∈ ω.2) ∧ 0 < (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0)) = 0 :=
    prob_false' _ fun ω h => h.2.1 h.1
  rw [h0]
  exact mul_nonneg (by norm_num) (prob_nonneg' _)

theorem share01 (ω : Ω4) :
    0 ≤ (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0) ∧ (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0) ≤ 1 := by
  by_cases h : (0 : Fin 2) ∈ ω.2 <;> simp [h]

theorem share_view (ω : Ω4) : (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0) = R4.d (R4.view ω) := by
  by_cases h : (0 : Fin 2) ∈ ω.2 <;> simp [R4, h]

theorem g4_hmu : OperandHMu (3 / 4) G4 := by
  refine And.intro share01 (And.intro ⟨Bool, Bool, R4, r4_pointMass, fun _ _ => rfl, share_view, ?_⟩
    (fun _ => trivial))
  intro ω h
  have h1 : (1 : Fin 2) ∈ ω.2 := h.2
  simp [R4, h1]

theorem g4_luck : LuckAt (7 / 8) G4.game :=
  Headline.luck_of_receipt_half G4.game G4 rfl g4_pinned g4_hmu g4_receipt_luck

/-- r3's premise excluded it, at every ring: `CubeOperands` fixes the share before the bits. -/
theorem g4_not_cube (R : Type) [CommRing R] : ¬ G4.CubeOperands R := by
  intro h
  obtain ⟨-, ⟨δ, e, -, hdδ, -, -⟩, -⟩ := h
  have h1 : (if (0 : Fin 2) ∈ ({0} : Finset (Fin 2)) then (1 : ℝ) else 0) = δ () := hdδ ((), {0})
  have h0 : (if (0 : Fin 2) ∈ (∅ : Finset (Fin 2)) then (1 : ℝ) else 0) = δ () := hdδ ((), ∅)
  rw [if_pos (by simp)] at h1
  rw [if_neg (by simp)] at h0
  linarith

/-! ## (H) a cheater whose real view pins the skipped value but who never uses it -/

/-- Its real view: both bits. The share is bit 0, the skipped value bit 1, and it commits the wrong one. -/
noncomputable def Rreal : SkipRound Ω4 (Bool × Bool) Bool Bool where
  view ω := (decide ((0 : Fin 2) ∈ ω.2), decide ((1 : Fin 2) ∈ ω.2))
  skipped ω := decide ((1 : Fin 2) ∈ ω.2)
  d o := if o.1 then 1 else 0
  committed o := !o.2
  honest _ x := x

theorem rreal_not_pointMass : ¬ Rreal.PointMass (3 / 4) := by
  intro h
  have h' := h (true, true) (by simp [Rreal]) true
  have e1 : prob (fun ω : Ω4 => Rreal.view ω = (true, true) ∧ Rreal.skipped ω = true) = ((1 : ℕ) : ℝ) / 4 :=
    prob4_eq (by first | decide | decide +kernel)
  have e2 : prob (fun ω : Ω4 => Rreal.view ω = (true, true)) = ((1 : ℕ) : ℝ) / 4 :=
    prob4_eq (by first | decide | decide +kernel)
  first
  | (rw [e1, e2] at h'; norm_num at h')
  | linarith [e1, e2]

/-- `Rreal` is a `SkipRound` reading of `Gd`: its share from the view, its words right only where the values agree. -/
theorem rreal_reads_gd (ω : Ω4) :
    (if (0 : Fin 2) ∈ ω.2 then (1 : ℝ) else 0) = Rreal.d (Rreal.view ω) ∧
      (False → Rreal.committed (Rreal.view ω) = Rreal.skipped ω) := by
  refine And.intro ?_ (fun h => False.elim h)
  by_cases h : (0 : Fin 2) ∈ ω.2 <;> simp [Rreal, h]

noncomputable def Gd : ReceiptGame Ω4 1 :=
  .round (fun ω => if (0 : Fin 2) ∈ ω.2 then 1 else 0) (fun _ => False) (fun _ => False) (fun _ => False)
    (fun _ => .done)

theorem gd_hmu : OperandHMu (3 / 4) Gd :=
  And.intro share01 (And.intro ⟨Bool, Bool, R4, r4_pointMass, fun _ _ => rfl, share_view, fun _ h => False.elim h⟩
    (fun _ => trivial))

/-! ## Axioms -/

#print axioms r4_pointMass
#print axioms g4_pinned
#print axioms g4_receipt_luck
#print axioms g4_hmu
#print axioms g4_luck
#print axioms g4_not_cube
#print axioms rreal_not_pointMass
#print axioms rreal_reads_gd
#print axioms gd_hmu

end RTHeadlineR4
