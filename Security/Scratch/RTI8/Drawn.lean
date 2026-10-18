import Proofs.Accounting.TS.Drawn

/-!
# RT-I8 scratch: `RightFixedDrawn` (#2466), what it says and what it excludes

Scratch for the red team of #2466 (never lands). Run from `Security/Proofs` with `lake env lean`.

* `rfd_off_E`, `rfd_iff_bound_of_exact`: given `Of`'s first half and `Pins`, `RightFixedDrawn` is the accounting conjunct
  of `ProofsSound` off the binding/soundness event, and with exact binding and soundness it is that conjunct on every
  accepted path.
* `rfd_of_full_share`: a share of `1` a round makes it true of any play.
* `bound_of_local`, `rfd_of_local`: a draw-free per-unit premise (`UnitFixed`) plus a definitional covering of the share
  (`Covers`) give it, without reading its antecedent.
* `drawn_right_off`, `proofsSound_of_drawn_joint`: the restatement: joint soundness, and the drawn-right event exported.
* Toy 1 (`pf1`, `P1`): every premise of `ProofsSoundOfDrawn` at `δ = 0`, an accepted path with a wrong undrawn unit.
* Toy 2 (same, share `0`): the all-units form holds and `RightFixedDrawn` fails.
* Toy 6 (`pf6`, `P6`): a producer and a consumer equivocate a shared leaf; no unit is `Moved`, `Forked` or `Unsound`,
  every unit is right at its committed values, and `RightFixedDrawn` fails.
-/

namespace RTI8

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

theorem prob_none {Ω : Type} [Fintype Ω] (p : Ω → Prop) (h : ∀ x, ¬ p x) : prob p = 0 := by
  classical
  rw [prob_eq_card]
  have he : Finset.univ.filter p = ∅ := Finset.filter_eq_empty_iff.mpr (fun x _ => h x)
  simp [he]

theorem prob_le_zero_of {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ x, ¬ p x) : prob p ≤ 0 := (prob_none p h).le

theorem committed_eq_of_inv {A B : Type} [Fintype A] [Fintype B] {S s : ℕ} {J V : Type}
    (pf : DrawnLayer A B S s J V) (x : Paths (A × B) S) (w : Fin S) (j : J)
    (h : ∀ z, pf.opened (spliceDraw w x z) w j = pf.opened x w j) : pf.committed x w j = pf.opened x w j := by
  unfold DrawnLayer.committed plurality
  split
  · exact h _
  · rfl

theorem coinAt_one {Ω : Type} (x : Paths Ω 1) (w : Fin 1) : coinAt x w = x.1 := by
  obtain ⟨_ | k, hw⟩ := w
  · rfl
  · omega

theorem coinAt_splice_one {A B : Type} (x z : Paths (A × B) 1) (w : Fin 1) :
    (coinAt (spliceDraw w x z) w).1 = (coinAt x w).1 := by
  obtain ⟨_ | k, hw⟩ := w
  · rfl
  · omega

section General

variable {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] {A B : Type} [Fintype A] [Fintype B]
  {S s : ℕ} {J V : Type}

/-- `ProofsSound`'s accounting conjunct at a path. -/
def Bound (P : Play (A × B) S K T L U) (Nw : ℝ) (x : Paths (A × B) S) : Prop :=
  (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed

/-- `RightFixed`'s antecedent over a `DrawnLayer`: every unit right at its committed values, drawn or not. -/
def RightFixedAll (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → (∀ w j, pf.right w j (pf.committed x w j)) → Bound P Nw x

theorem rfd_imp_all (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ)
    (h : RightFixedDrawn P pf Nw) : RightFixedAll P pf Nw :=
  fun x hx hall => h x hx fun w _ => hall w _

/-- **What `ProofsSoundOfDrawn` reads of `RightFixedDrawn`**: on an accepted path off the event (no drawn unit moved, no
drawn proof unsound), the accounting conjunct. -/
theorem rfd_off_E (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ) (hOf : pf.Of P)
    (hPin : pf.Pins) (hRF : RightFixedDrawn P pf Nw) (x : Paths (A × B) S) (hx : P.accepts x)
    (hM : ∀ w i, ¬ pf.Moved x w (pf.drawnAt x w i)) (hU : ∀ w i, ¬ pf.Unsound x w (pf.drawnAt x w i)) :
    Bound P Nw x :=
  hRF x hx (Accounting.SecurityProofs.TS.DrawnLayer.right_committed pf x (hOf.1 x hx) hM hU hPin)

/-- **With exact binding and soundness, `RightFixedDrawn` is the accounting conjunct on every accepted path**: its
antecedent then holds on every accepted path, so the premise is the conclusion's second half. -/
theorem rfd_iff_bound_of_exact (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ)
    (hOf1 : ∀ x, P.accepts x → ∀ w i, pf.accepted x w (pf.drawnAt x w i)) (hPin : pf.Pins)
    (hM : ∀ x w i, ¬ pf.Moved x w (pf.drawnAt x w i)) (hU : ∀ x w i, ¬ pf.Unsound x w (pf.drawnAt x w i)) :
    RightFixedDrawn P pf Nw ↔ ∀ x, P.accepts x → Bound P Nw x :=
  ⟨fun h x hx => h x hx
      (Accounting.SecurityProofs.TS.DrawnLayer.right_committed pf x (hOf1 x hx) (hM x) (hU x) hPin),
    fun h x hx _ => h x hx⟩

/-- **A share of `1` a round makes `RightFixedDrawn` true of any play** with nonnegative fixed words. -/
theorem rfd_of_full_share (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ) (hS : 0 < S)
    (hd : ∀ x, P.game.deficiency x = S) (hf : ∀ x, 0 ≤ (P.run x).fixed) : RightFixedDrawn P pf Nw := by
  intro x _ _
  have hS' : (S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  rw [hd x, div_self hS', sub_self, zero_mul]
  exact hf x

/-- **E1(c) per unit, draw-free**: on an accepted path, the fixed words are at least the work of the good units (a good
unit: its committed values and every value upstream of it right), per round. -/
def UnitFixed (P : Play (A × B) S K T L U) [Fintype J] (good : Paths (A × B) S → Fin S → J → Prop)
    [∀ x w j, Decidable (good x w j)] (wt : Fin S → J → ℝ) : Prop :=
  ∀ x, P.accepts x → (∑ w, ∑ j, if good x w j then wt w j else 0) / S ≤ (P.run x).fixed

/-- **The share covers every unit that isn't good** (a definition's obligation, checked, not assumed). -/
def Covers (P : Play (A × B) S K T L U) [Fintype J] (good : Paths (A × B) S → Fin S → J → Prop)
    [∀ x w j, Decidable (good x w j)] (wt : Fin S → J → ℝ) (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → (∑ w, ∑ j, if good x w j then 0 else wt w j) ≤ P.game.deficiency x * Nw

theorem bound_of_local (P : Play (A × B) S K T L U) [Fintype J] (Nw : ℝ) (hS : 0 < S)
    (good : Paths (A × B) S → Fin S → J → Prop) [∀ x w j, Decidable (good x w j)] (wt : Fin S → J → ℝ)
    (hsum : ∑ w, ∑ j, wt w j = S * Nw) (hfix : UnitFixed P good wt) (hcov : Covers P good wt Nw) :
    ∀ x, P.accepts x → Bound P Nw x := by
  intro x hx
  have hS' : (0 : ℝ) < S := by exact_mod_cast hS
  have hsplit : (∑ w, ∑ j, if good x w j then wt w j else 0) + (∑ w, ∑ j, if good x w j then 0 else wt w j) =
      S * Nw := by
    rw [← hsum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> simp
  have h1 : (∑ w, ∑ j, if good x w j then wt w j else 0) / S ≤ (P.run x).fixed := hfix x hx
  have h2 : (∑ w, ∑ j, if good x w j then 0 else wt w j) ≤ P.game.deficiency x * Nw := hcov x hx
  have h3 : (∑ w, ∑ j, if good x w j then wt w j else 0) / S =
      Nw - (∑ w, ∑ j, if good x w j then 0 else wt w j) / S := by
    rw [eq_sub_iff_add_eq, ← add_div, hsplit, mul_comm, mul_div_assoc, div_self hS'.ne', mul_one]
  have h4 : (∑ w, ∑ j, if good x w j then 0 else wt w j) / S ≤ P.game.deficiency x * Nw / S :=
    div_le_div_of_nonneg_right h2 hS'.le
  have h5 : (1 - P.game.deficiency x / S) * Nw = Nw - P.game.deficiency x * Nw / S := by ring
  unfold Bound
  linarith

/-- **`UnitFixed` and `Covers` give `RightFixedDrawn`**, and the proof never reads its antecedent. -/
theorem rfd_of_local (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) [Fintype J] (Nw : ℝ) (hS : 0 < S)
    (good : Paths (A × B) S → Fin S → J → Prop) [∀ x w j, Decidable (good x w j)] (wt : Fin S → J → ℝ)
    (hsum : ∑ w, ∑ j, wt w j = S * Nw) (hfix : UnitFixed P good wt) (hcov : Covers P good wt Nw) :
    RightFixedDrawn P pf Nw :=
  fun x hx _ => bound_of_local P Nw hS good wt hsum hfix hcov x hx

/-- **The crypto half, exported, with joint soundness**: outside an event of probability `S·(δ_bind + δ_ks)`, an accepted
path's drawn units are right at the values committed before the draws. -/
theorem drawn_right_off (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (δbind δks : ℝ)
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

/-- **`ProofsSound` from joint soundness** (`n_π = 1`, `δ_ZK := δ_ks`), the form Compose's `CircuitCredit` takes. -/
theorem proofsSound_of_drawn_joint (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw δbind δks : ℝ)
    (hOf : pf.Of P) (hB : ∀ w, prob (fun x => ∃ i, pf.Moved x w (pf.drawnAt x w i)) ≤ δbind)
    (hZ : ∀ w, prob (fun x => ∃ i, pf.Unsound x w (pf.drawnAt x w i)) ≤ δks) (hPin : pf.Pins)
    (hRF : RightFixedDrawn P pf Nw) : ProofsSound P Nw δbind δks 1 := by
  obtain ⟨E, hE, hoff⟩ := drawn_right_off P pf δbind δks hOf.1 hB hZ hPin
  exact ⟨E, by simpa using hE, fun x hx hx' => ⟨hOf.2 x (hoff x hx hx'), hRF x hx (hoff x hx hx')⟩⟩

end General

/-! ## Toy 1 and Toy 2: one round, the draw names one of two units, unit `false` is wrong and its proof rejected -/

/-- A run whose only relevant count is its fixed words. -/
def anat (f : ℝ) : Anatomy Unit Unit Unit Unit :=
  { run := { n := fun _ => 0, cycles := 0, clock := 1 }, nword := fun _ _ => 0, nfold := fun _ => 0,
    nfill := fun _ => 0, words := fun _ => 0, fixed := f }

/-- Units `Bool`; unit `j` opens `j`; it is right iff `true`; only unit `true`'s proof is accepted. -/
def pf1 : DrawnLayer Unit Bool 1 1 Bool Bool where
  draw b _ := b
  accepted _ _ j := j = true
  opened _ _ j := j
  rel _ _ v := v = true
  right _ _ v := v = true

/-- One round of share `d`, passing when the drawn unit is `true`. -/
def game1 (d : ℝ) : RoundGame (Unit × Bool) 1 := .round (fun _ => d) (fun ω => ω.2 = true) (fun _ => .done)

/-- The play: accepted when the drawn unit is `true`, its fixed words `f`. -/
def P1 (d f : ℝ) : Play (Unit × Bool) 1 Unit Unit Unit Unit where
  game := game1 d
  accepts x := x.1.2 = true
  run _ := anat f

theorem pf1_committed (x : Paths (Unit × Bool) 1) (w : Fin 1) (j : Bool) : pf1.committed x w j = j :=
  committed_eq_of_inv pf1 x w j (fun _ => rfl)

theorem pf1_drawnAt (x : Paths (Unit × Bool) 1) (w : Fin 1) (i : Fin 1) : pf1.drawnAt x w i = x.1.2 := by
  show pf1.draw (coinAt x w).2 i = x.1.2
  rw [coinAt_one]
  all_goals rfl

theorem toy1_noMoved : ∀ x w j, ¬ pf1.Moved x w j := fun x w j h => h.2 (pf1_committed x w j).symm

theorem toy1_noUnsound : ∀ x w j, ¬ pf1.Unsound x w j := fun _ _ _ h => h.2 h.1

theorem toy1_binding : ∀ w, prob (fun x => ∃ i, pf1.Moved x w (pf1.drawnAt x w i)) ≤ 0 := fun w =>
  prob_le_zero_of fun x hx => Exists.elim hx fun _ h => toy1_noMoved x w _ h

theorem toy1_sound : pf1.SoundAt 0 := by
  intro w i
  exact prob_le_zero_of fun x h => toy1_noUnsound x w _ h

theorem toy1_pins : pf1.Pins := by
  intro _ _ _ h
  exact h

theorem toy1_of (d f : ℝ) : pf1.Of (P1 d f) := by
  refine ⟨fun x hx w i => ?_, fun x hr => ?_⟩
  · show pf1.drawnAt x w i = true
    rw [pf1_drawnAt]
    exact hx
  · have h := hr 0 0
    rw [pf1_committed, pf1_drawnAt] at h
    exact ⟨h, trivial⟩

/-- **Toy 1, non-vacuity**: at share `½` (one of two equal units wrong) and fixed words `N_w/2`, every premise of
`ProofsSoundOfDrawn` holds at `δ_bind = δ_ZK = 0`, and an accepted path has a wrong undrawn unit. -/
theorem toy1_rfd (Nw : ℝ) : RightFixedDrawn (P1 (1 / 2) (Nw / 2)) pf1 Nw := by
  intro x _ _
  show (1 - ((1 : ℝ) / 2 + 0) / ((1 : ℕ) : ℝ)) * Nw ≤ Nw / 2
  norm_num <;> linarith

theorem toy1_nonvacuous (Nw : ℝ) :
    pf1.Of (P1 (1 / 2) (Nw / 2)) ∧ (∀ w, prob (fun x => ∃ i, pf1.Moved x w (pf1.drawnAt x w i)) ≤ 0) ∧
      pf1.SoundAt 0 ∧ pf1.Pins ∧ RightFixedDrawn (P1 (1 / 2) (Nw / 2)) pf1 Nw ∧
      ∃ x, (P1 (1 / 2) (Nw / 2)).accepts x ∧ ¬ pf1.right 0 false (pf1.committed x 0 false) := by
  refine ⟨toy1_of _ _, toy1_binding, toy1_sound, toy1_pins, toy1_rfd Nw, (((), true), ()), rfl, ?_⟩
  rw [pf1_committed]
  exact Bool.false_ne_true

/-- **Toy 2, strictly stronger**: at share `0`, the all-units form holds (vacuously: unit `false` is wrong on every path)
and `RightFixedDrawn` fails; whether it holds turns on the share alone. -/
theorem toy2_all (Nw : ℝ) : RightFixedAll (P1 0 (Nw / 2)) pf1 Nw := by
  intro x _ hall
  have h := hall 0 false
  rw [pf1_committed] at h
  exact absurd h Bool.false_ne_true

theorem toy2_not_rfd (Nw : ℝ) (hNw : 0 < Nw) : ¬ RightFixedDrawn (P1 0 (Nw / 2)) pf1 Nw := by
  intro h
  have hb := h (((), true), ()) rfl (fun w i => by
    show pf1.committed _ w (pf1.drawnAt _ w i) = true
    rw [pf1_committed, pf1_drawnAt]
    all_goals rfl)
  change (1 - ((0 : ℝ) + 0) / ((1 : ℕ) : ℝ)) * Nw ≤ Nw / 2 at hb
  norm_num at hb
  linarith

/-! ## Toy 6: a producer and a consumer equivocate a shared leaf -/

/-- The round's coin before the draw. -/
def sig (x : Paths (Bool × Fin 2) 1) (w : Fin 1) : Bool := (coinAt x w).1

theorem sig_splice (x z : Paths (Bool × Fin 2) 1) (w : Fin 1) : sig (spliceDraw w x z) w = sig x w :=
  coinAt_splice_one x z w

/-- Unit `0` (the producer) opens operand `false` and output `false`; unit `1` (the consumer) opens the leaf the
producer writes as its operand, but as `σ`, and its output from it. Each unit's relation and rightness: its output is
its operand (its own values only, as `DrawnLayer` has it). Every proof is accepted. -/
def pf6 : DrawnLayer Bool (Fin 2) 1 1 (Fin 2) (Bool × Bool) where
  draw b _ := b
  accepted _ _ _ := True
  opened x w j := if j = 0 then (false, false) else (sig x w, sig x w)
  rel _ _ v := v.2 = v.1
  right _ _ v := v.2 = v.1

/-- No share; fixed words `1` when the consumer reads the producer's output, `½` when its words are adaptive. -/
def P6 : Play (Bool × Fin 2) 1 Unit Unit Unit Unit where
  game := .round (fun _ => 0) (fun _ => True) (fun _ => .done)
  accepts _ := True
  run x := anat (if sig x 0 then 1 / 2 else 1)

theorem pf6_inv (x : Paths (Bool × Fin 2) 1) (w : Fin 1) (j : Fin 2) (z : Paths (Bool × Fin 2) 1) :
    pf6.opened (spliceDraw w x z) w j = pf6.opened x w j := by
  show (if j = 0 then (false, false) else (sig (spliceDraw w x z) w, sig (spliceDraw w x z) w)) =
    (if j = 0 then (false, false) else (sig x w, sig x w))
  rw [sig_splice]

theorem pf6_committed (x : Paths (Bool × Fin 2) 1) (w : Fin 1) (j : Fin 2) : pf6.committed x w j = pf6.opened x w j :=
  committed_eq_of_inv pf6 x w j (pf6_inv x w j)

theorem toy6_noMoved : ∀ x w j, ¬ pf6.Moved x w j := fun x w j h => h.2 (pf6_committed x w j).symm

theorem toy6_noForked : ∀ w j x z, ¬ pf6.Forked w j x z := fun w j x z h => h.2.2 (pf6_inv x w j z).symm

theorem toy6_rel (x : Paths (Bool × Fin 2) 1) (w : Fin 1) (j : Fin 2) : pf6.rel w j (pf6.opened x w j) := by
  show (if j = 0 then (false, false) else (sig x w, sig x w)).2 = (if j = 0 then (false, false) else (sig x w, sig x w)).1
  by_cases hj : j = 0 <;> simp [hj]

theorem toy6_noUnsound : ∀ x w j, ¬ pf6.Unsound x w j := fun x w j h => h.2 (toy6_rel x w j)

theorem toy6_allRight (x : Paths (Bool × Fin 2) 1) (w : Fin 1) (j : Fin 2) : pf6.right w j (pf6.committed x w j) := by
  rw [pf6_committed]
  exact toy6_rel x w j

theorem toy6_of : pf6.Of P6 := ⟨fun _ _ _ _ => trivial, fun _ _ => ⟨trivial, trivial⟩⟩

/-- **Toy 6**: every premise of `ProofsSoundOfDrawn` but `RightFixedDrawn` holds at `δ_bind = δ_ZK = 0`, the fork form
included (no unit is ever `Forked`); every unit, drawn or not, is right at its committed values; the producer's
committed output and the consumer's committed operand, one leaf, differ; and `RightFixedDrawn` fails. Only it excludes
this cheater, and in the deployed system what excludes it is SHA-512's collision resistance. -/
theorem toy6 :
    pf6.Of P6 ∧ (∀ w, prob (fun x => ∃ i, pf6.Moved x w (pf6.drawnAt x w i)) ≤ 0) ∧
      (∀ w j, prob (fun p : Paths (Bool × Fin 2) 1 × Paths (Bool × Fin 2) 1 => pf6.Forked w j p.1 p.2) ≤ 0) ∧
      pf6.SoundAt 0 ∧ pf6.Pins ∧ (∀ x w j, pf6.right w j (pf6.committed x w j)) ∧
      (∃ x, (pf6.committed x 0 0).2 ≠ (pf6.committed x 0 1).1) ∧ ¬ RightFixedDrawn P6 pf6 1 := by
  refine ⟨toy6_of, fun w => prob_le_zero_of fun x hx => Exists.elim hx fun _ h => toy6_noMoved x w _ h,
    fun w j => prob_le_zero_of fun p h => toy6_noForked w j p.1 p.2 h,
    by intro w i; exact prob_le_zero_of fun x h => toy6_noUnsound x w _ h, by intro _ _ _ h; exact h,
    toy6_allRight, ⟨((true, 0), ()), ?_⟩, ?_⟩
  · rw [pf6_committed, pf6_committed]
    decide
  · intro h
    have hb := h ((true, 0), ()) trivial (fun w i => toy6_allRight _ w _)
    change (1 - ((0 : ℝ) + 0) / ((1 : ℕ) : ℝ)) * 1 ≤ 1 / 2 at hb
    norm_num at hb

end RTI8

#print axioms RTI8.rfd_off_E
#print axioms RTI8.rfd_iff_bound_of_exact
#print axioms RTI8.rfd_of_full_share
#print axioms RTI8.rfd_of_local
#print axioms RTI8.drawn_right_off
#print axioms RTI8.proofsSound_of_drawn_joint
#print axioms RTI8.toy1_nonvacuous
#print axioms RTI8.toy2_all
#print axioms RTI8.toy2_not_rfd
#print axioms RTI8.toy6
