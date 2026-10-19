import Properties.Pouw
import Properties.Proofs.Pouw.Grid
import Properties.Proofs.Pouw.Draw

/-!
# Proofs of PoUW's window properties (`Properties.Pouw`)

The served numbers, by evaluation in the kernel (`decide +kernel`: no compiled code, only the kernel's reduction), and
the refusals, by unfolding. The grid's properties are in `Properties.Proofs.Pouw.Grid` and K′'s in `Properties.Proofs.Pouw.Draw`.
-/

namespace Properties.Pouw.Window

open PouwWindow

theorem levelsServed : LevelsServed := by
  unfold LevelsServed; decide +kernel

theorem credit8192 : Credit8192 := by
  unfold Credit8192; decide +kernel

theorem kPrimeServed : KPrimeServed := by
  unfold KPrimeServed; decide +kernel

theorem kPrimeExactAtEquality : KPrimeExactAtEquality := by
  unfold KPrimeExactAtEquality; decide +kernel

theorem publicItems_eq : PublicItems := by
  unfold PublicItems; decide +kernel

/-- Python's grid functions on any integer refuse exactly off their domain, and on it answer the function on naturals. -/
theorem grid_isOk (bits : Nat) (n lo : Int) (t : Nat) :
    (floorPoint? bits n).isOk = decide (1 ≤ n) ∧ (pad? bits n).isOk = decide (1 ≤ n) ∧
    (level? bits n).isOk = decide (1 ≤ n ∧ onGrid bits n.toNat = true) ∧
    (point? bits n).isOk = decide (0 ≤ n) ∧ (Work.at? bits n t).isOk = decide (0 ≤ n) ∧
    (levels? bits lo n).isOk = decide (1 ≤ lo ∧ 1 ≤ n) ∧
    (1 ≤ n → floorPoint? bits n = .ok (floorPoint bits n.toNat) ∧ pad? bits n = .ok (pad bits n.toNat)) ∧
    (1 ≤ n → onGrid bits n.toNat = true → level? bits n = .ok (level bits n.toNat)) ∧
    (0 ≤ n → point? bits n = .ok (point bits n.toNat) ∧ Work.at? bits n t = .ok (Work.at bits n.toNat t)) ∧
    (1 ≤ lo → 1 ≤ n → levels? bits lo n = .ok (levels bits lo.toNat n.toNat)) := by
  have fp : ∀ m : Int, floorPoint? bits m = if 1 ≤ m then .ok (floorPoint bits m.toNat) else
      .error s!"the grid is over positive integers, not {m}" := fun _ => rfl
  have pt : ∀ m : Int, point? bits m = if 0 ≤ m then .ok (point bits m.toNat) else
      .error s!"a level is a natural number, not {m}" := fun _ => rfl
  by_cases h1 : 1 ≤ n <;> by_cases h0 : 0 ≤ n <;> by_cases hl : 1 ≤ lo <;> cases hg : onGrid bits n.toNat <;>
    simp [pad?, level?, onGridInt, Work.at?, levels?, fp, pt, h1, h0, hl, hg, Except.isOk, Except.toBool,
      bind, Except.bind, pure, Except.pure] <;> omega

/-- `credit?` refuses exactly off `1 ≤ real < n`, and on it answers `credit`. -/
theorem credit_isOk (bits : Nat) (real n : Int) :
    (credit? bits real n).isOk = decide (1 ≤ real ∧ real < n) ∧
    (1 ≤ real → real < n → credit? bits real n = .ok (credit bits real.toNat n.toNat)) := by
  unfold credit?
  by_cases h : 1 ≤ real ∧ real < n
  · rw [ite_eq_left h]; exact ⟨by rw [decide_eq_true h]; rfl, fun _ _ => rfl⟩
  · rw [ite_eq_right h]; exact ⟨by rw [decide_eq_false h]; rfl, fun a b => absurd ⟨a, b⟩ h⟩

theorem ok_dom {α : Type} {x : Except String α} {a v : α} {P : Prop} [Decidable P] (hok : x.isOk = decide P)
    (hv : P → x = .ok v) (h : x = .ok a) : P ∧ a = v := by
  have hP : P := of_decide_eq_true (hok.symm.trans (by rw [h]; rfl))
  exact ⟨hP, Except.ok.inj (h.symm.trans (hv hP))⟩

theorem floorPointRefuses : FloorPointRefuses := fun bits n _ h =>
  ok_dom (grid_isOk bits n 0 0).1 (fun p => ((grid_isOk bits n 0 0).2.2.2.2.2.2.1 p).1) h

theorem padRefuses : PadRefuses := fun bits n _ h =>
  ok_dom (grid_isOk bits n 0 0).2.1 (fun p => ((grid_isOk bits n 0 0).2.2.2.2.2.2.1 p).2) h

theorem creditRefuses : CreditRefuses := fun bits real n _ h => by
  obtain ⟨⟨a, b⟩, c⟩ := ok_dom (credit_isOk bits real n).1 (fun p => (credit_isOk bits real n).2 p.1 p.2) h
  exact ⟨a, b, c⟩

theorem levelRefuses : LevelRefuses := fun bits n _ h => by
  obtain ⟨⟨a, b⟩, c⟩ :=
    ok_dom (grid_isOk bits n 0 0).2.2.1 (fun p => (grid_isOk bits n 0 0).2.2.2.2.2.2.2.1 p.1 p.2) h
  exact ⟨a, b, c⟩

theorem pointRefuses : PointRefuses := fun bits ℓ _ h =>
  ok_dom (grid_isOk bits ℓ 0 0).2.2.2.1 (fun p => ((grid_isOk bits ℓ 0 0).2.2.2.2.2.2.2.2.1 p).1) h

theorem workAtRefuses : WorkAtRefuses := fun bits ℓ t _ h =>
  ok_dom (grid_isOk bits ℓ 0 t).2.2.2.2.1 (fun p => ((grid_isOk bits ℓ 0 t).2.2.2.2.2.2.2.2.1 p).2) h

theorem levelsRefuses : LevelsRefuses := fun bits lo hi _ h => by
  obtain ⟨⟨a, b⟩, c⟩ :=
    ok_dom (grid_isOk bits hi lo 0).2.2.2.2.2.1 (fun p => (grid_isOk bits hi lo 0).2.2.2.2.2.2.2.2.2 p.1 p.2) h
  exact ⟨a, b, c⟩

theorem floorPointGreatest : FloorPointGreatest := by
  intro bits n g h
  obtain ⟨hn, rfl⟩ := floorPointRefuses bits n g h
  exact floorPoint_greatest bits n.toNat (by omega)

theorem padLeast : PadLeast := by
  intro bits n g h
  obtain ⟨hn, rfl⟩ := padRefuses bits n g h
  exact pad_least bits n.toNat (by omega)

theorem creditCredits : CreditCredits := by
  intro bits real n w h
  obtain ⟨hr, hn, rfl⟩ := creditRefuses bits real n w h
  exact credit_credits bits real.toNat n.toNat (by omega) (by omega)

theorem creditCost : CreditCost := by
  intro bits real n w h
  obtain ⟨-, -, rfl⟩ := creditRefuses bits real n w h
  exact credit_cost bits real.toNat n.toNat

theorem creditCost3 : CreditCost3 := by
  intro real n w h
  obtain ⟨-, -, rfl⟩ := creditRefuses gridBits real n w h
  exact credit_cost3 real.toNat n.toNat

theorem levelsCount : LevelsCount := by
  intro bits lo hi k h
  obtain ⟨hlo, -, rfl⟩ := levelsRefuses bits lo hi k h
  exact levels_count bits lo.toNat hi.toNat (by omega)

theorem kPrimeRefuses : KPrimeRefuses := by
  intro eps δ credited n w k hk
  unfold kPrime at hk
  split at hk
  · assumption
  · cases hk

theorem kPrimeLeast : KPrimeLeast := by
  intro eps δ credited n w k hk
  have h := kPrime_least eps δ credited n w (kPrimeRefuses eps δ credited n w k hk)
  dsimp only at h ⊢
  obtain ⟨k', hk', hr⟩ := h
  rw [hk] at hk'
  cases hk'
  exact hr

theorem drawMisses : DrawMisses := fun eps δ credited n w B k hk =>
  draw_misses eps δ credited n w B k (kPrimeRefuses eps δ credited n w k hk) hk

/-! ## Completeness

The other half of the refusals: every input on a function's domain is answered. Each comes with one answered input of the
served grid and draw. -/

theorem grid_complete (bits : Nat) (n lo : Int) (t : Nat) :
    (1 ≤ n → floorPoint? bits n = .ok (floorPoint bits n.toNat) ∧ pad? bits n = .ok (pad bits n.toNat)) ∧
    (1 ≤ n → onGrid bits n.toNat = true → level? bits n = .ok (level bits n.toNat)) ∧
    (0 ≤ n → point? bits n = .ok (point bits n.toNat) ∧ Work.at? bits n t = .ok (Work.at bits n.toNat t)) ∧
    (1 ≤ lo → 1 ≤ n → levels? bits lo n = .ok (levels bits lo.toNat n.toNat)) :=
  (grid_isOk bits n lo t).2.2.2.2.2.2

theorem credit_complete (bits : Nat) (real n : Int) (hr : 1 ≤ real) (hn : real < n) :
    credit? bits real n = .ok (credit bits real.toNat n.toNat) :=
  (credit_isOk bits real n).2 hr hn

theorem grid_instance :
    credit? gridBits 8192 8193 = .ok (credit gridBits 8192 8193) ∧ level? gridBits 8192 = .ok (level gridBits 8192) ∧
      Work.at? gridBits 87 8193 = .ok (Work.at gridBits 87 8193) ∧
      levels? gridBits 1 (2 ^ 40) = .ok (levels gridBits 1 (2 ^ 40)) :=
  ⟨credit_complete gridBits 8192 8193 (by decide) (by decide),
    (grid_complete gridBits 8192 1 0).2.1 (by decide) (by decide),
    ((grid_complete gridBits 87 1 8193).2.2.1 (by decide)).2,
    (grid_complete gridBits (2 ^ 40) 1 0).2.2.2 (by decide) (by decide)⟩

theorem kPrime_complete (eps δ : Rat) (credited n w : Int) (hd : KPrimeDomain eps δ credited n w) :
    ∃ k, kPrime eps δ credited n w = .ok k := by
  have h := kPrime_least eps δ credited n w hd
  dsimp only at h
  obtain ⟨k, hk, -⟩ := h
  exact ⟨k, hk⟩

theorem kPrime_instance : ∃ k, kPrime (1 / 2) (1 / 10) (8192 * 4096) 9216 4096 = .ok k :=
  kPrime_complete _ _ _ _ _ (by unfold KPrimeDomain; decide +kernel)

end Properties.Pouw.Window
