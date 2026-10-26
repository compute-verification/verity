import Properties.Pouw
import Proofs.Properties.Pouw.Grid
import Proofs.Properties.Pouw.Draw

/-!
# Proofs of PoUW's window properties (`Properties.Pouw`)

The served numbers, by evaluation in the kernel (`decide +kernel`: no compiled code, only the kernel's reduction), and
the refusals, by unfolding. The grid's properties are in `Proofs.Properties.Pouw.Grid` and K′'s in `Proofs.Properties.Pouw.Draw`.
-/

namespace Properties.Pouw.Window

open PouwWindow

theorem levelsServed : LevelsServed := by
  unfold LevelsServed; decide +kernel

theorem bucket8192 : Bucket8192 := by
  unfold Bucket8192; decide +kernel

theorem kPrimeServed : KPrimeServed := by
  unfold KPrimeServed; decide +kernel

theorem kPrimeExactAtEquality : KPrimeExactAtEquality := by
  unfold KPrimeExactAtEquality; decide +kernel

theorem publicItems_eq : PublicItems := by
  unfold PublicItems; decide +kernel

/-- Python's grid functions on any integer refuse exactly off their domain, and on it answer the function on naturals. -/
theorem grid_isOk (bits : Nat) (n lo : Int) :
    (floorPoint? bits n).isOk = decide (1 ≤ n) ∧ (pad? bits n).isOk = decide (1 ≤ n) ∧
    (bucket? bits n).isOk = decide (1 ≤ n) ∧ (level? bits n).isOk = decide (1 ≤ n ∧ onGrid bits n.toNat = true) ∧
    (point? bits n).isOk = decide (0 ≤ n) ∧ (Work.at? bits n).isOk = decide (0 ≤ n) ∧
    (levels? bits lo n).isOk = decide (1 ≤ lo ∧ 1 ≤ n) ∧
    (1 ≤ n → floorPoint? bits n = .ok (floorPoint bits n.toNat) ∧ pad? bits n = .ok (pad bits n.toNat) ∧
      bucket? bits n = .ok (bucket bits n.toNat)) ∧
    (1 ≤ n → onGrid bits n.toNat = true → level? bits n = .ok (level bits n.toNat)) ∧
    (0 ≤ n → point? bits n = .ok (point bits n.toNat) ∧ Work.at? bits n = .ok (Work.at bits n.toNat)) ∧
    (1 ≤ lo → 1 ≤ n → levels? bits lo n = .ok (levels bits lo.toNat n.toNat)) := by
  have fp : ∀ m : Int, floorPoint? bits m = if 1 ≤ m then .ok (floorPoint bits m.toNat) else
      .error s!"the grid is over positive integers, not {m}" := fun _ => rfl
  have pt : ∀ m : Int, point? bits m = if 0 ≤ m then .ok (point bits m.toNat) else
      .error s!"a level is a natural number, not {m}" := fun _ => rfl
  by_cases h1 : 1 ≤ n <;> by_cases h0 : 0 ≤ n <;> by_cases hl : 1 ≤ lo <;> cases hg : onGrid bits n.toNat <;>
    simp [pad?, bucket?, level?, onGridInt, Work.at?, levels?, fp, pt, h1, h0, hl, hg, Except.isOk, Except.toBool,
      bind, Except.bind, pure, Except.pure] <;> omega

theorem ok_dom {α : Type} {x : Except String α} {a v : α} {P : Prop} [Decidable P] (hok : x.isOk = decide P)
    (hv : P → x = .ok v) (h : x = .ok a) : P ∧ a = v := by
  have hP : P := of_decide_eq_true (hok.symm.trans (by rw [h]; rfl))
  exact ⟨hP, Except.ok.inj (h.symm.trans (hv hP))⟩

theorem floorPointRefuses : FloorPointRefuses := fun bits n _ h =>
  ok_dom (grid_isOk bits n 0).1 (fun p => ((grid_isOk bits n 0).2.2.2.2.2.2.2.1 p).1) h

theorem padRefuses : PadRefuses := fun bits n _ h =>
  ok_dom (grid_isOk bits n 0).2.1 (fun p => ((grid_isOk bits n 0).2.2.2.2.2.2.2.1 p).2.1) h

theorem bucketRefuses : BucketRefuses := fun bits n _ h =>
  ok_dom (grid_isOk bits n 0).2.2.1 (fun p => ((grid_isOk bits n 0).2.2.2.2.2.2.2.1 p).2.2) h

theorem levelRefuses : LevelRefuses := fun bits n _ h => by
  obtain ⟨⟨a, b⟩, c⟩ :=
    ok_dom (grid_isOk bits n 0).2.2.2.1 (fun p => (grid_isOk bits n 0).2.2.2.2.2.2.2.2.1 p.1 p.2) h
  exact ⟨a, b, c⟩

theorem pointRefuses : PointRefuses := fun bits ℓ _ h =>
  ok_dom (grid_isOk bits ℓ 0).2.2.2.2.1 (fun p => ((grid_isOk bits ℓ 0).2.2.2.2.2.2.2.2.2.1 p).1) h

theorem workAtRefuses : WorkAtRefuses := fun bits ℓ _ h =>
  ok_dom (grid_isOk bits ℓ 0).2.2.2.2.2.1 (fun p => ((grid_isOk bits ℓ 0).2.2.2.2.2.2.2.2.2.1 p).2) h

theorem levelsRefuses : LevelsRefuses := fun bits lo hi _ h => by
  obtain ⟨⟨a, b⟩, c⟩ :=
    ok_dom (grid_isOk bits hi lo).2.2.2.2.2.2.1 (fun p => (grid_isOk bits hi lo).2.2.2.2.2.2.2.2.2.2 p.1 p.2) h
  exact ⟨a, b, c⟩

theorem floorPointGreatest : FloorPointGreatest := by
  intro bits n g h
  obtain ⟨hn, rfl⟩ := floorPointRefuses bits n g h
  exact floorPoint_greatest bits n.toNat (by omega)

theorem padLeast : PadLeast := by
  intro bits n g h
  obtain ⟨hn, rfl⟩ := padRefuses bits n g h
  exact pad_least bits n.toNat (by omega)

theorem bucketCredits : BucketCredits := by
  intro bits real w h
  obtain ⟨hr, rfl⟩ := bucketRefuses bits real w h
  exact bucket_credits bits real.toNat (by omega)

theorem bucketCost : BucketCost := by
  intro bits real w h
  obtain ⟨-, rfl⟩ := bucketRefuses bits real w h
  exact bucket_cost bits real.toNat

theorem bucketCost3 : BucketCost3 := by
  intro real w h
  obtain ⟨-, rfl⟩ := bucketRefuses gridBits real w h
  exact bucket_cost3 real.toNat

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

theorem grid_complete (bits : Nat) (n lo : Int) :
    (1 ≤ n → floorPoint? bits n = .ok (floorPoint bits n.toNat) ∧ pad? bits n = .ok (pad bits n.toNat) ∧
      bucket? bits n = .ok (bucket bits n.toNat)) ∧
    (1 ≤ n → onGrid bits n.toNat = true → level? bits n = .ok (level bits n.toNat)) ∧
    (0 ≤ n → point? bits n = .ok (point bits n.toNat) ∧ Work.at? bits n = .ok (Work.at bits n.toNat)) ∧
    (1 ≤ lo → 1 ≤ n → levels? bits lo n = .ok (levels bits lo.toNat n.toNat)) :=
  (grid_isOk bits n lo).2.2.2.2.2.2.2

theorem grid_instance :
    bucket? gridBits 8192 = .ok (bucket gridBits 8192) ∧ level? gridBits 8192 = .ok (level gridBits 8192) ∧
      Work.at? gridBits 87 = .ok (Work.at gridBits 87) ∧ levels? gridBits 1 (2 ^ 40) = .ok (levels gridBits 1 (2 ^ 40)) :=
  ⟨((grid_complete gridBits 8192 1).1 (by decide)).2.2, (grid_complete gridBits 8192 1).2.1 (by decide) (by decide),
    ((grid_complete gridBits 87 1).2.2.1 (by decide)).2, (grid_complete gridBits (2 ^ 40) 1).2.2.2 (by decide) (by decide)⟩

theorem kPrime_complete (eps δ : Rat) (credited n w : Int) (hd : KPrimeDomain eps δ credited n w) :
    ∃ k, kPrime eps δ credited n w = .ok k := by
  have h := kPrime_least eps δ credited n w hd
  dsimp only at h
  obtain ⟨k, hk, -⟩ := h
  exact ⟨k, hk⟩

theorem kPrime_instance : ∃ k, kPrime (1 / 2) (1 / 10) (8192 * 4096) 9216 4096 = .ok k :=
  kPrime_complete _ _ _ _ _ (by unfold KPrimeDomain; decide +kernel)

end Properties.Pouw.Window
