module

public import Proofs.Flock.Claims.GS.Radius
public import Proofs.Flock.Claims.NumbersRadius
public import Definitions.Flock.Defs

/-!
# The radius decoder lists every message close to a table

A level-0 message is close to a table (`Model.Close`, at the level's radius `γ = 1 − √ρ − 1/200`) when its encoding
agrees with the table at enough positions. The radius decoder (`Claims.GS.flockDecodeR`) lists every message within
that same radius of the word it is given (`Claims.GS.listWithin_flockDecodeR`). So, given a table's columns written out
in order (`colsL`), it lists every message close to the table: `mem_flockDecodeR_of_close`.

A search of the decoder's list therefore finds a close message that packs a satisfying witness whenever one exists,
and no message is chosen (`Exec.candL`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-- **A word written out**: position `q` with `f q`, for each `q < n`, in order. -/
def colsOf {Col : Type} (n : ℕ) (f : ℕ → Col) : List (ℕ × Col) := (List.range n).map fun q => (q, f q)

/-- The written-out word holds `(i, f i)` as its first entry at position `i`. -/
theorem find?_colsOf {Col : Type} (f : ℕ → Col) {n i : ℕ} (hi : i < n) :
    (colsOf n f).find? (fun p => p.1 == i) = some (i, f i) := by
  unfold colsOf
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.find?_append]
    by_cases h : i < n
    · rw [ih h]; rfl
    · have hin : i = n := by omega
      subst hin
      have hnone : ((List.range i).map fun q => (q, f q)).find? (fun p => p.1 == i) = none := by
        rw [List.find?_eq_none]
        intro p hp hEq
        obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
        have hlt := List.mem_range.1 hq
        have : q = i := by simpa using hEq
        omega
      rw [hnone]
      simp

/-- **The zero-fill of a written-out word is the word**: nothing is left to fill. -/
theorem fill_colsOf {Col : Type} [Zero Col] (n : ℕ) (f : ℕ → Col) (i : Fin n) :
    Claims.Flock.fill n (colsOf n f) i = f i.val := by
  unfold Claims.Flock.fill
  rw [find?_colsOf f i.isLt]
  rfl

/-- **A table's columns, written out**: position `q`'s column of the oracle `C`, for each `q` below the level-0
length of `sch`, in order. The word the radius decoder reads. -/
def colsL (sch : Schedule) (C : Oracle 𝔽) : List (ℕ × (Fin (2 ^ sch.k0) → 𝔽)) :=
  colsOf (2 ^ sch.level₀.logLen) fun q ℓ => C ℓ.val q

/-- One filter of a finite type is no larger than another it implies, whichever way each is decided. -/
theorem card_filter_le_of_imp {α : Type} [Fintype α] (p q : α → Prop) (hp : DecidablePred p)
    (hq : DecidablePred q) (h : ∀ a, p a → q a) :
    (@Finset.filter α p hp Finset.univ).card ≤ (@Finset.filter α q hq Finset.univ).card :=
  Finset.card_le_card fun a ha => by
    rw [Finset.mem_filter] at ha ⊢
    exact ⟨ha.1, h a ha.2⟩

/-- **The radius decoder lists every message close to the table**: at a `fast100` schedule, a level-0 message whose
encoding is within the level-0 radius of the table `C` is in `flockDecodeR`'s list at `C`'s written-out columns. The
level's radius is the decoder's `θ` (`Numbers.level_radius_eq`), and closeness counts the positions where every lane
agrees, which are the positions where the columns agree. -/
theorem mem_flockDecodeR_of_close {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) (C : Oracle 𝔽)
    (M : Message 𝔽 sch.k0 sch.level₀.logCols)
    (hM : Model.Close sch.level₀.radius (asTable sch C) (encode execArith sch.level₀.logLen M)) :
    M ∈ Claims.GS.flockDecodeR h (colsL sch C) := by
  refine Claims.GS.listWithin_flockDecodeR h (colsL sch C) M ?_
  have himp : ∀ q : Fin (2 ^ sch.level₀.logLen),
      (∀ ℓ, asTable sch C ℓ q = encode execArith sch.level₀.logLen M ℓ q) →
      Claims.Flock.levelZeroEnc execArith sch.k0 sch.level₀.logCols M q.val =
        Claims.Flock.fill (2 ^ sch.level₀.logLen) (colsL sch C) q := by
    intro q hq
    unfold colsL
    rw [fill_colsOf]
    exact funext fun ℓ => (hq ℓ).symm
  have hM' : (1 - sch.level₀.radius) * (2 : ℝ) ^ sch.level₀.logLen ≤ _ := hM
  rw [Claims.Flock.Numbers.level_radius_eq] at hM'
  have e : ((2 ^ sch.level₀.logLen : ℕ) : ℝ) = (2 : ℝ) ^ sch.level₀.logLen := by norm_cast
  show (1 - (1 - Real.sqrt ((2 : ℝ) ^ sch.level₀.logCols / 2 ^ sch.level₀.logLen) - 1 / 200)) *
      ((2 ^ sch.level₀.logLen : ℕ) : ℝ) ≤ _
  rw [e]
  exact le_trans hM' (Nat.cast_le.2 (card_filter_le_of_imp _ _ _ _ himp))

end FlockSoundness.Discharge.Exec
