module

public import Proofs.Flock.Bridge.FilePart.Keys

/-!
# The sources `FilePart.ofCircuit` keeps

`mem_ofCircuit_srcs`: source `b` is kept exactly when it is the first stacked input (`firstKey`) with the key of a
stacked input `x` whose template column a computed row reads (`RecResiduals.readIns`) and whose key is not the zero's.
The kept sources ascend (`ofCircuit_srcs_sorted`), and `block_toList` is unit `u`'s block as lists.
-/

@[expose] public section

namespace Spot.FilePart

open Std FlockSoundness.Discharge

theorem find?_congr_mem {α : Type} {p q : α → Bool} : ∀ {l : List α}, (∀ a ∈ l, p a = q a) → l.find? p = l.find? q
  | [], _ => rfl
  | a :: l, h => by
    rw [List.find?_cons, List.find?_cons, h a List.mem_cons_self,
      find?_congr_mem fun b hb => h b (List.mem_cons_of_mem _ hb)]

theorem getD_set!_true (o : Array Bool) (i b : ℕ) :
    (o.set! i true).getD b false = true ↔ o.getD b false = true ∨ (i = b ∧ b < o.size) := by
  simp only [Array.set!, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases hib : i = b
  · subst hib
    by_cases hs : i < o.size
    · simp [hs]
    · simp [hs]
  · simp [hib]

/-- A fold that sets the entries `idx a` of the `a` that pass `P`, each in bounds, marks exactly those. -/
theorem foldl_set {α : Type} (P : α → Bool) (idx : α → ℕ) :
    ∀ (l : List α) (o : Array Bool), (∀ a ∈ l, P a = true → idx a < o.size) → ∀ b,
      (l.foldl (fun o a => if P a then o.set! (idx a) true else o) o).getD b false = true ↔
        o.getD b false = true ∨ ∃ a ∈ l, P a = true ∧ idx a = b
  | [], o, _, b => by simp
  | a :: l, o, h, b => by
    rw [List.foldl_cons]
    have hs : (if P a then o.set! (idx a) true else o).size = o.size := by split <;> simp
    rw [foldl_set P idx l _ (fun a' ha' hp => hs ▸ h a' (List.mem_cons_of_mem _ ha') hp)]
    by_cases hp : P a = true
    · simp only [hp, ite_true, getD_set!_true]
      have := h a List.mem_cons_self hp
      constructor
      · rintro ((h1 | ⟨h1, -⟩) | ⟨a', ha', h2⟩)
        · exact Or.inl h1
        · exact Or.inr ⟨a, List.mem_cons_self, hp, h1⟩
        · exact Or.inr ⟨a', List.mem_cons_of_mem _ ha', h2⟩
      · rintro (h1 | ⟨a', ha', h2⟩)
        · exact Or.inl (Or.inl h1)
        · rcases List.mem_cons.1 ha' with rfl | ha'
          · exact Or.inl (Or.inr ⟨h2.2, h2.2 ▸ this⟩)
          · exact Or.inr ⟨a', ha', h2⟩
    · simp only [hp, Bool.false_eq_true, ite_false]
      constructor
      · rintro (h1 | ⟨a', ha', h2⟩)
        · exact Or.inl h1
        · exact Or.inr ⟨a', List.mem_cons_of_mem _ ha', h2⟩
      · rintro (h1 | ⟨a', ha', h2⟩)
        · exact Or.inl h1
        · rcases List.mem_cons.1 ha' with rfl | ha'
          · exact absurd h2.1 hp
          · exact Or.inr ⟨a', ha', h2⟩

variable (c : Flock.Circuit) (n : ℕ)

/-- The template's input columns, `min (inWords · Flock.WORD) constPos`. -/
abbrev nIn0 : ℕ := min (c.unit.inWords * Flock.WORD) c.unit.constPos

/-- The first stacked input with `x`'s key (`Aliased.keyIdx`'s, as a number). -/
def firstKey (B x : ℕ) : Option ℕ := (List.range B).find? fun y => decide (Aliased.srcKey c y = Aliased.srcKey c x)

theorem ofCircuit_n : (ofCircuit c n).n = n := rfl

theorem ofCircuit_B : (ofCircuit c n).B = c.upv * nIn0 c := rfl

theorem ofCircuit_m : (ofCircuit c n).m = c.upv * (c.unit.constPos - nIn0 c + 1) := rfl

theorem ofCircuit_srcs_sorted : (ofCircuit c n).srcs.toList.Pairwise (· < ·) := by
  show ((Array.range _).filter _).toList.Pairwise _
  rw [Array.toList_filter, Array.toList_range]
  exact List.pairwise_lt_range.filter _

/-- **The sources `ofCircuit` keeps** are the first stacked input of each key some row reads, the zero's aside. -/
theorem mem_ofCircuit_srcs (b : ℕ) :
    b ∈ (ofCircuit c n).srcs.toList ↔ ∃ x < c.upv * nIn0 c,
      (Flock.RecResiduals.readIns c.unit).getD (x % nIn0 c) false = true ∧ Aliased.srcKey c x ≠ .zero ∧
        firstKey c (c.upv * nIn0 c) x = some b := by
  have hkey : ∀ x < c.upv * nIn0 c,
      ((Array.range (c.upv * nIn0 c)).map (srcKey c (Index.of c)))[x]! = srcKey c (Index.of c) x := fun x hx => by
    rw [getElem!_pos _ x (by simpa using hx)]
    simp
  have hfirst : ∀ x < c.upv * nIn0 c,
      ((Array.range (c.upv * nIn0 c)).foldl (fun (m : HashMap Src ℕ) y =>
        if m.contains ((Array.range (c.upv * nIn0 c)).map (srcKey c (Index.of c)))[y]! then m
        else m.insert ((Array.range (c.upv * nIn0 c)).map (srcKey c (Index.of c)))[y]! y) ∅).getD
          ((Array.range (c.upv * nIn0 c)).map (srcKey c (Index.of c)))[x]! x =
        (firstKey c (c.upv * nIn0 c) x).getD x := fun x hx => by
    rw [HashMap.getD_eq_getD_getElem?, ← Array.foldl_toList, Array.toList_range,
      foldl_first' (fun y => ((Array.range (c.upv * nIn0 c)).map (srcKey c (Index.of c)))[y]!) (fun y => y),
      Option.map_id']
    unfold firstKey
    congr 1
    refine find?_congr_mem fun y hy => ?_
    rw [List.mem_range] at hy
    rw [hkey x hx, hkey y hy]
    rw [Bool.eq_iff_iff, beq_iff_eq, decide_eq_true_eq]
    exact ⟨fun h => by rw [← srcKey_toA, ← srcKey_toA, h], fun h => Src.toA_inj (by rw [srcKey_toA, srcKey_toA, h])⟩
  have hsome : ∀ x < c.upv * nIn0 c, ∃ y < c.upv * nIn0 c, firstKey c (c.upv * nIn0 c) x = some y := fun x hx => by
    have hs : (firstKey c (c.upv * nIn0 c) x).isSome := by
      unfold firstKey; exact List.find?_isSome.2 ⟨x, List.mem_range.2 hx, by simp⟩
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hs
    exact ⟨y, List.mem_range.1 (List.mem_of_find?_eq_some hy), hy⟩
  show b ∈ ((Array.range _).filter _).toList ↔ _
  rw [Array.toList_filter, Array.toList_range, List.mem_filter, List.mem_range, ← Array.foldl_toList,
    Array.toList_range]
  rw [foldl_set _ _ _ _ (fun x hx _ => by
    rw [List.mem_range] at hx
    rw [hfirst x hx]
    obtain ⟨y, hy, hxy⟩ := hsome x hx
    rw [hxy]; simpa using hy)]
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate, List.mem_range]
  constructor
  · rintro ⟨hb, h | ⟨x, hx, hp, hi⟩⟩
    · split at h <;> simp at h
    · rw [Bool.and_eq_true, bne_iff_ne, ne_eq, hkey x hx] at hp
      rw [hfirst x hx] at hi
      obtain ⟨y, -, hxy⟩ := hsome x hx
      rw [hxy] at hi
      simp only [Option.getD_some] at hi
      subst hi
      refine ⟨x, hx, hp.1, fun h0 => hp.2 (Src.toA_eq_zero.1 ((srcKey_toA c x).trans h0)), hxy⟩
  · rintro ⟨x, hx, hrd, hz, hxy⟩
    obtain ⟨y, hy, -⟩ := hsome x hx
    have hb : b < c.upv * nIn0 c := List.mem_range.1 (List.mem_of_find?_eq_some hxy)
    refine ⟨hb, Or.inr ⟨x, hx, ?_, ?_⟩⟩
    · rw [Bool.and_eq_true, bne_iff_ne, ne_eq, hkey x hx]
      exact ⟨hrd, fun h0 => hz (by rw [← srcKey_toA, h0]; rfl)⟩
    · rw [hfirst x hx, hxy]; rfl

/-- **Unit `u`'s block, as lists**: its rows' lowered gates, then its sources' gates. -/
theorem block_toList (f : FilePart) (u : ℕ) :
    (f.block u).toList = (List.range (3 * f.m)).map (f.inGates + 3 * f.m * u + ·) ++ f.srcs.toList.map (2 + u * f.B + ·) := by
  simp [FilePart.block]

end Spot.FilePart
