import Definitions.Pouw.Lead.Model
import Proofs.Pouw.Lead.CoinBranch.Kept

/-!
# Coin branching at tile grain

At each coin value the execution's run is its own path, so a prover that branches on the coins pays, on each path, a
write for every distinct committed tile its memory didn't start with; revisits (a tile written twice) only add writes.

* `holds_iff`: a run holds a value in some slot at some point exactly when one of its write steps writes it or its
  initial memory holds it. Copies create no value, so "the run holds the tile" says nothing about the tile's words being
  in memory together.
* `distinct_le_writes`: the copy argument. If every committed tile is a write step's value or an initial tile
  (`TileGrain`), the distinct committed tiles number at most the write steps plus the table hits (`Kept.image_card_le`).
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Pouw.Lead Pouw.Dimension.Pointwise

/-- A run's last write is what its slot holds. -/
theorem run_write_last {A V : Type} [DecidableEq A] {w : ℕ} (m₀ : A → Fin w → V) (P : List (Step A V w)) (a : A)
    (p : Fin w) (v : V) : run m₀ (P ++ [.write a p v]) a p = v := by
  simp [run, List.foldl_append, Step.apply]

/-- **A run holds a value at some point exactly when it writes it or starts with it**: copies create no value. -/
theorem holds_iff {A V : Type} [DecidableEq A] (m₀ : A → Fin 1 → V) (P : List (Step A V 1)) (v : V) :
    (∃ k a, run m₀ (P.take k) a 0 = v) ↔ (∃ a p, Step.write a p v ∈ P) ∨ ∃ a, m₀ a 0 = v := by
  constructor
  · rintro ⟨k, a, h⟩
    induction P generalizing m₀ k with
    | nil => exact Or.inr ⟨a, by simpa [run] using h⟩
    | cons st P ih =>
      cases k with
      | zero => exact Or.inr ⟨a, by simpa [run] using h⟩
      | succ k =>
        rw [List.take_succ_cons] at h
        rcases ih (st.apply m₀) k h with ⟨a', p', hw⟩ | ⟨b, hb⟩
        · exact Or.inl ⟨a', p', List.mem_cons_of_mem _ hw⟩
        cases st with
        | write a' p' v' =>
          by_cases hba : b = a'
          · subst hba
            have hp : (0 : Fin 1) = p' := Subsingleton.elim _ _
            simp only [Step.apply, Function.update_self, hp] at hb
            exact Or.inl ⟨b, p', hb ▸ List.mem_cons_self⟩
          · simp only [Step.apply, Function.update_of_ne hba] at hb
            exact Or.inr ⟨b, hb⟩
        | copy s d =>
          by_cases hbd : b = d
          · subst hbd
            simp only [Step.apply, Function.update_self] at hb
            exact Or.inr ⟨s, hb⟩
          · simp only [Step.apply, Function.update_of_ne hbd] at hb
            exact Or.inr ⟨b, hb⟩
  · rintro (⟨a, p, h⟩ | ⟨a, h⟩)
    · obtain ⟨i, hi, hiv⟩ := List.getElem_of_mem h
      refine ⟨i + 1, a, ?_⟩
      rw [List.take_add_one, List.getElem?_eq_getElem hi, Option.toList_some, hiv, Subsingleton.elim (0 : Fin 1) p,
        run_write_last]
    · exact ⟨0, a, by simpa [run] using h⟩

/-- **Per path, distinct tiles cost writes**: if every committed tile is the value of one of the run's write steps or
a tile its memory starts with (`TileGrain` at a coin value with no egress whose execution hashes the honest words),
the committed tiles' distinct values number at most the run's whole-tile writes plus the committed tiles its memory
starts with (`tableHits`). -/
theorem distinct_le_writes {L : Pouw.Lead.Layout} (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L)
    (stream : CoinStream L) (ω : Coins L)
    (hrun : ∀ rb < L.rowBlocks, ∀ cb < L.colBlocks, ∀ t : Fin L.S,
      (∃ a p, Step.write a p (tileOf L κ ω rb cb t) ∈ (σ.exec ω).steps) ∨
        ∃ a, σ.init stream ω a 0 = tileOf L κ ω rb cb t) :
    ((tileIdx L).image fun x => tileOf L κ ω x.1 x.2.1 x.2.2).card ≤
      writes (σ.exec ω).steps + tableHits L κ σ stream ω := by
  classical
  refine (image_card_le (σ.init stream ω) (σ.exec ω).steps (tileIdx L) (fun x => tileOf L κ ω x.1 x.2.1 x.2.2)
    fun x hx => ?_).trans (Nat.add_le_add_left (Finset.card_le_card fun x hx => ?_) _)
  · simp only [tileIdx, Finset.mem_product, Finset.mem_range, Finset.mem_univ, and_true] at hx
    convert (holds_iff (σ.init stream ω) _ _).2 (hrun x.1 hx.1 x.2.1 hx.2 x.2.2)
  obtain ⟨hx, a, ha⟩ := Finset.mem_filter.1 hx
  refine Finset.mem_filter.2 ⟨hx, ?_⟩
  cases a with
  | inl b => exact Or.inl ⟨b, ha⟩
  | inr c => exact Or.inr ⟨c, ha⟩

end Pouw.SecurityProofs.Lead.CoinBranch
