module

public import Proofs.Flock.Bridge.FilePart.Inst
public import Proofs.Flock.Bridge.FilePart.Owned

/-!
# The kept sources, in the claims' terms

`mem_srcs_iff`: where `readIns` marks exactly the template inputs a computed row reads, the sources
`FilePart.ofCircuit` keeps (`mem_ofCircuit_srcs`'s form) are exactly the `b` some stacked row's input column has as its
key (`Aliased.keyIdx`).
-/

@[expose] public section

namespace Flock.Bridge.FilePart

open FlockSoundness FlockSoundness.Discharge

theorem map_val_finRange (n : ℕ) : (List.finRange n).map Fin.val = List.range n :=
  List.ext_getElem (by simp) fun i _ _ => by simp

/-- `keyIdx` as a number: none at the zero's key, else the first stacked input with the same key. -/
theorem keyIdx_map_val (c : Flock.Circuit) (B : ℕ) (x : Fin B) :
    (Aliased.keyIdx c B x).map Fin.val =
      if Aliased.srcKey c x.val = .zero then none else Spot.FilePart.firstKey c B x.val := by
  unfold Aliased.keyIdx Spot.FilePart.firstKey
  split
  · rfl
  · rw [← map_val_finRange, List.find?_map]
    rfl

/-- **The sources a unit owns, in the claims' terms**: a list of the first stacked input of each key some row reads,
the zero's aside, holds the `b < B` some row's input column has as its key (`Aliased.keyIdx`). -/
theorem mem_srcs_iff (c : Flock.Circuit) (T : Rows) (rd : Array Bool)
    (hread : ∀ y < T.nIn, rd.getD y false = true ↔ ∃ j : Fin T.nComp, ∃ c₀ ∈ T.a j ++ T.b j, c₀.val = y)
    (S : List ℕ) (hS : ∀ b, b ∈ S ↔ ∃ x < c.upv * T.nIn, rd.getD (x % T.nIn) false = true ∧
      Aliased.srcKey c x ≠ .zero ∧ Spot.FilePart.firstKey c (c.upv * T.nIn) x = some b) (b : ℕ) :
    b ∈ S ↔ b < (T.stack c.upv).nIn ∧ ∃ i : Fin (T.stack c.upv).nComp,
      ∃ col ∈ (T.stack c.upv).a i ++ (T.stack c.upv).b i, ∃ hc : col.val < (T.stack c.upv).nIn,
        (Aliased.keyIdx c (T.stack c.upv).nIn ⟨col.val, hc⟩).map Fin.val = some b := by
  have hn : ∀ x, x < c.upv * T.nIn → x % T.nIn < T.nIn := fun x hx =>
    Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega)
  rw [hS]
  constructor
  · rintro ⟨x, hx, hrd, hz, hf⟩
    have hb : b < c.upv * T.nIn := List.mem_range.1 (List.mem_of_find?_eq_some hf)
    obtain ⟨i, col, hcol, hcv⟩ := (stack_reads T c.upv x hx).2 ((hread _ (hn x hx)).1 hrd)
    refine ⟨hb, i, col, hcol, by rw [hcv]; exact hx, ?_⟩
    rw [keyIdx_map_val]
    simp only [hcv, hz, ite_false]
    exact hf
  · rintro ⟨hb, i, col, hcol, hc, hk⟩
    rw [keyIdx_map_val] at hk
    split at hk
    · cases hk
    · rename_i hz
      refine ⟨col.val, hc, ?_, hz, hk⟩
      exact (hread _ (hn _ hc)).2 ((stack_reads T c.upv col.val hc).1 ⟨i, col, hcol, rfl⟩)

end Flock.Bridge.FilePart
