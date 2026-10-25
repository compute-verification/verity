import Definitions.Flock.Merkle

/-!
# Merkle binding

**Binding** (`opening_binding`): two openings of one position that both verify against the same
cap (`Definitions.Flock.Merkle`) but carry different columns yield an explicit collision of `H`. This is the step both
reductions to collision resistance use: the straight-line one (a prover that knows its table)
and the rewinding one (every prover, `Rewinding`). The development states it for our tree shape
directly; VCVio proves the same fact for its inductive trees
(`InductiveMerkleTree.getPutativeRootWithHash_binding_collision`).

`opening_bindingL` is `opening_binding` for any leaf scheme (`Leaf`), salted ones included.
-/

namespace FlockSoundness.Merkle

variable {Col D : Type}

variable (H : List UInt8 → D) (E : Enc Col D)

theorem append_inj {a b a' b' : D} (h : E.dig a ++ E.dig b = E.dig a' ++ E.dig b') :
    a = a' ∧ b = b' := by
  have hl : (E.dig a).length = (E.dig a').length := E.dig_len a a'
  obtain ⟨h1, h2⟩ := List.append_inj h hl
  exact ⟨E.dig_inj h1, E.dig_inj h2⟩

/-- Two walks from different nodes that meet give a collision. -/
theorem walk_binding : ∀ (sibs sibs' : List D) (idx : ℕ) (a a' : D),
    sibs.length = sibs'.length → a ≠ a' → walk H E idx a sibs = walk H E idx a' sibs' →
      Collision H
  | [], [], _, a, a', _, hne, heq => absurd heq hne
  | [], _ :: _, _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, _, hlen, _, _ => by simp at hlen
  | s :: rest, s' :: rest', idx, a, a', hlen, hne, heq => by
      simp only [walk] at heq
      by_cases hup : up H E idx a s = up H E idx a' s'
      · -- the step itself collides: different inputs (the accumulators differ), equal hashes
        unfold up at hup
        split_ifs at hup with he
        · refine ⟨_, _, fun h => hne (append_inj E h).1, hup⟩
        · refine ⟨_, _, fun h => hne (append_inj E h).2, hup⟩
      · exact walk_binding rest rest' (idx / 2) _ _ (by simpa using hlen) hup heq

/-- **Merkle binding.** Two openings of one position that verify against one cap with different
columns yield a collision of `H`. -/
theorem opening_binding (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D)
    (hc : c ≠ c') (h : Verifies H E cap depth pos c sibs) (h' : Verifies H E cap depth pos c' sibs') :
    Collision H := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reach H E pos c sibs = reach H E pos c' sibs' := hr.trans hr'.symm
  by_cases hleaf : H (E.col c) = H (E.col c')
  · exact ⟨_, _, fun h => hc (E.col_inj h), hleaf⟩
  · exact walk_binding H E sibs sibs' pos _ _ (hl.trans hl'.symm) hleaf heq

variable {S : Type}

theorem verifiesL_plain (cap : ℕ → D) (depth pos : ℕ) (column : Col) (sibs : List D) :
    VerifiesL H E (Leaf.plain H E) cap depth pos column () sibs ↔ Verifies H E cap depth pos column sibs :=
  Iff.rfl

/-- **Merkle binding, for any leaf scheme.** Two openings of one position that verify against one cap with different
columns, under any salts, yield a collision of `H`. -/
theorem opening_bindingL (Lf : Leaf Col S D H) (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S)
    (sibs sibs' : List D) (hc : c ≠ c') (h : VerifiesL H E Lf cap depth pos c s sibs)
    (h' : VerifiesL H E Lf cap depth pos c' s' sibs') : Collision H := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reachL H E Lf pos c s sibs = reachL H E Lf pos c' s' sibs' := hr.trans hr'.symm
  by_cases hleaf : Lf.leaf c s = Lf.leaf c' s'
  · exact Lf.bind c c' s s' hc hleaf
  · exact walk_binding H E sibs sibs' pos _ _ (hl.trans hl'.symm) hleaf heq

end FlockSoundness.Merkle
