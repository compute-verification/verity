module

public import Definitions.Flock.MerklePair
public import Proofs.Flock.Soundness.Merkle
public import Mathlib.Tactic.Basic
import Mathlib.Tactic.Push
import Mathlib.Tactic.SplitIfs

/-!
# The pair of two Merkle openings is a collision

`Definitions.Flock.MerklePair` gives, for two openings, two byte strings. This file proves that they collide under
`H` whenever the two openings verify against one tree top and carry different columns:

* `walkPair_spec`: two walks of one length from different digests to one digest;
* `openPair_spec`: two verifying openings of one position to different columns, under the leaf `H (encode column)`;
* `openPairL_spec`: the same under a leaf scheme, given that the scheme's own pair is a collision whenever two
  different columns have one leaf.

Each names its pair, where `walk_binding`, `opening_binding` and `opening_bindingL` (`Proofs.Flock.Soundness.Merkle`)
conclude only that `H` has a collision.
-/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}
variable (H : List UInt8 → D) (E : Enc Col D)

/-- **`walkPair` is a collision**: two walks of one length that start from different digests and end at one digest
hash two different inputs to one digest at the step where they first agree. -/
theorem walkPair_spec : ∀ (sibs sibs' : List D) (idx : ℕ) (a a' : D),
    sibs.length = sibs'.length → a ≠ a' → walk H E idx a sibs = walk H E idx a' sibs' →
    Hash.Collision H (walkPair H E sibs sibs' idx a a').1 (walkPair H E sibs sibs' idx a a').2
  | [], [], _, a, a', _, hne, heq => absurd heq hne
  | [], _ :: _, _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, _, hlen, _, _ => by simp at hlen
  | s :: rest, s' :: rest', idx, a, a', hlen, hne, heq => by
      simp only [walk] at heq
      by_cases hup : E.dig (up H E idx a s) = E.dig (up H E idx a' s')
      · simp only [walkPair, ite_eq_left hup]
        refine ⟨fun hin => ?_, E.dig_inj hup⟩
        unfold upIn at hin
        split_ifs at hin
        · exact hne (append_inj E hin).1
        · exact hne (append_inj E hin).2
      · simp only [walkPair, ite_eq_right hup]
        exact walkPair_spec rest rest' (idx / 2) _ _ (by simpa using hlen) (fun h => hup (congrArg E.dig h)) heq

/-- **`openPair` is a collision**: two openings of one position that verify against one tree top and carry different
columns. -/
theorem openPair_spec (top : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D) (hc : c ≠ c')
    (h : Verifies H E top depth pos c sibs) (h' : Verifies H E top depth pos c' sibs') :
    Hash.Collision H (openPair H E pos c c' sibs sibs').1 (openPair H E pos c c' sibs sibs').2 := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reach H E pos c sibs = reach H E pos c' sibs' := hr.trans hr'.symm
  unfold openPair
  split_ifs with hleaf
  · exact ⟨fun e => hc (E.col_inj e), E.dig_inj hleaf⟩
  · exact walkPair_spec H E sibs sibs' pos _ _ (hl.trans hl'.symm) (fun e => hleaf (congrArg E.dig e)) heq

variable {S : Type}

/-- **`openPairL` is a collision**: two openings of one position that verify against one tree top under the leaf
scheme `Lf` and carry different columns, for a `pair` that is a collision whenever two different columns have one
leaf. -/
theorem openPairL_spec (Lf : Leaf Col S D H) (pair : Col → S → Col → S → List UInt8 × List UInt8)
    (hpair : ∀ c c' s s', c ≠ c' → Lf.leaf c s = Lf.leaf c' s' →
      Hash.Collision H (pair c s c' s').1 (pair c s c' s').2)
    (top : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S) (sibs sibs' : List D) (hc : c ≠ c')
    (h : VerifiesL H E Lf top depth pos c s sibs) (h' : VerifiesL H E Lf top depth pos c' s' sibs') :
    Hash.Collision H (openPairL H E Lf.leaf pair pos c c' s s' sibs sibs').1
      (openPairL H E Lf.leaf pair pos c c' s s' sibs sibs').2 := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reachL H E Lf pos c s sibs = reachL H E Lf pos c' s' sibs' := hr.trans hr'.symm
  unfold openPairL
  split_ifs with hleaf
  · exact hpair c c' s s' hc (E.dig_inj hleaf)
  · exact walkPair_spec H E sibs sibs' pos _ _ (hl.trans hl'.symm) (fun e => hleaf (congrArg E.dig e)) heq

end FlockSoundness.Merkle
