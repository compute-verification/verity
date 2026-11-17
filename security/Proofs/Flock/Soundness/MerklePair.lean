module

public import Definitions.Flock.MerklePair
public import Proofs.Flock.Soundness.Merkle
public import Mathlib.Tactic.Basic
import Mathlib.Tactic.Push
import Mathlib.Tactic.SplitIfs

/-! Two Merkle openings of one position that verify against one tree top and carry different columns give a collision
under `H`. -/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}
variable (H : List UInt8 → D) (E : Enc Col D)

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

theorem openCollPair_spec (top : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D) (hc : c ≠ c')
    (h : Verifies H E top depth pos c sibs) (h' : Verifies H E top depth pos c' sibs') :
    Hash.Collision H (openCollPair H E pos c c' sibs sibs').1 (openCollPair H E pos c c' sibs sibs').2 := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reach H E pos c sibs = reach H E pos c' sibs' := hr.trans hr'.symm
  unfold openCollPair
  split_ifs with hleaf
  · exact ⟨fun e => hc (E.col_inj e), E.dig_inj hleaf⟩
  · exact walkPair_spec H E sibs sibs' pos _ _ (hl.trans hl'.symm) (fun e => hleaf (congrArg E.dig e)) heq

variable {S : Type}

theorem openCollPairL_spec (Lf : Leaf Col S D H) (top : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S)
    (sibs sibs' : List D) (hc : c ≠ c') (h : VerifiesL H E Lf top depth pos c s sibs)
    (h' : VerifiesL H E Lf top depth pos c' s' sibs') :
    Hash.Collision H (openCollPairL H E Lf pos c c' s s' sibs sibs').1
      (openCollPairL H E Lf pos c c' s s' sibs sibs').2 := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reachL H E Lf pos c s sibs = reachL H E Lf pos c' s' sibs' := hr.trans hr'.symm
  unfold openCollPairL
  split_ifs with hleaf
  · exact Lf.pair_spec c c' s s' hc (E.dig_inj hleaf)
  · exact walkPair_spec H E sibs sibs' pos _ _ (hl.trans hl'.symm) (fun e => hleaf (congrArg E.dig e)) heq

theorem mem_pairsOf {γ : Type} {xs : List γ} {x y : γ} (hx : x ∈ xs) (hy : y ∈ xs) : (x, y) ∈ pairsOf xs :=
  List.mem_flatMap.2 ⟨x, hx, List.mem_map.2 ⟨y, hy, rfl⟩⟩

end FlockSoundness.Merkle
