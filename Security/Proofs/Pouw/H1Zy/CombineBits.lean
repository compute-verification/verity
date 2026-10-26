import Definitions.Pouw.H1Zy.CombineBits
import Proofs.Pouw.H1Zy.Combine

/-!
# The combine's pieces on bits compute their word views, and so does their composition

Each piece's lowering computes its word view on every input: `PouwXor32_v2`'s (`xor32Bits_bitsOf`), `PouwEq32_v2`'s
(`eq32Bits_bitsOf`), and `PouwAndBit_v2`'s, which is `andBit` on both sides. A tree whose operation a map carries to
another's is carried with it (`tree_map`), so the combine's body on bits, read at its words' bits, is its body on words
(`combineBits_bitsOf`).
-/

namespace Pouw.H1Zy

/-- A pass of the tree commutes with a map that carries `f` to `g`. -/
theorem pairs_map {α β : Type} (f : α → α → α) (g : β → β → β) (φ : α → β)
    (h : ∀ a b, φ (f a b) = g (φ a) (φ b)) : ∀ l : List α, pairs g (l.map φ) = (pairs f l).map φ
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by simp only [List.map_cons, pairs, h, pairs_map f g φ h rest]

/-- **The tree commutes with a map that carries `f` to `g` and `d` to `d'`.** -/
theorem tree_map {α β : Type} (f : α → α → α) (g : β → β → β) (φ : α → β) (d : α) (d' : β)
    (h : ∀ a b, φ (f a b) = g (φ a) (φ b)) (hd : φ d = d') : ∀ l : List α, tree g d' (l.map φ) = φ (tree f d l)
  | [] => by simp [tree, hd]
  | [a] => by simp [tree]
  | a :: b :: rest => by
      rw [List.map_cons, List.map_cons, tree, ← List.map_cons, ← List.map_cons, pairs_map f g φ h,
        tree_map f g φ d d' h hd (pairs f (a :: b :: rest)), tree]
termination_by l => l.length
decreasing_by exact length_pairs_lt f a b rest

/-- **`PouwXor32_v2`'s lowering computes `PouwXor32_v1`.** -/
theorem xor32Bits_bitsOf (a b : BitVec 32) : xor32Bits (bitsOf a) (bitsOf b) = bitsOf (xor32 a b) := by
  funext i
  simp [xor32Bits, bitsOf, xor32]

/-- **`PouwEq32_v2`'s lowering computes `PouwEq32_v1`.** -/
theorem eq32Bits_bitsOf (x y : BitVec 32) : eq32Bits (bitsOf x) (bitsOf y) = eq32 x y := by
  apply Bool.eq_iff_iff.2
  rw [eq32Bits, tree_andBit]
  simp only [List.mem_ofFn, forall_exists_index, forall_apply_eq_imp_iff, eq32, decide_eq_true_eq, bitsOf]
  constructor
  · intro h
    refine BitVec.eq_of_getLsbD_eq fun i hi => ?_
    have := h ⟨i, hi⟩
    revert this
    cases x.getLsbD i <;> cases y.getLsbD i <;> simp
  · rintro rfl _
    simp

/-- **The combine's body on bits computes it on words**: on the bits of any partials and receipt, `combineBits` is
`combineOk`. -/
theorem combineBits_bitsOf {L nw : ℕ} (p : Fin L → Fin nw → BitVec 32) (r : Fin nw → BitVec 32) :
    combineBits (fun l w => bitsOf (p l w)) (fun w => bitsOf (r w)) = combineOk p r := by
  unfold combineBits combineOk
  congr 2
  funext w
  rw [← eq32Bits_bitsOf]
  congr 1
  have h := tree_map xor32 xor32Bits bitsOf 0 (fun _ => false) (fun a b => (xor32Bits_bitsOf a b).symm)
    (by funext i; simp [bitsOf]) (List.ofFn fun l => p l w)
  rw [List.map_ofFn] at h
  exact h

end Pouw.H1Zy
