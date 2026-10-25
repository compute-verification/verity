import Definitions.Pouw.H1Zy.Combine

/-!
# The combine's `ok` is the receipt check

`Pouw.H1Zy.combineOk`, `PouwH1Combine_v1`'s body on words, is `true` exactly when, word by word, the left fold of XOR
over the lanes' words in lane order (from `0`) is the receipt's word (`combineOk_iff`): its `PouwXor32` tree is that fold
(`tree_xor32`), and its `PouwAndBit` tree is `true` exactly when every comparison is (`tree_andBit`). Both trees are the
fold of an associative operation from a left identity (`tree_eq_foldl`), so the tree's shape doesn't matter.
-/

namespace Pouw.H1Zy

/-- A pass of the tree keeps an associative fold. -/
theorem foldl_pairs {α : Type} (f : α → α → α) (hf : ∀ a b c, f (f a b) c = f a (f b c)) :
    ∀ (l : List α) (x : α), (pairs f l).foldl f x = l.foldl f x
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: rest, x => by
      simp only [pairs, List.foldl_cons]
      rw [foldl_pairs f hf rest, hf]

/-- **The tree is the fold**: for an associative `f` with `d` a left identity, `tree f d l` is `l`'s left fold from `d`. -/
theorem tree_eq_foldl {α : Type} (f : α → α → α) (d : α) (hf : ∀ a b c, f (f a b) c = f a (f b c))
    (hd : ∀ a, f d a = a) : ∀ l : List α, tree f d l = l.foldl f d
  | [] => by simp [tree]
  | [a] => by simp [tree, hd]
  | a :: b :: rest => by
      rw [tree, tree_eq_foldl f d hf hd (pairs f (a :: b :: rest)), foldl_pairs f hf]
termination_by l => l.length
decreasing_by exact length_pairs_lt f a b rest

/-- The `PouwXor32` tree is the left fold of XOR from `0`. -/
theorem tree_xor32 (l : List (BitVec 32)) : tree xor32 0 l = l.foldl (· ^^^ ·) 0 :=
  tree_eq_foldl xor32 0 (fun a b c => BitVec.xor_assoc a b c) (fun _ => BitVec.zero_xor) l

/-- The `PouwAndBit` tree is `true` exactly when every bit is. -/
theorem tree_andBit (l : List Bool) : tree andBit true l = true ↔ ∀ b ∈ l, b = true := by
  rw [tree_eq_foldl andBit true (fun a b c => Bool.and_assoc a b c) (fun a => Bool.true_and a) l]
  have key : ∀ (l : List Bool) (x : Bool), l.foldl andBit x = (x && l.all id) := by
    intro l
    induction l with
    | nil => intro x; simp
    | cons b l ih => intro x; simp [List.foldl_cons, ih, andBit, Bool.and_assoc]
  simp [key]

/-- **The combine's `ok` is the receipt check**: `true` exactly when, at every word `w`, the XOR of the lanes' words `w`
in lane order is the receipt's word `w`. -/
theorem combineOk_iff {L nw : ℕ} (partials : Fin L → Fin nw → BitVec 32) (receipt : Fin nw → BitVec 32) :
    combineOk partials receipt = true ↔
      ∀ w, (List.ofFn fun l => partials l w).foldl (· ^^^ ·) 0 = receipt w := by
  unfold combineOk
  rw [tree_andBit]
  simp only [List.mem_ofFn, forall_exists_index, forall_apply_eq_imp_iff, eq32, decide_eq_true_eq, tree_xor32]

end Pouw.H1Zy
