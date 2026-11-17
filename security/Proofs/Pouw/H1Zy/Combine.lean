module

public import Definitions.Pouw.H1Zy.Combine
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.Push
import Mathlib.Tactic.SetLike

/-! The combine's `ok` is the receipt check. -/

@[expose] public section

namespace Pouw.H1Zy

theorem foldl_pairs {α : Type} (f : α → α → α) (hf : ∀ a b c, f (f a b) c = f a (f b c)) :
    ∀ (l : List α) (x : α), (pairs f l).foldl f x = l.foldl f x
  | [], _ => rfl
  | [_], _ => rfl
  | a :: b :: rest, x => by
      simp only [pairs, List.foldl_cons]
      rw [foldl_pairs f hf rest, hf]

theorem tree_eq_foldl {α : Type} (f : α → α → α) (d : α) (hf : ∀ a b c, f (f a b) c = f a (f b c))
    (hd : ∀ a, f d a = a) : ∀ l : List α, tree f d l = l.foldl f d
  | [] => by simp [tree]
  | [a] => by simp [tree, hd]
  | a :: b :: rest => by
      rw [tree, tree_eq_foldl f d hf hd (pairs f (a :: b :: rest)), foldl_pairs f hf]
termination_by l => l.length
decreasing_by exact length_pairs_lt f a b rest

theorem tree_xor32 (l : List (BitVec 32)) : tree xor32 0 l = l.foldl (· ^^^ ·) 0 :=
  tree_eq_foldl xor32 0 (fun a b c => BitVec.xor_assoc a b c) (fun _ => BitVec.zero_xor) l

theorem tree_andBit (l : List Bool) : tree andBit true l = true ↔ ∀ b ∈ l, b = true := by
  rw [tree_eq_foldl andBit true (fun a b c => Bool.and_assoc a b c) (fun a => Bool.true_and a) l]
  have key : ∀ (l : List Bool) (x : Bool), l.foldl andBit x = (x && l.all id) := by
    intro l
    induction l with
    | nil => intro x; simp
    | cons b l ih => intro x; simp [List.foldl_cons, ih, andBit, Bool.and_assoc]
  simp [key]

theorem combineOk_iff {L nw : ℕ} (partials : Fin L → Fin nw → BitVec 32) (receipt : Fin nw → BitVec 32) :
    combineOk partials receipt = true ↔
      ∀ w, (List.ofFn fun l => partials l w).foldl (· ^^^ ·) 0 = receipt w := by
  unfold combineOk
  rw [tree_andBit]
  simp only [List.mem_ofFn, forall_exists_index, forall_apply_eq_imp_iff, eq32, decide_eq_true_eq, tree_xor32]

end Pouw.H1Zy
