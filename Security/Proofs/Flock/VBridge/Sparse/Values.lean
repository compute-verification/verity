import Proofs.Flock.VBridge.Sparse.Gadgets
import Proofs.Flock.Soundness.Refine.Arith

/-!
# The gadgets' values

What the mirrors in `Gadgets.lean` compute: `treeVal` is the factors' product in the field (`fOf_treeVal`), with as
many levels as factors; `sumVal` their sum (`fOf_sumVal`); and `demuxVal` at the bits `T_LOG − 1, …, 0` puts each slot's
vector at the entry its type's low `T_LOG` bits name, zero at every other (`demuxVal_eq`, `splits_range`, `low_testBit`).
-/

namespace FlockVBridge.Sparse

open Flock (F128)
open FlockSoundness.Refine (fOf fOf_add fOf_mul)

/-! ## The product tree and the sums -/

theorem prod_pairsVal : ∀ xs : List F128, ((pairsVal xs).map fOf).prod = (xs.map fOf).prod
  | [] => rfl
  | [_] => rfl
  | x :: y :: rest => by
      simp only [pairsVal, List.map_cons, List.prod_cons, fOf_mul, prod_pairsVal rest, mul_assoc]

theorem length_pairsVal : ∀ xs : List F128, 2 * (pairsVal xs).length ≤ xs.length + 1
  | [] => by simp [pairsVal]
  | [_] => by simp [pairsVal]
  | x :: y :: rest => by
      have := length_pairsVal rest
      simp only [pairsVal, List.length_cons]
      omega

/-- **The tree is the product**: with at least as many levels as factors less one, `treeVal` is the factors' product. -/
theorem fOf_treeVal : ∀ (n : ℕ) (xs : List F128), xs ≠ [] → xs.length ≤ n + 1 →
    fOf (treeVal n xs) = (xs.map fOf).prod
  | 0, [], h, _ => absurd rfl h
  | 0, [x], _, _ => by simp [treeVal]
  | 0, _ :: _ :: _, _, h => by simp only [List.length_cons] at h; omega
  | n + 1, [], h, _ => absurd rfl h
  | n + 1, [x], _, _ => by simp [treeVal]
  | n + 1, x :: y :: rest, _, h => by
      have hl := length_pairsVal (x :: y :: rest)
      simp only [List.length_cons] at h hl
      rw [treeVal, fOf_treeVal n (pairsVal (x :: y :: rest)) (by simp [pairsVal]) (by omega), prod_pairsVal]

theorem fOf_foldl_add : ∀ (xs : List F128) (x : F128), fOf (xs.foldl (· + ·) x) = fOf x + (xs.map fOf).sum
  | [], x => by simp
  | y :: ys, x => by
      simp only [List.foldl_cons, fOf_foldl_add ys, fOf_add, List.map_cons, List.sum_cons, add_assoc]

theorem fOf_sumVal (xs : List F128) (h : xs ≠ []) : fOf (sumVal xs) = (xs.map fOf).sum := by
  cases xs with
  | nil => exact absurd rfl h
  | cons x xs => rw [sumVal, fOf_foldl_add, List.map_cons, List.sum_cons]

/-! ## The demultiplexer -/

/-- The number bits `b 0 … b (m − 1)` spell, least significant first. -/
def low (b : ℕ → Bool) : ℕ → ℕ
  | 0 => 0
  | m + 1 => low b m + 2 ^ m * (b m).toNat

theorem low_lt (b : ℕ → Bool) : ∀ m, low b m < 2 ^ m
  | 0 => by simp [low]
  | m + 1 => by
      have := low_lt b m
      have := Bool.toNat_le (b m)
      rw [low, Nat.pow_succ]
      have : 2 ^ m * (b m).toNat ≤ 2 ^ m * 1 := Nat.mul_le_mul_left _ this
      omega

theorem low_congr {b b' : ℕ → Bool} : ∀ {m}, (∀ j < m, b j = b' j) → low b m = low b' m
  | 0, _ => rfl
  | m + 1, h => by rw [low, low, low_congr fun j hj => h j (by omega), h m (by omega)]

/-- The bits of `τ` spell `τ`'s low bits. -/
theorem low_testBit (τ : ℕ) : ∀ m, low τ.testBit m = τ % 2 ^ m
  | 0 => by simp [low, Nat.mod_one]
  | m + 1 => by rw [low, low_testBit τ m, Nat.mod_pow_succ, Nat.toNat_testBit]

/-- One slot's splits at the bits `js` in order, `b` its type's bits. -/
def splitsVal (b : ℕ → Bool) : List ℕ → List F128 → List F128
  | [], Ys => Ys
  | j :: js, Ys => splitsVal b js (splitVal (b j) Ys)

theorem splitsVal_append (b : ℕ → Bool) : ∀ (js : List ℕ) (A B : List F128),
    splitsVal b js (A ++ B) = splitsVal b js A ++ splitsVal b js B
  | [], _, _ => rfl
  | j :: js, A, B => by simp only [splitsVal, splitVal, List.flatMap_append, splitsVal_append b js]

/-- **The splits at `m − 1, …, 0` select by the low `m` bits**: entry `t` is the vector where the bits spell `t`. -/
theorem splits_range (b : ℕ → Bool) : ∀ (m : ℕ) (X : F128),
    splitsVal b (List.range m).reverse [X] = (List.range (2 ^ m)).map fun t => if low b m = t then X else 0
  | 0, X => by simp [splitsVal, low]
  | m + 1, X => by
      have hl := low_lt b m
      rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append, splitsVal,
        show splitVal (b m) [X] = [if b m then 0 else X] ++ [if b m then X else 0] by simp [splitVal],
        splitsVal_append, splits_range b m, splits_range b m, Nat.two_pow_succ, List.range_add, List.map_append,
        List.map_map]
      congr 1
      all_goals
        refine List.map_congr_left fun t ht => ?_
        rw [List.mem_range] at ht
        cases hb : b m <;>
          simp only [low, hb, Function.comp_apply, Bool.toNat_true, Bool.toNat_false, Bool.false_eq_true, eq_self,
            ↓reduceIte, Nat.mul_one, Nat.mul_zero, Nat.add_zero] <;>
          split_ifs <;> first | rfl | omega

theorem zipIdx_map_self {α β : Type} (f : α × ℕ → β) : ∀ (l : List α) (k : ℕ),
    ((l.zipIdx k).map f).zipIdx k = (l.zipIdx k).map fun p => (f p, p.2)
  | [], _ => rfl
  | x :: l, k => by simp only [List.zipIdx_cons, List.map_cons, zipIdx_map_self f l (k + 1)]

/-- **`demux` slot by slot**: slot `s`'s vectors split at the bits `js` of its type, `tb s`. -/
theorem demuxVal_eq (tb : ℕ → ℕ → Bool) : ∀ (js : List ℕ) (Xs : List (List F128)),
    demuxVal tb js Xs = (Xs.zipIdx 0).map fun p => splitsVal (tb p.2) js p.1
  | [], Xs => (List.zipIdx_map_fst 0 Xs).symm
  | j :: js, Xs => by
      rw [demuxVal, demuxVal_eq tb js, demuxLevelVal, zipIdx_map_self, List.map_map]
      rfl

end FlockVBridge.Sparse
