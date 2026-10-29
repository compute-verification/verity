import Proofs.Flock.Soundness.TmplFoldRef

/-!
# The template fold's arithmetic

What `Tmpl.foldDef` relies on besides the call tree:

- `sum_canon`: a row's columns summed after `canon` (sorted, equal pairs dropped) are summed before it, over GF(2^128).
- `laneSum_eq`: `laneSum f m count` is `Σ_{j < count} Π_{b < m} f b (bit b of j)`, a run's copies summed.
- `eqTail_at`, `eqTail_split`: the halves table read at `lo` is `eq(r[lo ..], ·)`, and `eq(r, u + 2^k v)` splits into
  `eq(r[0 .. k], u) · eq(r[k ..], v)`, the low and high factors of a summed class.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-! ## `canon` -/

theorem sum_dropPairs (f : ℕ → GF128) : ∀ l : List ℕ, ((dropPairs l).map f).sum = (l.map f).sum
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
    rw [dropPairs]
    split
    · rename_i h
      have hab : a = b := by simpa using h
      subst hab
      rw [sum_dropPairs f rest]
      simp only [List.map_cons, List.sum_cons]
      rw [← add_assoc, CharTwo.add_self_eq_zero, zero_add]
    · rw [List.map_cons, List.sum_cons, sum_dropPairs f (b :: rest)]
      simp only [List.map_cons, List.sum_cons]

theorem sum_canon (f : ℕ → GF128) (xs : Array ℕ) : ((canon xs).toList.map f).sum = (xs.toList.map f).sum := by
  show ((dropPairs xs.toList.mergeSort).map f).sum = _
  rw [sum_dropPairs]
  exact ((List.mergeSort_perm _ _).map f).sum_eq

/-! ## `laneSum` -/

theorem testBit_low (x m b : ℕ) (hb : b < m) : x.testBit b = (x % 2 ^ m).testBit b := by
  simp [Nat.testBit_mod_two_pow, hb]

theorem testBit_top (x m : ℕ) : x.testBit m = (x / 2 ^ m).testBit 0 := by
  rw [Nat.testBit_div_two_pow, Nat.zero_add]

theorem prod_bits_lo (g : ℕ → Bool → GF128) (m j : ℕ) (hj : j < 2 ^ m) :
    ∏ b : Fin (m + 1), g b (j.testBit b) = (∏ b : Fin m, g b (j.testBit b)) * g m false := by
  have ht : j.testBit m = false := by rw [testBit_top, Nat.div_eq_of_lt hj]; simp
  rw [Fin.prod_univ_castSucc]
  simp only [Fin.val_castSucc, Fin.val_last, ht]

theorem prod_bits_hi (g : ℕ → Bool → GF128) (m j : ℕ) (hj : j < 2 ^ m) :
    ∏ b : Fin (m + 1), g b ((2 ^ m + j).testBit b) = (∏ b : Fin m, g b (j.testBit b)) * g m true := by
  have ht : (2 ^ m + j).testBit m = true := by
    rw [testBit_top, Nat.add_div_left _ (Nat.two_pow_pos m), Nat.div_eq_of_lt hj]
    rfl
  rw [Fin.prod_univ_castSucc]
  simp only [Fin.val_castSucc, Fin.val_last, ht]
  congr 1
  refine Finset.prod_congr rfl fun b _ => ?_
  rw [testBit_low (2 ^ m + j) m b b.isLt, Nat.add_mod_left, Nat.mod_eq_of_lt hj]

/-- Every choice of bits, summed: `Σ_{j < 2^m} Π_{b < m} g b (bit b of j) = Π_{b < m} (g b 0 + g b 1)`. -/
theorem sum_bits (g : ℕ → Bool → GF128) : ∀ m : ℕ,
    ∑ j ∈ Finset.range (2 ^ m), ∏ b : Fin m, g b (j.testBit b) = ∏ b : Fin m, (g b false + g b true)
  | 0 => by simp
  | m + 1 => by
    rw [pow_succ, mul_two, Finset.sum_range_add]
    rw [Finset.sum_congr rfl fun j hj => prod_bits_lo g m j (Finset.mem_range.mp hj),
      Finset.sum_congr rfl fun j hj => prod_bits_hi g m j (Finset.mem_range.mp hj)]
    refine (congrArg₂ (· + ·) (Finset.sum_mul _ _ _).symm (Finset.sum_mul _ _ _).symm).trans ?_
    refine (congrArg₂ (fun x y => x * g m false + y * g m true) (sum_bits g m) (sum_bits g m)).trans ?_
    rw [Fin.prod_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    ring

theorem foldl_prod (f : ℕ → Bool → F128) : ∀ m : ℕ,
    of ((List.range m).foldl (fun acc b => acc * (f b false + f b true)) 1) =
      ∏ b : Fin m, (of (f b false) + of (f b true))
  | 0 => by simp [← one_eq]
  | m + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, of_mul, foldl_prod f m,
      Fin.prod_univ_castSucc, of_add]
    simp only [Fin.val_castSucc, Fin.val_last]

/-- **A run's lane sum**: `laneSum f m count` is the sum over the run's `count ≤ 2^m` copies of the product of `f` at
each copy index's bits. -/
theorem laneSum_eq (f : ℕ → Bool → F128) : ∀ (m count : ℕ), count ≤ 2 ^ m →
    of (laneSum f m count) = ∑ j ∈ Finset.range count, ∏ b : Fin m, of (f b (j.testBit b))
  | 0, count, h => by
    have h1 : count ≤ 1 := by simpa using h
    interval_cases count <;> simp [laneSum, of_zero', ← one_eq]
  | m + 1, count, h => by
    rw [laneSum]
    split
    · rename_i hc
      rw [of_mul, laneSum_eq f m count hc]
      refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun j hj => ?_)
      rw [prod_bits_lo (fun b x => of (f b x)) m j (lt_of_lt_of_le (Finset.mem_range.mp hj) hc)]
      ring
    · rename_i hc
      have h2 : 2 ^ (m + 1) = 2 ^ m + 2 ^ m := by rw [pow_succ, mul_two]
      have hc' : count - 2 ^ m ≤ 2 ^ m := by omega
      rw [of_add, of_mul, of_mul, laneSum_eq f m _ hc', foldl_prod,
        ← sum_bits (fun b x => of (f b x)) m,
        show Finset.range count = Finset.range (2 ^ m + (count - 2 ^ m)) by congr 1; omega,
        Finset.sum_range_add]
      refine congrArg₂ (· + ·) ((Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun j hj => ?_))
        ((Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun j hj => ?_))
      · rw [prod_bits_lo (fun b x => of (f b x)) m j (Finset.mem_range.mp hj)]
        ring
      · rw [prod_bits_hi (fun b x => of (f b x)) m j (by have := Finset.mem_range.mp hj; omega)]
        ring

/-! ## `eq` tables read at `lo` -/

/-- The halves table read at `lo` is `eq(r[lo ..], u)`. -/
theorem eqTail_at (r : Array F128) (lo u : ℕ) (hlo : lo ≤ r.size) (hu : u < 2 ^ (r.size - lo)) :
    of (Tmpl.eqTail (Tmpl.halves r) lo u) =
      FlockLevel3.eqAt (fun j : Fin (r.size - lo) => of (r.getD (lo + j.val) 0)) u := by
  have hh : (r.size - lo) / 2 ≤ r.size - lo := Nat.div_le_self _ _
  have h1 : u % 2 ^ ((r.size - lo) / 2) < 2 ^ ((r.size - lo) / 2) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have h2 : u / 2 ^ ((r.size - lo) / 2) < 2 ^ (r.size - lo - (r.size - lo) / 2) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← Nat.pow_add, Nat.sub_add_cancel hh]
    exact hu
  have hT : (Tmpl.halves r).getD lo (0, #[], #[]) =
      ((r.size - lo) / 2, (Array.range (2 ^ ((r.size - lo) / 2))).map (Tmpl.eqAt r lo ((r.size - lo) / 2)),
        (Array.range (2 ^ (r.size - lo - (r.size - lo) / 2))).map
          (Tmpl.eqAt r (lo + (r.size - lo) / 2) (r.size - lo - (r.size - lo) / 2))) := by
    simp [Tmpl.halves, Nat.lt_succ_of_le hlo]
  unfold Tmpl.eqTail
  rw [hT]
  dsimp only
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, h1, h2, ↓reduceIte,
    Option.map_some, Option.getD_some]
  rw [of_mul, of_eqAt r _ _ _ h1, of_eqAt r _ _ _ h2]
  simp only [Array.getD_eq_getD_getElem?]
  have e1 : FlockLevel3.eqAt (fun j : Fin (r.size - lo) => of (r[lo + j.val]?.getD 0)) u =
      FlockLevel3.eqAt (fun j : Fin ((r.size - lo) / 2 + (r.size - lo - (r.size - lo) / 2)) =>
        of (r[lo + j.val]?.getD 0)) u := by
    unfold FlockLevel3.eqAt
    refine Fintype.prod_equiv (finCongr (by omega)) _ _ (fun j => ?_)
    simp only [finCongr_apply, Fin.val_cast]
  rw [e1, eqAt_split]
  simp only [Fin.val_castAdd, Fin.val_natAdd, Nat.add_assoc]

/-- **`eq` split at bit `k`**: `eq(r, u + 2^k v) = eq(r[0 .. k], u) · eq(r[k ..], v)`. -/
theorem eqTail_split (r : Array F128) (k u v : ℕ) (hk : k ≤ r.size) (hu : u < 2 ^ k) (hv : v < 2 ^ (r.size - k)) :
    of (Tmpl.eqTail (Tmpl.halves r) 0 (u + 2 ^ k * v)) =
      of (Tmpl.eqAt r 0 k u) * of (Tmpl.eqTail (Tmpl.halves r) k v) := by
  have hlt : u + 2 ^ k * v < 2 ^ (r.size - 0) := by
    calc u + 2 ^ k * v < 2 ^ k + 2 ^ k * v := Nat.add_lt_add_right hu _
      _ = 2 ^ k * (v + 1) := by ring
      _ ≤ 2 ^ k * 2 ^ (r.size - k) := Nat.mul_le_mul_left _ hv
      _ = 2 ^ (r.size - 0) := by rw [← pow_add]; congr 1; omega
  rw [eqTail_at r 0 _ (Nat.zero_le _) hlt, of_eqAt r 0 k u hu, eqTail_at r k v hk hv]
  have e1 : FlockLevel3.eqAt (fun j : Fin (r.size - 0) => of (r.getD (0 + j.val) 0)) (u + 2 ^ k * v) =
      FlockLevel3.eqAt (fun j : Fin (k + (r.size - k)) => of (r.getD (0 + j.val) 0)) (u + 2 ^ k * v) := by
    unfold FlockLevel3.eqAt
    refine Fintype.prod_equiv (finCongr (by omega)) _ _ (fun j => ?_)
    simp only [finCongr_apply, Fin.val_cast]
  have hm : (u + 2 ^ k * v) % 2 ^ k = u := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hu]
  have hd : (u + 2 ^ k * v) / 2 ^ k = v := by
    rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos k), Nat.div_eq_of_lt hu, zero_add]
  rw [e1, eqAt_split, hm, hd]
  simp only [Fin.val_castAdd, Fin.val_natAdd, Nat.zero_add]

end FlockSoundness.TmplFold
