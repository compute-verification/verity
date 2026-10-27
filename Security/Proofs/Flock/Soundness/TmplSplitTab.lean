import Proofs.Flock.Soundness.TmplFoldClassOk
import Flock.TmplSplit

/-!
# The split fold's tables, valued

`Tmpl.walk` keeps a class's fold in sparse tables (`Tab`, `HashMap`s keyed by chunks). A table's value under a key
weight `f` is `hsum`: each entry's value times its key's weight.

- `hsum_superset`: the value is a sum over any finite set of keys holding the table's.
- `hsum_bump`: `bump m k v` adds `v · f k`.
- `fold_val`: a fold over a table whose every step adds a known term adds their sum.
- `eqHalf_at`, `eqTail_split_at`: the `eq` tables `walk` reads, split at a chunk boundary.
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

/-- A table's value: each entry's value times its key's weight. -/
noncomputable def hsum {κ : Type} [BEq κ] [Hashable κ] (m : HashMap κ F128) (f : κ → GF128) : GF128 :=
  (m.toList.map fun e => of e.2 * f e.1).sum

theorem hsum_superset {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] [DecidableEq κ]
    (m : HashMap κ F128) (f : κ → GF128) (K : Finset κ) (hK : ∀ x, x ∈ m → x ∈ K) :
    hsum m f = ∑ x ∈ K, of (m.getD x 0) * f x := by
  have h1 : hsum m f = (m.keys.map fun x => of (m.getD x 0) * f x).sum := by
    unfold hsum
    rw [← HashMap.map_fst_toList_eq_keys, List.map_map]
    congr 1
    refine List.map_congr_left fun e he => ?_
    have hv : m[e.1]? = some e.2 := (HashMap.mem_toList_iff_getElem?_eq_some).1 he
    simp only [Function.comp, HashMap.getD_eq_getD_getElem?, hv, Option.getD_some]
  rw [h1, ← List.sum_toFinset _ HashMap.nodup_keys]
  refine Finset.sum_subset (fun x hx => hK x (HashMap.mem_keys.1 (List.mem_toFinset.1 hx))) fun x _ hx => ?_
  have hn : ¬ x ∈ m := fun h => hx (List.mem_toFinset.2 (HashMap.mem_keys.2 h))
  rw [HashMap.getD_eq_fallback hn, of_zero', zero_mul]

theorem hsum_bump {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] [DecidableEq κ]
    (m : HashMap κ F128) (f : κ → GF128) (k : κ) (v : F128) :
    hsum (bump m k v) f = hsum m f + of v * f k := by
  let K := insert k m.keys.toFinset
  have hK : ∀ x, x ∈ m → x ∈ K := fun x hx => Finset.mem_insert_of_mem (List.mem_toFinset.2 (HashMap.mem_keys.2 hx))
  have hK' : ∀ x, x ∈ bump m k v → x ∈ K := fun x hx => by
    unfold bump at hx
    rcases HashMap.mem_insert.1 hx with h | h
    · rw [← beq_iff_eq.1 h]; exact Finset.mem_insert_self _ _
    · exact hK x h
  rw [hsum_superset _ f K hK', hsum_superset m f K hK]
  have hpt : ∀ x ∈ K, of ((bump m k v).getD x 0) * f x =
      of (m.getD x 0) * f x + if k = x then of v * f k else 0 := fun x _ => by
    unfold bump
    rw [HashMap.getD_insert]
    by_cases h : k = x
    · subst h
      simp [of_add, add_mul]
    · simp [h]
  rw [Finset.sum_congr rfl hpt, Finset.sum_add_distrib, Finset.sum_ite_eq _ _ _, if_pos (Finset.mem_insert_self _ _)]

/-- A fold over a table whose every step adds `G k v` to `V` adds their sum. -/
theorem fold_val {κ δ : Type} [BEq κ] [Hashable κ] (V : δ → GF128) (step : δ → κ → F128 → δ) (G : κ → F128 → GF128)
    (h : ∀ d k v, V (step d k v) = V d + G k v) (m : HashMap κ F128) (init : δ) :
    V (m.fold step init) = V init + (m.toList.map fun e => G e.1 e.2).sum := by
  rw [HashMap.fold_eq_foldl_toList]
  generalize m.toList = l
  induction l generalizing init with
  | nil => simp
  | cons e l ih => rw [List.foldl_cons, ih, h, List.map_cons, List.sum_cons, add_assoc]

/-- `eq(r[lo .. lo + k + m], u + 2^k j)` splits at bit `k`. -/
theorem eqAt_bits_at (r : Array F128) {lo k m u j : ℕ} (hu : u < 2 ^ k) (hj : j < 2 ^ m) :
    of (Tmpl.eqAt r lo (k + m) (u + 2 ^ k * j)) = of (Tmpl.eqAt r lo k u) * of (Tmpl.eqAt r (lo + k) m j) := by
  rw [of_eqAt r lo (k + m) _ (lane_lt hu hj), of_eqAt r lo k u hu, of_eqAt r (lo + k) m j hj, FlockLevel3.eqAt_split k m,
    show (u + 2 ^ k * j) % 2 ^ k = u by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hu],
    show (u + 2 ^ k * j) / 2 ^ k = j by rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos k), Nat.div_eq_of_lt hu, Nat.zero_add]]
  simp only [Fin.val_castAdd, Fin.val_natAdd, Nat.add_assoc]

/-- `walk`'s two-half `eq` table reads `eq(v[lo .. lo + n], u)`. -/
theorem eqHalf_at (v : Array F128) (lo n u : ℕ) (hu : u < 2 ^ n) :
    of ((eqHalf v lo n).at u) = of (Tmpl.eqAt v lo n u) := by
  have hh : n / 2 ≤ n := Nat.div_le_self _ _
  have h1 : u % 2 ^ (n / 2) < 2 ^ (n / 2) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have h2 : u / 2 ^ (n / 2) < 2 ^ (n - n / 2) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← Nat.pow_add, Nat.sub_add_cancel hh]
    exact hu
  unfold Half.at eqHalf
  dsimp only
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, h1, h2, ↓reduceIte,
    Option.map_some, Option.getD_some]
  rw [of_mul, ← eqAt_bits_at v h1 h2, Nat.mod_add_div, Nat.add_sub_cancel' hh]

/-- **`eq(r[lo ..], ·)` split at bit `k`**: `eq(r[lo ..], u + 2^k v) = eq(r[lo .. lo + k], u) · eq(r[lo + k ..], v)`. -/
theorem eqTail_split_at (r : Array F128) (lo k u v : ℕ) (hk : lo + k ≤ r.size) (hu : u < 2 ^ k) :
    of (Tmpl.eqTail (Tmpl.halves r) lo (u + 2 ^ k * v)) =
      of (Tmpl.eqAt r lo k u) * of (Tmpl.eqTail (Tmpl.halves r) (lo + k) v) := by
  by_cases hv : v < 2 ^ (r.size - (lo + k))
  · have hlt : u + 2 ^ k * v < 2 ^ (r.size - lo) := by
      calc u + 2 ^ k * v < 2 ^ k + 2 ^ k * v := Nat.add_lt_add_right hu _
        _ = 2 ^ k * (v + 1) := by ring
        _ ≤ 2 ^ k * 2 ^ (r.size - (lo + k)) := Nat.mul_le_mul_left _ hv
        _ = 2 ^ (r.size - lo) := by rw [← pow_add]; congr 1; omega
    rw [eqTail_at r lo _ (by omega) hlt, of_eqAt r lo k u hu, eqTail_at r (lo + k) v hk hv]
    have e1 : FlockLevel3.eqAt (fun j : Fin (r.size - lo) => of (r.getD (lo + j.val) 0)) (u + 2 ^ k * v) =
        FlockLevel3.eqAt (fun j : Fin (k + (r.size - (lo + k))) => of (r.getD (lo + j.val) 0)) (u + 2 ^ k * v) := by
      unfold FlockLevel3.eqAt
      refine Fintype.prod_equiv (finCongr (by omega)) _ _ (fun j => ?_)
      simp only [finCongr_apply, Fin.val_cast]
    have hm : (u + 2 ^ k * v) % 2 ^ k = u := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hu]
    have hd : (u + 2 ^ k * v) / 2 ^ k = v := by
      rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos k), Nat.div_eq_of_lt hu, zero_add]
    rw [e1, eqAt_split, hm, hd]
    simp only [Fin.val_castAdd, Fin.val_natAdd, Nat.add_assoc]
  · rw [eqTail_out r (lo + k) v hk (by omega), mul_zero, eqTail_out r lo _ (by omega)]
    calc 2 ^ (r.size - lo) = 2 ^ k * 2 ^ (r.size - (lo + k)) := by rw [← pow_add]; congr 1; omega
      _ ≤ 2 ^ k * v := Nat.mul_le_mul_left _ (by omega)
      _ ≤ u + 2 ^ k * v := Nat.le_add_left _ _

end FlockSoundness.TmplFold
