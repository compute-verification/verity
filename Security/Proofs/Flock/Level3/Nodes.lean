/-
`Arith.Correct.nodes` (§9): the skip nodes `S = φ8(0…63)` and the output coset `Λ = φ8(64…127)` are 128 distinct points.
`lagS`, `lagΛ` (the executable's `lagrange`) are the Lagrange weights on each, and `combW` (`combWeights`) evaluates a
polynomial of degree below 128 that vanishes on `S` from its values on `Λ`. φ8 is additive, so each node set is a
subspace or a coset, and the barycentric denominator `Π_{j ≠ i} (x_i − x_j)` is the executable's `den`.
-/
import Proofs.Flock.Level3.ArithFacts
import Mathlib.LinearAlgebra.Lagrange

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-! ### φ8 is additive and vanishes only at 0 -/

theorem of_add (a b : F128) : GF128.of (a + b) = GF128.of a + GF128.of b := (GF128.add_eq a b).symm
theorem of_mul (a b : F128) : GF128.of (a * b) = GF128.of a * GF128.of b := (GF128.mul_eq a b).symm
theorem of_zero : GF128.of F128.zero = 0 := GF128.zero_eq.symm

theorem phi8Sum_xor (a b : ℕ) : ∀ n,
    GF128.of (phi8Sum (a ^^^ b) n) = GF128.of (phi8Sum a n) + GF128.of (phi8Sum b n)
  | 0 => by simp only [phi8Sum, of_zero, add_zero]
  | n + 1 => by
    have ih := phi8Sum_xor a b n
    have h2 := CharTwo.add_self_eq_zero (GF128.of (phi8Pow n))
    simp only [phi8Sum, Nat.testBit_xor]
    cases a.testBit n <;> cases b.testBit n <;> simp only [Bool.xor_false, Bool.xor_true, Bool.not_false,
      Bool.not_true, ite_true, ite_false, Bool.false_eq_true, of_add, ih]
    · ring
    · ring
    · linear_combination (-1 : GF128) * h2

theorem phi8_eq (v : ℕ) (hv : v < 256) : phi8 v = phi8Sum v 8 := by
  simp [phi8, phi8Table, Nat.mod_eq_of_lt hv, Array.getD, hv]

theorem phi8_xor (a b : ℕ) (ha : a < 256) (hb : b < 256) :
    GF128.of (phi8 (a ^^^ b)) = GF128.of (phi8 a) + GF128.of (phi8 b) := by
  have hab : a ^^^ b < 256 := Nat.xor_lt_two_pow (n := 8) ha hb
  rw [phi8_eq _ hab, phi8_eq _ ha, phi8_eq _ hb, phi8Sum_xor]

/-- `g^i`, `i < 8`, as constants (checked once against `phi8Pow`). -/
def phi8PowC : Array F128 := #[⟨0x1, 0x0⟩, ⟨0x6b8330483c2e9849, 0xdcb364640a222fe⟩, ⟨0x7573da4a5f7710ed, 0x3d5bd35c94646a24⟩,
  ⟨0x41a12db1f974f3ac, 0x6d58c4e181f9199f⟩, ⟨0x5e2f716f4ede412f, 0xa72ec17764d7ced5⟩, ⟨0x5cb10fbabcf00118, 0x4d52354a3a3d8c86⟩,
  ⟨0x95ed1f57f3632d4d, 0x553e92e8bc0ae9a7⟩, ⟨0x512625b1f09fa87e, 0x93252331bf042b11⟩]

theorem phi8Pow_eq : ∀ i : Fin 8, phi8Pow i = phi8PowC[i.val]! := by decide +kernel

def phi8SumC (v : ℕ) : ℕ → F128
  | 0 => F128.zero
  | i + 1 => if v.testBit i then phi8SumC v i + phi8PowC[i]! else phi8SumC v i

theorem phi8Sum_eq_C (v : ℕ) : ∀ n ≤ 8, phi8Sum v n = phi8SumC v n
  | 0, _ => rfl
  | n + 1, h => by
    simp only [phi8Sum, phi8SumC, phi8Sum_eq_C v n (by omega), phi8Pow_eq ⟨n, by omega⟩]

theorem phi8SumC_ne_zero : ∀ v : Fin 256, v.val ≠ 0 → phi8SumC v.val 8 ≠ F128.zero := by decide +kernel

theorem phi8_ne_zero (v : ℕ) (hv : v < 256) (h0 : v ≠ 0) : GF128.of (phi8 v) ≠ 0 := by
  rw [phi8_eq v hv, phi8Sum_eq_C v 8 le_rfl, GF128.zero_eq]
  exact fun h => phi8SumC_ne_zero ⟨v, hv⟩ h0 h

theorem phi8_injOn (a b : ℕ) (ha : a < 256) (hb : b < 256) (h : GF128.of (phi8 a) = GF128.of (phi8 b)) : a = b := by
  by_contra hne
  have hx : a ^^^ b ≠ 0 := fun h0 => hne (Nat.eq_of_testBit_eq fun i => by
    simpa [Nat.testBit_xor] using congrArg (Nat.testBit · i) h0)
  apply phi8_ne_zero _ (Nat.xor_lt_two_pow (n := 8) ha hb) hx
  rw [phi8_xor a b ha hb, h, CharTwo.add_self_eq_zero]

/-! ### The node maps -/

noncomputable def nodeS (s : Fin 64) : GF128 := GF128.of (phi8 s.val)
noncomputable def nodeΛ (i : Fin 64) : GF128 := GF128.of (phi8 (64 + i.val))
/-- All 128 nodes, `S` then `Λ`. -/
noncomputable def node128 (t : Fin 128) : GF128 := GF128.of (phi8 t.val)

theorem nodeS_injective : Function.Injective nodeS :=
  fun a b h => Fin.ext (phi8_injOn _ _ (by omega) (by omega) h)

theorem nodeΛ_injective : Function.Injective nodeΛ :=
  fun a b h => Fin.ext (by have := phi8_injOn _ _ (by omega) (by omega) h; omega)

theorem nodeS_ne_nodeΛ (s i : Fin 64) : nodeS s ≠ nodeΛ i :=
  fun h => by have := phi8_injOn _ _ (by omega) (by omega) h; omega

theorem node128_injective : Function.Injective node128 :=
  fun a b h => Fin.ext (phi8_injOn _ _ (by omega) (by omega) h)

/-! ### Translation: `Π_{j ≠ i} (x_i + x_j)` is `den d` -/

theorem of_foldl_mul (f : ℕ → F128) (l : List ℕ) (acc : F128) :
    GF128.of (l.foldl (fun a i => a * f i) acc) = GF128.of acc * (l.map fun i => GF128.of (f i)).prod := by
  induction l generalizing acc with
  | nil => simp
  | cons x l ih => rw [List.foldl_cons, ih, of_mul, List.map_cons, List.prod_cons]; ring

theorem of_den (d : ℕ) : GF128.of (den d) = ∏ j ∈ Finset.range (2 ^ d - 1), GF128.of (phi8 (j + 1)) := by
  rw [den, ← Array.foldl_toList, of_foldl_mul, ← GF128.one_eq, one_mul, Array.toList_range]
  rfl

theorem xor_lt (a b d : ℕ) (ha : a < 2 ^ d) (hb : b < 2 ^ d) : a ^^^ b < 2 ^ d := Nat.xor_lt_two_pow ha hb

theorem xor_xor_cancel (k j : ℕ) : k ^^^ (k ^^^ j) = j := by rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

theorem pow_le_256 (d : ℕ) (hd : d ≤ 7) (x : ℕ) (hx : x < 2 ^ d) : x < 256 :=
  lt_of_lt_of_le hx (le_trans (Nat.pow_le_pow_right (by norm_num) hd) (by norm_num))

/-- **Translation.** For `k < 2^d`, `Π_{j < 2^d, j ≠ k} (φ8(k) + φ8(j)) = den d`: the differences of a subspace's points
to one of them are the subspace's nonzero points. -/
theorem prod_translate (d : ℕ) (hd : d ≤ 7) (k : ℕ) (hk : k < 2 ^ d) :
    ∏ j ∈ (Finset.range (2 ^ d)).erase k, (GF128.of (phi8 k) + GF128.of (phi8 j)) = GF128.of (den d) := by
  have h1 : ∏ j ∈ (Finset.range (2 ^ d)).erase k, (GF128.of (phi8 k) + GF128.of (phi8 j)) =
      ∏ j ∈ (Finset.range (2 ^ d)).erase k, GF128.of (phi8 (k ^^^ j)) := by
    refine Finset.prod_congr rfl fun j hj => ?_
    have hj' : j < 2 ^ d := Finset.mem_range.mp (Finset.mem_of_mem_erase hj)
    rw [phi8_xor k j (pow_le_256 d hd k hk) (pow_le_256 d hd j hj')]
  have h2 : ∏ j ∈ (Finset.range (2 ^ d)).erase k, GF128.of (phi8 (k ^^^ j)) =
      ∏ i ∈ (Finset.range (2 ^ d)).erase 0, GF128.of (phi8 i) := by
    refine Finset.prod_nbij' (fun j => k ^^^ j) (fun i => k ^^^ i) ?_ ?_ ?_ ?_ ?_
    · intro j hj
      simp only [Finset.mem_erase, Finset.mem_range] at hj ⊢
      refine ⟨fun h0 => hj.1 ?_, xor_lt _ _ _ hk hj.2⟩
      have := xor_xor_cancel k j; rw [h0, Nat.xor_zero] at this; exact this.symm
    · intro i hi
      simp only [Finset.mem_erase, Finset.mem_range] at hi ⊢
      refine ⟨fun h => hi.1 ?_, xor_lt _ _ _ hk hi.2⟩
      have := xor_xor_cancel k i; rw [h, Nat.xor_self] at this; exact this.symm
    · intro j _; exact xor_xor_cancel k j
    · intro i _; exact xor_xor_cancel k i
    · intro j _; rfl
  have h3 : ∏ i ∈ (Finset.range (2 ^ d)).erase 0, GF128.of (phi8 i) = ∏ j ∈ Finset.range (2 ^ d - 1), GF128.of (phi8 (j + 1)) := by
    have hpos : 0 < 2 ^ d := Nat.two_pow_pos d
    rw [Finset.range_eq_Ico, ← Nat.Ico_succ_left_eq_erase_Ico, Finset.prod_Ico_eq_prod_range]
    exact Finset.prod_congr rfl fun j _ => by rw [Nat.succ_eq_add_one, zero_add, add_comm]
  rw [h1, h2, h3, of_den]

theorem add_64 : ∀ i : Fin 64, 64 + i.val = 64 ^^^ i.val := by decide +kernel

theorem phi8_coset (i : ℕ) (hi : i < 64) : GF128.of (phi8 (64 + i)) = GF128.of (phi8 64) + GF128.of (phi8 i) := by
  rw [add_64 ⟨i, hi⟩, phi8_xor 64 i (by norm_num) (by omega)]

/-- The coset's differences are the subspace's: `Π_{j < 64, j ≠ k} (λ_k + λ_j) = den 6`. -/
theorem prod_translate_coset (k : ℕ) (hk : k < 64) :
    ∏ j ∈ (Finset.range 64).erase k, (GF128.of (phi8 (64 + k)) + GF128.of (phi8 (64 + j))) = GF128.of (den 6) := by
  rw [← prod_translate 6 (by norm_num) k (by norm_num; exact hk)]
  refine Finset.prod_congr rfl fun j hj => ?_
  have hj' : j < 64 := Finset.mem_range.mp (Finset.mem_of_mem_erase hj)
  rw [phi8_coset k hk, phi8_coset j hj']
  have h2 := CharTwo.add_self_eq_zero (GF128.of (phi8 64))
  linear_combination h2

/-! ### The executable's weights are the Lagrange basis at `ζ` -/

theorem of_inv (a : F128) : GF128.of a.inv = (GF128.of a)⁻¹ := (GF128.inv_eq a).symm

theorem getElem!_range_map (f : ℕ → F128) (n i : ℕ) (hi : i < n) : ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]; simp

theorem of_foldl_node (f : ℕ → F128) (n : ℕ) (z : F128) :
    GF128.of (((Array.range n).map f).foldl (fun acc x => acc * (z + x)) 1) =
      ∏ j ∈ Finset.range n, (GF128.of z + GF128.of (f j)) := by
  rw [← Array.foldl_toList, Array.toList_map, Array.toList_range, List.foldl_map]
  have := of_foldl_mul (fun j => z + f j) (List.range n) 1
  rw [this, ← GF128.one_eq, one_mul]
  simp only [of_add]; rfl

variable (f : ℕ → F128) (n d : ℕ)

/-- The node values. -/
noncomputable def nv (j : ℕ) : GF128 := GF128.of (f j)

theorem lagrange_weight (hinj : ∀ a < n, ∀ b < n, GF128.of (f a) = GF128.of (f b) → a = b)
    (htr : ∀ k < n, ∏ j ∈ (Finset.range n).erase k, (GF128.of (f k) + GF128.of (f j)) = GF128.of (den d))
    (z : F128) (i : ℕ) (hi : i < n) :
    GF128.of ((lagrange ((Array.range n).map f) (den d).inv z)[i]!) =
      eval (GF128.of z) (Lagrange.basis (Finset.range n) (nv f) i) := by
  have hinjOn : Set.InjOn (nv f) (Finset.range n : Set ℕ) := fun a ha b hb h =>
    hinj a (by simpa using ha) b (by simpa using hb) h
  unfold lagrange
  cases hfind : ((Array.range n).map f).findIdx? (fun x => decide (x = z)) with
  | some k =>
    obtain ⟨hk, hkz, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp hfind
    have hk' : k < n := by simpa using hk
    have hz : f k = z := by simpa using hkz
    simp only
    rw [getElem!_pos _ i (by simpa using hi)]
    simp only [Array.getElem_map, Array.getElem_range, Array.size_map, Array.size_range]
    rw [← hz]
    by_cases hik : i = k
    · subst hik
      simp only [beq_self_eq_true, ite_true]
      rw [← GF128.one_eq]
      exact (Lagrange.eval_basis_self hinjOn (by simpa using hi)).symm
    · rw [if_neg (by simpa using hik), ← GF128.zero_eq]
      exact (Lagrange.eval_basis_of_ne hik (by simpa using hk')).symm
  | none =>
    have hne : ∀ j < n, f j ≠ z := by
      intro j hj h
      have := Array.findIdx?_eq_none_iff.mp hfind (f j) (by simp; exact ⟨j, hj, rfl⟩)
      simp [h] at this
    simp only
    rw [getElem!_pos _ i (by simpa using hi)]
    simp only [Array.getElem_map, Array.getElem_range]
    rw [Lagrange.eval_basis_not_at_node (by simpa using hi) (fun h => hne i hi (by exact h.symm)),
      Lagrange.eval_nodal, Lagrange.nodalWeight, of_mul, of_mul, of_inv, of_inv, of_foldl_node, ← htr i hi,
      ← Finset.prod_inv_distrib]
    simp only [nv, CharTwo.sub_eq_add, of_add]
    ring

theorem degree_lt_of_natDegree_lt {p : GF128[X]} {n : ℕ} (hp : p.natDegree < n) : p.degree < n := by
  rcases eq_or_ne p 0 with h | h
  · rw [h, degree_zero]; exact WithBot.bot_lt_coe n
  · exact (natDegree_lt_iff_degree_lt h).mp hp

theorem lagrange_interp (hinj : ∀ a < n, ∀ b < n, GF128.of (f a) = GF128.of (f b) → a = b)
    (htr : ∀ k < n, ∏ j ∈ (Finset.range n).erase k, (GF128.of (f k) + GF128.of (f j)) = GF128.of (den d))
    (p : GF128[X]) (hp : p.natDegree < n) (ζ : GF128) :
    ∑ i ∈ Finset.range n, GF128.of ((lagrange ((Array.range n).map f) (den d).inv ζ.val)[i]!) * p.eval (nv f i) =
      p.eval ζ := by
  have hinjOn : Set.InjOn (nv f) (Finset.range n : Set ℕ) := fun a ha b hb h =>
    hinj a (by simpa using ha) b (by simpa using hb) h
  have hdeg : p.degree < (Finset.range n).card := by rw [Finset.card_range]; exact degree_lt_of_natDegree_lt hp
  conv_rhs => rw [Lagrange.eq_interpolate hinjOn hdeg, Lagrange.interpolate_apply, eval_finset_sum]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [eval_mul, eval_C, lagrange_weight f n d hinj htr ζ.val i (Finset.mem_range.mp hi), mul_comm]; rfl

/-! ### `S` and `Λ` -/

theorem phi8_inj (a b : ℕ) (ha : a < 128) (hb : b < 128) (h : GF128.of (phi8 a) = GF128.of (phi8 b)) : a = b :=
  phi8_injOn a b (by omega) (by omega) h

/-- `L_s(ζ)`, the executable's skip weights. -/
noncomputable def lagS (ζ : GF128) (s : Fin 64) : GF128 := GF128.of ((skipWeights ζ.val)[s.val]!)
/-- `L^Λ_i(ζ)`, the executable's output-coset weights (the ones `Pc` uses). -/
noncomputable def lagΛ (ζ : GF128) (i : Fin 64) : GF128 := GF128.of ((lagrange nodesL den6Inv ζ.val)[i.val]!)

theorem lagS_interp (p : GF128[X]) (hp : p.natDegree < 64) (ζ : GF128) :
    ∑ s, lagS ζ s * p.eval (nodeS s) = p.eval ζ := by
  rw [← lagrange_interp phi8 64 6 (fun a ha b hb h => phi8_inj a b (by omega) (by omega) h)
    (fun k hk => prod_translate 6 (by norm_num) k (by norm_num; exact hk)) p hp ζ]
  exact Fin.sum_univ_eq_sum_range (fun i => GF128.of ((lagrange ((Array.range 64).map phi8) (den 6).inv ζ.val)[i]!) *
    p.eval (nv phi8 i)) 64

theorem lagΛ_interp (p : GF128[X]) (hp : p.natDegree < 64) (ζ : GF128) :
    ∑ i, lagΛ ζ i * p.eval (nodeΛ i) = p.eval ζ := by
  rw [← lagrange_interp (fun i => phi8 (64 + i)) 64 6
    (fun a ha b hb h => by have := phi8_inj _ _ (by omega) (by omega) h; omega)
    (fun k hk => prod_translate_coset k hk) p hp ζ]
  exact Fin.sum_univ_eq_sum_range (fun i => GF128.of ((lagrange ((Array.range 64).map fun i => phi8 (64 + i))
    (den 6).inv ζ.val)[i]!) * p.eval (nv (fun i => phi8 (64 + i)) i)) 64

/-! ### `Pcomb`'s weights -/

/-- The weights of `Pcomb(ζ)`, the executable's `combWeights`. -/
noncomputable def combW (ζ : GF128) (i : Fin 64) : GF128 := GF128.of ((combWeights ζ.val)[i.val]!)

theorem nodes_all : nodesS ++ nodesL = (Array.range 128).map phi8 := by
  apply Array.toList_inj.mp
  simp only [nodesS, nodesL, Array.toList_append, Array.toList_map, Array.toList_range]
  rw [show 128 = 64 + 64 from rfl, List.range_add, List.map_append, List.map_map]; rfl

theorem combW_interp (p : GF128[X]) (hp : p.natDegree < 128) (hS : ∀ s, p.eval (nodeS s) = 0) (ζ : GF128) :
    ∑ i, combW ζ i * p.eval (nodeΛ i) = p.eval ζ := by
  simp only [combW]
  unfold combWeights
  cases h1 : nodesS.findIdx? (fun x => decide (x = ζ.val)) with
  | some k =>
    obtain ⟨hk, hkz, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp h1
    have hk' : k < 64 := by simpa [nodesS] using hk
    have hz : phi8 k = ζ.val := by simpa [nodesS] using hkz
    have hζ : ζ = nodeS ⟨k, hk'⟩ := by rw [nodeS]; exact congrArg GF128.of hz.symm
    simp only
    rw [hζ, hS]
    refine Finset.sum_eq_zero fun i _ => ?_
    rw [getElem!_pos _ i.val (by simp), Array.getElem_map, ← GF128.zero_eq, zero_mul]
  | none =>
    have hSne : ∀ j < 64, phi8 j ≠ ζ.val := by
      intro j hj h
      have := Array.findIdx?_eq_none_iff.mp h1 (phi8 j) (by simp [nodesS]; exact ⟨j, hj, rfl⟩)
      simp [h] at this
    cases h2 : nodesL.findIdx? (fun x => decide (x = ζ.val)) with
    | some k =>
      obtain ⟨hk, hkz, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp h2
      have hk' : k < 64 := by simpa [nodesL] using hk
      have hz : phi8 (64 + k) = ζ.val := by simpa [nodesL] using hkz
      have hζ : ζ = nodeΛ ⟨k, hk'⟩ := by rw [nodeΛ]; exact congrArg GF128.of hz.symm
      simp only
      rw [Finset.sum_eq_single ⟨k, hk'⟩]
      · rw [getElem!_pos _ k (by simpa using hk'), Array.getElem_map, Array.getElem_range, beq_self_eq_true, if_pos rfl,
          ← GF128.one_eq, one_mul, hζ]
      · intro i _ hik
        rw [getElem!_pos _ i.val (by simp), Array.getElem_map, Array.getElem_range,
          if_neg (by simpa [Fin.ext_iff] using hik), ← GF128.zero_eq, zero_mul]
      · simp
    | none =>
      have hLne : ∀ j < 64, phi8 (64 + j) ≠ ζ.val := by
        intro j hj h
        have := Array.findIdx?_eq_none_iff.mp h2 (phi8 (64 + j)) (by simp [nodesL]; exact ⟨j, hj, rfl⟩)
        simp [h] at this
      have hne : ∀ t < 128, phi8 t ≠ ζ.val := by
        intro t ht
        by_cases h : t < 64
        · exact hSne t h
        · have := hLne (t - 64) (by omega); rwa [show 64 + (t - 64) = t by omega] at this
      simp only
      let v : ℕ → GF128 := nv phi8
      have hinjOn : Set.InjOn v (Finset.range 128 : Set ℕ) := fun a ha b hb h =>
        phi8_inj a b (by simpa using ha) (by simpa using hb) h
      have hdeg : p.degree < (Finset.range 128).card := by
        rw [Finset.card_range]; exact degree_lt_of_natDegree_lt hp
      conv_rhs => rw [Lagrange.eq_interpolate hinjOn hdeg, Lagrange.interpolate_apply, eval_finset_sum]
      rw [show (128 : ℕ) = 64 + 64 from rfl, Finset.sum_range_add]
      have hlow : ∑ t ∈ Finset.range 64, eval ζ (C (eval (v t) p) * Lagrange.basis (Finset.range (64 + 64)) v t) = 0 :=
        Finset.sum_eq_zero fun t ht => by
          rw [eval_mul, eval_C, show v t = nodeS ⟨t, Finset.mem_range.mp ht⟩ from rfl, hS, zero_mul]
      rw [hlow, zero_add, ← Fin.sum_univ_eq_sum_range (fun i => eval ζ (C (eval (v (64 + i)) p) *
        Lagrange.basis (Finset.range (64 + 64)) v (64 + i))) 64]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [eval_mul, eval_C, show v (64 + i.val) = nodeΛ i from rfl, mul_comm]
      congr 1
      rw [getElem!_pos _ i.val (by simp), Array.getElem_map, Array.getElem_range]
      have hi : 64 + i.val ∈ Finset.range (64 + 64) := Finset.mem_range.mpr (by omega)
      rw [Lagrange.eval_basis_not_at_node hi (fun h => hne (64 + i.val) (by omega) (congrArg GF128.val h.symm)),
        Lagrange.eval_nodal, Lagrange.nodalWeight, of_mul, of_mul, of_inv, den7Inv, of_inv, nodes_all,
        of_foldl_node phi8 128, nodesL, getElem!_range_map _ 64 i.val i.isLt, show (64 + 64 : ℕ) = 2 ^ 7 from rfl,
        ← prod_translate 7 (by norm_num) (64 + i.val) (by omega), ← Finset.prod_inv_distrib]
      have hζ' : GF128.of ζ.val = ζ := rfl
      simp only [v, nv, CharTwo.sub_eq_add, of_add, hζ']
      ring

/-- **`Arith.Correct.nodes`**, in the field's shape. -/
theorem nodes : ∃ nS nΛ : Fin 64 → GF128, Function.Injective nS ∧ Function.Injective nΛ ∧ (∀ s i, nS s ≠ nΛ i) ∧
    (∀ p : GF128[X], p.natDegree < 64 → ∀ ζ, ∑ s, lagS ζ s * p.eval (nS s) = p.eval ζ) ∧
    (∀ p : GF128[X], p.natDegree < 64 → ∀ ζ, ∑ i, lagΛ ζ i * p.eval (nΛ i) = p.eval ζ) ∧
    (∀ p : GF128[X], p.natDegree < 128 → (∀ s, p.eval (nS s) = 0) →
      ∀ ζ, ∑ i, combW ζ i * p.eval (nΛ i) = p.eval ζ) :=
  ⟨nodeS, nodeΛ, nodeS_injective, nodeΛ_injective, nodeS_ne_nodeΛ, lagS_interp, lagΛ_interp, combW_interp⟩

end FlockLevel3
