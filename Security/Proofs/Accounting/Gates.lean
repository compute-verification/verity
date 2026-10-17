import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Ring.GeomSum
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Int.Interval
import Mathlib.Data.List.GetD
import Mathlib.Tactic.NormNum.Basic
import Mathlib.Tactic.Ring
import Models.Accounting.Guarantees.Gates

/-!
# Lower bounds in the circuit model, proved

`Fooling` by induction over the gates (every free gate is 2-adically triangular in its window) and a count of the
charged gates' joint values; `SquaringCarry` from it at `j = 1` on the sparse family, whose squares have no carries
between bytes; `EncodeBudget` and `EncodeCost` from `EncodeChainHardness` by summing the census over the windows.
-/

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates Accounting.Assumptions

variable {ι : Type}

/-! ## Runs -/

lemma foldl_length (C : Circuit ι) (x : ι → ℤ) (vs : List ℤ) :
    (C.foldl (fun vs g => vs ++ [g.value x vs]) vs).length = vs.length + C.length := by
  induction C generalizing vs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

lemma run_length (C : Circuit ι) (x : ι → ℤ) : (run C x).length = C.length := by
  simp [run, foldl_length]

lemma run_take_succ (C : Circuit ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    run (C.take (i + 1)) x = run (C.take i) x ++ [C[i].value x (run (C.take i) x)] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [run, List.foldl_append, List.foldl_cons, List.foldl_nil]

lemma run_take_length (C : Circuit ι) (x : ι → ℤ) (n : ℕ) (h : n ≤ C.length) : (run (C.take n) x).length = n := by
  rw [run_length, List.length_take]; omega

lemma getD_run_take (C : Circuit ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    ∀ n, i < n → n ≤ C.length → (run (C.take n) x).getD i 0 = C[i].value x (run (C.take i) x) := by
  intro n hin hn
  induction n with
  | zero => omega
  | succ n ih =>
    rw [run_take_succ C x n (by omega)]
    rcases Nat.lt_or_ge i n with hi | hi
    · rw [List.getD_append _ _ _ _ (by rw [run_take_length C x n (by omega)]; exact hi)]
      exact ih hi (by omega)
    · have : i = n := by omega
      subst this
      rw [List.getD_append_right _ _ _ _ (by rw [run_take_length C x i (by omega)]), run_take_length C x i (by omega)]
      simp

lemma val_eq (C : Circuit ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    val C x i = C[i].value x (run (C.take i) x) := by
  have := getD_run_take C x i h C.length h le_rfl
  rwa [List.take_length] at this

/-! ## Free gates are 2-adically triangular -/

lemma sum_zipWith_const_congr {m : ℤ} (F F' : ℕ → ℤ) (hF : ∀ j, F j ≡ F' j [ZMOD m]) :
    ∀ (cs : List ℤ) (u : List ℕ),
      (List.zipWith (· * ·) cs (u.map F)).sum ≡ (List.zipWith (· * ·) cs (u.map F')).sum [ZMOD m]
  | [], _ => by simp [Int.ModEq.refl]
  | _ :: _, [] => by simp [Int.ModEq.refl]
  | c :: cs, j :: u => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    exact ((hF j).mul_left c).add (sum_zipWith_const_congr F F' hF cs u)

lemma sum_zipWith_congr {m : ℤ} (F F' : ℕ → ℤ) (hF : ∀ j, F j ≡ F' j [ZMOD m]) :
    ∀ (u v : List ℕ),
      (List.zipWith (· * ·) (u.map F) (v.map F)).sum ≡ (List.zipWith (· * ·) (u.map F') (v.map F')).sum [ZMOD m]
  | [], _ => by simp [Int.ModEq.refl]
  | _ :: _, [] => by simp [Int.ModEq.refl]
  | i :: u, j :: v => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    exact ((hF i).mul (hF j)).add (sum_zipWith_congr F F' hF u v)

lemma clampS32_of {z : ℤ} (h1 : s32Min ≤ z) (h2 : z ≤ s32Max) : clampS32 z = z := by
  unfold clampS32; rw [min_eq_right h2, max_eq_right h1]

lemma wrapS32_of {z : ℤ} (h1 : s32Min ≤ z) (h2 : z ≤ s32Max) : wrapS32 z = z := by
  unfold wrapS32
  have hmin : s32Min = -2147483648 := by norm_num [s32Min]
  have hmax : s32Max = 2147483647 := by norm_num [s32Max]
  rw [Int.emod_eq_of_lt (by omega) (by rw [hmin] at h1 ⊢; rw [hmax] at h2; norm_num; omega)]
  omega

/-- A gate's value is its exact value in its window. -/
lemma value_eq_raw (g : Gate ι) (x : ι → ℤ) (vs : List ℤ) (hg : g.Ok x vs) : g.value x vs = g.raw x vs := by
  obtain ⟨h1, h2, -, h4⟩ := hg
  unfold Gate.value
  cases hop : g.op with
  | mma a b k sat =>
    have hb := h4 (by rw [hop]; trivial)
    cases sat
    · simp only [Op.out]; exact wrapS32_of (by omega) (by omega)
    · simp only [Op.out]; exact clampS32_of (by omega) (by omega)
  | _ => simp [Op.out]

/-- **The triangular step**: an uncharged gate in its window, on operands that agree mod `m`, takes values that agree
mod `m`, given that a registered input it reads agrees mod `m`. -/
lemma value_congr {m : ℤ} (g : Gate ι) (x x' : ι → ℤ) (vs vs' : List ℤ)
    (hv : ∀ j, vs.getD j 0 ≡ vs'.getD j 0 [ZMOD m]) (hc : g.op.charged = false)
    (hx : ∀ i, g.op = .input .registered i → x i ≡ x' i [ZMOD m]) (hg : g.Ok x vs) (hg' : g.Ok x' vs') :
    g.value x vs ≡ g.value x' vs' [ZMOD m] := by
  rw [value_eq_raw g x vs hg, value_eq_raw g x' vs' hg']
  unfold Gate.raw Gate.operands
  cases hop : g.op with
  | input b i =>
    cases b
    · exact hx i hop
    · rw [hop] at hc; simp [Op.charged] at hc
  | const c => exact Int.ModEq.refl _
  | affine c₀ cs =>
    simp only [Op.raw]
    exact (Int.ModEq.refl c₀).add (sum_zipWith_const_congr _ _ hv cs g.args)
  | mma a b k sat =>
    simp only [Op.raw, dot]
    generalize g.args = u
    cases u with
    | nil => simp [Int.ModEq.refl]
    | cons j u =>
      simp only [List.map_cons, List.headD_cons, List.tail_cons, ← List.map_take, ← List.map_drop]
      exact (hv j).add (sum_zipWith_congr _ _ hv _ _)
  | gather => rw [hop] at hc; simp [Op.charged] at hc
  | budgeted cls f => rw [hop] at hc; simp [Op.charged] at hc

/-- Two inputs on which `C` is in its window, that agree mod `m` in every registered input, and on which every
charged gate takes one value: every gate's values agree mod `m`. -/
lemma run_congr {m : ℤ} (C : Circuit ι) (x x' : ι → ℤ) (hw : InWindow C x) (hw' : InWindow C x')
    (hreg : ∀ i, (∃ g ∈ C, g.op = .input .registered i) → x i ≡ x' i [ZMOD m])
    (hch : ∀ i (h : i < C.length), C[i].op.charged = true → val C x i = val C x' i) :
    ∀ n, n ≤ C.length → ∀ o, (run (C.take n) x).getD o 0 ≡ (run (C.take n) x').getD o 0 [ZMOD m] := by
  intro n
  induction n with
  | zero => intro _ o; simp [run, Int.ModEq.refl]
  | succ n ih =>
    intro hn o
    have hn' : n < C.length := hn
    have hlen := run_take_length C x n (by omega)
    have hlen' := run_take_length C x' n (by omega)
    rw [run_take_succ C x n hn', run_take_succ C x' n hn']
    rcases Nat.lt_trichotomy o n with ho | ho | ho
    · rw [List.getD_append _ _ _ _ (by rw [hlen]; exact ho), List.getD_append _ _ _ _ (by rw [hlen']; exact ho)]
      exact ih (by omega) o
    · subst ho
      rw [List.getD_append_right _ _ _ _ (by rw [hlen]), List.getD_append_right _ _ _ _ (by rw [hlen']), hlen,
        hlen', Nat.sub_self, List.getD_cons_zero, List.getD_cons_zero]
      cases hc : C[o].op.charged
      · exact value_congr _ x x' _ _ (ih (by omega)) hc
          (fun i hi => hreg i ⟨C[o], List.getElem_mem hn', hi⟩) (hw o hn') (hw' o hn')
      · rw [← val_eq C x o hn', ← val_eq C x' o hn', hch o hn' hc]
    · rw [List.getD_append_right _ _ _ _ (by rw [hlen]; omega), List.getD_append_right _ _ _ _ (by rw [hlen']; omega),
        hlen, hlen']
      obtain ⟨k, hk⟩ : ∃ k, o - n = k + 1 := ⟨o - n - 1, by omega⟩
      rw [hk]
      simp [Int.ModEq.refl]

/-! ## The fooling bound -/

theorem Fooling : Accounting.Guarantees.Gates.Fooling := by
  intro ι κ C F x j out hw hreg hdist
  classical
  let T : κ → Fin C.length → ℤ := fun s i => if C[i.1].op.charged then val C (x s) i.1 else 0
  let box : Finset (Fin C.length → ℤ) :=
    Fintype.piFinset fun i => if C[i.1].op.charged then Finset.Icc C[i.1].lo C[i.1].hi else {0}
  have hmaps : Set.MapsTo T F box := by
    intro s hs
    simp only [Finset.mem_coe] at hs
    simp only [box, Finset.mem_coe, Fintype.mem_piFinset]
    intro i
    by_cases hc : C[i.1].op.charged
    · simp only [T, hc, ite_true]
      have hok := hw s hs i.1 i.2
      rw [Finset.mem_Icc, val_eq C (x s) i.1 i.2, value_eq_raw _ _ _ hok]
      exact ⟨hok.1, hok.2.1⟩
    · simp [T, hc]
  have hinj : Set.InjOn T F := by
    intro s hs s' hs' hT
    simp only [Finset.mem_coe] at hs hs'
    by_contra hne
    obtain ⟨o, -, ho⟩ := hdist s hs s' hs' hne
    apply ho
    have h := run_congr C (x s) (x s') (hw s hs) (hw s' hs') (hreg s hs s' hs')
      (fun i h hc => by simpa [T, hc] using congrFun hT ⟨i, h⟩) C.length le_rfl o
    rwa [List.take_length] at h
  calc F.card ≤ box.card := Finset.card_le_card_of_injOn T hmaps hinj
    _ = volume C := by
      rw [Fintype.card_piFinset, volume, ← Fin.prod_univ_fun_getElem C Gate.charge]
      apply Finset.prod_congr rfl
      intro i _
      unfold Gate.charge Gate.size
      split_ifs <;> simp [Int.card_Icc]

/-! ## The carry bound -/

/-- Byte `l` of a base-256 numeral whose digits are `f`. -/
lemma byte_sum : ∀ (M : ℕ) (f : ℕ → ℕ), (∀ m, f m < 256) → ∀ l,
    byte (∑ m ∈ Finset.range M, f m * 256 ^ m) l = if l < M then f l else 0
  | 0, f, _, l => by simp [byte]
  | M + 1, f, hf, l => by
    have hR : ∑ m ∈ Finset.range (M + 1), f m * 256 ^ m =
        f 0 + 256 * ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m := by
      rw [Finset.sum_range_succ', Finset.mul_sum, add_comm]
      congr 1
      · simp
      · apply Finset.sum_congr rfl; intro m _; ring
    rw [hR]
    have h0 := hf 0
    cases l with
    | zero =>
      simp only [byte, pow_zero, Nat.div_one]
      rw [ite_eq_left (Nat.succ_pos M)]
      omega
    | succ l =>
      have ih := byte_sum M (fun m => f (m + 1)) (fun m => hf _) l
      unfold byte at ih ⊢
      rw [pow_succ', ← Nat.div_div_eq_div_mul,
        show (f 0 + 256 * ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m) / 256 =
          ∑ m ∈ Finset.range M, f (m + 1) * 256 ^ m by omega, ih]
      simp

lemma byte_mul_256 (N l : ℕ) : byte (256 * N) (l + 1) = byte N l := by
  unfold byte
  rw [pow_succ', Nat.mul_div_mul_left _ _ (by norm_num : 0 < 256)]

/-- `ν_m`: the pairs of `S` that sum to `m`, the coefficient of `256^m` in `(Σ_{i ∈ S} 256^i)²`. -/
def nu (S : Finset ℕ) (m : ℕ) : ℕ := ((S ×ˢ S).filter fun q => q.1 + q.2 = m).card

lemma nu_le (S : Finset ℕ) (m : ℕ) : nu S m ≤ S.card := by
  unfold nu
  apply Finset.card_le_card_of_injOn Prod.fst
  · intro q hq
    simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq] at hq
    exact hq.1.1
  · intro q hq q' hq' h
    simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq] at hq hq'
    exact Prod.ext h (by omega)

lemma nu_two_mul_mod (S : Finset ℕ) (i : ℕ) : nu S (2 * i) % 2 = if i ∈ S then 1 else 0 := by
  have hsplit : nu S (2 * i) =
      ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 < q.2).card +
        ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.2 < q.1).card +
        ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 = q.2).card := by
    unfold nu
    simp only [Finset.card_filter, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro q _
    split_ifs <;> omega
  have hswap : ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 < q.2).card =
      ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.2 < q.1).card := by
    apply Finset.card_nbij' Prod.swap Prod.swap
    · intro q hq
      simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq, Prod.fst_swap, Prod.snd_swap] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, by omega, hq.2.2⟩
    · intro q hq
      simp only [Finset.coe_filter, Finset.mem_product, Set.mem_ofPred_eq, Prod.fst_swap, Prod.snd_swap] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, by omega, hq.2.2⟩
    · intro q _; exact Prod.swap_swap q
    · intro q _; exact Prod.swap_swap q
  have hdiag : ((S ×ˢ S).filter fun q => q.1 + q.2 = 2 * i ∧ q.1 = q.2).card = if i ∈ S then 1 else 0 := by
    by_cases hi : i ∈ S
    · rw [ite_eq_left hi, Finset.card_eq_one]
      refine ⟨(i, i), ?_⟩
      ext ⟨a, b⟩
      simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_singleton, Prod.mk.injEq]
      constructor
      · rintro ⟨-, h1, h2⟩; omega
      · rintro ⟨rfl, rfl⟩; exact ⟨⟨hi, hi⟩, by omega, rfl⟩
    · rw [ite_eq_right hi, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      rintro ⟨a, b⟩ hq ⟨h1, h2⟩
      rw [Finset.mem_product] at hq
      have : a = i := by simp only at h1 h2; omega
      exact hi (this ▸ hq.1)
  rw [hsplit, hswap, hdiag]
  split_ifs <;> omega

lemma sq_sum (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) :
    (∑ i ∈ S, 256 ^ i) ^ 2 = ∑ m ∈ Finset.range (2 * s), nu S m * 256 ^ m := by
  have hmaps : ∀ q ∈ S ×ˢ S, (fun q : ℕ × ℕ => q.1 + q.2) q ∈ Finset.range (2 * s) := by
    intro q hq
    rw [Finset.mem_product] at hq
    have h1 := Finset.mem_range.mp (hS hq.1)
    have h2 := Finset.mem_range.mp (hS hq.2)
    exact Finset.mem_range.mpr (by dsimp only; omega)
  rw [sq, Finset.sum_mul_sum, ← Finset.sum_product' (f := fun a b => 256 ^ a * 256 ^ b),
    ← Finset.sum_fiberwise_of_maps_to hmaps]
  apply Finset.sum_congr rfl
  intro m _
  rw [nu, Finset.card_eq_sum_ones, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro q hq
  rw [Finset.mem_filter] at hq
  rw [← pow_add, hq.2, one_mul]

lemma sum_pow_lt (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) : ∑ i ∈ S, 256 ^ i < 256 ^ s :=
  Nat.geomSum_lt (by norm_num) fun k hk => Finset.mem_range.mp (hS hk)

lemma sparse_sq_lt (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) : sparse S ^ 2 < 256 ^ (2 * s + 1) := by
  have h := sum_pow_lt S s hS
  have : sparse S < 16 * 256 ^ s := by unfold sparse; omega
  calc sparse S ^ 2 < (16 * 256 ^ s) ^ 2 := Nat.pow_lt_pow_left this (by norm_num)
    _ = 256 ^ (2 * s + 1) := by ring

lemma sparse_sq (S : Finset ℕ) : sparse S ^ 2 = 256 * (∑ i ∈ S, 256 ^ i) ^ 2 := by unfold sparse; ring

/-- Byte `2i + 1` of `a_S²` is `ν_{2i}`, odd exactly when `i ∈ S`. -/
lemma byte_sparse_sq (S : Finset ℕ) (s i : ℕ) (hs : s ≤ 255) (hS : S ⊆ Finset.range s) (hi : i < s) :
    byte (sparse S ^ 2) (2 * i + 1) % 2 = if i ∈ S then 1 else 0 := by
  have hcard : S.card ≤ s := by simpa using Finset.card_le_card hS
  rw [sparse_sq, byte_mul_256, sq_sum S s hS,
    byte_sum (2 * s) (nu S) (fun m => by have := nu_le S m; omega), ite_eq_left (by omega)]
  exact nu_two_mul_mod S i

lemma sparse_eq (S : Finset ℕ) (s : ℕ) (hS : S ⊆ Finset.range s) :
    sparse S = ∑ m ∈ Finset.range s, (if m ∈ S then 16 else 0) * 256 ^ m := by
  unfold sparse
  rw [Finset.mul_sum]
  simp only [ite_mul, zero_mul]
  rw [Finset.sum_ite_mem, Finset.inter_eq_right.mpr hS]

/-- Every byte of `a_S` is even. -/
lemma byte_sparse_even (S : Finset ℕ) (s l : ℕ) (hS : S ⊆ Finset.range s) : byte (sparse S) l % 2 = 0 := by
  rw [sparse_eq S s hS, byte_sum s (fun m => if m ∈ S then 16 else 0) (fun m => by split_ifs <;> norm_num)]
  split_ifs <;> norm_num

theorem SquaringCarry : Accounting.Guarantees.Gates.SquaringCarry := by
  intro ι C p n s G x slot out hs hp hn hG hreg hout
  have hsub : ∀ S ∈ G, S ⊆ Finset.range s := fun S hS => Finset.mem_powerset.mp (hG hS)
  have hmod : ∀ S ∈ G, sparse S ^ 2 % p = sparse S ^ 2 := fun S hS =>
    Nat.mod_eq_of_lt (lt_of_lt_of_le (sparse_sq_lt S s (hsub S hS)) hp)
  refine Fooling C G x 1 ((List.range n).map out) (fun S hS => (hout S hS).1) ?_ ?_
  · intro S hS S' hS' i hi
    rw [hreg S hS i hi, hreg S' hS' i hi, pow_one]
    have h1 := byte_sparse_even S s (slot i) (hsub S hS)
    have h2 := byte_sparse_even S' s (slot i) (hsub S' hS')
    unfold Int.ModEq
    omega
  · intro S hS S' hS' hne
    obtain ⟨i, hi⟩ : ∃ i, ¬ (i ∈ S ↔ i ∈ S') := by
      by_contra h
      exact hne (Finset.ext fun a => by by_contra ha; exact h ⟨a, ha⟩)
    have his : i < s := by
      by_cases h : i ∈ S
      · exact Finset.mem_range.mp (hsub S hS h)
      · have h' : i ∈ S' := by
          by_contra h'
          exact hi ⟨fun h1 => absurd h1 h, fun h1 => absurd h1 h'⟩
        exact Finset.mem_range.mp (hsub S' hS' h')
    refine ⟨out (2 * i + 1), List.mem_map.mpr ⟨2 * i + 1, List.mem_range.mpr (by omega), rfl⟩, ?_⟩
    rw [(hout S hS).2 _ (by omega), (hout S' hS').2 _ (by omega), hmod S hS, hmod S' hS', pow_one]
    have h1 := byte_sparse_sq S s i hs (hsub S hS) his
    have h2 := byte_sparse_sq S' s i hs (hsub S' hS') his
    unfold Int.ModEq
    by_cases h : i ∈ S
    · have h' : i ∉ S' := fun h' => hi ⟨fun _ => h', fun _ => h⟩
      rw [ite_eq_left h] at h1
      rw [ite_eq_right h'] at h2
      omega
    · have h' : i ∈ S' := by
        by_contra h'
        exact hi ⟨fun h1 => absurd h1 h, fun h1 => absurd h1 h'⟩
      rw [ite_eq_right h] at h1
      rw [ite_eq_left h'] at h2
      omega

/-! ## The encode budget -/

lemma countAt_cons (g : Gate ι) (C : Circuit ι) (s : ℕ → Prop) [DecidablePred s] (p : Gate ι → ℕ) :
    countAt (g :: C) s p = (if s g.step then p g else 0) + countAt C s p := by
  unfold countAt
  by_cases hs : s g.step <;> simp [hs]

/-- Summing a gate count over the windows: every gate is in one of them. -/
lemma sum_countAt (C : Circuit ι) (win : ℕ → ℕ) (N : ℕ) (p : Gate ι → ℕ) (h : ∀ g ∈ C, win g.step < N) :
    ∑ v ∈ Finset.range N, countAt C (fun t => win t = v) p = (C.map p).sum := by
  induction C with
  | nil => simp [countAt]
  | cons g C ih =>
    have hg := h g (List.mem_cons_self ..)
    have ih := ih fun g' hg' => h g' (List.mem_cons_of_mem _ hg')
    simp only [countAt_cons, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_range, hg, ite_true, ih,
      List.map_cons, List.sum_cons]

lemma budgeted_eq (C : Circuit ι) (A : Audit) (h : ∀ g ∈ C, A.win g.step < A.windows) :
    budgeted C = ∑ v ∈ Finset.range A.windows, (census C fun t => A.win t = v).budgeted := by
  have key : ∀ g : Gate ι, (if g.op.budget.isSome then 1 else 0) =
      g.isBudget .rounding + g.isBudget .compareSelect + g.isBudget .gather + g.isBudget .product := by
    intro g
    unfold Gate.isBudget
    rcases g.op.budget with _ | b
    · simp
    · cases b <;> simp
  unfold budgeted
  simp only [census, Census.budgeted, Finset.sum_add_distrib, sum_countAt C A.win A.windows _ h,
    ← List.sum_map_add]
  congr 1
  exact List.map_congr_left fun g _ => key g

theorem EncodeBudget : Accounting.Guarantees.Gates.EncodeBudget := by
  intro F k f w E εnum εden ι W caps A m x words out H hP hR hci hnd hlt hcw
  have h := H W m x words out hR hci hnd hlt hcw
  obtain ⟨hwin, hcap, -⟩ := hP
  have hb : budgeted W ≤ caps.total * ∑ v ∈ Finset.range A.windows, A.reads v := by
    rw [budgeted_eq W A hwin, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro v hv
    obtain ⟨h1, h2, h3, h4, -⟩ := hcap v (Finset.mem_range.mp hv)
    simp only [Census.budgeted, Caps.total, add_mul]
    omega
  have hr : runBits W ≤ 8 * caps.runBytes * A.windows := by
    rw [runBits, ← sum_countAt W A.win A.windows _ hwin]
    calc _ ≤ ∑ v ∈ Finset.range A.windows, 8 * caps.runBytes :=
          Finset.sum_le_sum fun v hv => (hcap v (Finset.mem_range.mp hv)).2.2.2.2
      _ = 8 * caps.runBytes * A.windows := by simp [mul_comm]
  calc words.length * E * (F.n - 1) ≤ budgeted W * (F.n - 1) + runBits W * E := h
    _ ≤ _ := add_le_add (Nat.mul_le_mul_right _ hb) (Nat.mul_le_mul_right _ hr)

theorem EncodeCost : Accounting.Guarantees.Gates.EncodeCost := by
  intro F k f w E εnum εden q H hn hne
  unfold encodeCost
  obtain ⟨ι, C, m, x, words, out, hlen, hnd, hlt, hR, hrb, hci, hcw, hc⟩ := Nat.sInf_mem hne
  have h := H C m x words out hR hci hnd hlt hcw
  rw [hrb, zero_mul, add_zero, hlen, hc] at h
  exact Nat.le_of_mul_le_mul_right h (by omega)

end Accounting.SecurityProofs.Gates
