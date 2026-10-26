import Definitions.Pouw.H1Zy.Tile
import Proofs.Pouw.H1Zy.Lane

/-!
# What the tile's `ok = 1` gives, on words

`Pouw.H1Zy.tileOk`, `PouwH1TileZy_v1{TILE}`'s `ok` on words, is `1` only where each cell's record z is the exact dot of
its A row's and B row's codes as an s32 word and its y the dequant of that z at the rows' scales (`tileOk_spec`), at a
`K` the tile's EXACT binds at (`32 ∣ K`, `K < 2^17`), for TILE either form (`TileForm.z_eq`):

* v1's z is EXACT, which never clamps there (`gemmAccS8_eq`);
* v3's z is the exact dot at any noise (`streamZ_eq`). Every word is the integer it names mod `2^32`, so the stream's
  `X W′`, less `Σ_d 128^d D_d`, plus the wrap correction, less the dither peel, is `Σ_k A_k B_k` entry by entry
  (`X = A + b - 256 xa`, `W′ = B + c + U - 256 wb⁺ + 256 wb⁻`), once the digits recombine to each residue sum exactly:
  at `K / 32 ≤ 4095` segments every rest is within s32 and the top digit within int8 (`digits_val`).
-/

namespace Pouw.H1Zy

/-! ### Words mod `2^32` as integers -/

theorem ofInt_sub32 (x y : ℤ) : BitVec.ofInt 32 (x - y) = BitVec.ofInt 32 x - BitVec.ofInt 32 y := by
  rw [sub_eq_add_neg, BitVec.ofInt_add, BitVec.ofInt_neg, ← sub_eq_add_neg]

theorem ofInt_sum32 {ι : Type} (s : Finset ι) (f : ι → ℤ) :
    BitVec.ofInt 32 (∑ i ∈ s, f i) = ∑ i ∈ s, BitVec.ofInt 32 (f i) :=
  map_sum (Int.castRingHom (BitVec 32)) f s

theorem ofInt_shiftLeft32 (v : ℤ) (n : ℕ) : BitVec.ofInt 32 v <<< n = BitVec.ofInt 32 (v * 2 ^ n) := by
  rw [BitVec.shiftLeft_eq_mul_twoPow, BitVec.ofInt_mul]
  congr 1
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_twoPow, show ((2 : ℤ) ^ n) = ((2 ^ n : ℕ) : ℤ) by push_cast; rfl, BitVec.ofInt_natCast,
    BitVec.toNat_ofNat]

/-- A fold whose every step adds a word adds their sum. -/
theorem segFold_add (f : ℕ → BitVec 32 → BitVec 32) (g : ℕ → ℤ)
    (hf : ∀ s acc, f s acc = acc + BitVec.ofInt 32 (g s)) (init : BitVec 32) :
    ∀ n, segFold f init n = init + BitVec.ofInt 32 (∑ s ∈ Finset.range n, g s)
  | 0 => by simp [segFold]
  | n + 1 => by
      rw [segFold, hf, segFold_add f g hf init n, Finset.sum_range_succ, BitVec.ofInt_add, add_assoc]

/-- A fold whose every step subtracts a word subtracts their sum. -/
theorem segFold_sub (f : ℕ → BitVec 32 → BitVec 32) (g : ℕ → ℤ)
    (hf : ∀ s acc, f s acc = acc - BitVec.ofInt 32 (g s)) (init : BitVec 32) (n : ℕ) :
    segFold f init n = init - BitVec.ofInt 32 (∑ s ∈ Finset.range n, g s) := by
  rw [segFold_add f (fun s => -g s) (fun s acc => by rw [hf, BitVec.ofInt_neg, sub_eq_add_neg]) init n,
    Finset.sum_neg_distrib, BitVec.ofInt_neg, ← sub_eq_add_neg]

/-- A sum over `n` segments of 32 entries is the sum over their `32 n` entries. -/
theorem sum_segs (g : ℕ → ℤ) : ∀ n,
    ∑ s ∈ Finset.range n, ∑ e : Fin 32, g (32 * s + e.val) = ∑ k ∈ Finset.range (32 * n), g k
  | 0 => by simp
  | n + 1 => by
      rw [Finset.sum_range_succ, sum_segs g n, show 32 * (n + 1) = 32 * n + 32 by ring, Finset.sum_range_add,
        Fin.sum_univ_eq_sum_range (fun e => g (32 * n + e))]

/-! ### int8 entries -/

theorem toInt8_bounds (a : BitVec 8) : -128 ≤ a.toInt ∧ a.toInt ≤ 127 := by
  have h1 := BitVec.le_toInt a
  have h2 := BitVec.toInt_lt (x := a)
  norm_num at h1 h2
  omega

theorem bmod256_of (v : ℤ) (h1 : -128 ≤ v) (h2 : v ≤ 127) : v.bmod (2 ^ 8) = v :=
  Int.bmod_eq_of_le (by norm_num; omega) (by norm_num; omega)

/-- `X = A + b` as an int8 is `A + b`, less 256 where it wraps (`PouwXWrap`). -/
theorem i8AddBit_toInt (a : BitVec 8) (d : Bool) :
    (i8AddBit a d).toInt = a.toInt + (if d then 1 else 0) - 256 * (if xWrap a d then 1 else 0) := by
  have ha := toInt8_bounds a
  cases d
  · simp [i8AddBit, xWrap]
  · simp only [i8AddBit, xWrap, if_true, Bool.and_true]
    rw [BitVec.toInt_add, show (1 : BitVec 8).toInt = 1 by decide]
    by_cases h : a = 127
    · subst h
      decide
    · have h' : a.toInt ≠ 127 := fun e => h (BitVec.toInt_inj.1 (e.trans (by decide)))
      rw [bmod256_of _ (by omega) (by omega), if_neg (by simpa using h)]
      ring

/-- `PouwU129`'s word is within `±64`. -/
theorem u129_toInt (x : BitVec 8) : (u129 x).toInt = -64 + ((x.toNat % 129 : ℕ) : ℤ) := by
  unfold u129
  rw [BitVec.toInt_ofInt]
  have : ((x.toNat % 129 : ℕ) : ℤ) < 129 := by exact_mod_cast Nat.mod_lt _ (by norm_num)
  exact bmod256_of _ (by omega) (by omega)

theorem u129_bounds (x : BitVec 8) : -64 ≤ (u129 x).toInt ∧ (u129 x).toInt ≤ 64 := by
  rw [u129_toInt]
  have : ((x.toNat % 129 : ℕ) : ℤ) < 129 := by exact_mod_cast Nat.mod_lt _ (by norm_num)
  omega

theorem i8Neg_toInt (u : BitVec 8) (h1 : -64 ≤ u.toInt) (h2 : u.toInt ≤ 64) : (i8Neg u).toInt = -u.toInt := by
  unfold i8Neg
  rw [BitVec.toInt_neg]
  exact bmod256_of _ (by omega) (by omega)

theorem bmod256_wrap (s : ℤ) (h1 : -384 ≤ s) (h2 : s ≤ 383) :
    s.bmod (2 ^ 8) = s - 256 * (if 127 < s then 1 else 0) + 256 * (if s < -128 then 1 else 0) := by
  rw [Int.bmod_def]
  norm_num
  split_ifs <;> omega

/-- `W′ = B + c + U` as an int8 is `B + c + U`, less 256 at `wb⁺` and plus 256 at `wb⁻` (`PouwWrapBits`). -/
theorem i8Add_toInt (b : BitVec 8) (c : Bool) (u : BitVec 8) (h1 : -64 ≤ u.toInt) (h2 : u.toInt ≤ 64) :
    (i8Add (i8AddBit b c) u).toInt = b.toInt + (if c then 1 else 0) + u.toInt -
      256 * (if (wrapBits b c u).1 then 1 else 0) + 256 * (if (wrapBits b c u).2 then 1 else 0) := by
  have hb := toInt8_bounds b
  have e : (i8Add (i8AddBit b c) u).toInt = (b.toInt + (if c then 1 else 0) + u.toInt).bmod (2 ^ 8) := by
    unfold i8Add i8AddBit
    rw [BitVec.toInt_add, BitVec.toInt_add, Int.bmod_add_bmod]
    cases c <;> simp
  have hc : (0 : ℤ) ≤ (if c then 1 else 0) ∧ (if c then 1 else 0 : ℤ) ≤ 1 := by cases c <;> simp
  rw [e, bmod256_wrap _ (by omega) (by omega)]
  simp only [wrapBits, decide_eq_true_eq]

/-! ### The digits -/

theorem toInt32_bounds (r : BitVec 32) : -2 ^ 31 ≤ r.toInt ∧ r.toInt < 2 ^ 31 := by
  have h1 := BitVec.le_toInt r
  have h2 := BitVec.toInt_lt (x := r)
  norm_num at h1 h2 ⊢
  omega

theorem toNat_emod128 (r : BitVec 32) : (r.toNat : ℤ) % 128 = r.toInt % 128 := by
  rw [BitVec.toInt_eq_toNat_cond]
  split <;> omega

theorem digitLo_toInt (r : BitVec 32) (z : Bool) :
    (digitLo r z).toInt = r.toInt % 128 - 128 * (if z then 1 else 0) := by
  unfold digitLo
  rw [BitVec.toInt_ofNat', ← toNat_emod128]
  have hn : r.toNat % 128 < 128 := Nat.mod_lt _ (by norm_num)
  rw [Int.bmod_def]
  cases z <;> norm_num <;> split_ifs <;> omega

theorem sshiftRight7_toInt (r : BitVec 32) : (r.sshiftRight 7).toInt = r.toInt / 128 := by
  rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
  norm_num

theorem digitHi_toInt (r : BitVec 32) (z : Bool) :
    (digitHi r z).toInt = r.toInt / 128 + (if z then 1 else 0) := by
  unfold digitHi
  rw [BitVec.toInt_add, sshiftRight7_toInt]
  have hb := toInt32_bounds r
  have e1 : (1 : BitVec 32).toInt = 1 := by decide
  cases z <;> simp only [e1, Bool.false_eq_true, if_false, if_true] <;>
    exact Int.bmod_eq_of_le (by norm_num; omega) (by norm_num; omega)

theorem digitZ_eq (r : BitVec 32) (z : Bool) : digitZ r z = if r.toInt / 128 = 127 then false else z := by
  unfold digitZ
  by_cases h : r.toInt / 128 = 127
  · have : r.sshiftRight 7 = 127 := BitVec.toInt_inj.1 ((sshiftRight7_toInt r).trans (h.trans (by decide)))
    simp [h, this]
  · have : r.sshiftRight 7 ≠ 127 := fun e => h ((sshiftRight7_toInt r).symm.trans (by rw [e]; decide))
    rw [if_neg this, if_neg h]

theorem low8_toInt (r : BitVec 32) (h1 : -128 ≤ r.toInt) (h2 : r.toInt ≤ 127) : (low8 r).toInt = r.toInt := by
  unfold low8
  rw [BitVec.toInt_setWidth]
  have hc := BitVec.toInt_eq_toNat_cond r
  rw [Int.bmod_def]
  norm_num
  split at hc <;> split_ifs <;> omega

theorem nDigits_one {ks : ℕ} (h : ks ≤ 1) : nDigits ks = 1 := by
  rcases (show ks = 0 ∨ ks = 1 by omega) with rfl | rfl <;> rfl

theorem nDigits_two {ks : ℕ} (h1 : 2 ≤ ks) (h2 : ks ≤ 128) : nDigits ks = 2 := by
  obtain ⟨f, rfl⟩ : ∃ f, ks = f + 1 + 1 := ⟨ks - 2, by omega⟩
  simp only [nDigits, nDigits.go]
  rw [if_neg (by omega), if_pos (by omega)]

theorem nDigits_three {ks : ℕ} (h1 : 129 ≤ ks) (h2 : ks ≤ 16384) : nDigits ks = 3 := by
  obtain ⟨f, rfl⟩ : ∃ f, ks = f + 1 + 1 + 1 := ⟨ks - 3, by omega⟩
  simp only [nDigits, nDigits.go]
  rw [if_neg (by omega), if_neg (by omega), if_pos (by omega)]

theorem b2i_bounds (z : Bool) : (0 : ℤ) ≤ (if z then 1 else 0) ∧ (if z then 1 else 0 : ℤ) ≤ 1 := by
  cases z <;> simp

/-- **The digits are exact**: at `KS ≤ 4095` segments (`K < 2^17`), a residue sum `r` of `KS` int8s has its `m` digits'
values sum, at powers of 128, to `r`: every rest is within s32 and the top digit within int8, so `PouwLow8` keeps it. -/
theorem digits_val (z : ℕ → Bool) (r : BitVec 32) (ks : ℕ) (hks : ks ≤ 4095)
    (h1 : -(128 * (ks : ℤ)) ≤ r.toInt) (h2 : r.toInt ≤ 127 * (ks : ℤ)) :
    ∑ d ∈ Finset.range (nDigits ks), 128 ^ d * ((digits (nDigits ks) z r).getD d 0).toInt = r.toInt := by
  rcases (show ks ≤ 1 ∨ (2 ≤ ks ∧ ks ≤ 128) ∨ (129 ≤ ks ∧ ks ≤ 4095) by omega) with h | ⟨ha, hb⟩ | ⟨ha, hb⟩
  · have hk : (ks : ℤ) ≤ 1 := by exact_mod_cast h
    rw [nDigits_one h, show digits 1 z r = digitsFrom z 0 0 r from rfl]
    simp only [digitsFrom, Finset.sum_range_one, pow_zero, one_mul, List.getD_cons_zero]
    exact low8_toInt r (by nlinarith) (by nlinarith)
  · have hk : (ks : ℤ) ≤ 128 := by exact_mod_cast hb
    rw [nDigits_two ha hb, show digits 2 z r = digitsFrom z 1 0 r from rfl]
    simp only [digitsFrom, Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one, one_mul,
      zero_add, List.getD_cons_zero, List.getD_cons_succ]
    have hz := b2i_bounds (digitZ r (z 0))
    have hhi := digitHi_toInt r (digitZ r (z 0))
    have hzv : digitZ r (z 0) = true → r.toInt / 128 ≠ 127 := by
      rw [digitZ_eq]; split_ifs <;> simp_all
    rw [digitLo_toInt, low8_toInt _ (by omega) ?_, hhi]
    · omega
    · rw [hhi]
      cases hd : digitZ r (z 0)
      · simp only [Bool.false_eq_true, if_false]; omega
      · have := hzv hd; simp only [if_true]; omega
  · have hk : (ks : ℤ) ≤ 4095 := by exact_mod_cast hb
    rw [nDigits_three ha (by omega), show digits 3 z r = digitsFrom z 2 0 r from rfl]
    simp only [digitsFrom, Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one, one_mul,
      zero_add, List.getD_cons_zero, List.getD_cons_succ]
    have hz0 := b2i_bounds (z 0)
    have hr1 := digitHi_toInt r (z 0)
    have hz1 : digitZ (digitHi r (z 0)) (z 1) = z 1 := by
      rw [digitZ_eq, hr1, if_neg (by have := b2i_bounds (z 0); omega)]
    rw [hz1]
    have hz1b := b2i_bounds (z 1)
    have hr2 := digitHi_toInt (digitHi r (z 0)) (z 1)
    rw [digitLo_toInt, digitLo_toInt, low8_toInt _ (by omega) (by omega), hr2, hr1]
    omega

theorem nDigits_pos {ks : ℕ} (h : ks ≤ 4095) : 1 ≤ nDigits ks := by
  rcases (show ks ≤ 1 ∨ (2 ≤ ks ∧ ks ≤ 128) ∨ (129 ≤ ks ∧ ks ≤ 16384) by omega) with h | ⟨ha, hb⟩ | ⟨ha, hb⟩
  · rw [nDigits_one h]
  · rw [nDigits_two ha hb]; omega
  · rw [nDigits_three ha hb]; omega

theorem sum_range_pred_succ (f : ℕ → ℤ) {m : ℕ} (hm : 1 ≤ m) :
    ∑ d ∈ Finset.range m, f d = f 0 + ∑ t : Fin (m - 1), f (t.val + 1) := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  rw [Finset.sum_range_succ', Fin.sum_univ_eq_sum_range (fun t => f (t + 1)), Nat.add_sub_cancel, add_comm]

/-! ### The stream's words -/

theorem sum_i8_bounds (f : ℕ → BitVec 8) : ∀ n : ℕ,
    -(128 * (n : ℤ)) ≤ ∑ s ∈ Finset.range n, (f s).toInt ∧ ∑ s ∈ Finset.range n, (f s).toInt ≤ 127 * n
  | 0 => by simp
  | n + 1 => by
      rw [Finset.sum_range_succ]
      have := sum_i8_bounds f n
      have := toInt8_bounds (f n)
      push_cast
      omega

theorem resSum_eq (x : ℕ → BitVec 8) (ρ : Fin 32) : ∀ n,
    resSum x ρ n = BitVec.ofInt 32 (∑ s ∈ Finset.range (n + 1), (x (32 * s + ρ.val)).toInt)
  | 0 => by simp [resSum, sx8]
  | n + 1 => by rw [resSum, resAdd, sx8, resSum_eq x ρ n, Finset.sum_range_succ (n := n + 1), BitVec.ofInt_add]

theorem segFold_segDot (x w : ℕ → BitVec 8) (init : BitVec 32) (n : ℕ) :
    segFold (fun s acc => segDot acc (seg x s) (seg w s)) init n =
      init + BitVec.ofInt 32 (∑ k ∈ Finset.range (32 * n), (x k).toInt * (w k).toInt) := by
  rw [segFold_add (fun s acc => segDot acc (seg x s) (seg w s))
    (fun s => ∑ e : Fin 32, (x (32 * s + e.val)).toInt * (w (32 * s + e.val)).toInt)
    (fun _ _ => rfl), sum_segs (fun k => (x k).toInt * (w k).toInt)]

theorem segFold_wrapSeg (x : ℕ → BitVec 8) (wb : ℕ → Bool × Bool) (init : BitVec 32) (n : ℕ) :
    segFold (fun s acc => wrapSeg acc (seg x s) (seg wb s)) init n =
      init + BitVec.ofInt 32 (∑ k ∈ Finset.range (32 * n),
        256 * ((x k).toInt * ((if (wb k).1 then 1 else 0) - (if (wb k).2 then 1 else 0)))) := by
  rw [segFold_add (fun s acc => wrapSeg acc (seg x s) (seg wb s))
    (fun s => ∑ e : Fin 32, 256 * ((x (32 * s + e.val)).toInt *
      ((if (wb (32 * s + e.val)).1 then 1 else 0) - (if (wb (32 * s + e.val)).2 then 1 else 0))))
    (fun _ _ => by simp only [wrapSeg, seg, Finset.mul_sum]; exact rfl),
    sum_segs (fun k => 256 * ((x k).toInt * ((if (wb k).1 then 1 else 0) - (if (wb k).2 then 1 else 0))))]

theorem segFold_peelSeg (x : ℕ → BitVec 8) (bd xa : ℕ → Bool) (w : ℕ → BitVec 8) (c : ℕ → Bool)
    (init : BitVec 32) (n : ℕ) :
    segFold (fun s acc => peelSeg acc (seg x s) (seg bd s) (seg xa s) (seg w s) (seg c s)) init n =
      init - BitVec.ofInt 32 (∑ k ∈ Finset.range (32 * n), ((x k).toInt * (if c k then 1 else 0) +
        (w k).toInt * ((if bd k then 1 else 0) - 256 * (if xa k then 1 else 0)))) := by
  rw [segFold_sub (fun s acc => peelSeg acc (seg x s) (seg bd s) (seg xa s) (seg w s) (seg c s))
    (fun s => ∑ e : Fin 32, ((x (32 * s + e.val)).toInt * (if c (32 * s + e.val) then 1 else 0) +
      (w (32 * s + e.val)).toInt * ((if bd (32 * s + e.val) then 1 else 0) -
        256 * (if xa (32 * s + e.val) then 1 else 0))))
    (fun _ _ => rfl),
    sum_segs (fun k => (x k).toInt * (if c k then 1 else 0) +
      (w k).toInt * ((if bd k then 1 else 0) - 256 * (if xa k then 1 else 0)))]

theorem oneStream_eq {m : ℕ} (op : ℤ) (d : Fin m → ℤ) :
    oneStream (BitVec.ofInt 32 op) (fun t => BitVec.ofInt 32 (d t)) =
      BitVec.ofInt 32 (op - ∑ t, d t * 2 ^ (7 * (t.val + 1))) := by
  unfold oneStream
  simp only [ofInt_shiftLeft32]
  rw [ofInt_sub32, ofInt_sum32]

theorem uOf_bounds {R C K : ℕ} (n : TileNoise R C K) (j : Fin C) (ρ : Fin 32) :
    -64 ≤ (uOf n j ρ).toInt ∧ (uOf n j ρ).toInt ≤ 64 :=
  u129_bounds _

theorem uOf_seg {R C K : ℕ} (n : TileNoise R C K) (j : Fin C) (s : ℕ) (e : Fin 32) :
    uOf n j ⟨(32 * s + e.val) % 32, Nat.mod_lt _ (by omega)⟩ = uOf n j e := by
  congr 1
  exact Fin.ext (by have := e.isLt; simp only []; omega)

/-- **v3's z is the exact dot** at a `K` EXACT binds at, whatever the noise: `X W′` less the one-stream's `X U`
(the residue sums' digits recombine exactly), plus the wrap correction, less the dither peel, is `A B` entry by
entry. -/
theorem streamZ_eq {R C K : ℕ} {P : Type} (n : TileNoise R C K) (x : TileIn R C K P) (i : Fin R) (j : Fin C)
    (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    streamZ n x i j = BitVec.ofInt 32 (∑ k, (x.a i k).toInt * (x.b j k).toInt) := by
  have hKS : K / 32 ≤ 4095 := by omega
  have hKSi : ((K / 32 : ℕ) : ℤ) ≤ 4095 := by exact_mod_cast hKS
  have hY : ∀ ρ : Fin 32, (resSum (xOf n x i) ρ (K / 32 - 1)).toInt =
      ∑ s ∈ Finset.range (K / 32), (xOf n x i (32 * s + ρ.val)).toInt := by
    intro ρ
    rw [resSum_eq]
    rcases Nat.eq_zero_or_pos (K / 32) with h0 | hpos
    · have hK0 : ¬ ρ.val < K := by omega
      rw [h0]
      simp [xOf, codeAt, bitAt, hK0, i8AddBit]
    · rw [Nat.sub_add_cancel hpos]
      have hb := sum_i8_bounds (fun s => xOf n x i (32 * s + ρ.val)) (K / 32)
      exact toInt_ofInt_of_range _ (by omega) (by omega)
  have hdig : ∀ ρ : Fin 32, (digitOf n x i 0 ρ).toInt +
      ∑ t : Fin (nDigits (K / 32) - 1), 128 ^ (t.val + 1) * (digitOf n x i (t.val + 1) ρ).toInt =
      ∑ s ∈ Finset.range (K / 32), (xOf n x i (32 * s + ρ.val)).toInt := by
    intro ρ
    have hb := sum_i8_bounds (fun s => xOf n x i (32 * s + ρ.val)) (K / 32)
    have h := digits_val (fun d' => n.zc i d' ρ) (resSum (xOf n x i) ρ (K / 32 - 1)) (K / 32) hKS
      (by rw [hY ρ]; exact hb.1) (by rw [hY ρ]; exact hb.2)
    rw [hY ρ, sum_range_pred_succ _ (nDigits_pos hKS)] at h
    simp only [pow_zero, one_mul] at h
    unfold digitOf
    exact h
  have hUY : ∑ ρ : Fin 32, (uOf n j ρ).toInt *
      ∑ s ∈ Finset.range (K / 32), (xOf n x i (32 * s + ρ.val)).toInt =
      ∑ k ∈ Finset.range (32 * (K / 32)),
        (xOf n x i k).toInt * (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩).toInt := by
    rw [← sum_segs (fun k => (xOf n x i k).toInt * (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩).toInt)]
    simp only [uOf_seg, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun e _ => mul_comm _ _
  have hmid : ∑ ρ : Fin 32, (i8Neg (uOf n j ρ)).toInt * (digitOf n x i 0 ρ).toInt -
      ∑ t : Fin (nDigits (K / 32) - 1),
        (∑ ρ : Fin 32, (uOf n j ρ).toInt * (digitOf n x i (t.val + 1) ρ).toInt) * 2 ^ (7 * (t.val + 1)) =
      -∑ k ∈ Finset.range (32 * (K / 32)),
        (xOf n x i k).toInt * (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩).toInt := by
    rw [← hUY]
    have e1 : ∑ t : Fin (nDigits (K / 32) - 1),
        (∑ ρ : Fin 32, (uOf n j ρ).toInt * (digitOf n x i (t.val + 1) ρ).toInt) * 2 ^ (7 * (t.val + 1)) =
        ∑ ρ : Fin 32, (uOf n j ρ).toInt *
          ∑ t : Fin (nDigits (K / 32) - 1), 128 ^ (t.val + 1) * (digitOf n x i (t.val + 1) ρ).toInt := by
      simp only [Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun ρ _ => Finset.sum_congr rfl fun t _ => ?_
      rw [pow_mul, show (2 : ℤ) ^ 7 = 128 by norm_num]
      ring
    rw [e1, ← Finset.sum_neg_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun ρ _ => ?_
    rw [i8Neg_toInt _ (uOf_bounds n j ρ).1 (uOf_bounds n j ρ).2, ← hdig ρ]
    ring
  have hR : ∑ k : Fin K, (x.a i k).toInt * (x.b j k).toInt =
      ∑ k ∈ Finset.range K, (codeAt (x.a i) k).toInt * (codeAt (x.b j) k).toInt := by
    rw [← Fin.sum_univ_eq_sum_range (fun k => (codeAt (x.a i) k).toInt * (codeAt (x.b j) k).toInt)]
    exact Finset.sum_congr rfl fun k _ => by simp [codeAt, k.isLt]
  unfold streamZ
  simp only []
  rw [segFold_peelSeg, segFold_wrapSeg, segFold_segDot]
  simp only [segDot0, segDot, zero_add]
  rw [← BitVec.ofInt_add, oneStream_eq, ← BitVec.ofInt_add, ← ofInt_sub32]
  congr 1
  have hsum : ∑ k ∈ Finset.range (32 * (K / 32)), (xOf n x i k).toInt * (wOf n x j k).toInt -
      ∑ k ∈ Finset.range (32 * (K / 32)),
        (xOf n x i k).toInt * (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩).toInt +
      ∑ k ∈ Finset.range (32 * (K / 32)), 256 * ((xOf n x i k).toInt *
        ((if (wrapBits (codeAt (x.b j) k) (bitAt (n.cc j) k) (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩)).1
          then 1 else 0) -
         (if (wrapBits (codeAt (x.b j) k) (bitAt (n.cc j) k) (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩)).2
          then 1 else 0))) -
      ∑ k ∈ Finset.range (32 * (K / 32)), ((xOf n x i k).toInt * (if bitAt (n.cc j) k then 1 else 0) +
        (codeAt (x.b j) k).toInt * ((if bitAt (n.bd i) k then 1 else 0) -
          256 * (if xWrap (codeAt (x.a i) k) (bitAt (n.bd i) k) then 1 else 0))) =
      ∑ k ∈ Finset.range K, (codeAt (x.a i) k).toInt * (codeAt (x.b j) k).toInt := by
    rw [Nat.mul_div_cancel' hK32, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    unfold xOf wOf
    rw [i8Add_toInt _ _ _ (uOf_bounds n j _).1 (uOf_bounds n j _).2, i8AddBit_toInt]
    ring
  rw [hR]
  linarith [hmid, hsum]

/-! ### The tile -/

/-- **TILE's z is the exact dot** in either form, at a `K` its EXACT binds at. -/
theorem TileForm.z_eq {R C K : ℕ} {P : Type} (form : TileForm R C K P) (x : TileIn R C K P) (i : Fin R) (j : Fin C)
    (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    form.z x i j = BitVec.ofInt 32 (∑ k, (x.a i k).toInt * (x.b j k).toInt) := by
  cases form with
  | exact => exact gemmAccS8_eq _ _ hK32 hK
  | stream noise => exact streamZ_eq _ _ _ _ hK32 hK

/-- **What the tile's `ok = 1` gives** (`PouwH1TileZy_v1{TILE}`, TILE v1 or v3, at a `K` its EXACT binds at): each
cell `(i, j)` has record z the s32 word of the exact dot of A row `i`'s and B row `j`'s codes and record y `dqW` of
that z at the rows' scales. -/
theorem tileOk_spec {R C K : ℕ} {P : Type} (form : TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : TileIn R C K P) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hok : tileOk form dqW x = true) (i : Fin R) (j : Fin C) :
    x.zr i j = BitVec.ofInt 32 (∑ k, (x.a i k).toInt * (x.b j k).toInt) ∧
      x.yr i j = dqW (BitVec.ofInt 32 (∑ k, (x.a i k).toInt * (x.b j k).toInt)) (x.sa i) (x.sw j) := by
  have h := (zyCheck_iff _ _ _ _ _ _).1 hok i j
  rw [TileForm.z_eq form x i j hK32 hK] at h
  exact h

end Pouw.H1Zy
