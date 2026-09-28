import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Discharge.Layout.Bytes

/-!
# What a region pins in a satisfying witness

The model's statement pins each region's bits (`Statement.Satisfies`): at block `o`, region bit `i` sits at the
in-block address `Region.inBlock i` and equals `bits o i`. For a region of the executable's statement, read as the
model's (`Refine.regionOf`), bit `k` of block `o` is bit `k % 8` of byte `k / 8` of the region's bytes there
(`regionOf_bit`), so a satisfying witness holds those bits (`region_read`).

Where they sit: a region `mkRegion` builds puts its free bits at the given positions and its fixed bits at `fixed`
(`mkRegion_spec`), so region bit `i`'s address is `fixed` plus `i`'s bits spread over the free positions
(`inBlock_mkRegion`, `spread`); and every in-block address of a region with distinct free bits is in the block
(`inBlock_lt`).
-/

namespace FlockSoundness.Discharge.Layout

open Model Flock

/-- Bit `k % 8` of a byte, as the executable reads it. -/
theorem uint8_bit (x : UInt8) (k : ℕ) : ((x >>> (k % 8).toUInt8) % 2 == 1) = x.toNat.testBit (k % 8) := by
  have h8 : (k % 8).toUInt8.toNat = k % 8 := by
    rw [Nat.toUInt8, UInt8.toNat_ofNat']; omega
  have h : ((x >>> (k % 8).toUInt8) % 2).toNat = x.toNat / 2 ^ (k % 8) % 2 := by
    rw [UInt8.toNat_mod, UInt8.toNat_shiftRight, h8, Nat.shiftRight_eq_div_pow]
    simp
  rw [Nat.testBit_eq_decide_div_mod_eq, ← h]
  generalize (x >>> (k % 8).toUInt8) % 2 = y
  apply Bool.eq_iff_iff.2
  rw [beq_iff_eq, decide_eq_true_iff, ← UInt8.toNat_inj]
  rfl

/-- **A region's bit, read as the model's**: bit `k` of block `b` is bit `k % 8` of byte `k / 8` of its bytes there. -/
theorem regionOf_bit (kl nbl : ℕ) (reg : Flock.Region) (b : Fin (2 ^ nbl)) (k : ℕ) :
    bit ((Refine.regionOf kl nbl reg).bits b k) = ((reg.bytes b.val).get! (k / 8)).toNat.testBit (k % 8) := by
  simp only [Refine.regionOf, uint8_bit]
  split_ifs with h <;> simp [bit, h]

/-- **What a satisfying witness holds at a region**: at block `o`, the bit at region bit `k`'s in-block address is bit
`k % 8` of byte `k / 8` of the region's bytes there. -/
theorem region_read {S : Statement} {z : Witness S.m} (hz : S.Satisfies z) {reg : Flock.Region}
    (hR : Refine.regionOf S.kLog (S.m - S.kLog) reg ∈ S.regions) (o : Fin (2 ^ S.nbl)) (k : ℕ)
    (hk : k < 2 ^ (Refine.regionOf S.kLog (S.m - S.kLog) reg).free.length)
    (ha : (Refine.regionOf S.kLog (S.m - S.kLog) reg).inBlock k < 2 ^ S.kLog) :
    bit (S.block z o ⟨_, ha⟩) = ((reg.bytes o.val).get! (k / 8)).toNat.testBit (k % 8) := by
  rw [hz.2.2 _ hR o k hk ha]
  exact regionOf_bit _ _ reg o k

/-! ### Where a region's bits sit -/

/-- The sum of `2^j` over the first `k` bit positions is `2^k - 1`, one short of `2^k`. -/
theorem sum_two_pow_fin : ∀ k : ℕ, ∑ j : Fin k, 2 ^ j.val + 1 = 2 ^ k
  | 0 => by simp
  | k + 1 => by
      rw [Fin.sum_univ_castSucc]
      simp only [Fin.val_castSucc, Fin.val_last]
      have := sum_two_pow_fin k
      rw [pow_succ]
      omega

/-- **Every in-block address of a region with distinct free bits is in the block**: the free bits and the fixed bits
off them are distinct positions below `k_log`. -/
theorem inBlock_lt {kl nbl : ℕ} (R : Model.Region kl nbl) (hnd : R.free.Nodup) (i : ℕ) : R.inBlock i < 2 ^ kl := by
  classical
  unfold Model.Region.inBlock
  have h1 : (∑ t : Fin R.free.length, if i.testBit t then 2 ^ (R.free.get t).val else 0) ≤
      ∑ j ∈ R.free.toFinset, 2 ^ j.val := by
    rw [List.sum_toFinset _ hnd]
    calc (∑ t : Fin R.free.length, if i.testBit t then 2 ^ (R.free.get t).val else 0)
        ≤ ∑ t : Fin R.free.length, 2 ^ (R.free.get t).val :=
          Finset.sum_le_sum fun t _ => by split_ifs <;> simp
      _ = (R.free.map fun j => 2 ^ j.val).sum := by
          rw [← List.sum_ofFn]
          congr 1
          exact List.ext_get (by simp) fun t h1 h2 => by simp
  have h2 : (∑ j : Fin kl, if j ∉ R.free ∧ R.fixedBits j then 2 ^ j.val else 0) ≤
      ∑ j ∈ Finset.univ.filter (· ∉ R.free.toFinset), 2 ^ j.val := by
    rw [Finset.sum_filter]
    exact Finset.sum_le_sum fun j _ => by
      by_cases hj : j ∈ R.free <;> simp only [hj, List.mem_toFinset, not_true_eq_false, not_false_eq_true,
        false_and, true_and, ite_false, ite_true, zero_le]
      split_ifs <;> simp
  have h3 : ∑ j ∈ R.free.toFinset, 2 ^ j.val + ∑ j ∈ Finset.univ.filter (· ∉ R.free.toFinset), 2 ^ j.val =
      ∑ j : Fin kl, 2 ^ j.val := by
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (· ∈ R.free.toFinset)]
    congr 1
    congr 1
    ext j
    simp
  have := sum_two_pow_fin kl
  omega

/-- A number's bits spread over the positions `F`: bit `t` of `i` to position `F[t]`. -/
def spread : List ℕ → ℕ → ℕ
  | [], _ => 0
  | f :: F, i => (if i.testBit 0 then 2 ^ f else 0) + spread F (i / 2)

theorem sum_free_spread {kl : ℕ} : ∀ (L : List (Fin kl)) (i : ℕ),
    (∑ t : Fin L.length, if i.testBit t then 2 ^ (L.get t).val else 0) = spread (L.map Fin.val) i
  | [], i => by simp [spread]
  | a :: L, i => by
      show (∑ t : Fin (L.length + 1), if i.testBit t then 2 ^ ((a :: L).get t).val else 0) = _
      rw [Fin.sum_univ_succ]
      simp only [List.map_cons, spread, Fin.val_zero, List.get_eq_getElem,
        Fin.val_succ, List.getElem_cons_succ, List.getElem_cons_zero, Nat.testBit_succ]
      congr 1
      have := sum_free_spread L (i / 2)
      simp only [List.get_eq_getElem] at this
      exact this

theorem spread_append : ∀ (A B : List ℕ) (i : ℕ), spread (A ++ B) i = spread A i + spread B (i / 2 ^ A.length)
  | [], B, i => by simp [spread]
  | a :: A, B, i => by
      simp only [List.cons_append, spread, List.length_cons]
      rw [spread_append A B (i / 2), Nat.div_div_eq_div_mul, ← pow_succ']
      ring

theorem spread_range : ∀ (n lo i : ℕ), spread ((List.range n).map (lo + ·)) i = i % 2 ^ n * 2 ^ lo
  | 0, lo, i => by simp [spread, Nat.mod_one]
  | n + 1, lo, i => by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      have hf : ((lo + ·) ∘ Nat.succ) = ((lo + 1) + ·) := funext fun j => by simp; omega
      rw [hf]
      simp only [spread, Nat.add_zero]
      rw [spread_range n (lo + 1) (i / 2), Nat.testBit_zero, show 2 ^ (n + 1) = 2 * 2 ^ n from pow_succ' 2 n,
        Nat.mod_mul, pow_succ]
      have h2 : i % 2 < 2 := Nat.mod_lt _ (by norm_num)
      by_cases h : i % 2 = 1
      · simp only [h, decide_true, ite_true]; ring
      · have h0 : i % 2 = 0 := by omega
        simp [h0]; ring

/-! ### The regions `mkRegion` builds -/

/-- The mask of a list of free bits, as `mkRegion` folds it. -/
def freeMask (free : Array ℕ) : ℕ := free.foldl (fun acc b => acc ||| 2 ^ b) 0

theorem testBit_foldl_or : ∀ (F : List ℕ) (x j : ℕ),
    (F.foldl (fun acc b => acc ||| 2 ^ b) x).testBit j = (x.testBit j || decide (j ∈ F))
  | [], x, j => by simp
  | b :: F, x, j => by
      rw [List.foldl_cons, testBit_foldl_or F, Nat.testBit_or, Nat.testBit_two_pow]
      by_cases h : j = b
      · subst h; simp
      · simp [h, Ne.symm h]

theorem testBit_freeMask (free : Array ℕ) (j : ℕ) : (freeMask free).testBit j = decide (j ∈ free.toList) := by
  rw [freeMask, ← Array.foldl_toList, testBit_foldl_or]
  simp

/-- **What `mkRegion` returns**: the region of the given name, free bits, fixed bits and bytes, with the mask of the
non-free in-block bits, once `fixed` is in the block and off the free bits, and the free bits are distinct, at most
`PT_LOCAL`, each in the block. -/
theorem mkRegion_spec {kl : ℕ} {name : String} {free : Array ℕ} {fixed : ℕ} {bytes : ℕ → ByteArray} {r : Region}
    (h : mkRegion kl name free fixed bytes = .ok r) :
    r = ⟨name, free, (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask free), fixed, bytes⟩ ∧ fixed < 2 ^ kl ∧
      fixed &&& freeMask free = 0 ∧ free.toList.Nodup ∧ free.size ≤ PT_LOCAL ∧ ∀ b ∈ free, b < kl := by
  have hok := mkRegion_ok h
  unfold mkRegion at h
  simp only at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i h1
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  simp only [Bool.or_eq_true, not_or, decide_eq_true_eq, not_le, bne_iff_ne, ne_eq, Decidable.not_not] at h1
  exact ⟨rfl, h1.1.1.1, h1.1.1.2, hok.2.1, hok.2.2.1, hok.2.2.2⟩

theorem filterMap_fin_map (kl : ℕ) : ∀ (L : List ℕ), (∀ b ∈ L, b < kl) →
    (L.filterMap fun b => if h : b < kl then some (⟨b, h⟩ : Fin kl) else none).map Fin.val = L
  | [], _ => rfl
  | b :: L, hlt => by
      rw [List.filterMap_cons]
      simp only [hlt b List.mem_cons_self, ↓reduceDIte, List.map_cons]
      rw [filterMap_fin_map kl L fun x hx => hlt x (List.mem_cons_of_mem _ hx)]

/-- The model's free bits of a region whose free bits are in the block: the same positions. -/
theorem regionOf_free_map (kl nbl : ℕ) (reg : Region) (hlt : ∀ b ∈ reg.free.toList, b < kl) :
    (Refine.regionOf kl nbl reg).free.map Fin.val = reg.free.toList :=
  filterMap_fin_map kl _ hlt

/-- **A region's in-block address**, for a region `mkRegion` builds: `fixed` plus the region bit's bits spread over the
free positions. -/
theorem inBlock_mkRegion {kl nbl : ℕ} {name : String} {free : Array ℕ} {fixed : ℕ} {bytes : ℕ → ByteArray}
    {r : Region} (h : mkRegion kl name free fixed bytes = .ok r) (i : ℕ) :
    (Refine.regionOf kl nbl r).inBlock i = spread free.toList i + fixed := by
  classical
  obtain ⟨rfl, hfix, hdis, -, -, hlt⟩ := mkRegion_spec h
  unfold Model.Region.inBlock
  rw [sum_free_spread, regionOf_free_map kl nbl _ fun b hb => hlt b (Array.mem_toList_iff.1 hb)]
  congr 1
  refine (Finset.sum_congr rfl fun j _ => ?_).trans (sum_testBit kl fixed hfix)
  have hfm : (freeMask free).testBit j = decide (j.val ∈ free.toList) := testBit_freeMask free j
  have hmem : j ∈ (Refine.regionOf kl nbl ⟨name, free, (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask free), fixed,
      bytes⟩).free ↔ j.val ∈ free.toList := by
    rw [← regionOf_free_map kl nbl ⟨name, free, _, fixed, bytes⟩ fun b hb => hlt b (Array.mem_toList_iff.1 hb)]
    constructor
    · intro hj; exact List.mem_map_of_mem hj
    · intro hj
      obtain ⟨j', hj', hjj⟩ := List.mem_map.1 hj
      exact (Fin.ext hjj) ▸ hj'
  have hbit : ∀ x : ℕ, ((x >>> j.val) % 2 == 1) = x.testBit j.val := by
    intro x
    rw [Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
    apply Bool.eq_iff_iff.2
    simp
  have hcond' : (j.val ∉ free.toList ∧ ((((2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask free)) >>> j.val) % 2 == 1 &&
      (fixed >>> j.val) % 2 == 1) = true) ↔ fixed.testBit j.val = true := by
    simp only [hbit, Bool.and_eq_true, Nat.testBit_and, Nat.testBit_xor, Nat.testBit_two_pow_sub_one, j.isLt,
      decide_true, Bool.true_and, hfm]
    constructor
    · exact fun h => h.2.2
    · intro hf
      have hnot : (freeMask free).testBit j.val = false := by
        have := congrArg (fun x => x.testBit j.val) hdis
        simp only [Nat.testBit_and, hf, Bool.true_and, Nat.zero_testBit] at this
        exact this
      rw [hfm] at hnot
      have hnm : j.val ∉ free.toList := by simpa using hnot
      exact ⟨hnm, by simpa using hnm, hf⟩
  have hcond : (j ∉ (Refine.regionOf kl nbl ⟨name, free, (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask free), fixed,
      bytes⟩).free ∧ (Refine.regionOf kl nbl ⟨name, free, (2 ^ kl - 1) &&& ((2 ^ kl - 1) ^^^ freeMask free), fixed,
      bytes⟩).fixedBits j = true) ↔ fixed.testBit j.val = true := by
    rw [hmem]
    exact hcond'
  split_ifs with h1 h2 h2
  · rfl
  · exact absurd (hcond.1 h1) h2
  · exact absurd (hcond.2 h2) h1
  · rfl

end FlockSoundness.Discharge.Layout
