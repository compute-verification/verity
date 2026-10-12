import Proofs.Flock.C3.Fork.OracleClose

/-!
# Piece (a), the oracle count: the hidden part's fold is the sparse phase's

A rep's lincheck runs on the public statement `cls.pubStatement regs` and takes the hidden part's fold `h` from the
prover (`holoLincheck`, L4′). At a registered table `R`, the hidden statement `R.toStatement τ regs` has the public
part plus, on each unit slot `u` of slot `σ`, type `τ σ`'s template. Its lincheck comb, folded by the column
challenges, is the public comb's fold plus `sparseFold` at `R`'s cells (`holoFold_eq`): the comb is linear in the
matrices (characteristic 2 casts `ZMod 2` sums to sums), a template's entries at row `i` and column `c` of unit `u` are
weighted `eq(ρ_in, i)·eq(ts, c)`, which `Φ` restates on the entries' planes, and the unit's part of the weights is the
slot's layout factor `U_σ` (`holoPub`), unit by unit, since every unit is inside the block (`hunits`).

So where the prover sends `h = hiddenFold`, `holoLincheck` checks what `lincheck` checks on the hidden statement.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Model

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The hidden part's fold** at `R`'s cells, at a rep's zerocheck output and its lincheck's `α` and column challenges
`ts`: the `h` an honest prover sends, and the one the sparse phase checks `h` against (`holoPub` reads no more of the
lincheck's output). -/
noncomputable def hiddenFold (A : Arith F K) (cls : HoloClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (R : RegTable cls) (zc : ZcOut F cls.m) (α : F)
    (ts : Fin (cls.kLog - 6) → F) : Fin 64 → F :=
  sparseFold A cls.shape (holoPub (cls.pubStatement regs) cls.shape cls.units zc ⟨⟨0, 0, 0⟩, α, ts, 0⟩)
    (fun σ => (τ σ).val) R.bits

namespace OracleFold

/-! ## A table's planes as field elements -/

omit [Fintype F] [DecidableEq F] in
theorem val_castHom [CharP F 2] (x : ZMod 2) : ((x.val : ℕ) : F) = ZMod.castHom (dvd_refl 2) F x := by
  rw [ZMod.castHom_apply, ZMod.natCast_val]

theorem val_cases (x : ZMod 2) : x.val = 0 ∨ x.val = 1 := by
  have := ZMod.val_lt x
  omega

variable {cls : HoloClass}

theorem testBit_word (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p₀ : ℕ) (k : Fin (2 ^ cls.nu)) (j : ℕ) :
    (R.word t p₀ k).testBit j = decide (j < cls.slotLog ∧ (R.plane t (p₀ + j) k).val = 1) := by
  have hw : R.word t p₀ k =
      ∑ i ∈ (Finset.range cls.slotLog).filter (fun i => (R.plane t (p₀ + i) k).val = 1), 2 ^ i := by
    unfold RegTable.word
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun i _ => ?_
    rcases val_cases (R.plane t (p₀ + i) k) with h | h <;> simp [h]
  rw [hw, testBit_sum_two_pow]
  simp only [Finset.mem_filter, Finset.mem_range]

theorem word_lt (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p₀ : ℕ) (k : Fin (2 ^ cls.nu)) :
    R.word t p₀ k < 2 ^ cls.slotLog :=
  Nat.lt_pow_two_of_testBit _ fun j hj => by rw [testBit_word, decide_eq_false_iff_not]; omega

omit [Fintype F] [DecidableEq F] in
theorem cast_plane (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p₀ : ℕ) (k : Fin (2 ^ cls.nu)) {j : ℕ}
    (hj : j < cls.slotLog) :
    ((R.plane t (p₀ + j) k).val : F) = if (R.word t p₀ k).testBit j then 1 else 0 := by
  rw [testBit_word]
  rcases val_cases (R.plane t (p₀ + j) k) with h | h <;> simp [h, hj]

omit [Fintype F] [DecidableEq F] in
/-- The `ℓ` row (or column) factors of `Φ` at an entry's planes are `eq` at the entry's row (or column). -/
theorem prod_eqAt [CharP F 2] (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p₀ : ℕ) (k : Fin (2 ^ cls.nu))
    (xv : ℕ → F) :
    ∏ j ∈ Finset.range cls.slotLog, (1 + xv j + ((R.plane t (p₀ + j) k).val : F)) =
      eqAt (fun j : Fin cls.slotLog => xv j.val) (R.word t p₀ k) := by
  rw [Finset.prod_range]
  unfold eqAt
  refine Finset.prod_congr rfl fun j _ => ?_
  rw [cast_plane R t p₀ k j.isLt]
  by_cases hb : (R.word t p₀ k).testBit j.val
  · rw [ite_eq_left hb, ite_eq_left hb, add_right_comm, CharTwo.add_self_eq_zero, zero_add]
  · rw [ite_eq_right hb, ite_eq_right hb, add_zero]

theorem decode3 {a b k p t : ℕ} (hk : k < 2 ^ a) (hp : p < 2 ^ b) :
    (k + 2 ^ a * (p + 2 ^ b * t)) / 2 ^ (a + b) = t ∧ (k + 2 ^ a * (p + 2 ^ b * t)) / 2 ^ a % 2 ^ b = p ∧
      (k + 2 ^ a * (p + 2 ^ b * t)) % 2 ^ a = k := by
  have h1 : (k + 2 ^ a * (p + 2 ^ b * t)) / 2 ^ a = p + 2 ^ b * t := by
    rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos a), Nat.div_eq_of_lt hk, zero_add]
  refine ⟨?_, ?_, ?_⟩
  · rw [pow_add, ← Nat.div_div_eq_div_mul, h1, Nat.add_mul_div_left _ _ (Nat.two_pow_pos b),
      Nat.div_eq_of_lt hp, zero_add]
  · rw [h1, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hk]

omit [Fintype F] [DecidableEq F] in
/-- The sparse phase's planes at `R.bits` are `R`'s. -/
theorem planeAt_bits (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) {p : ℕ} (hp : p < 2 ^ cls.shape.pLog)
    (k : Fin (2 ^ cls.nu)) : (planeAt cls.shape R.bits t.val p k.val : F) = ((R.plane t p k).val : F) := by
  have h : k.val + 2 ^ cls.shape.nu * (p + 2 ^ cls.shape.pLog * t.val) < 2 ^ cls.shape.mReg :=
    cell_lt t ⟨p, hp⟩ k
  have h' : k.val + 2 ^ cls.shape.nu * (p + 2 ^ cls.shape.pLog * t.val) <
      2 ^ (cls.nu + cls.shape.pLog + cls.tLog) := by
    have := idx_lt k.isLt (idx_lt hp t.isLt)
    rwa [← Nat.add_assoc] at this
  have d := decode3 (a := cls.nu) (t := t.val) k.isLt hp
  unfold planeAt
  rw [dite_eq_left h]
  unfold RegTable.bits RegTable.plane
  rw [dite_eq_left h', dite_eq_left hp]
  congr <;> first | exact Fin.ext d.1 | exact Fin.ext d.2.1 | exact Fin.ext d.2.2 | exact d.1 | exact d.2.1 |
    exact d.2.2

/-! ## `Φ` at a table's entry -/

/-- The value planes of entry `k` of type `t`: `α·A₀ + B₀`. -/
def planeV (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (α : F) (k : Fin (2 ^ cls.nu)) : F :=
  α * ((R.plane t (2 * cls.slotLog) k).val : F) + ((R.plane t (2 * cls.slotLog + 1) k).val : F)

omit [Fintype F] [DecidableEq F] in
theorem eqAt_cat6 [CharP F 2] {ℓ : ℕ} (h6 : 6 ≤ ℓ) (r : Fin 64) (xr : Fin (ℓ - 6) → F) (w : ℕ) :
    eqAt (fun j : Fin ℓ => cat6 (skipBits r) xr j.val) w = (if w % 64 = r.val then 1 else 0) * eqAt xr (w / 64) := by
  have h := eqAt_split_at 6 (ℓ - 6) (by omega) (fun j : Fin ℓ => cat6 (skipBits r) xr j.val) (w % 64) (w / 64)
    (Nat.mod_lt _ (by norm_num))
  rw [show w % 64 + 2 ^ 6 * (w / 64) = w by
    rw [show (2 : ℕ) ^ 6 = 64 by norm_num]; exact Nat.mod_add_div w 64] at h
  rw [h, ← eqAt_bool (R := F) (n := 6) (w % 64) r.val (Nat.mod_lt _ (by norm_num)) r.isLt]
  congr 1
  congr 1
  funext j
  simp [cat6, vget, j.isLt]

omit [Fintype F] [DecidableEq F] in
theorem phi_eq' [CharP F 2] (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (k : Fin (2 ^ cls.nu)) (α : F)
    (r s : Fin 64) (xr yr : Fin (cls.slotLog - 6) → F) :
    sparsePhi cls.slotLog α (cat6 (skipBits r) xr) (cat6 (skipBits s) yr)
        (fun p => planeAt cls.shape R.bits t.val p k.val) =
      planeV R t α k * ((if R.word t 0 k % 64 = r.val then 1 else 0) * eqAt xr (R.word t 0 k / 64)) *
        ((if R.word t cls.slotLog k % 64 = s.val then 1 else 0) * eqAt yr (R.word t cls.slotLog k / 64)) := by
  rw [SparseCount.sparsePhi_congr cls.slotLog α _ _ (q' := fun p => ((R.plane t p k).val : F))
    fun p hp => planeAt_bits R t (SparseCount.lt_two_pow_pLog cls.shape hp) k]
  have h0 := prod_eqAt R t 0 k (cat6 (skipBits r) xr)
  simp only [zero_add] at h0
  simp only [sparsePhi, Finset.prod_mul_distrib, h0, prod_eqAt R t cls.slotLog k (cat6 (skipBits s) yr),
    eqAt_cat6 cls.hs6 r xr, eqAt_cat6 cls.hs6 s yr]
  unfold planeV
  ring

omit [Fintype F] [DecidableEq F] in
/-- **`Φ` at entry `k` of type `t`**: its value planes, times the skip's indicators and `eq` past the skip at the
entry's row and column. -/
theorem phi_eq [CharP F 2] (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (k : Fin (2 ^ cls.nu)) (α : F)
    (r s : Fin 64) (xr yr : Fin (cls.shape.ell - 6) → F) :
    sparsePhi cls.shape.ell α (cat6 (skipBits r) xr) (cat6 (skipBits s) yr)
        (fun p => planeAt cls.shape R.bits t.val p k.val) =
      planeV R t α k * ((if R.word t 0 k % 64 = r.val then 1 else 0) * eqAt xr (R.word t 0 k / 64)) *
        ((if R.word t cls.slotLog k % 64 = s.val then 1 else 0) * eqAt yr (R.word t cls.slotLog k / 64)) :=
  phi_eq' R t k α r s xr yr

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- The skip's rows in `sparseFold`, at entry `k` of type `t`: only the entry's row survives. -/
theorem skip_sum [CharP F 2] (A : Arith F K) (R : RegTable cls) (pub : SparsePub F cls.shape)
    (t : Fin (2 ^ cls.tLog)) (k : Fin (2 ^ cls.nu)) (Uσ : F) (s : Fin 64) :
    ∑ r : Fin 64, A.lagS pub.z r * (Uσ * sparsePhi cls.shape.ell pub.alpha (cat6 (skipBits r) pub.xRest)
        (cat6 (skipBits s) pub.yRest) (fun p => planeAt cls.shape R.bits t.val p k.val)) =
      Uσ * (planeV R t pub.alpha k *
        (A.lagS pub.z ⟨R.word t 0 k % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt pub.xRest (R.word t 0 k / 64)) *
          ((if R.word t cls.slotLog k % 64 = s.val then 1 else 0) * eqAt pub.yRest (R.word t cls.slotLog k / 64))) := by
  simp only [phi_eq]
  rw [Finset.sum_eq_single ⟨R.word t 0 k % 64, Nat.mod_lt _ (by norm_num)⟩]
  · rw [ite_eq_left rfl]
    ring
  · intro r _ hr
    have : R.word t 0 k % 64 ≠ r.val := fun h => hr (Fin.ext h.symm)
    rw [ite_eq_right this]
    ring
  · exact fun h => absurd (Finset.mem_univ _) h

/-! ## The comb's hidden part -/

/-- The hidden part of the comb at column `c`: slot `σ`'s entry `k` sits at row `row + 2^ℓ·u` and column
`col + 2^ℓ·u` of each unit `u` the slot fills. -/
noncomputable def hidComb (A : Arith F K) (cls : HoloClass) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (R : RegTable cls) (z α : F) (ρ : Fin (cls.kLog - 6) → F) (c : ℕ) : F :=
  if c < 2 ^ cls.kLog then
    ∑ σ : Fin (2 ^ cls.sLog), ∑ k : Fin (2 ^ cls.nu), ∑ i : Fin (2 ^ cls.kLog),
      if i.val / 2 ^ cls.slotLog = c / 2 ^ cls.slotLog ∧ i.val / 2 ^ cls.slotLog ∈ cls.units σ ∧
          R.word (τ σ) 0 k = i.val % 2 ^ cls.slotLog ∧ R.word (τ σ) cls.slotLog k = c % 2 ^ cls.slotLog then
        A.lagS z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt ρ (i.val / 64) * planeV R (τ σ) α k
      else 0
  else 0

theorem ite_sum_ite {ι κ M : Type} [Fintype ι] [Fintype κ] [AddCommMonoid M] (p : Prop) [Decidable p]
    (q : ι → Prop) [DecidablePred q] (r : ι → κ → Prop) [∀ a, DecidablePred (r a)] (f : ι → κ → M) :
    (if p then ∑ a, (if q a then ∑ b, (if r a b then f a b else 0) else 0) else 0) =
      ∑ a, ∑ b, if p ∧ q a ∧ r a b then f a b else 0 := by
  by_cases hp : p
  · simp only [hp, true_and, ↓reduceIte]
    refine Finset.sum_congr rfl fun a _ => ?_
    by_cases hq : q a <;> simp [hq]
  · simp [hp]

omit [Fintype F] [DecidableEq F] in
theorem cast_side [CharP F 2] (R : RegTable cls) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (s : Bool)
    (i c : Fin (2 ^ cls.kLog)) :
    ((R.side τ s i c).val : F) = (((if s then cls.B₀pub else cls.A₀pub) i c).val : F) +
      ∑ σ : Fin (2 ^ cls.sLog), ∑ k : Fin (2 ^ cls.nu),
        if i.val / 2 ^ cls.slotLog = c.val / 2 ^ cls.slotLog ∧ i.val / 2 ^ cls.slotLog ∈ cls.units σ ∧
            R.word (τ σ) 0 k = i.val % 2 ^ cls.slotLog ∧ R.word (τ σ) cls.slotLog k = c.val % 2 ^ cls.slotLog then
          ((R.plane (τ σ) (2 * cls.slotLog + s.toNat) k).val : F)
        else 0 := by
  simp only [val_castHom, RegTable.side, RegTable.tmpl, map_add, map_sum,
    apply_ite (ZMod.castHom (dvd_refl 2) F), map_zero]
  congr 1
  exact ite_sum_ite _ _ _ _

theorem sum_swap3 {ι κ μ M : Type} [Fintype ι] [Fintype κ] [Fintype μ] [AddCommMonoid M] (g : ι → κ → μ → M) :
    ∑ a, ∑ b, ∑ c, g a b c = ∑ c, ∑ a, ∑ b, g a b c :=
  (Finset.sum_congr rfl fun _ _ => Finset.sum_comm).trans Finset.sum_comm

omit [Fintype F] [DecidableEq F] in
theorem sum_ite_lin {ι κ : Type} [Fintype ι] [Fintype κ] (C : ι → κ → Prop) [∀ a b, Decidable (C a b)]
    (x y : ι → κ → F) (L α : F) :
    ∑ a, ∑ b, (if C a b then L * (α * x a b + y a b) else 0) =
      L * (α * ∑ a, ∑ b, (if C a b then x a b else 0) + ∑ a, ∑ b, (if C a b then y a b else 0)) := by
  simp only [mul_add, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  split_ifs <;> ring

omit [DecidableEq F] [DecidableEq K] in
/-- **The comb is linear in the matrices**: the hidden statement's comb is the public one's plus the hidden part. -/
theorem comb_split (A : Arith F K) (hA : A.Correct) (cls : HoloClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (R : RegTable cls) (z α β : F) (ρ : Fin (cls.kLog - 6) → F) (c : ℕ) :
    comb A (R.toStatement τ regs) z α β ρ c = comb A (cls.pubStatement regs) z α β ρ c + hidComb A cls τ R z α ρ c := by
  have := hA.charTwo
  unfold comb hidComb
  show (if hc : c < 2 ^ cls.kLog then
      (∑ i : Fin (2 ^ cls.kLog), A.lagS z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt ρ (i.val / 64) *
        (α * ((R.side τ false i ⟨c, hc⟩).val : F) + ((R.side τ true i ⟨c, hc⟩).val : F))) +
        (if c = cls.pin.val then β else 0) else 0) =
    (if hc : c < 2 ^ cls.kLog then
      (∑ i : Fin (2 ^ cls.kLog), A.lagS z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt ρ (i.val / 64) *
        (α * ((cls.A₀pub i ⟨c, hc⟩).val : F) + ((cls.B₀pub i ⟨c, hc⟩).val : F))) +
        (if c = cls.pin.val then β else 0) else 0) + _
  by_cases hc : c < 2 ^ cls.kLog
  · simp only [hc, ↓reduceDIte, ↓reduceIte]
    rw [add_right_comm, sum_swap3, ← Finset.sum_add_distrib]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [cast_side R τ false i ⟨c, hc⟩, cast_side R τ true i ⟨c, hc⟩]
    simp only [Bool.false_eq_true, ↓reduceIte, Bool.toNat_false, Bool.toNat_true, add_zero, planeV]
    rw [sum_ite_lin]
    ring
  · simp only [hc, ↓reduceDIte, ↓reduceIte, add_zero]

/-! ## The block's units -/

omit [Fintype F] [DecidableEq F] in
theorem sum_hit_range {kl sl : ℕ} (h6 : 6 ≤ sl) (hsk : sl ≤ kl) (U : Finset ℕ) (hU : ∀ u ∈ U, u < 2 ^ (kl - sl))
    {a b : ℕ} (ha : a < 2 ^ sl) (hb : b < 2 ^ sl) (s : ℕ) (hs : s < 64) (G : ℕ → ℕ → F) :
    ∑ w ∈ Finset.range (2 ^ (kl - 6)), ∑ i ∈ Finset.range (2 ^ kl),
        (if i / 2 ^ sl = (s + 64 * w) / 2 ^ sl ∧ i / 2 ^ sl ∈ U ∧ a = i % 2 ^ sl ∧
            b = (s + 64 * w) % 2 ^ sl then G w i else 0) =
      ∑ u ∈ U, if b % 64 = s then G (b / 64 + 2 ^ (sl - 6) * u) (a + 2 ^ sl * u) else 0 := by
  have e64 : (2 : ℕ) ^ sl = 64 * 2 ^ (sl - 6) := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.add_sub_cancel' h6]
  have e1 : 2 ^ kl = 2 ^ sl * 2 ^ (kl - sl) := by rw [← pow_add, Nat.add_sub_cancel' hsk]
  have e2 : 2 ^ (kl - 6) = 2 ^ (sl - 6) * 2 ^ (kl - sl) := by rw [← pow_add]; congr 1; omega
  have hb64 : b / 64 < 2 ^ (sl - 6) := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num), mul_comm, ← e64]; exact hb
  calc _ = ∑ u ∈ Finset.range (2 ^ (kl - sl)),
        (if u ∈ U ∧ b % 64 = s then G (b / 64 + 2 ^ (sl - 6) * u) (a + 2 ^ sl * u) else 0) := by
        rw [e1, e2, SparseCount.sum_range_mul]
        refine Finset.sum_congr rfl fun u hu => ?_
        have hu := Finset.mem_range.mp hu
        have hai : a + 2 ^ sl * u < 2 ^ sl * 2 ^ (kl - sl) := by rw [← pow_add]; exact idx_lt ha hu
        have hda : (a + 2 ^ sl * u) / 2 ^ sl = u ∧ (a + 2 ^ sl * u) % 2 ^ sl = a := by
          rw [Nat.add_mul_div_left _ _ (Nat.two_pow_pos sl), Nat.div_eq_of_lt ha, zero_add,
            Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha]
          exact ⟨rfl, rfl⟩
        calc _ = ∑ wl ∈ Finset.range (2 ^ (sl - 6)),
              (if u ∈ U ∧ b = s + 64 * wl then G (wl + 2 ^ (sl - 6) * u) (a + 2 ^ sl * u) else 0) := by
              refine Finset.sum_congr rfl fun wl hwl => ?_
              have hwl := Finset.mem_range.mp hwl
              have hlt : s + 64 * wl < 2 ^ sl := by
                rw [e64]
                calc s + 64 * wl < 64 * (wl + 1) := by omega
                  _ ≤ 64 * 2 ^ (sl - 6) := Nat.mul_le_mul_left _ hwl
              have hc : s + 64 * (wl + 2 ^ (sl - 6) * u) = (s + 64 * wl) + 2 ^ sl * u := by rw [e64]; ring
              rw [hc, Nat.add_mul_div_left _ _ (Nat.two_pow_pos sl), Nat.div_eq_of_lt hlt, zero_add,
                Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]
              rw [Finset.sum_eq_single (a + 2 ^ sl * u)]
              · rw [hda.1, hda.2]
                simp only [true_and]
              · intro i _ hi
                refine ite_eq_right fun h => hi ?_
                rw [← Nat.mod_add_div i (2 ^ sl), ← h.2.2.1, h.1]
              · exact fun h => absurd (Finset.mem_range.mpr hai) h
          _ = _ := by
              by_cases hbs : b % 64 = s
              · rw [Finset.sum_eq_single (b / 64)]
                · have : b = s + 64 * (b / 64) := by omega
                  simp only [hbs, ← this, and_true]
                · intro wl _ hwl
                  exact ite_eq_right fun h => hwl (by omega)
                · exact fun h => absurd (Finset.mem_range.mpr hb64) h
              · rw [ite_eq_right fun h => hbs h.2]
                exact Finset.sum_eq_zero fun wl _ => ite_eq_right fun h => hbs (by omega)
    _ = ∑ u ∈ U, (if u ∈ U ∧ b % 64 = s then G (b / 64 + 2 ^ (sl - 6) * u) (a + 2 ^ sl * u) else 0) :=
        (Finset.sum_subset (fun u hu => Finset.mem_range.mpr (hU u hu)) fun u _ hu => by simp [hu]).symm
    _ = _ := Finset.sum_congr rfl fun u hu => by simp [hu]

omit [Fintype F] [DecidableEq F] in
/-- **The hidden entries' weights, unit by unit**: over the block's rows `i` and the folded columns `s + 64·w`, an
entry at slot row `a` and column `b` is hit once per unit `u`, at row `a + 2^ℓ·u` and column `b + 2^ℓ·u`. -/
theorem sum_hit {kl sl : ℕ} (h6 : 6 ≤ sl) (hsk : sl ≤ kl) (U : Finset ℕ) (hU : ∀ u ∈ U, u < 2 ^ (kl - sl))
    {a b : ℕ} (ha : a < 2 ^ sl) (hb : b < 2 ^ sl) (s : ℕ) (hs : s < 64) (G : ℕ → ℕ → F) :
    ∑ w : Fin (2 ^ (kl - 6)), ∑ i : Fin (2 ^ kl),
        (if i.val / 2 ^ sl = (s + 64 * w.val) / 2 ^ sl ∧ i.val / 2 ^ sl ∈ U ∧ a = i.val % 2 ^ sl ∧
            b = (s + 64 * w.val) % 2 ^ sl then G w.val i.val else 0) =
      ∑ u ∈ U, if b % 64 = s then G (b / 64 + 2 ^ (sl - 6) * u) (a + 2 ^ sl * u) else 0 := by
  have h := sum_hit_range h6 hsk U hU ha hb s hs G
  simp only [Finset.sum_range] at h
  exact h

omit [Fintype F] [DecidableEq F] in
/-- One entry's weights over its units: the skip's and `eq`'s parts at the row and column, times the slot's layout
factor. -/
theorem slot_term {kl sl : ℕ} (h6 : 6 ≤ sl) (hsk : sl ≤ kl) (lag : Fin 64 → F) (ρ y : Fin (kl - 6) → F)
    (U : Finset ℕ) {a b : ℕ} (ha : a < 2 ^ sl) (hb : b < 2 ^ sl) (s : ℕ) (V : F) :
    (∑ u ∈ U, if b % 64 = s then eqAt y (b / 64 + 2 ^ (sl - 6) * u) *
        (lag ⟨(a + 2 ^ sl * u) % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt ρ ((a + 2 ^ sl * u) / 64) * V) else 0) =
      (∑ u ∈ U, eqAt (fun j : Fin (kl - sl) => ρ ⟨sl - 6 + j.val, by omega⟩) u *
          eqAt (fun j : Fin (kl - sl) => y ⟨sl - 6 + j.val, by omega⟩) u) *
        (V * (lag ⟨a % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt (fun j : Fin (sl - 6) => ρ ⟨j.val, by omega⟩) (a / 64)) *
          ((if b % 64 = s then 1 else 0) * eqAt (fun j : Fin (sl - 6) => y ⟨j.val, by omega⟩) (b / 64))) := by
  have e64 : (2 : ℕ) ^ sl = 64 * 2 ^ (sl - 6) := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.add_sub_cancel' h6]
  have ha' : a / 64 < 2 ^ (sl - 6) := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num), mul_comm, ← e64]; exact ha
  have hb' : b / 64 < 2 ^ (sl - 6) := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num), mul_comm, ← e64]; exact hb
  have hN : sl - 6 + (kl - sl) = kl - 6 := by omega
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun u _ => ?_
  have hm : (a + 2 ^ sl * u) % 64 = a % 64 := by rw [e64, mul_assoc, Nat.add_mul_mod_self_left]
  have hd : (a + 2 ^ sl * u) / 64 = a / 64 + 2 ^ (sl - 6) * u := by
    rw [e64, mul_assoc, Nat.add_mul_div_left _ _ (by norm_num)]
  have hl : lag ⟨(a + 2 ^ sl * u) % 64, Nat.mod_lt _ (by norm_num)⟩ = lag ⟨a % 64, Nat.mod_lt _ (by norm_num)⟩ :=
    congrArg lag (Fin.ext hm)
  rw [hl, hd, eqAt_split_at (sl - 6) (kl - sl) hN ρ _ _ ha', eqAt_split_at (sl - 6) (kl - sl) hN y _ _ hb']
  split_ifs <;> ring

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem sparseFold_slots (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ' : Fin (2 ^ sh.sLog) → ℕ)
    (z : Witness sh.mReg) (s : Fin 64) :
    sparseFold A sh pub τ' z s = ∑ σ : Fin (2 ^ sh.sLog), ∑ k : Fin (2 ^ sh.nu), ∑ r : Fin 64,
      A.lagS pub.z r * (pub.U σ * sparsePhi sh.ell pub.alpha (cat6 (skipBits r) pub.xRest)
        (cat6 (skipBits s) pub.yRest) (fun p => planeAt sh z (τ' σ) p k.val)) := by
  unfold sparseFold
  simp only [Finset.mul_sum]
  exact (sum_swap3 _).symm

omit [DecidableEq F] [DecidableEq K] in
/-- **The hidden part's fold is the sparse phase's fold** at `R`'s cells. -/
theorem hidFold_eq (A : Arith F K) (hA : A.Correct) (cls : HoloClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (R : RegTable cls) (hunits : ∀ σ, ∀ u ∈ cls.units σ, u < 2 ^ (cls.kLog - cls.slotLog))
    (zc : ZcOut F cls.m) (α : F) (ts : Fin (cls.kLog - 6) → F) (s : Fin 64) :
    ∑ w : Fin (2 ^ (cls.kLog - 6)), eqAt (fun j : Fin (cls.kLog - 6) => ts ⟨cls.kLog - 6 - 1 - j.val, by omega⟩) w.val *
        hidComb A cls τ R zc.z α (fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩)
          (s.val + 64 * w.val) =
      hiddenFold A cls regs τ R zc α ts s := by
  have := hA.charTwo
  have h6 := cls.hs6
  have hsk := cls.hsk
  have hlt : ∀ w : Fin (2 ^ (cls.kLog - 6)), s.val + 64 * w.val < 2 ^ cls.kLog := fun w => by
    have hK : 2 ^ cls.kLog = 2 ^ (cls.kLog - 6) * 64 := by
      rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel cls.hk6]
    have hw := w.isLt
    have hs := s.isLt
    rw [hK]
    calc s.val + 64 * w.val < 64 * (w.val + 1) := by omega
      _ ≤ 64 * 2 ^ (cls.kLog - 6) := Nat.mul_le_mul_left _ hw
      _ = _ := mul_comm _ _
  have hx : (holoPub (cls.pubStatement regs) cls.shape cls.units zc ⟨⟨0, 0, 0⟩, α, ts, 0⟩).xRest =
      fun j : Fin (cls.slotLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩ := by
    funext j
    have hj : j.val < cls.slotLog - 6 := j.isLt
    exact dite_eq_left (show j.val < cls.kLog - 6 by omega)
  have hy : (holoPub (cls.pubStatement regs) cls.shape cls.units zc ⟨⟨0, 0, 0⟩, α, ts, 0⟩).yRest =
      fun j : Fin (cls.slotLog - 6) => ts ⟨cls.kLog - 6 - 1 - j.val, by omega⟩ := by
    funext j
    have hj : j.val < cls.slotLog - 6 := j.isLt
    exact dite_eq_left (show j.val < cls.kLog - 6 by omega)
  have hU : ∀ σ, (holoPub (cls.pubStatement regs) cls.shape cls.units zc ⟨⟨0, 0, 0⟩, α, ts, 0⟩).U σ =
      ∑ u ∈ cls.units σ,
        eqAt (fun j : Fin (cls.kLog - cls.slotLog) => zc.ρ ⟨cls.slotLog - 6 + j.val, by have := cls.hk; omega⟩) u *
          eqAt (fun j : Fin (cls.kLog - cls.slotLog) => ts ⟨cls.kLog - 6 - 1 - (cls.slotLog - 6 + j.val), by omega⟩)
            u := by
    intro σ
    refine Finset.sum_congr rfl fun u _ => congrArg₂ (· * ·) ?_ ?_
    · congr 1
      funext j
      have hj : j.val < cls.kLog - cls.slotLog := j.isLt
      exact dite_eq_left (show cls.slotLog - 6 + j.val < cls.kLog - 6 by omega)
    · congr 1
      funext j
      have hj : j.val < cls.kLog - cls.slotLog := j.isLt
      exact dite_eq_left (show cls.slotLog - 6 + j.val < cls.kLog - 6 by omega)
  simp only [hidComb, hlt, ↓reduceIte, Finset.mul_sum, mul_ite, mul_zero]
  rw [← sum_swap3]
  unfold hiddenFold
  rw [sparseFold_slots]
  refine Finset.sum_congr rfl fun σ _ => Finset.sum_congr rfl fun k _ => ?_
  rw [sum_hit h6 hsk (cls.units σ) (hunits σ) (word_lt R (τ σ) 0 k) (word_lt R (τ σ) cls.slotLog k) s.val s.isLt
    (fun w i => eqAt (fun j : Fin (cls.kLog - 6) => ts ⟨cls.kLog - 6 - 1 - j.val, by omega⟩) w *
      (A.lagS zc.z ⟨i % 64, Nat.mod_lt _ (by norm_num)⟩ *
        eqAt (fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩) (i / 64) * planeV R (τ σ) α k))]
  rw [slot_term h6 hsk _ _ _ _ (word_lt R (τ σ) 0 k) (word_lt R (τ σ) cls.slotLog k)]
  refine Eq.trans ?_ (skip_sum A R _ (τ σ) k _ s).symm
  rw [hU σ, hx, hy]
  rfl

end OracleFold

open OracleFold in
/-- **The hidden statement's folded comb is the public one's plus the hidden fold.** -/
theorem holoFold_eq (A : Arith F K) (hA : A.Correct) (cls : HoloClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (R : RegTable cls) (hunits : ∀ σ, ∀ u ∈ cls.units σ, u < 2 ^ (cls.kLog - cls.slotLog))
    (zc : ZcOut F cls.m) (α β : F) (ts : Fin (cls.kLog - 6) → F) (s : Fin 64) :
    foldAll (List.ofFn ts) (2 ^ cls.kLog) (comb A (R.toStatement τ regs) zc.z α β
        fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩) s.val =
      foldAll (List.ofFn ts) (2 ^ cls.kLog) (comb A (cls.pubStatement regs) zc.z α β
        fun j : Fin (cls.kLog - 6) => zc.ρ ⟨j.val, by have := cls.hk; omega⟩) s.val +
      hiddenFold A cls regs τ R zc α ts s := by
  have hK : 2 ^ cls.kLog = 2 ^ (cls.kLog - 6) * 64 := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel cls.hk6]
  have key : ∀ v : ℕ → F, foldAll (List.ofFn ts) (2 ^ cls.kLog) v s.val =
      ∑ w : Fin (2 ^ (cls.kLog - 6)), eqAt (fun j : Fin (cls.kLog - 6) => ts ⟨cls.kLog - 6 - 1 - j.val, by omega⟩)
        w.val * v (s.val + 64 * w.val) := fun v => by
    rw [hK]
    exact foldAll_eq_sum _ ts v s.val
  rw [key, key]
  simp only [comb_split A hA cls regs τ R, mul_add, Finset.sum_add_distrib]
  rw [hidFold_eq A hA cls regs τ R hunits zc α ts s]

end FlockC3.Fork
