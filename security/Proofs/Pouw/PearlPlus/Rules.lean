module

public import PouwPearlPlus.Rule
public import Proofs.Pouw.PearlC.CoreOps
public import Proofs.Pouw.PearlPlus.Words

/-!
# Pearl+'s bridge, part (c): the check's per-row rules and inert-work replay are the game's

`Rule.rowsFailure` decides each opened row by `row_passes` (the floor on the exact product ρ·α, and `live_row` on the
exact statistic), and `Rule.capCheck` the inert-work bound by `word_debit`, a word's flagged atoms and its later
promotions into a +0 total. The game reads `PearlCSem.RowOK` (the FMA floor `ρ·α − 1 ≥ 0` and `LiveRow` at δ = 1 under
`Stat.every8`) and the tile's inert work `tileDebitRev1`, on `Skip.unitFlags` and `laterZeroPromos`.

`rowsAreTheRules` proves the rules from the row statistics (`StatsAreTheRow`) and ρ's finiteness (`StatsRhoIsFinite`);
`rowPasses_iff` is the rules on one row given both. The floor agrees because ρ and α are finite FP32 words with
`0 ≤ ρ·α ≤ 113` (`RowCircuit.rho_alpha_le`): `ρ·α − 1`, when nonzero, is at least 2⁻⁴⁸ in magnitude, a
product of two 24-bit significands (`ra_gap`), far above FP32's least subnormal, so the FMA's rounding keeps its sign
(`floor_nonneg_iff`). Liveness agrees because `sq300` is a word's square times 2³⁰⁰ exactly (`sq300_val`) and the check
sums every 8th position as `Stat.every8` does (`liveRow_iff`).

The tile's cap is not stated here: the check's cap leaves out the game's forming term and reads its credited rows and
columns, where `TileGood` reads every column and `passRows`, so the cap's statement waits on the restated credit set and
acceptance event.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- **(c) The per-row rules are `RowOK`**: on a row of `k` finite FP32 words, `k` a multiple of 8 and at most `2^27`
(`StatsAreTheRow`'s domain), whose `s5_stats` are a value, `row_passes` on its ρ and α holds exactly when the semantics'
`RowOK` does. -/
def RowsAreTheRules : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (p k : ℕ) (W : Codes) (i : ℕ) (x : Array ℕ)
    (s ρw αw βw : ℕ), i < p → 0 < k → 8 ∣ k → k ≤ 2 ^ 27 → x.size = k →
    (∀ l < k, x[l]! = W i l ∧ W i l < 2 ^ 32 ∧ (wordQ (W i l)).isSome) →
    Words.s5Stats x = .ok (s, ρw, αw, βw) →
      (Rule.rowPasses ρw αw x = true ↔ (semH3 saltOf rootOf : PearlCSem Q R S).RowOK p k W i)

/-- **(c) `s5_stats`' ρ is a finite word** where the stats are a value: they refuse a non-finite partial sum or
`t = s + 4ρ`, and a NaN or infinite ρ makes `t` one. -/
def StatsRhoIsFinite : Prop :=
  ∀ (x : Array ℕ) (s ρw αw βw : ℕ), Words.s5Stats x = .ok (s, ρw, αw, βw) → (wordQ ρw).isSome

/-- **(c) A word's replayed debit is the game's**: where `word_debit` is a value on a word's two rows of codes, its flagged
atoms are `Skip.unitFlags`' and its later promotions into a +0 total are `laterZeroPromos`' at the H100 record. -/
def DebitIsTheReplay : Prop :=
  ∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Pouw.Fp8Atom.Code) (B : Matrix (Fin n) (Fin k) Pouw.Fp8Atom.Code)
    (i : Fin m) (j : Fin n) (f z : ℕ),
    Rule.wordDebit (Array.ofFn fun l => (A i l).val) (Array.ofFn fun l => (B j l).val) = .ok (f, z) →
      f = ((Finset.range (k / 32)).filter fun t => Pouw.PearlC.Skip.unitFlags A B i j t = true).card ∧
        z = Pouw.PearlC.laterZeroPromos Pouw.PearlC.devH100 A B i j

/-- **(c) The replay reads the semantics' codes**: under `WordsAreTheSemantics`' hypotheses, `word_debit` on the check's
A′ and B̃ codes of a word whose row passes the rules is `word_debit` on the semantics' noisy codes of that word
(`unitChain`). The check's code for a zero may be `0x80` where the semantics' is `0x00`, so this is stated on
`word_debit`'s result, not on the codes. -/
def ReplayReadsTheCodes : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S) (U : Pouw.PearlC.Workload) (u : ℕ)
    (act : Codes) (salt rootB : ByteArray) (wn : Words.FixedNoise) (rows cols : List ℕ) (ops : Words.Ops),
    bytesNat salt = saltOf s → bytesNat rootB = rootOf (U.layout.fixedId u) → (U.layout.shape u).k ≤ 2 ^ 16 →
    (∀ i ∈ rows, i < (U.layout.shape u).m) → (∀ j ∈ cols, j < (U.layout.shape u).n) →
    (∀ i ∈ rows, ∀ l < (U.layout.shape u).k, act i l < 2 ^ 32) →
    (∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.fixed (U.layout.fixedId u) j l < 2 ^ 32) →
    (∀ j ∈ cols, (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).n (U.layout.shape u).k
      (U.fixed (U.layout.fixedId u)) j) →
    Words.fixedNoise salt rootB (U.layout.shape u).k = .ok wn →
    Words.encode salt rootB wn u (U.layout.shape u).k rows (rows.map (rowBytesOf act (U.layout.shape u).k)) cols
      (cols.map (rowBytesOf (U.fixed (U.layout.fixedId u)) (U.layout.shape u).k)) = .ok ops →
    ∀ (x z : ℕ) (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n), rows[x]? = some i.val →
      cols[z]? = some j.val → (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).m (U.layout.shape u).k act i →
      Rule.wordDebit ops.a.codes[x]! ops.b.codes[z]! =
        Rule.wordDebit (Array.ofFn fun l => (((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).1 i l).val)
          (Array.ofFn fun l => (((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).2 j l).val)

/-! ## A word's fields, as `parts` reads them -/

theorem word_and_ff (w : ℕ) : w &&& 0xFF = w % 256 := by
  have h : (0xFF : ℕ) = 2 ^ 8 - 1 := by decide
  rw [h]; exact Nat.and_two_pow_sub_one_eq_mod w 8

theorem word_and_m (w : ℕ) : w &&& 0x7FFFFF = w % 2 ^ 23 := by
  have h : (0x7FFFFF : ℕ) = 2 ^ 23 - 1 := by decide
  rw [h]; exact Nat.and_two_pow_sub_one_eq_mod w 23

theorem parts_of_zero {w : ℕ} (h0 : (w >>> 23) % 256 = 0) :
    Rule.parts w = (w >>> 31 == 1, w % 2 ^ 23, -149) := by
  simp only [Rule.parts, word_and_ff, word_and_m, h0]; rfl

theorem parts_of_ne {w : ℕ} (h0 : (w >>> 23) % 256 ≠ 0) :
    Rule.parts w = (w >>> 31 == 1, w % 2 ^ 23 + 2 ^ 23, (((w >>> 23) % 256 : ℕ) : ℤ) - 150) := by
  simp only [Rule.parts, word_and_ff, word_and_m, beq_iff_eq, h0, ite_false]

theorem isNan_eq_false {w : ℕ} (he : (w >>> 23) % 256 ≠ 255) : Rule.isNan w = false := by
  simp [Rule.isNan, word_and_ff, he]

theorem isInf_eq_false {w : ℕ} (he : (w >>> 23) % 256 ≠ 255) : Rule.isInf w = false := by
  simp [Rule.isInf, word_and_ff, he]

theorem parts_sign_iff {w : ℕ} (h32 : w < 2 ^ 32) : (w >>> 31 == 1) = true ↔ (w >>> 31) % 2 = 1 := by
  have h31 : w >>> 31 < 2 := by rw [Nat.shiftRight_eq_div_pow]; omega
  rw [beq_iff_eq, Nat.mod_eq_of_lt h31]

/-- A finite word's value from `parts`: `±m·2^e`, a 24-bit significand at an exponent at least −149. -/
theorem parts_val {w : ℕ} {v : ℚ} (h32 : w < 2 ^ 32) (h : wordQ w = some v) :
    v = (if (Rule.parts w).1 = true then -1 else 1) * (((Rule.parts w).2.1 : ℕ) : ℚ) * 2 ^ (Rule.parts w).2.2 ∧
      (Rule.parts w).2.1 < 2 ^ 24 ∧ -149 ≤ (Rule.parts w).2.2 := by
  have he := RowCircuit.wordQ_exp h
  have hv := PeelExactProofs.wordQ_fields he
  rw [h, Option.some.injEq] at hv
  have hsg : (if (w >>> 31 == 1) = true then (-1 : ℚ) else 1) = if (w >>> 31) % 2 = 1 then -1 else 1 := by
    by_cases hs : (w >>> 31) % 2 = 1
    · rw [ite_eq_left ((parts_sign_iff h32).2 hs), ite_eq_left hs]
    · rw [ite_eq_right (fun h' => hs ((parts_sign_iff h32).1 h')), ite_eq_right hs]
  by_cases h0 : (w >>> 23) % 256 = 0
  · rw [parts_of_zero h0]
    refine ⟨?_, by dsimp only; omega, by dsimp only; omega⟩
    dsimp only
    rw [hsg, hv, h0, ite_eq_left (rfl : (0 : ℕ) = 0), Nat.zero_add, show max 0 1 = 1 by decide,
      show ((1 : ℕ) : ℤ) - 150 = -149 by norm_num]
    ring
  · rw [parts_of_ne h0]
    refine ⟨?_, by dsimp only; omega, by dsimp only; omega⟩
    dsimp only
    rw [hsg, hv, ite_eq_right h0, show max ((w >>> 23) % 256) 1 = (w >>> 23) % 256 by omega]
    push_cast
    ring

/-- A nonnegative `±m·2^e` is `m·2^e`. -/
theorem val_of_nonneg {s : Bool} {m : ℕ} {e : ℤ} {v : ℚ} (hv : v = (if s = true then -1 else 1) * (m : ℚ) * 2 ^ e)
    (hv0 : 0 ≤ v) : v = (m : ℚ) * 2 ^ e := by
  have h0 : 0 ≤ (m : ℚ) * 2 ^ e := mul_nonneg (Nat.cast_nonneg m) (zpow_pos (by norm_num) e).le
  cases s
  · rw [hv, ite_eq_right (by decide : ¬ (false = true)), one_mul]
  · rw [ite_eq_left (rfl : true = true)] at hv
    have hv' : v = -((m : ℚ) * 2 ^ e) := by rw [hv]; ring
    linarith

/-- A nonnegative `±m·2^e` with `m ≠ 0` has the sign bit clear. -/
theorem sign_of_nonneg {s : Bool} {m : ℕ} {e : ℤ} {v : ℚ} (hv : v = (if s = true then -1 else 1) * (m : ℚ) * 2 ^ e)
    (hv0 : 0 ≤ v) (hm : m ≠ 0) : s = false := by
  cases s
  · rfl
  · exfalso
    rw [ite_eq_left (rfl : true = true)] at hv
    have h0 : 0 < (m : ℚ) * 2 ^ e :=
      mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hm) (zpow_pos (by norm_num) e)
    have hv' : v = -((m : ℚ) * 2 ^ e) := by rw [hv]; ring
    linarith

/-! ## The floor -/

theorem floorOK_fin {rho alpha : ℕ} (hr : (rho >>> 23) % 256 ≠ 255) (ha : (alpha >>> 23) % 256 ≠ 255) :
    Rule.floorOK rho alpha =
      (if (Rule.parts rho).1 != (Rule.parts alpha).1 || (Rule.parts rho).2.1 * (Rule.parts alpha).2.1 == 0 then false
       else if (Rule.parts rho).2.2 + (Rule.parts alpha).2.2 ≥ 0 then true
       else decide ((Rule.parts rho).2.1 * (Rule.parts alpha).2.1 ≥
         2 ^ (-((Rule.parts rho).2.2 + (Rule.parts alpha).2.2)).toNat)) := by
  unfold Rule.floorOK
  rw [isNan_eq_false hr, isNan_eq_false ha, isInf_eq_false hr, isInf_eq_false ha]
  rfl

/-- **`floor_ok` is `ρ·α ≥ 1`** on two finite, nonnegative words. -/
theorem floorOK_iff {ρw αw : ℕ} {r a : ℚ} (hρ32 : ρw < 2 ^ 32) (hα32 : αw < 2 ^ 32) (hr : wordQ ρw = some r)
    (ha : wordQ αw = some a) (hr0 : 0 ≤ r) (ha0 : 0 ≤ a) : Rule.floorOK ρw αw = true ↔ 1 ≤ r * a := by
  rw [floorOK_fin (RowCircuit.wordQ_exp hr) (RowCircuit.wordQ_exp ha)]
  obtain ⟨hvr, -, -⟩ := parts_val hρ32 hr
  obtain ⟨hva, -, -⟩ := parts_val hα32 ha
  have hr' := val_of_nonneg hvr hr0
  have ha' := val_of_nonneg hva ha0
  generalize Rule.parts ρw = pr at hvr hr' ⊢
  generalize Rule.parts αw = pa at hva ha' ⊢
  obtain ⟨sr, mr, er⟩ := pr
  obtain ⟨sa, ma, ea⟩ := pa
  dsimp only at hvr hva hr' ha' ⊢
  have hra : r * a = ((mr * ma : ℕ) : ℚ) * 2 ^ (er + ea) := by
    rw [hr', ha', zpow_add₀ two_ne_zero, Nat.cast_mul]; ring
  by_cases hz : mr * ma = 0
  · have hc : (sr != sa || mr * ma == 0) = true := by simp [hz]
    rw [ite_eq_left hc, hra, hz, Nat.cast_zero, zero_mul]
    norm_num
  have hmr0 : mr ≠ 0 := fun h => hz (by rw [h, Nat.zero_mul])
  have hma0 : ma ≠ 0 := fun h => hz (by rw [h, Nat.mul_zero])
  obtain rfl := sign_of_nonneg hvr hr0 hmr0
  obtain rfl := sign_of_nonneg hva ha0 hma0
  have hc' : ¬ ((false != false || mr * ma == 0) = true) := by simp [hz]
  rw [ite_eq_right hc', hra]
  have hP : (1 : ℚ) ≤ ((mr * ma : ℕ) : ℚ) := by exact_mod_cast Nat.one_le_iff_ne_zero.2 hz
  by_cases hs : er + ea ≥ 0
  · rw [ite_eq_left hs]
    obtain ⟨t, ht⟩ : ∃ t : ℕ, er + ea = t := ⟨(er + ea).toNat, (Int.toNat_of_nonneg hs).symm⟩
    rw [ht, zpow_natCast]
    have h2 : (1 : ℚ) ≤ 2 ^ t := one_le_pow₀ (by norm_num)
    have h1 := mul_le_mul hP h2 zero_le_one (by positivity)
    rw [one_mul] at h1
    exact ⟨fun _ => h1, fun _ => rfl⟩
  · rw [ite_eq_right hs, decide_eq_true_iff]
    have hs' : 0 ≤ -(er + ea) := by omega
    obtain ⟨t, ht⟩ : ∃ t : ℕ, er + ea = -(t : ℤ) :=
      ⟨(-(er + ea)).toNat, by rw [Int.toNat_of_nonneg hs', neg_neg]⟩
    have h2t : (0 : ℚ) < 2 ^ t := by positivity
    rw [ht, neg_neg, Int.toNat_natCast, zpow_neg, zpow_natCast, ← div_eq_mul_inv, one_le_div h2t, ge_iff_le]
    constructor <;> intro h <;> exact_mod_cast h

/-- A natural below 2⁴⁸ times a power of two, when below 1, is at most `1 − 2⁻⁴⁸`. -/
theorem gap_of_parts {P : ℕ} {s : ℤ} (hP : (P : ℚ) < 2 ^ 48) (h : (P : ℚ) * 2 ^ s < 1) :
    (P : ℚ) * 2 ^ s ≤ 1 - 2 ^ (-48 : ℤ) := by
  have hG : (2 : ℚ) ^ (-48 : ℤ) = (2 ^ 48)⁻¹ := by norm_num
  rcases Nat.eq_zero_or_pos P with h0 | hP0
  · rw [h0, Nat.cast_zero, zero_mul, hG]; norm_num
  have hP1 : (1 : ℚ) ≤ P := by exact_mod_cast hP0
  rcases le_or_gt 0 s with hs | hs
  · obtain ⟨t, rfl⟩ : ∃ t : ℕ, s = t := ⟨s.toNat, (Int.toNat_of_nonneg hs).symm⟩
    rw [zpow_natCast] at h
    have h2 : (1 : ℚ) ≤ 2 ^ t := one_le_pow₀ (by norm_num)
    have h1 := mul_le_mul hP1 h2 zero_le_one (by positivity)
    linarith
  have hs' : 0 ≤ -s := by omega
  obtain ⟨t, rfl⟩ : ∃ t : ℕ, s = -(t : ℤ) := ⟨(-s).toNat, by rw [Int.toNat_of_nonneg hs', neg_neg]⟩
  have e : (2 : ℚ) ^ (-(t : ℤ)) = (2 ^ t)⁻¹ := by rw [zpow_neg, zpow_natCast]
  rw [e] at h ⊢
  have h2t : (0 : ℚ) < 2 ^ t := by positivity
  have hlt : P + 1 ≤ 2 ^ t := by
    have h' : (P : ℚ) < 2 ^ t := by rwa [← div_eq_mul_inv, div_lt_one h2t] at h
    have h'' : P < 2 ^ t := by exact_mod_cast h'
    omega
  have hle : (P : ℚ) + 1 ≤ 2 ^ t := by exact_mod_cast hlt
  rcases le_or_gt t 48 with ht | ht
  · have hinv : (2 : ℚ) ^ (-48 : ℤ) ≤ (2 ^ t)⁻¹ := by
      rw [hG]; exact inv_anti₀ h2t (pow_le_pow_right₀ (by norm_num) ht)
    calc (P : ℚ) * (2 ^ t)⁻¹ ≤ (2 ^ t - 1) * (2 ^ t)⁻¹ :=
          mul_le_mul_of_nonneg_right (by linarith) (by positivity)
      _ = 1 - (2 ^ t)⁻¹ := by rw [sub_mul, mul_inv_cancel₀ h2t.ne', one_mul]
      _ ≤ 1 - 2 ^ (-48 : ℤ) := by linarith
  · have ht' : 49 ≤ t := ht
    have hinv : ((2 : ℚ) ^ t)⁻¹ ≤ (2 ^ 49)⁻¹ := inv_anti₀ (by positivity) (pow_le_pow_right₀ (by norm_num) ht')
    calc (P : ℚ) * (2 ^ t)⁻¹ ≤ 2 ^ 48 * (2 ^ 49)⁻¹ := mul_le_mul hP.le hinv (by positivity) (by positivity)
      _ ≤ 1 - 2 ^ (-48 : ℤ) := by rw [hG]; norm_num

/-- **The gap**: a product of two finite, nonnegative words below 1 is at most `1 − 2⁻⁴⁸`. -/
theorem ra_gap {ρw αw : ℕ} {r a : ℚ} (hρ32 : ρw < 2 ^ 32) (hα32 : αw < 2 ^ 32) (hr : wordQ ρw = some r)
    (ha : wordQ αw = some a) (hr0 : 0 ≤ r) (ha0 : 0 ≤ a) (hlt : r * a < 1) : r * a ≤ 1 - 2 ^ (-48 : ℤ) := by
  obtain ⟨hvr, hmr, -⟩ := parts_val hρ32 hr
  obtain ⟨hva, hma, -⟩ := parts_val hα32 ha
  have hr' := val_of_nonneg hvr hr0
  have ha' := val_of_nonneg hva ha0
  have hP : (((Rule.parts ρw).2.1 * (Rule.parts αw).2.1 : ℕ) : ℚ) < 2 ^ 48 := by
    have h : (Rule.parts ρw).2.1 * (Rule.parts αw).2.1 < 2 ^ 48 :=
      lt_of_lt_of_le (Nat.mul_lt_mul'' hmr hma) (by norm_num)
    exact_mod_cast h
  have hra : r * a = (((Rule.parts ρw).2.1 * (Rule.parts αw).2.1 : ℕ) : ℚ) *
      2 ^ ((Rule.parts ρw).2.2 + (Rule.parts αw).2.2) := by
    rw [hr', ha', zpow_add₀ two_ne_zero, Nat.cast_mul]; ring
  rw [hra] at hlt ⊢
  exact gap_of_parts hP hlt

/-- **The FMA keeps the floor's sign**: with `0 ≤ ρ·α ≤ 113` and the gap below 1, `F32Fma(ρ, α, −1) ≥ 0` exactly when
`ρ·α ≥ 1`. -/
theorem floor_nonneg_iff {ρw αw : ℕ} {r a : ℚ} (hρ32 : ρw < 2 ^ 32) (hα32 : αw < 2 ^ 32) (hr : wordQ ρw = some r)
    (ha : wordQ αw = some a) (hra0 : 0 ≤ r * a) (hbd : r * a ≤ 113)
    (hgap : r * a < 1 → r * a ≤ 1 - 2 ^ (-48 : ℤ)) :
    0 ≤ Pouw.PearlC.wordVal (RowCircuit.Prim.eval RowCircuit.coreOps .fma [ρw, αw, 0xBF800000]) ↔ 1 ≤ r * a := by
  have h127 : (200 : ℚ) < 2 ^ 127 := by norm_num
  obtain ⟨fl, hfl, hfle⟩ := RowCircuit.floor_val Pouw.PearlC.CoreOps.pc8OpsAgree_coreOps hρ32 hα32 hr ha
    (abs_lt.2 ⟨by linarith, by linarith⟩)
  rw [RowCircuit.wordVal_of_wordQ hfl]
  have hE : (0 : ℚ) < 2 ^ (-150 : ℤ) := zpow_pos (by norm_num) _
  have hE2 : (2 : ℚ) ^ (-149 : ℤ) = 2 * 2 ^ (-150 : ℤ) := by norm_num
  have hEG : (2 : ℚ) ^ (-150 : ℤ) < 2 ^ (-48 : ℤ) * (1 - 1 / 16777216) := by norm_num
  have h24 : (2 : ℚ) ^ 24 = 16777216 := by norm_num
  rw [h24] at hfle
  constructor
  · intro hfl0
    by_contra hlt
    have hg := hgap (not_le.1 hlt)
    rw [abs_of_neg (by linarith : r * a - 1 < 0)] at hfle
    have h1 := (abs_le.1 hfle).2
    generalize (2 : ℚ) ^ (-150 : ℤ) = E at hE hEG h1
    generalize (2 : ℚ) ^ (-48 : ℤ) = G at hg hEG
    linarith
  · intro h1
    rw [abs_of_nonneg (by linarith : 0 ≤ r * a - 1)] at hfle
    have h2 := (abs_le.1 hfle).1
    have hflE : -((2 : ℚ) ^ (-150 : ℤ)) ≤ fl := by
      generalize (2 : ℚ) ^ (-150 : ℤ) = E at h2 ⊢
      linarith
    obtain ⟨m, hm⟩ := TileCircuit.wordQ_grid hfl
    rw [hm] at hflE ⊢
    have hm0 : (0 : ℚ) ≤ m := by
      by_contra hneg
      have hm1 : (m : ℚ) ≤ -1 := by
        have hm' : m < 0 := by exact_mod_cast not_le.1 hneg
        have hm'' : m ≤ -1 := by omega
        exact_mod_cast hm''
      have h3 : (m : ℚ) * 2 ^ (-149 : ℤ) ≤ -1 * 2 ^ (-149 : ℤ) :=
        mul_le_mul_of_nonneg_right hm1 (zpow_pos (by norm_num) _).le
      linarith
    exact mul_nonneg hm0 (zpow_pos (by norm_num) _).le

/-! ## Liveness -/

theorem sq300_eq (w : ℕ) : Rule.sq300 w = (Rule.parts w).2.1 * (Rule.parts w).2.1 *
    2 ^ (2 * (Rule.parts w).2.2 + 300).toNat := rfl

/-- **`sq300` is the square times 2³⁰⁰** on a finite word. -/
theorem sq300_val {w : ℕ} {v : ℚ} (h32 : w < 2 ^ 32) (h : wordQ w = some v) :
    (Rule.sq300 w : ℚ) = v ^ 2 * 2 ^ 300 := by
  obtain ⟨hv, -, hE⟩ := parts_val h32 h
  rw [sq300_eq]
  generalize Rule.parts w = pw at hv hE ⊢
  obtain ⟨s, m, e⟩ := pw
  dsimp only at hv hE ⊢
  have hnn : 0 ≤ 2 * e + 300 := by omega
  have h2a : ((2 ^ (2 * e + 300).toNat : ℕ) : ℚ) = (2 : ℚ) ^ (2 * e + 300) := by
    rw [Nat.cast_pow, Nat.cast_ofNat, ← zpow_natCast, Int.toNat_of_nonneg hnn]
  have h2b : (2 : ℚ) ^ (2 * e + 300) = 2 ^ e * 2 ^ e * 2 ^ (300 : ℕ) := by
    rw [show 2 * e + 300 = e + e + ((300 : ℕ) : ℤ) by push_cast; ring, zpow_add₀ two_ne_zero,
      zpow_add₀ two_ne_zero, zpow_natCast]
  rw [Nat.cast_mul, Nat.cast_mul, h2a, h2b, hv]
  generalize (2 : ℚ) ^ (300 : ℕ) = P
  cases s
  · rw [ite_eq_right (by decide : ¬ (false = true))]; ring
  · rw [ite_eq_left (rfl : true = true)]; ring

theorem foldl_count (x : Array ℕ) (q : ℕ → Prop) [DecidablePred q] :
    x.foldl (fun d w => if q w then d + 1 else d) 0 = ∑ l ∈ Finset.range x.size, if q x[l]! then 1 else 0 :=
  Array.foldl_induction (motive := fun i d => d = ∑ l ∈ Finset.range i, if q x[l]! then 1 else 0) (by simp)
    (fun i d hd => by
      rw [Finset.sum_range_succ, ← hd, getElem!_pos x i.1 i.2, Fin.getElem_fin]
      by_cases h : q x[i.1]
      · rw [ite_eq_left h, ite_eq_left h]
      · rw [ite_eq_right h, ite_eq_right h, Nat.add_zero])

theorem foldl_range_add (g : ℕ → ℕ) (n c : ℕ) :
    (List.range n).foldl (fun s i => s + g i) c = c + ∑ i ∈ Finset.range n, g i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.foldl_append, ih, Finset.sum_range_succ]; simp [Nat.add_assoc]

theorem liveRow_eq (x : Array ℕ) : Rule.liveRow x = decide (64 * (∑ l ∈ Finset.range x.size,
    if Rule.sq300 x[l]! * ((x.size + 7) / 8) ≥ 64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), Rule.sq300 x[8 * j]!
    then 1 else 0) ≤ x.size) := by
  unfold Rule.liveRow
  simp only [foldl_range_add, Nat.zero_add, foldl_count]

/-- **`live_row` is `LiveRow`** at δ = 1 under `Stat.every8`, on a row of finite words. -/
theorem liveRow_iff {x : Array ℕ} (hk : 0 < x.size) (hx : ∀ l < x.size, x[l]! < 2 ^ 32 ∧ (wordQ x[l]!).isSome) :
    Rule.liveRow x = true ↔ LiveRow Params.pi.σ (rowVals x.size x) Stat.every8 := by
  have hsq : ∀ l < x.size, (Rule.sq300 x[l]! : ℚ) = (wordQ x[l]!).getD 0 ^ 2 * 2 ^ 300 := by
    intro l hl
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (hx l hl).2
    rw [sq300_val (hx l hl).1 hv, hv, Option.getD_some]
  have hK : (0 : ℚ) < (((x.size + 7) / 8 : ℕ) : ℚ) := by
    have : 0 < (x.size + 7) / 8 := by omega
    exact_mod_cast this
  have h300 : (0 : ℚ) < 2 ^ 300 := by positivity
  have himg : (Finset.range x.size).filter (fun l => l % 8 = 0) =
      (Finset.range ((x.size + 7) / 8)).image (fun j => 8 * j) := by
    ext l
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_image]
    constructor
    · rintro ⟨hl, h8⟩
      exact ⟨l / 8, by omega, by omega⟩
    · rintro ⟨j, hj, rfl⟩
      exact ⟨by omega, by omega⟩
  have hinj : Set.InjOn (fun j => 8 * j) ↑(Finset.range ((x.size + 7) / 8)) := by
    intro a _ b _ h
    have h' : 8 * a = 8 * b := h
    omega
  have hT : ∑ j ∈ Finset.range ((x.size + 7) / 8), (Rule.sq300 x[8 * j]! : ℚ) =
      (∑ l ∈ Finset.univ.filter (fun l : Fin x.size => l.val % 8 = 0), rowVals x.size x l ^ 2) * 2 ^ 300 := by
    calc ∑ j ∈ Finset.range ((x.size + 7) / 8), (Rule.sq300 x[8 * j]! : ℚ)
        = ∑ j ∈ Finset.range ((x.size + 7) / 8), (fun n => (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300) (8 * j) :=
          Finset.sum_congr rfl fun j hj => hsq (8 * j) (by have := Finset.mem_range.1 hj; omega)
      _ = ∑ l ∈ (Finset.range ((x.size + 7) / 8)).image (fun j => 8 * j),
            (fun n => (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300) l :=
          (Finset.sum_image (f := fun n => (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300) hinj).symm
      _ = ∑ l ∈ Finset.range x.size,
            if l % 8 = 0 then (fun n => (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300) l else 0 := by
          rw [← himg, Finset.sum_filter]
      _ = ∑ l : Fin x.size, if l.val % 8 = 0 then (fun n => (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300) l.val else 0 :=
          (Fin.sum_univ_eq_sum_range (fun n => if n % 8 = 0 then (wordQ x[n]!).getD 0 ^ 2 * 2 ^ 300 else 0)
            x.size).symm
      _ = ∑ l : Fin x.size, if l.val % 8 = 0 then rowVals x.size x l ^ 2 * 2 ^ 300 else 0 := rfl
      _ = ∑ l ∈ Finset.univ.filter (fun l : Fin x.size => l.val % 8 = 0), rowVals x.size x l ^ 2 * 2 ^ 300 := by
          rw [Finset.sum_filter]
      _ = _ := (Finset.sum_mul _ _ _).symm
  have key : ∀ l (hl : l < x.size), (Rule.sq300 x[l]! * ((x.size + 7) / 8) ≥
      64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), Rule.sq300 x[8 * j]!) ↔
      64 * Params.pi.σ ^ 2 * Stat.every8 x.size (rowVals x.size x) ≤ rowVals x.size x ⟨l, hl⟩ ^ 2 := by
    intro l hl
    have hcast : (Rule.sq300 x[l]! * ((x.size + 7) / 8) ≥
        64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), Rule.sq300 x[8 * j]!) ↔
        64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), (Rule.sq300 x[8 * j]! : ℚ) ≤
          (Rule.sq300 x[l]! : ℚ) * (((x.size + 7) / 8 : ℕ) : ℚ) := by
      rw [ge_iff_le]; constructor <;> intro h <;> exact_mod_cast h
    rw [hcast, hsq l hl, hT]
    show _ ↔ 64 * (1 : ℚ) ^ 2 * ((∑ l ∈ Finset.univ.filter (fun l : Fin x.size => l.val % 8 = 0),
      rowVals x.size x l ^ 2) / (((x.size + 7) / 8 : ℕ) : ℚ)) ≤ (wordQ x[l]!).getD 0 ^ 2
    generalize (∑ l ∈ Finset.univ.filter (fun l : Fin x.size => l.val % 8 = 0), rowVals x.size x l ^ 2) = S
    generalize (((x.size + 7) / 8 : ℕ) : ℚ) = K at hK ⊢
    generalize (wordQ x[l]!).getD 0 = Y
    rw [one_pow, mul_one, ← mul_div_assoc, div_le_iff₀ hK, ← mul_assoc, mul_right_comm (Y ^ 2)]
    exact ⟨fun h => le_of_mul_le_mul_right h h300, fun h => mul_le_mul_of_nonneg_right h h300.le⟩
  have hcount : (Finset.univ.filter fun l : Fin x.size =>
      64 * Params.pi.σ ^ 2 * Stat.every8 x.size (rowVals x.size x) ≤ rowVals x.size x l ^ 2).card =
      ∑ l ∈ Finset.range x.size, if Rule.sq300 x[l]! * ((x.size + 7) / 8) ≥
        64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), Rule.sq300 x[8 * j]! then 1 else 0 := by
    rw [Finset.card_filter, ← Fin.sum_univ_eq_sum_range (fun l => if Rule.sq300 x[l]! * ((x.size + 7) / 8) ≥
        64 * ∑ j ∈ Finset.range ((x.size + 7) / 8), Rule.sq300 x[8 * j]! then 1 else 0) x.size]
    refine Finset.sum_congr rfl fun l _ => ?_
    by_cases h : 64 * Params.pi.σ ^ 2 * Stat.every8 x.size (rowVals x.size x) ≤ rowVals x.size x l ^ 2
    · rw [ite_eq_left h, ite_eq_left ((key l.val l.isLt).2 h)]
    · rw [ite_eq_right h, ite_eq_right fun h' => h ((key l.val l.isLt).1 h')]
  rw [liveRow_eq, decide_eq_true_iff, LiveRow, hcount]

/-! ## The rules -/

theorem alpha8_finite (x : ℕ → ℕ) (k : ℕ) : ∃ a, wordQ (RowCircuit.alpha8 RowCircuit.coreOps x k) = some a := by
  obtain ⟨-, a, ha, -⟩ := RowCircuit.alpha_spec Pouw.PearlC.CoreOps.pc8OpsAgree_coreOps
    (t := RowCircuit.Prim.eval RowCircuit.coreOps .add [RowCircuit.rowMax RowCircuit.coreOps .fabs x (k - 1),
      RowCircuit.Prim.eval RowCircuit.coreOps .mul [0x40800000, RowCircuit.rowRho RowCircuit.coreOps .id x k]])
    (RowCircuit.eval_lt32 .add rfl _) RowCircuit.wordQ_c448 (by norm_num) (by norm_num) (by norm_num)
  exact ⟨a, ha⟩

theorem semH3_floor {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (k : ℕ) (X : Fin k → ℚ) :
    (semH3 saltOf rootOf : PearlCSem Q R S).floor k X =
      if (wordQ (RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k X) k)).isSome then
        Pouw.PearlC.wordVal (RowCircuit.floor8 RowCircuit.coreOps (RowCircuit.encRow k X) k) + 1 else 0 := rfl

/-- **The rules on a row** whose ρ is a finite word and whose ρ and α are `pc8`'s. -/
theorem rowPasses_iff {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) {p k : ℕ} {W : Codes} {i : ℕ}
    {x : Array ℕ} {ρw αw : ℕ} (hi : i < p) (hk : 0 < k) (hxk : x.size = k)
    (hx : ∀ l < k, x[l]! = W i l ∧ W i l < 2 ^ 32 ∧ (wordQ (W i l)).isSome) (hρS : (wordQ ρw).isSome)
    (hρ : ρw = RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k)
    (hα : αw = RowCircuit.alpha8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k) :
    Rule.rowPasses ρw αw x = true ↔ (semH3 saltOf rootOf : PearlCSem Q R S).RowOK p k W i := by
  have hxw : ∀ l < k, x[l]! < 2 ^ 32 ∧ (wordQ x[l]!).isSome := fun l hl => by
    obtain ⟨he, h32, hS'⟩ := hx l hl
    rw [he]; exact ⟨h32, hS'⟩
  have hX : ∀ h : i < p, wordsVal p k W ⟨i, h⟩ = rowVals k x := fun h => funext fun l => by
    show (wordQ (W i l)).getD 0 = (wordQ x[l.val]!).getD 0
    rw [(hx l.val l.isLt).1]
  obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 hρS
  obtain ⟨a, ha⟩ := alpha8_finite (RowCircuit.encRow k (rowVals k x)) k
  have haw : wordQ αw = some a := by rw [hα]; exact ha
  have hr' : wordQ (RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k) = some r := by
    rw [← hρ]; exact hr
  have hxe : ∀ l, RowCircuit.encRow k (rowVals k x) l < 2 ^ 32 ∧
      (wordQ (RowCircuit.encRow k (rowVals k x) l)).isSome :=
    RowCircuit.encRow_words (w := fun l => x[l]!) fun l hl => (hxw l hl).2
  obtain ⟨hr0, ha0, hbd⟩ := RowCircuit.rho_alpha_le Pouw.PearlC.CoreOps.pc8OpsAgree_coreOps hxe hr' ha
  have hρ32 : ρw < 2 ^ 32 := by rw [hρ]; exact RowCircuit.eval_lt32 .sqrt rfl _
  have hα32 : αw < 2 ^ 32 := by rw [hα]; exact RowCircuit.eval_lt32 .div rfl _
  have hfl8 : RowCircuit.floor8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k =
      RowCircuit.Prim.eval RowCircuit.coreOps .fma [ρw, αw, 0xBF800000] := by
    subst hρ hα; rfl
  have hc : (wordQ (RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k)).isSome =
      true := by
    rw [← hρ]; exact hρS
  have hfloor : ∀ h : i < p, (semH3 saltOf rootOf : PearlCSem Q R S).floor k (wordsVal p k W ⟨i, h⟩) =
      Pouw.PearlC.wordVal (RowCircuit.Prim.eval RowCircuit.coreOps .fma [ρw, αw, 0xBF800000]) + 1 := by
    intro h
    rw [hX h, semH3_floor, ite_eq_left hc, hfl8]
  have hF : Rule.floorOK ρw αw = true ↔
      1 ≤ Pouw.PearlC.wordVal (RowCircuit.Prim.eval RowCircuit.coreOps .fma [ρw, αw, 0xBF800000]) + 1 := by
    rw [floorOK_iff hρ32 hα32 hr haw hr0 ha0, ← floor_nonneg_iff hρ32 hα32 hr haw (mul_nonneg hr0 ha0) hbd
      (ra_gap hρ32 hα32 hr haw hr0 ha0)]
    constructor <;> intro h <;> linarith
  have hL : Rule.liveRow x = true ↔ LiveRow Params.pi.σ (rowVals k x) Stat.every8 := by
    subst hxk; exact liveRow_iff hk hxw
  unfold PearlCSem.RowOK
  rw [Rule.rowPasses, Bool.and_eq_true]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun l hl => (hx l hl).2.2, fun h => ⟨?_, ?_⟩⟩
    · rw [hX h]; exact hL.1 h2
    · rw [hfloor h]; exact hF.1 h1
  · rintro ⟨-, h⟩
    obtain ⟨h1, h2⟩ := h hi
    rw [hX hi] at h1
    rw [hfloor hi] at h2
    exact ⟨hF.2 h2, hL.2 h1⟩

/-- **(c) The per-row rules, proved** from the row statistics and ρ's finiteness. -/
theorem rowsAreTheRules (hS : StatsAreTheRow) (hF : StatsRhoIsFinite) : RowsAreTheRules := by
  intro Q R S saltOf rootOf p k W i x s ρw αw βw hi hk _hk8 _hk27 hxk hx hst
  have _hxw : ∀ l < k, x[l]! < 2 ^ 32 ∧ (wordQ x[l]!).isSome := fun l hl => by
    obtain ⟨he, h32, hS'⟩ := hx l hl
    rw [he]; exact ⟨h32, hS'⟩
  obtain ⟨hρ, hα, -⟩ : ρw = RowCircuit.rowRho RowCircuit.coreOps .id (RowCircuit.encRow k (rowVals k x)) k ∧
      αw = RowCircuit.alpha8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k ∧
      βw = RowCircuit.beta8 RowCircuit.coreOps (RowCircuit.encRow k (rowVals k x)) k := by
    apply hS k x s ρw αw βw <;> assumption
  exact rowPasses_iff saltOf rootOf hi hk hxk hx (hF x s ρw αw βw hst) hρ hα

end Pouw.PearlPlus.Bridge
