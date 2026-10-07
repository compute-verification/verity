import Definitions.Pouw.PearlC.Accum
import Definitions.Pouw.PearlC.PeelExactP
import Proofs.Pouw.PearlC.PeelExactProofs

namespace Pouw.PearlC.Accum

open Pouw.Fp8Atom Pouw.PearlC Matrix

variable {m n k : Nat}

theorem groupWordP_hopper : groupWordP hopperE4m3K32 = groupWord := rfl

theorem totalP_hopper : totalP hopperE4m3K32 = total := rfl

theorem chainP_hopper (G : Nat) : chainP hopperE4m3K32 G = chain G := rfl

theorem ctChainP_hopper (G : Nat) : @ctChainP m n k hopperE4m3K32 G = @ctChain m n k G := rfl

/-- Π's chain (`G = 4`) is the chain twin at `hopperE4m3K32`. -/
theorem chainP_hopper_four : chainP hopperE4m3K32 4 = chain 4 := rfl

/-- Π's `C̃` (`G = 4`) is the chain twin's at `hopperE4m3K32`. -/
theorem ctChainP_hopper_four : @ctChainP m n k hopperE4m3K32 4 = @ctChain m n k 4 := rfl

/-- The H100 record's ticket chain is Π's `C̃`. -/
theorem chainAt_hopper : @chainAt m n k hopperE4m3K32 4 = @ctChain m n k 4 := rfl

end Pouw.PearlC.Accum

namespace Pouw.PearlC.AccumProofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.Accum Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP
  Matrix

variable {m n k : Nat}

theorem toChunks_zero_nil {α : Type} : ([] : List α).toChunks 0 = [] := rfl

theorem toChunks_zero_cons {α : Type} (x : α) (xs : List α) : (x :: xs).toChunks 0 = [x :: xs] := rfl

theorem promote_zero (w : Nat) : promote (some 0) w = wordQ w := by
  unfold promote
  rcases h : wordQ w with _ | x
  · rfl
  · simp only
    rw [rnAdd_exact 0 x (by rw [zero_add]; exact wordQ_representable h), zero_add]

theorem chainP_zero (p : Pipeline) (atoms : List AtomIn) : chainP p 0 atoms = pureChain p atoms := by
  cases atoms with
  | nil => exact Freeze.wordQ_zero.symm
  | cons x xs =>
    show totalP p [x :: xs] = _
    exact promote_zero _

theorem ctChainP_zero_eq_pure (p : Pipeline) : @ctChainP m n k p 0 = @ctPureChain m n k p := by
  funext Aq Bq i j
  exact chainP_zero p _

theorem chainAt_zero_eq_pure (p : Pipeline) : @chainAt m n k p 0 = @ctPureChain m n k p := ctChainP_zero_eq_pure p

/-! ## The words a step writes -/

/-- A term as FP32 carries it: zero, or a magnitude below `2^24` at exponent `−126` or more. -/
def NormT (t : Term) : Prop := t.mag = 0 ∨ (t.mag < 2 ^ 24 ∧ -126 ≤ t.exp)

theorem bitLength_spec (a : Nat) : a < 2 ^ bitLength a := by
  unfold bitLength
  split_ifs with h
  · subst h; norm_num
  · exact Nat.lt_log2_self

theorem normalize_normT (w : Nat) (E total : Int) : NormT (normalize w E total) := by
  unfold normalize
  split_ifs with h0
  · exact Or.inl rfl
  dsimp only
  set bl := bitLength total.natAbs
  have hbl := bitLength_spec total.natAbs
  have h1 : (if w < bl then total.natAbs >>> (bl - w) else total.natAbs <<< (w - bl)) < 2 ^ w := by
    split_ifs with hw
    · rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]
      exact lt_of_lt_of_le hbl (Nat.pow_le_pow_right (by norm_num) (by omega))
    · rw [Nat.shiftLeft_eq, show w = bl + (w - bl) by omega, pow_add, Nat.add_sub_cancel_left]
      exact Nat.mul_lt_mul_of_pos_right hbl (by positivity)
  set M := if w < bl then total.natAbs >>> (bl - w) else total.natAbs <<< (w - bl)
  have h2 : (if E + (bl : ℤ) - w < -126 then M >>> (-126 - (E + (bl : ℤ) - w)).toNat else M) < 2 ^ w := by
    split_ifs
    · exact lt_of_le_of_lt (Nat.shiftRight_le _ _) h1
    · exact h1
  set M' := if E + (bl : ℤ) - w < -126 then M >>> (-126 - (E + (bl : ℤ) - w)).toNat else M
  have h3 : fromAdder w M' < 2 ^ 24 := by
    unfold fromAdder
    split_ifs with hw
    · rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add,
        show 24 + (w - 24) = w by omega]
      exact h2
    · rw [Nat.shiftLeft_eq, show (24 : Nat) = w + (24 - w) by omega, pow_add, Nat.add_sub_cancel_left]
      exact Nat.mul_lt_mul_of_pos_right h2 (by positivity)
  split_ifs with h4
  all_goals first | exact Or.inl rfl | exact Or.inr ⟨h3, by dsimp only; omega⟩

theorem fp32Value_normT (acc : Nat) : ∀ t, fp32Value acc = .fin t → NormT t := by
  intro t h
  unfold fp32Value at h
  dsimp only at h
  split_ifs at h with h1 h2 h3 <;> simp only [Value.fin.injEq] at h <;> subst h
  · exact Or.inl rfl
  · exact Or.inr ⟨lt_trans (Nat.mod_lt _ (by norm_num)) (by norm_num), le_refl _⟩
  · refine Or.inr ⟨Nat.or_lt_two_pow (by decide) (lt_trans (Nat.mod_lt _ (by norm_num)) (by norm_num)), ?_⟩
    dsimp only; omega

theorem groupStepTotal_normT (w : Nat) (z : Int) (v : Value) (ps : List Value) :
    ∀ t, groupStepTotal w z v ps = .fin t → NormT t := by
  intro t h
  unfold groupStepTotal at h
  dsimp only at h
  split_ifs at h
  simp only [Value.fin.injEq] at h
  subst h
  exact normalize_normT _ _ _

theorem fold_normT (p : Pipeline) :
    ∀ (gs : List Nat) (v : Value) (ab : List (Code × Code)), (∀ t, v = .fin t → NormT t) →
      ∀ t, p.fold gs v ab = .fin t → NormT t
  | [], _, _, hv => hv
  | _ :: gs, _, _, _ => fold_normT p gs _ _ (groupStepTotal_normT _ _ _ _)

theorem or_fields (s e r : Nat) (he : e < 2 ^ 8) (hr : r < 2 ^ 23) :
    (s <<< 31) ||| (e <<< 23) ||| r = s * 2 ^ 31 + e * 2 ^ 23 + r := by
  have h1 : (s <<< 31) ||| (e <<< 23) = (s <<< 8 + e) <<< 23 := by
    rw [show (31 : ℕ) = 8 + 23 from rfl, Nat.shiftLeft_add, ← Nat.shiftLeft_or_distrib,
      ← Nat.shiftLeft_add_eq_or_of_lt he]
  rw [h1, ← Nat.shiftLeft_add_eq_or_of_lt hr, Nat.shiftLeft_eq, Nat.shiftLeft_eq]
  ring

theorem and_mask_lt (x : Nat) : x &&& 0x7FFFFF < 2 ^ 23 :=
  lt_of_le_of_lt Nat.and_le_right (by norm_num)

theorem packTerm_fin (t : Term) (h0 : t.mag ≠ 0) (h1 : ¬ 127 < t.exp)
    (he : (t.exp + 127 - if t.mag &&& 1 <<< 23 = 0 then 1 else 0).toNat < 2 ^ 8) :
    packTerm t = (if t.neg then 1 else 0) * 2 ^ 31 +
      (t.exp + 127 - if t.mag &&& 1 <<< 23 = 0 then 1 else 0).toNat * 2 ^ 23 + (t.mag &&& 0x7FFFFF) := by
  unfold packTerm
  rw [ite_eq_right h0, ite_eq_right h1]
  exact or_fields _ _ _ he (and_mask_lt _)

theorem packTerm_exp (t : Term) (h1 : ¬ 127 < t.exp) :
    (t.exp + 127 - if t.mag &&& 1 <<< 23 = 0 then 1 else 0).toNat < 2 ^ 8 := by
  split_ifs <;> omega

theorem packValue_lt (v : Value) : packValue v < 2 ^ 32 := by
  rcases v with _ | s | t
  · decide
  · cases s <;> decide
  · show packTerm t < _
    by_cases h0 : t.mag = 0
    · unfold packTerm; rw [ite_eq_left h0]; positivity
    by_cases h1 : 127 < t.exp
    · unfold packTerm; rw [ite_eq_right h0, ite_eq_left h1]; split <;> decide
    rw [packTerm_fin t h0 h1 (packTerm_exp t h1)]
    have := and_mask_lt t.mag
    have := packTerm_exp t h1
    have : (if t.neg = true then 1 else 0) ≤ 1 := by split <;> norm_num
    omega

theorem mag_bit23 {x : Nat} (hx : x < 2 ^ 24) (h : x &&& 1 <<< 23 = 0) : x < 2 ^ 23 := by
  rw [Nat.one_shiftLeft, Nat.and_two_pow] at h
  by_contra hc
  have : x.testBit 23 = true := by
    rw [Nat.testBit_eq_decide_div_mod_eq]
    simp only [decide_eq_true_eq]
    omega
  simp [this] at h

theorem packValue_ne_negZero (v : Value) (hv : ∀ t, v = .fin t → NormT t) : packValue v ≠ 0x80000000 := by
  rcases v with _ | s | t
  · decide
  · cases s <;> decide
  · have hN := hv t rfl
    show packTerm t ≠ _
    by_cases h0 : t.mag = 0
    · unfold packTerm; rw [ite_eq_left h0]; decide
    by_cases h1 : 127 < t.exp
    · unfold packTerm; rw [ite_eq_right h0, ite_eq_left h1]; split <;> decide
    rcases hN with hN | ⟨hm, he⟩
    · exact absurd hN h0
    rw [packTerm_fin t h0 h1 (packTerm_exp t h1)]
    have hs : (if t.neg = true then 1 else 0) ≤ 1 := by split <;> norm_num
    have hef := packTerm_exp t h1
    intro hw
    by_cases hb : t.mag &&& 1 <<< 23 = 0
    · have h23 := mag_bit23 hm hb
      have hr : t.mag &&& 0x7FFFFF = t.mag := by
        rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt h23]
      rw [hr, ite_eq_left hb] at hw
      rw [ite_eq_left hb] at hef
      omega
    · have := and_mask_lt t.mag
      rw [ite_eq_right hb] at hw hef
      omega

theorem step_lt (p : Pipeline) (acc : Nat) (a b : List Code) : p.step acc a b < 2 ^ 32 := packValue_lt _

theorem step_ne_negZero (p : Pipeline) (acc : Nat) (a b : List Code) : p.step acc a b ≠ 0x80000000 :=
  packValue_ne_negZero _ (fold_normT p _ _ _ (fp32Value_normT acc))

/-! ## `f32Word` of a word's value -/

theorem word_split {w : Nat} (hw : w < 2 ^ 32) :
    w = (w >>> 31) % 2 * 2 ^ 31 + (w >>> 23) % 256 * 2 ^ 23 + w % 2 ^ 23 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
  norm_num at hw ⊢
  omega

/-- **`f32Word` of a finite nonzero FP32 value is its word**, on the fields: sign `s`, biased exponent `e < 255`,
mantissa `mm`. -/
theorem f32Word_fields {s e mm : ℕ} (hs : s < 2) (he : e < 255) (hm : mm < 2 ^ 23)
    (h0 : (if e = 0 then 0 else 2 ^ 23) + mm ≠ 0) :
    f32Word ((if s = 1 then (-1 : ℚ) else 1) *
      ((((if e = 0 then 0 else 2 ^ 23) + mm : ℕ) : ℚ) * (2 : ℚ) ^ (((max e 1 : ℕ) : ℤ) - 150))) =
      s * 2 ^ 31 + e * 2 ^ 23 + mm := by
  set N : ℕ := (if e = 0 then 0 else 2 ^ 23) + mm with hN
  set E' : ℤ := ((max e 1 : ℕ) : ℤ) - 150 with hE'
  have hpos : 0 < (N : ℚ) * 2 ^ E' := mul_pos (by exact_mod_cast Nat.pos_of_ne_zero h0) (two_zpow_pos _)
  have hsgn : ((if s = 1 then (-1 : ℚ) else 1) * ((N : ℚ) * 2 ^ E') < 0) ↔ s = 1 := by
    split_ifs with h1
    · simp only [h1, iff_true]; linarith
    · simp only [one_mul, h1, iff_false, not_lt]; exact hpos.le
  have habs : |(if s = 1 then (-1 : ℚ) else 1) * ((N : ℚ) * 2 ^ E')| = (N : ℚ) * 2 ^ E' := by
    rw [abs_mul, abs_of_pos hpos]
    split_ifs <;> simp
  have hbin : binade ((N : ℚ) * 2 ^ E') = (N.log2 : ℤ) + E' := H1T.Proofs.binade_nat_mul h0 E'
  have hNlt : N < 2 ^ 24 := by rw [hN]; split_ifs <;> omega
  obtain ⟨Eb, hEb, hEbE, hNe⟩ : ∃ Eb : ℤ, max ((N.log2 : ℤ) + E') (-126) = Eb ∧ Eb - 23 = E' ∧
      (Eb = -126 ∧ N = mm ∧ e = 0 ∨ Eb = (e : ℤ) - 127 ∧ N = 2 ^ 23 + mm ∧ e ≠ 0) := by
    by_cases e0 : e = 0
    · have hNm : N = mm := by rw [hN, ite_eq_left e0, zero_add]
      have hl : N.log2 < 23 := (Nat.log2_lt h0).2 (by omega)
      have hE1 : E' = -149 := by rw [hE', e0]; rfl
      exact ⟨-126, by omega, by omega, Or.inl ⟨rfl, hNm, e0⟩⟩
    · have hNm : N = 2 ^ 23 + mm := by rw [hN, ite_eq_right e0]
      have hl : N.log2 = 23 :=
        le_antisymm (Nat.lt_succ_iff.1 ((Nat.log2_lt h0).2 hNlt)) ((Nat.le_log2 h0).2 (by omega))
      have hE1 : E' = (e : ℤ) - 150 := by rw [hE', max_eq_left (by omega : 1 ≤ e)]
      exact ⟨(e : ℤ) - 127, by omega, by omega, Or.inr ⟨rfl, hNm, e0⟩⟩
  have hdiv : (N : ℚ) * 2 ^ E' / 2 ^ (Eb - 23) = ((N : ℤ) : ℚ) := by
    rw [hEbE, mul_div_cancel_right₀ _ (two_zpow_pos _).ne']; norm_cast
  have hN24 : ¬ N = 2 ^ 24 := by omega
  have hE127 : ¬ 127 < Eb := by omega
  unfold f32Word
  dsimp only
  rw [habs, ite_eq_right hpos.ne', hbin, hEb, hdiv, roundEven_intCast, Int.toNat_natCast]
  simp only [hN24, hE127, ↓reduceIte]
  rw [if_congr hsgn rfl rfl]
  rcases hNe with ⟨hE, hNm, e0⟩ | ⟨hE, hNm, e0⟩
  · rw [ite_eq_left (by omega : N < 2 ^ 23)]
    split_ifs <;> omega
  · rw [ite_eq_right (by omega : ¬ N < 2 ^ 23)]
    split_ifs <;> omega

theorem finite?_exp {w : Nat} (hf : finite? w = some w) : (w >>> 23) % 256 ≠ 255 := by
  unfold finite? at hf
  split_ifs at hf with h
  exact h

/-- **`f32Word` of a finite nonzero word's value is the word.** -/
theorem f32Word_word {w : Nat} (hw : w < 2 ^ 32) (he : (w >>> 23) % 256 ≠ 255) {y : ℚ} (hy : wordQ w = some y)
    (h0 : y ≠ 0) : f32Word y = w := by
  rw [PeelExactProofs.wordQ_fields he] at hy
  obtain rfl := Option.some.inj hy
  have hN : (if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 ≠ 0 := by
    intro h; apply h0; rw [h]; simp
  have he' : (w >>> 23) % 256 < 255 := by have := Nat.mod_lt (w >>> 23) (by norm_num : 256 > 0); omega
  rw [f32Word_fields (Nat.mod_lt _ (by norm_num)) he' (Nat.mod_lt _ (by norm_num)) hN]
  exact (word_split hw).symm

/-! ## `G = 0` on words -/

/-- A word as a step leaves it: below `2^32`, not `−0`, and finite. -/
def StepWord (w : Nat) : Prop := w < 2 ^ 32 ∧ w ≠ 0x80000000 ∧ finite? w = some w

theorem wordQ_zero_word {w : Nat} (hw : StepWord w) (h : wordQ w = some 0) : w = 0 := by
  obtain ⟨hlt, hn, hf⟩ := hw
  have he := finite?_exp hf
  have hsplit := word_split hlt
  rw [PeelExactProofs.wordQ_fields he] at h
  have h' := Option.some.inj h
  have hs2 : (w >>> 31) % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hm2 : w % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  have hN : (if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 = 0 := by
    rcases mul_eq_zero.1 h' with hs | hp
    · split_ifs at hs <;> norm_num at hs
    · rcases mul_eq_zero.1 hp with hc | hz
      · exact_mod_cast hc
      · exact absurd hz (two_zpow_pos _).ne'
  split_ifs at hN with he0
  · omega
  · omega

theorem promoteWord_zero {w : Nat} (hw : StepWord w) : promoteWord? 0 w = some w := by
  obtain ⟨y, hy⟩ : ∃ y, wordQ w = some y := (PeelExactProofs.finite?_eq_some hw.2.2).2
  unfold promoteWord? addWord?
  rw [Freeze.wordQ_zero, hy]
  dsimp only
  have hsb : signBit 0 = false := by decide
  rw [hsb, Bool.and_false, Bool.false_and, zero_add]
  unfold rnWord
  by_cases h0 : y = 0
  · subst h0
    rw [wordQ_zero_word hw hy]
    decide
  · rw [ite_eq_right h0, f32Word_word hw.1 (finite?_exp hw.2.2) hy h0, hw.2.2, Option.map_some,
      ite_eq_right hw.2.1]

theorem pureWordP?_step (p : Pipeline) :
    ∀ (atoms : List AtomIn) (c w : Nat), StepWord c →
      atoms.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) c = some w → StepWord w
  | [], c, w, hc, h => by simp only [List.foldlM_nil, pure, Option.some.injEq] at h; exact h ▸ hc
  | x :: xs, c, w, _, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨c', hc', h⟩ := Option.bind_eq_some_iff.1 h
    refine pureWordP?_step p xs c' w ?_ h
    obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some hc'
    exact ⟨step_lt _ _ _ _, step_ne_negZero _ _ _ _, hc'⟩

/-- **`G = 0` on words**: the chain promoted once from `+0` is the pure chain, since that promotion returns the word. -/
theorem chainWordP?_zero (p : Pipeline) (atoms : List AtomIn) : chainWordP? p 0 atoms = pureWordP? p atoms := by
  cases atoms with
  | nil => rfl
  | cons x xs =>
    show (windowWordP? p (x :: xs)).bind (promoteWord? 0) >>= pure = pureWordP? p (x :: xs)
    change (pureWordP? p (x :: xs)).bind (promoteWord? 0) >>= pure = pureWordP? p (x :: xs)
    rcases h : pureWordP? p (x :: xs) with _ | w
    · rfl
    · have hw := pureWordP?_step p (x :: xs) 0 w ⟨by norm_num, by decide, by decide⟩ h
      simp only [Option.bind_some, promoteWord_zero hw]
      rfl

/-- **U at `G = 0`** is U with pure chains. -/
theorem peelExactP_zero_eq_pure (p : Pipeline) : @peelExactP m n k p 0 = @peelExactPureP m n k p := by
  funext Aq Bq FA FB beA beB i j
  unfold peelExactP peelExactPureP
  simp only [chainWordP?_zero]

theorem peelExactValP_zero_eq_pure (p : Pipeline) : @peelExactValP m n k p 0 = @peelExactPureValP m n k p := by
  funext Aq Bq FA FB beA beB i j
  unfold peelExactValP peelExactPureValP
  rw [peelExactP_zero_eq_pure]

end Pouw.PearlC.AccumProofs
