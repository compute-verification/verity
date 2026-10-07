import Proofs.Pouw.Fp8Atom.H1TSliceProofs
import Proofs.Pouw.PearlC.AccumProofs
import Proofs.Pouw.PearlC.FreezeP

/-!
# `pearl-c-sm120-v2`: the pure chain's lone skips and its identities (staging)

Without promotion (`G = 0`) the chain is `Accum.pureChain p`, one fold of `p`'s step over all of a word's atoms from
`+0`, and the debit is its lone skips (`SkipP.pureFlagsP`).

* **The device record's flags** (`flagsAt p G`, `unitFlagsAt p G`): `pureFlagsP p` at `G = 0`, `wordFlagsP p G`
  otherwise. At the H100 record they are `Skip.wordFlags` and `Skip.unitFlags` by `rfl` (`flagsAt_hopper`,
  `unitFlagsAt_hopper`).
* **Soundness and completeness** (`pureFlags_iff`): `pureFlagsP` flags atom `t` exactly when `t < k/32` and skipping it
  alone leaves the pure chain's output unchanged, which is `FreezeP.skipOneP` on the chain's one group
  (`pureFlags_iff_skipOne`) and `FreezeP.absorbedP` there (`pureFlags_iff_absorbed`, from `skipOne_iffP`). While the
  output is finite, unchanged output is unchanged word (`pureFlags_iff_word`). The skip resumes from the atom's incoming
  word (`pureSkip_resume`), so an identity atom is flagged (`pureFlags_of_identity`). The unit's form, the ticket
  `chainAt p 0` (`unitFlagsAt_zero_iff`), and `debitDev`'s count and `needed` at `G = 0` (`pureDebit_eq`,
  `pureNeeded_iff`).
* **Identities on sm_120** (`sm120_no_identity`, `sm120_pure_no_identity`): the pure chain's words stay `+0` or normal
  at exponent at least `−11 − w`, each at most the absolute mass `Σ |a·b|` of the atoms before it
  (`pure_inWordW`). By (a) (`FreezeP.identity_cellW`) an atom of exact sum `S`, products at most `M` and incoming mass
  `C` is not an identity once `9·2^−23·C + 2^−14·M < |S|`; with every product at most `P`, once
  `(9·2^−23·32t + 2^−14)·P < |S|` (`sm120_pure_no_identity_of_le`). No bound on the products alone excludes an
  identity: an atom whose products cancel is one on any word (`sm120_cancel_identity`, products `±1` on `1.0`).
-/

namespace Pouw.PearlC.FreezePPure

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Skip
  Pouw.PearlC.Freeze Pouw.PearlC.Accum Pouw.PearlC.SkipP Pouw.PearlC.FreezeP

/-! ## The device record's flags -/

/-- The H100 record's word flags are `flagsAt` at `hopperE4m3K32` and `G = 4`. -/
theorem flagsAt_hopper : flagsAt hopperE4m3K32 4 = wordFlags := rfl

/-- The H100 record's flags (`devH100.flags`) are `unitFlagsAt` at `hopperE4m3K32` and `G = 4`. -/
theorem unitFlagsAt_hopper {m n k : ℕ} : @unitFlagsAt hopperE4m3K32 m n k 4 = unitFlags := rfl

section Pure

variable (p : Pipeline)

theorem flagsAt_zero : flagsAt p 0 = pureFlagsP p := rfl

theorem flagsAt_of_ne_zero {G : Nat} (hG : G ≠ 0) : flagsAt p G = wordFlagsP p G := ite_eq_right hG

theorem length_rowAtoms {k : ℕ} (a b : Fin k → Code) : (rowAtoms a b).length = k / 32 := List.length_ofFn

/-! ## The lone skip -/

/-- Skipping one atom by a one-hot mask is folding the list without it. -/
theorem foldMask_oneMask {α β : Type} (f : β → α → β) (w : β) (xs : List α) (j : Nat) :
    foldMask f w xs (oneMask xs.length j) = (xs.eraseIdx j).foldl f w := by
  induction xs generalizing w j with
  | nil => rfl
  | cons x xs ih =>
    cases j with
    | zero =>
      rw [List.length_cons, oneMask, foldMask, ite_eq_left rfl, List.eraseIdx_cons_zero, foldMask_replicate]
    | succ j =>
      rw [List.length_cons, oneMask, foldMask, ite_eq_right Bool.false_ne_true, List.eraseIdx_cons_succ, List.foldl_cons,
        ih]

/-- **The lone skip resumes from the atom's incoming word**: the pure chain with atom `t` skipped folds the atoms after
`t` from the word before it (`FreezeP.inWordP`). -/
theorem pureSkip_resume (atoms : List AtomIn) (t : Nat) :
    pureSkipP p atoms t = wordQ ((atoms.drop (t + 1)).foldl (stepP p) (inWordP p atoms t)) := by
  rw [pureSkipP, pureChain, List.eraseIdx_eq_take_drop_succ, List.foldl_append]
  rfl

/-- The pure chain is the chain's one group, promoted once (`FreezeP.skipOneP` on `atoms.toChunks 0`). -/
theorem skipOne_single (g : List AtomIn) (t : Nat) : skipOneP p [g] 0 t = pureSkipP p g t := by
  rw [skipOneP, show [g].getD 0 [] = g from rfl, skipIn_eqP p [g] 0 _ (by simp)]
  change promote (some 0) (foldMask (stepP p) 0 g (oneMask g.length t)) = _
  rw [AccumProofs.promote_zero, foldMask_oneMask, pureSkipP, pureChain]
  rfl

theorem pureSkip_iff_skipOne (atoms : List AtomIn) (t : Nat) (ht : t < atoms.length) :
    skipOneP p (atoms.toChunks 0) 0 t = chainP p 0 atoms ↔ pureSkipP p atoms t = pureChain p atoms := by
  cases atoms with
  | nil => simp at ht
  | cons x xs =>
    rw [AccumProofs.chainP_zero, show (x :: xs).toChunks 0 = [x :: xs] from rfl, skipOne_single]

/-- An identity atom's lone skip leaves the pure chain unchanged. -/
theorem pureSkip_of_identity (atoms : List AtomIn) (t : Nat) (ht : t < atoms.length)
    (hid : stepP p (inWordP p atoms t) atoms[t] = inWordP p atoms t) : pureSkipP p atoms t = pureChain p atoms := by
  rw [pureSkip_resume, ← hid]
  change _ = wordQ (atoms.foldl (stepP p) 0)
  rw [foldl_split (stepP p) atoms t ht 0]
  rfl

/-! ## `pureFlagsP` is the lone skips -/

/-- **`pureFlagsP` is sound and complete**: it flags atom `t` of the word over `a` and `b` exactly when the word has
atom `t` and skipping it alone leaves the pure chain's output unchanged. -/
theorem pureFlags_iff (k : ℕ) (a b : Fin k → Code) (t : ℕ) :
    pureFlagsP p k a b t = true ↔
      t < k / 32 ∧ pureChain p ((rowAtoms a b).eraseIdx t) = pureChain p (rowAtoms a b) := by
  unfold pureFlagsP pureSkipP
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]

/-- `pureFlags_iff` on the chain with no promotion, `chainP p 0` and its one group: `FreezeP.skipOneP`. -/
theorem pureFlags_iff_skipOne (k : ℕ) (a b : Fin k → Code) (t : ℕ) :
    pureFlagsP p k a b t = true ↔
      t < k / 32 ∧ skipOneP p ((rowAtoms a b).toChunks 0) 0 t = chainP p 0 (rowAtoms a b) := by
  rw [pureFlags_iff]
  exact and_congr_right fun ht =>
    (pureSkip_iff_skipOne p _ t (by rwa [length_rowAtoms])).symm

/-- **The analogue of `skipOne_iff`**: `pureFlagsP` flags atom `t` exactly when it is absorbed in the chain's one group
(`FreezeP.absorbedP`: its lone skip leaves the group's word, or its promotion from `0`, unchanged). -/
theorem pureFlags_iff_absorbed (k : ℕ) (a b : Fin k → Code) (t : ℕ) :
    pureFlagsP p k a b t = true ↔ t < k / 32 ∧ absorbedP p ((rowAtoms a b).toChunks 0) 0 t = true := by
  rw [pureFlags_iff_skipOne]
  refine and_congr_right fun ht => ?_
  have hne : rowAtoms a b ≠ [] := by
    intro h; have := length_rowAtoms a b; rw [h, List.length_nil] at this; omega
  obtain ⟨x, xs, hx⟩ := List.exists_cons_of_ne_nil hne
  have hi : 0 < ((rowAtoms a b).toChunks 0).length := by rw [hx]; exact Nat.zero_lt_one
  exact skipOne_iffP p _ 0 t hi

/-- An identity atom of the pure chain is flagged. -/
theorem pureFlags_of_identity (k : ℕ) (a b : Fin k → Code) (t : ℕ) (ht : t < k / 32)
    (hid : stepP p (inWordP p (rowAtoms a b) t) ((rowAtoms a b)[t]'(by rwa [length_rowAtoms])) =
      inWordP p (rowAtoms a b) t) :
    pureFlagsP p k a b t = true :=
  (pureFlags_iff p k a b t).2 ⟨ht, pureSkip_of_identity p _ t _ hid⟩

/-! ## Unchanged output is unchanged word -/

theorem fold_word (atoms : List AtomIn) :
    ∀ c : Nat, c < 2 ^ 32 → c ≠ 0x80000000 →
      atoms.foldl (stepP p) c < 2 ^ 32 ∧ atoms.foldl (stepP p) c ≠ 0x80000000 := by
  induction atoms with
  | nil => exact fun c h1 h2 => ⟨h1, h2⟩
  | cons x xs ih =>
    intro c _ _
    exact ih _ (AccumProofs.step_lt _ _ _ _) (AccumProofs.step_ne_negZero _ _ _ _)

theorem exp_ne_of_wordQ {w : Nat} {y : ℚ} (h : wordQ w = some y) : (w >>> 23) % 256 ≠ 255 := by
  intro he
  simp only [wordQ, fp32Value, he, ite_true] at h
  split_ifs at h <;> simp [Value.finite?] at h

/-- Two words below `2^32`, neither `−0`, of the same finite value are equal. -/
theorem word_eq_of_wordQ {w w' : Nat} (hw : w < 2 ^ 32) (hw0 : w ≠ 0x80000000) (hw' : w' < 2 ^ 32)
    (hw0' : w' ≠ 0x80000000) {y : ℚ} (h : wordQ w = some y) (h' : wordQ w' = some y) : w = w' := by
  by_cases hy : y = 0
  · subst hy
    have hf : PeelExact.finite? w = some w := by
      unfold PeelExact.finite?; rw [ite_eq_right_iff.2 fun h0 => absurd h0 (exp_ne_of_wordQ h)]
    have hf' : PeelExact.finite? w' = some w' := by
      unfold PeelExact.finite?; rw [ite_eq_right_iff.2 fun h0 => absurd h0 (exp_ne_of_wordQ h')]
    rw [AccumProofs.wordQ_zero_word ⟨hw, hw0, hf⟩ h, AccumProofs.wordQ_zero_word ⟨hw', hw0', hf'⟩ h']
  · rw [← AccumProofs.f32Word_word hw (exp_ne_of_wordQ h) h hy,
      AccumProofs.f32Word_word hw' (exp_ne_of_wordQ h') h' hy]

/-- Two pure chains of the same finite output end on the same word. -/
theorem pureWord_eq {atoms atoms' : List AtomIn} (h : pureChain p atoms' = pureChain p atoms)
    (hf : pureChain p atoms ≠ none) : atoms'.foldl (stepP p) 0 = atoms.foldl (stepP p) 0 := by
  obtain ⟨y, hy⟩ := Option.ne_none_iff_exists'.1 hf
  have h1 := fold_word p atoms 0 (by norm_num) (by decide)
  have h2 := fold_word p atoms' 0 (by norm_num) (by decide)
  exact word_eq_of_wordQ h2.1 h2.2 h1.1 h1.2 (h.trans hy) hy

/-- **`pureFlags_iff` on words**: while the pure chain's output is finite, `pureFlagsP` flags atom `t` exactly when
skipping it alone leaves the final word unchanged. -/
theorem pureFlags_iff_word (k : ℕ) (a b : Fin k → Code) (t : ℕ) (hf : pureChain p (rowAtoms a b) ≠ none) :
    pureFlagsP p k a b t = true ↔
      t < k / 32 ∧ ((rowAtoms a b).eraseIdx t).foldl (stepP p) 0 = (rowAtoms a b).foldl (stepP p) 0 := by
  rw [pureFlags_iff]
  exact and_congr_right fun _ => ⟨fun h => pureWord_eq p h hf, fun h => congrArg wordQ h⟩

/-! ## The unit's form, for `debitDev` at `G = 0` -/

variable {m n k : ℕ}

/-- **The unit's flags at `G = 0`** flag atom `t` of word `(i, j)` exactly when skipping it alone leaves the ticket
`chainAt p 0` (the device record's `ticket`) unchanged. -/
theorem unitFlagsAt_zero_iff (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (i : Fin m)
    (j : Fin n) (t : ℕ) :
    unitFlagsAt p 0 Aq Bq i j t = true ↔
      t < k / 32 ∧ pureChain p ((rowAtoms (Aq i) (Bq j)).eraseIdx t) = chainAt p 0 Aq Bq i j := by
  rw [AccumProofs.chainAt_zero_eq_pure]
  exact pureFlags_iff p k (Aq i) (Bq j) t

/-- **The debit's atoms at `G = 0`** (`debitDev`'s `x.atoms`, `Skip.chainDebit`): the unit's atoms whose lone skip leaves
the ticket unchanged. -/
theorem pureDebit_eq (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (c : ℚ) :
    chainDebit k c (unitFlagsAt p 0 Aq Bq) = c * ∑ i : Fin m, ∑ j : Fin n, ((Finset.range (k / 32)).filter fun t =>
      pureChain p ((rowAtoms (Aq i) (Bq j)).eraseIdx t) = chainAt p 0 Aq Bq i j).card := by
  have h : ∀ i j, ((Finset.range (k / 32)).filter fun t => unitFlagsAt p 0 Aq Bq i j t = true) =
      (Finset.range (k / 32)).filter fun t =>
        pureChain p ((rowAtoms (Aq i) (Bq j)).eraseIdx t) = chainAt p 0 Aq Bq i j :=
    fun i j => Finset.filter_congr fun t ht => by
      rw [unitFlagsAt_zero_iff, and_iff_right (Finset.mem_range.1 ht)]
  unfold chainDebit
  simp only [h]

/-- **`needed` at `G = 0`**, on `k` a multiple of 32: element `(i, l)` of `A′` is needed exactly when some word `(i, j)`
changes when its atom `l / 32` is skipped alone. -/
theorem pureNeeded_iff (hk : 32 ∣ k) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (l : Fin k) :
    needed (unitFlagsAt p 0 Aq Bq) i l ↔
      ∃ j, pureChain p ((rowAtoms (Aq i) (Bq j)).eraseIdx ((l : ℕ) / 32)) ≠ chainAt p 0 Aq Bq i j := by
  have hl : (l : ℕ) / 32 < k / 32 := by obtain ⟨q, rfl⟩ := hk; have := l.isLt; omega
  unfold needed
  refine exists_congr fun j => ?_
  rw [Bool.eq_false_iff, ne_eq, unitFlagsAt_zero_iff, and_iff_right hl]

end Pure

/-! ## Identities of the pure chain on sm_120 -/

/-- An atom's exact sum `Σ a·b`. -/
def sumAt (x : AtomIn) : ℚ := ∑ l, val (x.1 l) * val (x.2 l)

/-- An atom's absolute mass `Σ |a·b|`. -/
def massAt (x : AtomIn) : ℚ := ∑ l, |val (x.1 l) * val (x.2 l)|

/-- The absolute mass of a list of atoms. -/
def mass (atoms : List AtomIn) : ℚ := (atoms.map massAt).sum

theorem massAt_nonneg (x : AtomIn) : 0 ≤ massAt x := Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem mass_nonneg (atoms : List AtomIn) : 0 ≤ mass atoms :=
  List.sum_nonneg fun q hq => by obtain ⟨x, -, rfl⟩ := List.mem_map.1 hq; exact massAt_nonneg x

theorem mass_take_succ (atoms : List AtomIn) (t : Nat) (ht : t < atoms.length) :
    mass (atoms.take (t + 1)) = mass (atoms.take t) + massAt atoms[t] := by
  rw [mass, mass, List.take_add_one, List.getElem?_eq_getElem ht, Option.toList_some, List.map_append,
    List.sum_append, List.map_singleton, List.sum_singleton]

theorem mass_take_le (atoms : List AtomIn) (t : Nat) : mass (atoms.take t) ≤ mass atoms := by
  conv_rhs => rw [← List.take_append_drop t atoms]
  rw [mass, mass, List.map_append, List.sum_append]
  exact le_add_of_nonneg_right (mass_nonneg _)

theorem term_abs_value (t : Term) : |t.value| = (t.mag : ℚ) * 2 ^ (t.exp - 23) := by
  unfold Term.value
  rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0 : ℚ) ≤ t.mag), abs_of_pos (two_zpow_pos _)]
  split <;> simp

/-- A term of magnitude at least `2^j` is at least `2^(exp − 23 + j)`. -/
theorem pow_le_abs_value (t : Term) (j : ℕ) (h : 2 ^ j ≤ t.mag) :
    (2 : ℚ) ^ (t.exp - 23 + j) ≤ |t.value| := by
  rw [term_abs_value, zpow_add₀ two_ne_zero, zpow_natCast]
  have h' : (2 : ℚ) ^ j ≤ (t.mag : ℚ) := by exact_mod_cast h
  calc (2 : ℚ) ^ (t.exp - 23) * 2 ^ j = 2 ^ j * 2 ^ (t.exp - 23) := mul_comm _ _
    _ ≤ (t.mag : ℚ) * 2 ^ (t.exp - 23) := mul_le_mul_of_nonneg_right h' (two_zpow_pos _).le

theorem product_mag_ge (a b : Code) (h : (product a b).mag ≠ 0) : 2 ^ 17 ≤ (product a b).mag := by
  have e : (product a b).mag = (a.sig * b.sig) <<< 17 := by
    unfold product; dsimp only; split_ifs with h' <;> simp [Term.zero, h']
  rw [e, Nat.shiftLeft_eq] at h ⊢
  generalize a.sig * b.sig = s at h ⊢
  have : s ≠ 0 := fun h0 => h (by rw [h0, zero_mul])
  omega

/-- A nonzero product's exponent is at most its value's binade plus 6. -/
theorem product_pow_le (a b : Code) (h : (product a b).mag ≠ 0) :
    (2 : ℚ) ^ (product a b).exp ≤ 64 * |(product a b).value| := by
  have h1 := pow_le_abs_value _ 17 (product_mag_ge a b h)
  generalize (product a b).value = v at h1 ⊢
  generalize (product a b).exp = e at h1 ⊢
  have h2 : (2 : ℚ) ^ e = 2 ^ (e - 23 + ((17 : ℕ) : ℤ)) * 2 ^ (6 : ℤ) := by
    rw [← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  rw [h2]
  have h64 : (2 : ℚ) ^ (6 : ℤ) = 64 := by norm_num
  rw [h64]
  linarith

/-- The alignment exponent is the floor or a nonzero term's exponent. -/
theorem maxExp_cases (z : ℤ) (ts : List Term) :
    maxExp z ts = z ∨ ∃ t ∈ ts, t.mag ≠ 0 ∧ maxExp z ts = t.exp := by
  induction ts generalizing z with
  | nil => left; rfl
  | cons s ts ih =>
    unfold maxExp at *
    rw [List.foldl_cons]
    split_ifs with hs
    · rcases ih s.exp with h | ⟨t, ht, hm, h⟩
      · exact Or.inr ⟨s, List.mem_cons_self, hs.1, h⟩
      · exact Or.inr ⟨t, List.mem_cons_of_mem _ ht, hm, h⟩
    · rcases ih z with h | ⟨t, ht, hm, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨t, List.mem_cons_of_mem _ ht, hm, h⟩

/-- With a nonzero term above the floor, `2^E ≤ B` once every nonzero term has `2^exp ≤ B`. -/
theorem pow_maxExp_le {z : ℤ} {B : ℚ} (ts : List Term) (hts : ∃ t ∈ ts, t.mag ≠ 0 ∧ z < t.exp)
    (h : ∀ t ∈ ts, t.mag ≠ 0 → (2 : ℚ) ^ t.exp ≤ B) : (2 : ℚ) ^ maxExp z ts ≤ B := by
  rcases maxExp_cases z ts with h1 | ⟨t, ht, htm, h1⟩
  · obtain ⟨t, ht, htm, hlt⟩ := hts
    have := le_maxExp z ts t ht htm
    omega
  · rw [h1]; exact h t ht htm

theorem val_mul_eq (a b : Code) (ha : a.isNaN = false) (hb : b.isNaN = false) :
    val a * val b = (product a b).value :=
  (productExact _ _ _ _ (val_eq ha) (val_eq hb)).symm

/-- On finite codes, a nonzero product's exponent is at most `2^6·M`, `M` bounding the products. -/
theorem prod_exp_le {a b : Fin 32 → Code} (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) {M : ℚ}
    (hM : ∀ l, |val (a l) * val (b l)| ≤ M) (l : Fin 32) (hl : (product (a l) (b l)).mag ≠ 0) :
    (2 : ℚ) ^ (product (a l) (b l)).exp ≤ 64 * M := by
  have := product_pow_le (a l) (b l) hl
  rw [← val_mul_eq _ _ (ha l) (hb l)] at this
  linarith [hM l]

/-- The grid of 32 products at width 26: `32·2^(E − 25) = 2^−20·2^E`. -/
theorem grid_bound {E : ℤ} {B : ℚ} (hE : (2 : ℚ) ^ E ≤ B) :
    (32 : ℚ) * 2 ^ (E - ((26 : ℕ) : ℤ) + 1) ≤ 2 ^ (-20 : ℤ) * B := by
  have : (32 : ℚ) * 2 ^ (E - ((26 : ℕ) : ℤ) + 1) = 2 ^ (-20 : ℤ) * 2 ^ E := by
    rw [show (32 : ℚ) = 2 ^ (5 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero, ← zpow_add₀ two_ne_zero]
    congr 1; push_cast; ring
  rw [this]; exact mul_le_mul_of_nonneg_left hE (by positivity)

/-- A running word of the pure chain at width `w`: `+0`, or normal at exponent at least `−11 − w` and of magnitude at
most `C`. -/
def PureWordW (w : Nat) (C : ℚ) (acc : Nat) : Prop :=
  acc = 0 ∨ ∃ tc, fp32Value acc = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧ -11 - (w : ℤ) ≤ tc.exp ∧ |tc.value| ≤ C

theorem bitLength_mul_pow {M : ℕ} (h1 : 2 ^ 23 ≤ M) (h2 : M < 2 ^ 24) (s : ℕ) : bitLength (M * 2 ^ s) = 24 + s := by
  have hn : M * 2 ^ s ≠ 0 := by positivity
  unfold bitLength
  rw [ite_eq_right hn]
  have hl : (M * 2 ^ s).log2 = 23 + s := by
    refine le_antisymm (Nat.lt_succ_iff.1 ((Nat.log2_lt hn).2 ?_)) ((Nat.le_log2 hn).2 ?_)
    · calc M * 2 ^ s < 2 ^ 24 * 2 ^ s := Nat.mul_lt_mul_of_pos_right h2 (by positivity)
        _ = 2 ^ (23 + s).succ := by rw [← pow_add]; congr 1; omega
    · calc 2 ^ (23 + s) = 2 ^ 23 * 2 ^ s := pow_add _ _ _
        _ ≤ M * 2 ^ s := Nat.mul_le_mul_right _ h1
  omega

/-- **One step of the pure chain** at width `24 ≤ w ≤ 115` and floor at most −126, on finite codes: from a word of
magnitude at most `C ≥ 0`, the next word is at most `C` plus the atom's mass, while that is at most `2^100`. -/
theorem pure_stepW {w : ℕ} (hw24 : 24 ≤ w) (hw : w ≤ 115) {z : ℤ} (hz : z ≤ -126) {C : ℚ} (hC0 : 0 ≤ C)
    {acc : Nat} (hacc : PureWordW w C acc) (x : AtomIn) (hx : FiniteAtom x) (hC : C + massAt x ≤ 2 ^ 100) :
    PureWordW w (C + massAt x) (stepP ⟨[32], w, z⟩ acc x) := by
  obtain ⟨ha, hb⟩ := hx
  obtain ⟨tc, hfw, htc, hcC⟩ : ∃ tc, fp32Value acc = .fin tc ∧
      (tc.mag = 0 ∨ (2 ^ 23 ≤ tc.mag ∧ -11 - (w : ℤ) ≤ tc.exp)) ∧ |tc.value| ≤ C := by
    rcases hacc with rfl | ⟨tc, h1, h2, h3, h4⟩
    · exact ⟨Term.zero, fp32Value_zero, Or.inl rfl, by simp [Term.value, Term.zero, hC0]⟩
    · exact ⟨tc, h1, Or.inr ⟨h2, h3⟩, h4⟩
  unfold stepP
  rw [step_fin hfw x.1 x.2 ha hb]
  unfold groupSum
  set E := maxExp z (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with hEdef
  set total := alignedSum w E (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with htdef
  change PureWordW w (C + massAt x) (packValue (let t := normalize w E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t))
  have hcE : tc.mag ≠ 0 → tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self
  have hle : ∀ l, (product (x.1 l) (x.2 l)).mag ≠ 0 → (product (x.1 l) (x.2 l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  clear_value total E
  have hsum : total = aligned w E tc + ∑ l, aligned w E (product (x.1 l) (x.2 l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have hct : truncTo (2 ^ (E - w + 1)) tc.value = (aligned w E tc : ℚ) * 2 ^ (E - w + 1) := by
    by_cases hm : tc.mag = 0
    · have : tc.value = 0 := by simp [Term.value, hm]
      rw [this, truncTo_zero]; simp [aligned, hm]
    · exact (aligned_valueW hw24 fun _ => hcE hm).symm
  have htot : (total : ℚ) * 2 ^ (E - w + 1) =
      truncTo (2 ^ (E - w + 1)) tc.value + ∑ l, truncTo (2 ^ (E - w + 1)) (val (x.1 l) * val (x.2 l)) := by
    rw [hsum]; push_cast
    rw [add_mul, Finset.sum_mul, hct, add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_valueW hw24 (hle l), val_mul_eq _ _ (ha l) (hb l)]
  have hg : (0 : ℚ) < 2 ^ (E - w + 1) := two_zpow_pos _
  have hbound : |(total : ℚ) * 2 ^ (E - w + 1)| ≤ C + massAt x := by
    rw [htot]
    refine (abs_add_le _ _).trans (add_le_add ((truncTo_abs_le hg _).trans hcC) ?_)
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun l _ => truncTo_abs_le hg _)
  by_cases h0 : total = 0
  · left; rw [h0]; simp [normalize, Term.zero, packValue, packTerm]
  have hn : total.natAbs ≠ 0 := by omega
  have hbl1 : 1 ≤ bitLength total.natAbs := by unfold bitLength; rw [ite_eq_right hn]; omega
  have hlow : -11 - (w : ℤ) ≤ E + (bitLength total.natAbs : ℤ) - w := by
    by_cases hE : -12 ≤ E
    · omega
    · have hp0 : ∀ l, (product (x.1 l) (x.2 l)).mag = 0 := fun l => by
        by_contra hp; have := (product_exp _ _ hp).1; have := hle l hp; omega
      have hc0 : tc.mag ≠ 0 := by
        intro hc; apply h0; rw [hsum]
        simp [aligned, hc, hp0]
      obtain ⟨hm1, hme⟩ := htc.resolve_left hc0
      have hEc : E = tc.exp := by
        refine le_antisymm ?_ (hcE hc0)
        rw [hEdef]
        refine maxExp_le _ _ _ (by omega) fun t ht htm => ?_
        rcases List.mem_cons.1 ht with rfl | ht
        · exact le_rfl
        · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
          exact absurd (hp0 l) htm
      have hal : (aligned w E tc).natAbs = tc.mag * 2 ^ (w - 24) := by
        unfold aligned toAdder
        rw [ite_eq_right hc0, ite_eq_left hw24, Nat.shiftLeft_eq, hEc, sub_self, Int.toNat_zero, Nat.shiftRight_zero]
        dsimp only
        split <;> simp only [Int.natAbs_neg, Int.natAbs_natCast]
      have hpz : ∑ l, aligned w E (product (x.1 l) (x.2 l)) = 0 :=
        Finset.sum_eq_zero fun l _ => by simp [aligned, hp0 l]
      have hta : total.natAbs = tc.mag * 2 ^ (w - 24) := by rw [hsum, hpz, add_zero]; exact hal
      rw [hta, bitLength_mul_pow hm1 (fp32Value_fin hfw).1]
      push_cast; omega
  obtain ⟨m1, m2, e1, hv⟩ := normalize_specW hw24 h0 (E := E) (by omega)
  have habs : |(normalize w E total).value| ≤ C + massAt x := by
    rw [hv]; exact (truncTo_abs_le (two_zpow_pos _) _).trans hbound
  have e2 : (normalize w E total).exp ≤ 127 := by
    have h1 := pow_le_abs_value _ 23 m1
    have h2 : (2 : ℚ) ^ (normalize w E total).exp ≤ 2 ^ (100 : ℤ) := by
      have : (normalize w E total).exp - 23 + ((23 : ℕ) : ℤ) = (normalize w E total).exp := by push_cast; ring
      rw [this] at h1
      calc (2 : ℚ) ^ (normalize w E total).exp ≤ C + massAt x := h1.trans habs
        _ ≤ 2 ^ (100 : ℤ) := by rw [zpow_ofNat]; exact hC
    have := (zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).1 h2
    omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (not_lt.2 e2))]
  right
  exact ⟨_, fp32Value_packTerm m1 m2 (by rw [e1]; omega) e2, m1, by rw [e1]; exact hlow, habs⟩

/-- **The pure chain's running words** at width `24 ≤ w ≤ 115`, floor at most −126, on finite atoms of total mass at
most `2^100`: the word before atom `t` is `+0` or normal at exponent at least `−11 − w`, of magnitude at most the mass
of the atoms before it. -/
theorem pure_inWordW {w : ℕ} (hw24 : 24 ≤ w) (hw : w ≤ 115) {z : ℤ} (hz : z ≤ -126) (atoms : List AtomIn)
    (hfin : ∀ x ∈ atoms, FiniteAtom x) (hm : mass atoms ≤ 2 ^ 100) :
    ∀ t ≤ atoms.length, PureWordW w (mass (atoms.take t)) (inWordP ⟨[32], w, z⟩ atoms t) := by
  intro t ht
  induction t with
  | zero => left; rfl
  | succ t ih =>
    have htl : t < atoms.length := by omega
    have : inWordP ⟨[32], w, z⟩ atoms (t + 1) = stepP ⟨[32], w, z⟩ (inWordP ⟨[32], w, z⟩ atoms t) atoms[t] := by
      unfold inWordP
      rw [List.take_add_one, List.getElem?_eq_getElem htl, Option.toList_some, List.foldl_append]
      rfl
    rw [this, mass_take_succ atoms t htl]
    refine pure_stepW hw24 hw hz (mass_nonneg _) (ih (by omega)) _ (hfin _ (List.getElem_mem htl)) ?_
    rw [← mass_take_succ atoms t htl]
    exact (mass_take_le _ _).trans hm

/-- **No identity on sm_120** (floor at most −126), on finite codes: on a running word `+0` or normal at exponent at
least `−37` of magnitude at most `C ≥ 0`, an atom whose products are at most `M` and whose exact sum `S` has
`9·2^−23·C + 2^−14·M < |S|` is not an identity. The first term is (a)'s nine ulps of the word; the second is the grid
of a product above the word (`2^6` for a subnormal product's exponent). -/
theorem sm120_no_identity {z : ℤ} (hz : z ≤ -126) {C : ℚ} (hC0 : 0 ≤ C) {acc : Nat} (hacc : PureWordW 26 C acc)
    (x : AtomIn) (hx : FiniteAtom x) {M : ℚ} (hM : ∀ l, |val (x.1 l) * val (x.2 l)| ≤ M)
    (h : 9 * 2 ^ (-23 : ℤ) * C + 2 ^ (-14 : ℤ) * M < |sumAt x|) :
    stepP (sm120E4m3K32 z) acc x ≠ acc := by
  obtain ⟨ha, hb⟩ := hx
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hS0 : sumAt x ≠ 0 := by
    intro h0; rw [h0, abs_zero] at h
    have : 0 ≤ 9 * 2 ^ (-23 : ℤ) * C + 2 ^ (-14 : ℤ) * M :=
      add_nonneg (mul_nonneg (by positivity) hC0) (mul_nonneg (by positivity) hM0)
    linarith
  obtain ⟨l0, hl0⟩ : ∃ l, (product (x.1 l) (x.2 l)).mag ≠ 0 := by
    by_contra hall
    push Not at hall
    apply hS0
    refine Finset.sum_eq_zero fun l _ => ?_
    rw [val_mul_eq _ _ (ha l) (hb l)]
    simp [Term.value, hall l]
  intro hid
  have hC9 : 0 ≤ 9 * 2 ^ (-23 : ℤ) * C := mul_nonneg (by positivity) hC0
  rcases hacc with rfl | ⟨tc, hfw, hm, he, hcC⟩
  · have hP := (sm120_step_zero_iff z x.1 x.2 ha hb).1 hid
    have hS : |sumAt x - preSumW z 26 x.1 x.2| <
        32 * 2 ^ (maxExp z (List.ofFn fun l => product (x.1 l) (x.2 l)) - ((26 : ℕ) : ℤ) + 1) :=
      abs_sum_sub_truncTo_lt (two_zpow_pos _) _
    have hE : (2 : ℚ) ^ maxExp z (List.ofFn fun l => product (x.1 l) (x.2 l)) ≤ 64 * M := by
      refine pow_maxExp_le (List.ofFn fun l => product (x.1 l) (x.2 l))
        ⟨product (x.1 l0) (x.2 l0), List.mem_ofFn.2 ⟨l0, rfl⟩, hl0, ?_⟩ fun t ht htm => ?_
      · have := (product_exp _ _ hl0).1; omega
      · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
        exact prod_exp_le ha hb hM l htm
    rw [hP, sub_zero] at hS
    generalize maxExp z (List.ofFn fun l => product (x.1 l) (x.2 l)) = E at hS hE
    have h1 := grid_bound hE
    have h2 : (2 : ℚ) ^ (-20 : ℤ) * (64 * M) = 2 ^ (-14 : ℤ) * M := by
      rw [show (2 : ℚ) ^ (-14 : ℤ) = 64 * 2 ^ (-20 : ℤ) by norm_num]; ring
    linarith
  · obtain ⟨h1, -, -⟩ := identity_cellW (w := 26) (by norm_num) hfw hm (by push_cast; omega) x.1 x.2 ha hb hid
    have hc : (2 : ℚ) ^ tc.exp ≤ C := by
      have := pow_le_abs_value tc 23 hm
      rw [show tc.exp - 23 + ((23 : ℕ) : ℤ) = tc.exp by push_cast; ring] at this
      linarith
    have hE : (2 : ℚ) ^ stepExpW z tc x.1 x.2 ≤ C + 64 * M := by
      refine pow_maxExp_le (tc :: List.ofFn fun l => product (x.1 l) (x.2 l))
        ⟨tc, List.mem_cons_self, by omega, by omega⟩ fun t ht htm => ?_
      rcases List.mem_cons.1 ht with rfl | ht
      · linarith
      · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
        linarith [prod_exp_le ha hb hM l htm]
    have hu : (2 : ℚ) ^ (tc.exp - 23) = 2 ^ (-23 : ℤ) * 2 ^ tc.exp := by
      rw [← zpow_add₀ two_ne_zero]; congr 1; ring
    rw [hu] at h1
    generalize stepExpW z tc x.1 x.2 = E at h1 hE
    generalize tc.exp = e at h1 hc
    have h2 := grid_bound hE
    have e2 : (2 : ℚ) ^ (-23 : ℤ) * 2 ^ e ≤ 2 ^ (-23 : ℤ) * C := mul_le_mul_of_nonneg_left hc (by positivity)
    have e3 : (2 : ℚ) ^ (-20 : ℤ) * (C + 64 * M) + 2 ^ (-23 : ℤ) * C =
        9 * 2 ^ (-23 : ℤ) * C + 2 ^ (-14 : ℤ) * M := by
      rw [show (2 : ℚ) ^ (-20 : ℤ) = 8 * 2 ^ (-23 : ℤ) by norm_num,
        show (2 : ℚ) ^ (-14 : ℤ) = 512 * 2 ^ (-23 : ℤ) by norm_num]
      ring
    unfold sumAt at h
    linarith

/-- **No identity in sm_120's pure chain** (floor at most −126), on finite atoms of total mass at most `2^100`: atom
`t`, of exact sum `S` and products at most `M`, is not an identity once `9·2^−23·C + 2^−14·M < |S|`, `C` the absolute
mass of the atoms before it. -/
theorem sm120_pure_no_identity {z : ℤ} (hz : z ≤ -126) (atoms : List AtomIn) (hfin : ∀ x ∈ atoms, FiniteAtom x)
    (hm : mass atoms ≤ 2 ^ 100) (t : Nat) (ht : t < atoms.length) {M : ℚ}
    (hM : ∀ l, |val (atoms[t].1 l) * val (atoms[t].2 l)| ≤ M)
    (h : 9 * 2 ^ (-23 : ℤ) * mass (atoms.take t) + 2 ^ (-14 : ℤ) * M < |sumAt atoms[t]|) :
    stepP (sm120E4m3K32 z) (inWordP (sm120E4m3K32 z) atoms t) atoms[t] ≠ inWordP (sm120E4m3K32 z) atoms t :=
  sm120_no_identity hz (mass_nonneg _)
    (pure_inWordW (w := 26) (by norm_num) (by norm_num) hz atoms hfin hm t ht.le) _
    (hfin _ (List.getElem_mem ht)) hM h

theorem mass_le_of_le {P : ℚ} (atoms : List AtomIn) (hP : ∀ x ∈ atoms, ∀ l, |val (x.1 l) * val (x.2 l)| ≤ P) :
    mass atoms ≤ 32 * atoms.length * P := by
  induction atoms with
  | nil => simp [mass]
  | cons x xs ih =>
    have h1 : massAt x ≤ 32 * P := by
      calc massAt x ≤ ∑ _l : Fin 32, P := Finset.sum_le_sum fun l _ => hP x List.mem_cons_self l
        _ = 32 * P := by simp
    have h2 := ih fun y hy => hP y (List.mem_cons_of_mem _ hy)
    rw [mass, List.map_cons, List.sum_cons, ← mass, List.length_cons]
    push_cast
    linarith

/-- **`sm120_pure_no_identity` with every product at most `P`** (Π's domain: `P ≤ 448²`, and at most `k/64` spike
entries): atom `t` is not an identity once `(9·2^−23·32t + 2^−14)·P < |S|`, i.e. about `k·P < 2^19.8·|S|` at the
chain's end, `k = 32t`. -/
theorem sm120_pure_no_identity_of_le {z : ℤ} (hz : z ≤ -126) (atoms : List AtomIn)
    (hfin : ∀ x ∈ atoms, FiniteAtom x) {P : ℚ} (hP : ∀ x ∈ atoms, ∀ l, |val (x.1 l) * val (x.2 l)| ≤ P)
    (hk : 32 * atoms.length * P ≤ 2 ^ 100) (t : Nat) (ht : t < atoms.length)
    (h : (9 * 2 ^ (-23 : ℤ) * (32 * t) + 2 ^ (-14 : ℤ)) * P < |sumAt atoms[t]|) :
    stepP (sm120E4m3K32 z) (inWordP (sm120E4m3K32 z) atoms t) atoms[t] ≠ inWordP (sm120E4m3K32 z) atoms t := by
  refine sm120_pure_no_identity hz atoms hfin ((mass_le_of_le atoms hP).trans hk) t ht
    (hP _ (List.getElem_mem ht)) ?_
  have h1 := mass_le_of_le (atoms.take t) fun x hx => hP x (List.mem_of_mem_take hx)
  rw [List.length_take, min_eq_left ht.le] at h1
  have : 9 * 2 ^ (-23 : ℤ) * mass (atoms.take t) ≤ 9 * 2 ^ (-23 : ℤ) * (32 * t * P) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  linarith

/-- **No bound on the products alone excludes an identity**: on the word `1.0` (`0x3F800000`), the atom with `a = 1.0`
throughout and `b = 1.0` on its first 16 positions and `−1.0` on the rest (products `±1`, of sum 0) is an identity on
sm_120, at any floor at most −126 (`FreezeP.sm120_floor`). -/
theorem sm120_cancel_identity {z : ℤ} (hz : z ≤ -126) :
    stepP (sm120E4m3K32 z) 0x3F800000 (fun _ => 0x38, fun l => if (l : ℕ) < 16 then 0x38 else 0xB8) = 0x3F800000 := by
  unfold stepP
  rw [sm120_floor hz (by norm_num : (-133 : ℤ) ≤ -126)]
  decide +kernel

theorem val_one : val 0x38 = 1 ∧ val 0xB8 = -1 := by decide +kernel

end Pouw.PearlC.FreezePPure
