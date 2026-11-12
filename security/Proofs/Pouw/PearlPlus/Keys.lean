module

public import PouwPearlPlus.Rule
public import Proofs.Pouw.PearlC.CoreOps
public import Proofs.Pouw.PearlC.RowStats
public import Proofs.Pouw.PearlC.TileValues

/-!
# Pearl+'s bridge, part (c), the debit's keys: `key` is the value order, `least` a threshold, `carried` the rounded add

`Rule.wordDebit` replays a word's later promotions on keys rather than on values: `Rule.key` writes a finite FP32 word as
an integer in [−TOP, TOP] in value order (±0 one key), `Rule.least` bisects for the least key at which a monotone predicate
holds, and `Rule.carried x w` is the key of RN(x's value + w's value). `keyVal` is a key's value; the three statements
here are what the debit's replay (`Debit`) reads of them.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- The value of a magnitude key `u`, a finite word's low 31 bits: `wordQ`'s field formula at exponent field `u / 2^23`. -/
def magVal (u : ℕ) : ℚ :=
  (((if u / 2 ^ 23 = 0 then 0 else 2 ^ 23) + u % 2 ^ 23 : ℕ) : ℚ) * (2 : ℚ) ^ (((max (u / 2 ^ 23) 1 : ℕ) : ℤ) - 150)

/-- The value of a key. -/
def keyVal (x : ℤ) : ℚ := if x < 0 then -magVal x.natAbs else magVal x.natAbs

/-- **The key is the value's order**: `keyVal` is strictly increasing; a finite word's key is in [−TOP, TOP] and its value
is the key's; and every key in [−TOP, TOP] is a finite word's, `Rule.wordOf`'s. -/
def KeyIsTheValue : Prop :=
  StrictMono keyVal ∧
    (∀ w : ℕ, w < 2 ^ 32 → (w >>> 23) % 256 ≠ 255 →
      wordQ w = some (keyVal (Rule.key w)) ∧ -Rule.TOP ≤ Rule.key w ∧ Rule.key w ≤ Rule.TOP) ∧
    ∀ x : ℤ, -Rule.TOP ≤ x → x ≤ Rule.TOP →
      Rule.wordOf x < 2 ^ 32 ∧ (Rule.wordOf x >>> 23) % 256 ≠ 255 ∧ Rule.key (Rule.wordOf x) = x

/-- **`least` is the threshold**: for a predicate monotone on [−TOP, TOP], `Rule.least` is in [−TOP, TOP + 1], and a key
in [−TOP, TOP] is at least it exactly when the predicate holds there. -/
def LeastIsTheThreshold : Prop :=
  ∀ pred : ℤ → Bool, (∀ x y, -Rule.TOP ≤ x → x ≤ y → y ≤ Rule.TOP → pred x = true → pred y = true) →
    -Rule.TOP ≤ Rule.least pred ∧ Rule.least pred ≤ Rule.TOP + 1 ∧
      ∀ x, -Rule.TOP ≤ x → x ≤ Rule.TOP → (Rule.least pred ≤ x ↔ pred x = true)

/-- **`carried` is the rounded add**: where the check's FP32 add is the semantics' on finite words, a key in [−TOP, TOP]
carried through a finite word of magnitude below 2⁴⁶ (a promotion group's word) stays in [−TOP, TOP], and its value is
`rnAdd` of the two values. -/
def CarriedIsTheAdd : Prop :=
  (∀ a b : ℕ, (wordQ a).isSome → (wordQ b).isSome → Evaluate.Fp32.f32AddV2 a b = TileCircuit.f32Add a b) →
    ∀ (x : ℤ) (w : ℕ) (y : ℚ), -Rule.TOP ≤ x → x ≤ Rule.TOP → wordQ w = some y → |y| < 2 ^ 46 →
      -Rule.TOP ≤ Rule.carried x w ∧ Rule.carried x w ≤ Rule.TOP ∧ keyVal (Rule.carried x w) = rnAdd (keyVal x) y

/-! ## Proofs -/

namespace KeysSim

/-! ### Keys and words -/

theorem or_top {n : ℕ} (h : n < 2 ^ 31) : n ||| 0x80000000 = n + 2 ^ 31 := by
  rw [Nat.or_comm, show (0x80000000 : ℕ) = 1 <<< 31 from rfl, ← Nat.shiftLeft_add_eq_or_of_lt h, Nat.shiftLeft_eq]
  omega

theorem key_cases {w : ℕ} (hw : w < 2 ^ 32) :
    ((w >>> 31) % 2 = 1 ∧ Rule.key w = -((w % 2 ^ 31 : ℕ) : ℤ)) ∨
      ((w >>> 31) % 2 = 0 ∧ w < 2 ^ 31 ∧ Rule.key w = (w : ℤ)) := by
  unfold Rule.key
  rw [Nat.shiftRight_eq_div_pow, CoreOps.and_7fffffff]
  by_cases h : w < 2 ^ 31
  · rw [show w / 2 ^ 31 = 0 by omega]
    exact Or.inr ⟨rfl, h, rfl⟩
  · rw [show w / 2 ^ 31 = 1 by omega]
    exact Or.inl ⟨rfl, rfl⟩

theorem key_bound {w : ℕ} (hw : w < 2 ^ 32) (he : (w >>> 23) % 256 ≠ 255) :
    -Rule.TOP ≤ Rule.key w ∧ Rule.key w ≤ Rule.TOP := by
  have hT : Rule.TOP = 0x7F7FFFFF := rfl
  rw [Nat.shiftRight_eq_div_pow] at he
  rcases key_cases hw with ⟨-, hk⟩ | ⟨-, hlt, hk⟩ <;> rw [hk] <;> omega

theorem wordOf_ok {x : ℤ} (h1 : -Rule.TOP ≤ x) (h2 : x ≤ Rule.TOP) :
    Rule.wordOf x < 2 ^ 32 ∧ (Rule.wordOf x >>> 23) % 256 ≠ 255 ∧ Rule.key (Rule.wordOf x) = x := by
  have hT : Rule.TOP = 0x7F7FFFFF := rfl
  unfold Rule.wordOf
  by_cases hx : x < 0
  · rw [ite_eq_left hx, or_top (by omega)]
    have hw : x.natAbs + 2 ^ 31 < 2 ^ 32 := by omega
    refine ⟨hw, by rw [Nat.shiftRight_eq_div_pow]; omega, ?_⟩
    rcases key_cases hw with ⟨-, hk⟩ | ⟨-, h, -⟩
    · rw [hk]; omega
    · omega
  · rw [ite_eq_right hx]
    have hw : x.toNat < 2 ^ 32 := by omega
    refine ⟨hw, by rw [Nat.shiftRight_eq_div_pow]; omega, ?_⟩
    rcases key_cases hw with ⟨h, -⟩ | ⟨-, -, hk⟩
    · rw [Nat.shiftRight_eq_div_pow] at h; omega
    · rw [hk]; omega

/-! ### Values -/

theorem magVal_zero : magVal 0 = 0 := by simp [magVal]

theorem magVal_lt_succ (u : ℕ) : magVal u < magVal (u + 1) := by
  unfold magVal
  rcases Nat.lt_or_ge (u % 2 ^ 23 + 1) (2 ^ 23) with h | h
  · rw [show (u + 1) / 2 ^ 23 = u / 2 ^ 23 by omega, show (u + 1) % 2 ^ 23 = u % 2 ^ 23 + 1 by omega]
    exact mul_lt_mul_of_pos_right (Nat.cast_lt.2 (by omega)) (zpow_pos (by norm_num) _)
  · rw [show (u + 1) / 2 ^ 23 = u / 2 ^ 23 + 1 by omega, show (u + 1) % 2 ^ 23 = 0 by omega,
      show u % 2 ^ 23 = 2 ^ 23 - 1 by omega]
    generalize u / 2 ^ 23 = e
    rcases Nat.eq_zero_or_pos e with rfl | he
    · rw [show max (0 : ℕ) 1 = 1 from rfl, show max (0 + 1 : ℕ) 1 = 1 from rfl]
      norm_num
    · rw [ite_eq_right (show ¬ e = 0 by omega), ite_eq_right (show ¬ e + 1 = 0 by omega),
        max_eq_left (show 1 ≤ e by omega), max_eq_left (show 1 ≤ e + 1 by omega)]
      have h2 : (2 : ℚ) ^ (((e + 1 : ℕ) : ℤ) - 150) = 2 ^ ((e : ℤ) - 150) * 2 := by
        rw [← zpow_add_one₀ (by norm_num : (2 : ℚ) ≠ 0)]
        congr 1
        push_cast
        ring
      rw [h2, show (((2 ^ 23 + (2 ^ 23 - 1) : ℕ)) : ℚ) = 16777215 by norm_num,
        show (((2 ^ 23 + 0 : ℕ)) : ℚ) = 8388608 by norm_num]
      have hp := zpow_pos (show (0 : ℚ) < 2 by norm_num) ((e : ℤ) - 150)
      linarith

theorem magVal_strictMono : StrictMono magVal := strictMono_nat_of_lt_succ magVal_lt_succ

theorem keyVal_strictMono : StrictMono keyVal := by
  have hm := magVal_strictMono
  intro a b hab
  unfold keyVal
  split_ifs with ha hb hb
  · exact neg_lt_neg (hm (by omega))
  · have h1 : magVal 0 < magVal a.natAbs := hm (by omega)
    have h2 : magVal 0 ≤ magVal b.natAbs := hm.monotone (Nat.zero_le _)
    rw [magVal_zero] at h1 h2
    linarith
  · omega
  · exact hm (by omega)

theorem key_value {w : ℕ} (hw : w < 2 ^ 32) (he : (w >>> 23) % 256 ≠ 255) :
    wordQ w = some (keyVal (Rule.key w)) := by
  rw [PeelExactProofs.wordQ_fields he]
  congr 1
  rw [show (w >>> 23) % 256 = w % 2 ^ 31 / 2 ^ 23 by rw [Nat.shiftRight_eq_div_pow]; omega,
    show w % 2 ^ 23 = w % 2 ^ 31 % 2 ^ 23 by omega]
  rcases key_cases hw with ⟨hs, hk⟩ | ⟨hs, hlt, hk⟩
  · rw [hs, hk]
    show (if (1 : ℕ) = 1 then (-1 : ℚ) else 1) * magVal (w % 2 ^ 31) = keyVal (-((w % 2 ^ 31 : ℕ) : ℤ))
    rw [ite_eq_left rfl, neg_one_mul]
    unfold keyVal
    rcases Nat.eq_zero_or_pos (w % 2 ^ 31) with h0 | h0
    · rw [h0]; simp [magVal_zero]
    · rw [ite_eq_left (show -((w % 2 ^ 31 : ℕ) : ℤ) < 0 by omega), Int.natAbs_neg, Int.natAbs_natCast]
  · rw [hs, hk, Nat.mod_eq_of_lt hlt]
    show (if (0 : ℕ) = 1 then (-1 : ℚ) else 1) * magVal w = keyVal (w : ℤ)
    rw [ite_eq_right (show ¬ (0 : ℕ) = 1 by norm_num), one_mul]
    unfold keyVal
    rw [ite_eq_right (show ¬ (w : ℤ) < 0 by omega), Int.natAbs_natCast]

theorem keyVal_zero : keyVal 0 = 0 := by simp [keyVal, magVal_zero]

theorem keyVal_neg {x : ℤ} (hx : 0 < x) : keyVal (-x) = -keyVal x := by
  unfold keyVal
  rw [ite_eq_left (show -x < 0 by omega), ite_eq_right (show ¬ x < 0 by omega), Int.natAbs_neg]

theorem keyVal_top : keyVal Rule.TOP = (2 ^ 24 - 1) * 2 ^ 104 := by
  have e : Rule.TOP.natAbs = 0x7F7FFFFF := rfl
  unfold keyVal
  rw [ite_eq_right (show ¬ Rule.TOP < 0 by decide), e]
  unfold magVal
  rw [show (0x7F7FFFFF : ℕ) / 2 ^ 23 = 254 by norm_num, show (0x7F7FFFFF : ℕ) % 2 ^ 23 = 2 ^ 23 - 1 by norm_num,
    show max (254 : ℕ) 1 = 254 from rfl]
  norm_num

theorem keyVal_abs_le {x : ℤ} (h1 : -Rule.TOP ≤ x) (h2 : x ≤ Rule.TOP) : |keyVal x| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
  have hm := keyVal_strictMono.monotone
  have a := hm h2
  have b := hm h1
  rw [keyVal_neg (show (0 : ℤ) < Rule.TOP by decide), keyVal_top] at b
  rw [keyVal_top] at a
  exact abs_le.2 ⟨by linarith, a⟩

/-- Below `(2²⁴ − 1/2)·2¹⁰⁴`, half an ulp past the largest finite value, a value's word is finite. -/
theorem f32Word_fin {v : ℚ} (hv : |v| < (2 ^ 24 - 1 / 2) * 2 ^ 104) : (f32Word v >>> 23) % 256 ≠ 255 := by
  by_cases h127 : |v| < 2 ^ 127
  · obtain ⟨t, ht, -⟩ := AtomErr.f32Word_spec h127
    exact RowCircuit.wordQ_exp (TileCircuit.wordQ_of_fin ht)
  have hv0 : v ≠ 0 := by rintro rfl; norm_num at h127
  have hge : (2 : ℚ) ^ 127 ≤ |v| := not_lt.1 h127
  have hb : binade |v| = 127 := Pouw.Fp8Atom.Proofs.binade_unique (abs_pos.2 hv0)
    (by rw [show (2 : ℚ) ^ (127 : ℤ) = 2 ^ 127 by norm_num]; exact hge)
    (by rw [show (2 : ℚ) ^ ((127 : ℤ) + 1) = 2 ^ 128 by norm_num]; linarith)
  have hu : (2 : ℚ) ^ ((127 : ℤ) - 23) = 2 ^ 104 := by norm_num
  have hR1 : roundEven (|v| / 2 ^ ((127 : ℤ) - 23)) ≤ 2 ^ 24 - 1 := by
    apply RowCircuit.roundEven_le_of_lt
    rw [hu, div_lt_iff₀ (by positivity)]
    push_cast
    linarith
  have hR2 : 2 ^ 23 ≤ roundEven (|v| / 2 ^ ((127 : ℤ) - 23)) := by
    refine le_trans (Int.le_floor.2 ?_) (Pouw.Fp8Atom.Proofs.floor_le_roundEven _)
    rw [hu, le_div_iff₀ (by positivity)]
    push_cast
    linarith
  unfold f32Word
  dsimp only
  rw [hb, show max (127 : ℤ) (-126) = 127 by norm_num]
  split_ifs <;> first | omega | (rw [Nat.shiftRight_eq_div_pow]; omega)

/-! ### Bisection -/

section Loops

variable {σ α : Type}

theorem forIn_list_id {l : List α} {body : α → σ → Id (ForInStep σ)} (P : ℕ → σ → Prop) :
    ∀ {i : ℕ} {x : σ}, P i x →
      (∀ j (hj : j < l.length) s, P (i + j) s → ∃ z, body l[j] s = .yield z ∧ P (i + j + 1) z) →
      P (i + l.length) (Id.run (forIn l x body)) := by
  induction l with
  | nil => intro i x hx _; exact hx
  | cons a l ih =>
    intro i x hx hstep
    obtain ⟨z, hz, hP⟩ := hstep 0 (by simp) x (by simpa using hx)
    rw [List.forIn_cons]
    simp only [List.getElem_cons_zero] at hz
    rw [hz]
    have := ih (i := i + 1) (x := z) hP (fun j hj s hs => by
      obtain ⟨z', hz', hP'⟩ := hstep (j + 1) (by simp only [List.length_cons]; omega) s
        (by rw [show i + (j + 1) = i + 1 + j by omega]; exact hs)
      exact ⟨z', by simpa using hz', by rw [show i + 1 + j + 1 = i + (j + 1) + 1 by omega]; exact hP'⟩)
    rw [show i + (a :: l).length = i + 1 + l.length by simp only [List.length_cons]; omega]
    exact this

theorem forIn_range_id {n : ℕ} {body : ℕ → σ → Id (ForInStep σ)} (P : ℕ → σ → Prop) {x : σ}
    (h0 : P 0 x) (hstep : ∀ i < n, ∀ s, P i s → ∃ z, body i s = .yield z ∧ P (i + 1) z) :
    P n (Id.run (forIn [:n] x body)) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_list_id (l := List.range' 0 [:n].size 1) (body := body) (i := 0) P h0 (fun j hj s hP => by
    simp only [List.length_range', Std.Legacy.Range.size] at hj
    simp only [List.getElem_range', Nat.zero_add, Nat.one_mul] at hP ⊢
    exact hstep j (by omega) s hP)
  simpa [Std.Legacy.Range.size] using this

end Loops

/-- `Rule.least`'s loop body on `(lo, hi)`. -/
def leastStep (pred : ℤ → Bool) : ℕ → ℤ × ℤ → Id (ForInStep (ℤ × ℤ)) := fun _ s =>
  if s.1 < s.2 then
    if pred ((s.1 + s.2).fdiv 2) = true then pure (ForInStep.yield (s.1, (s.1 + s.2).fdiv 2))
    else pure (ForInStep.yield ((s.1 + s.2).fdiv 2 + 1, s.2))
  else pure (ForInStep.yield (s.1, s.2))

/-- One bisection step halves `hi − lo`. -/
theorem half_step {lo hi m : ℤ} {i : ℕ} (hd : m * 2 ≤ hi - lo) (h6 : (hi - lo) * 2 ^ i ≤ 2 * Rule.TOP) :
    m * 2 ^ (i + 1) ≤ 2 * Rule.TOP := by
  have := mul_le_mul_of_nonneg_right hd (pow_nonneg (show (0 : ℤ) ≤ 2 by norm_num) i)
  rw [pow_succ, mul_comm ((2 : ℤ) ^ i) 2, ← mul_assoc]
  exact le_trans this h6

end KeysSim

/-- **The key is the value's order.** -/
theorem keyIsTheValue : KeyIsTheValue := by
  unfold KeyIsTheValue
  exact ⟨KeysSim.keyVal_strictMono, fun _ hw he => ⟨KeysSim.key_value hw he, KeysSim.key_bound hw he⟩,
    fun _ h1 h2 => KeysSim.wordOf_ok h1 h2⟩

/-- **`least` is the threshold**: the bisection keeps `pred hi`, `¬ pred` below `lo`, and `(hi − lo)·2^i ≤ 2·TOP`, so
after 64 steps `lo = hi`. -/
theorem leastIsTheThreshold : LeastIsTheThreshold := by
  intro pred hmono
  have hT : Rule.TOP = 0x7F7FFFFF := rfl
  by_cases htop : pred Rule.TOP = true
  · let P : ℕ → ℤ × ℤ → Prop := fun i s =>
      -Rule.TOP ≤ s.1 ∧ s.1 ≤ s.2 ∧ s.2 ≤ Rule.TOP ∧ pred s.2 = true ∧
        (∀ y, -Rule.TOP ≤ y → y < s.1 → pred y = false) ∧ (s.2 - s.1) * 2 ^ i ≤ 2 * Rule.TOP
    have hloop := KeysSim.forIn_range_id (n := 64) (x := (-Rule.TOP, Rule.TOP)) (body := KeysSim.leastStep pred) P
      ⟨by omega, by omega, by omega, htop, fun y h1 h2 => by omega, by rw [pow_zero, mul_one]; omega⟩
      (fun i _ s ⟨h1, h2, h3, h4, h5, h6⟩ => by
        obtain ⟨lo, hi⟩ := s
        dsimp only at h1 h2 h3 h4 h5 h6 ⊢
        unfold KeysSim.leastStep
        dsimp only
        have hfd := Int.fdiv_eq_ediv_of_nonneg (lo + hi) (b := 2) (by decide)
        by_cases hlt : lo < hi
        · rw [ite_eq_left hlt]
          by_cases hp : pred ((lo + hi).fdiv 2) = true
          · rw [ite_eq_left hp]
            exact ⟨_, rfl, by omega, by omega, by omega, hp, h5, KeysSim.half_step (by omega) h6⟩
          · rw [ite_eq_right hp]
            refine ⟨_, rfl, by omega, by omega, by omega, h4, fun y hy1 hy2 => ?_, KeysSim.half_step (by omega) h6⟩
            dsimp only at hy2
            by_cases hy : y < lo
            · exact h5 y hy1 hy
            · cases hpy : pred y with
              | false => rfl
              | true => exact absurd (hmono y _ hy1 (by omega) (by omega) hpy) hp
        · rw [ite_eq_right hlt]
          refine ⟨_, rfl, h1, h2, h3, h4, h5, ?_⟩
          show (hi - lo) * 2 ^ (i + 1) ≤ 2 * Rule.TOP
          rw [show hi - lo = 0 by omega, zero_mul]
          omega)
    have e : Rule.least pred = (Id.run (forIn [:64] (-Rule.TOP, Rule.TOP) (KeysSim.leastStep pred))).1 := by
      unfold Rule.least
      simp only [htop, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
      rfl
    rw [e]
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hloop
    generalize Id.run _ = s at h1 h2 h3 h4 h5 h6 ⊢
    have hd : s.2 - s.1 = 0 := by
      rw [show (2 : ℤ) ^ 64 = 18446744073709551616 by norm_num] at h6
      omega
    refine ⟨h1, by omega, fun x hx1 hx2 =>
      ⟨fun hx => hmono s.1 x h1 hx hx2 (by rw [show s.1 = s.2 by omega]; exact h4), fun hx => ?_⟩⟩
    refine not_lt.1 fun hc => ?_
    have := h5 x hx1 hc
    rw [hx] at this
    cases this
  · have e : Rule.least pred = Rule.TOP + 1 := by
      unfold Rule.least
      simp only [htop, Bool.not_false, ↓reduceIte]
      rfl
    rw [e]
    refine ⟨by omega, by omega, fun x hx1 hx2 => ⟨fun h => by omega, fun hx => ?_⟩⟩
    exact absurd (hmono x Rule.TOP hx1 hx2 (by omega) hx) htop

/-- **`carried` is the rounded add**: `wordOf x` is finite with value `keyVal x`, the sum is below half an ulp past the
largest finite value, so the promotion is finite and its value is `rnAdd` (`TileCircuit.promoteWord_wordQ`). -/
theorem carriedIsTheAdd : CarriedIsTheAdd := by
  intro H x w y hx1 hx2 hw hy
  obtain ⟨ha32, hae, hka⟩ := KeysSim.wordOf_ok hx1 hx2
  have hqa : wordQ (Rule.wordOf x) = some (keyVal x) := by
    have h := KeysSim.key_value ha32 hae
    rwa [hka] at h
  have hsum : |keyVal x + y| < (2 ^ 24 - 1 / 2) * 2 ^ 104 := by
    have h1 := KeysSim.keyVal_abs_le hx1 hx2
    have h2 := abs_add_le (keyVal x) y
    linarith
  have hH := H (Rule.wordOf x) w (by simp [hqa]) (by simp [hw])
  obtain ⟨nz, hnz⟩ := RowCircuit.f32Add_fin hqa hw
  have hfin : (TileCircuit.f32Add (Rule.wordOf x) w >>> 23) % 256 ≠ 255 := by
    rw [hnz]
    unfold PeelExact.rnWord
    split_ifs <;> first | decide | exact KeysSim.f32Word_fin hsum
  have hadd : PeelExact.addWord? (Rule.wordOf x) w = some (TileCircuit.f32Add (Rule.wordOf x) w) := by
    have e : PeelExact.addWord? (Rule.wordOf x) w = PeelExact.finite? (TileCircuit.f32Add (Rule.wordOf x) w) := by
      simp only [PeelExact.addWord?, TileCircuit.f32Add, hqa, hw]
    rw [e]
    unfold PeelExact.finite?
    rw [ite_eq_right hfin]
  have hlt := WholeCircuit.addWord?_lt hadd
  have hp : PeelExact.promoteWord? (Rule.wordOf x) w = some (if TileCircuit.f32Add (Rule.wordOf x) w = 0x80000000 then 0
      else TileCircuit.f32Add (Rule.wordOf x) w) := by
    unfold PeelExact.promoteWord?
    simp [hadd]
  have hval := TileCircuit.promoteWord_wordQ hqa hw hp
  unfold Rule.carried
  rw [hH]
  dsimp only
  generalize TileCircuit.f32Add (Rule.wordOf x) w = out at hfin hlt hval ⊢
  have hT : Rule.TOP = 0x7F7FFFFF := rfl
  by_cases h0 : out = 0x80000000
  · rw [ite_eq_left h0, CoreOps.wordQ_zero] at hval
    subst h0
    show -Rule.TOP ≤ 0 ∧ 0 ≤ Rule.TOP ∧ keyVal 0 = rnAdd (keyVal x) y
    exact ⟨by omega, by omega, by rw [KeysSim.keyVal_zero]; exact Option.some.inj hval⟩
  · rw [ite_eq_right h0] at hval
    simp only [beq_iff_eq, h0, ↓reduceIte]
    obtain ⟨hb1, hb2⟩ := KeysSim.key_bound hlt hfin
    exact ⟨hb1, hb2, Option.some.inj ((KeysSim.key_value hlt hfin).symm.trans hval)⟩

end Pouw.PearlPlus.Bridge
