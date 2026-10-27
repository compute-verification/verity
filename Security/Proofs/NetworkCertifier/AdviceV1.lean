import Proofs.NetworkCertifier.AdviceString
import Definitions.NetworkCertifier.Guarantees.AdviceV1Kraft
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Advice v1's charge counts: the proof

`Guarantees.AdviceV1`. A state's weight is its interval's width over `2^bits` (`wt`). A scaling doubles the width and
counts a bit (`scale_spec`), and a symbol's outcomes take disjoint sub-intervals of the width
(`sum_owt_countStep_le`), so no step adds weight. The finish charges at most what the weight pays for (`finish_le`).
Over the decisions (`sum_decisions_le`), n_arrived's bins and offsets (`sum_owt_arrive_le`) and the windows
(`sum_session_le`), the weights `2^(−charge)` of the streams no window refuses sum to at most 1, so at most `2^c` are
charged `c` or less (`encard_le_of_kraft`). That is `BitsCount` for advice v1's charge (`bitsCount`), and the leak
bounds follow from `AdviceString`'s, with that premise discharged.

Windows with counts: a window's weight with counts is its decisions' weight times its counts' (`stepWt_windowCounted`),
so over any finite set of decision streams and counts the weights sum to at most 1 (`sum_windowCounted_le`). Any run
whose windows each meet Kraft does too (`sum_runWt_le`, `encard_runOf_le`), which gives the count with counts
(`encard_chargedCounted_le`) and `BitsCount` (`bitsCount_counted`). Prefix-free codes meet Kraft
(`kraft_of_prefixFree`, `CountCode.ofPrefixFree`); the unary code is one for nonzero counts (`CountCode.unary`), and
under the code charging nothing for `Unit` the charges are advice v1's (`chargesCounted_zero`).
-/

namespace NetTiming.AdviceV1

open Finset

/-! ### The weight of a state -/

/-- The width of a state's interval, `high − low + 1`. -/
def width (st : Tally) : ℕ := st.high + 1 - st.low

/-- A state's weight: its width over `2^bits`. -/
noncomputable def wt (st : Tally) : ℝ := (width st : ℝ) * (2⁻¹ : ℝ) ^ st.bits

/-- A refused state weighs nothing. -/
noncomputable def owt (o : Option Tally) : ℝ := (o.map wt).getD 0

/-- The states a window reaches after each step: an interval of 32-bit ends wider than `Q1`, and the first state's
interval until the coder is used. -/
def Good (st : Tally) : Prop :=
  st.low ≤ st.high ∧ st.high < 2 ^ 32 ∧ Q1 < width st ∧ (st.used = false → st.low = 0 ∧ st.high = TOP)

theorem wt_nonneg (st : Tally) : 0 ≤ wt st := by unfold wt; positivity

theorem owt_nonneg (o : Option Tally) : 0 ≤ owt o := by
  cases o <;> simp [owt, wt_nonneg]

@[simp] theorem owt_none : owt none = 0 := rfl

@[simp] theorem owt_some (st : Tally) : owt (some st) = wt st := rfl

theorem good_fresh : Good Tally.fresh := by
  refine ⟨?_, ?_, ?_, fun _ => ⟨rfl, rfl⟩⟩ <;> simp [Tally.fresh, width, TOP, Q1]

theorem wt_fresh : wt Tally.fresh = 2 ^ 32 := by
  norm_num [wt, width, Tally.fresh, TOP]

/-! ### Scalings keep the weight -/

theorem scale_of_not_scales {k lo hi : ℕ} (h : scales lo hi = false) : scale k lo hi = some (lo, hi, 0) := by
  cases k <;> simp [scale, h]

theorem wide_of_not_scales {lo hi : ℕ} (hle : lo ≤ hi) (h : scales lo hi = false) : Q1 < hi + 1 - lo := by
  simp only [scales, decide_eq_false_iff_not, not_or, not_and, not_lt] at h
  simp only [HALF, Q1, Q3] at *
  omega

theorem shift_spec {lo hi : ℕ} (hle : lo ≤ hi) (hlt : hi < 2 ^ 32) (hs : scales lo hi = true) :
    (shift lo hi).1 ≤ (shift lo hi).2 ∧ (shift lo hi).2 < 2 ^ 32 ∧
      (shift lo hi).2 + 1 - (shift lo hi).1 = 2 * (hi + 1 - lo) := by
  simp only [scales, decide_eq_true_eq] at hs
  by_cases h : hi < HALF ∨ HALF ≤ lo
  · simp only [shift, ite_eq_left h]
    simp only [HALF, Q1, Q3] at *
    omega
  · simp only [shift, ite_eq_right h]
    simp only [HALF, Q1, Q3] at *
    omega

/-- **The scalings**: they end on an interval wider than `Q1`, its width the start's doubled once per scaling. -/
theorem scale_spec : ∀ (k lo hi : ℕ) {p : ℕ × ℕ × ℕ}, lo ≤ hi → hi < 2 ^ 32 → scale k lo hi = some p →
    p.1 ≤ p.2.1 ∧ p.2.1 < 2 ^ 32 ∧ Q1 < p.2.1 + 1 - p.1 ∧ p.2.1 + 1 - p.1 = (hi + 1 - lo) * 2 ^ p.2.2
  | 0, lo, hi, p, hle, hlt, h => by
    cases hs : scales lo hi
    · rw [scale_of_not_scales hs] at h
      cases h
      exact ⟨hle, hlt, wide_of_not_scales hle hs, by simp⟩
    · simp [scale, hs] at h
  | k + 1, lo, hi, p, hle, hlt, h => by
    cases hs : scales lo hi
    · rw [scale_of_not_scales hs] at h
      cases h
      exact ⟨hle, hlt, wide_of_not_scales hle hs, by simp⟩
    · simp only [scale, hs, ite_true, Option.map_eq_some_iff] at h
      obtain ⟨q, hq, rfl⟩ := h
      obtain ⟨h1, h2, h3⟩ := shift_spec hle hlt hs
      obtain ⟨i1, i2, i3, i4⟩ := scale_spec k _ _ h1 h2 hq
      refine ⟨i1, i2, i3, ?_⟩
      rw [i4, h3, pow_succ]
      ring

/-! ### A symbol splits the weight -/

theorem two_pow_mul_inv_pow (n : ℕ) : (2 : ℝ) ^ n * (2⁻¹ : ℝ) ^ n = 1 := by
  rw [← mul_pow]; norm_num

/-- **One symbol**: the state it leads to is one the window reaches, and weighs the symbol's sub-interval of the
width, `⌊r·(cum + freq)/total⌋ − ⌊r·cum/total⌋`, over `2^bits`. -/
theorem countStep_spec {st st' : Tally} {c f T : ℕ} (hst : Good st) (h : countStep st c f T = some st') :
    (1 ≤ f ∧ c + f ≤ T ∧ T ≤ MAX_TOTAL) ∧ Good st' ∧
      wt st' = ((width st * (c + f) / T - width st * c / T : ℕ) : ℝ) * (2⁻¹ : ℝ) ^ st.bits := by
  obtain ⟨hle, hlt, hw, hfr⟩ := hst
  have hwd : st.high + 1 - st.low = width st := rfl
  unfold countStep at h
  rw [hwd] at h
  split_ifs at h with hv
  obtain ⟨hf, hcf, hT⟩ := hv
  have hT0 : 0 < T := by omega
  have hrT : T ≤ width st := by simp only [MAX_TOTAL, Q1] at hT hw; omega
  have hab : width st * c / T + 1 ≤ width st * (c + f) / T := by
    rw [← Nat.add_div_right _ hT0]
    apply Nat.div_le_div_right
    nlinarith
  have hbr : width st * (c + f) / T ≤ width st := by
    calc width st * (c + f) / T ≤ width st * T / T := Nat.div_le_div_right (Nat.mul_le_mul_left _ hcf)
      _ = width st := Nat.mul_div_cancel _ hT0
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨p, hp, h2⟩ := h
  split_ifs at h2 with hb
  cases h2
  generalize ha : width st * c / T = a at hab hbr hp ⊢
  generalize hb' : width st * (c + f) / T = b at hab hbr hp ⊢
  obtain ⟨i1, i2, i3, i4⟩ := scale_spec K_MAX _ _ (show st.low + a ≤ st.low + b - 1 by omega)
    (show st.low + b - 1 < 2 ^ 32 by rw [← hwd] at hbr; omega) hp
  have hpre : st.low + b - 1 + 1 - (st.low + a) = b - a := by omega
  refine ⟨⟨hf, hcf, hT⟩, ⟨i1, i2, i3, fun hu => ?_⟩, ?_⟩
  · simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hu
    obtain ⟨hu, rfl⟩ := hu
    obtain ⟨h0, hTOP⟩ := hfr hu
    have hc : c = 0 := by omega
    subst hc
    have hwT : width st = 2 ^ 32 := by rw [← hwd, h0, hTOP]; norm_num [TOP]
    have ha0 : a = 0 := by rw [← ha, mul_zero, Nat.zero_div]
    have hb0 : b = 2 ^ 32 := by rw [← hb', zero_add, Nat.mul_div_cancel _ hT0, hwT]
    have hp' : scale K_MAX 0 TOP = some p := by rw [← hp, h0, ha0, hb0]; rfl
    rw [scale_of_not_scales (by decide)] at hp'
    cases hp'
    exact ⟨rfl, rfl⟩
  · show ((p.2.1 + 1 - p.1 : ℕ) : ℝ) * (2⁻¹ : ℝ) ^ (st.bits + p.2.2) = ((b - a : ℕ) : ℝ) * (2⁻¹ : ℝ) ^ st.bits
    rw [i4, hpre]
    push_cast
    rw [pow_add]
    calc ((b - a : ℕ) : ℝ) * 2 ^ p.2.2 * ((2⁻¹ : ℝ) ^ st.bits * (2⁻¹) ^ p.2.2)
        = ((b - a : ℕ) : ℝ) * (2⁻¹) ^ st.bits * (2 ^ p.2.2 * (2⁻¹) ^ p.2.2) := by ring
      _ = _ := by rw [two_pow_mul_inv_pow, mul_one]

theorem wt_countStep_le {st st' : Tally} {c f T : ℕ} (hst : Good st) (h : countStep st c f T = some st') :
    wt st' ≤ wt st := by
  obtain ⟨⟨hf, hcf, hT⟩, _, hw⟩ := countStep_spec hst h
  have hT0 : 0 < T := by omega
  have hbr : width st * (c + f) / T ≤ width st :=
    calc width st * (c + f) / T ≤ width st * T / T := Nat.div_le_div_right (Nat.mul_le_mul_left _ hcf)
      _ = width st := Nat.mul_div_cancel _ hT0
  rw [hw, wt]
  gcongr
  exact (Nat.sub_le _ _).trans hbr

/-- **A symbol's outcomes**: symbols whose `[cum, cum + freq)` are disjoint lead to states whose weights sum to at most
the state's. -/
theorem sum_owt_countStep_le {ι : Type*} (s : Finset ι) {st : Tally} (hst : Good st) (c f : ι → ℕ) (T : ℕ)
    (hdisj : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → c i + f i ≤ c j ∨ c j + f j ≤ c i) :
    ∑ i ∈ s, owt (countStep st (c i) (f i) T) ≤ wt st := by
  classical
  let J : ι → Finset ℕ := fun i =>
    if (countStep st (c i) (f i) T).isSome then Ico (width st * c i / T) (width st * (c i + f i) / T) else ∅
  have hJ : ∀ i, owt (countStep st (c i) (f i) T) = ((J i).card : ℝ) * (2⁻¹ : ℝ) ^ st.bits := by
    intro i
    cases h : countStep st (c i) (f i) T with
    | none => simp [J, h]
    | some st' =>
      rw [owt_some, (countStep_spec hst h).2.2]
      simp [J, h, Nat.card_Ico]
  have hvalid : ∀ i, (countStep st (c i) (f i) T).isSome → c i + f i ≤ T ∧ 0 < T := by
    intro i hi
    obtain ⟨st', h⟩ := Option.isSome_iff_exists.1 hi
    obtain ⟨⟨hf, hcf, _⟩, _⟩ := countStep_spec hst h
    exact ⟨hcf, by omega⟩
  have hsub : s.biUnion J ⊆ range (width st) := by
    intro x hx
    simp only [mem_biUnion] at hx
    obtain ⟨i, _, hx⟩ := hx
    simp only [J] at hx
    split_ifs at hx with hi
    · obtain ⟨hcf, hT0⟩ := hvalid i hi
      have : width st * (c i + f i) / T ≤ width st :=
        calc width st * (c i + f i) / T ≤ width st * T / T := Nat.div_le_div_right (Nat.mul_le_mul_left _ hcf)
          _ = width st := Nat.mul_div_cancel _ hT0
      simp only [mem_Ico] at hx
      simp only [mem_range]
      omega
    · simp at hx
  have hpd : (s : Set ι).PairwiseDisjoint J := by
    intro i hi j hj hij
    simp only [Function.onFun, J]
    split_ifs with h1 h2
    · rcases hdisj i hi j hj hij with h | h
      · have := Nat.div_le_div_right (c := T) (Nat.mul_le_mul_left (width st) h)
        exact Finset.disjoint_left.2 fun x hx hy => by simp only [mem_Ico] at hx hy; omega
      · have := Nat.div_le_div_right (c := T) (Nat.mul_le_mul_left (width st) h)
        exact Finset.disjoint_left.2 fun x hx hy => by simp only [mem_Ico] at hx hy; omega
    all_goals simp
  calc ∑ i ∈ s, owt (countStep st (c i) (f i) T) = (∑ i ∈ s, ((J i).card : ℝ)) * (2⁻¹ : ℝ) ^ st.bits := by
        rw [Finset.sum_mul]; exact sum_congr rfl fun i _ => hJ i
    _ = ((s.biUnion J).card : ℝ) * (2⁻¹ : ℝ) ^ st.bits := by rw [card_biUnion hpd]; push_cast; rfl
    _ ≤ (width st : ℝ) * (2⁻¹ : ℝ) ^ st.bits := by
        gcongr
        simpa using card_le_card hsub
    _ = wt st := rfl

/-- **A decision's two outcomes** weigh at most the state. -/
theorem owt_bitStep_add_le {st : Tally} (hst : Good st) (f0 f1 : ℕ) :
    owt (bitStep st f0 f1 true) + owt (bitStep st f0 f1 false) ≤ wt st := by
  by_cases h : 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ MAX_TOTAL
  · have := sum_owt_countStep_le (univ : Finset Bool) hst (fun b => if b then f0 else 0)
      (fun b => if b then f1 else f0) (f0 + f1) (by
        intro i _ j _ hij
        cases i <;> cases j <;> simp_all)
    simpa [Fintype.sum_bool, bitStep, h] using this
  · simp only [bitStep, ite_eq_right h, owt_none, add_zero]
    exact wt_nonneg st

theorem good_of_bitStep {st st' : Tally} {f0 f1 : ℕ} {b : Bool} (hst : Good st) (h : bitStep st f0 f1 b = some st') :
    Good st' := by
  unfold bitStep at h
  by_cases hv : 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ MAX_TOTAL
  · rw [ite_eq_left hv] at h
    exact (countStep_spec hst h).2.1
  · rw [ite_eq_right hv] at h
    cases h

/-! ### The window: zeros, finish, decisions, n_arrived -/

theorem zeros_spec : ∀ (zs : List (ℕ × ℕ)) {st st' : Tally}, Good st → zeros st zs = some st' →
    Good st' ∧ wt st' ≤ wt st
  | [], st, st', hst, h => by cases h; exact ⟨hst, le_rfl⟩
  | z :: zs, st, st', hst, h => by
    simp only [zeros, Option.bind_eq_some_iff] at h
    obtain ⟨s1, h1, h2⟩ := h
    obtain ⟨g2, w2⟩ := zeros_spec zs (countStep_spec hst h1).2.1 h2
    exact ⟨g2, w2.trans (wt_countStep_le hst h1)⟩

/-- **The finish** charges at most what the weight pays for: `2^(−charge) · 2^32 ≤ wt`. -/
theorem finish_le {st : Tally} (hst : Good st) {L : ℕ} (h : finish st = some L) : (2⁻¹ : ℝ) ^ L * 2 ^ 32 ≤ wt st := by
  unfold finish at h
  split_ifs at h
  cases h
  obtain ⟨_, _, hw, hfr⟩ := hst
  cases hu : st.used
  · obtain ⟨h0, hT⟩ := hfr hu
    have hw32 : width st = 2 ^ 32 := by unfold width; simp only [h0, hT, TOP]; norm_num
    rw [wt, hw32]
    apply le_of_eq
    simp only [Bool.toNat_false, mul_zero, add_zero]
    push_cast
    ring
  · have hw' : (2 : ℝ) ^ 30 ≤ width st := by
      have : 2 ^ 30 ≤ width st := by simp only [Q1] at hw; omega
      exact_mod_cast this
    simp only [Bool.toNat_true, wt, pow_add]
    calc (2⁻¹ : ℝ) ^ st.bits * (2⁻¹) ^ (2 * 1) * 2 ^ 32 = (2 : ℝ) ^ 30 * (2⁻¹) ^ st.bits := by norm_num; ring
      _ ≤ (width st : ℝ) * (2⁻¹) ^ st.bits := by gcongr

theorem sum_split_bool {n : ℕ} (g : (Fin (n + 1) → Bool) → ℝ) :
    ∑ v, g v = ∑ b : Bool, ∑ v : Fin n → Bool, g (Fin.cons b v) := by
  rw [← (Fin.consEquiv fun _ => Bool).sum_comp, Fintype.sum_prod_type]
  rfl

/-- **The decisions**: over every stream of `n` decisions, what follows them weighs at most the state, when it weighs at
most each state it starts from. -/
theorem sum_decisions_le (V : Tally → Carry → ℝ) (hV : ∀ st cy, Good st → V st cy ≤ wt st) :
    ∀ (n : ℕ) (st : Tally) (cy : Carry), Good st →
      ∑ v : Fin n → Bool, ((decisions st cy (List.ofFn v)).map fun p => V p.1 p.2).getD 0 ≤ wt st
  | 0, st, cy, hst => by simpa [decisions] using hV st cy hst
  | n + 1, st, cy, hst => by
    rw [sum_split_bool]
    simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, decisions]
    cases hf : ktFreq cy.counts cy.run with
    | none => simpa using wt_nonneg st
    | some f =>
      simp only [Option.bind_some]
      calc _ ≤ ∑ b : Bool, owt (bitStep st f.1 f.2 b) := by
            apply sum_le_sum
            intro b _
            cases hb : bitStep st f.1 f.2 b with
            | none => simp
            | some st' => simpa using sum_decisions_le V hV n st' _ (good_of_bitStep hst hb)
        _ ≤ wt st := by rw [Fintype.sum_bool]; exact owt_bitStep_add_le hst f.1 f.2

theorem good_of_arrive {bk : Book} {n : ℕ} {st : Tally} (h : arrive bk n = some st) : Good st := by
  unfold arrive at h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨s1, h1, h2⟩ := h
  have g1 := (countStep_spec good_fresh h1).2.1
  split_ifs at h2
  · exact (countStep_spec g1 h2).2.1
  · cases h2; exact g1

/-- A window's weight: `2^(−charge)`, or nothing when refused. -/
noncomputable def lw (bk : Book) (N : ℕ) (cy : Carry) (ds : List Bool) : ℝ :=
  ((window bk N cy ds).map fun p => (2⁻¹ : ℝ) ^ p.1).getD 0

theorem lw_nonneg (bk : Book) (N : ℕ) (cy : Carry) (ds : List Bool) : 0 ≤ lw bk N cy ds := by
  unfold lw
  cases window bk N cy ds <;> simp [pow_nonneg]

theorem sum_len_window_le (bk : Book) (N : ℕ) (cy : Carry) (n : ℕ) :
    ∑ v : Fin n → Bool, lw bk N cy (List.ofFn v) * 2 ^ 32 ≤ owt (arrive bk n) := by
  let V : Tally → Carry → ℝ := fun st _ => (((zeros st bk.zeros).bind finish).map fun L => (2⁻¹ : ℝ) ^ L * 2 ^ 32).getD 0
  have hV : ∀ st cy, Good st → V st cy ≤ wt st := by
    intro st cy hst
    simp only [V]
    cases hz : zeros st bk.zeros with
    | none => simpa using wt_nonneg st
    | some s2 =>
      obtain ⟨g2, w2⟩ := zeros_spec bk.zeros hst hz
      cases hfin : finish s2 with
      | none => simpa [hfin] using wt_nonneg st
      | some L => simpa [hfin] using (finish_le g2 hfin).trans w2
  by_cases hw : n ≤ N ∧ cy.run < 3
  · cases ha : arrive bk n with
    | none =>
      simp [lw, window, ha]
    | some st =>
      refine le_trans (le_of_eq ?_) (sum_decisions_le V hV n st cy (good_of_arrive ha))
      apply sum_congr rfl
      intro v _
      simp only [lw, window, List.length_ofFn, ite_eq_left hw, ha, Option.bind_some, V]
      cases decisions st cy (List.ofFn v) with
      | none => simp
      | some p =>
        simp only [Option.bind_some, Option.map_some, Option.getD_some]
        cases (zeros p.1 bk.zeros).bind finish <;> simp
  · have : ∀ v : Fin n → Bool, lw bk N cy (List.ofFn v) = 0 := by
      intro v
      simp [lw, window, List.length_ofFn, hw]
    simpa [this] using owt_nonneg _

theorem take_sum_add_getD_le (l : List ℕ) {i j : ℕ} (hij : i < j) : (l.take i).sum + l.getD i 0 ≤ (l.take j).sum := by
  have h1 : (l.take i).sum + l.getD i 0 = (l.take (i + 1)).sum := by
    by_cases hi : i < l.length
    · rw [List.sum_take_succ _ _ hi, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
    · rw [not_lt] at hi
      rw [List.take_of_length_le hi, List.take_of_length_le (by omega), List.getD_eq_getElem?_getD,
        List.getElem?_eq_none hi]
      simp
  rw [h1]
  have h2 := List.sum_take_add_sum_drop (l.take j) (i + 1)
  rw [List.take_take, min_eq_left (by omega)] at h2
  omega

theorem eq_of_size_eq {n m : ℕ} (h : n.size = m.size) (h2 : n.size < 2) : n = m := by
  have hn := Nat.lt_size_self n
  have hm := Nat.lt_size_self m
  rcases Nat.lt_or_ge n.size 1 with h0 | h1
  · have hn0 : n.size = 0 := by omega
    have hm0 : m.size = 0 := by omega
    rw [Nat.size_eq_zero.1 hn0, Nat.size_eq_zero.1 hm0]
  · have hn1 : 2 ^ 0 ≤ n := Nat.lt_size.1 (by omega)
    have hm1 : 2 ^ 0 ≤ m := Nat.lt_size.1 (by omega)
    have : n.size = 1 := by omega
    rw [this] at hn
    rw [← h, this] at hm
    simp at hn hm hn1 hm1
    omega

/-- **n_arrived**: over the window's possible counts `0 … N`, the states after its bin and offset weigh at most the
first state, `2^32`. -/
theorem sum_owt_arrive_le (bk : Book) (N : ℕ) : ∑ n ∈ range (N + 1), owt (arrive bk n) ≤ 2 ^ 32 := by
  classical
  rw [← sum_fiberwise_of_maps_to (g := Nat.size) (t := range (N.size + 1)) (fun n hn => by
    simp only [mem_range] at hn ⊢
    exact Nat.lt_succ_of_le (Nat.size_le_size (by omega)))]
  let B : ℕ → Option Tally := fun j =>
    countStep Tally.fresh (bk.arrived.take j).sum (bk.arrived.getD j 0) bk.total
  have hfib : ∀ j, ∑ n ∈ range (N + 1) with n.size = j, owt (arrive bk n) ≤ owt (B j) := by
    intro j
    have harr : ∀ n ∈ (range (N + 1)).filter (fun n => n.size = j), arrive bk n =
        (B j).bind fun st => if 2 ≤ j then countStep st (n - 2 ^ (j - 1)) 1 (2 ^ (j - 1)) else some st := by
      intro n hn
      simp only [mem_filter] at hn
      simp only [arrive, B, hn.2]
    rw [sum_congr rfl fun n hn => congrArg owt (harr n hn)]
    cases hB : B j with
    | none => simp
    | some s1 =>
      have g1 : Good s1 := (countStep_spec good_fresh hB).2.1
      simp only [Option.bind_some, owt_some]
      by_cases hj : 2 ≤ j
      · simp only [ite_eq_left hj]
        apply sum_owt_countStep_le _ g1
        intro n hn m hm hnm
        simp only [mem_filter] at hn hm
        have h1 : 2 ^ (j - 1) ≤ n := Nat.lt_size.1 (by omega)
        have h2 : 2 ^ (j - 1) ≤ m := Nat.lt_size.1 (by omega)
        omega
      · simp only [ite_eq_right hj, owt_some, sum_const, nsmul_eq_mul]
        have hc : ((range (N + 1)).filter fun n => n.size = j).card ≤ 1 := by
          rw [card_le_one]
          intro a ha b hb
          simp only [mem_filter] at ha hb
          exact eq_of_size_eq (ha.2.trans hb.2.symm) (by omega)
        calc (((range (N + 1)).filter fun n => n.size = j).card : ℝ) * wt s1 ≤ 1 * wt s1 := by
              gcongr
              · exact wt_nonneg s1
              · exact_mod_cast hc
          _ = wt s1 := one_mul _
  calc _ ≤ ∑ j ∈ range (N.size + 1), owt (B j) := sum_le_sum fun j _ => hfib j
    _ ≤ wt Tally.fresh := by
        apply sum_owt_countStep_le _ good_fresh
        intro i _ j _ hij
        rcases Nat.lt_or_gt_of_ne hij with h | h
        · exact Or.inl (take_sum_add_getD_le _ h)
        · exact Or.inr (take_sum_add_getD_le _ h)
    _ = 2 ^ 32 := wt_fresh

/-- **One window (Kraft)**: over any finite set of decision streams, the weights `2^(−charge)` of those the window
doesn't refuse sum to at most 1, from any carry. -/
theorem sum_window_le (bk : Book) (N : ℕ) (cy : Carry) (A : Finset (List Bool)) : ∑ ds ∈ A, lw bk N cy ds ≤ 1 := by
  classical
  let U : Finset (List Bool) := (range (N + 1)).biUnion fun n => univ.image (List.ofFn : (Fin n → Bool) → List Bool)
  have hU : ∑ ds ∈ U, lw bk N cy ds ≤ 1 := by
    rw [sum_biUnion (by
      intro n _ m _ hnm
      simp only [Function.onFun]
      exact Finset.disjoint_left.2 fun x hx hy => by
        simp only [mem_image, mem_univ, true_and] at hx hy
        obtain ⟨v, rfl⟩ := hx
        obtain ⟨w, hw⟩ := hy
        have := congrArg List.length hw
        simp at this
        exact hnm this.symm)]
    have : ∀ n ∈ range (N + 1),
        ∑ ds ∈ univ.image (List.ofFn : (Fin n → Bool) → List Bool), lw bk N cy ds * 2 ^ 32 ≤ owt (arrive bk n) := by
      intro n _
      rw [sum_image (fun x _ y _ h => List.ofFn_injective h)]
      exact sum_len_window_le bk N cy n
    have h2 := (sum_le_sum this).trans (sum_owt_arrive_le bk N)
    simp only [← sum_mul] at h2
    nlinarith
  calc ∑ ds ∈ A, lw bk N cy ds = ∑ ds ∈ A with ds.length ≤ N, lw bk N cy ds := by
        refine (sum_filter_of_ne fun ds _ h => ?_).symm
        by_contra hN
        exact h (by simp [lw, window, hN])
    _ ≤ ∑ ds ∈ U, lw bk N cy ds := by
        apply sum_le_sum_of_subset_of_nonneg
        · intro ds hds
          simp only [mem_filter] at hds
          simp only [U, mem_biUnion, mem_range, mem_image, mem_univ, true_and]
          exact ⟨ds.length, by omega, fun i => ds[i], List.ofFn_getElem⟩
        · exact fun ds _ _ => lw_nonneg bk N cy ds
    _ ≤ 1 := hU

/-! ### The run, and the count -/

/-- A run's weight: `2^(−Σ charges)`, or nothing when a window is refused. -/
noncomputable def lwS (bk : Book) (N : ℕ) (cy : Carry) (ws : List (List Bool)) : ℝ :=
  ((session bk N cy ws).map fun ls => (2⁻¹ : ℝ) ^ ls.sum).getD 0

theorem lwS_nonneg (bk : Book) (N : ℕ) (cy : Carry) (ws : List (List Bool)) : 0 ≤ lwS bk N cy ws := by
  unfold lwS
  cases session bk N cy ws <;> simp [pow_nonneg]

theorem lwS_cons (bk : Book) (N : ℕ) (cy : Carry) (ds : List Bool) (ws : List (List Bool)) :
    lwS bk N cy (ds :: ws) = ((window bk N cy ds).map fun p => (2⁻¹ : ℝ) ^ p.1 * lwS bk N p.2 ws).getD 0 := by
  unfold lwS
  simp only [session]
  cases window bk N cy ds with
  | none => simp
  | some p =>
    simp only [Option.bind_some, Option.map_some, Option.getD_some]
    cases session bk N p.2 ws <;> simp [pow_add, mul_comm]

/-- **The run (Kraft)**: over any finite set of decision streams of `W` windows, the weights `2^(−Σ charges)` of those
no window refuses sum to at most 1, from any carry. -/
theorem sum_session_le (bk : Book) (N : ℕ) :
    ∀ (W : ℕ) (cy : Carry) (A : Finset (Fin W → List Bool)), ∑ a ∈ A, lwS bk N cy (List.ofFn a) ≤ 1
  | 0, cy, A => by
    have hc : A.card ≤ 1 := (card_le_univ A).trans (by simp)
    have : ∀ a ∈ A, lwS bk N cy (List.ofFn a) = 1 := fun a _ => by simp [lwS, session]
    rw [sum_congr rfl this, sum_const, nsmul_eq_mul, mul_one]
    exact_mod_cast hc
  | W + 1, cy, A => by
    classical
    rw [← sum_fiberwise_of_maps_to (g := fun a => a 0) (t := A.image fun a => a 0)
      (fun a ha => mem_image_of_mem _ ha)]
    refine le_trans (sum_le_sum fun x _ => ?_) (sum_window_le bk N cy (A.image fun a => a 0))
    have hcons : ∀ a ∈ A.filter (fun a => a 0 = x), lwS bk N cy (List.ofFn a) =
        ((window bk N cy x).map fun p => (2⁻¹ : ℝ) ^ p.1 * lwS bk N p.2 (List.ofFn (Fin.tail a))).getD 0 := by
      intro a ha
      simp only [mem_filter] at ha
      rw [List.ofFn_succ, ha.2, lwS_cons]
      rfl
    rw [sum_congr rfl hcons]
    unfold lw
    cases window bk N cy x with
    | none => simp
    | some p =>
      simp only [Option.map_some, Option.getD_some]
      rw [← mul_sum]
      calc (2⁻¹ : ℝ) ^ p.1 * ∑ a ∈ A with a 0 = x, lwS bk N p.2 (List.ofFn (Fin.tail a))
          ≤ (2⁻¹ : ℝ) ^ p.1 * 1 := by
            gcongr
            rw [← sum_image (f := fun v => lwS bk N p.2 (List.ofFn v)) (fun a ha b hb h => by
              simp only [coe_filter, Set.mem_ofPred_eq] at ha hb
              rw [← Fin.cons_self_tail a, ← Fin.cons_self_tail b, ha.2, hb.2]
              exact congrArg _ h)]
            exact sum_session_le bk N W p.2 _
        _ = (2⁻¹ : ℝ) ^ p.1 := mul_one _

/-- **Kraft's count**: if the weights `2^(−L)` of any finite set of values with `P` sum to at most 1, at most `2^c`
values with `P` have `L ≤ c`. -/
theorem encard_le_of_kraft {α : Type*} (P : α → Prop) (L : α → ℕ)
    (h : ∀ A : Finset α, (∀ a ∈ A, P a) → ∑ a ∈ A, (2⁻¹ : ℝ) ^ L a ≤ 1) (c : ℕ) :
    {a | P a ∧ L a ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞) := by
  by_contra hlt
  rw [not_le] at hlt
  obtain ⟨t, hts, htc⟩ := Set.exists_subset_encard_eq (k := ((2 ^ c + 1 : ℕ) : ℕ∞)) (by
    rw [Nat.cast_add, Nat.cast_one]
    exact Order.add_one_le_of_lt hlt)
  have ht : t.Finite := Set.finite_of_encard_eq_coe htc
  have hcard : ht.toFinset.card = 2 ^ c + 1 := by
    have := ht.encard_eq_coe_toFinset_card
    rw [htc] at this
    exact_mod_cast this.symm
  have hsum := h ht.toFinset (fun a ha => (hts (ht.mem_toFinset.1 ha)).1)
  have hlow : ∑ a ∈ ht.toFinset, (2⁻¹ : ℝ) ^ c ≤ ∑ a ∈ ht.toFinset, (2⁻¹ : ℝ) ^ L a :=
    sum_le_sum fun a ha => pow_le_pow_of_le_one (by norm_num) (by norm_num) (hts (ht.mem_toFinset.1 ha)).2
  rw [sum_const, hcard, nsmul_eq_mul] at hlow
  have : ((2 ^ c + 1 : ℕ) : ℝ) * (2⁻¹ : ℝ) ^ c = 1 + (2⁻¹ : ℝ) ^ c := by
    push_cast
    rw [add_mul, two_pow_mul_inv_pow, one_mul]
  have hpos : 0 < (2⁻¹ : ℝ) ^ c := by positivity
  linarith

/-- **Advice v1's charge counts (Kraft).** -/
theorem encard_charged_le (bk : Book) (N W c : ℕ) :
    {a : Fin W → List Bool | ∃ ls, charges bk N a = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞) := by
  have hset : {a : Fin W → List Bool | ∃ ls, charges bk N a = some ls ∧ ls.sum ≤ c} =
      {a | (charges bk N a).isSome ∧ ((charges bk N a).getD []).sum ≤ c} := by
    ext a
    simp only [Set.mem_ofPred_eq]
    cases charges bk N a <;> simp
  rw [hset]
  apply encard_le_of_kraft
  intro A hA
  refine le_trans (le_of_eq (sum_congr rfl fun a ha => ?_)) (sum_session_le bk N W ⟨bk.priors, 0⟩ A)
  obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 (hA a ha)
  have hs : session bk N ⟨bk.priors, 0⟩ (List.ofFn a) = some ls := hls
  simp only [lwS, hls, hs, Option.map_some, Option.getD_some]

theorem length_session (bk : Book) (N : ℕ) :
    ∀ (cy : Carry) (ws : List (List Bool)) {ls : List ℕ}, session bk N cy ws = some ls → ls.length = ws.length
  | _, [], ls, h => by simp [session] at h; simp [← h]
  | cy, ds :: ws, ls, h => by
    simp only [session, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨p, _, ls', h2, rfl⟩ := h
    simp [length_session bk N p.2 ws h2]

theorem sum_getD_eq_sum : ∀ (ls : List ℕ) (W : ℕ), ls.length = W → ∑ w : Fin W, ls.getD w 0 = ls.sum
  | [], 0, _ => by simp
  | [], _ + 1, h => by simp at h
  | _ :: _, 0, h => by simp at h
  | x :: ls, W + 1, h => by
    rw [Fin.sum_univ_succ]
    simp only [Fin.val_zero, List.getD_cons_zero, Fin.val_succ, List.getD_cons_succ, List.sum_cons]
    rw [sum_getD_eq_sum ls W (by simpa using h)]

theorem sum_bits (bk : Book) (N : ℕ) {W : ℕ} (a : Streams bk N W) :
    ∑ w, bits bk N a w = ((charges bk N a.1).getD []).sum := by
  obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
  simp only [bits, hls, Option.getD_some]
  exact sum_getD_eq_sum ls W (by rw [length_session bk N _ _ hls, List.length_ofFn])

/-- **`BitsCount` holds for advice v1's charge**, under any registered book. -/
theorem bitsCount (bk : Book) (N W : ℕ) : BitsCount (bits bk N (W := W)) := by
  intro c
  calc {a : Streams bk N W | ∑ w, bits bk N a w ≤ c}.encard
      = (Subtype.val '' {a : Streams bk N W | ∑ w, bits bk N a w ≤ c}).encard :=
        (Subtype.val_injective.encard_image _).symm
    _ ≤ {a : Fin W → List Bool | ∃ ls, charges bk N a = some ls ∧ ls.sum ≤ c}.encard := by
        apply Set.encard_le_encard
        rintro _ ⟨a, ha, rfl⟩
        obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
        refine ⟨ls, hls, ?_⟩
        have := sum_bits bk N a
        rw [hls, Option.getD_some] at this
        rw [← this]
        exact ha
    _ ≤ _ := encard_charged_le bk N W c

/-! ### Any run of windows each meeting Kraft -/

/-- A run of an abstract window charge `step`: each window's charge from a carry, the carry passed along. -/
def runOf {σ α : Type*} (step : σ → α → Option (ℕ × σ)) : σ → List α → Option (List ℕ)
  | _, [] => some []
  | s, x :: xs => (step s x).bind fun p => (runOf step p.2 xs).map fun ls => p.1 :: ls

/-- A window's weight under `step`: `2^(−charge)`, or nothing when refused. -/
noncomputable def stepWt {σ α : Type*} (step : σ → α → Option (ℕ × σ)) (s : σ) (x : α) : ℝ :=
  ((step s x).map fun p => (2⁻¹ : ℝ) ^ p.1).getD 0

/-- A run's weight under `step`: `2^(−Σ charges)`, or nothing when a window is refused. -/
noncomputable def runWt {σ α : Type*} (step : σ → α → Option (ℕ × σ)) (s : σ) (xs : List α) : ℝ :=
  ((runOf step s xs).map fun ls => (2⁻¹ : ℝ) ^ ls.sum).getD 0

theorem runWt_cons {σ α : Type*} (step : σ → α → Option (ℕ × σ)) (s : σ) (x : α) (xs : List α) :
    runWt step s (x :: xs) = ((step s x).map fun p => (2⁻¹ : ℝ) ^ p.1 * runWt step p.2 xs).getD 0 := by
  unfold runWt
  simp only [runOf]
  cases step s x with
  | none => simp
  | some p =>
    simp only [Option.bind_some, Option.map_some, Option.getD_some]
    cases runOf step p.2 xs <;> simp [pow_add, mul_comm]

/-- **A run (Kraft)**: when every window's weights sum to at most 1 from any carry, so do a run's over any finite set
of `W` windows' advice. -/
theorem sum_runWt_le {σ α : Type*} (step : σ → α → Option (ℕ × σ))
    (hstep : ∀ (s : σ) (A : Finset α), ∑ x ∈ A, stepWt step s x ≤ 1) :
    ∀ (W : ℕ) (s : σ) (A : Finset (Fin W → α)), ∑ a ∈ A, runWt step s (List.ofFn a) ≤ 1
  | 0, s, A => by
    have hc : A.card ≤ 1 := (card_le_univ A).trans (by simp)
    have : ∀ a ∈ A, runWt step s (List.ofFn a) = 1 := fun a _ => by simp [runWt, runOf]
    rw [sum_congr rfl this, sum_const, nsmul_eq_mul, mul_one]
    exact_mod_cast hc
  | W + 1, s, A => by
    classical
    rw [← sum_fiberwise_of_maps_to (g := fun a => a 0) (t := A.image fun a => a 0)
      (fun a ha => mem_image_of_mem _ ha)]
    refine le_trans (sum_le_sum fun x _ => ?_) (hstep s (A.image fun a => a 0))
    have hcons : ∀ a ∈ A.filter (fun a => a 0 = x), runWt step s (List.ofFn a) =
        ((step s x).map fun p => (2⁻¹ : ℝ) ^ p.1 * runWt step p.2 (List.ofFn (Fin.tail a))).getD 0 := by
      intro a ha
      simp only [mem_filter] at ha
      rw [List.ofFn_succ, ha.2, runWt_cons]
      rfl
    rw [sum_congr rfl hcons]
    unfold stepWt
    cases step s x with
    | none => simp
    | some p =>
      simp only [Option.map_some, Option.getD_some]
      rw [← mul_sum]
      calc (2⁻¹ : ℝ) ^ p.1 * ∑ a ∈ A with a 0 = x, runWt step p.2 (List.ofFn (Fin.tail a))
          ≤ (2⁻¹ : ℝ) ^ p.1 * 1 := by
            gcongr
            rw [← sum_image (f := fun v => runWt step p.2 (List.ofFn v)) (fun a ha b hb h => by
              simp only [coe_filter, Set.mem_ofPred_eq] at ha hb
              rw [← Fin.cons_self_tail a, ← Fin.cons_self_tail b, ha.2, hb.2]
              exact congrArg _ h)]
            exact sum_runWt_le step hstep W p.2 _
        _ = (2⁻¹ : ℝ) ^ p.1 := mul_one _

theorem length_runOf {σ α : Type*} (step : σ → α → Option (ℕ × σ)) :
    ∀ (s : σ) (xs : List α) {ls : List ℕ}, runOf step s xs = some ls → ls.length = xs.length
  | _, [], ls, h => by simp [runOf] at h; simp [← h]
  | s, x :: xs, ls, h => by
    simp only [runOf, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨p, _, ls', h2, rfl⟩ := h
    simp [length_runOf step p.2 xs h2]

/-- **A run's count**: at most `2^c` advice values of `W` windows are charged `c` or less in all by a run whose
windows each meet Kraft. -/
theorem encard_runOf_le {σ α : Type*} (step : σ → α → Option (ℕ × σ))
    (hstep : ∀ (s : σ) (A : Finset α), ∑ x ∈ A, stepWt step s x ≤ 1) (s₀ : σ) (W c : ℕ) :
    {a : Fin W → α | ∃ ls, runOf step s₀ (List.ofFn a) = some ls ∧ ls.sum ≤ c}.encard ≤ ((2 ^ c : ℕ) : ℕ∞) := by
  have hset : {a : Fin W → α | ∃ ls, runOf step s₀ (List.ofFn a) = some ls ∧ ls.sum ≤ c} =
      {a | (runOf step s₀ (List.ofFn a)).isSome ∧ ((runOf step s₀ (List.ofFn a)).getD []).sum ≤ c} := by
    ext a
    simp only [Set.mem_ofPred_eq]
    cases runOf step s₀ (List.ofFn a) <;> simp
  rw [hset]
  apply encard_le_of_kraft
  intro A hA
  refine le_trans (le_of_eq (sum_congr rfl fun a ha => ?_)) (sum_runWt_le step hstep W s₀ A)
  obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 (hA a ha)
  simp only [runWt, hls, Option.map_some, Option.getD_some]

/-! ### Windows that send counts -/

/-- **Prefix-free codes meet Kraft**: if no codeword is a prefix of another's, the weights `2^(−length)` of the
codewords of any finite set of counts sum to at most 1. Each codeword's extensions to the longest one's length are
`2^(M − length)` strings of length `M`, and prefix-freeness keeps them apart. -/
theorem kraft_of_prefixFree {C : Type} (enc : C → Option (List Bool))
    (hpf : ∀ κ κ' w w', enc κ = some w → enc κ' = some w' → w <+: w' → κ = κ') (A : Finset C) :
    ∑ κ ∈ A, ((enc κ).map fun w => (2⁻¹ : ℝ) ^ w.length).getD 0 ≤ 1 := by
  classical
  set B := A.filter fun κ => (enc κ).isSome with hB
  let cw : C → List Bool := fun κ => (enc κ).getD []
  have hcw : ∀ κ ∈ B, enc κ = some (cw κ) := by
    intro κ hκ
    simp only [B, mem_filter] at hκ
    obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hκ.2
    simp [cw, hw]
  let M := B.sup fun κ => (cw κ).length
  have hM : ∀ κ ∈ B, (cw κ).length ≤ M := fun κ hκ => le_sup (f := fun κ => (cw κ).length) hκ
  let E : C → Finset (List Bool) := fun κ =>
    (univ : Finset (Fin (M - (cw κ).length) → Bool)).image fun v => cw κ ++ List.ofFn v
  have hEcard : ∀ κ, (E κ).card = 2 ^ (M - (cw κ).length) := by
    intro κ
    rw [card_image_of_injective _ (fun v v' h => List.ofFn_injective (List.append_cancel_left h))]
    simp
  have hlen : ∀ l ∈ B.biUnion E, l.length = M := by
    intro l hl
    simp only [mem_biUnion, E, mem_image, mem_univ, true_and] at hl
    obtain ⟨κ, hκ, v, rfl⟩ := hl
    simp only [List.length_append, List.length_ofFn]
    have := hM κ hκ
    omega
  have hpd : (B : Set C).PairwiseDisjoint E := by
    intro κ hκ κ' hκ' hne
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro l h1 h2
    simp only [E, mem_image, mem_univ, true_and] at h1 h2
    obtain ⟨v, rfl⟩ := h1
    obtain ⟨v', hv'⟩ := h2
    have p1 : cw κ <+: cw κ ++ List.ofFn v := List.prefix_append _ _
    have p2 : cw κ' <+: cw κ ++ List.ofFn v := hv' ▸ List.prefix_append _ _
    rcases le_total (cw κ).length (cw κ').length with hl | hl
    · exact hne (hpf κ κ' _ _ (hcw κ hκ) (hcw κ' hκ') (List.prefix_of_prefix_length_le p1 p2 hl))
    · exact hne (hpf κ' κ _ _ (hcw κ' hκ') (hcw κ hκ) (List.prefix_of_prefix_length_le p2 p1 hl)).symm
  have hle : ∑ κ ∈ B, (2 : ℝ) ^ (M - (cw κ).length) ≤ 2 ^ M := by
    have h1 : (B.biUnion E).card ≤ 2 ^ M := by
      calc (B.biUnion E).card ≤ (univ : Finset (Fin M → Bool)).card := by
            refine card_le_card_of_injOn (fun l (i : Fin M) => l.getD i false) (by intro a _; exact mem_univ _) ?_
            intro l hl l' hl' h
            apply List.ext_getElem (by rw [hlen l hl, hlen l' hl'])
            intro i h1 h2
            have := congrFun h ⟨i, by rw [← hlen l hl]; exact h1⟩
            simpa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2] using this
        _ = 2 ^ M := by simp
    rw [card_biUnion hpd] at h1
    simp only [hEcard] at h1
    exact_mod_cast h1
  have key : ∀ l, l ≤ M → (2⁻¹ : ℝ) ^ l = 2 ^ (M - l) * (2⁻¹ : ℝ) ^ M := by
    intro l hl
    obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_le hl
    rw [hk, Nat.add_sub_cancel_left, pow_add, ← mul_assoc, mul_comm ((2 : ℝ) ^ k), mul_assoc, two_pow_mul_inv_pow,
      mul_one]
  have hsum : ∑ κ ∈ A, ((enc κ).map fun w => (2⁻¹ : ℝ) ^ w.length).getD 0 = ∑ κ ∈ B, (2⁻¹ : ℝ) ^ (cw κ).length := by
    rw [hB, sum_filter]
    refine sum_congr rfl fun κ _ => ?_
    cases h : enc κ <;> simp [h, cw]
  rw [hsum]
  calc ∑ κ ∈ B, (2⁻¹ : ℝ) ^ (cw κ).length = (∑ κ ∈ B, (2 : ℝ) ^ (M - (cw κ).length)) * (2⁻¹ : ℝ) ^ M := by
        rw [sum_mul]
        exact sum_congr rfl fun κ hκ => key _ (hM κ hκ)
    _ ≤ 2 ^ M * (2⁻¹ : ℝ) ^ M := by gcongr
    _ = 1 := two_pow_mul_inv_pow M

/-- **A prefix-free code is a count code**, charging each counts its codeword's length. -/
def CountCode.ofPrefixFree {C : Type} (enc : C → Option (List Bool))
    (hpf : ∀ κ κ' w w', enc κ = some w → enc κ' = some w' → w <+: w' → κ = κ') : CountCode C where
  len κ := (enc κ).map List.length
  kraft A := le_trans (le_of_eq (sum_congr rfl fun κ _ => by cases enc κ <;> rfl)) (kraft_of_prefixFree enc hpf A)

/-- **A count code for nonzero counts**: the unary code, `n` as `n` zeros and a one, prefix-free, charging `n + 1`. -/
def CountCode.unary : CountCode ℕ :=
  CountCode.ofPrefixFree (fun n => some (List.replicate n false ++ [true])) fun a b w w' ha hb h => by
    cases ha
    cases hb
    have hl := h.length_le
    simp only [List.length_append, List.length_replicate, List.length_singleton] at hl
    rcases Nat.lt_or_ge a b with hab | hab
    · exfalso
      rw [List.prefix_iff_eq_take, List.length_append, List.length_replicate, List.length_singleton,
        List.take_append_of_le_length (by simp; omega), List.take_replicate, min_eq_left (by omega),
        List.replicate_succ'] at h
      simpa using List.append_cancel_left h
    · omega

/-- **Advice v1 is the case of no counts**: under the code that charges nothing for `Unit`, the charges with counts
are advice v1's. -/
def CountCode.zero : CountCode Unit where
  len _ := some 0
  kraft A := by
    have : A.card ≤ 1 := (card_le_univ A).trans (by simp)
    simpa using (show (A.card : ℝ) ≤ 1 by exact_mod_cast this)

theorem chargesCounted_zero (bk : Book) (N : ℕ) {W : ℕ} (a : Fin W → List Bool) :
    chargesCounted bk N CountCode.zero (fun w => (a w, ())) = charges bk N a := by
  suffices h : ∀ (cy : Carry) (ws : List (List Bool)),
      sessionCounted bk N CountCode.zero cy (ws.map fun ds => (ds, ())) = session bk N cy ws by
    simpa [chargesCounted, charges, List.map_ofFn, Function.comp_def] using h ⟨bk.priors, 0⟩ (List.ofFn a)
  intro cy ws
  induction ws generalizing cy with
  | nil => rfl
  | cons ds ws ih =>
    simp only [List.map_cons, sessionCounted, session, windowCounted, ih]
    cases window bk N cy ds <;> simp [CountCode.zero]

variable {C : Type}

/-- The counts' weight under the count code: `2^(−len)`, or nothing when refused. -/
noncomputable def kw (cc : CountCode C) (κ : C) : ℝ := ((cc.len κ).map fun l => (2⁻¹ : ℝ) ^ l).getD 0

theorem kw_nonneg (cc : CountCode C) (κ : C) : 0 ≤ kw cc κ := by
  unfold kw
  cases cc.len κ <;> simp [pow_nonneg]

theorem stepWt_windowCounted (bk : Book) (N : ℕ) (cc : CountCode C) (cy : Carry) (x : List Bool × C) :
    stepWt (windowCounted bk N cc) cy x = lw bk N cy x.1 * kw cc x.2 := by
  unfold stepWt windowCounted lw kw
  cases window bk N cy x.1 with
  | none => simp
  | some p =>
    cases cc.len x.2 with
    | none => simp
    | some l => simp [pow_add, mul_comm]

/-- **One window with counts (Kraft)**: the decisions' weights and the counts' multiply, and each sum is at most 1. -/
theorem sum_windowCounted_le (bk : Book) (N : ℕ) (cc : CountCode C) (cy : Carry) (A : Finset (List Bool × C)) :
    ∑ x ∈ A, stepWt (windowCounted bk N cc) cy x ≤ 1 := by
  classical
  calc ∑ x ∈ A, stepWt (windowCounted bk N cc) cy x = ∑ x ∈ A, lw bk N cy x.1 * kw cc x.2 :=
        sum_congr rfl fun x _ => stepWt_windowCounted bk N cc cy x
    _ ≤ ∑ x ∈ A.image Prod.fst ×ˢ A.image Prod.snd, lw bk N cy x.1 * kw cc x.2 := by
        apply sum_le_sum_of_subset_of_nonneg
        · intro x hx
          exact mem_product.2 ⟨mem_image_of_mem _ hx, mem_image_of_mem _ hx⟩
        · exact fun x _ _ => mul_nonneg (lw_nonneg bk N cy x.1) (kw_nonneg cc x.2)
    _ = (∑ ds ∈ A.image Prod.fst, lw bk N cy ds) * ∑ κ ∈ A.image Prod.snd, kw cc κ := by
        rw [sum_product, sum_mul]
        exact sum_congr rfl fun ds _ => by rw [mul_sum]
    _ ≤ 1 * 1 :=
        mul_le_mul (sum_window_le bk N cy _) (cc.kraft _) (sum_nonneg fun κ _ => kw_nonneg cc κ) zero_le_one
    _ = 1 := one_mul 1

theorem sessionCounted_eq (bk : Book) (N : ℕ) (cc : CountCode C) (cy : Carry) (ws : List (List Bool × C)) :
    sessionCounted bk N cc cy ws = runOf (windowCounted bk N cc) cy ws := by
  induction ws generalizing cy with
  | nil => rfl
  | cons x ws ih => simp only [sessionCounted, runOf, ih]

/-- **The charge with counts counts (Kraft).** -/
theorem encard_chargedCounted_le (bk : Book) (N W c : ℕ) (cc : CountCode C) :
    {a : Fin W → List Bool × C | ∃ ls, chargesCounted bk N cc a = some ls ∧ ls.sum ≤ c}.encard ≤
      ((2 ^ c : ℕ) : ℕ∞) := by
  simpa only [chargesCounted, sessionCounted_eq] using
    encard_runOf_le (windowCounted bk N cc) (sum_windowCounted_le bk N cc) ⟨bk.priors, 0⟩ W c

theorem sum_bitsCounted (bk : Book) (N : ℕ) (cc : CountCode C) {W : ℕ} (a : StreamsCounted bk N cc W) :
    ∑ w, bitsCounted bk N cc a w = ((chargesCounted bk N cc a.1).getD []).sum := by
  obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
  simp only [bitsCounted, hls, Option.getD_some]
  have h : runOf (windowCounted bk N cc) ⟨bk.priors, 0⟩ (List.ofFn a.1) = some ls := by
    rw [← sessionCounted_eq]
    exact hls
  exact sum_getD_eq_sum ls W (by rw [length_runOf _ _ _ h, List.length_ofFn])

/-- **`BitsCount` holds for the charge with counts**, under any registered book and any count code. -/
theorem bitsCount_counted (bk : Book) (N W : ℕ) (cc : CountCode C) : BitsCount (bitsCounted bk N cc (W := W)) := by
  intro c
  calc {a : StreamsCounted bk N cc W | ∑ w, bitsCounted bk N cc a w ≤ c}.encard
      = (Subtype.val '' {a : StreamsCounted bk N cc W | ∑ w, bitsCounted bk N cc a w ≤ c}).encard :=
        (Subtype.val_injective.encard_image _).symm
    _ ≤ {a : Fin W → List Bool × C | ∃ ls, chargesCounted bk N cc a = some ls ∧ ls.sum ≤ c}.encard := by
        apply Set.encard_le_encard
        rintro _ ⟨a, ha, rfl⟩
        obtain ⟨ls, hls⟩ := Option.isSome_iff_exists.1 a.2
        refine ⟨ls, hls, ?_⟩
        have := sum_bitsCounted bk N cc a
        rw [hls, Option.getD_some] at this
        rw [← this]
        exact ha
    _ ≤ _ := encard_chargedCounted_le bk N W c cc

end NetTiming.AdviceV1

namespace NetTiming

variable {L K S : ℕ} {In : Type}

/-- **The leak bound with advice v1's charge**: `encard_accSeqsWithin_le_constantRate`, its `BitsCount` premise
discharged by `AdviceV1.bitsCount`. -/
theorem encard_accSeqsWithin_le_constantRate_v1 {W : ℕ} (bk : AdviceV1.Book) (N : ℕ) (b : Fin W → ℕ) (T : ℕ)
    (ss : SyncSet) (frames : List In → AdviceV1.Streams bk N W → Fin L → List LogicalFrame)
    (op : ℕ → List (Obs L K S) → In) (n D m : ℕ) (hm : n ≤ m * T) :
    (AccSeqsWithin (AdviceV1.bits bk N) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_accSeqsWithin_le_constantRate (AdviceV1.bits bk N) (AdviceV1.bitsCount bk N W) b T ss frames op n D m hm

variable {Z Act Wire : Type}

/-- **The leak bound through the wire, with advice v1's charge**: `encard_decodableWithin_le_compiled`, its
`BitsCount` premise discharged by `AdviceV1.bitsCount`. -/
theorem encard_decodableWithin_le_compiled_v1 {Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire) (T : ℕ)
    (ss : SyncSet) (bk : AdviceV1.Book) (N : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (AdviceV1.Streams bk N W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (AdviceV1.bits bk N) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled I T ss (AdviceV1.bits bk N) (AdviceV1.bitsCount bk N W) b compilerOf env hpre
    hdet n D m hm view hW2 hJ Zs hacc dec hdec

/-- **The leak bound per window, with advice v1's charge**: `encard_decodableWithin_le_compiled_perWindow`, its
`BitsCount` premise discharged by `AdviceV1.bitsCount`. -/
theorem encard_decodableWithin_le_compiled_perWindow_v1 {Ω : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire)
    (T : ℕ) (ss : SyncSet) (bk : AdviceV1.Book) (N : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (AdviceV1.Streams bk N W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (AdviceV1.bits bk N) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z) :
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled_perWindow I T ss (AdviceV1.bits bk N) (AdviceV1.bitsCount bk N W) b compilerOf
    env hpre hdet D view hW2 hJ Zs hacc dec hdec

/-- **The leak bound with the charge with counts**: `encard_accSeqsWithin_le_constantRate`, its `BitsCount` premise
discharged by `AdviceV1.bitsCount_counted`. -/
theorem encard_accSeqsWithin_le_constantRate_v1Counted {W : ℕ} {C : Type} (bk : AdviceV1.Book) (N : ℕ)
    (cc : AdviceV1.CountCode C) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → AdviceV1.StreamsCounted bk N cc W → Fin L → List LogicalFrame)
    (op : ℕ → List (Obs L K S) → In) (n D m : ℕ) (hm : n ≤ m * T) :
    (AccSeqsWithin (AdviceV1.bitsCounted bk N cc) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_accSeqsWithin_le_constantRate (AdviceV1.bitsCounted bk N cc) (AdviceV1.bitsCount_counted bk N W cc) b T ss
    frames op n D m hm

/-- **The leak bound through the wire, with the charge with counts**: `encard_decodableWithin_le_compiled`, its
`BitsCount` premise discharged by `AdviceV1.bitsCount_counted`. -/
theorem encard_decodableWithin_le_compiled_v1Counted {Ω C : Type} {W : ℕ} (I : Interaction L K S Z In Act Wire)
    (T : ℕ) (ss : SyncSet) (bk : AdviceV1.Book) (N : ℕ) (cc : AdviceV1.CountCode C) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (AdviceV1.StreamsCounted bk N cc W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (AdviceV1.bitsCounted bk N cc) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled I T ss (AdviceV1.bitsCounted bk N cc) (AdviceV1.bitsCount_counted bk N W cc) b
    compilerOf env hpre hdet n D m hm view hW2 hJ Zs hacc dec hdec

/-- **The leak bound per window, with the charge with counts**: `encard_decodableWithin_le_compiled_perWindow`, its
`BitsCount` premise discharged by `AdviceV1.bitsCount_counted`. -/
theorem encard_decodableWithin_le_compiled_perWindow_v1Counted {Ω C : Type} {W : ℕ}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : AdviceV1.Book) (N : ℕ)
    (cc : AdviceV1.CountCode C) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (AdviceV1.StreamsCounted bk N cc W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (AdviceV1.bitsCounted bk N cc) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z) :
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞) :=
  encard_decodableWithin_le_compiled_perWindow I T ss (AdviceV1.bitsCounted bk N cc)
    (AdviceV1.bitsCount_counted bk N W cc) b compilerOf env hpre hdet D view hW2 hJ Zs hacc dec hdec

end NetTiming
