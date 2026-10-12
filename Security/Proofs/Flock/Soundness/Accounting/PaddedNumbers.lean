import Proofs.Flock.Soundness.Accounting.Padded
import Proofs.Flock.Soundness.Accounting.Numbers

/-!
# The numbers with C-Flock ZK's padded level 0

`padTableError` bounded rationally (`padTableError_le_up`), as `tableError` is in `Fast100`:
`√ρ'` by a rational upper bound `s̄` with `ρ' ≤ s̄²` (`padQueryError_le`), and the padded code's
MCA numerator by a crude closed form (`padMcaA_le`; its terms sit below 2^-200).

At C-Flock ZK's values (`verity/ml/flock/live/PROTOCOL.md` §7 item 6, the verifier's `Zk.queries0`: level-0
query counts `q₀ = 277, 242, 229` at `m = 25, 26, 27`, 219 at `m = 28..31` and 218 at `m = 32..35`, padding
`t = 2q₀`, a pair of extra lanes per rep), every table is sound to `2^-205` at `η = 1/200` for every statement
shape in range, at every `m` the verifier serves (`padTableError_le_m1`). The rational checks run in Lean's
kernel (`decide +kernel`).
-/

namespace FlockSoundness.Accounting

open Real

/-! ## Rational bounds on the padded level 0 -/

/-- `(√ρ' + η)^Q`, from a rational upper bound `s̄` on `√ρ'`. -/
def padQueryUp (l : Level) (sbar : ℚ) : ℚ := (sbar + 1 / 200) ^ l.queries

theorem padQueryError_le (l : Level) (t : ℕ) (sbar : ℚ) (h0 : 0 ≤ sbar)
    (hs : l.padRate t ≤ (sbar : ℝ) ^ 2) : l.padQueryError t ≤ (padQueryUp l sbar : ℝ) := by
  unfold Level.padQueryError padQueryUp eta
  push_cast
  have hsq : Real.sqrt (l.padRate t) ≤ (sbar : ℝ) :=
    Real.sqrt_le_iff.2 ⟨by exact_mod_cast h0, hs⟩
  gcongr

/-- A closed-form upper bound on the padded code's MCA numerator at rate `1/2`-length codewords. -/
def padMcaUp (l : Level) : ℚ := (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) * 4 / 3 * 2 ^ l.logLen + 401

/-- **The padded MCA numerator**, for level 0 (`d = cols + 1`) with at least one padding coefficient
and a positive radius. -/
theorem padMcaA_bounds (l : Level) (t : ℕ) (ht1 : 1 ≤ t) (hd : l.logLen = l.logCols + 1)
    (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen) (hr : 0 < l.padRadius t) :
    0 ≤ l.padMcaA t ∧ l.padMcaA t ≤ (padMcaUp l : ℝ) := by
  set n : ℝ := (2 : ℝ) ^ l.logLen with hn_def
  have hn : 0 < n := by positivity
  set ρr : ℝ := ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / n with hρr_def
  set ρ : ℝ := l.padRate t with hρ_def
  have hL : (2 : ℝ) ^ l.logLen = 2 * 2 ^ l.logCols := by rw [hd, pow_succ]; ring
  have hk' : ((2 ^ l.logCols + t : ℕ) : ℝ) ≤ n := by rw [hn_def]; exact_mod_cast hk
  have hρ : ρ = ((2 ^ l.logCols + t : ℕ) : ℝ) / n := rfl
  have hρ_le1 : ρ ≤ 1 := by rw [hρ, div_le_one hn]; exact hk'
  have hρr_le : ρr ≤ ρ := by
    rw [hρr_def, hρ]
    apply div_le_div_of_nonneg_right _ hn.le
    exact_mod_cast Nat.sub_le _ _
  have hρr_ge : 1 / 2 ≤ ρr := by
    rw [hρr_def, le_div_iff₀ hn, hn_def, hL]
    have : (2 ^ l.logCols : ℝ) ≤ ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) := by
      exact_mod_cast (by omega : 2 ^ l.logCols ≤ 2 ^ l.logCols + t - 1)
    linarith
  have hρr_pos : 0 < ρr := by linarith
  have hρr_le1 : ρr ≤ 1 := hρr_le.trans hρ_le1
  have hs : ρr ^ ((1 : ℝ) / 2) = Real.sqrt ρr := (Real.sqrt_eq_rpow ρr).symm
  have hs_le : Real.sqrt ρr ≤ Real.sqrt ρ := Real.sqrt_le_sqrt hρr_le
  have hs_ge : ρr ≤ Real.sqrt ρr := by
    rw [← hs]
    simpa using Real.rpow_le_rpow_of_exponent_ge hρr_pos hρr_le1 (by norm_num : (1 : ℝ) / 2 ≤ 1)
  have hs32 : ρr ^ 2 ≤ ρr ^ ((3 : ℝ) / 2) := by
    have := Real.rpow_le_rpow_of_exponent_ge hρr_pos hρr_le1 (by norm_num : (3 : ℝ) / 2 ≤ 2)
    simpa [Real.rpow_natCast] using this
  set δ : ℝ := l.padRadius t with hδ_def
  have hδ : δ = 1 - Real.sqrt ρ - 1 / 200 := by rw [hδ_def, Level.padRadius, eta]
  have hden : 1 / 200 ≤ 1 - Real.sqrt ρr - δ := by rw [hδ]; linarith
  have hx : Real.sqrt ρr / (1 - Real.sqrt ρr - δ) ≤ 200 := by
    rw [div_le_iff₀ (by linarith)]
    have : Real.sqrt ρr ≤ 1 := Real.sqrt_le_one.2 hρr_le1
    linarith
  set mm : ℝ := max (⌈ρr ^ ((1 : ℝ) / 2) / (1 - ρr ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) 3 with hmm_def
  have hceil : (⌈ρr ^ ((1 : ℝ) / 2) / (1 - ρr ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) ≤ 200 := by
    rw [hs]
    have : ⌈Real.sqrt ρr / (1 - Real.sqrt ρr - δ)⌉ ≤ 200 := Int.ceil_le.2 (by exact_mod_cast hx)
    exact_mod_cast this
  have hmm_le : mm ≤ 200 := max_le hceil (by norm_num)
  have hmm_ge : 3 ≤ mm := le_max_right _ _
  have hδ_le1 : δ ≤ 1 := by rw [hδ]; linarith [Real.sqrt_nonneg ρ]
  have hδρ : δ * ρr ≤ 1 := by
    have := mul_le_mul_of_nonneg_right hδ_le1 hρr_pos.le
    linarith
  have hδρ0 : 0 ≤ δ * ρr := mul_nonneg hr.le hρr_pos.le
  have hnum : 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr ≤
      2 * (401 / 2) ^ 5 + 3 * (401 / 2) := by
    have h5 : (mm + 1 / 2) ^ 5 ≤ (401 / 2 : ℝ) ^ 5 := by gcongr; linarith
    have : 3 * (mm + 1 / 2) * δ * ρr ≤ 3 * (401 / 2) := by
      have hpos : 0 ≤ 3 * (mm + 1 / 2) := by linarith
      calc 3 * (mm + 1 / 2) * δ * ρr = 3 * (mm + 1 / 2) * (δ * ρr) := by ring
        _ ≤ 3 * (mm + 1 / 2) * 1 := mul_le_mul_of_nonneg_left hδρ hpos
        _ ≤ 3 * (401 / 2) := by linarith
    linarith
  have hnum_nonneg : 0 ≤ 2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr := by
    have hpos : 0 ≤ mm + 1 / 2 := by linarith
    have : 3 * (mm + 1 / 2) * δ * ρr = 3 * ((mm + 1 / 2) * (δ * ρr)) := by ring
    rw [this]
    have := mul_nonneg hpos hδρ0
    positivity
  have hnum0 : (0 : ℝ) ≤ 2 * (401 / 2) ^ 5 + 3 * (401 / 2) := by norm_num
  have hρr2 : 1 / 4 ≤ ρr ^ 2 := by nlinarith
  have hT1 : (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) / (3 * ρr ^ ((3 : ℝ) / 2)) * n
      ≤ (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) * 4 / 3 * n := by
    have hD : 3 * (1 / 4) ≤ 3 * ρr ^ ((3 : ℝ) / 2) := by linarith
    calc (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) / (3 * ρr ^ ((3 : ℝ) / 2)) * n
        ≤ (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / (3 * (1 / 4)) * n := by
          have := div_le_div₀ hnum0 hnum (by norm_num) hD
          gcongr
      _ = (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) * 4 / 3 * n := by ring
  have hT2 : (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) ≤ 401 := by
    rw [hs]
    have hsq : 1 / 2 ≤ Real.sqrt ρr := hρr_ge.trans hs_ge
    calc (mm + 1 / 2) / Real.sqrt ρr ≤ (401 / 2) / Real.sqrt ρr := by gcongr; linarith
      _ ≤ (401 / 2) / (1 / 2) := div_le_div_of_nonneg_left (by norm_num) (by norm_num) hsq
      _ = 401 := by norm_num
  have hT1_nonneg : 0 ≤ (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr)
      / (3 * ρr ^ ((3 : ℝ) / 2)) * n :=
    mul_nonneg (div_nonneg hnum_nonneg (by positivity)) hn.le
  have hT2_nonneg : 0 ≤ (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) := by
    have : 0 ≤ mm + 1 / 2 := by linarith
    positivity
  have hA : l.padMcaA t = (2 * (mm + 1 / 2) ^ 5 + 3 * (mm + 1 / 2) * δ * ρr) /
      (3 * ρr ^ ((3 : ℝ) / 2)) * n + (mm + 1 / 2) / ρr ^ ((1 : ℝ) / 2) := rfl
  have hup : ((padMcaUp l : ℚ) : ℝ) = (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) * 4 / 3 * n + 401 := by
    simp only [padMcaUp, hn_def]; push_cast; ring
  rw [hA, hup]
  exact ⟨by linarith, by linarith⟩

/-- The padded level 0's folds and extra lanes, rationally. -/
def padFoldUp (l : Level) (t e : ℕ) : ℚ :=
  l.lanes * (2 * l.padListBound t) / 2 ^ 256 + (2 ^ l.lanes - 1) * padMcaUp l / 2 ^ 256
    + e * padMcaUp l / 2 ^ 256 + e * l.padListBound t / 2 ^ 256

/-- The padded level 0 followed by level `l'`, rationally. -/
def padMidUp (l l' : Level) (t e : ℕ) (sbar : ℚ) : ℚ :=
  padFoldUp l t e
    + pairsUp l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℚ) / 2 ^ 128) ^ 2
    + 2 * l'.listBound / 2 ^ 128
    + padQueryUp l sbar
    + l'.listBound * ceilLog2 l.queries / 2 ^ 128
    + l'.listBound / 2 ^ 128

/-- The padded level 0's conditions for the rational bounds. -/
structure PadNumOK (l : Level) (t : ℕ) (sbar : ℚ) : Prop where
  ht1 : 1 ≤ t
  hd : l.logLen = l.logCols + 1
  hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen
  h0 : 0 ≤ sbar
  hs : l.padRate t ≤ (sbar : ℝ) ^ 2
  hr : 0 < l.padRadius t

theorem padMidLevelError_le (l l' : Level) (t e : ℕ) (sbar : ℚ) (hp : PadNumOK l t sbar) :
    padMidLevelError l l' t e ≤ (padMidUp l l' t e sbar : ℝ) := by
  have hm := (padMcaA_bounds l t hp.ht1 hp.hd hp.hk hp.hr).2
  have hq := padQueryError_le l t sbar hp.h0 hp.hs
  have hl : (0 : ℝ) ≤ 2 ^ l.lanes - 1 := by
    have : (1 : ℝ) ≤ 2 ^ l.lanes := one_le_pow₀ (by norm_num)
    linarith
  unfold padMidLevelError Level.padFoldError padMidUp padFoldUp qK qF
  rw [pairs_eq]
  push_cast
  have e1 : (2 ^ l.lanes - 1) * l.padMcaA t / 2 ^ 256 ≤
      (2 ^ l.lanes - 1) * (padMcaUp l : ℝ) / 2 ^ 256 := by gcongr
  have e2 : (e : ℝ) * l.padMcaA t / 2 ^ 256 ≤ e * (padMcaUp l : ℝ) / 2 ^ 256 := by gcongr
  linarith

/-- One rep with the padded level 0, rationally, over the covered statements. -/
def padRepUp (s : Schedule) (t e : ℕ) (sbar : ℚ) : ℚ :=
  match s.levels with
  | l₀ :: l₁ :: rest =>
      (piopNumMax s.m (l₀.padListBound t) : ℚ) / 2 ^ 128 + padMidUp l₀ l₁ t e sbar + ligUp (l₁ :: rest)
  | _ => 1

/-- The link term with the padded list, rationally, over the covered statements. -/
def padLinkUp (s : Schedule) (t : ℕ) : ℚ :=
  (s.levels.headD default).padListBound t * 1024 * (64 / 2 ^ 128) ^ 2

theorem padRepError_le (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (t e : ℕ)
    (sbar : ℚ) (hp : PadNumOK l₀ t sbar) (hs : ∀ l ∈ l₁ :: rest, l.Ok) (st : Stmt)
    (hst : st.InRange) :
    padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e ≤
      (padRepUp ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e sbar : ℝ) := by
  show piopError m st.kLog st.nClaims (l₀.padListBound t) + padMidLevelError l₀ l₁ t e +
      ligeritoError (l₁ :: rest) ≤ ((((piopNumMax m (l₀.padListBound t) : ℚ) / 2 ^ 128 +
        padMidUp l₀ l₁ t e sbar + ligUp (l₁ :: rest)) : ℚ) : ℝ)
  push_cast
  have h1 := piopError_le m (l₀.padListBound t) st hst
  have h2 := padMidLevelError_le l₀ l₁ t e sbar hp
  have h3 := ligeritoError_le (l₁ :: rest) hs
  push_cast at h1
  linarith

theorem padLinkError_le (s : Schedule) (st : Stmt) (t : ℕ) (hst : st.InRange) :
    padLinkError s st t ≤ (padLinkUp s t : ℝ) := by
  obtain ⟨_, hR, hp⟩ := hst
  unfold padLinkError padLinkUp qF
  push_cast
  have hR' : (st.nRegions : ℝ) ≤ 1024 := by exact_mod_cast hR
  have hp' : (st.mPts : ℝ) ≤ 64 := by exact_mod_cast hp
  gcongr

theorem padTableError_le_up (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (t e : ℕ)
    (sbar : ℚ) (hp : PadNumOK l₀ t sbar) (hs : ∀ l ∈ l₁ :: rest, l.Ok) (st : Stmt)
    (hst : st.InRange) (h0 : 0 ≤ padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e) :
    padTableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e ≤
      ((padLinkUp ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t +
        padRepUp ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e sbar ^ 2 : ℚ) : ℝ) := by
  unfold padTableError
  push_cast
  have h1 := padLinkError_le ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t hst
  have h2 := padRepError_le m k0 l₀ l₁ rest ext t e sbar hp hs st hst
  nlinarith

/-! ## C-Flock ZK's values -/

/-- C-Flock ZK's level-0 query count at `m` (`PROTOCOL.md` §7 item 6, `Zk.queries0`), where it differs from
`fast100`'s 218. -/
def m1Queries₀ : ℕ → ℕ
  | 25 => 277
  | 26 => 242
  | 27 => 229
  | 28 | 29 | 30 | 31 => 219
  | _ => 218

/-- C-Flock ZK's schedule at `m`: `fast100` with level 0's query count for the padded rate. -/
def m1Schedule (m : ℕ) : Schedule := ((fast100 m).getD default).withQueries₀ (m1Queries₀ m)

/-- C-Flock ZK's padding at `m`: `t = 2q₀`, every distinct level-0 position the session opens. -/
def m1Pad (m : ℕ) : ℕ := 2 * m1Queries₀ m

/-- The rational upper bounds on `√ρ'` at `25 ≤ m ≤ 35` (eight digits, rounded up). -/
def m1SqrtUp : ℕ → ℚ
  | 25 => 37670511 / 50000000
  | 26 => 72769569 / 100000000
  | 27 => 71692193 / 100000000
  | 28 | 29 | 30 => 35414363 / 50000000
  | 31 => 70769727 / 100000000
  | 32 => 35370037 / 50000000
  | 33 => 35362689 / 50000000
  | 34 => 70718029 / 100000000
  | 35 => 35357177 / 50000000
  | _ => 1

/-- The rational check for `25 ≤ m ≤ 33`. -/
theorem m1_check_le_33 : ∀ m, 25 ≤ m → m ≤ 33 →
    padLinkUp (m1Schedule m) (m1Pad m) + padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check for `m = 34, 35`. -/
theorem m1_check_34_35 : ∀ m, 34 ≤ m → m ≤ 35 →
    padLinkUp (m1Schedule m) (m1Pad m) + padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check at `25 ≤ m ≤ 35`. -/
theorem m1_check : ∀ m, 25 ≤ m → m ≤ 35 →
    padLinkUp (m1Schedule m) (m1Pad m) + padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  rcases le_or_gt m 33 with h | h
  · exact m1_check_le_33 m h1 h
  · exact m1_check_34_35 m (by omega) h2

theorem padRadius_pos_of (l : Level) (t : ℕ) (sbar : ℚ) (h0 : 0 ≤ sbar)
    (hs : l.padRate t ≤ (sbar : ℝ) ^ 2) (hlt : sbar < 49 / 50) : 0 < l.padRadius t := by
  have hsq : Real.sqrt (l.padRate t) ≤ (sbar : ℝ) := Real.sqrt_le_iff.2 ⟨by exact_mod_cast h0, hs⟩
  have : (sbar : ℝ) < 49 / 50 := by
    have h := (Rat.cast_lt (K := ℝ)).2 hlt
    have e : ((49 / 50 : ℚ) : ℝ) = 49 / 50 := by push_cast; ring
    rwa [e] at h
  unfold Level.padRadius eta
  linarith

/-- `ρ' ≤ s̄²`, from the rational inequality `2^cols + t ≤ s̄²·2^d`. -/
theorem padRate_le_sq (l : Level) (t : ℕ) (sbar : ℚ)
    (h : ((2 ^ l.logCols + t : ℕ) : ℚ) ≤ sbar ^ 2 * 2 ^ l.logLen) : l.padRate t ≤ (sbar : ℝ) ^ 2 := by
  have h' := (Rat.cast_le (K := ℝ)).2 h
  push_cast at h'
  unfold Level.padRate
  rw [div_le_iff₀ (by positivity)]
  push_cast
  linarith

/-- C-Flock ZK's schedule at `m` has two levels or more, and its padded level 0 and its levels meet the
conditions of the rational bounds and of the soundness theorem: one check, run in the kernel. -/
def m1Facts (m : ℕ) : Bool :=
  let s := m1Schedule m
  let l := s.levels.headD default
  let t := m1Pad m
  decide (2 ≤ s.levels.length) && decide (s.m = m) && decide (1 ≤ t) && decide (l.logLen = l.logCols + 1) &&
    decide (l.logLen ≤ 40) && decide (t ≤ 2 ^ l.logCols) && decide (2 ^ l.logCols + t ≤ 2 ^ l.logLen) &&
    decide (1 < 2 ^ l.logCols + t) && decide (0 ≤ m1SqrtUp m) &&
    decide (((2 ^ l.logCols + t : ℕ) : ℚ) ≤ m1SqrtUp m ^ 2 * 2 ^ l.logLen) && decide (m1SqrtUp m < 49 / 50) &&
    s.levels.all (fun l => decide l.Ok)

theorem m1Facts_all : ∀ m, 25 ≤ m → m ≤ 35 → m1Facts m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem m1Facts_spec {m : ℕ} (h : m1Facts m = true) :
    2 ≤ (m1Schedule m).levels.length ∧ (m1Schedule m).m = m ∧
      PadNumOK ((m1Schedule m).levels.headD default) (m1Pad m) (m1SqrtUp m) ∧
      PadOK ((m1Schedule m).levels.headD default) (m1Pad m) ∧ ∀ l ∈ (m1Schedule m).levels, l.Ok := by
  simp only [m1Facts, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, and_assoc] at h
  obtain ⟨h2, hm, ht1, hd, hd40, ht, hk, hk1, h0, hq, hlt, hok⟩ := h
  have hs := padRate_le_sq _ _ _ hq
  have hr := padRadius_pos_of _ _ _ h0 hs hlt
  exact ⟨h2, hm, ⟨ht1, hd, hk, h0, hs, hr⟩, ⟨hd40, ht, hk, hk1, hr⟩, hok⟩

theorem m1Schedule_m : ∀ m, 25 ≤ m → m ≤ 35 → (m1Schedule m).m = m := fun m h1 h2 =>
  (m1Facts_spec (m1Facts_all m h1 h2)).2.1

/-- C-Flock ZK's padded level 0 meets the rational bounds' conditions. -/
theorem m1_padNumOK : ∀ m, 25 ≤ m → m ≤ 35 →
    PadNumOK ((m1Schedule m).levels.headD default) (m1Pad m) (m1SqrtUp m) := fun m h1 h2 =>
  (m1Facts_spec (m1Facts_all m h1 h2)).2.2.1

/-- C-Flock ZK's padded level 0 meets the soundness theorem's conditions. -/
theorem m1_padOK : ∀ m, 25 ≤ m → m ≤ 35 →
    PadOK ((m1Schedule m).levels.headD default) (m1Pad m) := fun m h1 h2 =>
  (m1Facts_spec (m1Facts_all m h1 h2)).2.2.2.1

theorem m1_levels_ok : ∀ m, 25 ≤ m → m ≤ 35 → ∀ l ∈ (m1Schedule m).levels, l.Ok := fun m h1 h2 =>
  (m1Facts_spec (m1Facts_all m h1 h2)).2.2.2.2

theorem padRepError_nonneg (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (t e : ℕ)
    (sbar : ℚ) (hp : PadNumOK l₀ t sbar) (hs : ∀ l ∈ l₁ :: rest, l.Ok) (st : Stmt) :
    0 ≤ padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e := by
  show 0 ≤ piopError m st.kLog st.nClaims (l₀.padListBound t) + padMidLevelError l₀ l₁ t e +
      ligeritoError (l₁ :: rest)
  have hq : (0 : ℝ) < qF := by unfold qF; positivity
  have hqK : (0 : ℝ) < qK := by unfold qK; positivity
  have hmca := (padMcaA_bounds l₀ t hp.ht1 hp.hd hp.hk hp.hr).1
  have h1 : 0 ≤ piopError m st.kLog st.nClaims (l₀.padListBound t) := by unfold piopError; positivity
  have h3 := ligeritoError_nonneg (l₁ :: rest) hs
  have h2l : (1 : ℝ) ≤ 2 ^ l₀.lanes := one_le_pow₀ (by norm_num)
  have hQ : 0 ≤ l₀.padQueryError t := by unfold Level.padQueryError eta; positivity
  have hp2 := pairs_nonneg l₁.listBound
  have hfold : 0 ≤ l₀.padFoldError t e (l₀.padListBound t) := by
    unfold Level.padFoldError
    have t1 : (0 : ℝ) ≤ l₀.lanes * (2 * (l₀.padListBound t : ℝ)) / qK :=
      div_nonneg (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg (by norm_num) (Nat.cast_nonneg _)))
        hqK.le
    have t2 : (0 : ℝ) ≤ (2 ^ l₀.lanes - 1) * l₀.padMcaA t / qK :=
      div_nonneg (mul_nonneg (by linarith) hmca) hqK.le
    have t3 : (0 : ℝ) ≤ e * l₀.padMcaA t / qK := div_nonneg (mul_nonneg (Nat.cast_nonneg _) hmca) hqK.le
    have t4 : (0 : ℝ) ≤ e * (l₀.padListBound t : ℝ) / qK :=
      div_nonneg (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) hqK.le
    linarith
  have h2 : 0 ≤ padMidLevelError l₀ l₁ t e := by
    unfold padMidLevelError
    have a1 : 0 ≤ pairs l₁.listBound * (((l₁.logCols + l₁.lanes : ℕ) : ℝ) / qF) ^ 2 :=
      mul_nonneg hp2 (sq_nonneg _)
    have a2 : (0 : ℝ) ≤ 2 * l₁.listBound / qF := by positivity
    have a3 : (0 : ℝ) ≤ l₁.listBound * ceilLog2 l₀.queries / qF := by positivity
    have a4 : (0 : ℝ) ≤ l₁.listBound / qF := by positivity
    linarith
  linarith

theorem padRepError_nonneg_of (s : Schedule) (st : Stmt) (t e : ℕ) (sbar : ℚ) (h2 : 2 ≤ s.levels.length)
    (hp : PadNumOK (s.levels.headD default) t sbar) (hok : ∀ l ∈ s.levels, l.Ok) :
    0 ≤ padRepError s st t e := by
  obtain ⟨m, k0, levels, ext⟩ := s
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at h2
  · simp at h2
  · exact padRepError_nonneg m k0 l₀ l₁ rest ext t e sbar hp
      (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st

/-- A schedule's table with the padded level 0 is sound to `2^-205` once its rational check passes. -/
theorem padTableError_le_of_check (s : Schedule) (st : Stmt) (hst : st.InRange) (t e : ℕ) (sbar : ℚ)
    (h2 : 2 ≤ s.levels.length) (hp : PadNumOK (s.levels.headD default) t sbar) (hok : ∀ l ∈ s.levels, l.Ok)
    (hc : padLinkUp s t + padRepUp s t e sbar ^ 2 ≤ (1 / 2) ^ 205) :
    padTableError s st t e ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  obtain ⟨m, k0, levels, ext⟩ := s
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at h2
  · simp at h2
  · have hs : ∀ l ∈ l₁ :: rest, l.Ok := fun l hl => hok l (List.mem_cons_of_mem _ hl)
    refine (padTableError_le_up m k0 l₀ l₁ rest ext t e sbar hp hs st hst
      (padRepError_nonneg m k0 l₀ l₁ rest ext t e sbar hp hs st)).trans (le_trans ?_ half_pow_205_le)
    have h := Rat.cast_le (K := ℝ) |>.2 hc
    refine h.trans (le_of_eq ?_); push_cast; ring

theorem m1_rep_nonneg : ∀ m, 25 ≤ m → m ≤ 35 → ∀ st : Stmt,
    0 ≤ padRepError (m1Schedule m) st (m1Pad m) 2 := by
  intro m h1 h2 st
  have h := m1Facts_spec (m1Facts_all m h1 h2)
  exact padRepError_nonneg_of (m1Schedule m) st (m1Pad m) 2 (m1SqrtUp m) h.1 h.2.2.1 h.2.2.2.2

/-- **The soundness number with C-Flock ZK's padded level 0**, `25 ≤ m ≤ 35`: at most `2^-205` per
table, for every statement shape in range. -/
theorem padTableError_le_m1 : ∀ m, 25 ≤ m → m ≤ 35 → ∀ st : Stmt, st.InRange →
    padTableError (m1Schedule m) st (m1Pad m) 2 ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  intro m h1 h2 st hst
  have h := m1Facts_spec (m1Facts_all m h1 h2)
  exact padTableError_le_of_check (m1Schedule m) st hst (m1Pad m) 2 (m1SqrtUp m) h.1 h.2.2.1 h.2.2.2.2
    (m1_check m h1 h2)

end FlockSoundness.Accounting
