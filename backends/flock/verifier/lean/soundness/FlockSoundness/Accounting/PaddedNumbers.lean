import FlockSoundness.Accounting.Padded
import FlockSoundness.Accounting.Numbers

/-!
# The numbers with M1's padded level 0

`padTableError` bounded rationally (`padTableError_le_up`), as `tableError` is in `Fast100`:
`√ρ'` by a rational upper bound `s̄` with `ρ' ≤ s̄²` (`padQueryError_le`), and the padded code's
MCA numerator by a crude closed form (`padMcaA_le`; its terms sit below 2^-200).

At M1's values (#123 at `8d621454`, `backends/flock/live/PROTOCOL.md` §7 item 6: level-0 query
counts `q₀ = 277, 242, 229` at `m = 25, 26, 27`, padding `t = 2q₀`, a pair of extra lanes per rep),
every table is sound to `2^-205` at `η = 1/200` for every statement shape in range (`padTableError_le_m1`).
The rational checks run in Lean's kernel (`decide +kernel`).
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

/-! ## M1's values -/

/-- M1's level-0 query count at `m` (#123's `PROTOCOL.md` §7 item 6), where it differs from
`fast100`'s 218. -/
def m1Queries₀ : ℕ → ℕ
  | 25 => 277
  | 26 => 242
  | 27 => 229
  | _ => 218

/-- M1's schedule at `m`: `fast100` with level 0's query count for the padded rate. -/
def m1Schedule (m : ℕ) : Schedule := ((fast100 m).getD default).withQueries₀ (m1Queries₀ m)

/-- M1's padding at `m`: `t = 2q₀`, every distinct level-0 position the session opens. -/
def m1Pad (m : ℕ) : ℕ := 2 * m1Queries₀ m

/-- The rational upper bounds on `√ρ'` at `m = 25, 26, 27` (eight digits, rounded up). -/
def m1SqrtUp : ℕ → ℚ
  | 25 => 37670511 / 50000000
  | 26 => 72769569 / 100000000
  | 27 => 71692193 / 100000000
  | _ => 1

/-- The rational check at `m = 25, 26, 27`. -/
theorem m1_check : ∀ m, 25 ≤ m → m ≤ 27 →
    padLinkUp (m1Schedule m) (m1Pad m) + padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem m1Schedule_25 : m1Schedule 25 =
    ⟨25, 6, [⟨1, 277, 12, 6⟩, ⟨2, 106, 9, 4⟩, ⟨3, 71, 6, 4⟩, ⟨4, 53, 3, 4⟩], 3⟩ := rfl
theorem m1Schedule_26 : m1Schedule 26 =
    ⟨26, 6, [⟨1, 242, 13, 6⟩, ⟨2, 106, 10, 4⟩, ⟨3, 71, 7, 4⟩, ⟨4, 53, 4, 4⟩], 4⟩ := rfl
theorem m1Schedule_27 : m1Schedule 27 =
    ⟨27, 6, [⟨1, 229, 14, 6⟩, ⟨2, 106, 11, 4⟩, ⟨3, 71, 8, 4⟩, ⟨4, 53, 5, 4⟩], 5⟩ := rfl

theorem m1Schedule_m : ∀ m, 25 ≤ m → m ≤ 27 → (m1Schedule m).m = m := by
  intro m h1 h2
  interval_cases m <;> rfl

theorem padRadius_pos_of (l : Level) (t : ℕ) (sbar : ℚ) (h0 : 0 ≤ sbar)
    (hs : l.padRate t ≤ (sbar : ℝ) ^ 2) (hlt : sbar < 49 / 50) : 0 < l.padRadius t := by
  have hsq : Real.sqrt (l.padRate t) ≤ (sbar : ℝ) := Real.sqrt_le_iff.2 ⟨by exact_mod_cast h0, hs⟩
  have : (sbar : ℝ) < 49 / 50 := by
    have h := (Rat.cast_lt (K := ℝ)).2 hlt
    have e : ((49 / 50 : ℚ) : ℝ) = 49 / 50 := by push_cast; ring
    rwa [e] at h
  unfold Level.padRadius eta
  linarith

/-- M1's padded level 0 meets the rational bounds' conditions. -/
theorem m1_padNumOK : ∀ m, 25 ≤ m → m ≤ 27 →
    PadNumOK ((m1Schedule m).levels.headD default) (m1Pad m) (m1SqrtUp m) := by
  intro m h1 h2
  have key : ∀ (l : Level) (t : ℕ) (sbar : ℚ), 1 ≤ t → l.logLen = l.logCols + 1 →
      2 ^ l.logCols + t ≤ 2 ^ l.logLen → 0 ≤ sbar → l.padRate t ≤ (sbar : ℝ) ^ 2 →
      sbar < 49 / 50 → PadNumOK l t sbar := fun l t sbar a b c d e f =>
    ⟨a, b, c, d, e, padRadius_pos_of l t sbar d e f⟩
  interval_cases m
  · rw [m1Schedule_25]
    refine key _ _ _ (by decide) rfl (by decide) (by norm_num [m1SqrtUp])
      ?_ (by norm_num [m1SqrtUp])
    norm_num [Level.padRate, Level.logLen, m1Pad, m1Queries₀, m1SqrtUp]
  · rw [m1Schedule_26]
    refine key _ _ _ (by decide) rfl (by decide) (by norm_num [m1SqrtUp])
      ?_ (by norm_num [m1SqrtUp])
    norm_num [Level.padRate, Level.logLen, m1Pad, m1Queries₀, m1SqrtUp]
  · rw [m1Schedule_27]
    refine key _ _ _ (by decide) rfl (by decide) (by norm_num [m1SqrtUp])
      ?_ (by norm_num [m1SqrtUp])
    norm_num [Level.padRate, Level.logLen, m1Pad, m1Queries₀, m1SqrtUp]

/-- M1's padded level 0 meets the soundness theorem's conditions. -/
theorem m1_padOK : ∀ m, 25 ≤ m → m ≤ 27 →
    PadOK ((m1Schedule m).levels.headD default) (m1Pad m) := by
  intro m h1 h2
  have hp := m1_padNumOK m h1 h2
  refine ⟨?_, ?_, hp.hk, ?_, hp.hr⟩ <;> (interval_cases m <;> decide)

theorem m1_levels_ok : ∀ m, 25 ≤ m → m ≤ 27 → ∀ l ∈ (m1Schedule m).levels, l.Ok := by
  intro m h1 h2 l hl
  have key : ∀ m', 25 ≤ m' → m' ≤ 27 → ((m1Schedule m').levels.all fun l => decide l.Ok) = true := by
    intro m' h1' h2'
    interval_cases m' <;> decide +kernel
  have := List.all_eq_true.1 (key m h1 h2) l hl
  simpa using this

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

theorem m1_rep_nonneg : ∀ m, 25 ≤ m → m ≤ 27 → ∀ st : Stmt,
    0 ≤ padRepError (m1Schedule m) st (m1Pad m) 2 := by
  intro m h1 h2 st
  have hp := m1_padNumOK m h1 h2
  have hok := m1_levels_ok m h1 h2
  interval_cases m
  · rw [m1Schedule_25] at hp hok ⊢
    exact padRepError_nonneg _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st
  · rw [m1Schedule_26] at hp hok ⊢
    exact padRepError_nonneg _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st
  · rw [m1Schedule_27] at hp hok ⊢
    exact padRepError_nonneg _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st

/-- **The soundness number with M1's padded level 0**, `m = 25, 26, 27`: at most `2^-205` per
table, for every statement shape in range. -/
theorem padTableError_le_m1 : ∀ m, 25 ≤ m → m ≤ 27 → ∀ st : Stmt, st.InRange →
    padTableError (m1Schedule m) st (m1Pad m) 2 ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  intro m h1 h2 st hst
  have hp := m1_padNumOK m h1 h2
  have hok := m1_levels_ok m h1 h2
  have hc := Rat.cast_le (K := ℝ) |>.2 (m1_check m h1 h2)
  have hfin := half_pow_205_le
  interval_cases m
  · rw [m1Schedule_25] at hp hok hc ⊢
    refine (padTableError_le_up _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l
      (List.mem_cons_of_mem _ hl)) st hst (padRepError_nonneg _ _ _ _ _ _ _ _ _ hp
        (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st)).trans (le_trans ?_ hfin)
    refine hc.trans (le_of_eq ?_); push_cast; ring
  · rw [m1Schedule_26] at hp hok hc ⊢
    refine (padTableError_le_up _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l
      (List.mem_cons_of_mem _ hl)) st hst (padRepError_nonneg _ _ _ _ _ _ _ _ _ hp
        (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st)).trans (le_trans ?_ hfin)
    refine hc.trans (le_of_eq ?_); push_cast; ring
  · rw [m1Schedule_27] at hp hok hc ⊢
    refine (padTableError_le_up _ _ _ _ _ _ _ _ _ hp (fun l hl => hok l
      (List.mem_cons_of_mem _ hl)) st hst (padRepError_nonneg _ _ _ _ _ _ _ _ _ hp
        (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st)).trans (le_trans ?_ hfin)
    refine hc.trans (le_of_eq ?_); push_cast; ring

end FlockSoundness.Accounting
