module

public import Definitions.Flock.Accounting.Bound
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The soundness bound with Flock ZK's padded level 0, a function of the padding `t` and the extra lanes `e`. -/

@[expose] public section

namespace FlockSoundness.Accounting

namespace Level

/-- The padded code's rate `ρ' = (2^cols + t)/2^d`. -/
noncomputable def padRate (l : Level) (t : ℕ) : ℝ := ((2 ^ l.logCols + t : ℕ) : ℝ) / 2 ^ l.logLen

/-- The padded code's proximity radius `γ' = 1 − √ρ' − η`. -/
noncomputable def padRadius (l : Level) (t : ℕ) : ℝ := 1 - Real.sqrt (l.padRate t) - eta

/-- ArkLib's Johnson bound at the padded rate: `⌊1/(2ηρ')⌋ = ⌊100·2^d/(2^cols + t)⌋`. -/
def padListBound (l : Level) (t : ℕ) : ℕ := 100 * 2 ^ l.logLen / (2 ^ l.logCols + t)

/-- The level's queries on the padded code: `(√ρ' + η)^Q`. -/
noncomputable def padQueryError (l : Level) (t : ℕ) : ℝ := (Real.sqrt (l.padRate t) + eta) ^ l.queries

/-- BCHKS25 Theorem 4.6's numerator for the padded code `RS[n = 2^d, k = 2^cols + t]` at `γ'`. -/
noncomputable def padMcaA (l : Level) (t : ℕ) : ℝ :=
  let n : ℝ := (2 : ℝ) ^ l.logLen
  let ρ : ℝ := ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / n
  let δ : ℝ := l.padRadius t
  let mm : ℝ := max (⌈ρ ^ ((1 : ℝ) / 2) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) 3
  (2 * (mm + 1/2) ^ 5 + 3 * (mm + 1/2) * δ * ρ) / (3 * ρ ^ ((3 : ℝ) / 2)) * n
    + (mm + 1/2) / ρ ^ ((1 : ℝ) / 2)

/-- The padded level 0's folds for a list of `L`: the lane rounds (as `foldError`, on the padded
code), then the `e` extra lanes' coins, each an affine line for mutual correlated agreement and a
coin on the restarted claim. -/
noncomputable def padFoldError (l : Level) (t e L : ℕ) : ℝ :=
  l.lanes * (2 * L) / qK + (2 ^ l.lanes - 1) * l.padMcaA t / qK + e * l.padMcaA t / qK
    + e * L / qK

end Level

/-- The padded level 0 followed by level `l'`: its folds and extra lanes, `l'`'s binding and
batching coins, and level 0's queries on the padded code (as `midLevelError`). -/
noncomputable def padMidLevelError (l l' : Level) (t e : ℕ) : ℝ :=
  l.padFoldError t e (l.padListBound t)
    + pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2
    + 2 * l'.listBound / qF
    + l.padQueryError t
    + l'.listBound * ceilLog2 l.queries / qF
    + l'.listBound / qF

noncomputable def padLigeritoError (s : Schedule) (t e : ℕ) : ℝ :=
  match s.levels with
  | l₀ :: l₁ :: rest =>
      (l₀.padListBound t : ℝ) / qF + padMidLevelError l₀ l₁ t e + ligeritoError (l₁ :: rest)
  | _ => 1

/-- **One rep with the padded level 0.** As `repError`, with the padded list bound before
Ligerito (whose level-0 out-of-domain coin `piopError` counts) and the padded level 0 in it. -/
noncomputable def padRepError (s : Schedule) (st : Stmt) (t e : ℕ) : ℝ :=
  match s.levels with
  | l₀ :: l₁ :: rest =>
      piopError s.m st.kLog st.nClaims (l₀.padListBound t) + padMidLevelError l₀ l₁ t e
        + ligeritoError (l₁ :: rest)
  | _ => 1

/-- The link points, over the padded list. -/
noncomputable def padLinkError (s : Schedule) (st : Stmt) (t : ℕ) : ℝ :=
  ((s.levels.headD default).padListBound t : ℝ) * st.nRegions * ((st.mPts : ℝ) / qF) ^ 2

noncomputable def padTableError (s : Schedule) (st : Stmt) (t e : ℕ) : ℝ :=
  padLinkError s st t + padRepError s st t e ^ 2

/-- The padded level 0's conditions: the padded code fits the novel basis at `d ≤ 40` bits, and
its radius is positive. -/
structure PadOK (l : Level) (t : ℕ) : Prop where
  hd : l.logLen ≤ 40
  ht : t ≤ 2 ^ l.logCols
  hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen
  hk1 : 1 < 2 ^ l.logCols + t
  hr : 0 < l.padRadius t

/-- A schedule with level 0's query count replaced (M1 sizes it for the padded rate). -/
def Schedule.withQueries₀ (s : Schedule) (q₀ : ℕ) : Schedule :=
  { s with levels := s.levels.modifyHead fun l => { l with queries := q₀ } }

end FlockSoundness.Accounting
