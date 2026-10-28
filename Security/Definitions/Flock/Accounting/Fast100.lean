import Definitions.Flock.Accounting.Bound
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.GCongr

/-!
# Rational upper bounds for `fast100`'s error, term by term

Each real term of the error is bounded by a rational one (`…Up`): `√(2^-r)` by a rational upper bound (`sqrtUp`), the
queries (`queryUp`), the mutual-correlated-agreement numerator of BCHKS25 Theorem 4.6 by a crude closed form (`mcaUp`),
and the folds, the level pairs and the last level; `ligUp`, `repUp` and `linkUp` sum them. `Level.Ok` and
`Stmt.InRange` are the levels and statements the bound covers. The bounds themselves are proved in
`Proofs.Flock.Soundness.Accounting.Fast100`.
-/

namespace FlockSoundness.Accounting

open Real

/-! ## Rational upper bounds, term by term -/

/-- A rational upper bound on `√(2^-r)`: exact for even `r`, `√(1/2) ≤ 0.70710679` for odd. -/
def sqrtUp (r : ℕ) : ℚ := (if r % 2 = 0 then 1 else 70710679 / 10 ^ 8) * (1 / 2) ^ (r / 2)

/-- `(√ρ + η)^Q`, rationally. -/
def queryUp (l : Level) : ℚ := (sqrtUp l.logInvRate + 1 / 200) ^ l.queries

/-- A closed-form upper bound on the MCA numerator `a` (`Level.mcaA`) for rates `2^-1 … 2^-7`. -/
def mcaUp (l : Level) : ℚ :=
  (2 * (401 / 2) ^ 5 + 3 * (401 / 2)) / 3 * 2 ^ (3 * l.logInvRate + 2 + l.logCols)
    + 401 / 2 * 2 ^ (l.logInvRate + 1)

/-- The folding rounds of a level, rationally. -/
def foldUp (l : Level) (L : ℕ) : ℚ :=
  l.lanes * (2 * L) / 2 ^ 256 + (2 ^ l.lanes - 1) * mcaUp l / 2 ^ 256

/-- Unordered pairs, rationally. -/
def pairsUp (L : ℕ) : ℚ := (L : ℚ) * ((L : ℚ) - 1) / 2

def midUp (l l' : Level) : ℚ :=
  foldUp l (max l.listBound l'.listBound)
    + pairsUp l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℚ) / 2 ^ 128) ^ 2
    + 2 * l'.listBound / 2 ^ 128
    + queryUp l
    + l'.listBound * ceilLog2 l.queries / 2 ^ 128
    + l'.listBound / 2 ^ 128

def lastUp (l : Level) : ℚ :=
  foldUp l l.listBound + queryUp l + ceilLog2 l.queries / 2 ^ 128 + 1 / 2 ^ 128

def ligUp : List Level → ℚ
  | [] => 0
  | [l] => lastUp l
  | l :: l' :: rest => midUp l l' + ligUp (l' :: rest)

/-- The level conditions the rational bounds need: rates `2^-1 … 2^-7`, at least two message
columns. Every `fast100` level satisfies them (checked below). -/
def Level.Ok (l : Level) : Prop := 1 ≤ l.logInvRate ∧ l.logInvRate ≤ 7 ∧ 1 ≤ l.logCols

instance instDecidableOk (l : Level) : Decidable l.Ok := by unfold Level.Ok; infer_instance

/-! ## The statement ranges -/

/-- The statement shapes covered: at most `2^27` witness bits per block, 1024 public regions,
64 coordinates per link point. -/
def Stmt.InRange (st : Stmt) : Prop := st.kLog ≤ 27 ∧ st.nRegions ≤ 1024 ∧ st.mPts ≤ 64

/-- The worst-case numerator of `piopError` over the covered statements. -/
def piopNumMax (m L₀ : ℕ) : ℕ :=
  (L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6)) + L₀ * m
    + L₀ * (2 + 2 * 21 + 63) + L₀ * (7 * 2050 + 1) + L₀

/-- The rational bound on one rep over the covered statements. -/
def repUp (s : Schedule) : ℚ := (piopNumMax s.m s.listBound₀ : ℚ) / 2 ^ 128 + ligUp s.levels

/-- The rational bound on the link term over the covered statements. -/
def linkUp (s : Schedule) : ℚ := s.listBound₀ * 1024 * (64 / 2 ^ 128) ^ 2

end FlockSoundness.Accounting
