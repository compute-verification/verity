import Definitions.Flock.Accounting.Schedule
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# The soundness bound as a function of the parameters

`repError s st` is the probability that one rep accepts when the committed values do not satisfy
the statement, and `tableError s st` the probability that a table's two reps both accept. They
are functions of the `fast100` schedule `s` (PROTOCOL.md §15) and of the statement's shape `st`
(§16), with one term per verifier coin of PROTOCOL.md §9–§13 (the coin's step id is in each
docstring). The soundness theorem proves that these are bounds; `Accounting.Fast100` evaluates them.

**Where each term comes from.** A coin drawn after the prover has fixed a nonzero polynomial of
total degree `D` hits a zero with probability at most `D / |field|` (Schwartz–Zippel). In the
list-decoding regime the level-0 commitment is close to up to `L₀` codewords, and a claim must
fail for every one of them, so every term before Ligerito's binding carries a factor `L₀` (paper
Appendix A, "Security"). Ligerito's per-level terms follow paper Appendix C.3 with the spec's
changes (§19, A11): folding challenges in GF(2^256), two out-of-domain samples per committed
level, stratified queries.

**List sizes.** `Level.listBound` is ArkLib's proved Johnson bound for interleaved Reed–Solomon
codes (`irs_lambda_le_johnson_mds`): at radius `1 − √ρ − η` at most `1 / (2ηρ) = 100·2^r`
codewords. Flock's own accounting uses the sharper `1 / (2η√ρ)`; the difference moves the total
by less than 0.001 bit.
-/

namespace FlockSoundness.Accounting

/-- `|GF(2^128)|`. -/
noncomputable def qF : ℝ := 2 ^ 128

/-- `|GF(2^256)|`. -/
noncomputable def qK : ℝ := 2 ^ 256

/-- The Johnson slack `η` of every level: the analysis's choice, which the verifier never reads.
Over the GF(2^256) fold field the MCA term stays far below the query terms at `1/200`. -/
noncomputable def eta : ℝ := 1 / 200

/-- The statement's shape: the parameters of §16 the bound depends on. -/
structure Stmt where
  /-- `log2` of the witness bits per block (`k_log`). -/
  kLog : ℕ
  /-- Public regions; each gives one extra claim at each of the two link points (§16.2). -/
  nRegions : ℕ
  /-- Coordinates of a link point (`m_pts`). -/
  mPts : ℕ
  deriving DecidableEq, Repr

/-- Claims opened per rep: `ab`, `c` and two per region (§11). -/
def Stmt.nClaims (st : Stmt) : ℕ := 2 + 2 * st.nRegions

/-- `⌈log₂ n⌉` for `1 ≤ n ≤ 2^64`: the number of consistency-batching coins for `n` queries
(§13.3 step 8, `sample_f128_vec(⌈log2 Q⌉)`). -/
def ceilLog2 (n : ℕ) : ℕ := ((List.range 65).find? fun k => decide (n ≤ 2 ^ k)).getD 64

example : [218, 106, 71, 53, 43, 36, 32].map ceilLog2 = [8, 7, 7, 6, 6, 6, 5] := by decide

/-- Unordered pairs among `L` list elements. -/
noncomputable def pairs (L : ℕ) : ℝ := (L : ℝ) * ((L : ℝ) - 1) / 2

namespace Level

/-- The code rate `ρ = 2^-r`. -/
noncomputable def rate (l : Level) : ℝ := (2⁻¹ : ℝ) ^ l.logInvRate

/-- ArkLib's proved list bound at this level's radius: `1 / (2ηρ) = 100·2^r`. -/
def listBound (l : Level) : ℕ := 100 * 2 ^ l.logInvRate

/-- The proximity radius `γ = 1 − √ρ − η`. -/
noncomputable def radius (l : Level) : ℝ := 1 - Real.sqrt l.rate - eta

/-- Queries on this level (§13.5): a committed word that is `γ`-far from the fold survives all
`Q` stratified queries with probability at most `(1 − γ)^Q = (√ρ + η)^Q`. -/
noncomputable def queryError (l : Level) : ℝ := (Real.sqrt l.rate + eta) ^ l.queries

/-- The numerator `a` of the Johnson-range mutual-correlated-agreement bound `a / |field|`
(BCHKS25 Theorem 4.6's printed numerator, which `mcaError_le_bchks25` proves from DKT26), for
this level's code `RS[n = 2^d, k = 2^logCols]` at radius `γ`. -/
noncomputable def mcaA (l : Level) : ℝ :=
  let n : ℝ := (2 : ℝ) ^ l.logLen
  let ρ : ℝ := ((2 ^ l.logCols - 1 : ℕ) : ℝ) / n
  let δ : ℝ := l.radius
  let mm : ℝ := max (⌈ρ ^ ((1 : ℝ) / 2) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ : ℝ) 3
  (2 * (mm + 1/2) ^ 5 + 3 * (mm + 1/2) * δ * ρ) / (3 * ρ ^ ((3 : ℝ) / 2)) * n
    + (mm + 1/2) / ρ ^ ((1 : ℝ) / 2)

/-- This level's folding rounds (§13.3 steps 4 and the recursive rounds): each GF(2^256)
challenge `r` fails the round's quadratic sumcheck for some list element with probability at most
`2L / 2^256`, and breaks mutual correlated agreement with probability at most `2^(k−i)·a / 2^256`
at round `i` (the row union of paper Lemma 9); summed over the `k` rounds. -/
noncomputable def foldError (l : Level) (L : ℕ) : ℝ :=
  l.lanes * (2 * L) / qK + (2 ^ l.lanes - 1) * l.mcaA / qK

end Level

/-- Level `l` followed by level `l'`, which is committed, bound by its two out-of-domain samples,
and then checked against `l` by `l`'s queries:
* folds of `l`;
* binding of `l'` (§13.3 step 6): two distinct list elements agree at both out-of-domain points
  with probability at most `(μ/2^128)^2`, for `μ` the variables of `l'`'s table, over all pairs;
* the two out-of-domain claims' batching coins `β`;
* `l`'s queries;
* the consistency coin `α` (degree `⌈log2 Q⌉`) and its batching coin `β`. -/
noncomputable def midLevelError (l l' : Level) : ℝ :=
  l.foldError (max l.listBound l'.listBound)
    + pairs l'.listBound * (((l'.logCols + l'.lanes : ℕ) : ℝ) / qF) ^ 2
    + 2 * l'.listBound / qF
    + l.queryError
    + l'.listBound * ceilLog2 l.queries / qF
    + l'.listBound / qF

/-- The last level: its folds, its queries against the clear residual `yr`, and the final
consistency and batching coins (no list: `yr` is sent in the clear). -/
noncomputable def lastLevelError (l : Level) : ℝ :=
  l.foldError l.listBound + l.queryError + ceilLog2 l.queries / qF + 1 / qF

/-- Ligerito (§13), level by level. -/
noncomputable def ligeritoError : List Level → ℝ
  | [] => 0
  | [l] => lastLevelError l
  | l :: l' :: rest => midLevelError l l' + ligeritoError (l' :: rest)

/-- Everything before Ligerito's first fold, each term times the level-0 list bound `L₀`:
* Z3, `r[13 … m−1]`: the eq-reduction of a nonzero Boolean table, degree `m − 13` (the seven
  pinned coordinates lose nothing, paper §4.3); plus the exceptional event that a sampled
  coordinate equals 1, which the `(G(1), G(∞))` encoding with `inv(0) = 0` cannot survive
  (§9.2), degree `m − 13`, charged once;
* Z5, `z`: the univariate skip, degree 127 (§19 A3);
* Z6, `ρ_i`: `m − 6` quadratic residual rounds;
* the pin: the constant column's multilinear extension equals 1 at `ρ_out` although the pin is 0
  in some block, degree `m − k_log` (§10, the `β` term, §19 A8);
* L1 `α`, L2 `β`: degree 1 each; L3 `t_j`: `k_log − 6` quadratic rounds; L5 `r'`: degree 63;
* O5 `r''_k`: degree 7 per claim; O6 `γ`: total degree 1 (§12);
* Ligerito's level-0 out-of-domain claim's batching coin (§13.3 step 2). -/
noncomputable def piopError (m kLog nClaims L₀ : ℕ) : ℝ :=
  (((L₀ + 1) * (m - 13) + L₀ * 127 + L₀ * (2 * (m - 6)) + L₀ * (m - kLog)
      + L₀ * (2 + 2 * (kLog - 6) + 63)
      + L₀ * (7 * nClaims + 1)
      + L₀ : ℕ) : ℝ) / qF

/-- The level-0 list bound. -/
def Schedule.listBound₀ (s : Schedule) : ℕ := (s.levels.head?.map Level.listBound).getD 0

/-- **One rep:** the probability that one rep of the table accepts although no codeword near
the committed table satisfies the statement. -/
noncomputable def repError (s : Schedule) (st : Stmt) : ℝ :=
  piopError s.m st.kLog st.nClaims s.listBound₀ + ligeritoError s.levels

/-- The link points (§16, drawn once per session, before both reps): a region whose committed
bits differ from the public bits has a nonzero multilinear difference in at most `m_pts`
variables, which vanishes at both independent points with probability at most `(m_pts/2^128)^2`;
over every region and every list element. -/
noncomputable def linkError (s : Schedule) (st : Stmt) : ℝ :=
  s.listBound₀ * st.nRegions * ((st.mPts : ℝ) / qF) ^ 2

/-- **One table:** the link points once, then the two reps on independent live coins, which
multiply (`Game.value_interleave_le`). -/
noncomputable def tableError (s : Schedule) (st : Stmt) : ℝ :=
  linkError s st + repError s st ^ 2

end FlockSoundness.Accounting
