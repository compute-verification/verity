import Proofs.Flock.Soundness.Discharge.ZkHidden.Rate
import Proofs.Flock.Soundness.Discharge.ZkLink.M1
import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
import Proofs.Flock.Soundness.Discharge.ZkLink.Link
import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
import Proofs.Flock.Soundness.Instance

/-!
# The hidden-output `--zk` tables are at C-Flock ZK's parameters, and `ε_ks` in closed form

`ZkLink/M1.lean` for the hidden-output plan of `J` tables (`ZkHidden.planZHJ`): every table it builds is at C-Flock
ZK's parameters, from what `verify --zk` checks at setup, so the knowledge term `ε_ks = ksAvgStrictZ` at that plan is at
most `ksAvgStrictZm1`, and at most one number that reads no table.

A refused hidden-output table carries the reference setup's regions and `contra`'s two (`DrawSetupZHJ.refused`), up to
1026 regions, where `TabZK.M1` reads `Stmt.InRange`'s 1024. So the tables meet `TabZK.M1W`, `M1` at
`Stmt.InRangeWide`, and the padded table error is checked there:

* `padTableError_le_m1_wide`: at 1026 regions, `padLinkError + padRepError² ≤ 3/2^206` at `m1Schedule m`,
  `25 ≤ m ≤ 35` (`m1_check_wide`, one kernel check per `m`, beside `m1_check`'s `2^-205` at 1024).
* `TabZK.bound_le_m1w`: a table's `bound` at `M1W` is at most `2^-203`, as `TabZK.bound_le_m1` at `M1`.
* `ksStrictZ_le_m1w`, `ksAvgStrictZ_le_m1w`: `ksStrictZ_le_m1` and `ksAvgStrictZ_le_m1` at `M1W`.
* `m1Ks_all`: the per-`m` kernel check for the hash-sum coefficient. At `m1Schedule m`, the sum over levels `ℓ ≥ 1`
  of `√(2^{log N_ℓ}·Q_ℓ)` is at most `25638` (each term below an integer `rootUp` whose square the kernel checks),
  and level 0's codeword is at most `2^23` long.
* `ksStrictZm1_le_closed`, `rateZ_le_m1w`, `ksAvgStrictZ_le_closed`: so at `M1W` tables
  $$\varepsilon_{ks}\le\frac{2^{-203}+(2\cdot25638+2^{14}+2)\sqrt{q_F^2/2^{513}}+q_S^2/2^{513}
  +\sqrt{k\,q_F^2/2^{513}}}{1-2^{23}/(ek)}$$
  (`ksClosedM1`, `rateM1`), whenever `2^23 < ek`.
* `DrawSetupZHJ.tabAt_M1W`, `table_M1W`, `refused_M1W`, `tableZHJ_M1W`, `planZHJ_M1W`: every table of `planZHJ` is at
  `M1W` (`Stmt.setupHidden`'s `checkInRange`, `Zk.schedule`'s `schedule_m1`, the inner code's `innerN_pow`), and
  `ksAvgStrictZ_planZHJ_le_m1` and `ksAvgStrictZ_planZHJ_le_closed` read the bounds at that plan (its rate cap is
  `ZkHidden.hr_planZHJ_23`).
* `qs_le_io`, `qs_le_mul`: `Q_s = Σ_c |Ω_c|/|Ω|`, the factor of `δ_link` (`linkBoundZC`), is at most the largest
  `Σ_{u ∈ S} |io u|` over draws `S`, `io` the committed wires each unit reads or writes in the circuit's partition, so
  at most `k_d·r` at draws of `k_d` units with `|io u| ≤ r`.

The file sits under `FailClosed/` because it reads `Stmt.InRangeWide` and `piopNumMaxWide` (`FailClosed/Numeric.lean`),
which import the `ZkHidden` headlines.
-/

namespace FlockSoundness.Accounting

/-- `padLinkUp` at 1026 regions. -/
def padLinkUpWide (s : Schedule) (t : ℕ) : ℚ :=
  (s.levels.headD default).padListBound t * 1026 * (64 / 2 ^ 128) ^ 2

/-- `padRepUp` at 1026 regions (`piopNumMaxWide`). -/
def padRepUpWide (s : Schedule) (t e : ℕ) (sbar : ℚ) : ℚ :=
  match s.levels with
  | l₀ :: l₁ :: rest =>
      (piopNumMaxWide s.m (l₀.padListBound t) : ℚ) / 2 ^ 128 + padMidUp l₀ l₁ t e sbar + ligUp (l₁ :: rest)
  | _ => 1

theorem padLinkError_le_wide (s : Schedule) (st : Stmt) (t : ℕ) (hst : st.InRangeWide) :
    padLinkError s st t ≤ (padLinkUpWide s t : ℝ) := by
  obtain ⟨_, hR, hp⟩ := hst
  unfold padLinkError padLinkUpWide qF
  push_cast
  have hR' : (st.nRegions : ℝ) ≤ 1026 := by exact_mod_cast hR
  have hp' : (st.mPts : ℝ) ≤ 64 := by exact_mod_cast hp
  gcongr

theorem padRepError_le_wide (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (t e : ℕ)
    (sbar : ℚ) (hp : PadNumOK l₀ t sbar) (hs : ∀ l ∈ l₁ :: rest, l.Ok) (st : Stmt) (hst : st.InRangeWide) :
    padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e ≤
      (padRepUpWide ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e sbar : ℝ) := by
  show piopError m st.kLog st.nClaims (l₀.padListBound t) + padMidLevelError l₀ l₁ t e +
      ligeritoError (l₁ :: rest) ≤ ((((piopNumMaxWide m (l₀.padListBound t) : ℚ) / 2 ^ 128 +
        padMidUp l₀ l₁ t e sbar + ligUp (l₁ :: rest)) : ℚ) : ℝ)
  push_cast
  have h1 := piopError_le_wide m (l₀.padListBound t) st hst
  have h2 := padMidLevelError_le l₀ l₁ t e sbar hp
  have h3 := ligeritoError_le (l₁ :: rest) hs
  push_cast at h1
  linarith

theorem padTableError_le_up_wide (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (t e : ℕ)
    (sbar : ℚ) (hp : PadNumOK l₀ t sbar) (hs : ∀ l ∈ l₁ :: rest, l.Ok) (st : Stmt)
    (hst : st.InRangeWide) (h0 : 0 ≤ padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e) :
    padTableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e ≤
      ((padLinkUpWide ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t +
        padRepUpWide ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e sbar ^ 2 : ℚ) : ℝ) := by
  unfold padTableError
  push_cast
  have h1 := padLinkError_le_wide ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t hst
  have h2 := padRepError_le_wide m k0 l₀ l₁ rest ext t e sbar hp hs st hst
  nlinarith

/-- A schedule's table with the padded level 0 is sound to `3/2^206` at 1026 regions once its rational check passes. -/
theorem padTableError_le_of_check_wide (s : Schedule) (st : Stmt) (hst : st.InRangeWide) (t e : ℕ) (sbar : ℚ)
    (h2 : 2 ≤ s.levels.length) (hp : PadNumOK (s.levels.headD default) t sbar) (hok : ∀ l ∈ s.levels, l.Ok)
    (hc : padLinkUpWide s t + padRepUpWide s t e sbar ^ 2 ≤ 3 / 2 ^ 206) :
    padTableError s st t e ≤ (3 : ℝ) / 2 ^ 206 := by
  obtain ⟨m, k0, levels, ext⟩ := s
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at h2
  · simp at h2
  · have hs : ∀ l ∈ l₁ :: rest, l.Ok := fun l hl => hok l (List.mem_cons_of_mem _ hl)
    refine (padTableError_le_up_wide m k0 l₀ l₁ rest ext t e sbar hp hs st hst
      (padRepError_nonneg m k0 l₀ l₁ rest ext t e sbar hp hs st)).trans ?_
    have h := Rat.cast_le (K := ℝ) |>.2 hc
    refine h.trans (le_of_eq ?_)
    norm_num

/-- The rational check at 1026 regions for `25 ≤ m ≤ 33`. -/
theorem m1_check_wide_le_33 : ∀ m, 25 ≤ m → m ≤ 33 →
    padLinkUpWide (m1Schedule m) (m1Pad m) + padRepUpWide (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      3 / 2 ^ 206 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check at 1026 regions for `m = 34, 35`. -/
theorem m1_check_wide_34_35 : ∀ m, 34 ≤ m → m ≤ 35 →
    padLinkUpWide (m1Schedule m) (m1Pad m) + padRepUpWide (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      3 / 2 ^ 206 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check at 1026 regions, `25 ≤ m ≤ 35`. -/
theorem m1_check_wide (m : ℕ) (h1 : 25 ≤ m) (h2 : m ≤ 35) :
    padLinkUpWide (m1Schedule m) (m1Pad m) + padRepUpWide (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) ^ 2 ≤
      3 / 2 ^ 206 := by
  rcases le_or_gt m 33 with h | h
  · exact m1_check_wide_le_33 m h1 h
  · exact m1_check_wide_34_35 m (by omega) h2

/-- **The padded table error at C-Flock ZK's parameters over 1026 regions**: at most `3/2^206` per table, for every
statement shape in `Stmt.InRangeWide`. -/
theorem padTableError_le_m1_wide : ∀ m, 25 ≤ m → m ≤ 35 → ∀ st : Stmt, st.InRangeWide →
    padTableError (m1Schedule m) st (m1Pad m) 2 ≤ (3 : ℝ) / 2 ^ 206 := by
  intro m h1 h2 st hst
  have h := m1Facts_spec (m1Facts_all m h1 h2)
  exact padTableError_le_of_check_wide (m1Schedule m) st hst (m1Pad m) 2 (m1SqrtUp m) h.1 h.2.2.1 h.2.2.2.2
    (m1_check_wide m h1 h2)

/-! ## The hash-sum coefficient -/

/-- `⌊√n⌋` by bisection on `[lo, hi)`, within `f` steps. Nothing rests on it being right: `m1Ks_all` checks each
square it is used at. -/
def sqrtBis (n : ℕ) : ℕ → ℕ → ℕ → ℕ
  | 0, lo, _ => lo
  | f + 1, lo, hi =>
    if hi ≤ lo + 1 then lo
    else if (lo + hi) / 2 * ((lo + hi) / 2) ≤ n then sqrtBis n f ((lo + hi) / 2) hi
    else sqrtBis n f lo ((lo + hi) / 2)

/-- An integer at least `√n` once `n ≤ rootUp n * rootUp n` (`m1Ks_all` checks it where it is read). -/
def rootUp (n : ℕ) : ℕ := sqrtBis n 64 0 (n + 1) + 1

/-- Level `ℓ`'s `2^{log N_ℓ}·Q_ℓ`, under the square root of the knowledge term's level-`ℓ` hash term. -/
def hashArg (s : Schedule) (ℓ : ℕ) : ℕ := 2 ^ (lvOf s ℓ).logLen * (lvOf s ℓ).queries

/-- The hash-sum coefficient, rounded up: `Σ_{1 ≤ ℓ ≤ L} rootUp (2^{log N_ℓ}·Q_ℓ)`. -/
def hashUp (s : Schedule) : ℕ := ∑ ℓ ∈ Finset.Icc 1 s.levels.length, rootUp (hashArg s ℓ)

/-- **The per-`m` check for the knowledge term**: at `m1Schedule m`, `25 ≤ m ≤ 35`, each `rootUp` is at least its
square root, the hash-sum coefficient is at most `25638` (`√` summed exactly: `25632.6`, at `m = 35`), and level 0's
codeword has length at most `2^23`. -/
theorem m1Ks_all : ∀ m, 25 ≤ m → m ≤ 35 →
    (∀ ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length,
      hashArg (m1Schedule m) ℓ ≤ rootUp (hashArg (m1Schedule m) ℓ) * rootUp (hashArg (m1Schedule m) ℓ)) ∧
    hashUp (m1Schedule m) ≤ 25638 ∧ (lvOf (m1Schedule m) 0).logLen ≤ 23 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem sqrt_cast_le_of_le_mul {n r : ℕ} (h : n ≤ r * r) : Real.sqrt n ≤ r :=
  Real.sqrt_le_iff.2 ⟨Nat.cast_nonneg _, by rw [sq]; exact_mod_cast h⟩

/-- **The hash sum at C-Flock ZK's schedules**: `Σ_{1 ≤ ℓ ≤ L} √(2^{log N_ℓ}·Q_ℓ·x) ≤ 25638·√x`. -/
theorem hashSum_le_m1 (m : ℕ) (h1 : 25 ≤ m) (h2 : m ≤ 35) (x : ℝ) :
    ∑ ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length,
        Real.sqrt (2 ^ (lvOf (m1Schedule m) ℓ).logLen * (lvOf (m1Schedule m) ℓ).queries * x) ≤
      25638 * Real.sqrt x := by
  obtain ⟨hr, hc, -⟩ := m1Ks_all m h1 h2
  calc ∑ ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length,
          Real.sqrt (2 ^ (lvOf (m1Schedule m) ℓ).logLen * (lvOf (m1Schedule m) ℓ).queries * x)
      ≤ ∑ ℓ ∈ Finset.Icc 1 (m1Schedule m).levels.length,
          (rootUp (hashArg (m1Schedule m) ℓ) : ℝ) * Real.sqrt x := by
        refine Finset.sum_le_sum fun ℓ hℓ => ?_
        have e : (2 : ℝ) ^ (lvOf (m1Schedule m) ℓ).logLen * ((lvOf (m1Schedule m) ℓ).queries : ℝ) * x =
            (hashArg (m1Schedule m) ℓ : ℝ) * x := by
          unfold hashArg
          push_cast
          ring
        rw [e, Real.sqrt_mul (Nat.cast_nonneg _)]
        exact mul_le_mul_of_nonneg_right (sqrt_cast_le_of_le_mul (hr ℓ hℓ)) (Real.sqrt_nonneg _)
    _ = (hashUp (m1Schedule m) : ℝ) * Real.sqrt x := by
        rw [← Finset.sum_mul, hashUp, Nat.cast_sum]
    _ ≤ 25638 * Real.sqrt x := by
        have hc' : (hashUp (m1Schedule m) : ℝ) ≤ 25638 := by exact_mod_cast hc
        exact mul_le_mul_of_nonneg_right hc' (Real.sqrt_nonneg _)

end FlockSoundness.Accounting

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting Audit
open scoped ENNReal NNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ}

/-- **A `--zk` table at C-Flock ZK's parameters over the plan's shapes**: `TabZK.M1` with the statement in
`Stmt.InRangeWide` (up to 1026 regions, a refused table's), not `Stmt.InRange`. -/
def TabZK.M1W (T : TabZK F mPts) : Prop :=
  ∃ m, 25 ≤ m ∧ m ≤ 35 ∧ T.sch = m1Schedule m ∧ T.t = m1Pad m ∧ T.e = 2 ∧ (T.S.shape mPts).InRangeWide ∧
    T.N ≤ 2 ^ 13 ∧ T.q = 168 ∧ 7 / 16 ≤ T.δ

theorem TabZK.M1.toW {T : TabZK F mPts} (hT : T.M1) : T.M1W := by
  obtain ⟨m, h1, h2, hsch, ht, he, ⟨hk, hR, hp⟩, hN, hq, hδ⟩ := hT
  exact ⟨m, h1, h2, hsch, ht, he, ⟨hk, by omega, hp⟩, hN, hq, hδ⟩

/-- **A `--zk` table's bound at `M1W`** is at most `2^-203` (`TabZK.bound_le_m1` at 1026 regions, from
`padTableError_le_m1_wide`'s `3/2^206`). -/
theorem TabZK.bound_le_m1w (hF : Fintype.card F = 2 ^ 128) (T : TabZK F mPts) (hT : T.M1W) :
    T.bound ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) := by
  obtain ⟨m, h1, h2, hsch, ht, he, hst, hN, hq, hδ⟩ := hT
  unfold TabZK.bound
  set r := padRepError T.sch (T.S.shape mPts) T.t T.e
  set l := padLinkError T.sch (T.S.shape mPts) T.t
  set b := εInner T.N F T.q T.δ with hbdef
  have hl : 0 ≤ l := by simp only [l, padLinkError]; positivity
  have hsq : (ENNReal.ofReal r + b) ^ 2 ≤ 2 * (ENNReal.ofReal r ^ 2 + b ^ 2) := by
    have := ENNReal.rpow_add_le_mul_rpow_add_rpow (ENNReal.ofReal r) b (p := 2) (by norm_num)
    rw [ENNReal.rpow_two, ENNReal.rpow_two, ENNReal.rpow_two] at this
    norm_num at this
    exact this
  have hr2 : ENNReal.ofReal r ^ 2 ≤ ENNReal.ofReal (r ^ 2) := by
    rcases le_or_gt 0 r with h | h
    · rw [ENNReal.ofReal_pow h]
    · rw [ENNReal.ofReal_of_nonpos h.le]
      simp
  have hpt : l + r ^ 2 ≤ (3 : ℝ) / 2 ^ 206 := by
    have e : padTableError T.sch (T.S.shape mPts) T.t T.e = l + r ^ 2 := rfl
    rw [← e, hsch, ht, he]
    exact padTableError_le_m1_wide m h1 h2 _ hst
  have hb : b ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(114 : ℝ))) := by
    rw [hbdef, hq]
    exact εInner_le_m1 hF hN hδ
  have hpos : (0 : ℝ) ≤ (3 : ℝ) / 2 ^ 206 := by positivity
  have hpos' : (0 : ℝ) ≤ (2 : ℝ) ^ (-(114 : ℝ)) := by positivity
  calc ENNReal.ofReal l + (ENNReal.ofReal r + b) ^ 2
      ≤ ENNReal.ofReal l + 2 * (ENNReal.ofReal (r ^ 2) + b ^ 2) := by
        gcongr
        exact hsq.trans (by gcongr)
    _ = ENNReal.ofReal l + 2 * ENNReal.ofReal (r ^ 2) + 2 * b ^ 2 := by ring
    _ ≤ 2 * ENNReal.ofReal l + 2 * ENNReal.ofReal (r ^ 2) + 2 * b ^ 2 := by
        gcongr
        exact le_mul_of_one_le_left zero_le one_le_two
    _ = 2 * ENNReal.ofReal (l + r ^ 2) + 2 * b ^ 2 := by
        rw [ENNReal.ofReal_add hl (sq_nonneg r)]
        ring
    _ ≤ 2 * ENNReal.ofReal ((3 : ℝ) / 2 ^ 206) + 2 * ENNReal.ofReal ((2 : ℝ) ^ (-(114 : ℝ))) ^ 2 := by
        gcongr
    _ = ENNReal.ofReal (2 * ((3 : ℝ) / 2 ^ 206) + 2 * ((2 : ℝ) ^ (-(114 : ℝ))) ^ 2) := by
        rw [ENNReal.ofReal_add (by positivity) (by positivity), ENNReal.ofReal_mul (by norm_num),
          ENNReal.ofReal_mul (by norm_num), ENNReal.ofReal_pow hpos', ENNReal.ofReal_ofNat]
    _ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [Real.rpow_neg (by norm_num), Real.rpow_neg (by norm_num),
          show (114 : ℝ) = ((114 : ℕ) : ℝ) by norm_num, show (203 : ℝ) = ((203 : ℕ) : ℝ) by norm_num,
          Real.rpow_natCast, Real.rpow_natCast]
        norm_num

end FlockSoundness.Discharge.ZkSession

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession StrictCR
open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ}

theorem ksStrictZ_le_m1w (hF : Fintype.card F = 2 ^ 128) (ts : List (TabZK F mPts)) (j : Fin ts.length)
    (hT : (ts.get j).M1W) (k : ℕ) (qF qS : ℝ) : ksStrictZ ts j k qF qS ≤ ksStrictZm1 ts j k qF qS := by
  unfold ksStrictZ ksStrictZm1
  gcongr
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) ((ts.get j).bound_le_m1w hF hT)

/-- **`ksAvgStrictZ_le_m1` at `M1W`**: with every table of the plan at `M1W`, `ksAvgStrictZ` is at most
`ksAvgStrictZm1`. -/
theorem ksAvgStrictZ_le_m1w (hF : Fintype.card F = 2 ^ 128) {n : ℕ} {Reg : Type}
    (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (hW : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1W) (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) :
    ksAvgStrictZ planZ L k qF qS R ≤ ksAvgStrictZm1 planZ L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  rcases le_or_gt (1 - rateZ (planZ (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos
      (div_nonpos_of_nonneg_of_nonpos (ksStrictZ_nonneg _ j k qF qS) h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal
      (div_le_div_of_nonneg_right (ksStrictZ_le_m1w hF _ j (hW _ R j) k qF qS) h0.le)

/-- **A `--zk` table's knowledge bound at C-Flock ZK's parameters, as one number**: `ksStrictZm1` with the hash sum
at its largest coefficient (`hashSum_le_m1`) and the inner code's length at `2^13`. -/
noncomputable def ksClosedM1 (k : ℕ) (qF qS : ℝ) : ℝ :=
  2 ^ (-(203 : ℝ)) + (2 * 25638 + 2 ^ 14 + 2) * Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) + qS ^ 2 / 2 ^ (513 : ℝ) +
    Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

/-- The level-0 rate cap at C-Flock ZK's parameters: `2^23/(ek)`, level 0's codeword at most `2^23` long. -/
noncomputable def rateM1 (k : ℕ) : ℝ := 2 ^ 23 / (Real.exp 1 * k)

theorem ksClosedM1_nonneg (k : ℕ) (qF qS : ℝ) : 0 ≤ ksClosedM1 k qF qS := by
  unfold ksClosedM1
  positivity

theorem ksStrictZm1_le_closed (ts : List (TabZK F mPts)) (j : Fin ts.length) (hT : (ts.get j).M1W) (k : ℕ)
    (qF qS : ℝ) : ksStrictZm1 ts j k qF qS ≤ ksClosedM1 k qF qS := by
  unfold ksStrictZm1 ksClosedM1
  obtain ⟨m, h1, h2, hsch, -, -, -, hN, -, -⟩ := hT
  set x := qF ^ 2 / 2 ^ (513 : ℝ) with hxdef
  have hsx := Real.sqrt_nonneg x
  have hH : ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (ts.get j).sch.levels.length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * x) ≤
        2 * 25638 * Real.sqrt x := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hsch]
    have := hashSum_le_m1 m h1 h2 x
    push_cast
    linarith
  have hNt : ∑ _r ∈ Finset.range 2, Real.sqrt ((ts.get j).N * (ts.get j).N * x) ≤ 2 ^ 14 * Real.sqrt x := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul,
      Real.sqrt_mul (mul_self_nonneg _), Real.sqrt_mul_self (Nat.cast_nonneg _)]
    have hN' : ((ts.get j).N : ℝ) ≤ 2 ^ 13 := by exact_mod_cast hN
    calc ((2 : ℕ) : ℝ) * (((ts.get j).N : ℝ) * Real.sqrt x) ≤ 2 * (2 ^ 13 * Real.sqrt x) := by
          push_cast
          gcongr
      _ = 2 ^ 14 * Real.sqrt x := by ring
  have hτ : ∑ _r ∈ Finset.range 2, Real.sqrt x = 2 * Real.sqrt x := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    push_cast
    ring
  linarith

/-- At `M1W`, the level-0 rate `N₀/(ek)` is at most `rateM1 k`. -/
theorem rateZ_le_m1w (ts : List (TabZK F mPts)) (j : Fin ts.length) (hT : (ts.get j).M1W) (k : ℕ) :
    rateZ ts j k ≤ rateM1 k := by
  obtain ⟨m, h1, h2, hsch, -⟩ := hT
  have hc := (m1Ks_all m h1 h2).2.2
  unfold rateZ rateM1
  rw [hsch]
  exact div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) hc) (by positivity)

/-- **The compiled `--zk` knowledge term at C-Flock ZK's parameters, as one number**: with every table of the plan at
`M1W` and `2^23 < ek`, `ksAvgStrictZ ≤ ksClosedM1/(1 − 2^23/(ek))`. -/
theorem ksAvgStrictZ_le_closed (hF : Fintype.card F = 2 ^ 128) {n : ℕ} {Reg : Type}
    (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (hW : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1W) (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg)
    (hk : rateM1 k < 1) :
    ksAvgStrictZ planZ L k qF qS R ≤ ENNReal.ofReal (ksClosedM1 k qF qS / (1 - rateM1 k)) := by
  have hc : ∀ ω : L.Ω, (⨆ j : Fin (planZ (L.draw ω) R).length,
      ENNReal.ofReal (ksStrictZ (planZ (L.draw ω) R) j k qF qS / (1 - rateZ (planZ (L.draw ω) R) j k))) ≤
        ENNReal.ofReal (ksClosedM1 k qF qS / (1 - rateM1 k)) := fun ω => iSup_le fun j => by
    have hS := (ksStrictZ_le_m1w hF _ j (hW _ R j) k qF qS).trans (ksStrictZm1_le_closed _ j (hW _ R j) k qF qS)
    have hr := rateZ_le_m1w _ j (hW _ R j) k
    rcases le_or_gt (1 - rateZ (planZ (L.draw ω) R) j k) 0 with h0 | h0
    · exact (le_of_eq (ENNReal.ofReal_of_nonpos
        (div_nonpos_of_nonneg_of_nonpos (ksStrictZ_nonneg _ j k qF qS) h0))).trans zero_le
    · refine ENNReal.ofReal_le_ofReal ((div_le_div_of_nonneg_right hS h0.le).trans ?_)
      exact div_le_div_of_nonneg_left (ksClosedM1_nonneg k qF qS) (by linarith) (by linarith)
  exact (avg_mono hc).trans (avg_const _).le

/-! ## `Q_s`, the expected number of commit strings the drawn units read -/

section Qs

variable {K : Type} [Field K] (A : Arith F K) {C : Audit.Circuit} {n : ℕ}
  (P : Audit.Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **`Q_s` from the partition**: `Q_s = Σ_c |Ω_c|/|Ω|`, `linkBoundZC`'s factor, is at most any bound `B` on the
committed wires the drawn units read or write, `Σ_{u ∈ S} |io u|`, at every draw `S`. A draw reads commit string `c`
only through a wire of a drawn unit's `io` that `c` holds (`ReadsZ`), so it reads at most that many strings, and
`Σ_c |Ω_c|/|Ω|` averages that count over the draws. -/
theorem qs_le_io (L : Law n) (B : ℕ) (hB : ∀ ω, ∑ u ∈ L.draw ω, (P.io u).card ≤ B) :
    ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω ≤ B := by
  classical
  have hω : ∀ ω, (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω).card ≤ B := fun ω => by
    refine le_trans (Finset.card_le_card (t := (L.draw ω).biUnion fun u => (P.io u).image vb.pos) ?_)
      (Finset.card_biUnion_le.trans ((Finset.sum_le_sum fun u _ => Finset.card_image_le).trans (hB ω)))
    intro c hc
    obtain ⟨u, hu, g, hg, rfl⟩ := (Finset.mem_filter.1 hc).2
    exact Finset.mem_biUnion.2 ⟨u, hu, Finset.mem_image.2 ⟨g, hg, rfl⟩⟩
  have hsum : ∑ c : vb.Pos, Fintype.card (ΩcZ A P planZ tab vb L c) ≤ Fintype.card L.Ω * B := by
    calc ∑ c : vb.Pos, Fintype.card (ΩcZ A P planZ tab vb L c)
        = ∑ c : vb.Pos, (Finset.univ.filter fun ω => ReadsDrawZ A P planZ tab vb c ω).card :=
          Finset.sum_congr rfl fun c _ => Fintype.card_of_subtype _ fun ω => by simp
      _ = ∑ ω : L.Ω, (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω).card := by
          simp only [Finset.card_filter]
          exact Finset.sum_comm
      _ ≤ ∑ _ω : L.Ω, B := Finset.sum_le_sum fun ω _ => hω ω
      _ = Fintype.card L.Ω * B := by simp
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  rw [← Finset.sum_div, div_le_iff₀ hΩ]
  have := (Nat.cast_le (α := ℝ)).2 hsum
  push_cast at this
  linarith [mul_comm (B : ℝ) (Fintype.card L.Ω : ℝ)]

/-- **`Q_s ≤ k_d·r`**: at draws of at most `k_d` units, each with at most `r` committed wires in its `io`. -/
theorem qs_le_mul (L : Law n) (kd r : ℕ) (hd : ∀ ω, (L.draw ω).card ≤ kd) (hr : ∀ u, (P.io u).card ≤ r) :
    ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω ≤ kd * r := by
  have h := qs_le_io A P planZ tab vb L (kd * r) fun ω =>
    (Finset.sum_le_sum fun u _ => hr u).trans (by
      rw [Finset.sum_const, smul_eq_mul]
      exact Nat.mul_le_mul_right r (hd ω))
  push_cast at h
  exact h

end Qs

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Flock FlockLevel3 Model Accounting Refine ZkSession ZkLink ZkExec Composed

variable {n mPts : ℕ}

namespace DrawSetupZHJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJ I' mPts S d)

/-- **The range setupHidden checked** (`checkInRange`): `k_log ≤ 27`, at most 1024 regions and `m_pts ≤ 64`. -/
theorem inRange : y.x.st.c.kLog ≤ 27 ∧ y.x.st.regions.size ≤ 1024 ∧ mPts ≤ 64 := by
  obtain ⟨hk, hr, hp⟩ := (checkInRange_ok _ _ _).1 (Hidden.setupHidden_spec y.x.h).inRange
  have h1 := y.x.wf.2.mPts
  have h2 := y.x.hmPts
  exact ⟨hk, hr, by omega⟩

/-- **A hidden-output `--zk` table of a draw `verify --zk` sets up is at C-Flock ZK's parameters**, at any regions of
at most 1026 (`DrawSetupZK.tabAt_M1`'s twin). -/
theorem tabAt_M1W (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) (hR : regs.length ≤ 1026) :
    (y.tabAt regs hregs).M1W := by
  obtain ⟨hk, -, hp⟩ := y.inRange
  obtain ⟨h1, h2, -⟩ := schedule_spec y.hz
  obtain ⟨hsch, hpad⟩ := schedule_m1 y.hz
  have h0 : y.sch.k0 ≤ 35 := by
    have := y.hshape.2.2.1
    have hm := y.hmsch
    simp only [schedOf] at this hm
    omega
  obtain ⟨b, hb, hb13⟩ := innerN_pow (layOf y.x.st y.sch) (layLen_le h2 hk h0)
  have h8 : 8 * ((layOf y.x.st y.sch).len + Zk.K_PAD) ≤ innerN (layOf y.x.st y.sch) := by
    have := Nat.le_nextPowerOfTwo ((layOf y.x.st y.sch).len + Zk.K_PAD)
    rw [ZkExec.innerN, Zk.INNER_LOG_INV_RATE, show (2 : ℕ) ^ 3 = 8 by norm_num]
    omega
  have : NeZero ((layOf y.x.st y.sch).len + Zk.K_PAD) := ⟨by simp [Zk.K_PAD]⟩
  refine ⟨y.x.st.m, h1, h2, hsch, hpad, rfl, ⟨hk, hR, hp⟩, ?_, rfl, relUDR_ge_m1 y.innerDom h8⟩
  show innerN (layOf y.x.st y.sch) ≤ 2 ^ 13
  rw [hb]
  exact Nat.pow_le_pow_right (by norm_num) hb13

/-- **The table's hidden-output `--zk` table is at C-Flock ZK's parameters**: setupHidden's regions, at most 1024. -/
theorem table_M1W : y.table.M1W :=
  y.tabAt_M1W _ _ (by
    have := y.inRange.2.1
    rw [FlockVBridge.Family.length_regionsOf]
    omega)

/-- **A refused hidden-output `--zk` table is at C-Flock ZK's parameters**: setupHidden's regions and `contra`'s two,
at most 1026. -/
theorem refused_M1W : y.refused.M1W :=
  y.tabAt_M1W _ _ (by
    have := y.inRange.2.1
    have hc : (Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)).length = 2 := by
      simp [Integrate.DrawSetup.contra]
    rw [List.length_append, FlockVBridge.Family.length_regionsOf, hc]
    omega)

end DrawSetupZHJ

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}

/-- **Every table `tableZHJ` builds is at C-Flock ZK's parameters**: table `j`'s own, or the reference setup's refused
one. -/
theorem tableZHJ_M1W (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableZHJ dj y₀ S j).M1W := by
  unfold tableZHJ
  split
  · exact DrawSetupZHJ.table_M1W _
  · exact y₀.refused_M1W

/-- **Every table of the hidden-output plan of `J` tables is at C-Flock ZK's parameters** (`tableZHJ_M1W`). -/
theorem planZHJ_M1W :
    ∀ S (R : Reg) (j : Fin (planZHJ (Reg := Reg) dj y₀ S R).length), ((planZHJ dj y₀ S R).get j).M1W := by
  intro S R j
  rw [planZHJ_get dj y₀ S R j]
  exact tableZHJ_M1W dj y₀ S _

/-- **`ε_ks` at the hidden-output plan is at most `ksAvgStrictZm1`** (`ksAvgStrictZ_le_m1w` at `planZHJ_M1W`). -/
theorem ksAvgStrictZ_planZHJ_le_m1 (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) :
    ksAvgStrictZ (planZHJ dj y₀) L k qF qS R ≤ ksAvgStrictZm1 (planZHJ dj y₀) L k qF qS R :=
  ksAvgStrictZ_le_m1w execArith_correct.card_F (planZHJ dj y₀) (planZHJ_M1W dj y₀) L k qF qS R

/-- **`ε_ks` at the hidden-output plan in closed form**: at most `ksClosedM1/(1 − 2^23/(ek))` once `2^23 < ek`, a
number that reads no table (`ksAvgStrictZ_le_closed` at `planZHJ_M1W`). -/
theorem ksAvgStrictZ_planZHJ_le_closed (L : Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) (hk : rateM1 k < 1) :
    ksAvgStrictZ (planZHJ dj y₀) L k qF qS R ≤ ENNReal.ofReal (ksClosedM1 k qF qS / (1 - rateM1 k)) :=
  ksAvgStrictZ_le_closed execArith_correct.card_F (planZHJ dj y₀) (planZHJ_M1W dj y₀) L k qF qS R hk

end FlockSoundness.Discharge.ZkHidden
