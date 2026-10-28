import Definitions.Accounting.Guarantees.HeadlineDc
import Proofs.Accounting.Headline.Encoding

/-!
# An honest play meets every premise of the five data-centre int8 headlines

The non-vacuity witnesses of `Sm90SheetHeadline`, `Sm90MeasuredHeadline`, `Sm90MeasuredI3Headline`,
`Sm100SheetHeadline` and `Sm100MeasuredHeadline` (`Definitions.Accounting.Guarantees.HeadlineDc`): on one-bit coins
(`Ω = Bool`) over the window's 48,000 rounds, one honest play meets every hypothesis of each statement at once and is
accepted on every path (`sm90Sheet_meets`, `sm90Measured_meets`, `sm90MeasuredI3_meets`, `sm100Sheet_meets`,
`sm100Measured_meets`).

* **The run** (`hAnat`): honest's MMA form makes every committed word, `N_w/ω` instructions of its class (IGMMA s8
  m64n256k32 on the H200, UTCIMMA m128n256 on the B200), the design's own fold (route 0, li) folds every word, there is
  no filler, and all `N_w` words are owned, right and fixed. Its counts are those parts' sum, its clock `f_max`, and its
  cycles `Σ_k n_k·Σ_r |a_{r,k}|`, at least every facet's load, so A1–A5 (on the sheets) and (I1) (`FacetCeilings`, on
  the measured rows) hold. A6 holds at `ω` words an instruction, A7 with no side unit and every word writer-headed.
  (I3) `ScalarRoute` holds at any floors (`hAnat_scalarRoute`): the run heads no vector or table word.
* **FOLD-OPT at `π*`** holds because the run folds by a route on the menu, which `φ*` prices at most (`honest_foldOpt`,
  at any certificate).
* **The game** (`honestGame`, `honestUf`): no round deficient or unfolded, every check passing, every value right and
  kept, so the openings are pinned and the three luck premises hold in their deficient cases vacuously.
* **The proofs** (`honestLayer`): one proof a round, always accepted, opening one value, its relation and rightness
  true, so the fork form of binding and proof soundness hold at `0`, the relations pin, the play is the proofs', and
  the budget `S·(√δ_fork + n_π·δ_ZK)` is `0`. E1(c) (`RightFixed`) holds with no deficiency and every word fixed.

The run's cycles are a ceiling, not honest's measured time: the witness shows that the premises hold together at an
honest play, not where honest's own kernel reads against the bound.
-/

namespace Accounting.SecurityProofs.TS.HeadlineDcMeets

open Finset Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

theorem prob_none {Ω : Type*} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  have hp : p = fun _ => False := funext fun ω => propext ⟨h ω, False.elim⟩
  subst hp
  unfold prob
  simp

/-! ## The honest game -/

/-- **The honest receipt game** over `n` rounds: no round deficient, every check passing, every value right and the
honest committed value kept. -/
noncomputable def honestGame (Ω : Type) : (n : ℕ) → ReceiptGame Ω n
  | 0 => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) (fun _ => honestGame Ω n)

/-- **The honest unfolded share**: no round skips a fold. -/
noncomputable def honestUf (Ω : Type) : (n : ℕ) → Unfolded Ω n
  | 0 => .done
  | n + 1 => .round 0 (fun _ => honestUf Ω n)

/-- The honest round at its receipt: nothing skipped, nothing deficient. -/
noncomputable def noSkip (Ω : Type) : Accounting.HMu.SkipRound Ω Unit Unit Unit where
  view _ := ()
  skipped _ := ()
  d _ := 0
  committed _ := ()
  honest _ x := x

theorem honest_accepts (Ω : Type) (n : ℕ) (x : Paths Ω n) : (honestGame Ω n).game.accepts x := by
  induction n with
  | zero => exact trivial
  | succ n ih => exact And.intro trivial (ih x.2)

theorem honest_deficiency (Ω : Type) (n : ℕ) (x : Paths Ω n) : (honestGame Ω n).game.deficiency x = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show (0 : ℝ) + (honestGame Ω n).game.deficiency x.2 = 0
    rw [ih, add_zero]

theorem honest_share (Ω : Type) (n : ℕ) (x : Paths Ω n) : (honestUf Ω n).share x = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show (0 : ℝ) + (honestUf Ω n).share x.2 = 0
    rw [ih, add_zero]

theorem honest_pinned (Ω : Type) (n : ℕ) : (honestGame Ω n).Pinned := by
  induction n with
  | zero => exact trivial
  | succ n ih => exact And.intro (fun _ _ => trivial) (fun _ => ih)

theorem noSkip_pointMass (Ω : Type) [Fintype Ω] (μ : ℝ) : (noSkip Ω).PointMass μ :=
  fun _ ho => absurd ho (lt_irrefl 0)

theorem honest_pointMass (Ω : Type) [Fintype Ω] (μ : ℝ) (n : ℕ) : (honestGame Ω n).OperandPointMass μ := by
  induction n with
  | zero => exact trivial
  | succ n ih =>
    exact And.intro (fun _ => ⟨le_rfl, zero_le_one⟩)
      (And.intro ⟨Unit, Unit, noSkip Ω, noSkip_pointMass Ω μ, fun _ _ => rfl, fun _ => rfl, fun _ _ => rfl⟩
        (fun _ => ih))

theorem honest_receiptLuck (Ω : Type) [Fintype Ω] (pr : ℝ) (n : ℕ) : (honestGame Ω n).ReceiptLuck pr := by
  induction n with
  | zero => exact trivial
  | succ n ih =>
    have h1 : prob (fun _ : Ω => True ∧ ¬True ∧ (0 : ℝ) < 0) = 0 := prob_none fun _ h => h.2.1 trivial
    have h2 : prob (fun _ : Ω => ¬True ∧ (0 : ℝ) < 0) = 0 := prob_none fun _ h => h.1 trivial
    refine And.intro ?_ (fun _ => ih)
    show prob (fun _ : Ω => True ∧ ¬True ∧ (0 : ℝ) < 0) ≤ pr * prob (fun _ : Ω => ¬True ∧ (0 : ℝ) < 0)
    simp only [h1, h2, mul_zero, le_refl]

theorem honest_wholeLuck (Ω : Type) [Fintype Ω] (p0 : ℝ) (n : ℕ) :
    (honestGame Ω n).WholeLuck p0 (honestUf Ω n) := by
  induction n with
  | zero => exact trivial
  | succ n ih =>
    exact And.intro ⟨le_rfl, zero_le_one⟩ (And.intro (fun h => absurd h (lt_irrefl 0)) (fun _ => ih))

/-! ## The honest proofs -/

/-- **An honest proof layer**: one proof a round, always accepted, opening `()`, its relation and rightness true. -/
def honestLayer (Ω : Type) (S : ℕ) : ProofLayer Ω S 1 Unit where
  accepted _ _ _ := True
  opened _ _ _ := ()
  rel _ _ _ _ := True
  right _ _ := True

theorem honestLayer_fork (Ω : Type) [Fintype Ω] (S : ℕ) : BindingForkAt (honestLayer Ω S) 0 := by
  intro w
  refine le_of_eq (prob_none fun p h => ?_)
  obtain ⟨_, _, i, hi⟩ := h
  exact hi rfl

theorem honestLayer_sound (Ω : Type) [Fintype Ω] (S : ℕ) : ProofSoundAt (honestLayer Ω S) 0 := by
  intro w i
  refine le_of_eq (prob_none fun x h => ?_)
  exact h.2 trivial

theorem honestLayer_pins (Ω : Type) (S : ℕ) : ProofsPin (honestLayer Ω S) := by
  intro _ _ _
  trivial

theorem budget_zero (n : ℕ) : (n : ℝ) * (Real.sqrt 0 + ((1 : ℕ) : ℝ) * 0) ≤ (1 / 2) ^ 41 := by
  simp only [Real.sqrt_zero, mul_zero, add_zero]
  positivity

/-! ## The honest run -/

section Run

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

open Classical in
/-- The honest run's word production: writer kind `t0`'s class, `N_w/ω_{t0}` instructions of it. -/
noncomputable def hNword (Z : Model K R T L U F X) (t0 : T) : Head T L U → K → ℝ
  | .writer t, k => if t = t0 ∧ k = Z.cls t0 then Z.Nw / Z.omega t0 else 0
  | .vector, _ => 0
  | .table _, _ => 0
  | .side _, _ => 0

/-- The honest run's fold: route counts `c` a word, for each of the `N_w` words. -/
noncomputable def hNfold (Z : Model K R T L U F X) (c : K → ℝ) : K → ℝ := fun k => Z.Nw * c k

/-- The honest run's counts: its word production, its fold and no filler. -/
noncomputable def hN (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ) : K → ℝ :=
  fun k => ∑ h, hNword Z t0 h k + hNfold Z c k + 0

open Classical in
/-- The honest run's words: all `N_w` headed by writer kind `t0`. -/
noncomputable def hWords (Z : Model K R T L U F X) (t0 : T) : Head T L U → ℝ
  | .writer t => if t = t0 then Z.Nw else 0
  | .vector => 0
  | .table _ => 0
  | .side _ => 0

/-- A run of counts `n` at `M`'s `f_max`, its cycles `Σ_k n_k·Σ_r |a_{r,k}|`, at least every facet's load. -/
noncomputable def runOf {R' : Type} [Fintype R'] (n : K → ℝ) (M : FacetCosts K R') : Run K where
  n := n
  cycles := ∑ k, n k * ∑ r, |M.a r k|
  clock := M.fmax

/-- **The honest run's labelling**: writer kind `t0` heads every word, route counts `c` fold every word, no filler,
every word owned, right and fixed; the run at `M`'s ceilings. -/
noncomputable def hAnat {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') : Anatomy K T L U where
  run := runOf (hN Z t0 c) M
  nword := hNword Z t0
  nfold := hNfold Z c
  nfill _ := 0
  words := hWords Z t0
  fixed := Z.Nw

/-- **The honest play**: the honest game, accepted on every path, the run `A` on each. -/
noncomputable def honestPlay (Ω : Type) (n : ℕ) (A : Anatomy K T L U) : Play Ω n K T L U where
  game := (honestGame Ω n).game
  accepts _ := True
  run _ := A

theorem load_le {R' : Type} [Fintype R'] (M : FacetCosts K R') (n : K → ℝ) (hn : ∀ k, 0 ≤ n k) (r : R') :
    M.load n r ≤ (runOf n M).cycles := by
  show ∑ k, M.a r k * n k ≤ ∑ k, n k * ∑ r', |M.a r' k|
  refine Finset.sum_le_sum fun k _ => ?_
  rw [mul_comm (n k)]
  exact mul_le_mul_of_nonneg_right ((le_abs_self _).trans
    (Finset.single_le_sum (f := fun r' => |M.a r' k|) (fun r' _ => abs_nonneg _) (Finset.mem_univ r))) (hn k)

theorem load_smul {R' : Type} (M : FacetCosts K R') (s : ℝ) (c : K → ℝ) (r : R') :
    M.load (fun k => s * c k) r = s * M.load c r := by
  simp only [FacetCosts.load, Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

theorem price_smul {R' : Type} [Fintype R'] (M : FacetCosts K R') (lam : R' → ℝ) (s : ℝ) (c : K → ℝ) :
    M.price lam (fun k => s * c k) = s * demandPrice lam (M.load c) := by
  unfold FacetCosts.price demandPrice
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [load_smul]
  ring

omit [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X] in
theorem hNword_nonneg (Z : Model K R T L U F X) (t0 : T) (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0)
    (h : Head T L U) (k : K) : 0 ≤ hNword Z t0 h k := by
  cases h with
  | writer t =>
    simp only [hNword]
    split_ifs
    · exact div_nonneg hNw hω.le
    · exact le_rfl
  | vector => exact le_rfl
  | table _ => exact le_rfl
  | side _ => exact le_rfl

omit [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X] in
theorem hWords_nonneg (Z : Model K R T L U F X) (t0 : T) (hNw : 0 ≤ Z.Nw) (h : Head T L U) :
    0 ≤ hWords Z t0 h := by
  cases h with
  | writer t =>
    simp only [hWords]
    split_ifs
    · exact hNw
    · exact le_rfl
  | vector => exact le_rfl
  | table _ => exact le_rfl
  | side _ => exact le_rfl

theorem hN_nonneg (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ) (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0)
    (hc : ∀ k, 0 ≤ c k) (k : K) : 0 ≤ hN Z t0 c k := by
  have h1 : 0 ≤ ∑ h, hNword Z t0 h k := Finset.sum_nonneg fun h _ => hNword_nonneg Z t0 hNw hω h k
  have h2 : 0 ≤ hNfold Z c k := mul_nonneg hNw (hc k)
  show 0 ≤ ∑ h, hNword Z t0 h k + hNfold Z c k + 0
  linarith

theorem hAnat_split {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ) (M : FacetCosts K R')
    (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0) (hc : ∀ k, 0 ≤ c k) : (hAnat Z t0 c M).Split :=
  And.intro (fun h k => hNword_nonneg Z t0 hNw hω h k) (And.intro (fun k => mul_nonneg hNw (hc k))
    (And.intro (fun _ => le_rfl) (And.intro (fun h => hWords_nonneg Z t0 hNw h) (fun _ => le_rfl))))

omit [Fintype R] [Fintype F] [Fintype X] in
theorem hAnat_sheetWords {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') (hω : 0 < Z.omega t0) : SheetWords Z (hAnat Z t0 c M) := by
  refine And.intro (fun t => ?_) (And.intro ?_ (fun l => ?_))
  · show hWords Z t0 (.writer t) ≤ Z.omega t * hNword Z t0 (.writer t) (Z.cls t)
    by_cases ht : t = t0
    · have e1 : hWords Z t0 (.writer t) = Z.Nw := by simp [hWords, ht]
      have e2 : hNword Z t0 (.writer t) (Z.cls t) = Z.Nw / Z.omega t0 := by simp [hNword, ht]
      rw [e1, e2, ht]
      exact le_of_eq (mul_div_cancel₀ Z.Nw hω.ne').symm
    · have e1 : hWords Z t0 (.writer t) = 0 := by simp [hWords, ht]
      have e2 : hNword Z t0 (.writer t) (Z.cls t) = 0 := by simp [hNword, ht]
      rw [e1, e2, mul_zero]
  · show hWords Z t0 .vector ≤ 2 * (32 * ∑ k ∈ Z.vec, hNword Z t0 .vector k)
    simp [hWords, hNword]
  · show hWords Z t0 (.table l) ≤ 2 * (32 * ∑ k ∈ Z.vec, hNword Z t0 (.table l) k)
    simp [hWords, hNword]

omit [Fintype R] [Fintype F] [Fintype X] in
/-- (I3) at any floors: the honest run heads no vector or table word. -/
theorem hAnat_scalarRoute {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') : ScalarRoute Z (hAnat Z t0 c M) := by
  refine And.intro ?_ (fun l => And.intro ?_ ?_)
  · show Z.q * hWords Z t0 .vector ≤ 32 * ∑ k ∈ Z.vec, hNword Z t0 .vector k
    simp [hWords, hNword]
  · show Z.qT * hWords Z t0 (.table l) ≤ 32 * ∑ k ∈ Z.vec, hNword Z t0 (.table l) k
    simp [hWords, hNword]
  · show Z.bT * hWords Z t0 (.table l) ≤ hNword Z t0 (.table l) (Z.bytes l)
    simp [hWords, hNword]

theorem hAnat_noSum {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ) (M : FacetCosts K R')
    (hNw : 0 ≤ Z.Nw) : NoSum (hAnat Z t0 c M) := by
  show Z.Nw ≤ ∑ h, hWords Z t0 h
  have e : hWords Z t0 (.writer t0) = Z.Nw := by simp [hWords]
  calc Z.Nw = hWords Z t0 (.writer t0) := e.symm
    _ ≤ ∑ h, hWords Z t0 h :=
      Finset.single_le_sum (f := hWords Z t0) (fun h _ => hWords_nonneg Z t0 hNw h) (Finset.mem_univ _)

omit [Fintype R] [Fintype F] [Fintype X] in
theorem hAnat_sideUnits {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') : SideUnits Z (hAnat Z t0 c M) := by
  intro u k
  show hWords Z t0 (.side u) * Z.side u k ≤ hNword Z t0 (.side u) k
  simp [hWords, hNword]

theorem hAnat_semantics {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') (hNw : 0 ≤ Z.Nw) : SheetSemantics Z (hAnat Z t0 c M) :=
  And.intro (hAnat_noSum Z t0 c M hNw) (hAnat_sideUnits Z t0 c M)

omit [Fintype X] in
/-- **FOLD-OPT at any certificate**: the run folds every word by route `φ0`, which `φ*` prices at most. -/
theorem hAnat_foldOpt {R' : Type} [Fintype R'] (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ) (M : FacetCosts K R')
    (hNw : 0 ≤ Z.Nw) (hmu : Z.muF = 0) (φ0 : F) (hφ : Z.fold φ0 = Z.G.load c) (lam : R → ℝ) :
    FoldOptFoldedAt Z (hAnat Z t0 c M) Z.Nw lam := by
  unfold FoldOptFoldedAt
  rw [hmu, sub_zero, one_mul]
  show Z.Nw * Z.phiStar lam ≤ Z.G.price lam (fun k => Z.Nw * c k)
  rw [price_smul, ← hφ]
  exact mul_le_mul_of_nonneg_left (Finite.ciInf_le (fun φ => demandPrice lam (Z.fold φ)) φ0) hNw

theorem honest_foldOpt {R' : Type} [Fintype R'] (Ω : Type) (n : ℕ) (Z : Model K R T L U F X) (t0 : T) (c : K → ℝ)
    (M : FacetCosts K R') (hNw : 0 ≤ Z.Nw) (hmu : Z.muF = 0) (φ0 : F) (hφ : Z.fold φ0 = Z.G.load c) (lam : R → ℝ)
    (x : Paths Ω n) : FoldOptFoldedAt Z (hAnat Z t0 c M) ((1 - (honestUf Ω n).share x / n) * Z.Nw) lam := by
  rw [honest_share, zero_div, sub_zero, one_mul]
  exact hAnat_foldOpt Z t0 c M hNw hmu φ0 hφ lam

omit [Fintype K] [Fintype T] [Fintype L] [Fintype U] in
theorem honestLayer_of (Ω : Type) [Fintype Ω] (n : ℕ) (P : Play Ω n K T L U) (hg : ∀ x, P.game.accepts x) :
    (honestLayer Ω n).Of P :=
  And.intro (fun _ _ _ _ => trivial) (fun x _ => hg x)

omit [Fintype K] [Fintype T] [Fintype L] [Fintype U] in
theorem honest_rightFixed (Ω : Type) [Fintype Ω] (n : ℕ) (A : Anatomy K T L U) {nπ : ℕ} {V : Type}
    (pf : ProofLayer Ω n nπ V) (Nw : ℝ) (hA : Nw ≤ A.fixed) : RightFixed (honestPlay Ω n A) pf Nw := by
  intro x _ _
  show (1 - (honestGame Ω n).game.deficiency x / n) * Nw ≤ A.fixed
  rw [honest_deficiency, zero_div, sub_zero, one_mul]
  exact hA

/-! ## The sheets' ceilings and the measured rows' -/

section Sheet

variable {P : Type} [Fintype P] [DecidableEq P]

theorem sheetClock_of (S : Sheet K P) (hf : 0 < S.fmax) (n : K → ℝ) : SheetClock S (runOf n S.machine) :=
  And.intro hf le_rfl

theorem sheetDispatch_of (S : Sheet K P) (n : K → ℝ) (hn : ∀ k, 0 ≤ n k) : SheetDispatch (runOf n S.machine) := by
  have h := load_le S.machine n hn SheetRow.issue
  have e : S.machine.load n SheetRow.issue = ∑ k, n k := by
    unfold FacetCosts.load
    exact Finset.sum_congr rfl fun k _ => one_mul (n k)
  rw [e] at h
  exact h

theorem sheetTensor_of (S : Sheet K P) (n : K → ℝ) (hn : ∀ k, 0 ≤ n k) : SheetTensor S (runOf n S.machine) :=
  load_le S.machine n hn SheetRow.tensor

theorem sheetPipe_of (S : Sheet K P) (hw : ∀ p, 0 < S.width p) (n : K → ℝ) (hn : ∀ k, 0 ≤ n k) :
    SheetPipe S (runOf n S.machine) := by
  intro p
  have h := load_le S.machine n hn (SheetRow.pipe p)
  have hwp : S.width p ≠ 0 := (hw p).ne'
  have e : ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * n k =
      S.width p / 4 * S.machine.load n (SheetRow.pipe p) := by
    unfold FacetCosts.load
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    show (if S.pipe k = some p then (32 : ℝ) else 0) * n k =
      S.width p / 4 * ((if S.pipe k = some p then 128 / S.width p else 0) * n k)
    split_ifs
    · rw [← mul_assoc, div_mul_div_comm, mul_comm (S.width p) 128, mul_div_mul_right _ _ hwp]
      norm_num
    · simp
  show ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * n k ≤ S.width p / 4 * (runOf n S.machine).cycles
  rw [e]
  exact mul_le_mul_of_nonneg_left h (div_nonneg (hw p).le (by norm_num))

theorem sheetSmem_of (S : Sheet K P) (n : K → ℝ) (hn : ∀ k, 0 ≤ n k) : SheetSmem S (runOf n S.machine) := by
  have h := load_le S.machine n hn SheetRow.smem
  have e : ∑ k, S.smem k * n k = 32 * S.machine.load n SheetRow.smem := by
    unfold FacetCosts.load
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    show S.smem k * n k = 32 * (S.smem k / 32 * n k)
    ring
  show ∑ k, S.smem k * n k ≤ 32 * (runOf n S.machine).cycles
  rw [e]
  linarith

end Sheet

theorem facetCeilings_of {R' : Type} [Fintype R'] (M : FacetCosts K R') (hf : 0 < M.fmax) (n : K → ℝ)
    (hn : ∀ k, 0 ≤ n k) : FacetCeilings M (runOf n M) := by
  have hcyc : 0 ≤ (runOf n M).cycles :=
    Finset.sum_nonneg fun k _ => mul_nonneg (hn k) (Finset.sum_nonneg fun r _ => abs_nonneg _)
  refine And.intro hn (And.intro hcyc (And.intro hf (And.intro le_rfl fun r => ?_)))
  refine And.intro (fun _ => load_le M n hn r) (fun _ => ?_)
  have e : M.fmax * (runOf n M).time = (runOf n M).cycles := mul_div_cancel₀ _ hf.ne'
  rw [e]
  exact load_le M n hn r

/-! ## The witnesses, generic -/

/-- **A sheet headline's premises at an honest play**, on any data sheet `S` and TS model `Z` over its rows. -/
theorem sheet_meets {P : Type} [Fintype P] [DecidableEq P] (S : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (n : ℕ) (t0 : T) (φ0 : F) (c : K → ℝ) (lam : SheetRow P → ℝ) (hf : 0 < S.fmax) (hw : ∀ p, 0 < S.width p)
    (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0) (hc : ∀ k, 0 ≤ c k) (hmu : Z.muF = 0)
    (hφ : Z.fold φ0 = Z.G.load c) :
    ∃ (Pl : Play Bool n K T L U) (G : ReceiptGame Bool n) (Uf : Unfolded Bool n) (pf : ProofLayer Bool n 1 Unit)
      (δfork δzk : ℝ),
      (n : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, SheetClock S (Pl.run x).run) ∧ (∀ x, SheetDispatch (Pl.run x).run) ∧
      (∀ x, SheetTensor S (Pl.run x).run) ∧ (∀ x, SheetPipe S (Pl.run x).run) ∧
      (∀ x, SheetSmem S (Pl.run x).run) ∧ (∀ x, SheetWords Z (Pl.run x)) ∧
      (∀ x, SheetSemantics Z (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / n) * Z.Nw) lam) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Z.Nw ∧
      ∀ x, Pl.accepts x := by
  have hn := hN_nonneg Z t0 c hNw hω hc
  exact ⟨honestPlay Bool n (hAnat Z t0 c S.machine), honestGame Bool n, honestUf Bool n, honestLayer Bool n, 0, 0,
    budget_zero n, fun _ => hAnat_split Z t0 c S.machine hNw hω hc, fun _ => sheetClock_of S hf (hN Z t0 c),
    fun _ => sheetDispatch_of S (hN Z t0 c) hn, fun _ => sheetTensor_of S (hN Z t0 c) hn,
    fun _ => sheetPipe_of S hw (hN Z t0 c) hn, fun _ => sheetSmem_of S (hN Z t0 c) hn,
    fun _ => hAnat_sheetWords Z t0 c S.machine hω, fun _ => hAnat_semantics Z t0 c S.machine hNw,
    fun x => honest_foldOpt Bool n Z t0 c S.machine hNw hmu φ0 hφ lam x, rfl, honest_pinned Bool n,
    honest_pointMass Bool (3 / 4) n, honest_receiptLuck Bool (1 / 2) n, honest_wholeLuck Bool (1 / 2) n,
    honestLayer_of Bool n _ (honest_accepts Bool n), honestLayer_fork Bool n, honestLayer_sound Bool n,
    honestLayer_pins Bool n, honest_rightFixed Bool n _ _ Z.Nw le_rfl, fun _ => trivial⟩

/-- **A measured headline's premises at an honest play**, on any measured machine `M` and TS model `Z` on it. -/
theorem measured_meets (M : FacetCosts K R) (Z : Model K R T L U F X) (n : ℕ) (t0 : T) (φ0 : F) (c : K → ℝ)
    (lam : R → ℝ) (hf : 0 < M.fmax) (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0) (hc : ∀ k, 0 ≤ c k) (hmu : Z.muF = 0)
    (hφ : Z.fold φ0 = Z.G.load c) :
    ∃ (Pl : Play Bool n K T L U) (G : ReceiptGame Bool n) (Uf : Unfolded Bool n) (pf : ProofLayer Bool n 1 Unit)
      (δfork δzk : ℝ),
      (n : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, FacetCeilings M (Pl.run x).run) ∧
      (∀ x, SheetWords Z (Pl.run x)) ∧ (∀ x, NoSum (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / n) * Z.Nw) lam) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Z.Nw ∧
      ∀ x, Pl.accepts x := by
  have hn := hN_nonneg Z t0 c hNw hω hc
  exact ⟨honestPlay Bool n (hAnat Z t0 c M), honestGame Bool n, honestUf Bool n, honestLayer Bool n, 0, 0,
    budget_zero n, fun _ => hAnat_split Z t0 c M hNw hω hc, fun _ => facetCeilings_of M hf (hN Z t0 c) hn,
    fun _ => hAnat_sheetWords Z t0 c M hω, fun _ => hAnat_noSum Z t0 c M hNw,
    fun x => honest_foldOpt Bool n Z t0 c M hNw hmu φ0 hφ lam x, rfl, honest_pinned Bool n,
    honest_pointMass Bool (3 / 4) n, honest_receiptLuck Bool (1 / 2) n, honest_wholeLuck Bool (1 / 2) n,
    honestLayer_of Bool n _ (honest_accepts Bool n), honestLayer_fork Bool n, honestLayer_sound Bool n,
    honestLayer_pins Bool n, honest_rightFixed Bool n _ _ Z.Nw le_rfl, fun _ => trivial⟩

/-- **A measured headline's premises at an honest play under (I3)**: `measured_meets` with (I2)/(I4)
`WordsWritten` and (I3) `ScalarRoute` in A6's place, at any floors. -/
theorem measured_meets_i3 (M : FacetCosts K R) (Z : Model K R T L U F X) (n : ℕ) (t0 : T) (φ0 : F) (c : K → ℝ)
    (lam : R → ℝ) (hf : 0 < M.fmax) (hNw : 0 ≤ Z.Nw) (hω : 0 < Z.omega t0) (hc : ∀ k, 0 ≤ c k) (hmu : Z.muF = 0)
    (hφ : Z.fold φ0 = Z.G.load c) :
    ∃ (Pl : Play Bool n K T L U) (G : ReceiptGame Bool n) (Uf : Unfolded Bool n) (pf : ProofLayer Bool n 1 Unit)
      (δfork δzk : ℝ),
      (n : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, FacetCeilings M (Pl.run x).run) ∧
      (∀ x, WordsWritten Z (Pl.run x)) ∧ (∀ x, ScalarRoute Z (Pl.run x)) ∧ (∀ x, NoSum (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / n) * Z.Nw) lam) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Z.Nw ∧
      ∀ x, Pl.accepts x := by
  have hn := hN_nonneg Z t0 c hNw hω hc
  exact ⟨honestPlay Bool n (hAnat Z t0 c M), honestGame Bool n, honestUf Bool n, honestLayer Bool n, 0, 0,
    budget_zero n, fun _ => hAnat_split Z t0 c M hNw hω hc, fun _ => facetCeilings_of M hf (hN Z t0 c) hn,
    fun _ => (hAnat_sheetWords Z t0 c M hω).1, fun _ => hAnat_scalarRoute Z t0 c M,
    fun _ => hAnat_noSum Z t0 c M hNw,
    fun x => honest_foldOpt Bool n Z t0 c M hNw hmu φ0 hφ lam x, rfl, honest_pinned Bool n,
    honest_pointMass Bool (3 / 4) n, honest_receiptLuck Bool (1 / 2) n, honest_wholeLuck Bool (1 / 2) n,
    honestLayer_of Bool n _ (honest_accepts Bool n), honestLayer_fork Bool n, honestLayer_sound Bool n,
    honestLayer_pins Bool n, honest_rightFixed Bool n _ _ Z.Nw le_rfl, fun _ => trivial⟩

end Run

/-! ## The five witnesses -/

theorem counts_nonneg {n : ℕ} (v : Sparse) (k : Fin n) : 0 ≤ Sparse.counts v k :=
  Rat.cast_nonneg.mpr (Accounting.SecurityProofs.TS.Headline.coeff_nonneg v k.val)

/-- **`Sm90SheetHeadline`'s hypotheses hold together at an honest play** on one-bit coins: honest's IGMMA s8
m64n256k32 (writer kind 19) heads every word and the design's li (route 0) folds every one. -/
theorem sm90Sheet_meets :
    ∃ (Pl : Play Bool Sm90.checks (Fin 50) (Fin 34) Unit Empty) (G : ReceiptGame Bool Sm90.checks)
      (Uf : Unfolded Bool Sm90.checks) (pf : ProofLayer Bool Sm90.checks 1 Unit) (δfork δzk : ℝ),
      (Sm90.checks : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, SheetClock Sm90.sheet (Pl.run x).run) ∧ (∀ x, SheetDispatch (Pl.run x).run) ∧
      (∀ x, SheetTensor Sm90.sheet (Pl.run x).run) ∧ (∀ x, SheetPipe Sm90.sheet (Pl.run x).run) ∧
      (∀ x, SheetSmem Sm90.sheet (Pl.run x).run) ∧ (∀ x, SheetWords Sm90.model (Pl.run x)) ∧
      (∀ x, SheetSemantics Sm90.model (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw) Sm90.lamStar) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Sm90.model.Nw ∧
      ∀ x, Pl.accepts x :=
  sheet_meets Sm90.sheet Sm90.model Sm90.checks 19 0 (Sparse.counts (Sm90.d2s.folds.getD (0 : Fin 56).val []))
    Sm90.lamStar (Nat.cast_pos.mpr (by decide))
    (fun p => by cases p <;> norm_num [Sm90.sheet, SheetDataQ.sheet, Sm90.ds90, Sm90.Pipe.width])
    (Nat.cast_nonneg _) (Nat.cast_pos.mpr (by decide)) (counts_nonneg _) rfl rfl

/-- **`Sm90MeasuredHeadline`'s hypotheses hold together at an honest play** on one-bit coins: honest's IGMMA s8
m64n256k32 (writer kind 0) heads every word and the design's li (route 0) folds every one, within DC-90-ri's rows. -/
theorem sm90Measured_meets :
    ∃ (Pl : Play Bool Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Bool Sm90.checks)
      (Uf : Unfolded Bool Sm90.checks) (pf : ProofLayer Bool Sm90.checks 1 Unit) (δfork δzk : ℝ),
      (Sm90.checks : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) ∧
      (∀ x, SheetWords Sm90Measured.model (Pl.run x)) ∧ (∀ x, NoSum (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Sm90Measured.model (Pl.run x)
        ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw) Sm90Measured.lamStar) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧
      RightFixed Pl pf Sm90Measured.model.Nw ∧ ∀ x, Pl.accepts x :=
  measured_meets Sm90Measured.machine Sm90Measured.model Sm90.checks 0 0
    (Sparse.counts (Sm90Measured.d2s.folds.getD (0 : Fin 56).val [])) Sm90Measured.lamStar
    (Nat.cast_pos.mpr (by decide)) (Nat.cast_nonneg _) (Nat.cast_pos.mpr (by decide)) (counts_nonneg _) rfl rfl

/-- **`Sm90MeasuredI3Headline`'s hypotheses hold together at an honest play** on one-bit coins: as
`sm90Measured_meets`, at (I3)'s floors `q = q_T = 3/5`, which the run meets since it heads no vector or table word. -/
theorem sm90MeasuredI3_meets :
    ∃ (Pl : Play Bool Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Bool Sm90.checks)
      (Uf : Unfolded Bool Sm90.checks) (pf : ProofLayer Bool Sm90.checks 1 Unit) (δfork δzk : ℝ),
      (Sm90.checks : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) ∧
      (∀ x, WordsWritten Sm90MeasuredI3.model (Pl.run x)) ∧ (∀ x, ScalarRoute Sm90MeasuredI3.model (Pl.run x)) ∧
      (∀ x, NoSum (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Sm90MeasuredI3.model (Pl.run x)
        ((1 - Uf.share x / Sm90.checks) * Sm90MeasuredI3.model.Nw) Sm90MeasuredI3.lamStar) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧
      RightFixed Pl pf Sm90MeasuredI3.model.Nw ∧ ∀ x, Pl.accepts x :=
  measured_meets_i3 Sm90Measured.machine Sm90MeasuredI3.model Sm90.checks 0 0
    (Sparse.counts (Sm90MeasuredI3.d2s.folds.getD (0 : Fin 56).val [])) Sm90MeasuredI3.lamStar
    (Nat.cast_pos.mpr (by decide)) (Nat.cast_nonneg _) (Nat.cast_pos.mpr (by decide)) (counts_nonneg _) rfl rfl

/-- **`Sm100SheetHeadline`'s hypotheses hold together at an honest play** on one-bit coins: honest's UTCIMMA
m128n256 (writer kind 10) heads every word and the design's li (route 0) folds every one. -/
theorem sm100Sheet_meets :
    ∃ (Pl : Play Bool Sm100.checks (Fin 52) (Fin 27) Unit Empty) (G : ReceiptGame Bool Sm100.checks)
      (Uf : Unfolded Bool Sm100.checks) (pf : ProofLayer Bool Sm100.checks 1 Unit) (δfork δzk : ℝ),
      (Sm100.checks : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, SheetClock Sm100.sheet (Pl.run x).run) ∧ (∀ x, SheetDispatch (Pl.run x).run) ∧
      (∀ x, SheetTensor Sm100.sheet (Pl.run x).run) ∧ (∀ x, SheetPipe Sm100.sheet (Pl.run x).run) ∧
      (∀ x, SheetSmem Sm100.sheet (Pl.run x).run) ∧ (∀ x, SheetWords Sm100.model (Pl.run x)) ∧
      (∀ x, SheetSemantics Sm100.model (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw)
        Sm100.lamStar) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Sm100.model.Nw ∧
      ∀ x, Pl.accepts x :=
  sheet_meets Sm100.sheet Sm100.model Sm100.checks 10 0 (Sparse.counts (Sm100.d2s.folds.getD (0 : Fin 14).val []))
    Sm100.lamStar (Nat.cast_pos.mpr (by decide))
    (fun p => by cases p <;> norm_num [Sm100.sheet, SheetDataQ.sheet, Sm100.ds100, Sm100.Pipe.width])
    (Nat.cast_nonneg _) (Nat.cast_pos.mpr (by decide)) (counts_nonneg _) rfl rfl

/-- **`Sm100MeasuredHeadline`'s hypotheses hold together at an honest play** on one-bit coins: honest's UTCIMMA
m128n256 (writer kind 4) heads every word and the design's li (route 0) folds every one, within DC-100-ri's rows. -/
theorem sm100Measured_meets :
    ∃ (Pl : Play Bool Sm100.checks (Fin 52) (Fin 13) Unit Empty) (G : ReceiptGame Bool Sm100.checks)
      (Uf : Unfolded Bool Sm100.checks) (pf : ProofLayer Bool Sm100.checks 1 Unit) (δfork δzk : ℝ),
      (Sm100.checks : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, FacetCeilings Sm100Measured.machine (Pl.run x).run) ∧
      (∀ x, SheetWords Sm100Measured.model (Pl.run x)) ∧ (∀ x, NoSum (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Sm100Measured.model (Pl.run x)
        ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw) Sm100Measured.lamStar) ∧
      Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
      pf.Of Pl ∧ BindingForkAt pf δfork ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧
      RightFixed Pl pf Sm100Measured.model.Nw ∧ ∀ x, Pl.accepts x :=
  measured_meets Sm100Measured.machine Sm100Measured.model Sm100.checks 4 0
    (Sparse.counts (Sm100Measured.d2s.folds.getD (0 : Fin 14).val [])) Sm100Measured.lamStar
    (Nat.cast_pos.mpr (by decide)) (Nat.cast_nonneg _) (Nat.cast_pos.mpr (by decide)) (counts_nonneg _) rfl rfl

end Accounting.SecurityProofs.TS.HeadlineDcMeets
