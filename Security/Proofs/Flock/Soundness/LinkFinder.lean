import Proofs.Flock.Soundness.ExtractWait
import Proofs.Flock.Soundness.Assumptions

/-!
# The link finder: a trial, then further trials until one reads the same string

The link theorem's collision finder (DESIGN.md §3), for any trial. A trial (`T`, played by `sT`)
reads a set of values behind one commit string (`rd`). The finder (`finderG`) runs a trial and then
up to `M` more. It succeeds (`Succ`) if

* its first trial reads two values (two drawn units opened the string differently), or
* its first trial reads one value, and the first further trial that reads anything reads another.

With `p v` the probability that a trial reads exactly `v`, `s` that it reads two or more values and
`r` that it reads anything, the finder succeeds with probability `s + geom r M · Σ_v p v (r − p v)`
(`expect_succ`). For the plurality `Xv` of `p` that is at least `s + Σ_{v ≠ Xv} p v − 1/(eM)`
(`succ_ge`): disagreeing with the plurality is at most as likely as two readings disagreeing.

Its cost stops at the first further trial that reads anything, so it is at most twice a trial's in
expectation, whatever `r` (`expect_finderCost_le`, a Wald identity). Under expected-time collision
resistance for the finder (`Assumptions.SHA512CRExpected`), and when every success outputs a
collision, `s + Σ_{v ≠ Xv} p v ≤ 2·E[trial cost]/2^256 + 1/(eM)` (`link_mass_le`).
-/

namespace FlockSoundness

namespace LinkFinder

open Game Extract

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (T : Game τ) (sT : Strategy T)
  (rd : τ → Finset Val)

/-- **The finder**: a trial, then `M` more. -/
def finderG (M : ℕ) : Game (τ × List τ) := T.bind fun e => (runsG T M).map fun aux => (e, aux)

/-- `sT` in every trial. -/
def finderStrat (M : ℕ) : Strategy (finderG T M) :=
  Strategy.bind T _ sT fun _ => Strategy.map _ _ (runsStrat T sT M)

theorem expect_finderG (M : ℕ) (F : τ × List τ → ℝ) :
    expect F (finderG T M) (finderStrat T sT M) =
      expect (fun e => expectN T sT M fun aux => F (e, aux)) T sT := by
  unfold finderG finderStrat
  rw [expect_bind]
  refine congrArg (fun f => expect f T sT) (funext fun e => ?_)
  rw [expect_map, expect_runsG]

open Classical in
/-- The first further trial that reads anything. -/
noncomputable def nextRead (aux : List τ) : Option τ := aux.find? fun e => decide (rd e).Nonempty

open Classical in
/-- **The finder succeeds**: its first trial reads two values, or it reads one and the first
further trial that reads anything reads another. -/
def Succ (x : τ × List τ) : Prop :=
  2 ≤ (rd x.1).card ∨ ∃ v, rd x.1 = {v} ∧ ∃ e', nextRead rd x.2 = some e' ∧ rd e' ≠ {v}

open Classical in
/-- The further trials' cost, up to and including the first that reads anything. -/
noncomputable def loopCost (c : τ → ℝ) : List τ → ℝ
  | [] => 0
  | e :: es => c e + if (rd e).Nonempty then 0 else loopCost c es

open Classical in
/-- **The finder's cost**: its first trial's, and when that read one value, the further trials' up
to the first that reads anything. -/
noncomputable def finderCost (c : τ → ℝ) (x : τ × List τ) : ℝ :=
  c x.1 + if (rd x.1).card = 1 then loopCost rd c x.2 else 0

/-- The probability that a trial reads exactly `v`. -/
noncomputable def pRd (v : Val) : ℝ := expect (ind fun e => rd e = {v}) T sT

/-- The probability that a trial reads two or more values. -/
noncomputable def sRd : ℝ := expect (ind fun e => 2 ≤ (rd e).card) T sT

/-- The probability that a trial reads anything. -/
noncomputable def rRd : ℝ := expect (ind fun e => (rd e).Nonempty) T sT

theorem ind_nonempty_eq (e : τ) :
    ind (fun e => (rd e).Nonempty) e =
      ∑ v, ind (fun e => rd e = {v}) e + ind (fun e => 2 ≤ (rd e).card) e := by
  classical
  rcases Nat.lt_or_ge (rd e).card 2 with h | h
  · have h2 : ¬ 2 ≤ (rd e).card := by omega
    rcases Nat.lt_or_ge (rd e).card 1 with h1 | h1
    · have h0 : rd e = ∅ := Finset.card_eq_zero.1 (show (rd e).card = 0 by omega)
      have hs : ∀ v, ind (fun e => rd e = {v}) e = 0 := fun v => by
        have : ¬ rd e = {v} := by rw [h0]; exact Finset.empty_ne_singleton v
        simp only [ind, this, ↓reduceIte]
      rw [Finset.sum_eq_zero fun v _ => hs v]
      simp [ind, h0]
    · obtain ⟨w, hw⟩ := Finset.card_eq_one.1 (show (rd e).card = 1 by omega)
      have hs : ∀ v, ind (fun e => rd e = {v}) e = if v = w then 1 else 0 := fun v => by
        by_cases hv : v = w
        · subst hv
          simp only [ind, hw, ↓reduceIte]
        · have : ¬ rd e = {v} := by
            rw [hw]; exact fun h' => hv (Finset.singleton_inj.1 h').symm
          simp only [ind, this, hv, ↓reduceIte]
      rw [Finset.sum_congr rfl fun v _ => hs v]
      simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
      simp [ind, hw]
  · have hne : (rd e).Nonempty := Finset.card_pos.1 (by omega)
    have hs : ∀ v, ind (fun e => rd e = {v}) e = 0 := fun v => by
      have : ¬ rd e = {v} := fun h' => by rw [h', Finset.card_singleton] at h; omega
      simp only [ind, this, ↓reduceIte]
    rw [Finset.sum_eq_zero fun v _ => hs v]
    simp [ind, hne, h]

theorem rRd_eq : rRd T sT rd = ∑ v, pRd T sT rd v + sRd T sT rd := by
  unfold rRd pRd sRd
  rw [← expect_sum, ← expect_add]
  exact congrArg (fun f => expect f T sT) (funext (ind_nonempty_eq rd))

omit [Fintype Val] [DecidableEq Val] in
theorem pRd_nonneg (v : Val) : 0 ≤ pRd T sT rd v := expect_nonneg (ind_nonneg _) _ _
omit [Fintype Val] [DecidableEq Val] in
theorem sRd_nonneg : 0 ≤ sRd T sT rd := expect_nonneg (ind_nonneg _) _ _
omit [Fintype Val] [DecidableEq Val] in
theorem rRd_le_one : rRd T sT rd ≤ 1 := expect_ind_le_one _ _ _

open Classical in
/-- **The finder's success probability**: `s + geom r M · Σ_v p v (r − p v)`. -/
theorem expect_succ (M : ℕ) :
    expect (ind (Succ rd)) (finderG T M) (finderStrat T sT M) =
      sRd T sT rd + geom (rRd T sT rd) M *
        ∑ v, pRd T sT rd v * (rRd T sT rd - pRd T sT rd v) := by
  rw [expect_finderG]
  set r := rRd T sT rd with hrdef
  -- a further trial that reads anything, other than `{v}`
  have hq : ∀ v, expect (fun e => ind (fun e => (rd e).Nonempty) e *
      ind (fun e => rd e ≠ {v}) e) T sT = r - pRd T sT rd v := by
    intro v
    rw [hrdef]
    unfold rRd pRd
    rw [← expect_sub]
    refine congrArg (fun f => expect f T sT) (funext fun e => ?_)
    by_cases h1 : (rd e).Nonempty <;> by_cases h2 : rd e = {v} <;> simp [ind, h1, h2]
  have hpt : ∀ e, expectN T sT M (fun aux => ind (Succ rd) (e, aux)) =
      ind (fun e => 2 ≤ (rd e).card) e +
        ∑ v, ind (fun e => rd e = {v}) e * (geom r M * (r - pRd T sT rd v)) := by
    intro e
    rcases Nat.lt_or_ge (rd e).card 2 with h | h
    · have h2 : ind (fun e => 2 ≤ (rd e).card) e = 0 := by simp [ind]; omega
      rcases Nat.lt_or_ge (rd e).card 1 with h1 | h1
      · have h0 : rd e = ∅ := Finset.card_eq_zero.1 (show (rd e).card = 0 by omega)
        have hS : ∀ aux, ind (Succ rd) (e, aux) = 0 := fun aux => by
          simp [ind, Succ, h0, Finset.empty_ne_singleton]
        simp only [hS, expectN_const, h2, zero_add]
        refine (Finset.sum_eq_zero fun v _ => ?_).symm
        simp [ind, h0, Finset.empty_ne_singleton]
      · obtain ⟨w, hw⟩ := Finset.card_eq_one.1 (show (rd e).card = 1 by omega)
        have hS : ∀ aux, ind (Succ rd) (e, aux) =
            (fun o : Option τ => o.elim 0 fun e' => ind (fun e => rd e ≠ {w}) e')
              (nextRead rd aux) := fun aux => by
          cases hn : nextRead rd aux with
          | none => simp [ind, Succ, hw, hn]
          | some e' =>
              by_cases he' : rd e' = {w}
              · simp [ind, Succ, hw, hn, he']
              · simp [ind, Succ, hw, hn, he']
        have hsum : ∑ v, ind (fun e => rd e = {v}) e * (geom r M * (r - pRd T sT rd v)) =
            geom r M * (r - pRd T sT rd w) := by
          rw [Finset.sum_eq_single w]
          · simp [ind, hw]
          · intro v _ hv
            simp [ind, hw, Finset.singleton_inj, Ne.symm hv]
          · simp
        simp only [hS, h2, zero_add, hsum]
        unfold nextRead
        rw [expectN_find T sT (fun e => (rd e).Nonempty)
          (fun o => o.elim 0 fun e' => ind (fun e => rd e ≠ {w}) e') M]
        simp only [Option.elim_none, mul_zero, add_zero, Option.elim_some]
        rw [hq w]
        rfl
    · have h2 : ind (fun e => 2 ≤ (rd e).card) e = 1 := by simp [ind, h]
      have hS : ∀ aux, ind (Succ rd) (e, aux) = 1 := fun aux => by simp [ind, Succ, h]
      have hsum : ∑ v, ind (fun e => rd e = {v}) e * (geom r M * (r - pRd T sT rd v)) = 0 :=
        Finset.sum_eq_zero fun v _ => by
          have : rd e ≠ {v} := fun h' => by rw [h', Finset.card_singleton] at h; omega
          simp [ind, this]
      simp only [hS, expectN_const, h2, hsum, add_zero]
  rw [show (fun e => expectN T sT M fun aux => ind (Succ rd) (e, aux)) = fun e =>
      ind (fun e => 2 ≤ (rd e).card) e +
        ∑ v, ind (fun e => rd e = {v}) e * (geom r M * (r - pRd T sT rd v)) from funext hpt,
    expect_add, expect_sum]
  unfold sRd
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [show (fun e => ind (fun e => rd e = {v}) e * (geom r M * (r - pRd T sT rd v))) =
    fun e => (geom r M * (r - pRd T sT rd v)) * ind (fun e => rd e = {v}) e from
    funext fun e => mul_comm _ _, expect_const_mul]
  unfold pRd
  ring

omit [DecidableEq Val] in
/-- **Plurality**: `Σ_v p v (r − p v) ≥ (Σ_v p v − p Xv)·r` when `r = Σ_v p v + s` and `Xv` is a
plurality of `p`. -/
theorem plurality_le {p : Val → ℝ} (hp : ∀ v, 0 ≤ p v) {s : ℝ} (hs : 0 ≤ s) {Xv : Val}
    (hX : ∀ v, p v ≤ p Xv) :
    ((∑ v, p v) - p Xv) * ((∑ v, p v) + s) ≤ ∑ v, p v * ((∑ v, p v) + s - p v) := by
  set P := ∑ v, p v
  have hsq : ∑ v, p v * p v ≤ p Xv * P := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun v _ => mul_le_mul_of_nonneg_right (hX v) (hp v)
  have hexp : ∑ v, p v * (P + s - p v) = (P + s) * P - ∑ v, p v * p v := by
    rw [Finset.sum_congr rfl fun v _ => mul_sub (p v) (P + s) (p v), Finset.sum_sub_distrib,
      ← Finset.sum_mul]
    ring
  rw [hexp]
  nlinarith [hX Xv, hp Xv]

theorem geom_mul_le_one {r : ℝ} (h1 : r ≤ 1) (M : ℕ) : r * geom r M ≤ 1 := by
  rw [mul_geom]
  have : 0 ≤ (1 - r) ^ M := pow_nonneg (by linarith) M
  linarith

/-- **The finder succeeds** with probability at least `s + Σ_{v ≠ Xv} p v − 1/(eM)`, for the
plurality `Xv` of the single readings. -/
theorem succ_ge {Xv : Val} (hX : ∀ v, pRd T sT rd v ≤ pRd T sT rd Xv) (M : ℕ) (hM : 1 ≤ M) :
    sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd Xv) - 1 / (Real.exp 1 * M) ≤
      expect (ind (Succ rd)) (finderG T M) (finderStrat T sT M) := by
  rw [expect_succ]
  set r := rRd T sT rd
  set B := (∑ v, pRd T sT rd v) - pRd T sT rd Xv
  have hr : r = (∑ v, pRd T sT rd v) + sRd T sT rd := rRd_eq T sT rd
  have hr0 : 0 ≤ r := expect_nonneg (ind_nonneg _) _ _
  have hr1 : r ≤ 1 := rRd_le_one T sT rd
  have hB0 : 0 ≤ B := by
    have := Finset.single_le_sum (f := pRd T sT rd) (fun v _ => pRd_nonneg T sT rd v)
      (Finset.mem_univ Xv)
    linarith
  have hBr : B ≤ r := by linarith [pRd_nonneg T sT rd Xv, sRd_nonneg T sT rd]
  have hpl := plurality_le (pRd_nonneg T sT rd) (sRd_nonneg T sT rd) hX
  rw [← hr] at hpl
  have hg0 : 0 ≤ geom r M := geom_nonneg hr1 M
  have hmiss : r * (1 - r) ^ M ≤ 1 / (Real.exp 1 * M) := mul_one_sub_pow_le r hr0 hr1 M hM
  have hgeo : geom r M * (B * r) = B * (1 - (1 - r) ^ M) := by
    rw [mul_comm B r, ← mul_assoc, mul_comm (geom r M) r, mul_geom]; ring
  have hB : B * (1 - r) ^ M ≤ r * (1 - r) ^ M :=
    mul_le_mul_of_nonneg_right hBr (pow_nonneg (by linarith) M)
  have key : B - 1 / (Real.exp 1 * M) ≤ geom r M * ∑ v, pRd T sT rd v * (r - pRd T sT rd v) :=
    calc B - 1 / (Real.exp 1 * M) ≤ B * (1 - (1 - r) ^ M) := by nlinarith
      _ = geom r M * (B * r) := hgeo.symm
      _ ≤ _ := mul_le_mul_of_nonneg_left hpl hg0
  linarith

omit [Fintype Val] [DecidableEq Val] in
open Classical in
theorem expectN_loopCost (c : τ → ℝ) : ∀ M : ℕ,
    expectN T sT M (loopCost rd c) = geom (rRd T sT rd) M * expect c T sT
  | 0 => by simp [expectN, loopCost, geom]
  | M + 1 => by
      have ih := expectN_loopCost c M
      have h : ∀ e, expectN T sT M (fun es => loopCost rd c (e :: es)) =
          c e + (1 - ind (fun e => (rd e).Nonempty) e) * (geom (rRd T sT rd) M * expect c T sT) := by
        intro e
        by_cases he : (rd e).Nonempty
        · simp only [loopCost, he, ↓reduceIte, add_zero, expectN_const]
          simp [ind, he]
        · simp only [loopCost, he, ↓reduceIte]
          rw [expectN_add, expectN_const, ih]
          simp [ind, he]
      simp only [expectN]
      rw [show (fun e => expectN T sT M fun es => loopCost rd c (e :: es)) = fun e =>
          c e + (geom (rRd T sT rd) M * expect c T sT) *
            (1 - ind (fun e => (rd e).Nonempty) e) from funext fun e => by rw [h e]; ring,
        expect_add, expect_const_mul, expect_sub, expect_const, geom_succ]
      unfold rRd
      ring

omit [Fintype Val] [DecidableEq Val] in
open Classical in
/-- **The finder's expected cost** is at most twice a trial's. -/
theorem expect_finderCost_le (c : τ → ℝ) (hc : ∀ e, 0 ≤ c e) (M : ℕ) :
    expect (finderCost rd c) (finderG T M) (finderStrat T sT M) ≤ 2 * expect c T sT := by
  rw [expect_finderG]
  have hc0 : 0 ≤ expect c T sT := expect_nonneg hc _ _
  have hpt : ∀ e, expectN T sT M (fun aux => finderCost rd c (e, aux)) =
      c e + ind (fun e => (rd e).card = 1) e * (geom (rRd T sT rd) M * expect c T sT) := by
    intro e
    by_cases h1 : (rd e).card = 1
    · simp only [finderCost, h1, ↓reduceIte]
      rw [expectN_add, expectN_const, expectN_loopCost]
      simp [ind, h1]
    · simp only [finderCost, h1, ↓reduceIte, add_zero, expectN_const]
      simp [ind, h1]
  rw [show (fun e => expectN T sT M fun aux => finderCost rd c (e, aux)) = fun e =>
      c e + (geom (rRd T sT rd) M * expect c T sT) * ind (fun e => (rd e).card = 1) e from
      funext fun e => by rw [hpt e]; ring, expect_add, expect_const_mul]
  set r := rRd T sT rd
  -- a single reading is a reading
  have h1r : expect (ind fun e => (rd e).card = 1) T sT ≤ r :=
    expect_mono (ind_mono fun e (h : (rd e).card = 1) => Finset.card_pos.1 (by omega)) T sT
  have hr0 : 0 ≤ r := expect_nonneg (ind_nonneg _) _ _
  have hrg := geom_mul_le_one (rRd_le_one T sT rd) M
  have hg0 : 0 ≤ geom r M := geom_nonneg (rRd_le_one T sT rd) M
  have h1 : 0 ≤ expect (ind fun e => (rd e).card = 1) T sT := expect_nonneg (ind_nonneg _) _ _
  have : expect (ind fun e => (rd e).card = 1) T sT * geom r M ≤ 1 :=
    (mul_le_mul_of_nonneg_right h1r hg0).trans hrg
  nlinarith

/-- **The link mass under expected-time collision resistance.** If every success of the finder
outputs a collision of `Hc`, and the finder obeys the generic bound, then
`s + Σ_{v ≠ Xv} p v ≤ 2·E[trial cost]/2^256 + 1/(eM)` for the plurality `Xv`. -/
theorem link_mass_le {Dh : Type} (Hc : List UInt8 → Dh)
    (out : τ × List τ → Option (List UInt8 × List UInt8)) (c : τ → ℝ) (hc : ∀ e, 0 ≤ c e)
    (M : ℕ) (hM : 1 ≤ M)
    (hout : ∀ x, Succ rd x → ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b)
    (hCR : Assumptions.SHA512CRExpected Hc (finderG T M) (finderStrat T sT M) out
      (finderCost rd c))
    {Xv : Val} (hX : ∀ v, pRd T sT rd v ≤ pRd T sT rd Xv) :
    sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd Xv) ≤
      2 * expect c T sT / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  have h1 := succ_ge T sT rd hX M hM
  have h2 : expect (ind (Succ rd)) (finderG T M) (finderStrat T sT M) ≤
      (prob (fun x => ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b) (finderG T M)
        (finderStrat T sT M)).toReal := by
    rw [prob_toReal]
    exact expect_mono (ind_mono hout) _ _
  have h3 := hCR
  unfold Assumptions.SHA512CRExpected at h3
  have h4 := expect_finderCost_le T sT rd c hc M
  have h5 : expect (finderCost rd c) (finderG T M) (finderStrat T sT M) / 2 ^ (256 : ℝ) ≤
      2 * expect c T sT / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right h4 (by positivity)
  linarith

end LinkFinder

end FlockSoundness
