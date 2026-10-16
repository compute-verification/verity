import Proofs.Flock.Claims.Prob

/-!
# `E^rep`: its reruns and its amplification

`E^rep` replays the judged run and, if it accepted, makes `m` waits, each over `K` fresh coin draws and stopping at `k`
accepted reruns; it returns the first assembled candidate that `check` accepts.

* **Worst case** (`ERep_runs_le`): at most `1 + m·K` reruns, on every prover, tape and coin (the first half of
  `ERepCost`; the expected half, and `ERepCost` itself, are in `ERepCost`).
* **The bad event factors** (`ERep_bad_le`): when `check` is sound for `R`, `E^rep` fails on an accepted run only if
  every wait fails, and the waits are independent given the tape, so
  `Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼_ω[ε_ω · f_ω^m]`, with `ε_ω` the acceptance probability at tape `ω` and `f_ω` one wait's
  failure probability there.
* **Amplification** (`amplify`, `ERep_amplify`): if each tape has `ε_ω · f_ω ≤ a_ω + rate · ε_ω` (one wait's
  knowledge bound), then `Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼[a]/rate + (2·rate)^m`: a tape with `a_ω ≤ rate·ε_ω` has
  `f_ω ≤ 2·rate`, and any other has `ε_ω < a_ω/rate`.
-/

namespace Claims

/-! ## Rerun programs -/

/-- Running a bound program: its result is the continuation's on the first part's result, and its reruns add. -/
theorem RP.eval_bind {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).1 = ((f (p.eval H run).1).eval H run).1 ∧
      ((p.bind f).eval H run).2.2 = (p.eval H run).2.2 + ((f (p.eval H run).1).eval H run).2.2 := by
  induction p with
  | ret a => simp [RP.bind, RP.eval]
  | hash q k ih =>
    simp only [RP.bind, RP.eval]
    exact ⟨(ih _).1, (ih _).2⟩
  | rerun r k ih =>
    simp only [RP.bind, RP.eval]
    refine ⟨(ih _).1, ?_⟩
    rw [(ih _).2]; ring

/-- A wait reruns the prover at most once per coin draw it holds. -/
theorem waitAcc_runs_le {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List Seen), ((waitAcc k rs acc).eval H run).2.2 ≤ rs.length := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAcc, RP.eval]
  | cons r rs ih =>
    intro acc
    simp only [waitAcc]
    split
    · simp [RP.eval]
    · simp only [RP.eval, List.length_cons]
      exact Nat.add_le_add_right (ih _) 1

/-- The waits rerun the prover at most once per coin draw they hold. -/
theorem firstWait_runs_le {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).2.2 ≤ (rss.map List.length).sum := by
  intro rss
  induction rss with
  | nil => simp [firstWait, RP.eval]
  | cons rs rss ih =>
    simp only [firstWait, List.map_cons, List.sum_cons]
    rw [(RP.eval_bind H run _ _).2]
    refine Nat.add_le_add (waitAcc_runs_le H run k rs []) ?_
    split
    · split
      · simp [RP.eval]
      · exact ih
    · exact ih

/-- What one wait over the coin draws `rs` yields: some candidate that `check` accepts. -/
def WaitOk {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (rs : List R) : Prop :=
  ∃ w, assemble (((waitAcc k rs []).eval H run).1.flatMap opens) = some w ∧ check w = true

/-- The waits return only candidates `check` accepts. -/
theorem firstWait_some {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ (rss : List (List R)) (w : W), ((firstWait k opens assemble check rss).eval H run).1 = some w →
      check w = true := by
  intro rss
  induction rss with
  | nil => intro w h; simp [firstWait, RP.eval] at h
  | cons rs rss ih =>
    intro w h
    simp only [firstWait] at h
    rw [(RP.eval_bind H run _ _).1] at h
    split at h
    · split at h
      · simp only [RP.eval, Option.some.injEq] at h
        subst h; assumption
      · exact ih w h
    · exact ih w h

/-- The waits return nothing only if every wait failed. -/
theorem firstWait_none {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), ((firstWait k opens assemble check rss).eval H run).1 = none →
      ∀ rs ∈ rss, ¬ WaitOk H run k opens assemble check rs := by
  intro rss
  induction rss with
  | nil => intro _ rs h; simp at h
  | cons rs rss ih =>
    intro h rs' hrs' hok
    simp only [firstWait] at h
    rw [(RP.eval_bind H run _ _).1] at h
    rcases List.mem_cons.1 hrs' with rfl | hrs'
    · obtain ⟨w, hw, hc⟩ := hok
      rw [hw] at h
      simp only [hc, ↓reduceIte, RP.eval, reduceCtorEq] at h
    · split at h
      · split at h
        · simp [RP.eval] at h
        · exact ih h rs' hrs' hok
      · exact ih h rs' hrs' hok

/-! ## `E^rep`'s worst case -/

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`E^rep` reruns the prover at most `1 + m·K` times**, on every prover, tape and coin. -/
theorem ERep_runs_le (H : Msg → Msg) (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    ExtractRunsLe H (ERep V m k K opens assemble check) (fun x => 1 + m x * K x) := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval, ERep, RP.eval]
  rw [Nat.add_comm 1]
  refine Nat.add_le_add_right ?_ 1
  split
  · refine (firstWait_runs_le H _ _ _ _ _ _).trans (le_of_eq ?_)
    simp [List.map_map, Function.comp_def, List.sum_replicate]
  · simp [RP.eval]

/-- `ε_ω`: the probability that a run at tape `ω` is accepted. -/
noncomputable def accPr (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) : ℝ :=
  Claims.Pr fun r : Fin (V.k x) → Coin => (run H P V x ω r).out = .acc

/-- `f_ω`: the probability that one of `E^rep`'s waits at tape `ω`, over `K` fresh coin draws, fails. -/
noncomputable def waitFail (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) : ℝ :=
  Claims.Pr fun ρ : Fin K → Fin (V.k x) → Coin =>
    ¬ WaitOk H (run H P V x ω) k opens assemble check ((List.finRange K).map ρ)

/-- **The bad event factors**: when `check` is sound for `R`, `Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼_ω[ε_ω · f_ω^m]`. -/
theorem ERep_bad_le (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (hcheck : ∀ x w, check x w = true → R x w)
    (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H V R (ERep V m k K opens assemble check) x P ≤
      Ex fun ω => accPr H V P x ω * waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ m x := by
  unfold PrBad accPr waitFail
  simp only [← Pr_pi]
  rw [← Pr_indep]
  refine Pr_mono fun s ⟨hacc, hbad⟩ => ⟨hacc, fun i hok => hbad ?_⟩
  simp only [Extractor.eval, ERep, RP.eval, Trace.seen, hacc, ↓reduceIte] at hbad ⊢
  generalize hres : ((firstWait (k x) opens (assemble x) (check x)
    ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (s.2.2 i))).eval H (run H P V x s.1)).1 = res
    at hbad ⊢
  cases res with
  | some w => exact hcheck x w (firstWait_some H _ _ _ _ _ _ w hres)
  | none =>
    exact absurd hok (firstWait_none H _ _ _ _ _ _ hres _ (List.mem_map.2 ⟨i, List.mem_finRange i, rfl⟩))

end ERep

/-! ## Amplification -/

/-- **One tape's amplification**: from one wait's bound `ε·f ≤ a + rate·ε`, `m` independent waits give
`ε·f^m ≤ a/rate + (2·rate)^m`. -/
theorem amplify {ε f a rate : ℝ} (m : ℕ) (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hf : 0 ≤ f) (hf1 : f ≤ 1) (ha : 0 ≤ a)
    (hrate : 0 < rate) (h : ε * f ≤ a + rate * ε) : ε * f ^ m ≤ a / rate + (2 * rate) ^ m := by
  have hdiv : 0 ≤ a / rate := div_nonneg ha hrate.le
  have hpow : 0 ≤ (2 * rate) ^ m := pow_nonneg (by linarith) m
  have hfm1 : f ^ m ≤ 1 := pow_le_one₀ hf hf1
  by_cases hcase : a ≤ rate * ε
  · rcases hε.lt_or_eq with hε0 | rfl
    · have hf2 : f ≤ 2 * rate := by
        by_contra hlt
        push Not at hlt
        have : ε * (2 * rate) < ε * f := mul_lt_mul_of_pos_left hlt hε0
        linarith
      have : f ^ m ≤ (2 * rate) ^ m := pow_le_pow_left₀ hf hf2 m
      have : ε * f ^ m ≤ f ^ m := mul_le_of_le_one_left (pow_nonneg hf m) hε1
      linarith
    · simp only [zero_mul]; positivity
  · push Not at hcase
    have hεa : ε ≤ a / rate := by rw [le_div_iff₀ hrate]; linarith
    have : ε * f ^ m ≤ ε := mul_le_of_le_one_right hε hfm1
    linarith

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`E^rep`'s amplification**: when `check` is sound for `R` and each tape has one wait's knowledge bound
`ε_ω · f_ω ≤ a_ω + rate · ε_ω`, `Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼[a]/rate + (2·rate)^m`. -/
theorem ERep_amplify (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (hcheck : ∀ x w, check x w = true → R x w)
    (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (a : Ω → ℝ) (ha : ∀ ω, 0 ≤ a ω) {rate : ℝ}
    (hrate : 0 < rate)
    (hwait : ∀ ω, accPr H V P x ω * waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ≤
      a ω + rate * accPr H V P x ω) :
    PrBad H V R (ERep V m k K opens assemble check) x P ≤ Ex a / rate + (2 * rate) ^ m x := by
  refine (ERep_bad_le H V R m k K opens assemble check hcheck x P).trans ?_
  calc Ex (fun ω => accPr H V P x ω * waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ m x)
      ≤ Ex (fun ω => a ω / rate + (2 * rate) ^ m x) :=
        Ex_mono fun ω => amplify (m x) (Pr_nonneg _) (Pr_le_one _) (Pr_nonneg _) (Pr_le_one _) (ha ω) hrate
          (hwait ω)
    _ = Ex a / rate + (2 * rate) ^ m x := by rw [Ex_add, Ex_div, Ex_const]

end ERep

end Claims
