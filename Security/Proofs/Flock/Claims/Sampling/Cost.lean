import Definitions.Claims.Sampling
import Proofs.Flock.Claims.Sampling.Extract

/-!
# The sampling extractor's rerun programs: reruns, own calls, and what they return

The claims block's extractor `E` (`Definitions.Claims.Sampling`) is built of `ofOC`, `flatMapRP`, `waitAccC` and
`checkRevealed`. These are their laws, on every run.

* `ofOC_runs`, `flatMapRP_runs`, `waitAccC_runs_eq`: their reruns. `ownLog_bind`, `ownLog_flatMapRP`, `ownLog_ofOC`,
  `ownLog_of_noHash`: their own `H` calls.
* `eval_ofOC`, `flatMapRP_mem`, `checkRevealed_mem`: what they return. Each opening `checkRevealed` returns passes the
  commitment's check under its roots, and that check's calls are among its own.
* `waitAccC_sum`, `accExp_le`: the weights of a wait's accepted draws, `ε · G(K, a) ≤ (k − a) · 𝔼[1[acc] · g]`.
-/

namespace Claims.Sampling

open Claims

/-! ## The block's rerun programs: reruns and own calls -/

theorem ofOC_runs {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ o : OC α, ((ofOC o : RP R α).eval H run).2.2 = 0
  | .ret _ => rfl
  | .query q k => by
    simp only [ofOC, RP.eval]
    exact ofOC_runs H run (k (H q))

theorem flatMapRP_runs {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ l : List β, ((flatMapRP f l).eval H run).2.2 = (l.map fun b => ((f b).eval H run).2.2).sum
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.map_cons, List.sum_cons]
    rw [(RP.eval_bind H run _ _).2, (RP.eval_bind H run _ _).2, flatMapRP_runs H run f bs]
    simp [RP.eval]

/-- A bound program's own calls: the first part's, then the continuation's on its result. -/
theorem ownLog_bind {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ownLog H run (p.bind f) = ownLog H run p ++ ownLog H run (f (p.eval H run).1) := by
  induction p with
  | ret a => simp [RP.bind, ownLog, RP.eval]
  | hash q k ih => simp only [RP.bind, ownLog, RP.eval, ih, List.cons_append]
  | rerun r k ih => simp only [RP.bind, ownLog, RP.eval, ih]

theorem ownLog_of_noHash {R α : Type} (H : Msg → Msg) (run : R → Trace) {p : RP R α} (hp : NoHash p) :
    ownLog H run p = [] := by
  induction hp with
  | ret a => rfl
  | rerun r k _ ih => exact ih _

theorem ownLog_flatMapRP {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ l : List β, (ownLog H run (flatMapRP f l)).length = (l.map fun b => (ownLog H run (f b)).length).sum
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.map_cons, List.sum_cons]
    rw [ownLog_bind, ownLog_bind, List.length_append, List.length_append, ownLog_flatMapRP H run f bs]
    simp [ownLog]

theorem NoHash.waitAccC {R : Type} (k : ℕ) :
    ∀ (rs : List R) (acc : List (R × Seen)), NoHash (Sampling.waitAccC k rs acc) := by
  intro rs
  induction rs with
  | nil => intro acc; exact NoHash.ret _
  | cons r rs ih =>
    intro acc
    simp only [Sampling.waitAccC]
    split
    · exact NoHash.ret _
    · exact NoHash.rerun r _ fun v => ih _

theorem waitAccC_runs_eq {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List (R × Seen)),
      ((waitAccC k rs acc).eval H run).2.2 = waitRuns run k rs acc.length := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAccC, waitRuns, RP.eval]
  | cons r rs ih =>
    intro acc
    by_cases hk : k ≤ acc.length
    · simp [waitAccC, waitRuns, hk, RP.eval]
    · simp only [waitAccC, waitRuns, hk, ↓reduceIte, RP.eval, ih]
      by_cases ha : (run r).out = .acc
      · have hs : (run r).seen.out = .acc := ha
        simp [hs, ha]
      · have hs : ¬ (run r).seen.out = .acc := ha
        simp [hs, ha]

theorem waitRuns_le_length {R : Type} (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (a : ℕ), waitRuns run k rs a ≤ rs.length := by
  intro rs
  induction rs with
  | nil => intro a; simp [waitRuns]
  | cons r rs ih =>
    intro a
    simp only [waitRuns, List.length_cons]
    split
    · exact Nat.zero_le _
    · exact Nat.add_le_add_right (ih _) 1

theorem waitAccC_length_le {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List (R × Seen)), acc.length ≤ k → ((waitAccC k rs acc).eval H run).1.length ≤ k := by
  intro rs
  induction rs with
  | nil => intro acc h; simpa [waitAccC, RP.eval] using h
  | cons r rs ih =>
    intro acc h
    by_cases hk : k ≤ acc.length
    · simp only [waitAccC, hk, ↓reduceIte, RP.eval]
      exact h
    · simp only [waitAccC, hk, ↓reduceIte, RP.eval]
      refine ih _ ?_
      split
      · rw [List.length_append, List.length_singleton]
        omega
      · exact h

/-- The weights `g` of a wait's accepted draws among `rs`, from `a` accepted so far. -/
noncomputable def accSum {R : Type} (run : R → Trace) (k : ℕ) (g : R → ℝ) : List R → ℕ → ℝ
  | [], _ => 0
  | r :: rs, a =>
    if k ≤ a then 0 else if (run r).out = .acc then g r + accSum run k g rs (a + 1) else accSum run k g rs a

/-- **The accepted reruns' weights**: those `waitAccC` keeps are the accepted draws `accSum` counts. -/
theorem waitAccC_sum {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (g : R → ℝ) :
    ∀ (rs : List R) (acc : List (R × Seen)),
      (((waitAccC k rs acc).eval H run).1.map fun p => g p.1).sum =
        (acc.map fun p => g p.1).sum + accSum run k g rs acc.length := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAccC, RP.eval, accSum]
  | cons r rs ih =>
    intro acc
    by_cases hk : k ≤ acc.length
    · simp [waitAccC, accSum, hk, RP.eval]
    · simp only [waitAccC, accSum, hk, ↓reduceIte, RP.eval]
      rw [ih]
      by_cases ha : (run r).out = .acc
      · have hs : (run r).seen.out = .acc := ha
        simp only [hs, ha, ↓reduceIte, List.map_append, List.sum_append, List.length_append,
          List.length_singleton, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
        ring
      · have hs : ¬ (run r).seen.out = .acc := ha
        simp only [hs, ha, ↓reduceIte]

/-- `G(K, a)`: the accepted draws' weights over `K` uniform draws, from `a` accepted so far. -/
noncomputable def accExp {D : Type} [Fintype D] (run : D → Trace) (k : ℕ) (g : D → ℝ) (K a : ℕ) : ℝ :=
  Ex fun σ : Fin K → D => accSum run k g ((List.finRange K).map σ) a

/-- **The accepted draws' weights**: `ε · G(K, a) ≤ (k − a) · 𝔼[1[acc] · g]`, so the `k` accepted draws weigh at most
`k · 𝔼[g | acc]` in expectation. -/
theorem accExp_le {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (k : ℕ) (g : D → ℝ) (hg : ∀ d, 0 ≤ g d)
    (K : ℕ) : ∀ a, a ≤ k → Pr (fun d => (run d).out = .acc) * accExp run k g K a ≤
      (k - a) * Ex (fun d => (if (run d).out = .acc then (1 : ℝ) else 0) * g d) := by
  have hμ : 0 ≤ Ex (fun d => (if (run d).out = .acc then (1 : ℝ) else 0) * g d) :=
    Ex_nonneg fun d => mul_nonneg (by split_ifs <;> norm_num) (hg d)
  induction K with
  | zero =>
    intro a ha
    have h0 : accExp run k g 0 a = 0 := by simp [accExp, accSum, Ex]
    have hka : (0 : ℝ) ≤ k - a := by
      have : (a : ℝ) ≤ k := by exact_mod_cast ha
      linarith
    rw [h0, mul_zero]
    exact mul_nonneg hka hμ
  | succ K ih =>
    intro a ha
    obtain hlt | rfl := ha.lt_or_eq
    · let F : D → (Fin K → D) → ℝ := fun d σ => (if (run d).out = .acc then (1 : ℝ) else 0) * g d +
        accSum run k g ((List.finRange K).map σ) (if (run d).out = .acc then a + 1 else a)
      have hstep : ∀ ρ : Fin (K + 1) → D,
          accSum run k g ((List.finRange (K + 1)).map ρ) a = F (ρ 0) fun i => ρ i.succ := by
        intro ρ
        rw [List.finRange_succ, List.map_cons, List.map_map]
        simp only [F, accSum, not_le.2 hlt, ↓reduceIte, Function.comp_def]
        split_ifs <;> ring
      have h1 : accExp run k g (K + 1) a = Ex fun d => Ex fun σ : Fin K → D => F d σ :=
        (congrArg Ex (funext hstep)).trans (Ex_cons K F)
      have h2 : ∀ d : D, (Ex fun σ : Fin K → D => F d σ) =
          (if (run d).out = .acc then (1 : ℝ) else 0) * g d +
            (if (run d).out = .acc then accExp run k g K (a + 1) else accExp run k g K a) := by
        intro d
        simp only [F]
        rw [Ex_add, Ex_const]
        by_cases hc : (run d).out = .acc <;> simp only [hc, ↓reduceIte, accExp]
      have hE : accExp run k g (K + 1) a =
          Ex (fun d => (if (run d).out = .acc then (1 : ℝ) else 0) * g d) +
            (Pr (fun d => (run d).out = .acc) * accExp run k g K (a + 1) +
              (1 - Pr fun d => (run d).out = .acc) * accExp run k g K a) := by
        rw [h1, congrArg Ex (funext h2), Ex_add, Ex_ite]
      have hε0 := Pr_nonneg fun d => (run d).out = .acc
      have hε1 := Pr_le_one fun d => (run d).out = .acc
      have ih1 := mul_le_mul_of_nonneg_left (ih (a + 1) (Nat.succ_le_of_lt hlt)) hε0
      have ih0 := mul_le_mul_of_nonneg_left (ih a ha) (sub_nonneg.2 hε1)
      push_cast at ih1
      rw [hE]
      set ε := Pr fun d => (run d).out = .acc
      set μ := Ex (fun d => (if (run d).out = .acc then (1 : ℝ) else 0) * g d)
      calc ε * (μ + (ε * accExp run k g K (a + 1) + (1 - ε) * accExp run k g K a))
          = ε * μ + ε * (ε * accExp run k g K (a + 1)) + (1 - ε) * (ε * accExp run k g K a) := by ring
        _ ≤ ε * μ + ε * ((k - (a + 1)) * μ) + (1 - ε) * ((k - a) * μ) := by linarith
        _ = (k - a) * μ := by ring
    · simp [accExp, accSum, List.finRange_succ, Ex]

theorem waitRuns_map {R R' : Type} (run : R → Trace) (k : ℕ) (f : R' → R) :
    ∀ (l : List R') (a : ℕ), waitRuns run k (l.map f) a = waitRuns (run ∘ f) k l a := by
  intro l
  induction l with
  | nil => intro a; rfl
  | cons r l ih => intro a; simp only [List.map_cons, waitRuns, ih, Function.comp]

theorem Ex_listSum {α ι : Type} [Fintype α] (l : List ι) (f : ι → α → ℝ) :
    Ex (fun a => (l.map fun i => f i a).sum) = (l.map fun i => Ex (f i)).sum := by
  induction l with
  | nil => simp [Ex]
  | cons i l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [Ex_add (f i) (fun a => (l.map fun i => f i a).sum), ih]

/-! ## What they return -/

theorem LogColl.mono {l l' : List (Msg × Msg)} (h : LogColl l) (hs : ∀ q ∈ l, q ∈ l') : LogColl l' := by
  obtain ⟨p, hp, p', hp', hne, heq⟩ := h
  exact ⟨p, hs p hp, p', hs p' hp', hne, heq⟩

theorem ownLog_ofOC {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ c : OC α, ownLog H run (ofOC c : RP R α) = (c.eval H).2
  | .ret _ => rfl
  | .query q k => by
    simp only [ofOC, ownLog, OC.eval]
    rw [ownLog_ofOC H run (k (H q))]

theorem eval_ofOC {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ c : OC α, ((ofOC c : RP R α).eval H run).1 = (c.eval H).1
  | .ret _ => rfl
  | .query q k => by
    simp only [ofOC, RP.eval, OC.eval]
    exact eval_ofOC H run (k (H q))

/-- What `flatMapRP` returns comes from one element's program, whose own calls are among the whole program's. -/
theorem flatMapRP_mem {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ (l : List β), ∀ c ∈ ((flatMapRP f l).eval H run).1,
      ∃ b ∈ l, c ∈ ((f b).eval H run).1 ∧ ∀ q ∈ ownLog H run (f b), q ∈ ownLog H run (flatMapRP f l)
  | [] => by simp [flatMapRP, RP.eval]
  | b :: bs => by
    intro c hc
    simp only [flatMapRP] at hc ⊢
    rw [(RP.eval_bind H run _ _).1, (RP.eval_bind H run _ _).1] at hc
    simp only [RP.eval, List.mem_append] at hc
    rw [ownLog_bind, ownLog_bind]
    rcases hc with hc | hc
    · exact ⟨b, by simp, hc, fun q hq => List.mem_append_left _ hq⟩
    · obtain ⟨b', hb', hc', hq⟩ := flatMapRP_mem H run f bs c hc
      exact ⟨b', List.mem_cons_of_mem _ hb', hc', fun q h => List.mem_append_right _ (List.mem_append_left _ (hq q h))⟩

/-- **What `checkRevealed` returns passes its check**: each pair is one of `rs`, its opening passes the commitment's
check under its roots, and that check's calls are among `checkRevealed`'s own. -/
theorem checkRevealed_mem {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : CommitmentScheme Bool)
    (rs : List (List Msg × Opening Bool)) :
    ∀ p ∈ ((checkRevealed (R := R) Cm rs).eval H run).1, p ∈ rs ∧ Cm.Opens H p.1 p.2 = true ∧
      ∀ q ∈ ((Cm.opens p.1 p.2).eval H).2, q ∈ ownLog H run (checkRevealed (R := R) Cm rs) := by
  intro p hp
  obtain ⟨p', hp', hc, hq⟩ := flatMapRP_mem H run _ rs p hp
  rw [(RP.eval_bind H run _ _).1, eval_ofOC] at hc
  rw [ownLog_bind, ownLog_ofOC] at hq
  simp only [RP.eval] at hc
  split_ifs at hc with hok
  · obtain rfl := List.mem_singleton.1 hc
    exact ⟨hp', hok, fun q h => hq q (List.mem_append_left _ h)⟩
  · simp at hc

theorem checkRevealed_runs {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : CommitmentScheme Bool)
    (rs : List (List Msg × Opening Bool)) : ((checkRevealed (R := R) Cm rs).eval H run).2.2 = 0 := by
  unfold checkRevealed
  rw [flatMapRP_runs]
  have h : ∀ p : List Msg × Opening Bool, (((ofOC (Cm.opens p.1 p.2)).bind fun ok =>
      (RP.ret (if ok then [p] else []) : RP R (List (List Msg × Opening Bool)))).eval H run).2.2 = 0 := fun p => by
    rw [(RP.eval_bind H run _ _).2, ofOC_runs]
    rfl
  simp [h]

end Claims.Sampling
