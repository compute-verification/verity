import Definitions.Claims.Sampling
import Proofs.Flock.Claims.Sampling.Extract

/-!
# The sampling extractor's cost

The claims block's `Efficient` (`Definitions.Claims.Sampling`) states the sampling extractor `E`'s reruns: at most
`1 + K + k·m′·K′`, and `1 + k·(1 + m′·k′)` in expectation. Both hold on every deployment, prover and tape.

`E` replays the judged run; if it accepted, it waits for `k` accepted reruns among `K` fresh draws (`waitAccC`), and at
each accepted rerun `r` runs the session's `E^rep` (`inner`): `m′` waits for `k′` accepted reruns among `K′` draws that
keep `r`'s draw coin (`atDraw r`). The session's waits at `r` make `g(r) = m′·N′(r)` reruns in expectation, with
`a(r)·N′(r) ≤ k′` for `a(r)` the acceptance probability of the runs at `r`'s draw (`waitRuns_exp`). The accepted reruns'
weights `g`, from `a` accepted so far, average `G(K, a)` with `ε·G(K, a) ≤ (k − a)·μ` for `μ = 𝔼[1[acc]·g]`
(`accExp_le`, the induction of `waitRuns_exp`), and `μ = 𝔼[a·g] ≤ m′·k′` since `g` reads only the draw coin
(`Ex_acc_forkOn` off coin 0). So an accepted judged run (probability `ε`) adds at most `ε·(k/ε + k·m′k′/ε)`.

* `E_runs_le`, `E_runs_exp`: the two bounds.
* `OpenCallsLe`, `E_own_le`: `E`'s own SHA-512 calls are its openings' checks, at most `k·openCalls` when one rerun's
  checks make at most `openCalls` (what the block's `Deploy.openCalls` describes, and does not state).
* `Efficient_of_decoder`: under `OpenCallsLe`, `Efficient` is a decoder that meets Flock's `ListComplete`, with its
  stated complexity.
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

theorem checkOpens_runs {Coin Col T R : Type} (H : Msg → Msg) (run : R → Trace) (D : Deploy Coin Col T) (s : Stmt)
    (os : List Opening) : ((checkOpens (R := R) D s os).eval H run).2.2 = 0 := by
  unfold checkOpens
  rw [flatMapRP_runs]
  have h : ∀ o : Opening, (((ofOC (openOC D.path s o)).bind fun ok =>
      (RP.ret (if ok then [o] else []) : RP R (List Opening))).eval H run).2.2 = 0 := fun o => by
    rw [(RP.eval_bind H run _ _).2, ofOC_runs]
    rfl
  simp [h]

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

/-! ## The session's reruns at an accepted rerun -/

section Cost

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The session's waits' reruns at an accepted rerun `r`: each wait over `K′` draws at `r`'s draw coin. -/
def innerRuns {n : ℕ} (run : (Fin n → Coin) → Trace) (k' K' : ℕ) {m' : ℕ} (r : Fin n → Coin)
    (ρ : Fin m' → Fin K' → Fin n → Coin) : ℕ :=
  ∑ i, waitRuns (run ∘ atDraw r) k' ((List.finRange K').map (ρ i)) 0

theorem waitRuns_map {R R' : Type} (run : R → Trace) (k : ℕ) (f : R' → R) :
    ∀ (l : List R') (a : ℕ), waitRuns run k (l.map f) a = waitRuns (run ∘ f) k l a := by
  intro l
  induction l with
  | nil => intro a; rfl
  | cons r l ih => intro a; simp only [List.map_cons, waitRuns, ih, Function.comp]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem inner_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (r : Fin (D.V.k x) → Coin) (v : Seen)
    (ρ : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin) (run : (Fin (D.V.k x) → Coin) → Trace) :
    ((inner H D decode x s r v ρ).eval H run).2.2 ≤ innerRuns run (D.innerK x) (D.innerCap x) r ρ := by
  have hsum : (((List.finRange (D.innerWaits x)).map fun i =>
      (List.finRange (D.innerCap x)).map fun j => atDraw r (ρ i j)).map fun rs => waitRuns run (D.innerK x) rs 0).sum =
      innerRuns run (D.innerK x) (D.innerCap x) r ρ := by
    rw [List.map_map, innerRuns, Fin.sum_univ_def]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]
    rw [← waitRuns_map, List.map_map]
    rfl
  unfold inner
  split
  · simp [RP.eval]
  · next xS _ _ =>
    rw [(RP.eval_bind H run _ _).2]
    have hw := firstWait_runs_le_sum H run (D.innerK x) D.sessionCols
      (Flock.assemble H D.flock.parse decode D.flock.read xS) (Flock.check H D.flock.parse xS)
      ((List.finRange (D.innerWaits x)).map fun i => (List.finRange (D.innerCap x)).map fun j => atDraw r (ρ i j))
    rw [hsum] at hw
    split
    · simpa [RP.eval] using hw
    · rw [checkOpens_runs, Nat.add_zero]
      exact hw

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem inner_runs_le_cap (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (r : Fin (D.V.k x) → Coin) (v : Seen)
    (ρ : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin) (run : (Fin (D.V.k x) → Coin) → Trace) :
    ((inner H D decode x s r v ρ).eval H run).2.2 ≤ D.innerWaits x * D.innerCap x := by
  refine (inner_runs_le H D decode x s r v ρ run).trans ?_
  unfold innerRuns
  refine (Finset.sum_le_card_nsmul _ _ (D.innerCap x) fun i _ => ?_).trans_eq ?_
  · exact (waitRuns_le_length _ _ _ _).trans_eq (by simp)
  · simp

/-! ## The extractor's reruns -/

/-- **`E` reruns the prover at most `1 + K + k·m′·K′` times**, on every prover, tape and coin. -/
theorem E_runs_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    ExtractRunsLe H (E H D decode) (fun x => 1 + D.cap x + D.k x * (D.innerWaits x * D.innerCap x)) := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval, E, RP.eval]
  have key : ∀ n a b : ℕ, n ≤ a + b → n + 1 ≤ 1 + a + b := by omega
  refine key _ _ _ ?_
  split
  · simp [RP.eval]
  · split
    · rw [(RP.eval_bind H _ _ _).2, (RP.eval_bind H _ _ _).2, flatMapRP_runs, waitAccC_runs_eq]
      simp only [RP.eval, Nat.add_zero]
      refine Nat.add_le_add ((waitRuns_le_length _ _ _ _).trans_eq (by simp)) ?_
      refine (List.sum_le_length_nsmul _ (D.innerWaits x * D.innerCap x) fun n hn => ?_).trans ?_
      · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hn
        exact inner_runs_le_cap H D decode x _ _ _ _ _
      · rw [smul_eq_mul, List.length_map, List.length_zip, List.length_finRange]
        exact Nat.mul_le_mul_right _ (min_le_left _ _)
    · simp [RP.eval]

/-- `E`'s reruns at one judged run and coin: the replay, and if it accepted, the wait's and each accepted rerun's
session's. -/
theorem E_runs_pt (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : (E H D decode).C x) :
    ((E H D decode).eval H P x ω r₀ c).2.2 ≤ 1 + if (run H P D.V x ω r₀).out = .acc then
      waitRuns (run H P D.V x ω) (D.k x) ((List.finRange (D.cap x)).map c.1) 0 +
        (((List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).eval H
          (run H P D.V x ω)).1).map fun p => innerRuns (run H P D.V x ω) (D.innerK x) (D.innerCap x) p.2.1
            (c.2 p.1)).sum
      else 0 := by
  simp only [Extractor.eval, E, RP.eval]
  rw [Nat.add_comm]
  refine Nat.add_le_add_left ?_ 1
  split
  · simp [RP.eval]
  · by_cases h : (run H P D.V x ω r₀).out = .acc
    · have hs : (run H P D.V x ω r₀).seen.out = .acc := h
      simp only [hs, h, ↓reduceIte]
      rw [(RP.eval_bind H _ _ _).2, (RP.eval_bind H _ _ _).2, flatMapRP_runs, waitAccC_runs_eq, List.length_nil]
      simp only [RP.eval, Nat.add_zero]
      exact Nat.add_le_add_left (List.sum_le_sum fun p _ => inner_runs_le H D decode x _ _ _ _ _) _
    · have hs : ¬ (run H P D.V x ω r₀).seen.out = .acc := h
      simp [hs, h, RP.eval]

theorem Ex_listSum {α ι : Type} [Fintype α] (l : List ι) (f : ι → α → ℝ) :
    Ex (fun a => (l.map fun i => f i a).sum) = (l.map fun i => Ex (f i)).sum := by
  induction l with
  | nil => simp [Ex]
  | cons i l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [Ex_add (f i) (fun a => (l.map fun i => f i a).sum), ih]

/-- The coins off the draw coin. -/
def offDraw (n : ℕ) : Finset (Fin n) := Finset.univ.filter fun t => t.val ≠ 0

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem forkOn_offDraw {n : ℕ} (r e : Fin n → Coin) : forkOn (offDraw n) r e = atDraw r e := by
  funext t
  by_cases h : t.val = 0 <;> simp [forkOn, atDraw, offDraw, h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem atDraw_forkOn {n : ℕ} (r e : Fin n → Coin) : atDraw (forkOn (offDraw n) r e) = atDraw r := by
  funext ρ t
  by_cases h : t.val = 0 <;> simp [forkOn, atDraw, offDraw, h]

omit [Inhabited Coin] in
/-- **The sampled waits' expected reruns**: over the judged run's coins and the extractor's, at most
`1 + k·(1 + m′·k′)`. -/
theorem sampled_runs_exp (H : Msg → Msg) {n : ℕ} (run' : (Fin n → Coin) → Trace) (k K m' k' K' : ℕ) :
    Ex (fun s : (Fin n → Coin) × ((Fin K → Fin n → Coin) × (Fin k → Fin m' → Fin K' → Fin n → Coin)) =>
      1 + (if (run' s.1).out = .acc then (1 : ℝ) else 0) *
        ((waitRuns run' k ((List.finRange K).map s.2.1) 0 : ℝ) +
          (((List.finRange k).zip ((waitAccC k ((List.finRange K).map s.2.1) []).eval H run').1).map
            fun p => (innerRuns run' k' K' p.2.1 (s.2.2 p.1) : ℝ)).sum)) ≤ 1 + k * (1 + m' * k') := by
  let g : (Fin n → Coin) → ℝ := fun r =>
    Ex fun ρ : Fin m' → Fin K' → Fin n → Coin => (innerRuns run' k' K' r ρ : ℝ)
  have hg0 : ∀ r, 0 ≤ g r := fun r => Ex_nonneg fun _ => Nat.cast_nonneg _
  have hgW : ∀ r, g r = m' * waitExp (run' ∘ atDraw r) k' K' 0 := by
    intro r
    simp only [g, innerRuns, Nat.cast_sum]
    exact Ex_sum_pi (fun σ => (waitRuns (run' ∘ atDraw r) k' ((List.finRange K').map σ) 0 : ℝ)) m'
  have hgI : ∀ r e, g (forkOn (offDraw n) r e) = g r := by
    intro r e
    simp only [g, innerRuns, atDraw_forkOn]
  have hμ : Ex (fun d => (if (run' d).out = .acc then (1 : ℝ) else 0) * g d) ≤ m' * k' := by
    rw [Ex_acc_forkOn (offDraw n) (fun d => (run' d).out = .acc) g hgI]
    refine (Ex_mono fun r => ?_).trans_eq (Ex_const (α := Fin n → Coin) _)
    have hw := waitRuns_exp (run' ∘ atDraw r) k' K' 0 (Nat.zero_le _)
    simp only [Nat.cast_zero, sub_zero, Function.comp_apply] at hw
    simp only [forkOn_offDraw]
    rw [hgW r]
    calc Pr (fun e => (run' (atDraw r e)).out = .acc) * (m' * waitExp (run' ∘ atDraw r) k' K' 0)
        = m' * (Pr (fun e => (run' (atDraw r e)).out = .acc) * waitExp (run' ∘ atDraw r) k' K' 0) := by ring
      _ ≤ m' * k' := mul_le_mul_of_nonneg_left hw (Nat.cast_nonneg _)
  have hS : ∀ c1 : Fin K → Fin n → Coin,
      Ex (fun c2 : Fin k → Fin m' → Fin K' → Fin n → Coin =>
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map c1) []).eval H run').1).map
          fun p => (innerRuns run' k' K' p.2.1 (c2 p.1) : ℝ)).sum) =
        accSum run' k g ((List.finRange K).map c1) 0 := by
    intro c1
    rw [Ex_listSum]
    have hpi : ∀ p : Fin k × ((Fin n → Coin) × Seen),
        Ex (fun c2 : Fin k → Fin m' → Fin K' → Fin n → Coin => (innerRuns run' k' K' p.2.1 (c2 p.1) : ℝ)) =
          g p.2.1 := fun p => Ex_pi_eval p.1 fun ρ => (innerRuns run' k' K' p.2.1 ρ : ℝ)
    simp only [hpi]
    have hlen := waitAccC_length_le H run' k ((List.finRange K).map c1) [] (Nat.zero_le _)
    have hz := List.map_snd_zip (l₁ := List.finRange k) (l₂ := ((waitAccC k ((List.finRange K).map c1) []).eval H run').1)
      (by rw [List.length_finRange]; exact hlen)
    have := waitAccC_sum H run' k g ((List.finRange K).map c1) []
    simp only [List.map_nil, List.sum_nil, zero_add, List.length_nil] at this
    rw [← this]
    conv_rhs => rw [← hz]
    rw [List.map_map]
    rfl
  have hY : Ex (fun c : (Fin K → Fin n → Coin) × (Fin k → Fin m' → Fin K' → Fin n → Coin) =>
      (waitRuns run' k ((List.finRange K).map c.1) 0 : ℝ) +
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map c.1) []).eval H run').1).map
          fun p => (innerRuns run' k' K' p.2.1 (c.2 p.1) : ℝ)).sum) =
      waitExp run' k K 0 + accExp run' k g K 0 := by
    rw [Ex_prod]
    simp only [Ex_add, Ex_const, hS]
    rfl
  have hW := waitRuns_exp run' k K 0 (Nat.zero_le _)
  simp only [Nat.cast_zero, sub_zero] at hW
  have hA := accExp_le run' k g hg0 K 0 (Nat.zero_le _)
  simp only [Nat.cast_zero, sub_zero] at hA
  rw [Ex_add, Ex_const, Ex_prod]
  have hin : ∀ r : Fin n → Coin, (Ex fun c : (Fin K → Fin n → Coin) × (Fin k → Fin m' → Fin K' → Fin n → Coin) =>
      (if (run' r).out = .acc then (1 : ℝ) else 0) *
        ((waitRuns run' k ((List.finRange K).map c.1) 0 : ℝ) +
          (((List.finRange k).zip ((waitAccC k ((List.finRange K).map c.1) []).eval H run').1).map
            fun p => (innerRuns run' k' K' p.2.1 (c.2 p.1) : ℝ)).sum)) =
      (if (run' r).out = .acc then (1 : ℝ) else 0) * (waitExp run' k K 0 + accExp run' k g K 0) := by
    intro r
    rw [Ex_mul_left]
    exact congrArg _ hY
  rw [congrArg Ex (funext hin), Ex_mul_right, Ex_ind]
  have hA' := hA.trans (mul_le_mul_of_nonneg_left hμ (Nat.cast_nonneg k))
  have hexp : (Pr fun d : Fin n → Coin => (run' d).out = .acc) * (waitExp run' k K 0 + accExp run' k g K 0) =
      (Pr fun d : Fin n → Coin => (run' d).out = .acc) * waitExp run' k K 0 +
        (Pr fun d : Fin n → Coin => (run' d).out = .acc) * accExp run' k g K 0 := by ring
  rw [hexp]
  nlinarith

/-- **`E` reruns the prover at most `1 + k·(1 + m′·k′)` times in expectation**, at every prover and tape. -/
theorem E_runs_exp (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    ExtractRunsExpLe H (E H D decode) (fun x => 1 + D.k x * (1 + D.innerWaits x * D.innerK x)) := by
  intro x Ω P ω
  refine (Ex_mono fun s => ?_).trans (sampled_runs_exp H (run H P D.V x ω) (D.k x) (D.cap x) (D.innerWaits x)
    (D.innerK x) (D.innerCap x))
  have h := E_runs_pt H D decode x P ω s.1 s.2
  split_ifs at h with hacc
  · simp only [hacc, ↓reduceIte, one_mul]
    refine (Nat.cast_le.2 h).trans_eq ?_
    simp only [Nat.cast_add, Nat.cast_one, Nat.cast_list_sum, List.map_map]
    rfl
  · simp only [hacc, ↓reduceIte, zero_mul, add_zero]
    exact_mod_cast h

/-! ## The extractor's own calls -/

/-- **One rerun's openings checks make at most `openCalls(x)` SHA-512 calls**: what `Deploy.openCalls` is described as. -/
def OpenCallsLe (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (run : (Fin (D.V.k x) → Coin) → Trace) (s : Stmt) (v : Seen) (w : Flock.Wit),
    (ownLog H run (checkOpens D s (D.openings s v w))).length ≤ D.openCalls x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem inner_own_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (r : Fin (D.V.k x) → Coin) (v : Seen)
    (ρ : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin) (run : (Fin (D.V.k x) → Coin) → Trace)
    (h : OpenCallsLe H D) : (ownLog H run (inner H D decode x s r v ρ)).length ≤ D.openCalls x := by
  unfold inner
  split
  · simp [ownLog]
  · rw [ownLog_bind, ownLog_of_noHash H run (NoHash.firstWait _ _ _ _ _), List.nil_append]
    split
    · simp [ownLog]
    · exact h x run s v _

/-- **`E` makes at most `k·openCalls` SHA-512 calls of its own**, under `OpenCallsLe`. -/
theorem E_own_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (h : OpenCallsLe H D) :
    ExtractOwnLe H (E H D decode) (fun x => D.k x * D.openCalls x) := by
  intro x Ω P ω r₀ c
  simp only [E, ownLog]
  split
  · simp [ownLog]
  · split
    · rw [ownLog_bind, ownLog_bind, ownLog_of_noHash H _ (NoHash.waitAccC _ _ _), List.nil_append,
        List.length_append, ownLog_flatMapRP]
      simp only [ownLog, List.length_nil, Nat.add_zero]
      refine (List.sum_le_length_nsmul _ (D.openCalls x) fun n hn => ?_).trans ?_
      · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hn
        exact inner_own_le H D decode x _ _ _ _ _ h
      · rw [smul_eq_mul, List.length_map, List.length_zip, List.length_finRange]
        exact Nat.mul_le_mul_right _ (min_le_left _ _)
    · simp [ownLog]

/-- **Sampling: efficiency rests on the decoder**, once one rerun's openings checks are bounded (`OpenCallsLe`): a
decoder that meets Flock's list-decoding spec, with its stated complexity. -/
theorem Efficient_of_decoder [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin Col T)
    (decode : List (ℕ × Col) → List T)
    (counted : List (ℕ × Col) → List T × ℕ) (size : List (ℕ × Col) → ℕ) (p : Polynomial ℕ) (h : OpenCallsLe H D)
    (hl : Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode)
    (hd : DecoderComplexity decode counted size p) : Efficient H D decode counted size p :=
  ⟨E_runs_le H D decode, E_runs_exp H D decode, E_own_le H D decode h, hl, hd⟩

end Cost

end Claims.Sampling
