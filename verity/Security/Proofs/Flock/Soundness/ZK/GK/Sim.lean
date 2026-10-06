import Proofs.Flock.Soundness.ZK.GK.Rand
import Proofs.Flock.Soundness.ZK.GK.Extract

/-!
# Zero knowledge: the Goldreich–Kahan simulator for C-Flock ZK

The simulator of `backends/flock/live/PROTOCOL.md` §9 (`gk_simulate` in `backends/flock/live/src/bin/flock-circuit.rs`),
as a finite uniform program (`Rand`). It plays one dummy run, the protocol with a dummy witness, against the verifier `V*`.
If `V*` does not complete it (it aborts, or its coin openings fail), that view is the output. Otherwise:

* **estimation** (`estimate`): fresh dummy runs until `t` of them complete, at most `M` runs; `N` is the number of runs,
  so `p̃ = t/N` estimates the completion rate. Reaching the cap first is a failure (`Out.fail`).
* **rewinding** (`rewinds`): at most `4N = 4t/p̃` runs from the coins `V*` opened in the first run (`rw`, which draws fresh
  dummy openings for those coins), until one completes, whose view is the output. A rewound run in which `V*` opens a coin
  differently from the first run (`bad`) is a break of the coin commitment, `Out.bind`; running out of rewinds is
  `Out.fail`.

`gk` counts the runs of `V*` it plays alongside its outcome. This file proves the loops' laws that the hybrid argument
(`GK/Theorem.lean`) reads: `estimate_runs` (`q·E[N] ≤ t` at completion rate `q`), `estimate_cap` (`M·Pr[cap] ≤ E[N]`),
`estimate_abort` (`E[(1 − p)^{4N}] ≤ 3^{-t}` once `(1 − p)^4·(1 + 2q) ≤ 1`), `rewinds_close` (one run's law changed by
`δ` changes `n` rewinds' by at most `n·δ`), `rewinds_couple` (two loops that agree until `bad` differ by at most the
probability of `bind`) and `ev_rewinds_real` (rewinding with real runs reaches a completed view with probability
`1 − (1 − p)^n`).
-/

namespace FlockSoundness.ZK.GK

open Finset Game

/-- The simulator's outcome: a view, a break of the coin commitment between the first run's view and a rewound one, or a
failure (the estimation's cap or the rewinds ran out). -/
inductive Out (V : Type) : Type where
  | view : V → Out V
  | bind : V → V → Out V
  | fail : Out V

namespace Out

variable {V : Type}

/-- A test's value at an outcome: `φ` at a view, `0` at `bind` and `fail`. -/
def val (φ : V → ℝ) : Out V → ℝ
  | view v => φ v
  | _ => 0

/-- The outcome is `bind` at a pair of views `bad` flags. -/
def bindBy (bad : V → V → Bool) : Out V → Prop
  | bind v₀ v => bad v₀ v = true
  | _ => False

/-- The finder's output at an outcome: `pick` at `bind`, nothing otherwise. -/
def pickBy (pick : V → V → Pair) : Out V → Pair
  | bind v₀ v => pick v₀ v
  | _ => none

theorem val_nonneg {φ : V → ℝ} (h0 : ∀ v, 0 ≤ φ v) : ∀ o : Out V, 0 ≤ val φ o
  | view v => h0 v
  | bind _ _ => le_rfl
  | fail => le_rfl

theorem val_le_one {φ : V → ℝ} (h1 : ∀ v, φ v ≤ 1) : ∀ o : Out V, val φ o ≤ 1
  | view v => h1 v
  | bind _ _ => zero_le_one
  | fail => zero_le_one

end Out

variable {V : Type}

/-- **Estimation**: fresh runs (coin `C`, completed when `comp`) until `k` complete, at most `b` runs. The result is the
number of runs and whether `k` completed within the cap. -/
def estimate (C : Type) [Fintype C] [Nonempty C] (comp : C → Bool) : ℕ → ℕ → Rand (ℕ × Bool)
  | _, 0 => .ret (0, true)
  | 0, _ + 1 => .ret (0, false)
  | b + 1, k + 1 => .coin C fun ω => (estimate C comp b (if comp ω then k else k + 1)).map fun r => (r.1 + 1, r.2)

/-- **Rewinding**: at most `n` runs `rw σ` on fresh coins `σ`. A run `bad` flags is `bind` against the first view `v₀`;
the first completed run is the output; none is `fail`. The count is the number of runs played. -/
def rewinds (C : Type) [Fintype C] [Nonempty C] (v₀ : V) (rw : C → V) (comp bad : V → Bool) : ℕ → Rand (Out V × ℕ)
  | 0 => .ret (.fail, 0)
  | n + 1 => .coin C fun σ =>
      if bad (rw σ) then .ret (.bind v₀ (rw σ), 1)
      else if comp (rw σ) then .ret (.view (rw σ), 1)
      else (rewinds C v₀ rw comp bad n).map fun o => (o.1, o.2 + 1)

/-- **The Goldreich–Kahan simulator**, with completion threshold `t` and estimation cap `M`: one dummy run `dummy ω₀`;
if it completes, the estimation on fresh dummy runs, then at most `4N` rewinds `rw (dummy ω₀)` from it. It reads `V*`
only through its runs: `dummy`, `rw`, `comp` and `bad` are functions of `V*`'s views and of the simulator's own coins,
with no witness. -/
def gk (Cd Cr : Type) [Fintype Cd] [Nonempty Cd] [Fintype Cr] [Nonempty Cr] (dummy : Cd → V) (rw : V → Cr → V)
    (comp : V → Bool) (bad : V → V → Bool) (t M : ℕ) : Rand (Out V × ℕ) :=
  .coin Cd fun ω₀ =>
    if comp (dummy ω₀) then
      (estimate Cd (fun ω => comp (dummy ω)) M t).bind fun r =>
        if r.2 then
          (rewinds Cr (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * r.1)).map fun o => (o.1, o.2 + 1 + r.1)
        else .ret (.fail, 1 + r.1)
    else .ret (.view (dummy ω₀), 1)

/-! ## Unfolding -/

section Unfold

variable (C : Type) [Fintype C] [Nonempty C]

theorem ev_estimate_succ (comp : C → Bool) (f : ℕ × Bool → ℝ) (b k : ℕ) :
    Rand.ev f (estimate C comp (b + 1) (k + 1)) =
      avgR fun ω => Rand.ev (fun r => f (r.1 + 1, r.2)) (estimate C comp b (if comp ω then k else k + 1)) := by
  rw [estimate, Rand.ev_coin]
  simp only [Rand.ev_map]

theorem ev_rewinds_succ (v₀ : V) (rw : C → V) (comp bad : V → Bool) (f : Out V × ℕ → ℝ) (n : ℕ) :
    Rand.ev f (rewinds C v₀ rw comp bad (n + 1)) =
      avgR fun σ => if bad (rw σ) = true then f (.bind v₀ (rw σ), 1)
        else if comp (rw σ) = true then f (.view (rw σ), 1)
        else Rand.ev (fun o => f (o.1, o.2 + 1)) (rewinds C v₀ rw comp bad n) := by
  rw [rewinds, Rand.ev_coin]
  congr 1
  funext σ
  split_ifs
  · rfl
  · rfl
  · exact Rand.ev_map _ _ _

theorem ev_gk (Cr : Type) [Fintype Cr] [Nonempty Cr] (dummy : C → V) (rw : V → Cr → V) (comp : V → Bool)
    (bad : V → V → Bool) (t M : ℕ) (f : Out V × ℕ → ℝ) :
    Rand.ev f (gk C Cr dummy rw comp bad t M) =
      avgR fun ω₀ => if comp (dummy ω₀) = true then
        Rand.ev (fun r => if r.2 = true then
            Rand.ev (fun o => f (o.1, o.2 + 1 + r.1)) (rewinds Cr (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * r.1))
          else f (.fail, 1 + r.1)) (estimate C (fun ω => comp (dummy ω)) M t)
      else f (.view (dummy ω₀), 1) := by
  rw [gk, Rand.ev_coin]
  congr 1
  funext ω₀
  split_ifs
  · rw [Rand.ev_bind]
    congr 1
    funext r
    split_ifs
    · exact Rand.ev_map _ _ _
    · rfl
  · rfl

end Unfold

/-! ## The estimation's laws -/

section Estimate

variable (C : Type) [Fintype C] [Nonempty C] (comp : C → Bool)

/-- The completion rate of one run. -/
noncomputable def rate : ℝ := avgR (ind fun ω => comp ω = true)

omit [Nonempty C] in
theorem rate_nonneg : 0 ≤ rate C comp := avgR_nonneg fun ω => ind_nonneg _ ω

theorem rate_le_one : rate C comp ≤ 1 := avgR_le_one fun ω => by unfold ind; split_ifs <;> norm_num

/-- Splitting one run on whether it completes. -/
theorem avgR_split (a b : ℝ) : (avgR fun ω => if comp ω = true then a else b) = rate C comp * a + (1 - rate C comp) * b :=
  avgR_ite _ a b

/-- The estimation stays within its cap, and fails only at the cap. -/
theorem estimate_all : ∀ b k : ℕ, Rand.All (fun r : ℕ × Bool => r.1 ≤ b ∧ (r.2 = false → r.1 = b)) (estimate C comp b k)
  | _, 0 => by simp [estimate, Rand.All]
  | 0, k + 1 => by simp [estimate, Rand.All]
  | b + 1, k + 1 => by
      rw [estimate]
      intro ω
      refine Rand.All.map _ (fun r hr => ?_) _ (estimate_all b _)
      exact ⟨Nat.succ_le_succ hr.1, fun h => by rw [hr.2 h]⟩

/-- **`q·E[N] ≤ t`**: at completion rate `q`, the estimation's expected number of runs is at most `t/q`. -/
theorem estimate_runs : ∀ b k : ℕ, rate C comp * Rand.ev (fun r => (r.1 : ℝ)) (estimate C comp b k) ≤ k
  | _, 0 => by simp [estimate]
  | 0, k + 1 => by simp [estimate]; positivity
  | b + 1, k + 1 => by
      rw [ev_estimate_succ]
      have e : (fun ω => Rand.ev (fun r : ℕ × Bool => ((r.1 + 1 : ℕ) : ℝ)) (estimate C comp b (if comp ω then k else k + 1)))
          = fun ω => if comp ω = true then Rand.ev (fun r => (r.1 : ℝ)) (estimate C comp b k) + 1
            else Rand.ev (fun r => (r.1 : ℝ)) (estimate C comp b (k + 1)) + 1 := by
        funext ω
        split_ifs <;>
        · simp only [Nat.cast_add, Nat.cast_one]
          rw [Rand.ev_add, Rand.ev_const]
      rw [e, avgR_split]
      have h1 := estimate_runs b k
      have h2 := estimate_runs b (k + 1)
      have hq0 := rate_nonneg C comp
      have hq1 := rate_le_one C comp
      have a1 := mul_le_mul_of_nonneg_left h1 hq0
      have a2 := mul_le_mul_of_nonneg_left h2 (sub_nonneg.2 hq1)
      push_cast at h2 a2 ⊢
      nlinarith

/-- **`M·Pr[cap] ≤ E[N]`** (Markov): the estimation fails only after `M` runs. -/
theorem estimate_cap (b k : ℕ) :
    (b : ℝ) * Rand.ev (fun r => if r.2 = true then 0 else 1) (estimate C comp b k) ≤
      Rand.ev (fun r => (r.1 : ℝ)) (estimate C comp b k) := by
  rw [← Rand.ev_const_mul]
  refine Rand.ev_le_of_All (fun r hr => ?_) _ (estimate_all C comp b k)
  rcases r with ⟨n, _ | _⟩
  · have h2 : n = b := hr.2 rfl
    subst h2
    simp
  · simp

/-- **The rewinds' abort bound**: if `a·(1 + 2q) ≤ 1` at completion rate `q`, then `E[a^N]` over a successful estimation
is at most `3^{-k}`: the generating function of `k` geometric variables at `a`. -/
theorem estimate_abort {a : ℝ} (ha : 0 ≤ a) (hq : a * (1 + 2 * rate C comp) ≤ 1) :
    ∀ b k : ℕ, Rand.ev (fun r => if r.2 = true then a ^ r.1 else 0) (estimate C comp b k) ≤ (1 / 3) ^ k
  | _, 0 => by simp [estimate]
  | 0, k + 1 => by simp [estimate]
  | b + 1, k + 1 => by
      rw [ev_estimate_succ]
      have e : ∀ k', Rand.ev (fun r : ℕ × Bool => if r.2 = true then a ^ (r.1 + 1) else 0) (estimate C comp b k') =
          a * Rand.ev (fun r => if r.2 = true then a ^ r.1 else 0) (estimate C comp b k') := by
        intro k'
        rw [← Rand.ev_const_mul]
        congr 1
        funext r
        split_ifs <;> ring
      simp only [e]
      have hle : ∀ ω, a * Rand.ev (fun r : ℕ × Bool => if r.2 = true then a ^ r.1 else 0)
            (estimate C comp b (if comp ω then k else k + 1)) ≤
          if comp ω = true then a * (1 / 3) ^ k else a * (1 / 3) ^ (k + 1) := by
        intro ω
        split_ifs with h
        · exact mul_le_mul_of_nonneg_left (estimate_abort ha hq b k) ha
        · exact mul_le_mul_of_nonneg_left (estimate_abort ha hq b (k + 1)) ha
      refine (avgR_mono hle).trans ?_
      rw [avgR_split]
      have hx : (0 : ℝ) ≤ (1 / 3) ^ k := by positivity
      have := mul_le_mul_of_nonneg_right hq hx
      rw [pow_succ]
      nlinarith

end Estimate

/-! ## The rewinds' laws -/

section Rewinds

/-- The rewinds stay within their budget. -/
theorem rewinds_all (C : Type) [Fintype C] [Nonempty C] (v₀ : V) (rw : C → V) (comp bad : V → Bool) :
    ∀ n : ℕ, Rand.All (fun o : Out V × ℕ => o.2 ≤ n) (rewinds C v₀ rw comp bad n)
  | 0 => by simp [rewinds, Rand.All]
  | n + 1 => by
      rw [rewinds]
      intro σ
      dsimp only
      split_ifs
      · simp [Rand.All]
      · simp [Rand.All]
      · exact Rand.All.map (P := fun o : Out V × ℕ => o.2 ≤ n) (Q := fun o : Out V × ℕ => o.2 ≤ n + 1)
          (fun o => (o.1, o.2 + 1)) (fun o ho => Nat.succ_le_succ ho) _ (rewinds_all C v₀ rw comp bad n)

/-- **One run's law changed by `δ` changes `n` rewinds' by at most `n·δ`**, for every test with values in `[0, 1]`. -/
theorem rewinds_close {C₁ C₂ : Type} [Fintype C₁] [Nonempty C₁] [Fintype C₂] [Nonempty C₂] (v₀ : V) (rw₁ : C₁ → V)
    (rw₂ : C₂ → V) (comp bad : V → Bool) {δ : ℝ}
    (h : ∀ U : V → Prop, |avgR (fun σ => ind U (rw₁ σ)) - avgR (fun σ => ind U (rw₂ σ))| ≤ δ) :
    ∀ (n : ℕ) (f : Out V × ℕ → ℝ), (∀ x, 0 ≤ f x) → (∀ x, f x ≤ 1) →
      |Rand.ev f (rewinds C₁ v₀ rw₁ comp bad n) - Rand.ev f (rewinds C₂ v₀ rw₂ comp bad n)| ≤ n * δ
  | 0, f, _, _ => by simp [rewinds]
  | n + 1, f, h0, h1 => by
      rw [ev_rewinds_succ, ev_rewinds_succ]
      set c₁ := Rand.ev (fun o => f (o.1, o.2 + 1)) (rewinds C₁ v₀ rw₁ comp bad n)
      set c₂ := Rand.ev (fun o => f (o.1, o.2 + 1)) (rewinds C₂ v₀ rw₂ comp bad n)
      have ih : |c₁ - c₂| ≤ n * δ :=
        rewinds_close v₀ rw₁ rw₂ comp bad h n _ (fun x => h0 _) (fun x => h1 _)
      have hc0 : 0 ≤ c₁ := Rand.ev_nonneg (fun x => h0 _) _
      have hc1 : c₁ ≤ 1 := Rand.ev_le_one (fun x => h1 _) _
      let ψ : ℝ → V → ℝ := fun c v => if bad v = true then f (.bind v₀ v, 1)
        else if comp v = true then f (.view v, 1) else c
      have step₁ : |avgR (fun σ => ψ c₁ (rw₁ σ)) - avgR (fun σ => ψ c₁ (rw₂ σ))| ≤ δ :=
        abs_avgR_sub_le_of_events rw₁ rw₂ h (ψ c₁)
          (fun v => by simp only [ψ]; split_ifs <;> first | exact h0 _ | exact hc0)
          (fun v => by simp only [ψ]; split_ifs <;> first | exact h1 _ | exact hc1)
      have step₂ : |avgR (fun σ => ψ c₁ (rw₂ σ)) - avgR (fun σ => ψ c₂ (rw₂ σ))| ≤ n * δ := by
        refine (abs_avgR_sub_le (k := fun _ => |c₁ - c₂|) fun σ => ?_).trans ?_
        · simp only [ψ]
          split_ifs <;> simp
        · rw [avgR_const]
          exact ih
      show |avgR (fun σ => ψ c₁ (rw₁ σ)) - avgR (fun σ => ψ c₂ (rw₂ σ))| ≤ (↑(n + 1) : ℝ) * δ
      push_cast
      calc _ ≤ |avgR (fun σ => ψ c₁ (rw₁ σ)) - avgR (fun σ => ψ c₁ (rw₂ σ))| +
            |avgR (fun σ => ψ c₁ (rw₂ σ)) - avgR (fun σ => ψ c₂ (rw₂ σ))| := abs_sub_le _ _ _
        _ ≤ δ + n * δ := add_le_add step₁ step₂
        _ = (n + 1) * δ := by ring

/-- **Identical until bad**: two rewinding loops on the same coins whose runs agree whenever the first's isn't `bad`, the
second never checking, differ on every `[0, 1]` test by at most the probability of the first's `bind` (any event `B`
that holds at the first's `bind` outcomes). -/
theorem rewinds_couple (C : Type) [Fintype C] [Nonempty C] (v₀ : V) (rw₁ rw₂ : C → V) (comp bad : V → Bool)
    (B : Out V → Prop) (hB : ∀ v, bad v = true → B (.bind v₀ v)) (hc : ∀ σ, bad (rw₁ σ) = false → rw₁ σ = rw₂ σ) :
    ∀ (n : ℕ) (f : Out V × ℕ → ℝ), (∀ x, 0 ≤ f x) → (∀ x, f x ≤ 1) →
      |Rand.ev f (rewinds C v₀ rw₁ comp bad n) - Rand.ev f (rewinds C v₀ rw₂ comp (fun _ => false) n)| ≤
        Rand.ev (fun o => ind B o.1) (rewinds C v₀ rw₁ comp bad n)
  | 0, f, _, _ => by simp [rewinds, ind_nonneg]
  | n + 1, f, h0, h1 => by
      rw [ev_rewinds_succ, ev_rewinds_succ, ev_rewinds_succ]
      refine abs_avgR_sub_le fun σ => ?_
      have ih := rewinds_couple C v₀ rw₁ rw₂ comp bad B hB hc n (fun o => f (o.1, o.2 + 1)) (fun x => h0 _)
        (fun x => h1 _)
      by_cases hb : bad (rw₁ σ) = true
      · have hB' : ind B (Out.bind v₀ (rw₁ σ)) = 1 := by simp [ind, hB _ hb]
        simp only [hb, hB', Bool.false_eq_true, ↓reduceIte]
        have a0 := h0 (Out.bind v₀ (rw₁ σ), 1)
        have a1 := h1 (Out.bind v₀ (rw₁ σ), 1)
        have e0 := Rand.ev_nonneg (f := fun o => f (o.1, o.2 + 1)) (fun x => h0 _)
          (rewinds C v₀ rw₂ comp (fun _ => false) n)
        have e1 := Rand.ev_le_one (f := fun o => f (o.1, o.2 + 1)) (fun x => h1 _)
          (rewinds C v₀ rw₂ comp (fun _ => false) n)
        rw [abs_le]
        split_ifs <;> constructor <;> linarith [h0 (Out.view (rw₂ σ), 1), h1 (Out.view (rw₂ σ), 1)]
      · have hb' : bad (rw₁ σ) = false := by simpa using hb
        have he := hc σ hb'
        simp only [hb', Bool.false_eq_true, ↓reduceIte, ← he]
        split_ifs
        · simp [ind_nonneg]
        · exact ih

/-- **Rewinding with real runs**: if `S = p·A`, where `p` is the completion rate of `real` and `S` the average of `φ` on
its completed runs, then `n` rewinds output a completed view with value `A·(1 − (1 − p)^n)` in expectation. -/
theorem ev_rewinds_real (C : Type) [Fintype C] [Nonempty C] (v₀ : V) (real : C → V) (comp : V → Bool) (φ : V → ℝ)
    (A : ℝ) (hS : avgR (fun ω => if comp (real ω) = true then φ (real ω) else 0) = rate C (fun ω => comp (real ω)) * A) :
    ∀ n : ℕ, Rand.ev (fun o => Out.val φ o.1) (rewinds C v₀ real comp (fun _ => false) n) =
      A * (1 - (1 - rate C (fun ω => comp (real ω))) ^ n)
  | 0 => by simp [rewinds, Out.val]
  | n + 1 => by
      rw [ev_rewinds_succ]
      have ih := ev_rewinds_real C v₀ real comp φ A hS n
      simp only [Bool.false_eq_true, ↓reduceIte]
      have e : (fun σ => if comp (real σ) = true then Out.val φ (Out.view (real σ))
            else Rand.ev (fun o : Out V × ℕ => Out.val φ o.1) (rewinds C v₀ real comp (fun _ => false) n)) =
          fun σ => (if comp (real σ) = true then φ (real σ) else 0) +
            (if comp (real σ) = true then 0 else A * (1 - (1 - rate C (fun ω => comp (real ω))) ^ n)) := by
        funext σ
        split_ifs
        · simp [Out.val]
        · rw [ih]
          ring
      rw [e, avgR_add, hS, avgR_split C (fun ω => comp (real ω))]
      ring

end Rewinds

end FlockSoundness.ZK.GK
