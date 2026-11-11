module

public import Proofs.Flock.SamplingGeneric.RegAt
public import Proofs.Flock.SamplingGeneric.RegX
public import Proofs.Flock.SamplingGeneric.TwoKSAt
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext

/-!
# `B_conf,X`'s cost at a budget on a run's calls

`B_conf,X` (`Compiled.BconfX`) makes two runs, each at its own tape, and extracts every session of each. Its cost is
what those log: both runs' SHA-512 calls, both parties', and both extractions', their reruns' included. Where every run
of the prover logs at most `T` calls, at the contract's cost facts and the opening checks' count:

* its expected cost is at most `c_T = 2T·(1 + n_sess·e_s) + 2·n_PU·openCalls` (`Compiled.BconfX_cost_le`): `T` for each
  of its two runs, and for each run's extraction its own calls, at most `n_PU·openCalls` (`runExt_own_le`), and `T`
  for each of its reruns (`Sampling.calls_le_own`), at most `n_sess·e_s` expected over the run's coins and the
  extractor's at every tape (`runExt_runs_exp`), the two runs' tapes and coins independent (`Ex_two_tapes`);
* on every coin it is at most `2T·(1 + n_sess·ρ_s) + 2·n_PU·openCalls` (`Compiled.BconfX_cost_strict`,
  `runExt_runs_le`).

So where `ExpectedCR` holds at `B_conf,X`, `Pr[B_conf,X] ≤ c_T/2^256` (`Compiled.pr_BconfX_le`), at a two-stage
verifier too, whose contract gives the cost facts (`CompiledTwo.pr_BconfX_le`), and the event of `KSRegTwoX` with the
by-position finders' terms has that for its `Pr[B_conf,X]` (`CompiledTwo.eventTwoX_le_firstHit_at_budget`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo ownLog)

/-- Two runs, each at its own tape and on its own coins: a cost `c` plus each run's, whose expectation over its coins
is at most `b` at every tape, is at most `c + 2b` in expectation. -/
theorem Ex_two_tapes {Ω X : Type} [Fintype Ω] [Nonempty Ω] [Fintype X] [Nonempty X] (G : Ω → X → ℝ) (c b : ℝ)
    (hG : ∀ ω, Ex (G ω) ≤ b) :
    Ex (fun q : (Ω × X) × (Ω × X) => c + G q.1.1 q.1.2 + G q.2.1 q.2.2) ≤ c + 2 * b := by
  have hb : Ex (fun p : Ω × X => G p.1 p.2) ≤ b := by
    rw [Ex_prod]
    exact (Ex_mono hG).trans_eq (Ex_const b)
  have h1 : Ex (fun q : (Ω × X) × (Ω × X) => G q.1.1 q.1.2) ≤ b :=
    ((Ex_prod fun q : (Ω × X) × (Ω × X) => G q.1.1 q.1.2).trans
      (congrArg Ex (funext fun a => Ex_const (α := Ω × X) (G a.1 a.2)))).trans_le hb
  have h2 : Ex (fun q : (Ω × X) × (Ω × X) => G q.2.1 q.2.2) ≤ b :=
    ((Ex_prod fun q : (Ω × X) × (Ω × X) => G q.2.1 q.2.2).trans
      (Ex_const (α := Ω × X) (Ex fun p : Ω × X => G p.1 p.2))).trans_le hb
  rw [Ex_add, Ex_add, Ex_const]
  linarith

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`B_conf,X`'s expected cost at a budget on a run's calls**: at most `c_T` (`bconfCost`) at the contract's cost
facts and the opening checks' count, where every run of the prover logs at most `T` SHA-512 calls, both parties'. Each
of its two runs logs at most `T`, and each run's extraction calls at most its own calls and `T` a rerun
(`Sampling.calls_le_own`), which is its cost at `T` (`runCostAt_exp`), over its own tape and coins (`Ex_two_tapes`). -/
theorem BconfX_cost_le {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) :
    Ex (fun q => ((BconfX H L Ω P x).cost q : ℝ)) ≤ L.bconfCost x T := by
  delta BconfX
  rcases hs : L.parse x with _ | s
  · show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) ≤ _
    rw [Nat.cast_zero, Ex_const]
    exact L.bconfCost_nonneg x T
  · refine (Ex_mono (g := fun q : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
          (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        (2 * T : ℝ) + L.runCostAt H T x s P q.1.1 q.1.2 + L.runCostAt H T x s P q.2.1 q.2.2) fun q => ?_).trans
      ((Ex_two_tapes (fun ω rc => L.runCostAt H T x s P ω rc) _ _ fun ω =>
        runCostAt_exp hS hopen T hs P ω).trans_eq ?_)
    · have h₁ := (Nat.cast_le (α := ℝ)).2 (Sampling.calls_le_own H (run H P L.V x q.1.1) T (hT q.1.1)
        (L.runExt x s q.1.2.1 (run H P L.V x q.1.1 q.1.2.1).seen q.1.2.2))
      have h₂ := (Nat.cast_le (α := ℝ)).2 (Sampling.calls_le_own H (run H P L.V x q.2.1) T (hT q.2.1)
        (L.runExt x s q.2.2.1 (run H P L.V x q.2.1 q.2.2.1).seen q.2.2.2))
      have h₃ := (Nat.cast_le (α := ℝ)).2 (hT q.1.1 q.1.2.1)
      have h₄ := (Nat.cast_le (α := ℝ)).2 (hT q.2.1 q.2.2.1)
      simp only [runCostAt]
      push_cast at h₁ h₂ h₃ h₄ ⊢
      linarith
    · unfold bconfCost
      ring

/-- **`B_conf,X`'s cost on every coin, at a budget on a run's calls**: at most
`2T·(1 + n_sess·ρ_s) + 2·n_PU·openCalls`, each run logging at most `T`, and each run's extraction rerunning at most
`n_sess·ρ_s` times (`runExt_runs_le`), `T` a rerun, and calling at most `n_PU·openCalls` times itself
(`runExt_own_le`). -/
theorem BconfX_cost_strict {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) :
    ∀ q, (BconfX H L Ω P x).cost q ≤ 2 * T * (1 + L.nSess x * L.ρs x) + 2 * (L.nPU x * L.openCalls x) := by
  delta BconfX
  rcases hs : L.parse x with _ | s
  · exact fun _ => Nat.zero_le _
  · rintro ⟨⟨ω₁, r₁, c₁⟩, ⟨ω₂, r₂, c₂⟩⟩
    dsimp only
    have h₁ := Nat.mul_le_mul_left T (runExt_runs_le hS P ω₁ s r₁ c₁)
    have h₂ := Nat.mul_le_mul_left T (runExt_runs_le hS P ω₂ s r₂ c₂)
    have h₃ := runExt_own_le hS hopen hs P ω₁ r₁ c₁
    have h₄ := runExt_own_le hS hopen hs P ω₂ r₂ c₂
    have h₅ := Sampling.calls_le_own H (run H P L.V x ω₁) T (hT ω₁)
      (L.runExt x s r₁ (run H P L.V x ω₁ r₁).seen c₁)
    have h₆ := Sampling.calls_le_own H (run H P L.V x ω₂) T (hT ω₂)
      (L.runExt x s r₂ (run H P L.V x ω₂ r₂).seen c₂)
    have h₇ := hT ω₁ r₁
    have h₈ := hT ω₂ r₂
    have e : T + T + (L.nPU x * L.openCalls x + T * (L.nSess x * L.ρs x)) +
        (L.nPU x * L.openCalls x + T * (L.nSess x * L.ρs x)) =
        2 * T * (1 + L.nSess x * L.ρs x) + 2 * (L.nPU x * L.openCalls x) := by ring
    rw [← e]
    omega

/-- **`Pr[B_conf,X]` at a budget on a run's calls**: where `ExpectedCR` holds at `B_conf,X` and every run of the prover
logs at most `T` SHA-512 calls, `Pr[B_conf,X] ≤ c_T/2^256` (`BconfX_cost_le`). -/
theorem pr_BconfX_le {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T)
    (hE : ExpectedCR H (BconfX H L Ω P x)) :
    (BconfX H L Ω P x).Pr H ≤ L.bconfCost x T / 2 ^ (256 : ℝ) := by
  unfold ExpectedCR at hE
  exact hE.trans (div_le_div_of_nonneg_right (BconfX_cost_le hS hopen T Ω P x hT) (by positivity))

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`Pr[B_conf,X]` at a budget on a run's calls, at two stages**: `Compiled.pr_BconfX_le` at the two-stage
verifier, whose contract gives the cost facts (`Sessions.toCostContract`). -/
theorem pr_BconfX_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T)
    (hE : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x)) :
    (Compiled.BconfX H L.toCompiled Ω P x).Pr H ≤ L.bconfCost x T / 2 ^ (256 : ℝ) :=
  Compiled.pr_BconfX_le hS.toCostContract hopen T Ω P x hT hE

/-- **The event of `KSRegTwoX` with the by-position finders' terms and `Pr[B_conf,X]` at a budget on a run's calls**:
`eventTwoX_le_firstHit_at` where `ExpectedCR` holds at `B_conf,X` too and every run of the prover logs at most `T`
SHA-512 calls (`pr_BconfX_le`): at most
`κ_s + √(k·γ_s·Pr[B]) + c_T/2^256 + 4·E[c + Σ_{g ∈ sample} c_g]/2^256`. -/
theorem eventTwoX_le_firstHit_at_budget {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXAt H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderIAt H P x s d.part needs K)) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + L.bconfCost x T / 2 ^ (256 : ℝ) +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_firstHit_at hA hbind hS hs hPt hd hin P needs hpos hX hI
  have hb := pr_BconfX_le hS hopen T Ω P x hT hB
  linarith

end CompiledTwo

end Claims.SamplingGeneric
