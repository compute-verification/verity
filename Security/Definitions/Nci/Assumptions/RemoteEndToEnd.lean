import Definitions.Nci.Remote

/-!
# The remote protocol end to end: its named premises

`Nci.Guarantees.RemoteEndToEnd` carries `Nci.Guarantees.DampenerRemotePeriod` to a task. Past `DampenerRemotePeriod`'s
own premises it names three, each a `Prop` here:

* the two protocol bridges, `PouwPerWindow` ([POUW-PER-WINDOW]) and `SampledPerWindow` ([SAMPLED-PER-WINDOW]):
  `DampenerRemotePeriod`'s PoUW leg and its four sampled-proofs legs, as one `Prop` each. They are what PR #4653's
  guarantees conclude (`Accounting.Guarantees.Compose.CircuitCreditPerWindow` and the sampled-proofs bridge,
  `Flock.Guarantees.HiddenWindowPerWindow` there), each from a premise that the deployment's window is a play its
  protocol's guarantee covers. Until #4653 lands they are premises; once it does, each is discharged by its bridge and
  `RemoteEndToEnd` takes the bridge's premise instead;
* the task model, `Coverage` ([COVERAGE]): a run reaches the task with fewer than `N` counted multiplications with
  probability at most `η`. It is the remote form of the spine's `Coverage 𝒢 N η` (note
  training-20261027T0134Z-draft-e2e-overhead-proof, Step 6), and the overhead core's definition replaces it when it lands.
-/

namespace Nci.Assumptions

open Nci.Dampener

/-- **PoUW's certificates, window by window** ([POUW-PER-WINDOW]; the bridge
`Accounting.Guarantees.Compose.CircuitCreditPerWindow`, PR #4653). Over a monitoring period of `T` windows, window `t`'s
coins `Ω t` uniform and independent of the other windows', window `t`'s certificate is wrong (`wrong t`: accepted with
the window's time below `(1 − ε)·T_cred`) with probability at most `p` over its own coins (`ξ[t ↦ a]`, `a` uniform on
`Ω t`), at every value `ξ` of the others'.

It is `DampenerRemotePeriod`'s PoUW hypothesis. #4653 proves it when each window, at each value of the other windows'
coins, is a play `CircuitCredit` covers whose failure `wrong t` implies (`CreditPlayOn`), at
`p ≥ δ_s + δ_bind + δ_ks`: `3 · 2⁻⁴⁰` for a uniform `K_d`-subset draw, which is not yet the served window's (#4653's
`CreditPlayOn` docstring). Nothing in Lean builds that play from the deployment. -/
def PouwPerWindow {T : ℕ} (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p : ℝ) : Prop :=
  ∀ t (ξ : (t : Fin T) → Ω t), prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p

/-- **Sampled proofs, window by window** ([SAMPLED-PER-WINDOW]; the bridge `Flock.Guarantees.HiddenWindowPerWindow`,
PR #4653, being renamed there). Over a monitoring period of `T` windows, window `t`'s coins `Ω t` uniform and
independent of the others', the developer's faulty proof blocks `bad ξ`, block `b` in window `win b`, the units each
window draws (`drawn`) and the faulty blocks it detects (`detected`):

* the faulty blocks of window `t` are fixed by the windows before it;
* its drawn and detected blocks are fixed by the windows up to it;
* its draw misses its `m` faulty blocks with probability at most `(1 − q)^m`;
* it detects none of them while it draws one with probability at most `ks t`;

each bound over window `t`'s own coins (`ξ[t ↦ a]`, `a` uniform on `Ω t`) at every value `ξ` of the others'.

It is `DampenerRemotePeriod`'s four sampled-proofs hypotheses, in their order. #4653 proves it when every window that
holds a block is a `HiddenWindow` played on its coins, chosen from the earlier windows' coins alone, under A3 and
`cr/sha-512`, with `q ≤ k_d/n` and `ε_ks + δ_link ≤ ks t`. Nothing in Lean ties the sampled-proofs service's window to
such a play. -/
def SampledPerWindow {T : ℕ} (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)] {β : Type}
    (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T) (q : ℝ) (ks : Fin T → ℝ) : Prop :=
  (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) ∧
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
          ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t)

/-- **The task needs `N` counted multiplications** ([COVERAGE]; the task model, the spine's `Coverage 𝒢 N η`). Over a
run's outcomes `Ω`, drawn from `μ`, a run that reaches the task (`success`) does so with fewer than `N` counted
multiplications (`M`) with probability at most `η`: reaching the task takes `N` of them but on an exception of
probability `η`.

It is a premise about the task, not about any protocol: what `N` and `η` are for a task (a target loss on a model
family, a benchmark score) is a claim about training, falsified by a run that reaches the task with fewer. It reads
the run's own count `M`, which the remote dampener bounds, so the guarantee that takes it turns a bound on the compute
efficiency into one on the hardware needed to reach the task. -/
def Coverage {Ω : Type} [Fintype Ω] (μ : PMF Ω) (success : Ω → Prop) (M : Ω → ℕ) (N : ℕ) (η : ℝ) : Prop :=
  prob μ (fun ω => success ω ∧ M ω < N) ≤ η

end Nci.Assumptions
