import Definitions.Accounting.Confidence
import Definitions.Pouw.Game.Deadline

/-!
# The cost model of `A`'s provers (trusted)

Section 8 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.4), B1: PoUW's game (`Pouw.CostModel`) at
the provers of actual circuits. The verifier sits at the leaves `V`, releases the salt at `tₛ` and decodes the
transcript (`read`, the protocol's) from what it receives from outside before the deadline `t_d`; `ℓ` is the latency
between it and the prover. A program (`Actual.Prover`) is a strategy over the online phase's inputs (oracle,
workload, activation input, preprocessed state, salt), with runs in the feasible set `𝓕` and causal from every tick;
what the prover (off `V`) can tell when the salt reaches it, at `tₛ + ℓ`, doesn't depend on the salt (preprocessing is
everything before `tₛ + ℓ`, and it hands on what the prover holds then); and its oracle calls are the hash's gates
(`hash`). Its cost is the prover's work at `π` in `[tₛ + ℓ, t_d − ℓ)` (`Actual.costModel`, `CM_A`).
`Actual.timedRun` is an adversary's timed run: the salt released at `tₛ`, the transcript received at `t_d`, and by
each time `t` the prover's work placed before `t` spent.

Runs don't confine the oracle to the hash's gates (a gate's operation is any function of what it reads, and a strategy
any map from the oracle to runs), so a program carries red-team B6's form as a field (`queries`): on the other inputs
fixed, what the verifier receives before `t_d` is an oracle program's output and the hash's gates are its queries.
-/

namespace Accounting

/-- The online phase's inputs: the oracle, the workload, the activation input, the preprocessed state and the salt. -/
abbrev Actual.Inputs (Q R S : Type) : Type := (Q → R) × Pouw.Workload × (ℕ → Pouw.Mat) × Pouw.State × S

open Classical in
/-- The prover's work at `π` during `I`: the work of the computed gates placed off `V` during `I`. -/
noncomputable def Actual.Run.workIn {L : Type*} (ρ : Actual.Run L) (π : Fin ρ.A.N → ℕ) (V : Finset L) (I : Set ℕ) :
    ℕ :=
  ρ.A.work π (ρ.A.computed.filter fun g => ρ.A.x g ∉ V ∧ ρ.A.t g ∈ I)

/-- **A prover** (v2.4, B1): a strategy over the online phase's inputs with runs in `𝓕`, causal from every tick, such
that what the prover can tell at `tₛ + ℓ` (`history`) doesn't depend on the salt, and, on the other inputs fixed, what
the verifier receives before `t_d`, decoded by `read`, and the number of the hash's gates are an oracle program's
output and queries. -/
structure Actual.Prover (Q R S : Type) {L : Type} [LinearOrder L] [Fintype L] (𝓕 : Set (Actual.Run L))
    (V : Finset L) (tₛ t_d ℓ : ℕ) (read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript)
    (hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)) where
  strategy : Actual.Inputs Q R S → Actual.Run L
  feasible : ∀ x, strategy x ∈ 𝓕
  causal : CausalFrom strategy
  fresh : ∀ H U act σ s s', (strategy (H, U, act, σ, s)).history Vᶜ (tₛ + ℓ) =
    (strategy (H, U, act, σ, s')).history Vᶜ (tₛ + ℓ)
  queries : ∀ U act σ s, ∃ t : Pouw.QTree Q R Pouw.Transcript, ∀ H,
    read ((strategy (H, U, act, σ, s)).crossing (↑V)ᶜ ↑V (Set.Iio t_d)) = t.eval H ∧
      (hash (strategy (H, U, act, σ, s))).card = t.calls H

/-- **`CM_A`, the cost model of `A`'s provers** (v2.4, B1): its programs are the provers; a program outputs the
transcript the verifier decodes from what it receives before `t_d`, costs the prover's work at `π` in
`[tₛ + ℓ, t_d − ℓ)` and makes as many oracle calls as its run has hash's gates. -/
noncomputable def Actual.costModel (Q R S : Type) {L : Type} [LinearOrder L] [Fintype L] (𝓕 : Set (Actual.Run L))
    (V : Finset L) (tₛ t_d ℓ : ℕ) (read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript)
    (hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)) (π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ) :
    Pouw.CostModel Q R S where
  Prog := Actual.Prover Q R S 𝓕 V tₛ t_d ℓ read hash
  run P H U act σ s := read ((P.strategy (H, U, act, σ, s)).crossing (↑V)ᶜ ↑V (Set.Iio t_d))
  cost P H U act σ s := ((P.strategy (H, U, act, σ, s)).workIn (π _) V (Set.Ico (tₛ + ℓ) (t_d - ℓ)) : ℝ)
  calls P H U act σ s := (hash (P.strategy (H, U, act, σ, s))).card

/-- **An adversary's timed run** in `CM_A` on the workload `U`: the salt is released at `tₛ`, the transcript is
received at `t_d` (it is what the verifier receives before then), and by each time `t` the prover has spent its work
at `π` placed before `t`. -/
noncomputable def Actual.timedRun {Q R S : Type} {L : Type} [LinearOrder L] [Fintype L] {𝓕 : Set (Actual.Run L)}
    {V : Finset L} {tₛ t_d ℓ : ℕ} {read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript}
    {hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)} {π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ}
    (A : Pouw.Adversary (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π)) (U : Pouw.Workload) :
    Pouw.TimedRun Q R S where
  ts := tₛ
  tr _ _ := t_d
  spent H s t :=
    let P : Actual.Prover Q R S 𝓕 V tₛ t_d ℓ read hash := A.online
    ((P.strategy (H, U, A.act, A.pre.eval H, s)).workIn (π _) V {u | (u : ℝ) < t} : ℝ)

end Accounting
