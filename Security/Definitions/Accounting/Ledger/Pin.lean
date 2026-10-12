import Mathlib.Data.Set.Card
import Definitions.Accounting.Ledger.Physical
import Definitions.Pouw.Basic.Prob

/-!
# Pins, requirements and additivity (trusted)

An **experiment** (`Experiment`) gives, for each adversary `a` and each value `ω` of the verifier's coins, the actual
circuit the hardware runs and the protocol's verdict. Probability is `Pouw.pr`, uniform over the coins.

* **A pin** is what a protocol establishes on accept. A lower pin (`Experiment.Pin`) bounds a meter of a region's ledger
  from below: against every adversary, the protocol accepts with the meter below `π` with probability at most `ε`. Work
  after the seed, bits held at the challenges and store bits read in a window are lower pins
  (`Definitions.Accounting.Ledger.Pins`). Today's guarantees state most of them in **budget form**
  (`Experiment.PinBudget`): against an adversary whose meter is at most `B` on every coin, the protocol accepts with
  `B < π` with probability at most `ε`. That is weaker: it bounds the meter's worst case over the coins, not its value
  at the coin drawn (`Accounting.Ledger.Experiment.Pin.budget`). The two are the readings (`Reading`, `PinAs`). A claim
  holds in either, since a capacity bounds the meter on every coin; budget pins on two regions don't compose, since the
  regions' worst cases may sit on different coins. An upper pin (`ContentAtMost`) bounds the undeclared content of a
  boundary crossing: the network certifier's.
* **A requirement** (`Experiment.Requires`) bounds from below the meter of every actual circuit that runs a task in a
  region. A **policy** is a requirement of a prohibited class of computations, such as what training a model of a given
  size needs.
* **Additivity** (`Experiment.Additive`) is the assumption that makes claims into arithmetic: running the declared work
  and a task `P` together costs at least what each needs. If the protocol pins `π` and `P` requires `R`, the protocol
  run alongside `P` (`Experiment.alongside`) pins `π + R` at the same error, in the same reading. It is one named shape
  per resource (`ComputeAdditive`, `BitsAdditive`, `CrossingAdditive`), each with its independence condition `Indep`
  on tasks. For
  bits it is information theory, but only for a task independent of the declared data: fine-tuning the declared model
  reuses its weights, so the width they take counts for both. For compute it is a conjecture, a direct sum across
  different tasks, stronger than Ball et al.'s within one task. `Indep` must exclude every task that reuses the
  declared work or data; with `Indep` true of every task the shape is false.
* **A claim** is arithmetic: if `P`'s requirement exceeds capacity minus pin in some meter, `P` didn't run there,
  except with the pin's error (`Accounting.Ledger.Experiment.claim`).
-/

namespace Accounting.Ledger

/-- An experiment over leaves `Λ`, gate types `G`, adversaries `Adv` and the verifier's coins `Ω`: adversary `a`'s actual
circuit at coins `ω`, and the protocol's verdict. -/
structure Experiment (Λ G Adv Ω : Type) where
  /-- the actual circuit, by its annotation's counts -/
  circuit : Adv → Ω → Annotation Λ G
  /-- the protocol accepts -/
  accepts : Adv → Ω → Prop

/-- A task: the adversaries and coins at which the actual circuit runs a given computation in a given region. -/
abbrev Task (Adv Ω : Type) : Type := Adv → Ω → Prop

/-- How a lower pin reads its meter: at the coins drawn, or its worst case over them. -/
inductive Reading
  /-- at the coins drawn (`Experiment.Pin`) -/
  | perRun
  /-- its worst case over the coins (`Experiment.PinBudget`) -/
  | budget

namespace Experiment

variable {Λ G Adv Ω : Type} [Fintype Λ] [DecidableEq Λ] [Fintype G] (E : Experiment Λ G Adv Ω)

/-- Meter `m` of region `ρ`'s ledger, in adversary `a`'s actual circuit at coins `ω`. -/
def read (ρ : Region Λ) (m : Meter G) (a : Adv) (ω : Ω) : ℕ := m.eval ((E.circuit a ω).ledger ρ)

/-- The experiment in which the protocol's verdict also requires task `P` to run: the declared work alongside `P`. -/
def alongside (P : Task Adv Ω) : Experiment Λ G Adv Ω := ⟨E.circuit, fun a ω => E.accepts a ω ∧ P a ω⟩

/-- **A requirement**: every actual circuit that runs task `P` has meter `m` of region `ρ` at least `R`. -/
def Requires (P : Task Adv Ω) (ρ : Region Λ) (m : Meter G) (R : ℝ) : Prop :=
  ∀ a ω, P a ω → R ≤ (E.read ρ m a ω : ℝ)

variable [Fintype Ω]

/-- **A lower pin**: against every adversary, the protocol accepts with meter `m` of region `ρ` below `π` with
probability at most `ε` over the coins. -/
def Pin (ρ : Region Λ) (m : Meter G) (π ε : ℝ) : Prop :=
  ∀ a, Pouw.pr (fun ω => E.accepts a ω ∧ (E.read ρ m a ω : ℝ) < π) ≤ ε

/-- **A lower pin in budget form**: against every adversary whose meter `m` of region `ρ` is at most `B ≥ 0` on every
coin, the protocol accepts with `B < π` with probability at most `ε`. PoUW's `EndToEnd` and the device model's
`Run.ComputeSound`, `Run.BandwidthSound` and `Run.CapacitySound` are in this form. -/
def PinBudget (ρ : Region Λ) (m : Meter G) (π ε : ℝ) : Prop :=
  ∀ a (B : ℝ), 0 ≤ B → (∀ ω, (E.read ρ m a ω : ℝ) ≤ B) → Pouw.pr (fun ω => E.accepts a ω ∧ B < π) ≤ ε

/-- A lower pin in reading `r`: per run (`Pin`) or in budget form (`PinBudget`). -/
def PinAs : Reading → Region Λ → Meter G → ℝ → ℝ → Prop
  | .perRun => E.Pin
  | .budget => E.PinBudget

/-- **Additivity** of meter `m` in region `ρ`, in reading `r`, under independence condition `Indep`: if the protocol
pins `π` of `m` at error `ε`, and a task `P` with `Indep P` requires `R` of `m`, the protocol run alongside `P` pins
`π + R` at error `ε`. -/
def Additive (r : Reading) (Indep : Task Adv Ω → Prop) (ρ : Region Λ) (m : Meter G) : Prop :=
  ∀ (P : Task Adv Ω) (π R ε : ℝ), Indep P → E.PinAs r ρ m π ε → E.Requires P ρ m R →
    (E.alongside P).PinAs r ρ m (π + R) ε

/-- **Additivity for compute** at price `p`: a direct sum across tasks. Work that runs an independent task does not
also produce the declared transcript. A conjecture for hard matmuls, stronger than non-amortizability within one task. -/
def ComputeAdditive (r : Reading) (Indep : Task Adv Ω → Prop) (ρ : Region Λ) (p : G → ℕ) : Prop :=
  E.Additive r Indep ρ (Meter.priced p)

/-- **Additivity for bits held**: width a task independent of the declared data needs is not also the declared data's.
-/
def BitsAdditive [DecidableEq G] (r : Reading) (Indep : Task Adv Ω → Prop) (ρ : Region Λ) : Prop :=
  E.Additive r Indep ρ (Meter.entry .held)

/-- **Additivity for a crossing**, outward from region `ρ`'s part (from device memory into the cores, when the part is
the memory): bits an independent task needs to move are not also the declared reads. -/
def CrossingAdditive [DecidableEq G] (r : Reading) (Indep : Task Adv Ω → Prop) (ρ : Region Λ) : Prop :=
  E.Additive r Indep ρ (Meter.entry .crossOut)

end Experiment

/-- **An upper pin on content**: a covert sender encodes a secret `z : Z` in what crosses a boundary, `wire z` is the
content that crossed in the run with secret `z`, and `accepted z` the audit's verdict on that run. No decoder of the
wire recovers more than `N` secrets whose runs are all accepted, so the undeclared content is at most `log₂ N` bits. -/
def ContentAtMost {Z W : Type} (accepted : Z → Prop) (wire : Z → W) (N : ℕ∞) : Prop :=
  ∀ Zs : Set Z, (∀ z ∈ Zs, accepted z) → ∀ dec : W → Z, (∀ z ∈ Zs, dec (wire z) = z) → Zs.encard ≤ N

end Accounting.Ledger
