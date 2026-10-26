import Definitions.Core.Accel.Machine
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Theorem O at the committed checkpoint: the program's one computational slot (trusted)

Every coin and key the honest receipt of a round reads (the noise and dither coins, the salt, the fold's start state
and keys) is read by the prover before it commits that receipt, since it needs them to compute the product. The only
coins it has not read at commit time are the draw's and the proofs', and neither enters the honest receipt. So whether
a cheaper program commits the honest checkpoint is a question about the model's costs, never about luck over coins the
cheater lacks. The receipt-side lucks (`Accounting.Assumptions.TS.OperandHMu`, `ReceiptLuckAt`,
`Accounting.Assumptions.Compose.GuessLuckAt`, H-μ⁺'s point mass) are that question asked per word, per receipt and
per unit, and this module asks it once, about the checkpoints a round commits.

* `Round`: a round on MODEL-SPEC's machine (`Verity.Accel.Machine`): its live coins, how its committed inputs and
  coins are written into the state a program starts from, where the committed checkpoints are read, the honest
  checkpoints, the draw and the check on the drawn units (drawn after the commit), and the honest kernel's priced
  cost `H`.
* `priced`: a run's cost at prices `λ`, `Σ_t Σ_r λ_r u_r(t) / ρ_r`.
* `moment`: the average over the round's coins and draw of `e^{η (c − cost/H)₊}` where the committed checkpoints pass
  the check, `0` elsewhere.
* **`Slot`**: Theorem O at the checkpoint. Every program and starting state, fixed before the coins, on every schedule
  chosen after them, whose runs meet the model (`Machine.Fits`), has moment at most 1.
* `Tail`, `TailOfSlot`: the slot's tail form, `Pr[match ∧ cost ≤ (c − δ)H] ≤ e^{−ηδ}`, and that the slot gives it
  (Markov).
* `HonestMeets`, `HonestTight`: non-vacuity. The honest kernel meets the slot's hypotheses, commits the honest
  checkpoints on every coin, and its moment is exactly 1, so the bound is attained at a saving of 0.
-/

namespace Pouw.TheoremO

open Verity.Accel

variable {ι : Type} {m : Machine ι} {Θ : Type} [DecidableEq Θ]

/-- **A round** as the slot reads it, on the machine `m` with threads `Θ`. The adversary's state before the round is
any state; `write` puts the round's committed inputs and its live coins into it, at the protocol's locations. `out`
reads the checkpoints the run commits from its final stores, and `honest` is the fold of the honest words, a function
of the committed inputs and the coins. `H` is the honest kernel's priced cost of the round. -/
structure Round (m : Machine ι) (Θ : Type) where
  /-- the round's live coins, uniform -/
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  /-- the committed checkpoints -/
  V : Type
  /-- the round's committed inputs and its coins, written into the state the program starts from -/
  write : Ω → m.State Θ → m.State Θ
  /-- the checkpoints a run commits, read from its final stores -/
  out : m.Mem → V
  /-- the honest checkpoints -/
  honest : Ω → V
  /-- the draw and check coins, uniform, drawn after the run commits -/
  D : Type
  [fintypeD : Fintype D]
  [nonemptyD : Nonempty D]
  /-- the check at draw `d`: the committed checkpoints pass against the honest ones at the drawn units -/
  ok : D → V → V → Prop
  /-- the honest checkpoints pass at every draw -/
  ok_refl : ∀ d v, ok d v v
  /-- the honest kernel's priced cost of the round -/
  H : ℝ

attribute [instance] Round.fintype Round.nonempty Round.fintypeD Round.nonemptyD

variable [Fintype m.Res]

/-- **A run's priced cost** at prices `λ` on the resources: `Σ_t Σ_r λ_r · u_r(t) / ρ_r` over the ticks the run
issues in, in device-ticks when `λ` is a probability vector (DITHER-THM's `Accel.priced` on the model's runs). An
import is priced as the resource that carries it. -/
noncomputable def priced (lam : m.Res → ℝ) (iss : List (List m.Instr)) : ℝ :=
  ∑ t ∈ Finset.range iss.length, ∑ r, lam r * (Machine.usage iss t r : ℝ) / (m.ρ r : ℝ)

/-- The run of `P` on coin `ω` from the adversary's state `σ₀`, on the schedule `sched ω`. -/
def Round.run (R : Round m Θ) (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)) (ω : R.Ω) :
    m.State Θ × List (List m.Instr) :=
  Machine.run P (R.write ω σ₀) (sched ω)

open Classical in
/-- The run on coin `ω` passes the check at draw `d`. -/
def Round.passes (R : Round m Θ) (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)) (ω : R.Ω)
    (d : R.D) : Prop :=
  R.ok d (R.out (R.run P σ₀ sched ω).1.mem) (R.honest ω)

open Classical in
/-- **A program's moment** on the round at prices `λ`, model constant `c` and rate `η`: the average over the round's
coins and its draw of `e^{η (c − cost/H)₊}` where the run's committed checkpoints pass the check, and `0` where they
don't. -/
noncomputable def moment (R : Round m Θ) (lam : m.Res → ℝ) (c η : ℝ) (P : m.Prog Θ) (σ₀ : m.State Θ)
    (sched : R.Ω → List (List Θ)) : ℝ :=
  (∑ ω, ∑ d, if R.passes P σ₀ sched ω d then
      Real.exp (η * max 0 (c - priced lam (R.run P σ₀ sched ω).2 / R.H)) else 0) /
    (Fintype.card R.Ω * Fintype.card R.D)

/-- **Theorem O at the committed checkpoint**: in the accelerator model `m`, every program `P` and starting state `σ₀`,
both fixed before the round's coins, on every schedule chosen after them (`sched ω`), whose runs meet the model on
every coin (`Machine.Fits`: (G1) and (G2)), has moment at most 1 at prices `λ`, constant `c` and rate `η`:
`E_{ω,d}[1(the committed checkpoints pass at draw d) · e^{η (c − cost_λ/H)₊}] ≤ 1`. By Markov (`TailOfSlot`), a
program that saves `δ·H` below `c·H` passes on at most an `e^{−ηδ}` share of the coins and draws. With `ok d x y`
the equality `x = y` this is the plain form: the program commits every honest checkpoint.

**This is a theorem to be proved in the model, not an assumption.** It is true or false of `m` and the round alone,
and it is the program's one computational slot: DITHER-THM's statement reads it where it reads Theorem O, and the
receipt-side lucks are its special cases, per word (`OperandHMu`), per receipt with wrong words (`ReceiptLuckAt`) and
per unit skipped whole (`GuessLuckAt`). Until it is proved it is exactly what Daniel's pending decision on symmetric
algorithmic knowledge for the work off the tensor cores would assume: that no program does the round's checkpointed
work, the fold and the instructions around the products, at a cost below what the model charges the honest kernel. -/
def Slot (R : Round m Θ) (lam : m.Res → ℝ) (c η : ℝ) : Prop :=
  ∀ (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)),
    (∀ ω, Machine.Fits P (R.write ω σ₀) (sched ω)) → moment R lam c η P σ₀ sched ≤ 1

open Classical in
/-- The share of the round's coins and draws on which a run passes the check at priced cost at most `(c − δ)·H`. -/
noncomputable def matchShare (R : Round m Θ) (lam : m.Res → ℝ) (c δ : ℝ) (P : m.Prog Θ) (σ₀ : m.State Θ)
    (sched : R.Ω → List (List Θ)) : ℝ :=
  ((Finset.univ.filter fun x : R.Ω × R.D => R.passes P σ₀ sched x.1 x.2 ∧
      priced lam (R.run P σ₀ sched x.1).2 ≤ (c - δ) * R.H).card : ℝ) / (Fintype.card R.Ω * Fintype.card R.D)

/-- **The slot's tail form**: every program and starting state whose runs meet the model, at every saving `δ ≥ 0`,
passes the check at priced cost at most `(c − δ)·H` on at most an `e^{−ηδ}` share of the coins and draws. At `δ = 0`
the bound is 1, which the honest kernel attains (`HonestTight`). -/
def Tail (R : Round m Θ) (lam : m.Res → ℝ) (c η : ℝ) : Prop :=
  ∀ (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)),
    (∀ ω, Machine.Fits P (R.write ω σ₀) (sched ω)) →
      ∀ δ, 0 ≤ δ → matchShare R lam c δ P σ₀ sched ≤ Real.exp (-(η * δ))

/-- **The slot gives its tail form** (Markov): for every round with `H > 0` and every rate `η ≥ 0`, `Slot` implies
`Tail`. -/
def TailOfSlot : Prop :=
  ∀ {ι : Type} {m : Machine ι} {Θ : Type} [DecidableEq Θ] [Fintype m.Res] (R : Round m Θ) (lam : m.Res → ℝ)
    (c η : ℝ), 0 < R.H → 0 ≤ η → Slot R lam c η → Tail R lam c η

/-- **The honest kernel meets the slot's hypotheses**: some program and starting state, on some schedule, run the
round within the model on every coin, commit the honest checkpoints on every coin, and are priced at least `c·H`.
An instance proves it with its honest kernel (DITHER-THM's `honest_witness`, rule I). -/
def HonestMeets (R : Round m Θ) (lam : m.Res → ℝ) (c : ℝ) : Prop :=
  ∃ (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)), ∀ ω,
    Machine.Fits P (R.write ω σ₀) (sched ω) ∧ R.out (R.run P σ₀ sched ω).1.mem = R.honest ω ∧
      c * R.H ≤ priced lam (R.run P σ₀ sched ω).2

/-- **Non-vacuity at a saving of 0**: for every round with `H > 0` that the honest kernel meets, some program meeting
the slot's hypotheses has moment exactly 1. It passes every draw on every coin (`ok_refl`) and saves nothing, so the
slot's bound is attained, and `Slot` with a bound below 1 would be false. -/
def HonestTight : Prop :=
  ∀ {ι : Type} {m : Machine ι} {Θ : Type} [DecidableEq Θ] [Fintype m.Res] (R : Round m Θ) (lam : m.Res → ℝ)
    (c η : ℝ), 0 < R.H → HonestMeets R lam c →
      ∃ (P : m.Prog Θ) (σ₀ : m.State Θ) (sched : R.Ω → List (List Θ)),
        (∀ ω, Machine.Fits P (R.write ω σ₀) (sched ω)) ∧ moment R lam c η P σ₀ sched = 1

end Pouw.TheoremO
