module

public import Definitions.Accounting.JointPeriod
public import Definitions.NetworkCertifier.PooledLeak
public import Definitions.NetworkCertifier.Checked
public import Definitions.NetworkCertifier.Tap
public import Definitions.Core.Assumptions
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# A period whose communication audit is the network certifier's monitoring period of sampled audits, pooled (trusted)

A period's run leaves the network certifier's claim (`commWithin`) and its verdict (`accepts .network`) as fields.
`RealizesCommOfPooledLeakAccepted` is a run whose communication audit is the monitoring period of `m` execution
phases that `NetTiming.Guarantees.PooledLeakAccepted` is about (its windows): each execution phase registered as
`sw i` and audited in its verification phase on its own live coins, the network certifier recording each one's
observations. The network audit's own coins are the `m` verification phases' coins (`coins`), and the run's claim at
`K` is the monitoring period's good event, `NetTiming.PooledGood`: the incorrect sets of the witnesses computed from
each registration hold fewer than `Fmax` proof blocks in total, and every execution phase's observations meet its leak
relation at its own incorrect set. That is the event whose secrets `PooledLeakAccepted` counts, with the incorrect
proof blocks paid for once over the monitoring period, so `K` is the bits of that count: the pooled reading of the
monitoring period, where the realization beside this one is the reading per phase. The run carries the deployed
network certifier in `PooledLeakAccepted`'s form, and its trust, `trusted-host` (`Verity.Assumptions.TrustedHost`), is
a field.

The period's operator (`op`) is `PooledLeakAccepted`'s: fixed before the period, a function of the observations
alone. Execution phase `i`'s operator at the record `o` is `op i (runsBefore o i)`, which reads only the execution
phases closed before its own, and its own only through the observations before each step.

`PhaseLeaks` names the second of the two inputs `PooledLeakAccepted` takes at each execution phase, the leak relation
at the witness's incorrect set and at that phase's operator. The first, the miss probability given the earlier phases'
coins, has its name among the network certifier's definitions.
-/

@[expose] public section

namespace Accounting.JointBridge

open Claims Claims.Sampling NetTiming

/-- **The leak relation at each execution phase's own incorrect set**, the second input `PooledLeakAccepted` takes
(its `hleak`): at each secret `z ∈ Zs`, over the monitoring period's coins `ω`, execution phase `i`'s audit accepts
while its observations `obs z ω i` fail the leak relation at the incorrect set `W z ω i` (`LeakAt`) with probability
at most `e' i`. The relation is read at the phase's operator at the record's earlier phases,
`op i (runsBefore (obs z ω) i)`: the period's operator `op` is fixed before the period, and execution phase `i`'s
reads only the execution phases closed before its own. The leak guarantee of one sampled audit, on live coins and at
the incorrect set of the witness computed before the sample, is what gives it at an execution phase. It is the input
the pooled bounds over checked inputs name for their leak relations, here for `LeakAt`. -/
def PhaseLeaks {L K S m : ℕ} {Ω Z : Type} [Fintype Ω] (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement)
    (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (Zs : Set Z)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop) (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (obs : Z → (Fin m → Ω) → Fin m → List (Obs L K S)) (e' : Fin m → ℝ) : Prop :=
  ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
    ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i)) (W z ω i) (obs z ω i)) ≤ e' i

/-- **A period whose communication audit is the network certifier's monitoring period of `m` sampled audits,
pooled**, at the public layouts `lay`, horizons `k`, registered inputs `reg`, jitter bound `Dj` and the bound `Fmax`
on the period's incorrect proof blocks, with the claim at `K`. Every field is a property of the deployed run:

* the period's execution phases, as `PooledLeakAccepted` takes them: each one's registration (`sw`), its partition
  into proof blocks (`Pw`) and its registered values (`ρ`); and the period's operator (`op`), fixed before the period:
  execution phase `i`'s operator at the record `o` is `op i (runsBefore o i)`, which reads only the execution phases
  closed before its own;
* the network audit's own coins are the `m` verification phases' coins, each a `Coin` (`coins`). Every field below
  reads the run at the other audits' coins `ξ` with the network audit's own set to the phases' coins `ω`
  (`Function.update ξ (r.own .network) (coins.symm ω)`, as an audit's guarantee, `Sound`, does), so nothing reads the
  coordinate of `ξ` at the network audit;
* the deployed network certifier, in `PooledLeakAccepted`'s form: one device (`device`) on the server's uplink, with
  its connections `Conn`, cells `Cell` and the server's state `Srv`, whose run (`run`) on a schedule and the server's
  state gives the cells the client receives and its record of the period's execution phases; its record format
  (`fmt`) and what reaches it from the server on each connection (`arr`); and, at the coins, its schedule (`sched`),
  which the format holds (`fits`), and the server's state (`srv`). The record that the claim and the audits read is
  its run's on those two (`record`);
* the network certifier runs its code on its host (`trusted`, `trusted-host`);
* at execution phase `i`, after the earlier phases' coins: its audit's verdict at its own coins `x` (`acc ξ i ω x`),
  and the incorrect set of the witness computed from its registration, before its sample (`W ξ ω i`). Neither reads
  the coins of phase `i` or of a later one through `ω` (`acc_prefix`, `W_prefix`), which holds when each execution
  phase is registered before its coins are flipped and each verification phase's verdict is fixed before the next
  one's are. Without it a prover that saw those coins could place its incorrect proof blocks where no sample falls,
  and the miss probabilities would not multiply;
* the network audit accepts only if every verification phase's audit does (`accepts_imp`);
* the claim at `K` is the monitoring period's good event at those incorrect sets and the network certifier's record
  (`comm_iff`), each execution phase's leak relation read at its operator at the record's earlier phases: that is the
  event whose secrets `PooledLeakAccepted` counts, so `K` is the bits of that count, `log2`
  of the bound it gives. No field ties the number `K` to the count yet, and none holds the count's own parameters
  (each execution phase's number of proof blocks, the bits a proof block commits, the run's schedules and the most
  cells each gives a connection): they come with that tie.

The two inputs that `PooledLeakAccepted` takes at each execution phase are not fields: the miss probability given the
earlier phases' coins (`NetTiming.PeriodEscapes` at `acc` and `W`) and the leak relation at the witness's incorrect
set (`PhaseLeaks` at `op`, `acc`, `W` and `record`), each at every value of the other audits' coins, are the premises of
`Accounting.Guarantees.Bridge.CommOfPooledLeakAccepted` (`hesc`, `hleak`). -/
structure RealizesCommOfPooledLeakAccepted {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)]
    (r : PeriodRun d α C Ω) (K : ℝ) (L Kₒ S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ)
    (Dj m Fmax : ℕ) where
  /-- each execution phase's registration -/
  sw : Fin m → AuditStatement
  /-- each execution phase's partition into proof blocks -/
  Pw : Fin m → Partition
  /-- each execution phase's registered values at its registered inputs -/
  ρ : Fin m → ℕ → Bool
  /-- the period's operator, fixed before the period: execution phase `i`'s, at the observations of the execution
  phases closed before it, in order, gives the ingress at each step from the observations before that step -/
  op : Fin m → List (List (Obs L Kₒ S)) → ℕ → List (Obs L Kₒ S) → List Bool
  /-- one verification phase's coins -/
  Coin : Type
  [coinFintype : Fintype Coin]
  [coinNonempty : Nonempty Coin]
  /-- the network audit's own coins are the `m` verification phases' coins, phase `i`'s at `coins x i` -/
  coins : Ω (r.own .network) ≃ (Fin m → Coin)
  /-- the connections on the server's uplink that the network certifier carries -/
  Conn : Type
  [connFintype : Fintype Conn]
  /-- the cells it carries -/
  Cell : Type
  /-- the server's state, which fixes what reaches the network certifier -/
  Srv : Type
  /-- the network certifier's devices -/
  Dev : Type
  /-- what reaches the network certifier from the server on each connection, at the server's state -/
  arr : Srv → Conn → Arrivals Cell
  /-- the network certifier's record format over the period's execution phases -/
  fmt : RecordFormat Conn Cell (Fin m → List (Obs L Kₒ S))
  /-- the deployed network certifier -/
  device : Dev
  /-- what the network certifier does on a schedule and the server's state: the cells the client receives and its
  record of the period's execution phases -/
  run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L Kₒ S))
  /-- the network certifier runs its code (`recordingCode`) on its host: `trusted-host`, the trust the network audit
  rests on. `PooledLeakAccepted` takes it, and its record is what the leak relation (`hleak`) is about. -/
  trusted : Verity.Assumptions.TrustedHost device run fun p => recordingCode fmt p.1 (arr p.2)
  /-- the network certifier's schedule at the coins -/
  sched : ((i : I) → Ω i) → CertifierSchedule Conn
  /-- the server's state at the coins -/
  srv : ((i : I) → Ω i) → Srv
  /-- the record format holds every schedule of the run -/
  fits : ∀ ξ, fmt.fits (sched ξ)
  /-- execution phase `i`'s audit accepts at its own coins `x`, after the earlier phases' coins in `ω`, at the other
  audits' coins `ξ` -/
  acc : ((i : I) → Ω i) → Fin m → (Fin m → Coin) → Coin → Prop
  /-- a verification phase's verdict reads `ω` only at the phases before it -/
  acc_prefix : ∀ ξ i ω ω', (∀ j, j < i → ω j = ω' j) → acc ξ i ω = acc ξ i ω'
  /-- the incorrect set of the witness computed from execution phase `i`'s registration, before its sample, after the
  earlier phases' coins in `ω`, at the other audits' coins `ξ` -/
  W : ((i : I) → Ω i) → (Fin m → Coin) → Fin m → Finset ℕ
  /-- an execution phase's incorrect set reads `ω` only at the phases before it -/
  W_prefix : ∀ ξ i ω ω', (∀ j, j < i → ω j = ω' j) → W ξ ω i = W ξ ω' i
  /-- the network audit accepts only if every verification phase's audit accepts at its own coins -/
  accepts_imp : ∀ ξ ω, r.accepts .network (Function.update ξ (r.own .network) (coins.symm ω)) → ∀ i, acc ξ i ω (ω i)
  /-- the claim at `K`: the monitoring period's good event, at the witnesses' incorrect sets and the network
  certifier's record of the run -/
  comm_iff : ∀ ξ ω, r.commWithin K (Function.update ξ (r.own .network) (coins.symm ω)) ↔
    PooledGood lay k reg Dj sw Pw ρ op Fmax (W ξ ω)
      (run device (sched (Function.update ξ (r.own .network) (coins.symm ω)),
        srv (Function.update ξ (r.own .network) (coins.symm ω)))).2

attribute [instance] RealizesCommOfPooledLeakAccepted.coinFintype RealizesCommOfPooledLeakAccepted.coinNonempty
  RealizesCommOfPooledLeakAccepted.connFintype

/-- **The network certifier's record of the period's execution phases**, at the other audits' coins `ξ` and the
verification phases' coins `ω`: its run's record on the schedule and the server's state at those coins, which the
claim (`comm_iff`) and the leak relation (`PhaseLeaks`) read. -/
def RealizesCommOfPooledLeakAccepted.record {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} {K : ℝ} {L Kₒ S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ}
    {Dj m Fmax : ℕ} (R : RealizesCommOfPooledLeakAccepted r K L Kₒ S lay k reg Dj m Fmax) (ξ : (i : I) → Ω i)
    (ω : Fin m → R.Coin) : Fin m → List (Obs L Kₒ S) :=
  (R.run R.device (R.sched (Function.update ξ (r.own .network) (R.coins.symm ω)),
    R.srv (Function.update ξ (r.own .network) (R.coins.symm ω)))).2

end Accounting.JointBridge
