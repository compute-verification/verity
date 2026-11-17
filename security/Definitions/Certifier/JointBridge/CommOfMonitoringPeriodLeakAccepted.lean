module

public import Definitions.Compose.JointPeriod
public import Definitions.Certifier.MonitoringPeriod
public import Definitions.Certifier.Tap
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

/-! A period whose communication audit is the network attester's period of sampled audits, its trust `trusted-device` a field. -/

@[expose] public section

namespace ResourceAccounting.JointBridge

open Claims Claims.Sampling NetTiming
open Claims.SamplingGeneric (Arg Compiled)

/-- **A period whose communication audit is the network attester's period of `n` sampled audits**, at the
compiled sampled verifier `D`, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter
bound `Dj`, at `δ`, with the claim at `K`. Every field is a property of the deployed run:

* the period's windows, as `MonitoringPeriodLeakAccepted` takes them: each window's registration (`sw`) and its registered
  values (`ρ`), and the network attester's openings at each statement (`tops`);
* at the coins, each window's statement (`stmt`), the audit's extracted witness there (`wit`) and the network
  attester's record of the window (`obs`);
* the claim at `K` is that every window meets `ExecutionPhaseGood` (`comm_iff`): that is the event whose secrets
  `MonitoringPeriodLeakAccepted`'s count bounds, so `K` is the bits of that count,
  `log2 (#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #incorrectSets_i(δ) · 2^β_i · Dj^k_i)`. No field ties the
  number `K` to the count yet (the follow-up: `Properties.NetworkPeriod.rowsKBoundsCount`, that the rows form's `K` is
  at least `⌈log2⌉` of the count);
* the statements are in the sampled audit's scope (`scope`): the registration refuses any other;
* the deployed network attester, in `MonitoringPeriodLeakAccepted`'s form: one device (`device`) on the server's uplink, with
  its connections `Conn`, cells `Cell` and the server's state `Srv`, whose run (`run`) on a schedule and the server's
  state gives the cells the client receives and its records of the period's windows; its record format (`fmt`) and
  what reaches it from the server on each connection (`arr`);
* the network attester runs its code on its host (`trusted`, `trusted-device`);
* at each window and every value of the other coins, the network audit's own coins are a run of the sampled verifier
  at the window's statement against a prover (`prover`, on a finite tape `Tape`, the rest of the coins), and the
  probability over the audit's own coins that the network attester accepts while the window fails `ExecutionPhaseGood` is at
  most that run's `PrBad` at the leak relation at the network attester's rows, `RLeakCertifier` (`period_le`). Past
  `RLeakCertifier` and the scope, `ExecutionPhaseGood` holds the trace's values at the registered inputs and the records' coverage
  (`Covered`, length and jitter): `period_le` holds them on acceptance. The registered values are the gap
  `MonitoringPeriodLeakAccepted` names, which the registration's commitment binds; no proof here gives it. `period_le` is the
  per-window bound that the trusted network attester's records give; deriving it from `MonitoringPeriodLeakAccepted` at the
  deployed period and `trusted`, rather than reading it as a field, is the follow-up. -/
structure RealizesCommOfMonitoringPeriodLeakAccepted {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin : Type} [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] {A : Arg Coin}
    (r : PeriodRun d α C Ω) (K : ℝ) (H : Msg → Msg) (D : Compiled Coin A) (L Kₒ S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ)
    (Dj n : ℕ) where
  sw : Fin n → AuditStatement
  ρ : Fin n → ℕ → Bool
  tops : AuditStatement → List (Opening Bool)
  stmt : ((i : I) → Ω i) → Fin n → Msg
  wit : ((i : I) → Ω i) → Fin n → Wit
  obs : ((i : I) → Ω i) → Fin n → List (Obs L Kₒ S)
  comm_iff : ∀ ξ, r.commWithin K ξ ↔
    ∀ i, ExecutionPhaseGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) tops (stmt ξ i) (wit ξ i) (obs ξ i)
  scope : ∀ ξ i, ScopeAcc D lay k (stmt ξ i)
  Tape : ((i : I) → Ω i) → Fin n → Type
  [tapeFintype : ∀ ξ i, Fintype (Tape ξ i)]
  [tapeNonempty : ∀ ξ i, Nonempty (Tape ξ i)]
  prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i)
  Conn : Type
  [connFintype : Fintype Conn]
  Cell : Type
  Srv : Type
  Dev : Type
  arr : Srv → Conn → Arrivals Cell
  fmt : RecordFormat Conn Cell (Fin n → List (Obs L Kₒ S))
  device : Dev
  run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L Kₒ S))
  trusted : Verity.Assumptions.TrustedDevice device run fun p => recordingCode fmt p.1 (arr p.2)
  /-- A field: nothing derives it from `MonitoringPeriodLeakAccepted` at the deployed period. -/
  period_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ ExecutionPhaseGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) tops (stmt (Function.update ξ (r.own .network) x) i)
          (wit (Function.update ξ (r.own .network) x) i) (obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V
      (RLeakCertifier H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
      D.E (stmt ξ i) (prover ξ i)

attribute [instance] RealizesCommOfMonitoringPeriodLeakAccepted.tapeFintype RealizesCommOfMonitoringPeriodLeakAccepted.tapeNonempty
  RealizesCommOfMonitoringPeriodLeakAccepted.connFintype

end ResourceAccounting.JointBridge
