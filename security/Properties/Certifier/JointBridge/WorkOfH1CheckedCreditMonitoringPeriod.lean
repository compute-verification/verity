module

public import Definitions.Pouw.Assumptions.CreditedWords
public import Properties.Pouw.MonitoringPeriod
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

/-! The joint period's PoUW guarantee from H1's checked credit over the torch loop. -/

@[expose] public section

namespace Properties.Certifier.JointBridge

open Properties.Pouw.MonitoringPeriod PouwCheck
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **PoUW's per-period guarantee in the joint period, from H1's checked credit on the torch loop, at price `p`,
`macs k` multiply-adds a word of kind `k`, `u` a unit and `b_unit` bits a unit.** On any device and any period of a
node's run on one coin space whose PoUW audit credits H1's checked words (`ResourceAccounting.RealizesCreditedWords`: PoUW's
coordinate of the coins is the source game's, its verdict the play's acceptance, and the work certificate within the
check's credit at `m = minWordMacs Sc Cd macs` multiply-adds a word, less the `K / b_unit` units the network
attester's bits can carry at `u` multiply-adds each, all at `p .tc`), each right word of a unit the device didn't
outsource costing `m` multiply-adds of priced work (`WordCostsMacs` at `n` times the run's fixed words and `help`
outsourced units), at most `K / b_unit` units outsourced (`HelpWithinCertifier`), and the source's instance at each
value `ξ` of the coins meeting `H1CheckedCreditMonitoringPeriod`'s premises: PoUW's per-period guarantee holds at certificates
`c`, price `p` and error `δ_s + n·(δ_bind + δ_ks)`, i.e. at any value of the other coins, the probability over PoUW's
own coins that it accepts while the compute account did less than `W` of work at `p` is at most that. -/
def WorkOfH1CheckedCreditMonitoringPeriodAt (p : ResourceAccounting.Price) (macs : ResourceAccounting.Compose.Kind → ℕ) (u bUnit : ℕ) : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : ResourceAccounting.Device ι) (C : Verity.Circuit) (r : ResourceAccounting.PeriodRun d α C Ω) (c : ResourceAccounting.Certificates)
    {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι') {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params) (enc : Cn → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
    (ws : ((i : I) → Ω i) → Executions (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray)
    (Wp : ((i : I) → Ω i) →
      ExecutionPhasePlay Cn (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : ((i : I) → Ω i) → Play (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (help : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (real : ResourceAccounting.RealizesCreditedWords r c p (ResourceAccounting.Compose.minWordMacs Sc Cd macs) u bUnit
      (Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n) (fun ξ => (Pl ξ).accepts)
      (fun ξ x => creditPrefix Sc Cd 144 ((ws ξ).prefixLen prm x) (checkedLive Sc Cd (Wp ξ) prm (ws ξ) x) 1 1 (67 / 8)
        δs)),
    ResourceAccounting.Assumptions.WordCostsMacs r p (ResourceAccounting.Compose.minWordMacs Sc Cd macs) u real.coin
      (fun ξ => (Pl ξ).accepts) (fun ξ x => n * ((Pl ξ).run x).fixed) help →
    ResourceAccounting.Assumptions.HelpWithinCertifier r c bUnit real.coin (fun ξ => (Pl ξ).accepts) help →
    Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, stageShare (tileStage Cd 144) (shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < shareCone Sc Cd 144 t → tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    (∀ ξ, (Pl ξ).game = monitoringPeriodGame (tileStage Cd 144) (liveCone Sc Cd 144) 1 1 24 (Wp ξ)) →
    (∀ ξ, (Wp ξ).EveryExecutionPhase (fun live _ => DeadConeDead Sc Cd 144 live)) →
    (∀ ξ, PrefixOf (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (Pl ξ).game) →
    (∀ ξ w, prob (fun x =>
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)).MovedIn x w) ≤
        δbind) →
    (∀ ξ, RegisteredBeforeDraw (ws ξ)) → (∀ ξ, SampleIsCoin hK prm (ws ξ)) →
    (∀ ξ, ServiceSound prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) δks) →
    (∀ ξ, SaltIsLiveCoin hr (salt ξ) enc) → StagedPins staged flat →
    (∀ ξ, (Wp ξ).EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G))) →
    (∀ ξ, (Wp ξ).EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G))) →
    (∀ ξ, (Wp ξ).EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0))) →
    (∀ ξ, UnitFixed (Pl ξ)
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (monitoringPeriodMode Cd (Wp ξ) (checkedRun hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)))
      (fun _ t => (tileWords Sc Cd 144 t : ℝ)) (folded ξ)) →
    r.Sound c p .pouw (δs + n * (δbind + δks))

/-- **PoUW's per-period guarantee in the joint period from H1's checked credit**, at every price `p`, per-kind
multiply-add count `macs`, unit count `u` and unit width `b_unit` (`WorkOfH1CheckedCreditMonitoringPeriodAt`). -/
def WorkOfH1CheckedCreditMonitoringPeriod : Prop :=
  ∀ (p : ResourceAccounting.Price) (macs : ResourceAccounting.Compose.Kind → ℕ) (u bUnit : ℕ),
    WorkOfH1CheckedCreditMonitoringPeriodAt p macs u bUnit

end Properties.Certifier.JointBridge
