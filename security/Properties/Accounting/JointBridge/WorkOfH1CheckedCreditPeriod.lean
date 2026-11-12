module

public import Definitions.Accounting.Assumptions.CreditedWords
public import Properties.Pouw.Period
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
# The joint period's PoUW guarantee from H1's checked credit over the torch loop

Each audit's per-period guarantee in the coins model is `Accounting.PeriodRun.Sound`, which the one theorem's targets
couple into the causal game (`Accounting.PeriodLive`). `WorkOfH1CheckedCreditPeriodAt p macs u b_unit` gives PoUW's, at the work certificate `W` and price `p`,
for a period whose PoUW audit credits H1's checked words (`Accounting.RealizesCreditedWords` at the play's acceptance
and the check's credit over the accepted prefix, `m = Accounting.Compose.minWordMacs Sc Cd macs` multiply-adds a word,
`u` a unit and `b_unit` bits a unit), from `Properties.Pouw.Period.H1CheckedCreditPeriod` and its premises, with its
error `δ_s + n·(δ_bind + δ_ks)`. It is `Properties.Accounting.JointBridge.WorkOfCreditedWords` at H1's instance.
`WorkOfH1CheckedCreditPeriod` is it at every price, word and unit count and unit width. Its instance at the H200's INT8
price is a corollary, stated beside its proof.

The other audits' coins fix the adversary the device is in H1's game, so the windows, the service's openings, the
salts, the period's live tiles, the play, its folded words and the right words it imported are functions of them
(`ws ξ`, …), and each premise of the source that reads them is taken at every value `ξ` of the coins. The scheme, the
declared circuit, the number of windows, the check's parameters, the salt's encoding, the staged and flat statements
and the errors are the protocol's and are fixed. The premises are the source's, unchanged; the gaps it names
(`RegisteredBeforeDraw` and the binding it reads on today's tags, `SaltIsLiveCoin`, `StagedPins`) stay premises here.
The bridge adds two: each right word of a unit the device didn't outsource cost `m` multiply-adds of
priced work on it (`Accounting.Assumptions.WordCostsMacs`, at `n` times the run's fixed words), and it outsourced at
most `K / b_unit` units (`Accounting.Assumptions.HelpWithinCertifier`).
-/

@[expose] public section

namespace Properties.Accounting.JointBridge

open Properties.Pouw.Period PouwCheck
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **PoUW's per-period guarantee in the joint period, from H1's checked credit on the torch loop, at price `p`,
`macs k` multiply-adds a word of kind `k`, `u` a unit and `b_unit` bits a unit.** On any device and any period of a
node's run on one coin space whose PoUW audit credits H1's checked words (`Accounting.RealizesCreditedWords`: PoUW's
coordinate of the coins is the source game's, its verdict the play's acceptance, and the work certificate within the
check's credit at `m = minWordMacs Sc Cd macs` multiply-adds a word, less the `K / b_unit` units the network
certifier's bits can carry at `u` multiply-adds each, all at `p .tc`), each right word of a unit the device didn't
outsource costing `m` multiply-adds of priced work (`WordCostsMacs` at `n` times the run's fixed words and `help`
outsourced units), at most `K / b_unit` units outsourced (`HelpWithinCertifier`), and the source's instance at each
value `ξ` of the coins meeting `H1CheckedCreditPeriod`'s premises: PoUW's per-period guarantee holds at certificates
`c`, price `p` and error `δ_s + n·(δ_bind + δ_ks)`, i.e. at any value of the other coins, the probability over PoUW's
own coins that it accepts while the compute account did less than `W` of work at `p` is at most that. -/
def WorkOfH1CheckedCreditPeriodAt (p : Accounting.Price) (macs : Accounting.Compose.Kind → ℕ) (u bUnit : ℕ) : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : Accounting.Device ι) (C : Verity.Circuit) (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates)
    {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι') {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params) (enc : Cn → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
    (ws : ((i : I) → Ω i) → Executions (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray)
    (Wp : ((i : I) → Ω i) →
      WindowPlay Cn (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : ((i : I) → Ω i) → Play (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (help : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (real : Accounting.RealizesCreditedWords r c p (Accounting.Compose.minWordMacs Sc Cd macs) u bUnit
      (Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n) (fun ξ => (Pl ξ).accepts)
      (fun ξ x => creditPrefix Sc Cd 144 ((ws ξ).prefixLen prm x) (checkedLive Sc Cd (Wp ξ) prm (ws ξ) x) 1 1 (67 / 8)
        δs)),
    Accounting.Assumptions.WordCostsMacs r p (Accounting.Compose.minWordMacs Sc Cd macs) u real.coin
      (fun ξ => (Pl ξ).accepts) (fun ξ x => n * ((Pl ξ).run x).fixed) help →
    Accounting.Assumptions.HelpWithinCertifier r c bUnit real.coin (fun ξ => (Pl ξ).accepts) help →
    Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, stageShare (tileStage Cd 144) (shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < shareCone Sc Cd 144 t → tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    (∀ ξ, (Pl ξ).game = periodGame (tileStage Cd 144) (liveCone Sc Cd 144) 1 1 24 (Wp ξ)) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live _ => DeadConeDead Sc Cd 144 live)) →
    (∀ ξ, PrefixOf (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (Pl ξ).game) →
    (∀ ξ w, prob (fun x =>
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)).MovedIn x w) ≤
        δbind) →
    (∀ ξ, RegisteredBeforeDraw (ws ξ)) → (∀ ξ, SampleIsCoin hK prm (ws ξ)) →
    (∀ ξ, ServiceSound prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) δks) →
    (∀ ξ, SaltIsLiveCoin hr (salt ξ) enc) → StagedPins staged flat →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G))) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G))) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0))) →
    (∀ ξ, UnitFixed (Pl ξ)
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (periodMode Cd (Wp ξ) (checkedRun hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)))
      (fun _ t => (tileWords Sc Cd 144 t : ℝ)) (folded ξ)) →
    r.Sound c p .pouw (δs + n * (δbind + δks))

/-- **PoUW's per-period guarantee in the joint period from H1's checked credit**, at every price `p`, per-kind
multiply-add count `macs`, unit count `u` and unit width `b_unit` (`WorkOfH1CheckedCreditPeriodAt`). -/
def WorkOfH1CheckedCreditPeriod : Prop :=
  ∀ (p : Accounting.Price) (macs : Accounting.Compose.Kind → ℕ) (u bUnit : ℕ),
    WorkOfH1CheckedCreditPeriodAt p macs u bUnit

end Properties.Accounting.JointBridge
