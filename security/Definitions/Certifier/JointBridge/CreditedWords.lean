module

public import Definitions.Compose.JointPeriod
public import Definitions.Compose.Compose
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

/-! A credit in words priced as work on the device model, on the premises `WordCostsMacs` and `HelpWithinCertifier`, since
nothing PoUW's check sees ties a word to work. -/

@[expose] public section

namespace ResourceAccounting

/-- **The units the network attester's bits can carry out**: `K / b_unit`, at `b_unit` bits a unit (about `1024` for a
unit whose inputs are seed-derived: a SHA-512 seed out and a SHA-512 root in). At `b_unit = 0` it is `0`. -/
noncomputable def Certificates.helpUnits (c : Certificates) (bUnit : ℕ) : ℝ := c.K / bUnit

/-- **A period whose PoUW audit credits words**, at certificates `c`, price `p`, `m` multiply-adds a word, `u`
multiply-adds a unit and `b_unit` bits a unit, over a source game on the coins `X` with acceptance `acc` and credit
`credit` (in words) at each value `ξ` of the period's coins (read only off PoUW's coordinate: the other audits' coins
fix the adversary the device is in the source game). Each field is a property of the deployed run. -/
structure RealizesCreditedWords {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (p : Price) (m u bUnit : ℕ) (X : Type)
    (acc : ((i : I) → Ω i) → X → Prop) (credit : ((i : I) → Ω i) → X → ℝ) where
  coin : Ω (r.own .pouw) ≃ X
  accepts_iff : ∀ ξ y, r.accepts .pouw (Function.update ξ (r.own .pouw) y) ↔ acc ξ (coin y)
  cert_le : ∀ ξ y, acc ξ (coin y) → c.W ≤ m * p .tc * credit ξ (coin y) - u * p .tc * c.helpUnits bUnit

/-- A CP-W checkpoint binds one 32-wide `K`-segment of the tile's `R·C` products and is two committed words
(`Gates.Place.parts`), so `32·R·C / 2`. -/
def receiptWordMacs (R C : ℕ) : ℕ := 16 * R * C

/-- Its `2·(K/32)` credited words (`Gates.roleOf`) at `receiptWordMacs R C` each: `K·R·C` when `32 ∣ K`. -/
def receiptUnitMacs (K R C : ℕ) : ℕ := 2 * (K / 32) * receiptWordMacs R C

/-- **The fewest multiply-adds any credited word of the registered circuit stands for**, at `macs k` for a word of kind
`k` (`receiptWordMacs R C` for an H1 + fx1 tile's): the minimum over its credited words (`Compose.creditedWords`), and
`0` for a circuit of none. It is exact for a circuit whose words are all of one kind (H1 + fx1 alone). -/
def Compose.minWordMacs {ι K : Type} (Sc : Compose.Scheme K) (Cd : Compose.Declared ι) (macs : Compose.Kind → ℕ) : ℕ :=
  ((Compose.creditedWords Sc Cd).map fun i => macs (Compose.kindAt Cd i)).min?.getD 0

end ResourceAccounting
