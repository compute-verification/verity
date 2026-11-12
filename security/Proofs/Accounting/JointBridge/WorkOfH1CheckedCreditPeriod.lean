module

public import Definitions.Accounting.Device
public import Definitions.Pouw.PearlC.W1
public import Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
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
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Pouw.Period
import Std.Tactic.BVDecide.Normalize.Prop
public import Proofs.Pouw.PearlC.Records

/-!
# Proof of the joint period's PoUW guarantee from H1's checked credit

`Accounting.SecurityProofs.Bridge.WorkOfCreditedWords` at H1's instance: at each value `ξ` of the other audits' coins,
`Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod` at the source's instance there bounds the play's
acceptance with `n` times its fixed words below the check's credit, which is the bound the bridge takes, read through
`Pouw.pr` (`pr_le_prob_of_equiv` at the identity). The H200's instance is it at `Accounting.h200S8Price`, the H200's
price with INT8, `receiptWordMacs R C` multiply-adds a word and `receiptUnitMacs K R C` a unit.

That price is stated here, with `Accounting.w1Price`, which it is an instance of. `Accounting.Price` prices the device
model's three op classes; W1 (`Pouw.PearlC.W1.W1Prices`) prices each tensor-core format's multiply-add and each
CUDA-core instruction, in one dense E4M3 multiply-add. `w1Price pr f` reads the first off the second for a run whose
tensor-core multiply-adds are of format `f`, and a card's price is it at that card's W1 prices and a format.
-/

@[expose] public section

namespace Accounting

open Pouw.PearlC.W1 in
/-- **The device model's price at W1's prices `pr`**, for tensor-core multiply-adds of format `f`: a tensor-core
multiply-add at `f`'s W1 price, a CUDA-core op at the cheaper of W1's FP32 and INT32 instructions, and an oracle query
at one SHA-512 compression, each rounded up to a whole unit (one dense E4M3 multiply-add). -/
def w1Price (pr : W1Prices) (f : Fmt) : Price := fun o =>
  match o with
  | .tc => ⌈pr.mac f⌉₊
  | .cuda => ⌈min pr.fp32 pr.int32⌉₊
  | .oracle => ⌈pr.sha512⌉₊

/-- **The H200's price with INT8**: `W1Device.s8H100`'s prices (`W1Prices.h100`, the H100's and the H200's, one GH100)
for S8 multiply-adds: an INT8 tensor-core multiply-add at W1's S8 price, `1`; a CUDA-core op at FP32's `32` (INT32's
is `64`); an oracle query at `sha` rounded up. -/
def h200S8Price (sha fetch : ℚ) : Price := w1Price (Pouw.PearlC.W1.W1Device.s8H100 sha fetch).prices .s8

end Accounting

namespace Accounting.SecurityProofs.Bridge

open Properties.Pouw.Period

/-- The uniform probability of `E` on `A` is at most that of `F` on `X` when `e : A ≃ X` maps `E` into `F`. -/
theorem pr_le_prob_of_equiv {A X : Type} [Fintype A] [Fintype X] (e : A ≃ X) {E : A → Prop} {F : X → Prop}
    (h : ∀ a, E a → F (e a)) : Pouw.pr E ≤ Pouw.RelativeSecurity.prob F := by
  classical
  unfold Pouw.pr Pouw.RelativeSecurity.prob Claims.Pr Claims.prUniform
  rw [Fintype.card_congr e]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hmap : ((Finset.univ.filter E).map e.toEmbedding) ⊆ Finset.univ.filter F := by
    intro x hx
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Equiv.coe_toEmbedding] at hx ⊢
    obtain ⟨a, ha, rfl⟩ := hx
    exact h a ha
  have := Finset.card_le_card hmap
  rw [Finset.card_map] at this
  convert (Nat.cast_le (α := ℝ)).2 this using 3

/-- `WorkOfH1CheckedCreditPeriodAt p macs u b_unit` from `H1CheckedCreditPeriod`, at the source's instance at each
value of the coins. -/
theorem workOfH1CheckedCreditPeriodAt_of (hsrc : H1CheckedCreditPeriod) (p : Price) (macs : Compose.Kind → ℕ)
    (u bUnit : ℕ) : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodAt p macs u bUnit := by
  intro ι α I _ _ Ω _ d C r c ι' K T L U Cn Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr hM hK prm enc staged flat δs δbind
    δks ws opened salt Wp Pl folded help real hcost hhelp hScope hρ hw hluck hgame hdead hpre hbind hreg hdraw hsvc
    hsalt hpins hmu hrb hguess hfix
  refine WorkOfCreditedWords ι α I Ω d C r c p _ u bUnit _ _ (fun ξ x => n * ((Pl ξ).run x).fixed) _ help
    (δs + n * (δbind + δks)) real hcost hhelp fun ξ => ?_
  have hs := hsrc Sc Cd hn hN hr hM hK prm (ws ξ) (opened ξ) (salt ξ) enc staged flat (Wp ξ) (Pl ξ) (folded ξ) δs
    δbind δks hScope hρ hw hluck (hgame ξ) (hdead ξ) (hpre ξ) (hbind ξ) (hreg ξ) (hdraw ξ) (hsvc ξ) (hsalt ξ) hpins
    (hmu ξ) (hrb ξ) (hguess ξ) (hfix ξ)
  refine le_trans ?_ hs
  exact pr_le_prob_of_equiv (Equiv.refl _) fun _ h => h

/-- **The joint period's PoUW guarantee from H1's checked credit**, at every price, per-kind multiply-add count, unit
count and unit width (`Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`). -/
theorem WorkOfH1CheckedCreditPeriod : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod :=
  workOfH1CheckedCreditPeriodAt_of Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod

/-- **PoUW's per-period guarantee in the joint period from H1's checked credit on the H200, at INT8, for `K`-deep
`R × C` tiles**: the guarantee above at `Accounting.h200S8Price sha fetch` (an INT8 tensor-core multiply-add `1`, a
CUDA-core op `32`, an oracle query `⌈sha⌉`, in dense E4M3 multiply-adds, W1's prices on `W1Device.s8H100`), every
word at the tile's `receiptWordMacs R C = 16·R·C` multiply-adds, every unit at its credited words'
`receiptUnitMacs K R C = 2·(K/32)·16·R·C`, and `b_unit` bits a unit, for every SHA-512 and fetch price, tile shape and
unit width. `PouwH1Tile_v1`'s is `K = 64`, `R × C = 4 × 8`: `512` a word, `2048` a unit; `PouwH1Tile_v5` and `_v6` (v1
and v3 under J10-R's noise, with v1's receipt) at a warp, `32 × 32`: `16384` a word, `4194304` a unit at `K = 4096`.
The certificate it admits on an accepted period is `W ≤ 16·R·C·credit − 2·(K/32)·16·R·C·K_net/b_unit` dense E4M3
multiply-adds, `K_net` the network certifier's bits, for a circuit with a credited word (`m` is `0` for one with
none); a circuit of several tile shapes takes the smallest word. -/
theorem WorkOfH1CheckedCreditPeriodH200 :
    ∀ (sha fetch : ℚ) (K R C bUnit : ℕ),
      Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodAt (h200S8Price sha fetch)
        (fun _ => receiptWordMacs R C) (receiptUnitMacs K R C) bUnit :=
  fun sha fetch K R C bUnit =>
    WorkOfH1CheckedCreditPeriod (h200S8Price sha fetch) (fun _ => receiptWordMacs R C) (receiptUnitMacs K R C) bUnit

end Accounting.SecurityProofs.Bridge
