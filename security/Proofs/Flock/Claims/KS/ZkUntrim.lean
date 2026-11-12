module

public import Proofs.Flock.Claims.KS.TrimBridge
public import Proofs.Flock.Claims.KS.ZkFork0
public import Proofs.Flock.Claims.KS.ZkTauSlot
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
import Proofs.Flock.Claims.KS.ZkBatchSim
import Proofs.Flock.Claims.KS.ZkClashPads
import Proofs.Flock.Claims.KS.ZkForkClash
import Proofs.Flock.Claims.KS.ZkPadsSlot

/-!
# KS at the untrimmed masked instance from `ExtendJ` alone

The trimmed record's KS holds with no premise (`zk_KS`), and does not give the untrimmed instance's
(`Flock.TrimBridge.hole_agree`). That one takes `zk_KS_of_kinds`'s premises and `ExtendJ` at full-width columns
(`TrimBridge.zk_KS_untrim_of_kinds`), and the premises hold at every message:

* `zkSimForkLeCapTau`: the fork bound at the caps and `τ`, from the caps' alignment and clash (`zkSlotAlign`,
  `zkForkClash`) and `τ`'s bound (`zkSimForkLeTau`);
* `zk_oneWait_untrim_of_extendJ` and `zk_KS_untrim_of_extendJ`: with the pads' fork bound and view clash
  (`zkSimForkLePads`, `zkViewClashQP`), one wait's bound and KS at the untrimmed instance at SHA-512 at the deployed
  relation from `ExtendJ` alone, which `ZkExtend` proves (`zk_extendJ`, `zk_oneWait_untrim`, `zk_KS_untrim`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at the caps and `τ`, with no hypothesis.** -/
theorem zkSimForkLeCapTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  zkSimForkLeOn_capTau (zkSimForkLeCap_of (zkForkAlign_of_slot (zkSlotAlign x)) (zkForkClash x)) (zkSimForkLeTau x)

end Gen

/-- **The fork bound at the caps and `τ`, with no hypothesis.** -/
theorem zkSimForkLeCapTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  Gen.zkSimForkLeCapTau (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **One wait's bound at the untrimmed masked instance**, at SHA-512 at the deployed relation, from `ExtendJ` at
full-width columns alone (`TrimBridge.zk_oneWait_untrim_of_kinds`). -/
theorem zk_oneWait_untrim_of_extendJ
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.OneWait sha512 (deployZk Readable.relation) :=
  TrimBridge.zk_oneWait_untrim_of_kinds (fun x _ => zkSimForkLeCapTau x) (fun x _ => zkSimForkLePads x)
    (fun x _ => zkViewClashQP x) hext

/-- **KS at the untrimmed masked instance**, at SHA-512 at the deployed relation, from `ExtendJ` at full-width columns
alone (`zk_oneWait_untrim_of_extendJ`). -/
theorem zk_KS_untrim_of_extendJ
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_of_oneWait_sha512 Readable.relation zk_splits (zk_oneWait_untrim_of_extendJ hext)

end Gen

/-- **KS at the untrimmed masked instance**, at SHA-512 at the deployed relation, from `ExtendJ` at full-width columns
alone (`zk_oneWait_untrim_of_extendJ`). -/
theorem zk_KS_untrim_of_extendJ
    (hext : Claims.Flock.ExtendJ (deployZk Readable.relation) (wordLenJ true) (wideJ true)) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  Gen.zk_KS_untrim_of_extendJ (S := S₀) (hext := hext)

end FlockDeploy.KS
