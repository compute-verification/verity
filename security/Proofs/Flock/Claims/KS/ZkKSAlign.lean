module

public import Proofs.Flock.Claims.KS.ZkFork
public import Proofs.Flock.Claims.KS.ZkForkClash
public import Proofs.Flock.Claims.KS.ZkPadsSlot
public import Proofs.Flock.Claims.KS.ZkTauFork
public import Proofs.Flock.Claims.KS.ZkTauPair
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

/-!
# Masked KS from the `τ` alignment

`zk_KS_of_slot` takes five premises at every message the masked verifier accepts at. Four of them hold at every
message: the caps' alignment (`zkSlotAlign`) and clash (`zkForkClash`), the pads' fork bound (`zkSimForkLePads`) and
the pads' view clash (`zkViewClashQP`). The fifth, the fork bound at `τ`, rests on `τ`'s alignment alone
(`zkSimForkLeTau_of_align`), so masked KS has the one premise `ZkTauAlign` (`zk_KS_of_tauAlign`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at the `τ` events**, from `ZkTauAlign` (`zkSimForkLeTau_of_align`). -/
theorem zkSimForkLeTauZ_of_align {x : Claims.Msg} (ha : ZkTauAlign x) : ZkSimForkLeOn x fun _ i => IsTauZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((_ | (_ | r)) | ⟨⟩)
  · exact False.elim hi
  · exact False.elim hi
  · exact zkSimForkLeTau_of_align ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ ⟨r, rfl⟩
  · exact False.elim hi

end Gen

/-- **The fork bound at the `τ` events**, from `ZkTauAlign` (`zkSimForkLeTau_of_align`). -/
theorem zkSimForkLeTauZ_of_align {x : Claims.Msg} (ha : ZkTauAlign x) : ZkSimForkLeOn x fun _ i => IsTauZ i :=
  Gen.zkSimForkLeTauZ_of_align (S := S₀) (x := x) (ha := ha)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked verifier: KS from the `τ` alignment**, at SHA-512 at the trimmed record at the deployed
relation: `zk_KS_of_slot` at `zkSlotAlign`, `zkForkClash`, `zkSimForkLePads` and `zkViewClashQP`. -/
theorem zk_KS_of_tauAlign (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkTauAlign x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_slot (fun x _ => zkSlotAlign x) (fun x _ => zkForkClash x)
    (fun x hno => zkSimForkLeTauZ_of_align (ha x hno)) (fun x _ => zkSimForkLePads x) (fun x _ => zkViewClashQP x)

end Gen

/-- **The deployed masked verifier: KS from the `τ` alignment**, at SHA-512 at the trimmed record at the deployed
relation: `zk_KS_of_slot` at `zkSlotAlign`, `zkForkClash`, `zkSimForkLePads` and `zkViewClashQP`. -/
theorem zk_KS_of_tauAlign (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkTauAlign x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_tauAlign (S := S₀) (ha := ha)

end FlockDeploy.KS
