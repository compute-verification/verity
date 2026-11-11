module

public import Proofs.Flock.Claims.Recursion.NextRHonest
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
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# An honest rep from the completeness model's shape

`HonestRep`'s lengths and levels are what `FlockComplete`'s `repShaped` gives the honest rep's proof: `ProofShaped` at
the shape the verifier decodes it at. A shaped proof decodes from its file (`decodeProof_encodeProof`), so it has the
lengths a decoded proof has (`decodeProof_sizes`), and its schedule has at least two levels. Its nonces are zero, as
`Lengths` has them since the verifier refuses any other (R4), so with the unabsorbed fields as V\* takes them it is an
honest rep (`honestRep_of_shaped`).
-/

@[expose] public section

namespace FlockDeploy

open Flock

/-- **A shaped proof is an honest rep**, at `2 + e` claims, where its `params`, `initialCap` and `finalC` are V\*'s: its
nonces are zero (`ZkView.NoncesZero`, the last of its `Lengths`), the only ones the verifier reads (R4). -/
theorem honestRep_of_shaped {c : RepCtx} {p : Flock.Proof} {e : ℕ} (hn : c.n = 2 + e)
    (hp : FlockVBridge.Family.ProofShaped (FlockSoundness.Refine.shapeOf c.st c.sch e) p)
    (hpa : p.params = pcsParams c.st.merkle c.sch) (hic : p.ligerito.initialCap = p.cap)
    (hfc : p.zerocheck.finalC = dot (lagrange nodesL den6Inv c.z) p.zerocheck.round1c) : HonestRep c p := by
  have hL : 1 ≤ c.sch.levels.size - 1 := hp.2.1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hz⟩ := hp.2.2.2
  exact ⟨⟨e, hn, FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _
    (FlockVBridge.Family.decodeProof_encodeProof _ _ hp)⟩, by omega, hz, hpa, hic, hfc⟩

end FlockDeploy
