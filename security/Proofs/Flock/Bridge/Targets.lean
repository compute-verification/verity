module

public import Proofs.Flock.Bridge.Statement
public import Proofs.Flock.EndToEndHidden.Statement
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

/-! The bridge's lemmas as `Prop` statements that plug into `SampledKSLive` (targets, not proved). -/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit

def FaultyRightTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {Pos Val Opn Com : Type}
    (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com)
    (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (roots : List Claims.Msg)
    (Xr : Fin C.N → Bool) (κ : Fin C.N → Option Bool) (u : Fin n),
    (∃ op : Pos → Opn, ∀ g ∈ P.io u, com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g) →
    (∀ g ∈ P.io u, κ g = none → ∀ o : Opn, com o = cs (pos g) →
      ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, bit (val o) g, d⟩ = true) →
    (∀ g ∈ P.io u, ∀ b, κ g = some b → C.eval Xr g = b) →
    ∀ (ns : List ℕ) (r : Claims.Sampling.OneStage) (desc : Claims.Msg) (q : Claims.Sampling.Query), ns.sum = n →
      u.val ∉ Claims.Sampling.faulty H Cm (stmtOf P κ r desc q roots) (partOf P κ ns) (witDep C Cm H roots Xr)

def PartBlocksTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (ns : List ℕ), ns.sum = n → (partOf P κ ns).blocks.length = n

def PaTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (ns : List ℕ), ns.sum = n → ∀ (r : Claims.Sampling.OneStage) (B : Finset ℕ), B ⊆ Finset.range n →
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns B

def CountTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (ns : List ℕ), ns.sum = n →
    ∀ (kd K₀ : ℕ), 1 ≤ kd → kd ≤ n →
    ∀ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (κ : Fin C.N → Option Bool)
      (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) (w : Claims.Sampling.Wit),
      ¬ Claims.Sampling.RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ))
          (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w →
      min K₀ (n - kd + 1) ≤
        ((Claims.Sampling.faulty H Cm (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w).filter (· < n)).card

def AssemblyTarget : Prop :=
  TransportsLive → FaultyRightTarget → PartBlocksTarget → CountTarget → Flock.Guarantees.EndToEndHidden →
    Flock.Guarantees.SampledKSLive

def RelDeployTarget : Prop :=
  ∀ {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (cd : SamplingDeploy.Code)
    {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) (ns : List ℕ)
    (r : Claims.Sampling.OneStage) (κ : Fin C.N → Option Bool) (desc : Claims.Msg) (q : Claims.Sampling.Query)
    (roots : List Claims.Msg) (Xr : Fin C.N → Bool) (δ : ℝ) (x : Claims.Msg),
    LayoutIs C rows c pub nl roots cd →
    cd.parse x = some (stmtOf P κ r desc q roots) → cd.partition (stmtOf P κ r desc q roots) = some (partOf P κ ns) →
    ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg (commitDep C rows c pub nl roots) δ (stmtOf P κ r desc q roots)
      (partOf P κ ns) (witDep C (commitDep C rows c pub nl roots) Claims.Sampling.sha512Msg roots Xr) →
    ¬ Claims.Sampling.R Claims.Sampling.sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep C (commitDep C rows c pub nl roots) Claims.Sampling.sha512Msg roots Xr)

end Flock.Bridge
