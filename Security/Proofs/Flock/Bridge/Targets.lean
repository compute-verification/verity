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

/-!
# The bridge's lemmas, as statements (targets, not proved)

Each is a `def … : Prop` over `Bridge.Defs` and `Bridge.Deployed`, so a proof plugs into `SampledKSLive` as stated:

* `FaultyRightTarget` (M1's core): a unit whose committed wires each carry, in some opening of their registered commit
  string, their value in the evaluation on `Xr`'s inputs (`RightRegZ`'s second conjunct), and each of whose registered
  openings at a wire the constants `κ` don't fix opens its wire's claims position under `Cm` (`TransportsLive`'s
  conclusion), and whose evaluation is `κ`'s constant at each constant wire, is not faulty in `witDep Xr`, at every
  design and every strata of the population. So `faulty ⊆ wrongRegZ`.
* `PartBlocksTarget`: `partOf P κ ns` has `n` slots, one per unit, each the unit's proof block with the input rows it
  owns, at strata `ns` that sum to `n`.
* `PaTarget` (M4): on `partOf P κ ns`, a one-stage design `r` misses a set `B` of population slots with `r`'s own miss on
  the strata `ns`.
* `CountTarget` (M4): outside `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` under `subset kd`, at least `min K₀ (n − kd + 1)`
  population slots are faulty.
* `AssemblyTarget` (M4): `TransportsLive`, those and `EndToEndHidden` give `SampledKSLive`.
* `RelDeployTarget` (rel_deploy): at the deployed sampling record (`SamplingDeploy.deployOf`) of a verifier's code
  whose layout is `commitDep`'s (`LayoutIs`) and which parses the statement's bytes to `stmtOf` and partitions it to
  `partOf`, a `witDep` outside `RelS` at SHA-512 and `commitDep` is outside the record's `R`.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit

/-- **M1's core**: a unit whose committed wires each carry, in some opening of their registered commit string, their
value in `C`'s evaluation on `Xr`'s inputs, each of whose registered openings at a wire the constants `κ` don't fix
opens its wire's position under `Cm` and `roots`, and whose evaluation is `κ`'s constant at each constant wire it
reads, is not faulty in `witDep C Cm H roots Xr` at the statement at constants `κ`, under any design `r` and strata
`ns` of the population. -/
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

/-- `partOf P κ ns` has one slot per unit, at strata that sum to `n`. -/
def PartBlocksTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (ns : List ℕ), ns.sum = n → (partOf P κ ns).blocks.length = n

/-- **The Pa lemma**: on `partOf P κ ns`, at strata that sum to `n`, the one-stage design `r` misses a set `B` of
population slots with its own miss on the strata `ns`, `r.Pa ns B`; under `subset kd`, `C(n − |B|, kd)/C(n, kd)`. -/
def PaTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (ns : List ℕ), ns.sum = n → ∀ (r : Claims.Sampling.OneStage) (B : Finset ℕ), B ⊆ Finset.range n →
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns B

/-- **Outside `RelS` is a count**: under `subset kd`, at `δ = C(n − K₀, kd)/C(n, kd)`, `1 ≤ kd ≤ n`, a witness outside
`RelS` has at least `min K₀ (n − kd + 1)` faulty population slots, slots below `n`, whatever the strata. -/
def CountTarget : Prop :=
  ∀ {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (ns : List ℕ), ns.sum = n →
    ∀ (kd K₀ : ℕ), 1 ≤ kd → kd ≤ n →
    ∀ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (κ : Fin C.N → Option Bool)
      (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) (w : Claims.Sampling.Wit),
      ¬ Claims.Sampling.RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ))
          (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w →
      min K₀ (n - kd + 1) ≤
        ((Claims.Sampling.faulty H Cm (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w).filter (· < n)).card

/-- **The assembly**: the registered openings opening the deployed commitment, M1's core, the slot count and the count
lemma turn `EndToEndHidden` into `SampledKSLive`. -/
def AssemblyTarget : Prop :=
  TransportsLive → FaultyRightTarget → PartBlocksTarget → CountTarget → Flock.Guarantees.EndToEndHidden →
    Flock.Guarantees.SampledKSLive

/-- **rel_deploy**: at the deployed sampling record of a verifier's code whose layout is `commitDep`'s, and which
parses `x` to `stmtOf` and partitions it to `partOf`, a `witDep` outside `RelS` at SHA-512 and `commitDep` is outside
the record's `R`. At the key program of a circuit in scope, the partition hypothesis holds where the deployed partition
is `Spot.FilePart.ofCircuit`'s blocks in the registration's strata (`Bridge.relDeploy_ofCircuit`, from
`FilePartTarget` and the scope check's `unitLog ≤ 32`), which the v2 audit's population check
(`Spot.FilePart.check`, `Bridge.check_ok`) holds to the circuit file's units in order. -/
def RelDeployTarget : Prop :=
  ∀ {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (cd : SamplingDeploy.Code)
    {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ) (ns : List ℕ) (r : Claims.Sampling.OneStage)
    (κ : Fin C.N → Option Bool) (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg)
    (Xr : Fin C.N → Bool) (δ : ℝ) (x : Claims.Msg),
    LayoutIs C rows c nl roots cd →
    cd.parse x = some (stmtOf P κ r desc q roots) → cd.partition (stmtOf P κ r desc q roots) = some (partOf P κ ns) →
    ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg (commitDep C rows c nl roots) δ (stmtOf P κ r desc q roots)
      (partOf P κ ns) (witDep C (commitDep C rows c nl roots) Claims.Sampling.sha512Msg roots Xr) →
    ¬ Claims.Sampling.R Claims.Sampling.sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep C (commitDep C rows c nl roots) Claims.Sampling.sha512Msg roots Xr)

end Flock.Bridge
