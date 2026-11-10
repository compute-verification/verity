module

public import Proofs.Flock.Claims.DeployEfficient
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.RunsExp
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Universal
public import Proofs.Flock.SamplingBoundary
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
# The sampling block's claims, with what remains as hypotheses

`sampling_assembled`: at a sampling deployment `D`, at every hash `H`, the conjunction of the sampling claims below
but `ZK`, which the deployed record's theorems add (`SamplingDeploy.sampling_deployed`). Knowledge soundness, its count
form, bits and efficiency are the compiled sampled verifier's at the deployed record (`SamplingDeploy.ks_compiled`, at
`Claims.SamplingGeneric.Compiled.KS`), from Flock's KS at its sessions. The claims are:

* `Complete` and `ZK`: hypotheses;
* `ScopeMatches`, from the deployed design's well-formedness (`scopeMatches_of`, `designWellFormed`) where the scope
  check is the readable scope's program part;
* `Universal`, from the unit's Definition (`Universal_of`);
* `Binding`, at hm96 leaves under the registered frame-v3-sha512 path check (`Binding_of_hm96`); at a commitment of
  any form, a hypothesis (`sampling_assembled_bind`);
* `FaultyBoundary` (`faultyBoundary`).

## What remains: the hypotheses

* `hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer` and `hrow : RowPlaces row`: the
  deployed trace commitment's opening program, hm96 leaves of rows that place the opened bit under the verifier's own
  registered path check; at the deployed record by definition and proved (`SamplingDeploy.deploy_commit`,
  `SamplingDeploy.row_places`).
* `DrawSlots` (`Sampling.Select.DrawLaw`): on every statement that parses and whose query evaluates, the partition and
  design are the verifier's own: the partition has the shape of its layout, and the design is the one `Slots.select`
  samples, from uniform bytes of the coins, every slot checked where it is refused; the Lean sampled-proofs verifier's
  run, which builds `D`.
* `ProgramScope` (`Sampling.Select.Scope`): the scope check is the readable scope's program part (a topologically
  ordered program, partitioned into proof blocks). The design part is a theorem (`designWellFormed`); the program part
  waits on the claims' program and the verifier's being one. The claims are over word gates: the program is the
  statement's bit gates (`prog`), the Boolean lowering of its word gates as the statement brings it, and a `Q_word` unit
  is the set of its word gates' bit gates, so no Lean lowering of the served primitives enters. The verifier's reading
  of the descriptor is total (Flock's verifier has no `partial` def); what remains is its map from each word gate to
  that word gate's bit gates in `prog`, under which its `Q_word` evaluation is `partition`.
* `huniv : UniversalIs D`: the deployed universal unit computes what its Definition's gates do (`universalProg`, the
  port of `verity.ml.private_circuits.universal`'s `build`); the code side's binding of the compiled unit. That those
  gates compute `PrivateCircuit.eval` is proved (`universal_correct`); the deployed record's unit is the Definition's
  (`SamplingDeploy.universalIs_deploy`).
* `hComplete : Complete H D`: of the record's reference prover (`D.P`) at its error (`D.δc H`); at the deployed record,
  from Flock's masked completeness and the sampled verifier's and the code's facts (`SamplingDeploy.complete_deploy`).
* `hZK : ZK H (D.mask Nz Ωz tapeOf S) εzk γzk`, at the deployed record: the zero knowledge of the record's reference
  prover on the masked tapes `tapeOf (ν, r)`, with the leaf layer's simulator `S`, its binding term `γzk` times
  `Fork`'s collisions among the honest prover's calls in two real runs on one nonce; `Sampling.ZK` reduces it to a
  committed-coin protocol's models (`SamplingDeploy.zk_deploy_of_models_gap`).
-/

@[expose] public section

namespace Claims.Sampling

/-- **`D` masked**: the record with its reference prover's tape read off a nonce and the rest (`tapeOf`), the same at
every statement, and the simulator `S`. -/
def Deploy.mask {Coin Col T : Type} (D : Deploy Coin Col T) (Nonce Rest : Type) [Fintype Nonce] [Nonempty Nonce]
    [Fintype Rest] [Nonempty Rest] (tapeOf : Nonce × Rest → D.Tape) (S : SimLeak) : ZkDeploy Coin Col T :=
  { toDeploy := D, Nonce := fun _ => Nonce, Rest := fun _ => Rest, tapeOf := fun _ => tapeOf, S := S }

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at a deployment whose commitment binds** (`hbind`), at every hash: `sampling_assembled`'s
conclusion, at a commitment of any form. -/
theorem sampling_assembled_bind {T : Type}
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg) (hbind : Binding D)
    (hslots : DrawSlots D) (hprog : ProgramScope D) (huniv : UniversalIs D) (hComplete : Complete H D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ Complete H D :=
  ⟨scopeMatches_of D hslots hprog, hbind, Universal_of huniv, faultyBoundary, hComplete⟩

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at a deployment**, at every hash: the readable scope is the verifier's; the trace commitment
binds; the universal unit runs its private circuit; the incorrect set is the boundary form's; and completeness. The
hypotheses are what remains (the module docstring says what discharges each). -/
theorem sampling_assembled {T : Type}
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hslots : DrawSlots D) (hprog : ProgramScope D) (huniv : UniversalIs D) (hComplete : Complete H D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ Complete H D :=
  sampling_assembled_bind D H (Binding_of_hm96 D own at_ row salt π layer hcommit hrow) hslots hprog huniv hComplete

/-- **The masked record's decoder lists within the radius**: it is the masked radius decoder
(`FlockDeploy.decode_zk`, `FlockDeploy.listWithin_zk`). -/
theorem listWithin_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Flock.ListWithin (FlockDeploy.deployZk rel).len (FlockDeploy.deployZk rel).enc (FlockDeploy.deployZk rel).θ
      (FlockDeploy.deployZk rel).decode := by
  rw [show (FlockDeploy.deployZk rel).decode = Claims.GS.zkDecodeRD from FlockDeploy.decode_zk rel]
  exact FlockDeploy.listWithin_zk rel

/-- **The trimmed plain record's decoder lists within the radius**: it is the plain radius decoder
(`FlockDeploy.decode_plain`, `FlockDeploy.listWithin_plainTrim`). -/
theorem listWithin_deployPlainTrim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Flock.ListWithin (FlockDeploy.deployPlainTrim rel).len (FlockDeploy.deployPlainTrim rel).enc
      (FlockDeploy.deployPlainTrim rel).θ (FlockDeploy.deployPlainTrim rel).decode := by
  rw [FlockDeploy.decode_plain rel]
  exact FlockDeploy.listWithin_plainTrim rel

end Claims.Sampling
