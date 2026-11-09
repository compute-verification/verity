module

public import Definitions.Core.Circuit
public import Definitions.Core.Digest
public import Definitions.Core.Partition
public import Definitions.Core.Annotation
public import Definitions.Core.Advice
public import Definitions.Core.PrivatePartition
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Boolean
public import Definitions.Core.Game
public import Definitions.Core.Uniform
public import Definitions.Core.Draw
public import Definitions.Core.Challenger
public import Definitions.Core.Decisive
public import Definitions.Core.Scalar
public import Definitions.Core.Prims
public import Definitions.Core.TC
public import Definitions.Core.Fp32
public import Definitions.Core.Formalism
public import Definitions.Core.Expand
public import Definitions.Core.Expand.Instances
public import Definitions.Core.NaiveStrassen
public import Definitions.Core.Site
public import Definitions.Core.ValueSalts
public import Definitions.Core.MemCheck

/-!
# Core: the model every area states its guarantees in

The developer registers a circuit `D` with its registered information `θ` (`RegisteredCircuit`), runs `D` and commits
the values of its committed wires. The verifier's live coins (`Game`) come from the challenger's log (`Challenger`) and
select a sample of proof blocks by a sampling design (`Draw`). Every area states its guarantees about these objects
and defines none of its own: a circuit and its trace, a partition into proof blocks and a proof block's correctness are
the ones here.

* Circuits: `Circuit` (gates, traces, `Implements`), `Boolean` (the Boolean profile), `Expand` (a multi-bit circuit as
  a Boolean one), `Digest`.
* Partitions: `Partition` (committed wires, a proof block's correctness), `Annotation`, `PrivatePartition`,
  `Formalism` (`(c, k)`-partitions and local description length).
* What the developer commits: `RegisteredCircuit` (the registered circuit and `θ`), `Advice` (the advice root), `Site`
  (the registered hardware and its registration).
* Coins and soundness: `Game` (`Sound`, live-coin interactions), `Draw` (sampling designs), `Challenger`, `Decisive`.
* Memory checking: `MemCheck` (a multiset's fingerprint and a key's print over a finite field).
* Word semantics: `Scalar`, `Prims`, `Fp32`, `TC` (the tensor cores); `NaiveStrassen` (two tilings in the tensor-core
  model).

The guarantees are in `Guarantees` (read `Verity.Guarantees.SampleAfterRecorded` and `Verity.Guarantees.SoundBatch`
first) and the named assumptions in `Assumptions`. This module, `Assumptions` and `Guarantees` are the spec: they import
only one another and Mathlib, so their diff is everything a statement reviewer reads. The proofs and their lemmas are in
`Security/Proofs/Core/`.
-/

@[expose] public section
