import Definitions.Core.Circuit
import Definitions.Core.Digest
import Definitions.Core.Partition
import Definitions.Core.Annotation
import Definitions.Core.Advice
import Definitions.Core.PrivatePartition
import Definitions.Core.Declaration
import Definitions.Core.Boolean
import Definitions.Core.Game
import Definitions.Core.Draw
import Definitions.Core.Challenger
import Definitions.Core.Decisive
import Definitions.Core.Scalar
import Definitions.Core.Prims
import Definitions.Core.TC
import Definitions.Core.Fp32
import Definitions.Core.Formalism
import Definitions.Core.Expand
import Definitions.Core.NaiveStrassen
import Definitions.Core.Site

/-!
# Core: the model every area states its guarantees in

The developer registers a circuit `D` with its registered information `θ` (`Declaration`), runs `D` and commits the
values of its committed wires. The verifier's live coins (`Game`) come from the challenger's log (`Challenger`) and
select a sample of proof blocks by a sampling design (`Draw`). Every area states its guarantees about these objects
and defines none of its own: a circuit and its trace, a partition into proof blocks and a proof block's correctness are
the ones here.

* Circuits: `Circuit` (gates, traces, `Implements`), `Boolean` (the Boolean profile), `Expand` (a multi-bit circuit as
  a Boolean one), `Digest`.
* Partitions: `Partition` (committed wires, a proof block's correctness), `Annotation`, `PrivatePartition`,
  `Formalism` (`(c, k)`-partitions and local description length).
* What the developer commits: `Declaration` (the registered circuit and `θ`), `Advice` (the advice root), `Site` (the
  registered hardware and its registration).
* Coins and soundness: `Game` (`Sound`, live-coin interactions), `Draw` (sampling designs), `Challenger`, `Decisive`.
* Word semantics: `Scalar`, `Prims`, `Fp32`, `TC` (the tensor cores); `NaiveStrassen` (two tilings in the tensor-core
  model).

The guarantees are in `Guarantees` (read `Verity.Guarantees.SampleAfterRecorded` and `Verity.Guarantees.SoundBatch`
first) and the named assumptions in `Assumptions`. This module, `Assumptions` and `Guarantees` are the spec: they import
only one another and Mathlib, so their diff is everything a statement reviewer reads. The proofs and their lemmas are in
`Security/Proofs/Core/`.
-/
