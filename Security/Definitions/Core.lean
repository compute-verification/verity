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
import Definitions.Core.Decisive
import Definitions.Core.Scalar
import Definitions.Core.Prims
import Definitions.Core.TC
import Definitions.Core.Fp32
import Definitions.Core.Formalism
import Definitions.Core.Expand
import Definitions.Core.NaiveStrassen

/-!
# Core's protocol: the model every protocol states its guarantees in

The developer declares a circuit `D` and its baseline `θ` (`Declaration`), runs `D` and commits the values of its
committed wires, and the verifier draws from its own coins (`Game`), by a draw law over units (`Draw`). Every protocol package states its guarantees about
these objects and defines none of its own: a circuit, its transcripts, a partition into units and a unit's correctness
are the ones here.

This module, `Assumptions` and `Guarantees` are the trusted text: they import only one another and Mathlib, so their
diff is everything a statement reviewer reads. `SecurityProofs` holds the proofs and every lemma, which nobody reviews
for meaning.
-/
