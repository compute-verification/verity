module

public import Definitions.Core.Circuit
public import Definitions.Core.Digest
public import Definitions.Core.Partition
public import Definitions.Core.Annotation
public import Definitions.Core.Advice
public import Definitions.Core.PrivatePartition
public import Definitions.Core.Declaration
public import Definitions.Core.Boolean
public import Definitions.Core.Game
public import Definitions.Core.Draw
public import Definitions.Core.Challenger
public import Definitions.Core.Decisive
public import Definitions.Core.Scalar
public import Definitions.Core.Prims
public import Definitions.Core.TC
public import Definitions.Core.Fp32
public import Definitions.Core.Formalism
public import Definitions.Core.Expand
public import Definitions.Core.NaiveStrassen
public import Definitions.Core.Site

/-!
# Core's protocol: the model every protocol states its guarantees in

The developer declares a circuit `D` and its baseline `θ` (`Declaration`), runs `D` and commits the values of its
committed wires, and the verifier draws with live coins (`Game`), which it reads from a challenger's log
(`Challenger`), by a draw law over units (`Draw`). Every protocol package states its guarantees about
these objects and defines none of its own: a circuit, its transcripts, a partition into units and a unit's correctness
are the ones here.

This module, `Assumptions` and `Guarantees` are the trusted text: they import only one another and Mathlib, so their
diff is everything a statement reviewer reads. `SecurityProofs` holds the proofs and every lemma, which nobody reviews
for meaning.
-/

@[expose] public section
