import Definitions.Pouw.Fp8Atom.Atom
import Definitions.Pouw.Fp8Atom.E4M3
import Definitions.Pouw.Fp8Atom.Fp32
import Definitions.Pouw.Fp8Atom.H1T
import Definitions.Pouw.Fp8Atom.H1TLemmas
import Models.Pouw.Assumptions.Fp8Atom.H1TAssumptions
import Models.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions
/-!
# The FP8 atom's guarantees (trusted)

Each guarantee is a `def G : Prop`, with the docstring, binders and `open`s of the statement in its source
module. Its proof is the theorem `Pouw.SecurityProofs.Fp8Atom.G : Pouw.Guarantees.Fp8Atom.G`.
-/

namespace Pouw.Fp8Atom.H1T.Proofs
open Pouw.Fp8Atom Pouw.Fp8Atom.H1T

def _root_.Pouw.Guarantees.Fp8Atom.H1T.H1tDistinctLive : Prop :=
  ∀ (hR : H1TRunningWords),
    DistinctLiveH1T

end Pouw.Fp8Atom.H1T.Proofs

namespace Pouw.Fp8Atom.H1T.Proofs
open Pouw.Fp8Atom Pouw.Fp8Atom.H1T

def _root_.Pouw.Guarantees.Fp8Atom.H1T.H1tRunningOfDistinct : Prop :=
  ∀ (hD : DistinctLiveH1T),
    H1TRunningWords

end Pouw.Fp8Atom.H1T.Proofs
