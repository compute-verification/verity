import Proofs.Flock.VBridge.Sha
import Proofs.Flock.VBridge.Algebra.Structure
import Proofs.Flock.VBridge.Algebra.Sim
import Proofs.Flock.VBridge.Algebra.Build
import Proofs.Flock.VBridge.Algebra.Eval
import Proofs.Flock.VBridge.Algebra.Zerocheck
import Proofs.Flock.VBridge.Algebra.Lincheck
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.VBridge.Algebra.Ligerito

/-! The bridge from V*, the C-Flock verifier as a circuit, to C-Flock's verifier of record: V*'s gadgets compute what the
verifier of record computes. -/
