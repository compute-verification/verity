import Proofs.Flock.VBridge.Sha
import Proofs.Flock.VBridge.Karatsuba
import Proofs.Flock.VBridge.Residuals
import Proofs.Flock.VBridge.Climb
import Proofs.Flock.VBridge.Open
import Proofs.Flock.VBridge.RecOpen
import Proofs.Flock.VBridge.Net
import Proofs.Flock.VBridge.Unit
import Proofs.Flock.VBridge.Keyed
import Proofs.Flock.VBridge.Inputs
import Proofs.Flock.VBridge.Verifier
import Proofs.Flock.VBridge.OfVStar

/-! The bridge from V*, the C-Flock verifier as a circuit, to C-Flock's verifier of record: V*'s gadgets compute what the
verifier of record computes. -/
