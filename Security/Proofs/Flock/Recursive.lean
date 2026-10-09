import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Recursive.Fork
import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Recursive.Schedule
import Proofs.Flock.Recursive.Target
import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Stage
import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Recursive.Private

/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), and the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered netlist's in `Private`). -/
