import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Recursive.Fork
import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Recursive.Finds
import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Recursive.Schedule
import Proofs.Flock.Recursive.Target
import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Stage
import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Audit
import Proofs.Flock.Recursive.ReadsOpen

/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered netlist's in `Private`), and its soundness at the compiled inner
protocol with the compile lemma open (`Compile`), and the registered reads read off an outcome, whose commit strings open
the registration's root when every session verifies (`Reads`, `ReadsOpen`). -/
