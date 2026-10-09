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
import Proofs.Flock.Recursive.VStarStrata
import Proofs.Flock.Recursive.VStarU2
import Proofs.Flock.Recursive.Universal
import Proofs.Flock.Recursive.UniversalClass

/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered netlist's in `Private`), its soundness at the compiled inner
protocol with the compile lemma open (`Compile`), V*'s laws from their executable strata (`VStarStrata`) and those
strata's tiling from the verifier's U2 check (`VStarU2`), the compile premise at a class whose schedule is
`fast100`'s (`Universal`), and the universal unit's class of private circuits, its descriptions read in gate order
(`UniversalClass`). -/
