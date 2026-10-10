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
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.Description

/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered private circuit's in `Private`), its soundness at the compiled inner
protocol with the compile lemma open (`Compile`), V*'s laws from their executable strata (`VStarStrata`) and those
strata's tiling from the verifier's U2 check (`VStarU2`), the compile premise at a class whose schedule is
`fast100`'s (`Universal`), the universal unit's class of private circuits, its descriptions read in gate order
(`UniversalClass`), the class, regions and schedule of an executable statement (`ExecClass`), the inner statement's
description read from its own rows, which are the registered ones or SHA-512 collides (`Description`), V*'s sessions
as the prover supplies them, typed where they type and refusing elsewhere, with the prover's strategy carried across
(`Raw`), and where V*'s statement enters a session, with the law's population V*'s units (`Loads`). -/
