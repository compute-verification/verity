import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Recursive.Fork
import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Recursive.Finds
import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Recursive.Schedule
import Proofs.Flock.Recursive.Target
import Proofs.Flock.Recursive.Holo
import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Stage
import Proofs.Flock.Recursive.StageRead
import Proofs.Flock.Recursive.Registrants
import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Recursive.OsSalts
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Audit
import Proofs.Flock.Recursive.Universal
import Proofs.Flock.Recursive.UniversalClass
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.OuterAbort
import Proofs.Flock.Recursive.Description
import Proofs.Flock.Recursive.Sessions
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.FirewallRun
import Proofs.Flock.Recursive.FirewallFollows
import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.PublicAuditProof
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.StageKinds
import Proofs.Flock.Recursive.HoloCoins
import Proofs.Flock.Recursive.HoloSched
import Proofs.Flock.Recursive.HoloSchedRead
import Proofs.Flock.Recursive.ForkLast
import Proofs.Flock.Recursive.ForkInnerC
import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.RegExtract
import Proofs.Flock.Recursive.AuditC3Proof
import Proofs.Flock.Recursive.Witness

/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), hidden unit circuits' class,
registered table and registered schedule, with the holographic table at them (`Holo`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered private circuit's in `Private`), its zero knowledge at the operating
system's salts (`OsSalts`), its soundness at the compiled inner
protocol with the compile lemma open (`Compile`), the compile premise at a class whose schedule is `fast100`'s
(`Universal`), the universal unit's class of private circuits, its descriptions read in gate order
(`UniversalClass`), the class, regions and schedule of an executable statement (`ExecClass`), the inner statement's
description read from its own rows, which are the registered ones or SHA-512 collides (`Description`), V*'s sessions
as the prover supplies them, typed where they type and refusing elsewhere, with the prover's strategy carried across
(`Raw`), where V*'s statement enters a session (`Loads`), the verifier's check that a session's circuit file is one of
V*'s (`VStarCheck`, which `RecursiveAudit` doesn't read yet), V*'s outer sessions with the honest prover's self-check
in them, which never stops (`OuterAbort`), and V*'s outer sessions (`Sessions`) and their zero knowledge, (ii)'s
`_hOuter`, at the honest prover's code, its self-check included (`OuterZK`), and (ii) stated about the firewall's code
and V*'s, with no zero knowledge of V*'s sessions taken (`Firewall`), its proof (`FirewallZK`), `FirewallSends`'
template half proved about the firewall's code, what it sends at every salt list fixed by one run (`FirewallRun`), the
honest worker's half left an obligation (`FirewallFollows`), and the recursive audit at public units, the inner
statement the verifier's own from its partition object and program (`PublicAudit`), its proof
(`PublicAuditProof`), the same audit at candidate 3's table (`HoloExec`, `AuditC3`, its proof `AuditC3Proof` from the
pieces under `Proofs.Flock.C3` and gap 2 at the word the prover's runs open, `RegExtract`), and
`flock-witness`'s checks of recorded sessions, each the model's predicate (`Witness`). -/
