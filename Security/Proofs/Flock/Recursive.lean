import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Recursive.Fork
import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Recursive.Finds
import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Recursive.Schedule
import Proofs.Flock.Recursive.Target
import Definitions.Flock.Recursive.Holo
import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Stage
import Proofs.Flock.Recursive.StageRead
import Proofs.Flock.Recursive.Registrants
import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Recursive.OsSalts
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.Audit
import Proofs.Flock.Recursive.SoundExec
import Proofs.Flock.Recursive.Universal
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Written
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.OuterAbort
import Proofs.Flock.Recursive.Description
import Proofs.Flock.Recursive.Sessions
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.OuterSessionZK
import Proofs.Flock.Recursive.FirewallRun
import Proofs.Flock.Recursive.FirewallTemplate
import Proofs.Flock.Recursive.FirewallLive
import Proofs.Flock.Recursive.FirewallAccepts
import Proofs.Flock.Recursive.FirewallWire
import Proofs.Flock.Recursive.FirewallJson
import Proofs.Flock.Recursive.FirewallAny
import Proofs.Flock.Recursive.FirewallAdaptive
import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.PublicAuditProof
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.FirewallObj
import Proofs.Flock.Recursive.FirewallSteps
import Proofs.Flock.Recursive.FirewallHonest
import Proofs.Flock.Recursive.StageKinds
import Proofs.Flock.Recursive.HoloCoins
import Proofs.Flock.Recursive.HoloSched
import Proofs.Flock.Recursive.HoloSchedRead
import Proofs.Flock.Recursive.HoloSchedCheck
import Proofs.Flock.Recursive.HoloRepCoins
import Proofs.Flock.Recursive.HoloMap
import Proofs.Flock.Recursive.ForkLast
import Proofs.Flock.Recursive.ForkInnerC
import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.RegExtract
import Proofs.Flock.Recursive.AuditC3Proof
import Proofs.Flock.Recursive.Witness
import Proofs.Flock.Recursive.TopRoot
import Proofs.Flock.Recursive.TopFinder
import Proofs.Flock.Recursive.SaltFinder
import Proofs.Flock.Recursive.ValueSrc
import Proofs.Flock.Recursive.ValueFinder
import Proofs.Flock.Recursive.ValueSource
import Proofs.Flock.Recursive.ValueBridge
import Proofs.Flock.Recursive.LeafFinder
import Proofs.Flock.Recursive.ValueChecked
import Proofs.Flock.Recursive.ValueInner
import Proofs.Flock.Recursive.PartTie
import Proofs.Flock.C3.RegLevels.VRows
import Proofs.Flock.C3.RegLevels.OutRegs
import Proofs.Flock.Recursive.HPortTie
import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Recursive.KSNontrivial
import Proofs.Flock.Recursive.KS
import Proofs.Flock.Recursive.KSV
import Proofs.Flock.Recursive.KSDeployed
import Proofs.Flock.Recursive.KSProved
import Proofs.Flock.Recursive.KSCoinTreeProved
import Proofs.Flock.Recursive.KSCoinTreeWitnesses
import Proofs.Flock.Recursive.KSLinked
import Proofs.Flock.Claims.Recursion.DeployedCounts
import Proofs.Flock.Claims.Recursion.LamDeployedTight
import Proofs.Flock.Claims.Recursion.KappaEpsDeployed
import Proofs.Flock.Recursive.BitsK
import Proofs.Flock.Claims.Recursion.KappaEpsTight
import Proofs.Flock.Recursive.BitsKTight
import Proofs.Flock.Claims.Recursion.AcceptedScope
import Proofs.Flock.Claims.Recursion.InnerSetupCount
import Proofs.Flock.Claims.Recursion.IVStarCalls
import Proofs.Flock.Recursive.Loop
import Proofs.Flock.Recursive.LoopZK
import Proofs.Flock.Recursive.FirewallSalted
import Proofs.Flock.Recursive.LoopAny
import Proofs.Flock.Recursive.InnerRows
import Proofs.Flock.Recursive.LinksOuter
import Proofs.Flock.Recursive.LinksDeployed
/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), hidden unit circuits' class,
registered table and registered schedule, with the holographic table at them (`Holo`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered private circuit's in `Private`; what a stop leaks, the counting lemma
and a recursive session's stops, in `StopLeak`), its zero knowledge at the operating
system's salts (`OsSalts`), its soundness at the compiled inner
protocol with the compile lemma open (`Compile`), the compile premise at a class whose schedule is `fast100`'s
(`Universal`), the universal unit's class of private circuits, its descriptions read in gate order
(`UniversalClass`), the class, regions and schedule of an executable statement (`ExecClass`), the inner statement's
description read from its own rows, which are the registered ones or SHA-512 collides (`Description`), V*'s sessions
as the prover supplies them, typed where they type and refusing elsewhere, with the prover's strategy carried across
(`Raw`), every outcome of V*'s sessions a transcript they write (`Written`), where V*'s statement enters a session
(`Loads`), the verifier's check that a session's circuit file is one of
V*'s (`VStarCheck`, which `RecursiveAudit` doesn't read yet), V*'s outer sessions with the honest prover's self-check
in them, which never stops (`OuterAbort`), and V*'s outer sessions (`Sessions`) and their zero knowledge, (ii)'s
`_hOuter`, at the honest prover's code, its self-check included (`OuterZK`), and (ii) stated about the firewall's code
and V*'s, with no zero knowledge of V*'s sessions taken (`Firewall`), its proof (`FirewallZK`), the committed-coin
statements' `_hOuter` (`OuterSessionsZK`) at V*'s sessions on V*'s staging (`OuterSessionZK`) and at V*'s statements
run by the session loop's switched prover (`Loop`), `RecursiveZK` there with the session released at a stop the coins
fix, its simulator reading only V*'s public statements (`LoopZK`), `FirewallZK`'s bound when each digest reads the
salts the firewall answered before it (`FirewallSalted`), what the firewall's code sends at every salt list fixed by
one run (`FirewallRun`), a finished session's sends the schedule's transcript (`FirewallTemplate`), `flock-firewall
live`, the deployed send check, sending what the firewall's run sends (`FirewallLive`), and `FirewallSends` from them
with the honest worker's half left an obligation (`FirewallAccepts`, with (ii) from it and the live send check's sends
from it), the honest worker's requests and rounds read back by `decode`, `frames` and `split` as it meant them
(`FirewallWire`), a plain value's text parsing back to its canonical form with no fraction or exponent
(`FirewallJson`), the live firewall's sends at any requests (`FirewallAny`) and at any worker that reads its answers
(`FirewallAdaptive`), `RecursiveZK` at the loop at such a worker, with no `FirewallSends` (`LoopAny`), and the
recursive audit at public units, the inner
statement the verifier's own from its partition object and program (`PublicAudit`), its proof
(`PublicAuditProof`), the same audit at candidate 3's table (`HoloExec`, its coins `HoloCoins`, its schedule and the
verifier's map of its records `HoloSched` and `HoloMap`, `AuditC3`, its proof `AuditC3Proof` from the
pieces under `Proofs.Flock.C3` and gap 2 at the word the prover's runs open, `RegExtract`),
`flock-witness`'s checks of recorded sessions, each the model's predicate (`Witness`), and a unit's rows read off the
inner relation's witness, each the commit string its public file publishes (`InnerRows`). -/
