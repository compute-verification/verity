module

public import Proofs.Flock.Recursive.Defs
public import Proofs.Flock.Recursive.Compose
public import Proofs.Flock.Recursive.Fork
public import Proofs.Flock.Recursive.Sound
public import Proofs.Flock.Recursive.Multi
public import Proofs.Flock.Recursive.Finds
public import Proofs.Flock.Recursive.Flock
public import Proofs.Flock.Recursive.Schedule
public import Proofs.Flock.Recursive.Target
public import Definitions.Flock.Recursive.Holo
public import Proofs.Flock.Soundness.Defs
public import Proofs.Flock.Recursive.Transfer
public import Proofs.Flock.Recursive.Stage
public import Proofs.Flock.Recursive.StageRead
public import Proofs.Flock.Recursive.Registrants
public import Proofs.Flock.Recursive.Guarantees
public import Proofs.Flock.Recursive.OsSalts
public import Proofs.Flock.Recursive.Private
public import Proofs.Flock.Recursive.Compile
public import Proofs.Flock.Recursive.Audit
public import Proofs.Flock.Recursive.SoundExec
public import Proofs.Flock.Recursive.Universal
public import Proofs.Flock.Recursive.ExecClass
public import Proofs.Flock.Recursive.Written
public import Proofs.Flock.Recursive.Loads
public import Proofs.Flock.Recursive.VStarCheck
public import Proofs.Flock.Recursive.Raw
public import Proofs.Flock.Recursive.OuterAbort
public import Proofs.Flock.Recursive.Description
public import Proofs.Flock.Recursive.Sessions
public import Proofs.Flock.Recursive.OuterZK
public import Proofs.Flock.Recursive.Firewall
public import Proofs.Flock.Recursive.FirewallZK
public import Proofs.Flock.Recursive.OuterSessionZK
public import Proofs.Flock.Recursive.FirewallRun
public import Proofs.Flock.Recursive.PublicAudit
public import Proofs.Flock.Recursive.PublicAuditProof
public import Proofs.Flock.Recursive.HoloExec
public import Proofs.Flock.Recursive.StageKinds
public import Proofs.Flock.Recursive.HoloCoins
public import Proofs.Flock.Recursive.HoloSched
public import Proofs.Flock.Recursive.HoloSchedRead
public import Proofs.Flock.Recursive.HoloSchedCheck
public import Proofs.Flock.Recursive.HoloRepCoins
public import Proofs.Flock.Recursive.HoloMap
public import Proofs.Flock.Recursive.ForkLast
public import Proofs.Flock.Recursive.ForkInnerC
public import Proofs.Flock.Recursive.AuditC3
public import Proofs.Flock.Recursive.RegExtract
public import Proofs.Flock.Recursive.AuditC3Proof
public import Proofs.Flock.Recursive.Witness
public import Proofs.Flock.Recursive.TopRoot
public import Proofs.Flock.Recursive.TopFinder
public import Proofs.Flock.Recursive.SaltFinder
public import Proofs.Flock.Recursive.ValueSrc
public import Proofs.Flock.Recursive.ValueFinder
public import Proofs.Flock.Recursive.ValueSource
public import Proofs.Flock.Recursive.ValueBridge
public import Proofs.Flock.Recursive.LeafFinder
public import Proofs.Flock.Recursive.ValueChecked
public import Proofs.Flock.Recursive.ValueInner
public import Proofs.Flock.Recursive.PartTie
public import Proofs.Flock.C3.RegLevels.VRows
public import Proofs.Flock.C3.RegLevels.OutRegs
public import Proofs.Flock.Recursive.HPortTie
public import Proofs.Flock.Recursive.KSStatement
public import Proofs.Flock.Recursive.KSNontrivial
public import Proofs.Flock.Recursive.KS
public import Proofs.Flock.Recursive.KSV
public import Proofs.Flock.Recursive.KSDeployed
public import Proofs.Flock.Recursive.KSProved
public import Proofs.Flock.Recursive.KSLinked
public import Proofs.Flock.Recursive.KSCoinTreeProved
public import Proofs.Flock.Recursive.KSCoinTreeWitnesses
public import Proofs.Flock.Claims.Recursion.DeployedCounts
public import Proofs.Flock.Claims.Recursion.LamDeployedTight
public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed
public import Proofs.Flock.Recursive.BitsK
public import Proofs.Flock.Claims.Recursion.AcceptedScope
public import Proofs.Flock.Claims.Recursion.InnerSetupCount
public import Proofs.Flock.Claims.Recursion.IVStarCalls
public import Proofs.Flock.Recursive.InnerRows
public import Proofs.Flock.Recursive.LinksOuter
public import Proofs.Flock.Recursive.LinksDeployed
/-! The recursive audit: its model and soundness (`Defs` to `Stage`, with `ZkReg/`), hidden unit circuits' class,
registered table and registered schedule, with the holographic table at them (`Holo`), the proofs of its guarantees
(`Guarantees`; circuit privacy and the registered private circuit's in `Private`), its zero knowledge at the operating
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
statements' `_hOuter` (`OuterSessionsZK`) at V*'s sessions on V*'s staging (`OuterSessionZK`), `FirewallSends`'
template half proved about the firewall's code, what it sends at every salt list fixed by one run (`FirewallRun`), the
honest worker's half left an obligation (`FirewallFollows`), and the recursive audit at public units, the inner
statement the verifier's own from its partition object and program (`PublicAudit`), its proof
(`PublicAuditProof`), the same audit at candidate 3's table (`HoloExec`, its coins `HoloCoins`, its schedule and the
verifier's map of its records `HoloSched` and `HoloMap`, `AuditC3`, its proof `AuditC3Proof` from the
pieces under `Proofs.Flock.C3` and gap 2 at the word the prover's runs open, `RegExtract`),
`flock-witness`'s checks of recorded sessions, each the model's predicate (`Witness`), and a unit's rows read off the
inner relation's witness, each the commit string its public file publishes (`InnerRows`). -/

@[expose] public section
