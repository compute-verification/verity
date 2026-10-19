import Proofs.Flock.Claims.Complete.Honest
import Proofs.Flock.Claims.Complete.Run
import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.Ready
import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Complete.Verdict
import Proofs.Flock.Claims.Complete.Skeleton
import Proofs.Flock.Claims.Complete.Discharge
import Proofs.Flock.Claims.Complete.Shape
import Proofs.Flock.Claims.Complete.Link
import Proofs.Flock.Claims.Complete.Count
import Proofs.Flock.Claims.Complete.Json
import Proofs.Flock.Claims.Complete.Custody
import Proofs.Flock.Claims.Complete.Files
import Proofs.Flock.Claims.Complete.Runs
import Proofs.Flock.Claims.Complete.Reps
import Proofs.Flock.Claims.Complete.Zk
import Proofs.Flock.Claims.Complete.Replay
import Proofs.Flock.Claims.Complete.Walk
import Proofs.Flock.Claims.Complete.WalkPacked
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Complete.Zk

/-! Flock's completeness at the deployed plain verifier: the honest prover (`Honest`: `honestP`, on the session prover
of `Prover/`: the transcript, the PIOP, Ligerito and the session record), its run against the verifier (`Run`:
`run_acc_of`, acceptance from the honest run's shape and its last message's checks), that run as a pure session (`Sim`:
`honestRun_eq`), the record's error `δc` and where it is `1` (`Ready`: `ready`, `δc`, `δcZk`, and the masked record's
`ZkTables.ready`, `ZkTables.δc`), its bound and the
composition into `Claims.Flock.Complete` (`Coins`: `pr_bad_le`, `complete_of`, and `Complete_meets_plain`,
`Complete_meets_zk`, the error below `1`), the prover's messages read back (`Encode`: `linkOf_linkBody`,
`finalOf_encode`), and the verdict and B's checks accepting when each of their checks passes (`Verdict`:
`verifyRepWith_ok`, `verifyWith_ok`, `checks_ok`). The skeleton (`Skeleton`: `complete_of_goals`) composes them, from
one named goal per check (`Goals`); `Discharge` proves the goals that follow from the scope alone (`tables_goal`,
`schedules_goal`), `Shape` those that read what the pure session returns (`Ret`, `proofCount_goal`), `Link` the first
send (`linkFirst_goal`), and `Count` the sends (`sendCount_goal`). `Replay` replays the verifier's phases on the honest
prover's rounds (`zerocheckWith_of`, `lincheckWith_of`, `ringSwitchWith_of`, and the goals `zerocheck_goal`,
`lincheck_goal`, `ringSwitch_goal`). `Shaped` proves each proof an honest rep returns fits the shape the verifier
decodes it at (`repShaped`). `Walk` aligns every rep's recorded stream with its run (`Run.walkAt`) and composes the
goals into `Claims.Flock.Complete` at the deployed relation from named premises (`complete_plain`). `WalkPacked` runs
every rep on the trace's packed table (`Run.repWalkP`), so the premise `LgFinal` rests on the honest Ligerito's final
identity alone (`lgFinal_walk`, `complete_plain_honest`), which holds (`Lg.honest`), so completeness rests on the
record's JSON alone (`complete_plain_json`), beside the oracle's 64-byte outputs that `Claims.Complete` reads itself. -/
