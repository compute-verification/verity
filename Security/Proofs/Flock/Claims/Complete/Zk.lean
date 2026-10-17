import Proofs.Flock.Claims.Complete.Zk.Tape
import Proofs.Flock.Claims.Complete.Zk.Piop
import Proofs.Flock.Claims.Complete.Zk.Ligerito
import Proofs.Flock.Claims.Complete.Zk.Inner
import Proofs.Flock.Claims.Complete.Zk.Session
import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.Complete.Zk.Sim
import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Claims.Complete.Zk.Verdict
import Proofs.Flock.Claims.Complete.Zk.Skeleton
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.Complete.Zk.Link
import Proofs.Flock.Claims.Complete.Zk.Json.QSort
import Proofs.Flock.Claims.Complete.Zk.Json.Record
import Proofs.Flock.Claims.Complete.Zk.Json.Decode
import Proofs.Flock.Claims.Complete.Zk.Json.Session
import Proofs.Flock.Claims.Complete.Zk.Json.Goals
import Proofs.Flock.Claims.Complete.Zk.Json.Files

/-! Flock's completeness at the deployed masked (`--zk`) verifier: the honest masked prover (`Honest`: `honestPZk`, on
the masked tape `TapeZk` and pads `padsOf` of `Tape`, the masked zerocheck and lincheck of `Piop`, level 0's masked
Ligerito of `Ligerito`, the pads' commitment and inner proof of `Inner`, the session of `Session`), its run against
`V true` (`Run`: `run_acc_of_zk`), that run as a pure session (`Sim`: `honestRunZk_eq`), `δcZk H` with the composition
into `Claims.Flock.Complete` (`Coins`: `pr_badZk_le`, `complete_zk_of`), and the masked verdict accepting when each of
its checks passes (`Verdict`: `zk_verifyRepWith_ok`, `zk_verifyWith_ok`). The skeleton (`Skeleton`:
`complete_zk_of_goals`) composes them from one named goal per check (`GoalsZk`), proving the keyed coin tree from
`readyZk` (`coinTree_goal`); `Discharge` proves those that follow
from the scope and the session's shape (`tablesZk_goal`, `schedulesZk_goal`, `proofCountZk_goal`), and `Link` the first
send (`linkFirstZk_goal`). `Json.Record` is the masked record's JSON (`recordJson_eqZk`) and the masked decoder's hello
and coin-tree checks passing on it (`checkHello_ok`, `checkRecord_ok`, with `Json.QSort`'s `qsort_coinTree`), after
which it decodes as the plain record of `specZ` (`decodeWith_specZ`); `Json.Decode` reads it back: decoded, live, and
its draw (`decode_recordObjZk`, `live_recordObjZk`, `recordDraw_recordObjZk`); `Json.Session` is what the one-table
masked session returns and sends (`sessionPZk_facts`), its reps' rounds their sends (`repPZk_rec`); `Json.Goals` proves
the record's goals from `StmtOkZk` and `ProverOkZk` (`RunZk.json`, `decodesZk_goal`, …, `repExtrasZk_goal`);
`Json.Files` the proof files' goals, given `RepShapedZk` (`RunZk.rep`, `repDecodesZk_goal`, `repParamsZk_goal`,
`padsRoot_goal`, `capCommittedZk_goal`, and `finalFitsZk_goal` given `RecordFitsZk`). -/
