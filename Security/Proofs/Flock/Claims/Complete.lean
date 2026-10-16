import Proofs.Flock.Claims.Complete.Honest
import Proofs.Flock.Claims.Complete.Run
import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Complete.Verdict

/-! Flock's completeness at the deployed plain verifier: the honest prover (`Honest`: `honestP`, on the session prover
of `Prover/`: the transcript, the PIOP, Ligerito and the session record), its run against the verifier (`Run`:
`run_acc_of`, acceptance from the honest run's shape and its last message's checks), that run as a pure session
(`Sim`: `honestRun_eq`), and `δc` with the composition into `Claims.Flock.Complete` (`Coins`: `pr_bad_le`,
`complete_of`), the prover's messages read back (`Encode`: `linkOf_linkBody`, `finalOf_encode`), and the verdict and
B's checks accepting when each of their checks passes (`Verdict`: `verifyRepWith_ok`, `verifyWith_ok`, `checks_ok`). -/
