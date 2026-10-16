import Proofs.Flock.Claims.Complete.Honest
import Proofs.Flock.Claims.Complete.Run
import Proofs.Flock.Claims.Complete.Sim

/-! Flock's completeness at the deployed plain verifier: the honest prover (`Honest`: `honestP`, on the session prover
of `Prover/`: the transcript, the PIOP, Ligerito and the session record), its run against the verifier (`Run`:
`run_acc_of`, acceptance from the honest run's shape and its last message's checks), and that run as a pure session
(`Sim`: `honestRun_eq`). -/
