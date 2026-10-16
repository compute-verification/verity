import Proofs.Flock.Claims.Complete.Honest
import Proofs.Flock.Claims.Complete.Run

/-! Flock's completeness at the deployed plain verifier: the honest prover (`Honest`: `honestP`, on the session prover
of `Prover/`: the transcript, the PIOP, Ligerito and the session record), and its run against the verifier (`Run`:
`run_acc_of`, acceptance from the honest run's shape and its last message's checks). -/
