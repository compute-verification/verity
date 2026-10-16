import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.KS.Game
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.Cover
import Proofs.Flock.Claims.KS.Radius
import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Claims.KS.Trace

/-! Flock's knowledge soundness at `E^rep`, in parts: one wait truncated to `E^rep`'s `K` draws, by a coupling with `k`
independent accepted runs and a Chernoff bound (`Wait`, `ERep_trunc`); the soundness proof's escape events, each a
collision among the verifier's own calls in a fork or in one run, covered by `B` and gathered into one square root
(`Fork`, `Escapes.le`); `Flock.KS` from one wait's bound on the opened columns (`Flock`, `Flock.KS_of_oneWait`); at a
game's verifier (`Bridge.ofGame`), escapes tied to the verifier's log through the decision's calls (`Game`); the
deployed plain and masked instances, from one wait's bound (`Deploy`, `FlockDeploy.KS.plain_of_oneWait`,
`FlockDeploy.KS.zk_of_oneWait`), with `k ≥ 8` wherever the verifier accepts at SHA-512 (`Range`,
`FlockDeploy.KS.plain_of_oneWait_sha512`, `FlockDeploy.KS.zk_of_oneWait_sha512`); and extraction from a filled word
close to a codeword, with the coverage of `nMin` positions it needs and that a prover's accepted runs need not give
(`Cover`, `Flock.extracts_of_fill`, `Flock.acc_le_of_KS`, `FlockDeploy.KS.plain_acc_le_of_KS`).

The radius form (`Radius`): the extractor decodes the filled word, and the decoder lists every trace within `θ·N` of
it, per whole column (`Flock.ListCompleteθ`, `Flock.ExtractsFill`, `Flock.OneWaitθ`, `Flock.KSθ`); KS from one wait's
bound there (`Flock.KSθ_of_oneWaitθ`, `FlockDeploy.KS.plain_of_oneWaitθ_sha512`, `FlockDeploy.KS.zk_of_oneWaitθ_sha512`).
One wait's bound from a fixed table's bound (`Joint`): the old theorems' `joint_acc_le` on the deployed verifier's own
run law (`Flock.runGame`, `Flock.oneWait_lhs_eq`), with three obligations: a fixed table's bound over tables of
well-formed columns, accepted runs opening only well-formed columns, and conflicts tied to collisions
(`Flock.TableBoundθ`, `Flock.ColsOk`, `Flock.ConflictTied`, `Flock.oneWaitθ_of_tableBound`,
`FlockDeploy.KS.plain_KSθ_of_tableBound`, `FlockDeploy.KS.zk_KSθ_of_tableBound`). The old table at the filled word
(`Trace`): a word as the old oracle, `Close` bounding the distance per whole column (`Flock.hdist_le_of_close`), the
deployed plain code, read step, schedule and radius as the old model's, so an old `Committed` table extracts
(`FlockDeploy.KS.good_of_committed_plain`, `FlockDeploy.KS.not_committed_plain`), and the masked code and read step as
the old padded ones (`Flock.zkEnc_eq_encAt`, `Flock.zkReadD_eq_padWit`, `Flock.radius_zk`). -/
