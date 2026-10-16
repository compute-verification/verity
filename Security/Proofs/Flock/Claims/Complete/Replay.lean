import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.Claims.Complete.Replay.Zerocheck
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Claims.Complete.Replay.ZerocheckGoal
import Proofs.Flock.Claims.Complete.Replay.Lincheck
import Proofs.Flock.Claims.Complete.Replay.RingSwitch
import Proofs.Flock.Claims.Complete.Replay.Trace
import Proofs.Flock.Claims.Complete.Replay.Satisfies
import Proofs.Flock.Claims.Complete.Replay.Ligerito
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra
import Proofs.Flock.Claims.Complete.Replay.RingSwitchAlgebra
import Proofs.Flock.Claims.Complete.Replay.LigeritoModel
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal

/-! The verifier's phases replayed on the honest prover's rounds: the steps of both sides (`Steps`: the prover's in the
pure session, `squeezeP_go`, `forIn_go`; the verifier's on rounds that carry the prover's, `squeezeWith_sent`, `Agrees`),
the zerocheck (`Zerocheck`: the prover's in closed form, `zerocheckP_go`, and the verifier's on its rounds,
`zerocheckWith_of`; `ZerocheckAlgebra`: on an R1CS-valid trace whose outer coins avoid 1 the residual claim ends at
final_a · final_b, `fold_final`; `ZerocheckGoal`: the skeleton's `Zerocheck` from R1CS-valid traces and the session
walk's alignment of each rep, `zerocheck_goal`, no outer coin `1` at a good case, `Run.outer_ne_one`, and a returning
rep's zerocheck, `repP_zerocheck`), the lincheck (`Lincheck`: the prover's in closed form, `lincheckP_go`, the verifier's
on its rounds, `lincheckWith_of`, and the skeleton's `Lincheck` given the lincheck's algebra, `lincheck_goal`), and the
ring switch (`RingSwitch`: the prover's in closed form, `ringSwitchP_go`, the verifier's on its rounds,
`ringSwitchWith_of`, and the skeleton's `RingSwitch` given each claim's switched vector meets its value,
`ringSwitch_goal`), and the honest trace at the deployed relation (`Trace`: the witness's bits are the trace, which
satisfies the block R1CS by `RelS`, `traceR1CS`, so the goals hold there without it, `zerocheck_goal'`,
`lincheck_goal'`, `ringSwitch_goal'`; `Satisfies`: the case's table is `stmtOf`'s, its matrices `matA`, `matB`,
satisfied by the witness's bits, `Case.model`; `LincheckAlgebra`: the lincheck's algebra there from `RelS`, the folded
circuit the halved fold with `β` at the pin, `folded_ok`, the start `α·v_A + v_B + β = Σ comb·zr` and the honest rounds,
`lcAlgebra`, so `Lincheck` holds without it, `lincheck_relation`; `RingSwitchAlgebra`: the ring switch's algebra there
from `RelS`, every claim's value the witness's extension at its point, lincheck's `w`, `Lc.ab_eval`, zerocheck's
`P_c(ζ)`, `Zc.c_claim_eval`, and the extra claims at link points of `m_pts` coordinates, `Region.regionClaim_holds`,
`rsAlgebra`, so `RingSwitch` holds without it, `ringSwitch_relation`), and Ligerito (`LigeritoModel`: the verifier's
steps on rounds that carry a model's coins, `Runs`, composing by `Runs.bind` and `Runs.loop`; `LigeritoRun`: the
verifier's Ligerito in closed form, `out`, and its return on the model's rounds given the Merkle openings and the final
identity, `ligeritoWith_of`; `LigeritoGoal`: the skeleton's `Ligerito` at the rep's coins and decoded sizes, given the
session walk's alignment of each rep's Ligerito, `LgAt`, each level's Merkle opening, `LgMerkle`, and the final
identity, `LgFinal`, `ligerito_goal`, and `Consumed` once each rep's stream holds exactly the model's rounds, `LgCount`,
`consumed_goal`). -/
