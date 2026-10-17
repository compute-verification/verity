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
import Proofs.Flock.Claims.Complete.Replay.Walk
import Proofs.Flock.Claims.Complete.Replay.LigeritoModel
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal
import Proofs.Flock.Claims.Complete.Replay.LigeritoProver
import Proofs.Flock.Claims.Complete.Replay.LigeritoRep
import Proofs.Flock.Claims.Complete.Replay.LigeritoMerkle
import Proofs.Flock.Claims.Complete.Replay.LigeritoWalk
import Proofs.Flock.Claims.Complete.Replay.RepRounds
import Proofs.Flock.Claims.Complete.Replay.ProverOk
import Proofs.Flock.Claims.Complete.Replay.LigeritoFinal
import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis
import Proofs.Flock.Claims.Complete.Replay.LigeritoSum
import Proofs.Flock.Claims.Complete.Replay.LigeritoSync
import Proofs.Flock.Claims.Complete.Replay.LigeritoCons
import Proofs.Flock.Claims.Complete.Replay.LigeritoWeight
import Proofs.Flock.Claims.Complete.Replay.LigeritoTrack
import Proofs.Flock.Claims.Complete.Replay.LigeritoHonest
import Proofs.Flock.Claims.Complete.Replay.LigeritoHonestRun

/-! The verifier's phases replayed on the honest prover's rounds: the steps of both sides (`Steps`: the prover's in the
pure session, `squeezeP_go`, `forIn_go`; the verifier's on rounds that carry the prover's, `squeezeWith_sent`,
`Agrees`), the zerocheck (`Zerocheck`: the prover's in closed form, `zerocheckP_go`, and the verifier's on its rounds,
`zerocheckWith_of`; `ZerocheckAlgebra`: on an R1CS-valid trace whose outer coins avoid 1 the residual claim ends at
final_a · final_b, `fold_final`; `ZerocheckGoal`: the skeleton's `Zerocheck` from R1CS-valid traces and the session
walk's alignment of each rep, `zerocheck_goal`, no outer coin `1` at a good case, `Run.outer_ne_one`, and a returning
rep's zerocheck, `repP_zerocheck`), the lincheck (`Lincheck`: the prover's in closed form, `lincheckP_go`, the
verifier's on its rounds, `lincheckWith_of`, and the skeleton's `Lincheck` given the lincheck's algebra,
`lincheck_goal`), and the ring switch (`RingSwitch`: the prover's in closed form, `ringSwitchP_go`, the verifier's on
its rounds, `ringSwitchWith_of`, and the skeleton's `RingSwitch` given each claim's switched vector meets its value,
`ringSwitch_goal`), and the honest trace at the deployed relation (`Trace`: the witness's bits are the trace, which
satisfies the block R1CS by `RelS`, `traceR1CS`, so the goals hold there without it, `zerocheck_goal'`,
`lincheck_goal'`, `ringSwitch_goal'`; `Satisfies`: the case's table is `stmtOf`'s, its matrices `matA`, `matB`,
satisfied by the witness's bits, `Case.model`; `LincheckAlgebra`: the lincheck's algebra there from `RelS`, the folded
circuit the halved fold with `β` at the pin, `folded_ok`, the start `α·v_A + v_B + β = Σ comb·zr` and the honest rounds,
`lcAlgebra`, so `Lincheck` holds without it, `lincheck_relation`; `RingSwitchAlgebra`: the ring switch's algebra there
from `RelS`, every claim's value the witness's extension at its point, lincheck's `w`, `Lc.ab_eval`, zerocheck's
`P_c(ζ)`, `Zc.c_claim_eval`, and the extra claims at link points of `m_pts` coordinates, `Region.regionClaim_holds`,
`rsAlgebra`, so `RingSwitch` holds without it, `ringSwitch_relation`; `Walk`: the statement's circuit fold returns at
every rep, `Run.fold_ok`, so the three goals hold there from the session walk alone, `ZcAt`, `LcWalk` and `RsAt`,
`replay_goals`), and Ligerito (`LigeritoModel`: the verifier's
steps on rounds that carry a model's coins, `Runs`, composing by `Runs.bind` and `Runs.loop`; `LigeritoRun`: the
verifier's Ligerito in closed form, `out`, and its return on the model's rounds given the Merkle openings and the final
identity, `ligeritoWith_of`; `LigeritoGoal`: the skeleton's `Ligerito` at the rep's coins and decoded sizes, given the
session walk's alignment of each rep's Ligerito, `LgAt`, each level's Merkle opening, `LgMerkle`, and the final
identity, `LgFinal`, `ligerito_goal`, and `Consumed` once each rep's stream holds exactly the model's rounds, `LgCount`,
`consumed_goal`; `LigeritoProver`: the honest prover's Ligerito ends where the model does on the proof it returns,
`ligeritoP_go`; `LigeritoRep`: a returning rep's transcript ends at the model's, `repP_ligerito`, so `LgAt` and
`LgCount` follow from the recorded stream carrying the rep's rounds, `Run.lgAt_of`, `Run.lgCount_of`, and lincheck's
circuit fold returns at a returning rep, `Run.lcFold_of`; `LigeritoMerkle`: a committed table's Merkle openings pass,
`commitP_tree`, `merkle_open`, the honest prover opens at the model's query rounds, `ligeritoP_opens`, and so every
level's Merkle check passes, `merkleOk_of_opens`; `LigeritoWalk`: a returning rep opens honestly, `repP_opens`, so
`LgMerkle` follows from the session walk at each rep, `RepWalk`, and the schemes' widths, `SchemeWidths`,
`lgMerkle_of_walk`, and `Ligerito` and `Consumed` rest on those and `LgFinal`, `ligerito_goal_walk`,
`consumed_goal_walk`; `LigeritoFinal`: a returning rep runs the honest Ligerito on its table at ring switching's `T` and
basis, `repP_lig`, so `LgFinal` holds at every rep the walk runs on the trace's packed table, `RepWalkP`, given the
honest Ligerito's final identity, `Lg.Honest`, ring switching's claim on the packed table, `RsSum`, and the verifier's
basis as the batched basis's extension, `BasisOk`, `lgFinal_of`; `LigeritoBasis`: every claim a rep opens has `m − 6`
coordinates, `claims_xsize`, so the verifier's basis is the extension of the prover's batched basis on the cube,
`Lg.fOf_basisVals`, `Lg.basis_bHat`, and `BasisOk` holds at the deployed relation, `basisOk_relation`; `LigeritoSum`: a
bit of a sum is the sum of the bits, `Lg.bF_bit_sumF`, so ring switching's `T` is the packed table's sum against the
batched basis, `Lg.rsSum_eq`, `RsSum` holds at the deployed relation, `rsSum_relation`, and `LgFinal` there rests on
`RepWalkP` and `Lg.Honest`, `lgFinal_relation`; `LigeritoSync`: every step of the verifier's model keeps its claim the
honest prover's sum of table against weight, `Lg.Sync`: a challenge, `Lg.Sync.chal`, one that splits,
`Lg.Sync.chalSplit`, an out-of-domain or consistency claim, `Lg.Sync.ood`, `Lg.Sync.add`, and the final check's sum is
the claim, `Lg.lhsE_honest`;
`LigeritoCons`: on rows the prover committed, the consistency value is its F-table folded
at the level's challenges against its consistency weight, `Lg.consE_honest`, `Lg.consE_fold`, the prover's folds are the
model's, `Lg.gv_foldKs`, and a level's consistency claim keeps `Lg.Sync`, `Lg.Sync.close`; `LigeritoWeight`: the
prover's weight is the model's at every stage: level 0's, `Lg.gv_w0`, after level 0's challenges, `Lg.fold_init`, on a
split table, `Lg.gv_liftU_preW`, with an out-of-domain or consistency claim on it, `Lg.gv_addOod`, `Lg.gv_addCons` (the
consistency weight being the model's, `Lg.embed_consW`), and the final residual weights are the model's, `Lg.gOf_resE`,
the last weight plus the last consistency claim, `Lg.res_honest`; `LigeritoTrack`: the prover's steps on the model's
with what they do to its table and weight, `Lg.HRuns`, composing by `Lg.HRuns.bindD` and `Lg.HRuns.loop`, and against
the model's own steps a challenge, `Lg.HRuns.challenge`, an out-of-domain intro, `Lg.HRuns.oodIntro`, and a level's
query, `Lg.HRuns.query`; `LigeritoHonest`: the claim and the weight kept through the model's steps, `Lg.sync_close`,
`Lg.WOK`, `Lg.wok_level`, the challenge loops, `Lg.HRuns.initChals`, `Lg.HRuns.levelChals`, and the prover between
levels, `Lg.Mid`, reached after level 0, `Lg.mid_lv0`, kept by an inner level's step, `Lg.mid_stepN`, from which the
last level meets the final identity, `Lg.goal_last`; `LigeritoHonestRun`: the honest Ligerito meets it, `Lg.honest`, so
`Lg.Honest` holds, and at the deployed relation `LgFinal` rests on `RepWalkP` alone, `lgFinal_walkP`, and `Ligerito` and
`Consumed` on the walk's facts and the schemes' widths, `ligerito_walkP`, `consumed_walkP`), and the prover's pieces
(`RepRounds`: a returning rep records a round, each of at most `65536` coins, `repP_repOk`, every squeeze of the models
being few coins, `Bnd`; `ProverOk`: the case's table has a registered scheme and at most 1,024 regions, `Case.schemeOk`,
so at an oracle of 64-byte outputs its digests have the scheme's width, `opsWith_widths`, `schemeWidths`, its level-0
cap is non-empty and under `2^64` bytes, `commitP_cap`, and `ProverOk` holds, `proverOk`). -/
