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
import Proofs.Flock.Claims.Complete.Zk.Json.Runs
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.ProverOk
import Proofs.Flock.Claims.Complete.Zk.RepRounds
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Zk.Consumed
import Proofs.Flock.Claims.Complete.Zk.Transfer
import Proofs.Flock.Claims.Complete.Zk.Walk
import Proofs.Flock.Claims.Complete.Zk.Replay
import Proofs.Flock.Claims.Complete.Zk.PiopGo
import Proofs.Flock.Claims.Complete.Zk.ReplayPos
import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Extras
import Proofs.Flock.Claims.Complete.Zk.LigeritoPos
import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Claims.Complete.Zk.InnerMerkle
import Proofs.Flock.Claims.Complete.Zk.InnerGo
import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Claims.Complete.Zk.Stops
import Proofs.Flock.Claims.Complete.Zk.ConsZerocheck
import Proofs.Flock.Claims.Complete.Zk.ConsLincheck
import Proofs.Flock.Claims.Complete.Zk.ConsRingSwitch
import Proofs.Flock.Claims.Complete.Zk.ConsReplay
import Proofs.Flock.Claims.Complete.Zk.ConsCoins
import Proofs.Flock.Claims.Complete.Zk.MaskRank
import Proofs.Flock.Claims.Complete.Zk.Layout
import Proofs.Flock.Claims.Complete.Zk.RankBridge
import Proofs.Flock.Claims.Complete.Zk.RankDual
import Proofs.Flock.Claims.Complete.Zk.RankPrimal
import Proofs.Flock.Claims.Complete.Zk.RankDeployed
import Proofs.Flock.Claims.Complete.Zk.Complete
import Proofs.Flock.Claims.Complete.Zk.Level0Pad
import Proofs.Flock.Claims.Complete.Zk.Level0Run
import Proofs.Flock.Claims.Complete.Zk.Level0Lanes
import Proofs.Flock.Claims.Complete.Zk.Level0Shape
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.Zk.Level0Model
import Proofs.Flock.Claims.Complete.Zk.Level0Honest
import Proofs.Flock.Claims.Complete.Zk.Level0Opens
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo

/-! Flock's completeness at the deployed masked (`--zk`) verifier at one table (`ZkTables` lifts it to any number, at
the masked record's prover `ZkTables.honestPR`): the honest masked prover (`Honest`: `honestPZk`, on
the masked tape `TapeZk` and pads `padsOf` of `Tape`, the masked zerocheck and lincheck of `Piop`, level 0's masked
Ligerito of `Ligerito`, the pads' commitment and inner proof of `Inner`, the session of `Session`), its run against
`V true` (`Run`: `run_acc_of_zk`), that run as a pure session (`Sim`: `honestRunZk_eq`), `δcZk H` with the composition
into `Claims.Flock.Complete`'s statement at `honestPZk` (`CompleteWith`; `Coins`: `pr_badZk_le`, `complete_zk_of`),
and the masked verdict accepting when each of its checks passes (`Verdict`: `zk_verifyRepWith_ok`,
`zk_verifyWith_ok`). The skeleton (`Skeleton`: `complete_zk_of_goals`) composes them from one named goal per check
(`GoalsZk`), proving the keyed coin tree from `readyZk` (`coinTree_goal`); `Discharge` proves those that follow
from the scope and the session's shape (`tablesZk_goal`, `schedulesZk_goal`, `proofCountZk_goal`), and `Link` the first
send (`linkFirstZk_goal`). `Json.Record` is the masked record's JSON (`recordJson_eqZk`) and the masked decoder's hello
and coin-tree checks passing on it (`checkHello_ok`, `checkRecord_ok`, with `Json.QSort`'s `qsort_coinTree`), after
which it decodes as the plain record of `specZ` (`decodeWith_specZ`); `Json.Decode` reads it back: decoded, live, and
its draw (`decode_recordObjZk`, `live_recordObjZk`, `recordDraw_recordObjZk`); `Json.Session` is what the one-table
masked session returns and sends (`sessionPZk_facts`), its reps' rounds their sends (`repPZk_rec`); `Json.Goals` proves
the record's goals from `StmtOkZk` and `ProverOkZk` (`RunZk.json`, `decodesZk_goal`, …, `repExtrasZk_goal`), and
`Json.Stmt` `StmtOkZk` from the case's setup (`stmtOkZk`, with `Same.good` and `good_subset`);
`Json.Files` the proof files' goals, given `RepShapedZk` (`RunZk.rep`, `repDecodesZk_goal`, `repParamsZk_goal`,
`padsRoot_goal`, `capCommittedZk_goal`, and `finalFitsZk_goal`, the record's length from `readyZk`); `Json.Runs` the
session running, given `ProverRunsZk` (`sessionRunsZk_goal`); `Count` the sends fitting the rounds (`sendCountZk_goal`);
`ProverOk` `ProverOkZk` from `Out64` and its rep half `RepRoundsZk` (`proverOkZk`), and `RepRounds` that half
(`repRoundsZk_goal`); `Lanes` level 0's lane phase from the masked Ligerito returning (`laneConstraints_goal`);
`Consumed` every recorded round consumed where the rep's steps return (`consumedZk_goal`); `Transfer` a replay
returning on any rounds that carry the prover's, whatever the hashes (`replayWith_xfer`); `Walk` the session rep by
rep, each stream and proof file the rounds and proof of the rep that made it (`sessionPZk_walk`, `RunZk.walk`);
`Replay` the masked replay passing, since the prover ran it on rounds its stream carries (`replay_goal`); `PiopGo` the
masked prover's zerocheck and lincheck in closed form (`zerocheckPZk_go`, `lincheckPZk_go`); `ReplayPos` where a
masked replay that returns stops and the coins its claims read (`replayWith_ret`); `RingSwitch` the ring switch
passing on the replay's claims, given the extra claims' algebra `RsExtrasZk` (`ringSwitchZk_goal`); `Extras` that
algebra at the deployed relation, from `RelS` (`CaseZk.model`, `rsExtrasZk`); `LigeritoPos` what a masked Ligerito
that returns gives, by coin position: its record and where it stops (`zkLigeritoWith_pos`, `RunZk.lg_pos`), and
`LigeritoProverPos` what the masked prover records, by send position (`ligeritoPZk_pos`); `PadAlgebra` the honest pads'
algebra: at a rep's pads the triple's constraints vanish at every
sacrifice coin, so the batched constraint `innerP` checks holds at every coin once the rep's own constraints vanish
(`triple_holds`, `innerP_batched`), and the verifier's batched and column checks hold on the honest `Y`
(`innerY_batched`, `innerY_column`); `InnerMerkle` the pads' commitment opening at every column, each column's siblings
climbing from its leaf to the root (`innerCommitP_tree`, `inner_column_opens`); `InnerGo` the masked prover's inner
proof in closed form at the replies' coins, where the batched constraint holds on the pads and `β ≠ 0` (`innerP_go`),
and the verifier's inner check passing on it, on rounds carrying its sends, at a rep's pads whose constraints vanish
(`verifyInnerWith_of`); `Exact` a masked rep that returns sending exactly its rounds, its pair coins independent, its
`β ≠ 0` and its rank check passed (`repPZk_exact`), and `Stops` a masked rep that stops, stopping right after the send
whose reply it refused (`repPZk_stop`, `repPZk_rho_stop`, `repPZk_beta_stop`); `ConsZerocheck`, `ConsLincheck` and `ConsRingSwitch` the replay's message-dependent rows
vanishing at the pads, from the plain algebra moved by the masks (`zcZ_masks`, `lcZ_masks`), and `ConsReplay` the five
together on `Zk.replay`'s output wherever the plain checks pass at its coins (`replay_rows_zero`), and `ConsCoins`
those coins the prover's on rounds carrying its sends, with the rows as `ConsHold` (`replay_coins`,
`replay_consHold_agrees`); `MaskRank` the mask slot's span, the prover's rank check over the slot's words
(`RankFull`, `maskRankOk`, `maskRankOk_iff`), a rep's claim points from its coins (`claimPoints`, `claimsAt`), the
slot's size (`maskWords_ge`: an in-scope masked table has at least `1024` words), and the rank bound the prover's
refusal needs (`DeployedRankBound`; at the verifier's coins `MaskRankBound`, a premise of `complete_zk`); `Layout`
the layout every table of a masked session in scope has (`layoutOk_of_scope`); `RankBridge` `MaskRankBound` from
`DeployedRankBound` and the small slots' `SmallSlotBound` (`maskRankBound_of`), which holds since the verifier refuses
a slot under `2^13` bits (`smallSlotBound`); `RankDual` the bound's dual side, the deficient coins summed over the
killing functionals' supports (`card_deficient_le`); `RankPrimal` its primal side, the spans' sizes step by step
(`sum_env_le`, `chain`); `RankDeployed` `DeployedRankBound` from the two (`deployedRankBound`), so `MaskRankBound`
(`maskRankBound`); `Complete`
assembles them into `complete_zk`, from those premises and the goals not yet proved, and `complete_zk_relation` at
the deployed relation.
`Level0Pad` is level 0's masked query identity: a row's padded lanes combine to the blinded table's, the padding
cancelling (`row_comb`, `consE0_honestZk`); `Level0Run` the masked Ligerito prover returning where its pair's coins are
independent (`ligeritoPZk_ret`); `Level0Lanes` its lane phase in closed form (`ligeritoPZk_lanes`); `Level0Shape` its
proof's lengths (`ligeritoPZk_shape`); `Level0Premises` the premises of `ZkTables.Level0` proved (`proverRuns`,
`proverShape`, `lanes`); `Level0Model` the masked verifier's Ligerito returning on rounds carrying its closed-form
model, given its openings and final identity at the model's values (`ligeritoWithZ_of`); `Level0Honest` the honest
masked prover's Ligerito against that model (`mid_lv0Z`); `Level0Opens` its openings passing the masked Merkle
checks, level 0's over rows `EXTRA_LANES` wider (`ligeritoPZk_opens`, `merkleOkZ_of_opens`). -/
