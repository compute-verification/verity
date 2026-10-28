import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.ZkTables.Faithful
import Proofs.Flock.Claims.Complete.ZkTables.Ready
import Proofs.Flock.Claims.Complete.ZkTables.MaskFree
import Proofs.Flock.Claims.Complete.ZkTables.Run
import Proofs.Flock.Claims.Complete.ZkTables.Skeleton
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Discharge
import Proofs.Flock.Claims.Complete.ZkTables.Link
import Proofs.Flock.Claims.Complete.ZkTables.Count
import Proofs.Flock.Claims.Complete.ZkTables.Trace
import Proofs.Flock.Claims.Complete.ZkTables.Runs
import Proofs.Flock.Claims.Complete.ZkTables.Complete
import Proofs.Flock.Claims.Complete.ZkTables.One
import Proofs.Flock.Claims.Complete.ZkTables.Names
import Proofs.Flock.Claims.Complete.ZkTables.Session
import Proofs.Flock.Claims.Complete.ZkTables.Decode
import Proofs.Flock.Claims.Complete.ZkTables.Prover
import Proofs.Flock.Claims.Complete.ZkTables.Goals
import Proofs.Flock.Claims.Complete.ZkTables.Files
import Proofs.Flock.Claims.Complete.ZkTables.Reps
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Complete.ZkTables.RingSwitch
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.ZkTables.RepSplit
import Proofs.Flock.Claims.Complete.ZkTables.View
import Proofs.Flock.Claims.Complete.ZkTables.RepRuns
import Proofs.Flock.Claims.Complete.ZkTables.Parts
import Proofs.Flock.Claims.Complete.ZkTables.LaneRows
import Proofs.Flock.Claims.Complete.ZkTables.ReplayGo
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.ZkTables.ShapeGo
import Proofs.Flock.Claims.Complete.ZkTables.Round0
import Proofs.Flock.Claims.Complete.ZkTables.Pass
import Proofs.Flock.Claims.Complete.ZkTables.Refuse
import Proofs.Flock.Claims.Complete.ZkTables.Round1
import Proofs.Flock.Claims.Complete.ZkTables.Round2
import Proofs.Flock.Claims.Complete.ZkTables.Round3

/-! Flock's completeness at the deployed masked verifier at any number of tables. The honest masked prover runs each
table on its share of the witness (`Honest`: `traces`, `honestP`, which is `honestPZk` at one table); the record's
prover is `flock-circuit --zk`'s (`Faithful`: `honestPR`, `honestP` at the tape's mask words, `maskWOf`, stopped at an
outer coin `1`), whose run is `honestP`'s outside `BadZk` (`Ready`: `run_honestPR`), at a witness the deployed relation
still holds (`MaskFree`: `maskFree_relation`); the masked
verifier's error drops `readyZk`'s one-table condition and asks the regions and the honest record to fit (`Ready`:
`ready`, `RecordFitsAt`, `δc`, `complete_of`; `δc H` is `δcZk H` where both `readyZk` and `ready` hold); the honest run
is the pure masked session (`Run`: `run_acc_of`, `honestRun_eq`); the skeleton is `Zk/Skeleton.lean`'s at this run
(`Skeleton`: `complete_of_goals`, and the record's size from `ready`, `Run.recordFits`). The scope holds per table
(`Scope`: `Case.tables`, every table of `25 ≤ m ≤ 35` and `hm96-sha512/v1`, the spec naming one table per statement)
and discharges five goals at any number of tables (`Discharge`: `tables_goal`, `schedules_goal`, `proofCount_goal`;
`Link`: `linkFirst_goal`; `Count`: `sends_goal`, each table's two reps' rounds summed). At the deployed relation each
table's trace is its share's and satisfies its block R1CS (`Trace`: `Case.traceR1CS`, `traceR1CS_goal`, from
`Readable.relS_only`), which `tracesOf` reads only at one table. The session runs at a good case from the statement's
`hello` and the prover's pieces, its reps run in stream order (`Runs`: `Case.sessionRuns` from `StmtAt`,
`ProverRunsAt`; at several tables `sessionRunsJ_goal`). `Complete` composes them (`complete_of_at`; `complete`: from
`complete_zk`'s premises, which give every check at one table, and at several tables `StmtOkJ`,
`ProverRunsJ`, `RecordJ` and `OpenJ`). The bound at several tables loses nothing at one (`One`: `completeZk_of`,
`honestP`'s completeness with error `δc H` gives `honestPZk`'s with error `δcZk H`). Toward `RecordAt`, the tables'
names and the rep streams' are distinct (`Names`: `tableNames_nodup`, `repNames_nodup`, `rootBList_distinct`,
`Case.tables_nodup`), and the session returns the printed honest masked record of one `root_B` per table and sends
the link of them, then the streams' rounds (`Session`: `sessionPZk_factsJ`); that record decodes, each table's
`root_B` read under its name and each rep stream found under its name, reads live and carries the draw (`Decode`:
`decode_recordObjZkJ`, `live_recordObjZkJ`, `recordDraw_recordObjZkJ`). Every table's cap and reps are what the record
needs (`Prover`: `Case.points_regions`, `proverOkJ`), so the record's checks hold but what they ask of the proof files
(`Goals`: `Run.jsonJ`, `recordJsonJ`, `FilesAt`, `recordAt_of`, `recordJ_of`), and `complete` holds with `FilesJ` for
`RecordJ` (`completeJ`). Each table's reps' proof files decode to proofs of its parameters, its `root_B` as their cap
and a 64-byte pads' root (`Files`: `sessionPZk_filesJ`, `Run.repJ`), so `FilesAt` holds from each table's
`RepShapedZk` (`filesAt` from `RepShapedAt`, `filesJ` from `RepShapedJ`; `completeJR`). Each proof file is the proof
its rep returned, ending at its stream (`Reps`: `Run.walkJ`), so the masked replay passes (`replayJ`), level 0's lanes
follow from the masked Ligerito (`lanesJ`), and every round is consumed (`consumedJ`) at every table; so the rep's
checks reduce to what `complete_zk` takes at one table, per rep the ring switch, the masked Ligerito and the inner proof
(`RepChecksAt`, `openAt_of`; at several tables `RepChecksJ`, `openJ_of`, `completeJO`). None of these asks for several
tables, so completeness holds at every number of tables from the four per-table premises at every good case
(`completeAll`: `StmtAt`, `ProverRunsAt`, `RepShapedAt`, `RepChecksAt`), with no premise of `complete_zk`'s. The
statement's premise holds at every case of several tables (`Stmt`: `Case.stmtAt`, `stmtOkJ`, from the setup of several
tables, `setupTables_spec`, a printable subset or stratified draw with no closure, `good_subset`,
`closure_stratified`, and the masked `Hello`'s read-back, `hello_ok`), so
`completeJS` and `completeAllS` drop it, keeping `StmtOkZk` at one table. The ring switch replays at every table given
the extra claims' algebra (`RingSwitch`: `ringSwitchJ`, `RsExtrasAt`), which holds at the deployed relation from
`RelS`, each table satisfied by its share (`Case.model`, `Run.rsExtras`), so there completeness asks per rep only the
masked Ligerito and the inner proof (`LgInnerAt`; `completeRel`, and `completeJRel` from `complete_zk_relation`'s
premises at one table). The statement's premise holds at one table too (`StmtOne`: `stmt_of_one`, from `setupOne`'s
statement, its header's draw the statement's, a printable stratified or subset draw, `good_stratified`), so `StmtAt`
holds at every good case (`stmtAt_all`) and `StmtOkZk` holds (`stmtOkZk`), and `completeRelS`, `completeJRelS` and
`complete_zk_relationS` drop it. The four premises left each say something of one rep of one table, so they are
stated once at a table, its trace, its tape row, its domain and its coins from the rep's first send (`Table`:
`TableT`, `RepIn`, `CoinsT`, `MaskRankT`; `RepShapedT`, `ProverRunsT`, `LigeritoT`, `InnerT`), and give `RepShapedAt`,
`ProverRunsAt` and `LgInnerAt` at each table `i` of every good case, its rep `r` starting after the `2 i + r` earlier
reps (`repStartZk`; `Case.coinsT`, `Run.start_eq`), its rank check passing by `¬ BadZk` (`Case.maskRankT`) at its rep
`0`'s claim points read off rep `0`'s rounds (`repT_points`), so masked completeness at any number of tables follows
from the JSON specification, those four and the mask-rank bound (`complete_jsonT`); at table `0` of a good masked
case or run (`CaseZk.toCase`, `RunZk.toRun`) they give `complete_zk`'s `RepShapedZk`, `ProverRunsZk`, `LigeritoZk` and `Inner`
(`repShapedZk_of`, `proverRunsZk_of`, `lgInnerZk_of`). Each of the four splits into level 0's masked Ligerito,
stated once (`Level0`: `ProverRuns`, `ProverShape`, `Lanes`, `Verifier`), and the rest (`Parts`); a returning rep runs
stage by stage (`RepSplit`: `repPZk_split`), so given `Level0.Verifier` the verifier's view of it is the prover's
(`View`: `view`), and `LigeritoT` and `InnerT` hold (`ligeritoRest`, `innerRest`; `complete_jsonT_of_parts`). A
returning rep records `repRoundsT` rounds (`RepRuns`: `rep_rounds`), level 0's two lane rows vanish at the honest pads
where its lane phase is in closed form (`LaneRows`: `lanes_hold`), and the verifier's masked replay returns on rounds
carrying the prover's zerocheck and lincheck (`ReplayGo`: `replayWith_ok`). So a rep returns at good coins where its
rank check passes, the rounds it reads giving its claim points at the coins (`lcRounds_points`), its replay's and
lanes' rows vanishing at its pads, and `RunsRest` holds (`RunsGo`: `repT_go`, `runsRest`); a returning
rep's proof has the masked PIOP's closed-form lengths, level 0's cap and Ligerito lengths, the inner proof's 192-byte
salts and 64-byte path entries, and a file under `2^64` bytes at a good case's at most 2,048 extra claims
(`Case.extras_le`), so `ShapeRest` holds (`ShapeGo`: `shapeRest`), leaving `complete_jsonT` to level 0's four and the
mask-rank bound (`complete_jsonT_of_level0`); `Zk.Level0Premises` and `Zk.Level0Verifier` prove level 0's four
(`L0.complete_jsonT_closed`), and `Zk.RankDeployed` the bound, so `complete_jsonT` rests on the JSON specification
alone (`L0.complete_jsonT_of_spec`).
For zero knowledge, `honestPR`'s first message is the link of
the tables' level-0 caps, each cap a function of its level's leaves, over `encodeRowsZk`'s rows (`Round0`:
`honestPR_first`, `commitsPZk_caps`, `encodeRowsZk_entry`), and a session that returns passed every rep's `β`,
pair-coin and rank refusals at their slots (`Pass`: `sessionPZk_pass`, each rep sending exactly its rounds,
`repPZk_exact`, its rank check at its claim points at its sends' coins, `repPZk_rank`),
so one that does not return is sent as `[]` once its sends are answered (`Refuse`: `honestP_err`, `honestPR_err`,
`feed_sim_ge`). The session's second body is table `0` rep `0`'s first zerocheck squeeze, from the domain's transcript
with the digest, the level-0 cap and the pads' root absorbed, or, where that table's extra claims throw, the session
stops after the link message (`Round1`: `sessionPZk_go_round1`, `sessionPZk_go_round1_err`). Its third is that rep's
outer zerocheck point squeeze, taken with nothing absorbed since the first: the header of `m − 13` coins of kind `2`
alone (`Round2`: `sessionPZk_go_round2`). Its bodies from the second on are that rep's zerocheck's sends
(`Round3`: `Replay.ZcZ.sends`, `sessionPZk_go_sends`), its fourth `z`'s squeeze frame after the padded `round1_ab` and
`round1_c` (`sessionPZk_go_round3`). -/
