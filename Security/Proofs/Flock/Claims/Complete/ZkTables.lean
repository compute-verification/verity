import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.ZkTables.Ready
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

/-! Flock's completeness at the deployed masked verifier at any number of tables. The honest masked prover runs each
table on its share of the witness (`Honest`: `traces`, `honestP`, which is `honestPZk` at one table); the masked
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
`complete_zk`'s premises but `RecordFitsZk`, which give every check at one table, and at several tables `StmtOkJ`,
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
tables, `setupTables_spec`, a printable subset draw, `good_subset`, and the masked `Hello`'s read-back, `hello_ok`), so
`completeJS` and `completeAllS` drop it, keeping `StmtOkZk` at one table. The ring switch replays at every table given
the extra claims' algebra (`RingSwitch`: `ringSwitchJ`, `RsExtrasAt`), which holds at the deployed relation from
`RelS`, each table satisfied by its share (`Case.model`, `Run.rsExtras`), so there completeness asks per rep only the
masked Ligerito and the inner proof (`LgInnerAt`; `completeRel`, and `completeJRel` from `complete_zk_relation`'s
premises at one table). -/
