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

/-! Flock's completeness at the deployed masked verifier at any number of tables. The honest masked prover runs each
table on its share of the witness (`Honest`: `traces`, `honestP`, which is `honestPZk` at one table); the masked
verifier's error drops `readyZk`'s one-table condition (`Ready`: `ready`, `δc`, `complete_of`; `δc H` is `δcZk H` where
`readyZk` holds and the regions fit); the honest run is the pure masked session (`Run`: `run_acc_of`, `honestRun_eq`);
the skeleton is `Zk/Skeleton.lean`'s at this run (`Skeleton`: `complete_of_goals`). The scope holds per table
(`Scope`: `Case.tables`, every table of `25 ≤ m ≤ 35` and `hm96-sha512/v1`, the spec naming one table per statement)
and discharges five goals at any number of tables (`Discharge`: `tables_goal`, `schedules_goal`, `proofCount_goal`;
`Link`: `linkFirst_goal`; `Count`: `sends_goal`, each table's two reps' rounds summed). At the deployed relation each
table's trace is its share's and satisfies its block R1CS (`Trace`: `Case.traceR1CS`, `traceR1CS_goal`, from
`Readable.relS_only`), which `tracesOf` reads only at one table. The session runs at several tables from the statement's
`hello` and the prover's pieces, its reps run in stream order (`Runs`: `sessionRunsJ_goal` from `StmtOkJ`,
`ProverRunsJ`). `Complete` composes them (`complete`): from `complete_zk`'s premises, which give every check at one
table, and at several tables `StmtOkJ`, `ProverRunsJ`, `RecordJ` and `OpenJ`. The bound at several tables loses nothing
at one (`One`: `completeZk_of`, `honestP`'s completeness with error `δc H` gives `honestPZk`'s with error `δcZk H`). -/
