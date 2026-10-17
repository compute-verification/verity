import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.ZkTables.Ready
import Proofs.Flock.Claims.Complete.ZkTables.Run
import Proofs.Flock.Claims.Complete.ZkTables.Skeleton
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Discharge
import Proofs.Flock.Claims.Complete.ZkTables.Link

/-! Flock's completeness at the deployed masked verifier at any number of tables: the honest masked prover on each
table's share of the witness (`Honest`: `traces`, `honestP`, which is `honestPZk` at one table), the masked verifier's
error with `readyZk`'s one-table condition dropped (`Ready`: `ready`, `δc`, `complete_of`; `δc H` is `δcZk H` where
`readyZk` holds and the regions fit), the honest run as the pure masked session (`Run`: `run_acc_of`, `honestRun_eq`),
the skeleton (`Skeleton`: `complete_of_goals`, from `Zk/Skeleton.lean`'s goals at this run), the scope at several
tables (`Scope`: `Case.tables`, every table of `25 ≤ m ≤ 35` and `hm96-sha512/v1`, the spec naming one table per
statement), and the goals it discharges (`Discharge`: `tables_goal`, `schedules_goal`, `proofCount_goal`; `Link`:
`linkFirst_goal`). -/
