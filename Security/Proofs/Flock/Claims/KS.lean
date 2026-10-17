import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.KS.Game
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Claims.KS.Trace
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.KS.ConflictOC
import Proofs.Flock.Claims.KS.ConflictMerkle
import Proofs.Flock.Claims.KS.ConflictWithin
import Proofs.Flock.Claims.KS.PlainQueryRound
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.ConflictTied
import Proofs.Flock.Claims.KS.TableBound
import Proofs.Flock.Claims.KS.Refit
import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Claims.KS.Mixture
import Proofs.Flock.Claims.KS.PlainBound
import Proofs.Flock.Claims.KS.PlainKS
import Proofs.Flock.Claims.KS.Escapes
import Proofs.Flock.Claims.KS.OnAcc

/-! Flock's knowledge soundness at `E^rep`, in parts: one wait truncated to `E^rep`'s `K` draws, by a coupling with `k`
independent accepted runs and a Chernoff bound (`Wait`, `ERep_trunc`); the soundness proof's escape events, each a
collision among the verifier's own calls in a fork or in one run, covered by `B` and gathered into one square root
(`Fork`, `Escapes.le`); `Flock.KS` from one wait's bound on the opened columns, where an opened word within `θ` of a
witness's codeword assembles it (`Flock`, `Flock.Extracts`, `Flock.KS_of_oneWait`); at a game's verifier
(`Bridge.ofGame`), escapes tied to the verifier's log through the decision's calls (`Game`); the deployed plain and
masked instances, from one wait's bound (`Deploy`, `FlockDeploy.KS.plain_of_oneWait`, `FlockDeploy.KS.zk_of_oneWait`),
with `k ≥ 8` wherever the verifier accepts at SHA-512 (`Range`, `FlockDeploy.KS.plain_of_oneWait_sha512`,
`FlockDeploy.KS.zk_of_oneWait_sha512`).

One wait's bound from a fixed table's bound (`Joint`): the old theorems' `joint_acc_le` on the deployed verifier's own
run law (`Flock.runGame`, `Flock.oneWait_lhs_eq`), with three obligations: a fixed table's bound over tables of
well-formed or unopened (`0`) columns, accepted runs opening only well-formed columns, and conflicts tied to collisions
(`Flock.TableBoundθ`, `Flock.orZero`, `Flock.ColsOk`, `Flock.ConflictTied`, `Flock.oneWait_of_tableBound`,
`FlockDeploy.KS.plain_KS_of_tableBound`, `FlockDeploy.KS.zk_KS_of_tableBound`); the bound over tables with unopened
columns from the one over full tables, where a table that does not extract extends to a full one that does not either
(`Flock.Extendθ`, `Flock.tableBoundθ_of_extend`). The old table at the opened word (`Trace`): a word as the old oracle,
`Close` bounding the agreement (`Flock.agree_of_close`), the deployed plain code, read step, schedule and radius as the
old model's (`Flock.radius_plain`), so an old `Committed` table extracts (`FlockDeploy.KS.good_of_committed_plain`,
`FlockDeploy.KS.not_committed_plain`), and the masked code and read step as the old padded ones
(`Flock.zkEnc_eq_encAt`, `Flock.zkReadD_eq_padWit`, `Flock.radius_zk`), so an old `CommittedPad` table extracts
(`FlockDeploy.KS.good_of_committed_zk`, `FlockDeploy.KS.not_committed_zk`). Accepted runs open only full-width columns
at both records (`Width`, `FlockDeploy.KS.plain_colsOk`, `FlockDeploy.KS.zk_colsOk`), and conflicts are tied to
collisions at both (`ConflictTied`, `FlockDeploy.KS.plain_conflictTied`, `FlockDeploy.KS.zk_conflictTied`), which
leaves `TableBoundθ` and one table at every accepted statement for KS at SHA-512 (`FlockDeploy.KS.plain_KS_of_table`,
`FlockDeploy.KS.zk_KS_of_table`); the plain scope has one (`FlockDeploy.KS.plain_hone`). Toward
`ConflictTied`: a run over `OC` within a log, peeled step by step, with the
shipped step's verdict alongside (`ConflictOC`, `FlockDeploy.KS.Runs`, `FlockDeploy.KS.shadow`); the shipped
`hm96-sha512/v1` Merkle check logs every SHA-512 input of each opening (`ConflictMerkle`,
`FlockDeploy.KS.merkleCheck_logs`), and two accepted openings of one position against one cap with different rows
collide in the two logs (`FlockDeploy.KS.tiedColl_of_opens`). SHA-512
answered only within a log (`ConflictWithin`, `FlockDeploy.KS.hashesIn`) carries a run within its log to a verdict in
`V` and back (`FlockDeploy.KS.within`, `FlockDeploy.KS.back`), so the verifier's lemmas under every oracle hold of it:
among them where level 0's query coins sit in a plain rep's stream, as `ZkQueryRound` has it for the masked one
(`PlainQueryRound`, `FlockDeploy.KS.plainQueryRound_verifyWithM`). An accepted run decides within its own verifier
log, so its verdict accepts with SHA-512 answered only within that log, on the link of the tape's first message and
on tables committed under `hm96-sha512/v1` (`ConflictRun`, `FlockDeploy.KS.acc_within`, `FlockDeploy.KS.checks_hm96`).
So an accepted run's column at `q` is an opening of `q` against table 0's `root_B` in the tape's link message, with
every SHA-512 input in the run's log (`FlockDeploy.KS.Opening`, `FlockDeploy.KS.plain_opening`,
`FlockDeploy.KS.zk_opening`), and two runs at one tape opening different columns at `q` collide
(`FlockDeploy.KS.tiedColl_of_openings`). `TableBoundθ` statement by statement (`TableBound`,
`Flock.TableBoundθAt`): where the verifier accepts no run it holds with no escapes (`Flock.tableBoundθAt_of_noAcc`),
the weight covering `k` at every statement (`FlockDeploy.KS.k_le_weight`), and a statement some run accepts at SHA-512
parses in scope (`FlockDeploy.KS.acc_inScope`), so at both records it is its bound at the accepted statements
(`FlockDeploy.KS.plain_tableBoundθ_of_acc`, `FlockDeploy.KS.zk_tableBoundθ_of_acc`,
`FlockDeploy.KS.plain_KS_of_acc`, `FlockDeploy.KS.zk_KS_of_acc`); at the deployed relation, a table of
full-width columns that does not extract is not committed to the plain record's table 0
(`FlockDeploy.KS.plain_hwit`, `FlockDeploy.KS.plain_not_committed`), and under `--zk` at one table to the masked
record's (`FlockDeploy.KS.zk_hwit`, `FlockDeploy.KS.zk_not_committed`). The old bound's terms at one table (`Refit`):
an opening off the old oracle is a looked-up column off the word (`Flock.exists_ne_of_off`, `Flock.Pr_le_of_off`), the
old table error is `ε_table` (`FlockDeploy.KS.εtable_one_plain`, `FlockDeploy.KS.bound_le_εtable_zk`), and the old
fork and self-clash terms are escapes' (`Flock.oldEsc`, `Flock.oldEsc_term`; masked, with the pads roots' and `τ`'s,
`Flock.oldEscZ`, `Flock.oldEscZ_term`) that fit the weight with the fork event of weight `k`
(`FlockDeploy.KS.oldEsc_fitsFork_plain`, `FlockDeploy.KS.oldEscZ_fitsFork_zk`), so the salted and masked sessions'
`ε_c⁻` are at most `ε_table` and their term (`FlockDeploy.KS.epsSL_le`, `FlockDeploy.KS.epsSZ_le`), also averaged over
a mixture of strategies (`FlockDeploy.KS.mean_epsSL_le`, `FlockDeploy.KS.mean_epsSZ_le`). The plain fork rounds
(`Rounds`): rep `r`'s level-`ℓ + 1` cap round at table 0, no two of them one (`FlockDeploy.KS.plainJ`,
`FlockDeploy.KS.plainJ_inj`), at which the old escapes fit the plain weight (`FlockDeploy.KS.oldEsc_fitsFork_plainJ`).
At one table the model does not commit to, a live verdict is no likelier than the simulated strategy's `ε_c⁻` after
`Commit` and its probability of accepting off the table (`Mixture`: `FlockDeploy.KS.live_le_epsSL`; over a mixture of
live strategies, `FlockDeploy.KS.mean_live_le`). At a plain statement of one table, `TableBoundθAt` follows from, at
each table `C`, a mixture of salted sessions after `Commit` whose acceptance with no opening off `C` bounds the deployed
runs accepting on `C`, and whose `Adv` and self-clash the escapes' events bound, the escapes at the plain fork rounds
(`PlainBound`: `FlockDeploy.KS.plain_tableBoundθAt_of_sessions`). `Flock.KS` at the plain record follows from the live
sessions' bound at table 0 and `Extendθ` (`PlainKS`: `FlockDeploy.KS.plain_KS_of_live`). The self-clash after `Commit`
against the simulated strategy arises only on live runs the live verdict accepts, whose decoded outputs self-clash
(`Escapes`: `FlockDeploy.KS.selfClash_le_live`, `FlockDeploy.KS.mean_selfClash_le_live`). Escapes on accepted runs
(`OnAcc`, `Claims.Escapes.OnAcc`): `Flock.TableBoundθAccAt` is `TableBoundθAt` with them, which gives it
(`Flock.TableBoundθAccAt.tableBoundθAt`) and holds with none where no run accepts (`Flock.tableBoundθAccAt_of_noAcc`);
the old escapes are on accepted runs where their events are (`Flock.oldEsc_onAcc`, `Flock.oldEscZ_onAcc`). -/
