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
import Proofs.Flock.Claims.KS.SessionSC
import Proofs.Flock.Claims.KS.SessionRaw
import Proofs.Flock.Claims.KS.OffAlign
import Proofs.Flock.Claims.KS.PlainOff
import Proofs.Flock.Claims.KS.OffSim
import Proofs.Flock.Claims.KS.PlainOffLive
import Proofs.Flock.Claims.KS.PlainOffRd
import Proofs.Flock.Claims.KS.CapChunk
import Proofs.Flock.Claims.KS.PlainCapPrefix
import Proofs.Flock.Claims.KS.PlainQueried
import Proofs.Flock.Claims.KS.PlainBound
import Proofs.Flock.Claims.KS.Trim
import Proofs.Flock.Claims.KS.PlainKS
import Proofs.Flock.Claims.KS.Escapes
import Proofs.Flock.Claims.KS.OnAcc
import Proofs.Flock.Claims.KS.PlainAcc
import Proofs.Flock.Claims.KS.RoundsN
import Proofs.Flock.Claims.KS.MixtureZk
import Proofs.Flock.Claims.KS.ZkBound
import Proofs.Flock.Claims.KS.ZkKS
import Proofs.Flock.Claims.KS.JointJ
import Proofs.Flock.Claims.KS.ZkTables
import Proofs.Flock.Claims.KS.ZkBoundJ
import Proofs.Flock.Claims.KS.ZkLiveJ
import Proofs.Flock.Claims.KS.ZkAccJ
import Proofs.Flock.Claims.KS.RoundsNZk
import Proofs.Flock.Claims.KS.LevelZero
import Proofs.Flock.Claims.KS.LevelZeroAcc
import Proofs.Flock.Claims.KS.OffSC
import Proofs.Flock.Claims.KS.Entries
import Proofs.Flock.Claims.KS.DecodeSC
import Proofs.Flock.Claims.KS.DecodeDeploy
import Proofs.Flock.Claims.KS.DecodeHidden
import Proofs.Flock.Claims.KS.PlainWrP
import Proofs.Flock.Claims.KS.PlainEvent
import Proofs.Flock.Claims.KS.LiveMix
import Proofs.Flock.Claims.KS.PlainFirst
import Proofs.Flock.Claims.KS.PlainThird
import Proofs.Flock.Claims.KS.ForkSlack
import Proofs.Flock.Claims.KS.PlainSimW
import Proofs.Flock.Claims.KS.FnFork
import Proofs.Flock.Claims.KS.PlainWire
import Proofs.Flock.Claims.KS.PlainClashL
import Proofs.Flock.Claims.KS.ForkCoupling
import Proofs.Flock.Claims.KS.PlainDraw
import Proofs.Flock.Claims.KS.PlainDrawAcc
import Proofs.Flock.Claims.KS.PlainDrawDec
import Proofs.Flock.Claims.KS.PlainDrawSim
import Proofs.Flock.Claims.KS.PlainDrawKDefs
import Proofs.Flock.Claims.KS.PlainDrawRCols
import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainDrawK
import Proofs.Flock.Claims.KS.PlainDrawAccK
import Proofs.Flock.Claims.KS.PlainDrawDecK
import Proofs.Flock.Claims.KS.PlainLeafDecK
import Proofs.Flock.Claims.KS.PlainClashLaterK
import Proofs.Flock.Claims.KS.PlainClash0K
import Proofs.Flock.Claims.KS.PlainLeafClashK
import Proofs.Flock.Claims.KS.PlainDrawJ
import Proofs.Flock.Claims.KS.PlainDrawJMove
import Proofs.Flock.Claims.KS.PlainDrawJRun
import Proofs.Flock.Claims.KS.PlainDrawJCols
import Proofs.Flock.Claims.KS.PlainSlackFit
import Proofs.Flock.Claims.KS.PlainSim
import Proofs.Flock.Claims.KS.PlainLock
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.KS.PlainParts
import Proofs.Flock.Claims.KS.PlainClash0
import Proofs.Flock.Claims.KS.PlainLevels
import Proofs.Flock.Claims.KS.LevelsAll
import Proofs.Flock.Claims.KS.PlainClashLater
import Proofs.Flock.Claims.KS.ForkSim
import Proofs.Flock.Claims.KS.PlainFork
import Proofs.Flock.Claims.KS.PlainForkClash
import Proofs.Flock.Claims.KS.PlainCapsDep
import Proofs.Flock.Claims.KS.ForkAlign
import Proofs.Flock.Claims.KS.SlotDist
import Proofs.Flock.Claims.KS.RepDist
import Proofs.Flock.Claims.KS.RepDistDeploy
import Proofs.Flock.Claims.KS.SlotLead
import Proofs.Flock.Claims.KS.PlainAlign
import Proofs.Flock.Claims.KS.PlainSched
import Proofs.Flock.Claims.KS.SimLead
import Proofs.Flock.Claims.KS.PlainLead
import Proofs.Flock.Claims.KS.PlainLive
import Proofs.Flock.Claims.KS.PlainLeafKS
import Proofs.Flock.Claims.KS.PlainRawL
import Proofs.Flock.Claims.KS.PlainRdRun
import Proofs.Flock.Claims.KS.PlainKSLin
import Proofs.Flock.Claims.KS.ZkPlanJ
import Proofs.Flock.Claims.KS.ZkPlanHJ
import Proofs.Flock.Claims.KS.ZkDrawJ
import Proofs.Flock.Claims.KS.ZkDecJ
import Proofs.Flock.Claims.KS.ZkXferJ
import Proofs.Flock.Claims.KS.ZkSessionSC
import Proofs.Flock.Claims.KS.ZkPads
import Proofs.Flock.Claims.KS.ZkSessionW
import Proofs.Flock.Claims.KS.ZkOffSC
import Proofs.Flock.Claims.KS.ZkEntries
import Proofs.Flock.Claims.KS.ZkDecodeW
import Proofs.Flock.Claims.KS.ZkEvent
import Proofs.Flock.Claims.KS.ZkFirst
import Proofs.Flock.Claims.KS.FitsR
import Proofs.Flock.Claims.KS.ZkFitsW
import Proofs.Flock.Claims.KS.ZkBoundW
import Proofs.Flock.Claims.KS.ZkLiveW
import Proofs.Flock.Claims.KS.JointR
import Proofs.Flock.Claims.KS.ZkXferW
import Proofs.Flock.Claims.KS.ZkTrimW
import Proofs.Flock.Claims.KS.ZkSimW
import Proofs.Flock.Claims.KS.ZkAbortR
import Proofs.Flock.Claims.KS.ZkAbortJR
import Proofs.Flock.Claims.KS.ZkClashRaw
import Proofs.Flock.Claims.KS.ZkSlackW
import Proofs.Flock.Claims.KS.ZkSlackFit
import Proofs.Flock.Claims.KS.ZkSim
import Proofs.Flock.Claims.KS.ZkLock
import Proofs.Flock.Claims.KS.ZkStep
import Proofs.Flock.Claims.KS.ZkClash0
import Proofs.Flock.Claims.KS.ZkLevelsAll
import Proofs.Flock.Claims.KS.ZkPadsPair
import Proofs.Flock.Claims.KS.ZkTauFork
import Proofs.Flock.Claims.KS.ZkTauPair
import Proofs.Flock.Claims.KS.ZkLevels
import Proofs.Flock.Claims.KS.ZkClashL
import Proofs.Flock.Claims.KS.ZkFork0
import Proofs.Flock.Claims.KS.ZkPadsLog
import Proofs.Flock.Claims.KS.ZkClashPads
import Proofs.Flock.Claims.KS.TrimBridge
import Proofs.Flock.Claims.KS.ZkFork
import Proofs.Flock.Claims.KS.ZkAlign
import Proofs.Flock.Claims.KS.RepDistZ
import Proofs.Flock.Claims.KS.ZkSched
import Proofs.Flock.Claims.KS.SlotBatch
import Proofs.Flock.Claims.KS.PushSched
import Proofs.Flock.Claims.KS.ZkBatch
import Proofs.Flock.Claims.KS.BatchProj
import Proofs.Flock.Claims.KS.BatchSteps
import Proofs.Flock.Claims.KS.BatchSched
import Proofs.Flock.Claims.KS.BatchMove
import Proofs.Flock.Claims.KS.BatchSim
import Proofs.Flock.Claims.KS.ZkLiveSched
import Proofs.Flock.Claims.KS.ZkBatchSim
import Proofs.Flock.Claims.KS.ZkForkClash
import Proofs.Flock.Claims.KS.ZkPadsSlot
import Proofs.Flock.Claims.KS.ZkKSAlign
import Proofs.Flock.Claims.KS.SlotDistOn
import Proofs.Flock.Claims.KS.ZkTauSlot
import Proofs.Flock.Claims.KS.ZkUntrim
import Proofs.Flock.Claims.KS.Extend
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.KS.PlainUntrim
import Proofs.Flock.Claims.KS.ZkOuter
import Proofs.Flock.Claims.KS.PlainAccPrR

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
`FlockDeploy.KS.plain_KS_of_tableBound`, `FlockDeploy.KS.zk_KS_of_tableBound`). At the records the claims read, whose
columns are trimmed of trailing zeros so `0` is the trimmed zero column (`FlockDeploy.deployPlainTrim`,
`FlockDeploy.deployZkTrim`), the three obligations carry over from the untrimmed records over full-width columns,
the table's bound by padding back (`Trim`, `Flock.tableBoundθ_mapCols`, `FlockDeploy.KS.plain_KS_trim`,
`FlockDeploy.KS.zk_KS_trim`). The old table at the opened word (`Trace`): a word as the old oracle,
`Close` bounding the agreement (`Flock.agree_of_close`), the deployed plain code, read step, schedule and radius as the
old model's (`Flock.radius_plain`), so an old `Committed` table extracts (`FlockDeploy.KS.good_of_committed_plain`,
`FlockDeploy.KS.not_committed_plain`), and the masked code and read step as the old padded ones (`Flock.zkEnc_eq_encAt`,
`Flock.zkReadD_eq_padWit`, `Flock.radius_zk`), so an old `CommittedPad` table extracts
(`FlockDeploy.KS.good_of_committed_zk`, `FlockDeploy.KS.not_committed_zk`). Accepted runs open only full-width columns
at both records (`Width`, `FlockDeploy.KS.plain_colsOk`, `FlockDeploy.KS.zk_colsOk`), and conflicts are tied to
collisions at both (`ConflictTied`, `FlockDeploy.KS.plain_conflictTied`, `FlockDeploy.KS.zk_conflictTied`), which leaves
`TableBoundθ` and one table at every accepted statement for KS at SHA-512 (`FlockDeploy.KS.plain_KS_of_table`,
`FlockDeploy.KS.zk_KS_of_table`); the plain scope has one (`FlockDeploy.KS.plain_hone`). Toward `ConflictTied`: a run
over `OC` within a log, peeled step by step, with the shipped step's verdict alongside (`ConflictOC`,
`FlockDeploy.KS.Runs`, `FlockDeploy.KS.shadow`); the shipped `hm96-sha512/v1` Merkle check logs every SHA-512 input of
each opening (`ConflictMerkle`, `FlockDeploy.KS.merkleCheck_logs`), and so does the plain-leaves one
(`FlockDeploy.KS.merkleCheck_logsP`), and two accepted openings of one position against one cap with different rows
collide in the two logs (`FlockDeploy.KS.tiedColl_of_opens`, `FlockDeploy.KS.tiedColl_of_opensP`). SHA-512 answered
only within a log (`ConflictWithin`, `FlockDeploy.KS.hashesIn`) carries a run within its log to a verdict in `V` and back
(`FlockDeploy.KS.within`, `FlockDeploy.KS.back`), so the verifier's lemmas under every oracle hold of it: among them
where level 0's query coins sit in a plain rep's stream, as `ZkQueryRound` has it for the masked one (`PlainQueryRound`,
`FlockDeploy.KS.plainQueryRound_verifyWithM`). An accepted run decides within its own verifier log, so its verdict
accepts with SHA-512 answered only within that log, on the link of the tape's first message and on tables committed
under `hm96-sha512/v1`, or without `--zk` under plain SHA-512 leaves (`ConflictRun`, `FlockDeploy.KS.acc_within`,
`FlockDeploy.KS.checks_leaf`, `FlockDeploy.KS.checks_hm96`). So an accepted run's
column at `q` is an opening of `q` against table 0's `root_B` in the tape's link message, with every SHA-512 input in
the run's log (`FlockDeploy.KS.Opening`, `FlockDeploy.KS.plain_opening`, `FlockDeploy.KS.zk_opening`), and two runs at
one tape opening different columns at `q` collide (`FlockDeploy.KS.tiedColl_of_openings`). `TableBoundθ` statement by
statement (`TableBound`, `Flock.TableBoundθAt`): where the verifier accepts no run it holds with no escapes
(`Flock.tableBoundθAt_of_noAcc`), the weight covering `k` at every statement (`FlockDeploy.KS.k_le_weight`), and a
statement some run accepts at SHA-512 parses in scope (`FlockDeploy.KS.acc_inScope`), so at both records it is its bound
at the accepted statements (`FlockDeploy.KS.plain_tableBoundθ_of_acc`, `FlockDeploy.KS.zk_tableBoundθ_of_acc`,
`FlockDeploy.KS.plain_KS_of_acc`, `FlockDeploy.KS.zk_KS_of_acc`); at the deployed relation, a table of full-width
columns that does not extract is not committed to the plain record's table 0 (`FlockDeploy.KS.plain_hwit`,
`FlockDeploy.KS.plain_not_committed`), and under `--zk` at one table to the masked record's (`FlockDeploy.KS.zk_hwit`,
`FlockDeploy.KS.zk_not_committed`). The old bound's terms at one table (`Refit`): an opening off the old oracle is a
looked-up column off the word (`Flock.exists_ne_of_off`, `Flock.Pr_le_of_off`), the old table error is `ε_table`
(`FlockDeploy.KS.εtable_one_plain`, `FlockDeploy.KS.bound_le_εtable_zk`), and the old fork and self-clash terms are
escapes' (`Flock.oldEsc`, `Flock.oldEsc_term`; masked, with the pads roots' and `τ`'s, `Flock.oldEscZ`,
`Flock.oldEscZ_term`) that fit the weight with the fork event of weight `k` (`FlockDeploy.KS.oldEsc_fitsFork_plain`,
`FlockDeploy.KS.oldEscZ_fitsFork_zk`; the plain weight with level 0's share to spare, `FlockDeploy.KS.plain_fork_room`),
so the salted and masked sessions' `ε_c⁻` are at most `ε_table` and their term
(`FlockDeploy.KS.epsSL_le`, `FlockDeploy.KS.epsSZ_le`), also averaged over a mixture of strategies
(`FlockDeploy.KS.mean_epsSL_le`, `FlockDeploy.KS.mean_epsSZ_le`). The plain fork rounds (`Rounds`): rep `r`'s
level-`ℓ + 1` cap round at table 0, no two of them one (`FlockDeploy.KS.plainJ`, `FlockDeploy.KS.plainJ_inj`), at which
the old escapes fit the plain weight (`FlockDeploy.KS.oldEsc_fitsFork_plainJ`). At one table the model does not commit
to, a live verdict is no likelier than the simulated strategy's `ε_c⁻` after `Commit` and its probability of accepting
off the table (`Mixture`: `FlockDeploy.KS.live_le_epsSL`; over a mixture of live strategies,
`FlockDeploy.KS.mean_live_le`). The session's bound with its level-0 term only on runs that do not self-clash, a
self-clash being in `ε_c⁻` already (`SessionSC`: `FlockSoundness.Discharge.Exec.bad_pointwise_ofL_sc`,
`FlockSoundness.Discharge.Exec.session_sound_of_table_accL_sc`). At a plain statement of one table, `TableBoundθAt`
follows from, at each table `C`, a mixture of salted sessions after `Commit` whose acceptance with no opening off `C`
unless it self-clashes bounds the deployed runs accepting on `C`, and whose `Adv` and self-clash the escapes' events
bound, the escapes at the plain fork rounds (`PlainBound`: `FlockDeploy.KS.plain_tableBoundθAt_of_sessions`). `Flock.KS`
at the trimmed plain record follows from the live sessions' bound at table 0 (`PlainKS`:
`FlockDeploy.KS.plain_KS_of_live`). The self-clash after `Commit` against the simulated strategy arises only on live
runs the live verdict accepts, whose decoded outputs self-clash (`Escapes`: `FlockDeploy.KS.selfClash_le_live`,
`FlockDeploy.KS.mean_selfClash_le_live`). Escapes on accepted runs (`OnAcc`, `Claims.Escapes.OnAcc`):
`Flock.TableBoundθAccAt` is `TableBoundθAt` with them, which gives it (`Flock.TableBoundθAccAt.tableBoundθAt`) and holds
with none where no run accepts (`Flock.tableBoundθAccAt_of_noAcc`); the old escapes are on accepted runs where their
events are (`Flock.oldEsc_onAcc`, `Flock.oldEscZ_onAcc`). With the live sessions' bound at escapes on accepted runs,
whose events bound the decoded self-clashes on accepted live runs (`PlainAcc`: `FlockDeploy.KS.PlainLiveAcc`, which
gives `PlainLive`, `FlockDeploy.KS.plainLive_of_acc`), the plain table bound holds with escapes on accepted runs at
every statement (`FlockDeploy.KS.plain_tableBoundθAccAt`). Each round of an accepted plain rep issued the coins its
statement fixes (`RoundsN`: `FlockDeploy.repNs`, `FlockDeploy.verifyRepWith_readsN`; per rep stream,
`FlockDeploy.plainRepNs`, `FlockDeploy.streams_ns_plain`), and custody reads each round message's coin at the session's
count (`FlockDeploy.plainNs`, `FlockDeploy.custody_vals_plain`). At `--zk`, one table's compiled session is `Commit`'s
cap and then `sessAtZ` with no other game, so a live verdict at one masked table is no likelier than any verdict after
`Commit` the decoder is right for (`MixtureZk`: `FlockDeploy.KS.live_le_simAtZ`, `FlockDeploy.KS.mean_live_le_simAtZ`),
and at a masked statement of one table, `TableBoundθAt` follows from a mixture of masked sessions after `Commit` at a
table `T` at C-Flock ZK's parameters, the escapes `Flock.oldEscZ` at injective fork rounds, and no table that does not
extract committed to `T` (`ZkBound`: `FlockDeploy.KS.zk_tableBoundθAt_of_sessions`,
`FlockDeploy.KS.zk_tableBoundθAt_of_live`, at the live sessions' bound `FlockDeploy.KS.ZkLiveAt`). At the deployed
masked record, `T` is C-Flock ZK's padded table of the statement's table 0 (`ZkKS`: `FlockDeploy.KS.zkTab`,
`FlockDeploy.KS.zkTab_m1`), so `Flock.KS` at the trimmed record follows from one table at every statement some run
accepts (`FlockDeploy.KS.ZkOne`, which the masked scope's several tables do not give) and the live sessions' bound at
`zkTab` (`FlockDeploy.KS.ZkLive`, `FlockDeploy.KS.zk_KS_of_live`). At several tables (`JointJ`), one wait follows
from every table's bound with one family of escapes, over every table's word's positions (`Flock.TableBoundJ`,
`Flock.ColsOkJ`, `Flock.ConflictTiedJ`, `Flock.oneWait_of_tableBoundJ`), and at the masked record accepted runs open
table `j`'s columns at its width and as its level-0 openings (`ZkTables`: `FlockDeploy.KS.wordLenJ`,
`FlockDeploy.KS.zk_colsOkJ`, `FlockDeploy.KS.zk_openingJ`, `FlockDeploy.KS.zk_conflictTiedJ`), so `Flock.KS` there
follows from every table's bound alone, at every number of tables (`FlockDeploy.KS.zk_KS_of_tableBoundJ`). The batched
tables' old fork events share rounds, so joined by round they fit the masked weight (`ZkBoundJ`: `Flock.joinEscZ`,
`FlockDeploy.KS.joinEscZ_fitsFork_zk`, `Flock.oldEscZ_term_le_join`); at table `j` the old padded bound is at most
`ε_table` (`FlockDeploy.KS.mean_epsSZ_leJ`), and a table that does not extract table `j` is not committed to its model
statement (`FlockDeploy.KS.zk_not_committedJ`). So every table's bound at a statement follows from masked sessions after
`Commit` at every table (`FlockDeploy.KS.ZkSessJ`, `FlockDeploy.KS.zk_tableBoundJAt_of_sessions`), needed only where
some run accepts (`FlockDeploy.KS.zk_tableBoundJ_of_acc`). After `Commit` table `j` is at position `j` of the batch, so
a live verdict is no likelier than table `j`'s verdict after `Commit` (`ZkLiveJ`: `FlockDeploy.KS.simAtJZ`,
`FlockDeploy.KS.mean_live_le_simAtJZ`), and the masked sessions follow from the live sessions' bound at every table
(`FlockDeploy.KS.ZkLiveAtJ`, `FlockDeploy.KS.zkSessJ_of_liveJ`). At the deployed masked record the tables are C-Flock
ZK's padded tables of the statement's tables (`FlockDeploy.KS.zkTabs`), so `Flock.KS` follows at every number of tables
from the live sessions' bound there (`FlockDeploy.KS.ZkLiveJ`) and `ExtendJ` (`FlockDeploy.KS.zk_KS_of_liveJ`). At the
masked record an output at the empty openings never self-clashes, so table `j`'s self-clash after `Commit` arises only
on accepted live runs (`ZkAccJ`: `FlockDeploy.KS.not_selfClashZ_nul`, `FlockDeploy.KS.mean_selfClash_le_liveJZ`); with
the live sessions' bound at escapes on accepted runs (`FlockDeploy.KS.ZkLiveAccJ`, which gives `ZkLiveJ`,
`FlockDeploy.KS.zkLiveJ_of_acc`), every table's bound holds with escapes on accepted runs at every statement
(`Flock.TableBoundJAccAt`, `FlockDeploy.KS.zk_tableBoundJAccAt`), and so table 0's (`Flock.TableBoundθAccAt`,
`Flock.TableBoundJAccAt.tableBoundθAccAt`, `FlockDeploy.KS.zk_tableBoundθAccAt`). Each round of an accepted masked rep
issued the coins its statement fixes (`RoundsNZk`: `FlockDeploy.zkRepNs`, `FlockDeploy.zkVerifyRepWith_readsN`; per rep
stream, `FlockDeploy.zkRepNsOf`, `FlockDeploy.streams_ns_zk`), and custody reads each round message's coin at the
session's count (`FlockDeploy.zkNs`, `FlockDeploy.custody_vals_zk`). An accepted Ligerito run's first model request is
level 0's, at the positions of the coins of the round `k0 + 6` after Ligerito's first (`LevelZero`:
`FlockSoundness.Refine.ligerito_refines_req0`, `FlockSoundness.Refine.ligerito_accepts_opening_req0`), so an accepted
rep's is at the stream's round `plainQueryRound` (`FlockSoundness.Refine.lincheck_moves`,
`FlockSoundness.Refine.ringSwitch_moves`, `FlockSoundness.Refine.rep_refines_req0`). At flock-verify's acceptance, every
result of each table's reps' runs requests level 0 first there, and every request has its level's lanes and path length
(`LevelZeroAcc`: `FlockSoundness.Discharge.Exec.accepts_req0`). A position the deployed reader is given below `2^d` is
looked up, to a column it was given there, so an output whose level-0 openings verify and are the reader's entries opens
a position off `C` without a self-clash only where the reader looks up a column off `C` (`OffSC`:
`Claims.Flock.lookCol_distinctBelow_of_mem`, `Claims.Flock.exists_ne_of_off₀L_sc`); an accepted salted table's level-0
openings verify (`Claims.Flock.verified_opens₀L`). At flock-verify's acceptance, every accepted salted output of a
table's reps' runs has its level-0 openings verifying below `2^d`, with positions and rows its reps' `plainRepColsOf`
entries (`Entries`: `FlockSoundness.Discharge.Exec.accepts_opens₀L`, `Claims.Flock.exists_ne_of_entries`). So at `wrP`,
at one table, the deployed plain decoders are right for accepting with no level-0 opening off `C` unless it
self-clashes, on the outcomes flock-verify accepts at the live root where the deployed reader looks up no column off `C`
(`DecodeSC`: `Claims.Bridge.DecodePlain.TxNoOff`, `Claims.Bridge.DecodePlain.decodesTable_wrP_sc`). At a plain statement
in scope with public outputs, that table is `plainTab`, so those decoders are the ones `PlainLiveAccAt` asks for
(`DecodeDeploy`: `FlockDeploy.KS.plain_hdec`). With hidden outputs and `hm96-sha512/v1` leaves, the hidden table is
`plainTab` too, and the same holds at its openings (`DecodeHidden`: `Claims.Bridge.DecodePlain.decodesTable_wrP_scH`,
`FlockDeploy.KS.plain_hdecH`). A statement some run accepts sets up at a draw, so the plain chain asks only for the
mixture at those decoders, whether or not the outputs are hidden, and for `PlainLiveAccAt` itself at hidden outputs
under the plain SHA-512 leaves the scope also admits, where the decoders at `leaf512` are not right (`PlainWrP`:
`FlockDeploy.KS.acc_draw`, `FlockDeploy.KS.PlainLiveWrPAt`,
`FlockDeploy.KS.PlainLiveW`, `FlockDeploy.KS.plain_KS_of_w`). On an accepting deployed run whose record carries the
statement's draw, the event the mixture's first bound reads gives that verdict at the run's live outcome (`PlainEvent`:
`FlockDeploy.KS.cols_plain`, `FlockDeploy.KS.plainVerdict_of_run`). A prover that plays the live session with the rest
of each deployed coin on its tape bounds an event of the deployed coins by a mixture of live strategies, the merged
coins being uniform, and the mixture by an event of the deployed coins the live event implies (`LiveMix`:
`Claims.Bridge.Live.Pr_mergeC`, `Claims.Bridge.Live.Pr_mergeAt`, `Claims.Bridge.Live.liveMix_eq`,
`Claims.Bridge.Live.Pr_le_liveMix`, `Claims.Bridge.Live.liveMix_le_Pr`). A live prover simulates the deployed prover on
coins of uniform law where its live outcome is the run's wherever the run accepts, and at the coins that merge each live
coin with the rest of a deployed coin the law holds, so the simulation of the runs alone suffices (`PlainFirst`:
`FlockDeploy.KS.PlainSim`, `FlockDeploy.KS.PlainSimRun`, `FlockDeploy.KS.plainCoins`,
`FlockDeploy.KS.plainSim_of_run`); the checks compare draws only up to `JsonEquiv`, which holds between different
terms (`FlockDeploy.KS.drawEq_ne`), so acceptance gives the draw only up to it (`FlockDeploy.KS.RecordDrawEquiv`,
`FlockDeploy.KS.recordDrawEquiv`). At those coins, a deployed event that the live event implies bounds the mixture, the
third bound's form (`PlainThird`: `FlockDeploy.KS.plainMix_le_of_run`). The simulator's bounds read an accepting run or
fork with its collision, and its fork bound holds up to a slack (`PlainSimW`: `FlockDeploy.KS.plainEv₁`,
`FlockDeploy.KS.plainForkEv`, `FlockDeploy.KS.PlainClashRun`). The slack is the later rounds' forks and
the one-run collision at the chances `δ_t` that a fork's coins at round `t` agree where the simulator's tape reads them,
`Σ_{j<i<K} Δ_{j,i} Pr[F_i] + Δ_{j,K} Pr[F_K]` with `Δ_{j,i} = Π_{j≤t<i} δ_t` and `F_K` the one-run collision; a coupling
of the fork's coin law gives it (`ForkSlack`: `Claims.Flock.ForkCoinCoupling`, `Claims.Flock.slack_of_coupling`), and it
folds into one combined fork event per round with the old escapes, which fit the plain weight in level 0's room
(`Claims.Flock.slackEsc`, `Claims.Flock.slackEsc_term`, `FlockDeploy.KS.slackEsc_fitsFork_plain`). The tape may hold the
rest of each deployed coin (`plainCoins`) or uniform functions read at the live coins seen so far, so that a fork that
redraws them reads fresh entries (`LiveMix`: `Claims.Bridge.Live.fnC`, `Claims.Bridge.Live.Pr_fnC`,
`Claims.Bridge.Live.Pr_fnAt`). Two different columns an accepted plain run's reps open at one position collide in its
own log (`ConflictTied`: `FlockDeploy.KS.plainRawCols`, `FlockDeploy.KS.plain_opening_raw`,
`FlockDeploy.KS.plain_selfClash_tied`), so the simulator's self-clashes at level 0 need only give such columns
(`PlainSimW`: `FlockDeploy.KS.PlainRawClashRun`, `FlockDeploy.KS.plainClashRun_of_raw`), and past level 0 two openings
of one position to different rows against one root in the run's log (`PlainClashL`: `FlockDeploy.KS.PlainLevelClash`,
`FlockDeploy.KS.PlainRawClashRunL`, `FlockDeploy.KS.plainClashRun_of_rawL`). At the deployed plain verifier's message
formats (`PlainWire`: `FlockDeploy.KS.plainWire`), the simulation is the live outcome's roots, link points, reps' rounds
and proof files at the view's (`FlockDeploy.KS.PlainSimRoot`, `FlockDeploy.KS.PlainSimPts`,
`FlockDeploy.KS.PlainSimReps`, `FlockDeploy.KS.PlainSimProofs`, `FlockDeploy.KS.plainSimRun_of_parts`), and a simulator
that aborts on rejected runs gives the clash's acceptance (`FlockDeploy.KS.PlainSimAbort`,
`FlockDeploy.KS.plainRawClashRun_of_abort`). At a tape of uniform functions read at a key that reads and tells
apart the revealed live history, the simulated fork at round `t` is at most the deployed fork there and `δ_t` times the
simulated fork at `t + 1`, with `δ_t` the chance that two coins agree on the entries the key reads,
`|X|^{-min (n t) M}`, which is `ForkCoinCoupling`'s shape (`FnFork`: `Claims.Flock.FnFork.PrefixKey`,
`Claims.Flock.FnFork.prefixKey_histKey`, `Claims.Flock.FnFork.fnC_coupling`, `Claims.Flock.FnFork.fnAt_coupling`,
`Claims.Flock.FnFork.δAt_eq`), so the simulator on such a tape has the coupling at those `δ_t`, and its slack
(`ForkCoupling`: `Claims.Flock.forkCoinCoupling_fnAt`, `Claims.Flock.slack_fnAt`,
`FlockDeploy.KS.forkCoinCoupling_plain`). The verdict at the statement's draw reads `verify` at the setup at that draw,
not the record's, so an accepting run gives it at its live outcome whatever draw its record carries (`PlainDraw`:
`Claims.Bridge.PlainDraw.AcceptsAtD`, `Claims.Bridge.PlainDraw.wrPD`, `FlockDeploy.KS.plainVerdictAt`,
`FlockDeploy.KS.plainVerdictAt_of_run`), and at every `keep` an accepting run of `VK keep` gives it read through
`keep`, its rounds the live outcome's up to their digests (`PlainDrawK`:
`Claims.Bridge.PlainDraw.AcceptsAtDK`, `Claims.Bridge.PlainDraw.RecordsLiveAtDK`,
`FlockDeploy.KS.plainVerdictAtK_of_run`), and the acceptance lemmas and decoders below hold at it by their proofs
(`PlainDrawAccK`, `PlainDrawDecK`: `FlockDeploy.KS.plain_hdecAtK`, `FlockDeploy.KS.plain_hdecHAtK`), the rep's run
reading no round's digest (`FlockSoundness.Discharge.PlaysCoin.repRun_of_erase`), and so do the decoders at plain
SHA-512 leaves and the later entries (`PlainLeafDecK`: `FlockDeploy.KS.plain_hdecHUAtK`; `PlainClashLaterK`:
`FlockDeploy.KS.plain_laterClashAtK`, `FlockDeploy.KS.plain_laterClashHAtK`) and the level-0 clashes
(`PlainClash0K`, `PlainLeafClashK`: `FlockDeploy.KS.plain_clash₀AtK`, `FlockDeploy.KS.plain_clash₀HUAtK`,
`FlockDeploy.KS.plain_laterClashHUAtK`). The acceptance lemmas the plain
decoders read hold at a draw value by their proofs, through `keep` and at `keep := id`
(`PlainDrawAccK`: `FlockSoundness.Discharge.AtDraw.accepts_oneAnyDK`; `PlainDrawAcc`:
`FlockSoundness.Discharge.AtDraw.tableAt_of_acceptsD`), so the plain decoders decode the verdict at the draw
(`PlainDrawDec`: `Claims.Bridge.PlainDraw.decodesTable_wrPD_sc`, `FlockDeploy.KS.plain_hdecAt`,
`FlockDeploy.KS.plain_hdecHAt`), and the plain table bound at a statement holds from a simulator at that verdict, with
nothing asked of the record's draw (`PlainDrawSim`: `FlockDeploy.KS.plain_first_of_simD`,
`FlockDeploy.KS.PlainSimWrPAtD`, `FlockDeploy.KS.PlainSimWD`, `FlockDeploy.KS.plain_tableBoundθAt_of_simD`). At the
coupling's
chances with every revealed length before the last round positive, each `δ_t` there is at most `2^{-128}`, the plain
fork rounds are before the last round, a fork's slack weights fall geometrically past its round, and the slack fits
level 0's room in the plain weight whatever the number of rounds, and a fork bound at most the simulated fork at
`fnAt` is at most the plain fork's and that slack (`PlainSlackFit`: `Claims.Flock.sum_slackW_le`,
`Claims.Flock.sum_slackC_le`, `Claims.Flock.slack_fits_fast100`, `FlockDeploy.KS.plainJ_succ_lt`,
`FlockDeploy.KS.plain_slack_fits`, `FlockDeploy.KS.plain_fork_le_fnAt`). The plain simulator runs `P` round by round on
the plain wire, at deployed coins it rebuilds from a tape of uniform functions of the revealed live history and the
entries the live session revealed, which are the coins `fnAt` gives on the live history; a live prover whose messages
along the live history are known has those rounds as its live outcome (`PlainSim`: `FlockDeploy.KS.plainSim`,
`FlockDeploy.KS.plainΦ`, `FlockDeploy.KS.Pr_plainΦ`, `FlockDeploy.KS.liveOut_known`, `FlockDeploy.KS.plainSim_root`). A
deployed run's messages are `P`'s at the history of its coins' bytes, and its acceptance does not read its last coin
(`PlainLock`: `FlockDeploy.KS.run_dep`, `FlockDeploy.KS.depMsg_congr`, `FlockDeploy.KS.run_acc_iff_last`); along the
live history the simulator's coins are `plainΦ`'s wherever it has revealed them, so it sends `P`'s rounds and its live
outcome is theirs (`FlockDeploy.KS.simCoins_agree`, `FlockDeploy.KS.plainSim_msg`, `FlockDeploy.KS.plainSim_liveOut`).
Where `P`'s run decides and accepts, the record's two rep streams hold `P`'s round messages with their coins, each of
the table's coin counts, so the simulator's live outcome has the record's reps and proof files at every plain statement
in scope, whose link points fit a live coin (`PlainStep`: `FlockDeploy.KS.plain_decides_record`,
`FlockDeploy.KS.plain_mPts`, `FlockDeploy.KS.PlainSimLockstep`, `FlockDeploy.KS.plainSimLockstep`), and with the link
coin's points it is the record's view (`FlockDeploy.KS.plainSimPts`, `FlockDeploy.KS.plainSimRun`); where it is
rejected, the simulator's last message is empty, and the verdict at the draw refuses an outcome without proof files
(`FlockDeploy.KS.PlainSimAbortAt`, `FlockDeploy.KS.plainSimAbortAt`). So `PlainSimWD` holds from two parts of the
concrete simulator: its live self-clashes give `P`'s accepting runs a raw clash, and its fork advantage is at most its
simulated fork (`PlainParts`: `FlockDeploy.KS.PlainSimClashAt`, `FlockDeploy.KS.PlainSimForkLe`,
`FlockDeploy.KS.plainSimWrPAtD_of_parts`, `FlockDeploy.KS.plainSimWD_of_parts`).
The level-0 case of that clash holds: a level-0 self-clash there is two of the reps' entries at one position with
different columns, which are the run's raw columns, so the clash needs only its case past level 0 (`PlainClash0`:
`FlockDeploy.KS.plainClashQ₀`, `FlockDeploy.KS.PlainSimLaterClashAt`, `FlockDeploy.KS.plainSimClashAt_of_later`).
Past level 0, an accepted plain rep passed every later level's Merkle check on its proof's rows at the positions of
that level's query round, against the flattened recursive cap before it, so two rows a rep opens at one later-level
position that differ clash in the run's log (`PlainLevels`: `FlockDeploy.KS.plainLevels_verifyWithM`,
`FlockDeploy.KS.plain_levelClash`). On the model's side, the same accepted rep's requests after level 0's are
levels `1, …, R` in order, each at the positions of the coins of its query round (`LevelsAll`:
`FlockSoundness.Refine.LaterAt`, `FlockSoundness.Refine.find?_later`, `FlockSoundness.Refine.rep_refines_reqs`).
So a self-clash past level 0 at the concrete simulator's outcome is two of one rep's level-`ℓ` entries at one position
with different columns, which are the run's level-`ℓ` rows and clash in its log, so the clash holds at every
statement (`PlainClashLater`: `FlockSoundness.Discharge.AtDraw.accepts_opensLD`, `FlockDeploy.KS.plainClashQL`,
`FlockDeploy.KS.plainSimLaterClashAt`).
Two runs forked after `j` coins pass through a simulation, which matches the model's coins with the live ones one to
one, and at the live session, whose every coin answers its own send, `B`'s fork at round `j` is the fork after `j`
coins (`ForkSim`: `FlockSoundness.Game.forkC`, `Claims.Bridge.Sim.forkC_le`, `Claims.Bridge.Live.forkX_sessionK`).
So the simulator's fork bound at a plain fork round holds where, against the simulated strategy after `Commit`, the
fork at the round's table node is at most the fork after as many coins of two runs recording one cap and opening a
position of it to different rows, and that live event gives `P`'s two accepting runs a collision (`PlainFork`:
`FlockDeploy.KS.forkC_simAt_le`, `FlockDeploy.KS.forkC_liveStrat`, `FlockDeploy.KS.PlainForkAlign`,
`FlockDeploy.KS.PlainForkClash`, `FlockDeploy.KS.plainSimForkLe_of`).
That live event's collision holds: each live outcome is its run's, whose rep records its proof's recursive caps and
opens its deployed later rows, each an `Opening` in the run's log, so one recorded cap is one root and two rows opened
at one position of it collide (`PlainForkClash`: `FlockDeploy.KS.plain_liveOpening`, `FlockDeploy.KS.capDig_inj`,
`FlockDeploy.KS.plainForkQ_of`, `FlockDeploy.KS.plainForkClash`).
A fork at a table node is at most the fork after `j` coins where the strategy reaches that node after `j` coins on every
path, or forks to nothing on the paths that don't, and the fork event at the node's cap implies the other on the
outcomes its leaves relate (`ForkAlign`: `FlockSoundness.Game.Lock.SlotAt`,
`FlockSoundness.Game.Lock.forkE_le_forkC`). The session's leaves record the cap of the entry they hold wherever
they open a position at it, so the simulated fork's alignment is the coin count alone, and holds where the simulated
strategy reaches each plain fork slot after `plainJ r ℓ` coins (`PlainAlign`: `FlockDeploy.KS.capAtL_of_sessRelL`,
`FlockDeploy.KS.advRSL_le_forkC`, `FlockDeploy.KS.PlainSlotAlign`, `FlockDeploy.KS.plainForkAlign_of_slot`). A dead
branch answers the openings with
`nulOp`, so two of its runs open only the empty row, never conflict, and fork to nothing at any chooser
(`FlockSoundness.Game.Lock.forkE_answerWith`, `FlockDeploy.KS.slotAt_answerWith_nul`).
Under a schedule that steps the first of two interleaved games until it ends, after `n` coins, except below a part of
the run whose every reached outcome is dead, a slot of the first lockstep is met after its own count and a slot of the
second after `n` more (`SlotLead`: `FlockSoundness.Game.Lock.Lead`, `FlockSoundness.Game.Lock.SlotAt.lead_fst`,
`FlockSoundness.Game.Lock.SlotAt.lead_snd`), so the plain slot alignment holds where the simulated strategy submits
rep 0's `repRounds` coins before rep 1's at every link point, with outcomes that open only the empty row as the dead
ones (`PlainSched`: `FlockDeploy.KS.LeadsAfter`, `FlockDeploy.KS.slotRSL_lead`, `FlockDeploy.KS.PlainSchedAlign`,
`FlockDeploy.KS.plainSlotAlign_of_sched`).
The simulation's strategy, read at its one table, leads with rep 0 for `n` coins wherever the live strategy submits its
next `n` rounds on rep 0 and the rest on rep 1 (`SimLead`: `FlockDeploy.KS.SchedL`, `FlockDeploy.KS.lead_simRJ`), and
`simAt` is that strategy at each link point (`PlainLead`: `FlockDeploy.KS.simAt_heq`, `FlockDeploy.KS.leadsAfter_simAt`,
`FlockDeploy.KS.leadsAfter_plainSimAt`). The plain simulator sends deployed round `k + 1` on rep `plainRep (k + 1)`,
so its live strategy submits `repRounds` rounds on rep 0 first wherever `fast100` has at least two levels, and
`PlainSchedAlign` holds at every statement (`PlainLive`: `FlockDeploy.KS.schedL_of_schedK`,
`FlockDeploy.KS.plainLive_sched`, `FlockDeploy.KS.plainRepNs_length`, `FlockDeploy.KS.plainSchedAlign`), so
`PlainSimWD` holds at every statement. Hidden outputs under plain SHA-512 leaves, the one leaf other than hm96's the
plain scope admits, are decoded right at the model's plain leaf, where the simulator's chain runs as at hm96 leaves, so
KS at the deployed plain verifier's trimmed record holds with no hypothesis (`PlainLeafDec`, `PlainLeafClash`,
`PlainLeafFork`, `PlainLeafKS`: `FlockDeploy.KS.plain_hdecHUAt`, `FlockDeploy.KS.plainSimClashHU`,
`FlockDeploy.KS.plainAdv_leU`, `FlockDeploy.KS.plain_leafU`, `FlockDeploy.KS.plainSimWD_all`,
`FlockDeploy.KS.plain_tableBoundθAt_all`, `FlockDeploy.KS.plain_KS`).

Where `verify --zk` sets its session up at a draw (`FlockDeploy.KS.setupAt`), the old `--zk` plan's tables are `zkTabs`
(`ZkPlanJ`: `FlockDeploy.KS.table_eq_zkTab`, `FlockDeploy.KS.planZJ_eq_tabsJ`), and so are the hidden-output plan's
(`ZkPlanHJ`: `FlockDeploy.KS.tableH_eq_zkTab`, `FlockDeploy.KS.planZHJ_eq_tabsJ`). `Deploy.V`'s checks pin the record's
draw to the statement's only up to key order, so the old discharge is read at the setup at the statement's draw, which
reads no draw in the record, with a writer that records the live session there (`ZkDrawJ`:
`FlockSoundness.Discharge.Composed.AcceptsAt`, `FlockSoundness.Discharge.Composed.runsAtZCJ_live`,
`FlockSoundness.Discharge.ZkHidden.runsAtZCJH_live`, `Claims.Bridge.Custody.wrS`,
`Claims.Bridge.Custody.recordsLiveAtS_wrS`), and an accepting run of `Deploy.V` is accepted there
(`Claims.Bridge.Custody.acceptsAt_wrS`). So at the empty draw the deployed `--zk` decoders are right at `zkTabs` at
public and hidden outputs (`FlockDeploy.KS.drawOkZKJ_empty`, `FlockDeploy.KS.decodesTable_zkTabs`,
`FlockDeploy.KS.decodesTableH_zkTabs`), and so at every masked statement in scope that `Deploy.V` sets up at a draw,
which every statement some run accepts carries (`FlockDeploy.KS.acc_draw_of_setup`; `ZkDecJ`:
`FlockDeploy.KS.decodesTable_deploy`), `ZkLiveAccAtJ`'s `hdec` at the verdict `accepted`. The masked session's bound
reads its level-0 term only on runs that do not self-clash, a self-clash being in `ε_c⁻` already (`ZkSessionSC`:
`FlockSoundness.Discharge.ZkLink.bad_pointwise_ofZ_sc`,
`FlockSoundness.Discharge.ZkLink.session_sound_of_table_accZ_sc`), so the masked and live sessions' verdicts need table
`j` accepted with no opening off `C` only unless it self-clashes (`FlockDeploy.KS.ZkSessJ`, `FlockDeploy.KS.ZkLiveAtJ`,
`FlockDeploy.KS.ZkLiveAccAtJ`). Read with the live outcomes whose decoded table `j` has no opening off `C` without a
self-clash (`ZkXferJ`: `FlockDeploy.KS.OffNoClashZ`, `FlockDeploy.KS.AccOffJ`, `FlockDeploy.KS.OkOffJ`,
`FlockDeploy.KS.decodesOff_deploy`), the deployed decoders are right at a verdict accepting with none, so `ZkLiveAccJ`
follows from the live sessions' bound at the deployed decoders (`FlockDeploy.KS.ZkXferJ`,
`FlockDeploy.KS.zkLiveAccJ_of_xferJ`) and `Flock.KS` from it and `ExtendJ` (`FlockDeploy.KS.zk_KS_of_xferJ`), table 0's
bound with escapes on accepted runs from it alone (`FlockDeploy.KS.zk_tableBoundθAccAt_of_xferJ`). Level 0's masked rows
carry both reps' extra lanes, which the session's level-0 term reads and the claim's witness-lane columns do not, so a
verifying opening with a nonzero extra lane is off every table of those columns (`ZkPads`:
`FlockDeploy.KS.off₀Z_oracleOf_of_extra`, `FlockDeploy.KS.offNoClashZ_oracleOf_of_extra`), while whether a table is
committed reads only its witness lanes (`FlockDeploy.KS.committed_congr_lanes`). So the masked session's bound holds
with its level-0 term on the witness lanes, the extra lanes the plurality of the openings on `C`'s, at the old level-0
term `√(2^{d₀}·2Q₀·Adv₀)` (`ZkSessionW`: `FlockSoundness.Discharge.ZkLink.OffW₀Z`,
`FlockSoundness.Discharge.ZkLink.session_sound_of_table_accZ_w`). The masked reader keeps each opened row's witness
lanes, so a masked output whose level-0 openings verify and are, with those lanes, the reader's entries opens a position
off `C` on a witness lane without a self-clash only where the reader looks up a column off `C` (`ZkOffSC`:
`Claims.Flock.exists_ne_of_offW₀Z_sc`, `Claims.Flock.exists_ne_of_zk_entries`). A compiled masked rep's first request is
level 0's, at the positions of the coins of its stream's round `zkQueryRound`
(`FlockSoundness.Discharge.ZkExec.zk_rep_reqs`), so at a rep's run its level-0 openings, on the witness lanes, are the
masked reader's entries, and an accepted masked table's level-0 openings verify at `width₀` (`ZkEntries`:
`Claims.Flock.slotOpensZ_entries`, `Claims.Flock.zkRepColsOf_ok`, `FlockSoundness.Discharge.ZkExec.zk_rep_entries`,
`Claims.Flock.verified_opens₀Z`); so where `Zk.verifyRep` accepts both reps of a table, every accepted output of their
results has its level-0 openings verifying at `width₀` below `2^d`, with positions and witness lanes its reps'
`zkRepColsOf` entries (`FlockSoundness.Discharge.ZkExec.zk_table_opens₀Z`), and so does every accepted output of an
outcome `verify --zk` accepts at the setup at the draw (`FlockSoundness.Discharge.Composed.OpensEntries`,
`FlockSoundness.Discharge.Composed.runsAtZCJ_live_entries`,
`FlockSoundness.Discharge.ZkHidden.runsAtZCJH_live_entries`). There, where the deployed masked reader looks up no column
of table `j` off `C`, table `j`'s accepted output opens no position off `C` on a witness lane without a self-clash
(`ZkDecodeW`: `FlockSoundness.Discharge.Composed.ZkTxNoOff`, `FlockSoundness.Discharge.Composed.not_offW_of_entries`,
`FlockSoundness.Discharge.Composed.runsAtZCJ_live_offW`, `FlockSoundness.Discharge.ZkHidden.runsAtZCJH_live_offW`), and
the deployed decoders are right at that verdict on those outcomes, at a masked statement in scope
(`FlockDeploy.KS.OkOffWJ`, `FlockDeploy.KS.AccTxJ`, `FlockDeploy.KS.decodesTableW_zkTabs`,
`FlockDeploy.KS.decodesTableHW_zkTabs`, `FlockDeploy.KS.decodesTableW_deploy`). On an accepting deployed masked run, the
event of no column of table `j` off `C` gives that verdict at the run's live outcome (`ZkEvent`:
`FlockDeploy.KS.cols_zk`, `FlockDeploy.KS.accTxJ_of_run`), with no premise on the record's draw, so a simulation of
the deployed masked prover by a live prover on coins of uniform law gives the masked mixture's first bound at that
verdict, and at the coins that merge each live coin with the rest of a deployed coin, a deployed event a live event
implies bounds the mixture (`ZkFirst`: `FlockDeploy.KS.ZkSimRun`, `FlockDeploy.KS.ZkSim`, `FlockDeploy.KS.zkCoins`,
`FlockDeploy.KS.zk_first_of_sim`, `FlockDeploy.KS.zk_first_of_run`, `FlockDeploy.KS.zkMix_le_of_run`).

The witness-lane bound's level-0 term is one more fork event per table, of weight `2·2^{d₀}·Q₀`. `Escapes.FitsFork`
charges every fork event at the busiest round's count, two with the conflict fork at round `0`, and at that count the
masked tables' weight spends level 0's share on `k`'s ceiling (`FlockDeploy.KS.cZ_sum_le_tableWeight`). Counted at
their own rounds, only round `0`'s events are charged twice, and the escapes' root is `Escapes.le`'s (`FitsR`:
`Claims.collapseR`, `Claims.Escapes.FitsR`, `Claims.Escapes.FitsForkR`, `Claims.Escapes.le_of_fitsR`,
`Claims.Escapes.Fits.fitsR`, `Claims.Escapes.FitsFork.fitsForkR`, `Claims.Escapes.fitsR_addFork`,
`Claims.Escapes.fitsForkR_of_round`). Joined by round with level 0's events (`ZkFitsW`: `Claims.Flock.IZW`,
`Claims.Flock.cZW`, `Claims.Flock.joinEscW`, `Claims.Flock.joinEscW_onAcc`), each table puts at most one event at
round `0` and every event but one weighs at least `2` (`Claims.Flock.joinEscW_c_le`), and each table's weight pays
exactly for its events with level 0's and its share of `k` (`FlockDeploy.KS.cZW_sum_le_tableWeight`), so the masked
escapes with level 0's fit the deployed weight with the fork event of weight `k`, counted per round
(`FlockDeploy.KS.joinEscW_fitsForkR_zk`). The fork bound's slack adds one fork event per round. Where round `0` holds
only level 0's event of each table, the fork of weight `k` and the joined escapes, round `0`'s charged twice, leave
every table's level caps' and pads roots' weights of twice the deployed masked weight (`ZkSlackW`:
`Claims.Flock.joinCW`, `Claims.Flock.roomW`, `Claims.Flock.joinCW_zero_le`, `Claims.Flock.roomW_add`,
`FlockDeploy.KS.zk_fork_room`), and the slack's events merged by round with the joined escapes, at weights
`(√J_t + √S_t)²`, fit the deployed weight per round where the slack fits that room (`Claims.Flock.slackEscW`,
`Claims.Flock.mergeCW`, `Claims.Flock.mergeCW_le`, `Claims.Flock.SlackFitW`, `FlockDeploy.KS.slackEscW_fitsForkR_zk`),
on accepted first runs (`Claims.Escapes.OnAcc₁`, `Claims.Flock.slackEscW_onAcc₁`), each table's fork terms at the
slack under their term (`Claims.Flock.slackEscW_term`). At one tape and table, the deployed masked runs accept with no
column of table `j` off `C` with probability at most `ε_table` and those escapes' term when the mixture bounds them by
its acceptance with no level-0 opening off `C` on a witness lane and each fork term's mean `Adv` by its event's
probability and the slack (`ZkBoundW`: `FlockDeploy.KS.mean_epsSZ_le_rawJ`, `FlockDeploy.KS.zk_le_sessionsJW`), so the
masked sessions bounding every table at that verdict with events on accepted runs give every masked table's bound with
the escapes counted per round on accepted first runs (`FlockDeploy.KS.ZkSessWJ`,
`Claims.Flock.TableBoundJAccAtR`, `Claims.Flock.TableBoundJAccAt.tableBoundJAccAtR`,
`FlockDeploy.KS.zk_tableBoundJAccAtR_of_sessionsW`). The live sessions' bound at that verdict, with level 0's events
bounding the simulated strategies' `Adv₀` after `Commit`, gives those sessions (`ZkLiveW`:
`FlockDeploy.KS.ZkLiveAccAtWJ`, `FlockDeploy.KS.zkSessWJ_of_liveW`, `FlockDeploy.KS.zk_tableBoundJAccAtR_of_liveW`),
so at the deployed masked record every table's bound and table 0's with escapes on accepted runs counted per round
follow from `ZkLiveAccWJ` where some run accepts (`FlockDeploy.KS.ZkLiveAccWJ`, `FlockDeploy.KS.zk_tableBoundJAccAtR`,
`Claims.Flock.TableBoundθAccAtR`, `Claims.Flock.TableBoundJAccAtR.tableBoundθAccAtR`,
`FlockDeploy.KS.zk_tableBoundθAccAtR`). Escapes fitting per round fit as `Escapes.Fits` asks once their fork events are
merged by round, with a term no smaller (`JointR`: `Claims.Escapes.mergeR`, `Claims.Escapes.FitsR.fits_mergeR`,
`Claims.Escapes.term_le_mergeR`), so `OneWait` follows from every table's bound counted per round
(`Claims.Flock.TableBoundJR`, `Claims.Flock.oneWait_of_tableBoundJR`, `Claims.Flock.tableBoundJR_of_extend`), and the
deployed masked verifier is KS from `ZkLiveAccWJ` where some run accepts and `ExtendJ`
(`FlockDeploy.KS.zk_KS_of_tableBoundJR`, `FlockDeploy.KS.zk_KS_of_liveAccWJ`). At a masked statement in scope table
`j`'s word has level 0's `2^logLen` columns (`ZkXferW`: `FlockDeploy.KS.zkLevel0_d_eq`,
`FlockDeploy.KS.wordLenJ_zkTabs`), so the live sessions' bound at the deployed decoders right at the witness-lane
verdict (`FlockDeploy.KS.ZkXferWJ`) gives `ZkLiveAccWJ` (`FlockDeploy.KS.zkLiveAccWJ_of_xferWJ`): with `ExtendJ`,
`Flock.KS` at the deployed masked record (`FlockDeploy.KS.zk_KS_of_xferWJ`), and alone, table 0's bound counted per
round (`FlockDeploy.KS.zk_tableBoundθAccAtR_of_xferWJ`). Every table's bound counted per round carries to the trimmed
record by padding back (`ZkTrimW`: `Claims.Flock.OkThroughJ`, `Claims.Flock.colsOkJ_mapCols`,
`Claims.Flock.conflictTiedJ_mapCols`, `Claims.Flock.tableBoundJR_mapCols`), where `[]` is the trim of table `j`'s zero
column at a position of its word (`FlockDeploy.KS.okThroughJ_nil`), so with no `ExtendJ`
(`FlockDeploy.KS.zk_tableBoundJR_trim`): `Flock.KS` at the trimmed masked record follows from every table's bound
counted per round at full-width columns (`FlockDeploy.KS.zk_KS_trim_of_tableBoundJR`), so from `ZkLiveAccWJ` or
`ZkXferWJ` where some run accepts alone (`FlockDeploy.KS.zk_KS_trim_of_liveAccWJ`,
`FlockDeploy.KS.zk_KS_trim_of_xferWJ`). With an accepting run or fork with its collision as the events, `ZkXferWJ`
holds from a live prover for each prover, with a finite tape and coins of uniform law, that sends its roots, simulates
its accepting runs, whose self-clashes give its accepting runs with a collision and whose fork bounds hold (`ZkSimW`:
`FlockDeploy.KS.zkRoot`, `FlockDeploy.KS.zkRoot_of_first`, `FlockDeploy.KS.ZkClashRun`, `FlockDeploy.KS.ZkSimWJ`,
`FlockDeploy.KS.zkXferWJ_of_simWJ`), with no premise on the record's draw, so `Flock.KS` at the trimmed masked record
and table 0's bound counted per round follow from it (`FlockDeploy.KS.zk_KS_trim_of_simWJ`,
`FlockDeploy.KS.zk_tableBoundθAccAtR_of_simWJ`). The recursion's chain at the abort law holds with the table bound
counted per round once the escapes with the conflict fork are merged by round (`ZkAbortR`:
`Claims.Recursion.TableBoundθOnAccR`, `Claims.Recursion.EscapesD.mergeR`,
`Claims.Recursion.oneWaitAbort_of_tableBoundR`, `Claims.Recursion.KSAbort_of_tableBoundOnAccR`), so the masked record
the claims read meets `Inner.KSAbort`, and the recursion's KS over it holds, from that bound and one table wherever
some run accepts (`FlockDeploy.KS.zk_KSAbortR`, `FlockDeploy.KS.zk_KSFlockR`), so from a simulator of each prover
(`FlockDeploy.KS.zk_KSAbort_of_simWJ`, `FlockDeploy.KS.zk_KSFlock_of_simWJ`). At every number of tables, with no
premise of one table, one wait over draws holds at every table's positions (`ZkAbortJR`: `Claims.Recursion.oneWaitDJ`),
so the chain holds from every table's bound counted per round (`Claims.Recursion.TableBoundJOnAccR`,
`Claims.Recursion.oneWaitAbort_of_tableBoundJR`, `Claims.Recursion.KSAbort_of_tableBoundJOnAccR`), and the masked
record the claims read meets `Inner.KSAbort`, and the recursion's KS over it holds, from that bound alone
(`FlockDeploy.KS.zk_KSAbortJR`, `FlockDeploy.KS.zk_KSFlockJR`), so from a simulator of each prover
(`FlockDeploy.KS.zk_KSAbort_of_simWJ_tables`, `FlockDeploy.KS.zk_KSFlock_of_simWJ_tables`). `ZkSimWJ`'s
self-clash clause follows from a live self-clash giving an accepting run whose reps open two different columns at one
position of some table, or a collision: each such column is a level-0 opening in the run's own log (`ZkClashRaw`:
`FlockDeploy.KS.zkRawColsJ`, `FlockDeploy.KS.zk_openingJ_raw`, `FlockDeploy.KS.zk_selfClash_tiedJ`,
`FlockDeploy.KS.ZkRawClashRun`, `FlockDeploy.KS.zkClashRun_of_raw`). At chances at most `ε ≤ 1/2` before the last round
and at most `1` there, a
fork's slack weights before the last round fall geometrically past its round (`PlainSlackFit`:
`Claims.Flock.sum_slackW_le`, `Claims.Flock.δN_le_inv`, `Claims.Flock.δN_le_one`). So each table's slack weights total
at most its weight times `|IZW|·8ε` (Cauchy–Schwarz), and at C-Flock ZK's schedules, with at most `7` levels, level 1
a query and level 0's weight under `2^40`, the joined weights are at most `2^41` times the room and the slack fits it
at chances at most `2^{-128}`, so at the coupling's chances where every revealed length before the last round is
positive (`ZkSlackFit`: `Claims.Flock.sum_slackCW_le`, `Claims.Flock.sum_slackTW_le`, `Claims.Flock.slackFitW_of_le`,
`Claims.Flock.joinCW_le_room`, `FlockDeploy.KS.m1_slackNums`, `FlockDeploy.KS.zk_slackFitW`,
`FlockDeploy.KS.zk_slackFitW_δN`). The masked simulator runs each prover round by round at uniform functions of the
live coins' history, its rounds on the masked wire at their stream and count, with each table's fork rounds at its
reps' caps, pads and `τ` and level 0 at the link coin; it meets `ZkSimWJ`, so masked KS and the recursion's premise
hold, from four named Props: its lockstep, its abort, its fork bounds and a view's self-clash giving a column clash
(`ZkSim`: `FlockDeploy.KS.zkSim`, `FlockDeploy.KS.zkΦ`, `FlockDeploy.KS.zkRd`, `FlockDeploy.KS.zk_simFork_le`,
`FlockDeploy.KS.ZkSimRunAt`, `FlockDeploy.KS.ZkSimAbortAt`, `FlockDeploy.KS.ZkSimForkLe`,
`FlockDeploy.KS.ZkViewClash`, `FlockDeploy.KS.zkSimWJ_of_parts`, `FlockDeploy.KS.zk_KS_of_parts`,
`FlockDeploy.KS.zk_tableBoundJOnAccR_of_parts`, `FlockDeploy.KS.zk_KSFlock_of_parts`). Its lockstep and abort hold at
every statement: it sends `P`'s messages at `zkΦ`'s coins along the live history and aborts where `V true` rejects
(`ZkLock`: `FlockDeploy.KS.zkSim_liveOut`, `FlockDeploy.KS.zkSimAbortAt`), and where `P`'s run accepts its live
outcome is the record's view, the record's rounds read back stream by stream (`ZkStep`: `FlockDeploy.KS.zkSim_reps`,
`FlockDeploy.KS.zkSimRun`, `FlockDeploy.KS.zkSimRunAt`), so masked KS and the recursion's premise rest on its fork
bounds and view clashes alone (`FlockDeploy.KS.zk_KS_of_fork`, `FlockDeploy.KS.zk_tableBoundJOnAccR_of_fork`,
`FlockDeploy.KS.zk_KSFlock_of_fork`). The view's clash splits along the self-clash's three cases (`ZkClash0`:
`FlockDeploy.KS.ZkViewClashQ`, `FlockDeploy.KS.zkViewClash_of_Q`). At level 0, where the clash compares whole rows,
the deployed decoders are right at the reps' opened rows at the view (`ZkRows`: `FlockDeploy.KS.decodesTableR_deploy`),
which are the run's rows, each a Merkle opening in its log, so two that differ at one position collide there
(`FlockDeploy.KS.rawRows_zk`, `FlockDeploy.KS.zk_rowClash_tiedJ`, `FlockDeploy.KS.zkViewClashQ₀`); masked KS and the
recursion's premise rest on the fork bounds and the view's clashes past level 0 and at the pads
(`FlockDeploy.KS.zk_KS_of_later`, `FlockDeploy.KS.zk_tableBoundJOnAccR_of_later`,
`FlockDeploy.KS.zk_KSFlock_of_later`). Past level 0, each of an accepted masked rep's later requests is one level's
(`ZkLevelsAll`: `FlockSoundness.Discharge.ZkExec.zk_rep_reqsL`), whose queries sit at round `zkLevelRound` of the rep's
stream and pass the level's Merkle check against the proof's recursive cap (`ZkLevels`: `FlockDeploy.KS.zkLevelRound`,
`FlockDeploy.KS.zkLevels_verifyWithM`), so two of one rep's level-`ℓ` rows at one position that differ collide in its
log (`FlockDeploy.KS.zk_levelClash_tiedJ`). The decoders are right at those rows at the view too
(`FlockSoundness.Discharge.ZkExec.zk_rep_rowsL`), so the view's clash past level 0 is such a collision (`ZkClashL`:
`FlockDeploy.KS.zkViewClashQL`), and masked KS and the recursion's premise rest on the fork bounds and the view's pads
clash alone (`FlockDeploy.KS.zk_KS_of_pads`, `FlockDeploy.KS.zk_tableBoundJOnAccR_of_pads`,
`FlockDeploy.KS.zk_KSFlock_of_pads`). At level 0 the masked simulator's fork bound holds (`ZkFork0`:
`FlockDeploy.KS.zkSimForkLe₀`): `Adv₀` after `Commit` is two independent runs, at most two independent live runs
decoding to a level-0 clash (`FlockDeploy.KS.adv₀SZ_simAtJZ_le`, through the simulation after `j` coins of `ForkSim`),
which are two accepting runs of `P` whose rows at one position differ and collide in their logs
(`FlockDeploy.KS.zkPair₀`). So masked KS and the recursion's premise rest on the cap/`τ` and pads fork bounds and the
pads clash (`FlockDeploy.KS.zk_KS_of_kinds`, `FlockDeploy.KS.zk_tableBoundJOnAccR_of_kinds`,
`FlockDeploy.KS.zk_KSFlock_of_kinds`). At the caps the fork bound splits as plain's does (`ZkFork`): against the
simulated strategy after `Commit`, the fork at a cap's table node is at most the fork after `zkRd j i` coins of two
runs recording one cap and opening a position of it to different rows (`FlockDeploy.KS.ZkForkAlign`), and two
accepting runs whose live outcomes decode to that collide in their logs (`FlockDeploy.KS.ZkForkClash`;
`FlockDeploy.KS.zkSimForkLeCap_of`, `FlockDeploy.KS.zk_KS_of_align`). The first is the fork after the slot's coins
(`ZkAlign`: `FlockDeploy.KS.ZkSlotAlign`, `FlockDeploy.KS.zkForkAlign_of_slot`, `FlockDeploy.KS.zk_KS_of_slot`), since
pushing a table's answer does not change a fork (`FlockSoundness.Discharge.ZkLink.forkC_pushStrat`). A masked rep
meets each cap's slot after a count fixed by the statement and the schedule (`RepDistZ`:
`FlockSoundness.Discharge.ZkLink.dist_lockRepZ`; `ZkSched`: `FlockDeploy.KS.dist_lockRepZ_zkCap`,
`FlockDeploy.KS.zkRdAt_cap`), plain's past level 0's lane rounds and extra draws. A rep whose opening rejects skips
Ligerito (`restZC`, `PendL.reject`), so rep 1 and the later tables start after fewer coins on that run, but
`advRSZ`'s event reads the openings only on runs accepting every table up to the session's, and is `0` past a rejected
one (`FlockDeploy.KS.ind_conflictAcc_dead`). So the slot is counted on the simulated strategy's schedule alone: the
tables in front of the session's advance to their sends and wait, then the session's table plays rep 0 first
(`ZkBatch`: `FlockDeploy.KS.ZkBatchAlign`, `FlockDeploy.KS.zkFrontAlign_of_batch`). The masked simulator submits its
rounds stream by stream, each table's rep 0 then its rep 1 (`ZkLiveSched`: `FlockDeploy.KS.zkLive_sched`,
`FlockDeploy.KS.zkStreams_split`), so its simulated strategy follows that schedule at the deployed tables' counts, and
`ZkSlotAlign` holds at every message (`ZkBatchSim`: `FlockDeploy.KS.schedC_simAtJZ`, `FlockDeploy.KS.zkBatchAlign`,
`FlockDeploy.KS.zkSlotAlign`). The clash holds at every message too: at an accepting run's view each rep records its
decoded proof's recursive caps and opens rows of them in the run's log, so a cap two runs record is one root, and two
rows they open at one position of it collide (`ZkForkClash`: `FlockDeploy.KS.zkAccOpening`,
`FlockDeploy.KS.zkForkClash`). The pads and `τ` split the same way, into an alignment and a clash, and both clashes
hold at every message (`ZkPadsPair`: `FlockDeploy.KS.PadsForkEvZ`, `FlockDeploy.KS.zkPairPads`,
`FlockDeploy.KS.zkSimForkLePads_of`; `ZkTauFork`: `FlockDeploy.KS.ZkTauAlign`, `FlockDeploy.KS.zkSimForkLeTau_of`;
`ZkTauPair`: `FlockDeploy.KS.zkTauClash`, `FlockDeploy.KS.zkSimForkLeTau_of_align`). A rep's pads root is its first
table node, so the pads' fork round is the caps' schedule at distance `0`, and their alignment holds at every message
(`ZkPadsSlot`: `FlockDeploy.KS.slotPadsZ_of_batch`, `FlockDeploy.KS.zkPadsAlign`,
`FlockDeploy.KS.zkSimForkLePads`). So masked KS rests on `τ`'s alignment alone (`ZkKSAlign`:
`FlockDeploy.KS.zk_KS_of_tauAlign`). `τ`'s commitment is two coins into the inner proof, past the masked phase, whose
coins are fixed only on the runs whose opening can end well; on the others the rep's Ligerito part is `PendL.reject`
and the table's output is not accepted, so it opens no `τ`. A distance on those paths (`SlotDistOn`:
`FlockSoundness.Game.Lock.DistOn`, `FlockSoundness.Game.Lock.SlotAtD.lead_fst_on`,
`FlockSoundness.Game.Lock.SlotAtD.lead_snd_on`) and the schedule short of the runs whose table opens nothing or is not
accepted (`FlockDeploy.KS.schedC_simAtJZ_dead`) give `τ`'s alignment at every message, its count `zkPre + zkLig + 2`
coins into a rep (`ZkTauSlot`: `FlockDeploy.KS.zkTau_dist`, `FlockDeploy.KS.slotTauZ_of_batch`,
`FlockDeploy.KS.zkTauAlign`, `FlockDeploy.KS.zkSimForkLeTau`). So masked KS holds with no premise
(`FlockDeploy.KS.zk_KS`). KS at an untrimmed instance gives it at its trimmed record
(`Flock.TrimBridge.KS_mapCols_of_KS`), but not conversely: an unopened position reads `[]`, the trim of every zero
column and equal to no full-width one (`Flock.TrimBridge.hole_agree`). Both follow from the one full-width table bound,
the untrimmed with `ExtendJ` (`Flock.TrimBridge.KS_both_of_tableBoundJR`), so from `zk_KS_of_kinds`'s premises
(`FlockDeploy.KS.TrimBridge.zk_KS_both_of_kinds`). Those hold at every message, so the untrimmed instance's KS rests
on `ExtendJ` alone (`ZkUntrim`: `FlockDeploy.KS.zkSimForkLeCapTau`, `FlockDeploy.KS.zk_KS_untrim_of_extendJ`). And
`ExtendJ` holds at the untrimmed instance at every relation: an unopened position takes a full-width column whose first
lane no trace one agreement short of its radius has there, which Johnson's bound leaves room for in GF(2^128), so the
table still does not extract (`Extend`: `Claims.GS.exists_avoid`, `Claims.Flock.fill_step`,
`Claims.Flock.extendJ_of_fill`; `ZkExtend`: `FlockDeploy.KS.zk_extendJ`). So masked KS at the untrimmed instance holds
with no premise too (`FlockDeploy.KS.zk_KS_untrim`). The same fill, at the plain code, extends plain's one-table form
(`Claims.Flock.extend_of_fill`), which carries plain's table bound to columns that may be zero
(`Claims.Flock.tableBoundθ_of_extend`), so plain KS holds at the untrimmed record too, with no premise (`PlainUntrim`:
`FlockDeploy.KS.plain_extend`, `FlockDeploy.KS.plain_KS_untrim`). The chain reads the verifier only through its setup (`FlockDeploy.KS.ZkSetup`), and
its setup-dependent part is restated over one (`FlockDeploy.KS.Gen`), `deployZk`'s at `FlockDeploy.KS.S₀`. At the
recursion's outer setup (`FlockDeploy.KS.SO`: the recursive setup, and the verdict's last refusals after the checks)
the chain's instance is the outer verifier's record (`ZkOuter`: `FlockDeploy.KS.deployZk_SO`), so masked KS holds
there with no premise (`FlockDeploy.KS.zk_KS_outer`). -/
