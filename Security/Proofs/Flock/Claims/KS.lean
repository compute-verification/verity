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
import Proofs.Flock.Claims.KS.PlainSlackFit
import Proofs.Flock.Claims.KS.PlainSim
import Proofs.Flock.Claims.KS.PlainLock
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.KS.PlainParts
import Proofs.Flock.Claims.KS.PlainClash0
import Proofs.Flock.Claims.KS.PlainLevels
import Proofs.Flock.Claims.KS.LevelsAll
import Proofs.Flock.Claims.KS.PlainClashLater

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
(`FlockDeploy.KS.oldEsc_fitsFork_plain`, `FlockDeploy.KS.oldEscZ_fitsFork_zk`; the plain weight with level 0's share to
spare, `FlockDeploy.KS.plain_fork_room`), so the salted and masked sessions'
`ε_c⁻` are at most `ε_table` and their term (`FlockDeploy.KS.epsSL_le`, `FlockDeploy.KS.epsSZ_le`), also averaged over
a mixture of strategies (`FlockDeploy.KS.mean_epsSL_le`, `FlockDeploy.KS.mean_epsSZ_le`). The plain fork rounds
(`Rounds`): rep `r`'s level-`ℓ + 1` cap round at table 0, no two of them one (`FlockDeploy.KS.plainJ`,
`FlockDeploy.KS.plainJ_inj`), at which the old escapes fit the plain weight (`FlockDeploy.KS.oldEsc_fitsFork_plainJ`).
At one table the model does not commit to, a live verdict is no likelier than the simulated strategy's `ε_c⁻` after
`Commit` and its probability of accepting off the table (`Mixture`: `FlockDeploy.KS.live_le_epsSL`; over a mixture of
live strategies, `FlockDeploy.KS.mean_live_le`). The session's bound with its level-0 term only on runs that do not
self-clash, a self-clash being in `ε_c⁻` already (`SessionSC`: `FlockSoundness.Discharge.Exec.bad_pointwise_ofL_sc`,
`FlockSoundness.Discharge.Exec.session_sound_of_table_accL_sc`). At a plain statement of one table, `TableBoundθAt`
follows from, at each table `C`, a mixture of salted sessions after `Commit` whose acceptance with no opening off `C`
unless it self-clashes bounds the deployed runs accepting on `C`, and whose `Adv` and self-clash the escapes' events
bound, the escapes at the plain fork rounds (`PlainBound`: `FlockDeploy.KS.plain_tableBoundθAt_of_sessions`).
`Flock.KS` at the trimmed plain record follows from the live sessions' bound at table 0 (`PlainKS`:
`FlockDeploy.KS.plain_KS_of_live`). The self-clash after `Commit`
against the simulated strategy arises only on live runs the live verdict accepts, whose decoded outputs self-clash
(`Escapes`: `FlockDeploy.KS.selfClash_le_live`, `FlockDeploy.KS.mean_selfClash_le_live`). Escapes on accepted runs
(`OnAcc`, `Claims.Escapes.OnAcc`): `Flock.TableBoundθAccAt` is `TableBoundθAt` with them, which gives it
(`Flock.TableBoundθAccAt.tableBoundθAt`) and holds with none where no run accepts (`Flock.tableBoundθAccAt_of_noAcc`);
the old escapes are on accepted runs where their events are (`Flock.oldEsc_onAcc`, `Flock.oldEscZ_onAcc`). With the
live sessions' bound at escapes on accepted runs, whose events bound the decoded self-clashes on accepted live runs
(`PlainAcc`: `FlockDeploy.KS.PlainLiveAcc`, which gives `PlainLive`, `FlockDeploy.KS.plainLive_of_acc`), the plain
table bound holds with escapes on accepted runs at every statement (`FlockDeploy.KS.plain_tableBoundθAccAt`).
At `--zk`, one table's compiled session is `Commit`'s cap and then `sessAtZ` with no other game, so a live verdict at
one masked table is no likelier than any verdict after `Commit` the decoder is right for (`MixtureZk`:
`FlockDeploy.KS.live_le_simAtZ`, `FlockDeploy.KS.mean_live_le_simAtZ`), and at a masked statement of one table,
`TableBoundθAt` follows from a mixture of masked sessions after `Commit` at a table `T` at C-Flock ZK's parameters, the
escapes `Flock.oldEscZ` at injective fork rounds, and no table that does not extract committed to `T` (`ZkBound`:
`FlockDeploy.KS.zk_tableBoundθAt_of_sessions`, `FlockDeploy.KS.zk_tableBoundθAt_of_live`, at the live sessions' bound
`FlockDeploy.KS.ZkLiveAt`). At the deployed masked record, `T` is C-Flock ZK's padded table of the statement's table 0
(`ZkKS`: `FlockDeploy.KS.zkTab`, `FlockDeploy.KS.zkTab_m1`), so `Flock.KS` at the trimmed record follows from one
table at every statement some run accepts (`FlockDeploy.KS.ZkOne`, which the masked scope's several tables do not give)
and the live sessions' bound at `zkTab` (`FlockDeploy.KS.ZkLive`, `FlockDeploy.KS.zk_KS_of_live`). At several tables
(`JointJ`), one wait follows from every table's bound with one family of escapes, over every table's word's positions
(`Flock.TableBoundJ`, `Flock.ColsOkJ`, `Flock.ConflictTiedJ`, `Flock.oneWait_of_tableBoundJ`), and at the masked record
accepted runs open table `j`'s columns at its width and as its level-0 openings (`ZkTables`: `FlockDeploy.KS.wordLenJ`,
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
(`Flock.TableBoundJAccAt`, `FlockDeploy.KS.zk_tableBoundJAccAt`). Each round of an accepted plain rep issued the coins
its statement fixes (`RoundsN`: `FlockDeploy.repNs`, `FlockDeploy.verifyRepWith_readsN`; per rep stream,
`FlockDeploy.plainRepNs`, `FlockDeploy.streams_ns_plain`), and custody reads each round message's coin at the session's
count (`FlockDeploy.plainNs`, `FlockDeploy.custody_vals_plain`). So under `--zk` (`RoundsNZk`): each round of an
accepted masked rep issued its statement's coins (`FlockDeploy.zkRepNs`, `FlockDeploy.zkVerifyRepWith_readsN`; per rep
stream, `FlockDeploy.zkRepNsOf`, `FlockDeploy.streams_ns_zk`), and custody reads each masked round message's coin at the
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
(`DecodeDeploy`: `FlockDeploy.KS.plain_hdec`). With hidden outputs, the hidden table is `plainTab` too, and the same
holds at its openings (`DecodeHidden`: `Claims.Bridge.DecodePlain.decodesTable_wrP_scH`, `FlockDeploy.KS.plain_hdecH`).
A statement some run accepts sets up at a draw, so the plain chain asks only for the mixture at those decoders, whether
or not the outputs are hidden (`PlainWrP`: `FlockDeploy.KS.acc_draw`, `FlockDeploy.KS.PlainLiveWrPAt`,
`FlockDeploy.KS.PlainLiveW`, `FlockDeploy.KS.plain_KS_of_w`). On an accepting deployed run whose record carries the
statement's draw, the event the mixture's first bound reads gives that verdict at the run's live outcome (`PlainEvent`:
`FlockDeploy.KS.cols_plain`, `FlockDeploy.KS.plainVerdict_of_run`). A prover that plays the live session with the rest
of each deployed coin on its tape bounds an event of the deployed coins by a mixture of live strategies, the merged
coins being uniform, and the mixture by an event of the deployed coins the live event implies (`LiveMix`:
`Claims.Bridge.Live.Pr_mergeC`, `Claims.Bridge.Live.Pr_mergeAt`, `Claims.Bridge.Live.liveMix_eq`,
`Claims.Bridge.Live.Pr_le_liveMix`, `Claims.Bridge.Live.liveMix_le_Pr`). From a simulation of the deployed prover by a
live prover on coins of uniform law, and the record's draw on accepting runs, the mixture's first bound holds
(`PlainFirst`: `FlockDeploy.KS.PlainSim`, `FlockDeploy.KS.RecordDrawVerbatim`, `FlockDeploy.KS.plain_first_of_sim`), and
at the coins that merge each live coin with the rest of a deployed coin, from the simulation of the runs alone
(`FlockDeploy.KS.PlainSimRun`, `FlockDeploy.KS.plainCoins`, `FlockDeploy.KS.plainSim_of_run`,
`FlockDeploy.KS.plain_first_of_run`); the checks compare draws only up to `JsonEquiv`, which holds between different
terms (`FlockDeploy.KS.drawEq_ne`), so acceptance gives the draw only up to it (`FlockDeploy.KS.RecordDrawEquiv`,
`FlockDeploy.KS.recordDrawEquiv`). At those coins, a deployed event that the live event implies bounds the mixture, the
third bound's form (`PlainThird`: `FlockDeploy.KS.plainMix_le_of_run`). With an accepting run or fork with its collision
as the events, KS at the deployed plain verifier's trimmed record holds from a live prover for each prover, with a
finite tape and coins of uniform law, that simulates its accepting runs, whose self-clashes give its accepting runs with
a collision and whose fork bound holds up to a slack, and the record's draw verbatim (`PlainSimW`:
`FlockDeploy.KS.PlainClashRun`, `FlockDeploy.KS.PlainSimWrPAt`, `FlockDeploy.KS.PlainSimW`,
`FlockDeploy.KS.plain_tableBoundθAt_of_sim`, `FlockDeploy.KS.plain_KS_of_sim`). The slack is the later rounds' forks and
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
`FlockDeploy.KS.plainVerdictAt_of_run`). The acceptance lemmas the plain decoders read hold at a draw value by their
proofs (`PlainDrawAcc`: `FlockSoundness.Discharge.AtDraw.accepts_oneAnyD`,
`FlockSoundness.Discharge.AtDraw.tableAt_of_acceptsD`), so the plain decoders decode the verdict at the draw
(`PlainDrawDec`: `Claims.Bridge.PlainDraw.decodesTable_wrPD_sc`, `FlockDeploy.KS.plain_hdecAt`,
`FlockDeploy.KS.plain_hdecHAt`), and KS at the deployed plain verifier's trimmed record holds from a simulator at
that verdict alone, with nothing asked of the record's draw (`PlainDrawSim`: `FlockDeploy.KS.plain_first_of_simD`,
`FlockDeploy.KS.PlainSimWrPAtD`, `FlockDeploy.KS.PlainSimWD`, `FlockDeploy.KS.plain_KS_of_simD`). At the coupling's
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
(`FlockDeploy.KS.PlainSimAbortAt`, `FlockDeploy.KS.plainSimAbortAt`). So KS at the deployed plain verifier's trimmed
record holds from two parts of the concrete simulator: its live self-clashes give `P`'s accepting runs a raw clash, and
its fork advantage is at most its simulated fork (`PlainParts`: `FlockDeploy.KS.PlainSimClashAt`,
`FlockDeploy.KS.PlainSimForkLe`, `FlockDeploy.KS.plainSimWrPAtD_of_parts`, `FlockDeploy.KS.plain_KS_of_parts`).
The level-0 case of that clash holds: a level-0 self-clash there is two of the reps' entries at one position with
different columns, which are the run's raw columns, so the clash needs only its case past level 0 (`PlainClash0`:
`FlockDeploy.KS.plainClashQ₀`, `FlockDeploy.KS.PlainSimLaterClashAt`, `FlockDeploy.KS.plain_KS_of_later`).
Past level 0, an accepted plain rep passed every later level's Merkle check on its proof's rows at the positions of
that level's query round, against the flattened recursive cap before it, so two rows a rep opens at one later-level
position that differ clash in the run's log (`PlainLevels`: `FlockDeploy.KS.plainLevels_verifyWithM`,
`FlockDeploy.KS.plain_levelClash`). On the model's side, the same accepted rep's requests after level 0's are
levels `1, …, R` in order, each at the positions of the coins of its query round (`LevelsAll`:
`FlockSoundness.Refine.LaterAt`, `FlockSoundness.Refine.find?_later`, `FlockSoundness.Refine.rep_refines_reqs`).
So a self-clash past level 0 at the concrete simulator's outcome is two of one rep's level-`ℓ` entries at one position
with different columns, which are the run's level-`ℓ` rows and clash in its log, and KS at the deployed plain
verifier's trimmed record holds from the simulator's fork bound alone (`PlainClashLater`:
`FlockSoundness.Discharge.AtDraw.accepts_opensLD`, `FlockDeploy.KS.plainClashQL`,
`FlockDeploy.KS.plainSimLaterClashAt`, `FlockDeploy.KS.plain_KS_of_fork`).
-/
