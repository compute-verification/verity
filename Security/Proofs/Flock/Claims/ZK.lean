import Proofs.Flock.Claims.ZK.Statement
import Proofs.Flock.Claims.ZK.GK
import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Coins
import Proofs.Flock.Claims.ZK.Congr
import Proofs.Flock.Claims.ZK.Pin
import Proofs.Flock.Claims.ZK.Forced
import Proofs.Flock.Claims.ZK.Rewind
import Proofs.Flock.Claims.ZK.GKProof
import Proofs.Flock.Claims.ZK.GKCost
import Proofs.Flock.Claims.ZK.GKExp
import Proofs.Flock.Claims.ZK.CoinCheck
import Proofs.Flock.Claims.ZK.CoinBind
import Proofs.Flock.Claims.ZK.Prefix
import Proofs.Flock.Claims.ZK.Shvzk
import Proofs.Flock.Claims.ZK.Flock
import Proofs.Flock.Claims.ZK.Hypotheses
import Proofs.Flock.Claims.ZK.Prefinal
import Proofs.Flock.Claims.ZK.Deployed
import Proofs.Flock.Claims.ZK.Reference
import Proofs.Flock.Claims.ZK.CoinLayer
import Proofs.Flock.Claims.ZK.CoinOpen
import Proofs.Flock.Claims.ZK.Tape
import Proofs.Flock.Claims.ZK.Capacity
import Proofs.Flock.Claims.ZK.Session
import Proofs.Flock.Claims.ZK.MaskedTables
import Proofs.Flock.Claims.ZK.Masked
import Proofs.Flock.Claims.ZK.MaskTape
import Proofs.Flock.Claims.ZK.MaskFamily
import Proofs.Flock.Claims.ZK.OneTape
import Proofs.Flock.Claims.ZK.Round0
import Proofs.Flock.Claims.ZK.ExeLayout
import Proofs.Flock.Claims.ZK.Refusals
import Proofs.Flock.Claims.ZK.FirstRefusal
import Proofs.Flock.Claims.ZK.Stopped
import Proofs.Flock.Claims.ZK.HiSalts
import Proofs.Flock.Claims.ZK.SaltProver
import Proofs.Flock.Claims.ZK.CutSession
import Proofs.Flock.Claims.ZK.ExePlan
import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.SimCommit
import Proofs.Flock.Claims.ZK.PrefixLive
import Proofs.Flock.Claims.ZK.SimSession
import Proofs.Flock.Claims.ZK.SimInner
import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.InnerSlotTriple
import Proofs.Flock.Claims.ZK.InnerLeaf
import Proofs.Flock.Claims.ZK.InnerY
import Proofs.Flock.Claims.ZK.InnerBatched
import Proofs.Flock.Claims.ZK.InnerTauVal
import Proofs.Flock.Claims.ZK.InnerTauRep
import Proofs.Flock.Claims.ZK.InnerTauLig
import Proofs.Flock.Claims.ZK.Assemble
import Proofs.Flock.Claims.ZK.Sent
import Proofs.Flock.Claims.ZK.Round1
import Proofs.Flock.Claims.ZK.Round1Pads
import Proofs.Flock.Claims.ZK.Round1Root
import Proofs.Flock.Claims.ZK.Round1Enc
import Proofs.Flock.Claims.ZK.Round2
import Proofs.Flock.Claims.ZK.Round3Vals
import Proofs.Flock.Claims.ZK.SimZerocheck
import Proofs.Flock.Claims.ZK.SimPiopGo
import Proofs.Flock.Claims.ZK.SimPiopOf
import Proofs.Flock.Claims.ZK.PiopSrcVals
import Proofs.Flock.Claims.ZK.PiopSrcLc
import Proofs.Flock.Claims.ZK.SentE
import Proofs.Flock.Claims.ZK.PiopSrcSv
import Proofs.Flock.Claims.ZK.SimPiop
import Proofs.Flock.Claims.ZK.SimLig
import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.FinPads
import Proofs.Flock.Claims.ZK.CausalExe
import Proofs.Flock.Claims.ZK.LiveEnd
import Proofs.Flock.Claims.ZK.FinTau
import Proofs.Flock.Claims.ZK.FinInner
import Proofs.Flock.Claims.ZK.Worker
import Proofs.Flock.Claims.ZK.PadFree
import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server
import Proofs.Flock.Claims.ZK.Blind
import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Claims.ZK.Refine
import Proofs.Flock.Claims.ZK.CoinPass
import Proofs.Flock.Claims.ZK.Os
import Proofs.Flock.Claims.ZK.FrameTape
import Proofs.Flock.Claims.ZK.FrameParse
import Proofs.Flock.Claims.ZK.FrameMask
import Proofs.Flock.Claims.ZK.FrameRounds
import Proofs.Flock.Claims.ZK.FrameOpen
import Proofs.Flock.Claims.ZK.FrameOpenCut
import Proofs.Flock.Claims.ZK.FrameOpenLive
import Proofs.Flock.Claims.ZK.FramePhase
import Proofs.Flock.Claims.ZK.FrameState
import Proofs.Flock.Claims.ZK.FrameRun
import Proofs.Flock.Claims.ZK.FrameShape
import Proofs.Flock.Claims.ZK.FrameOwn
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Claims.ZK.FrameSched
import Proofs.Flock.Claims.ZK.FrameProg
import Proofs.Flock.Claims.ZK.FrameCount
import Proofs.Flock.Claims.ZK.FrameAsks
import Proofs.Flock.Claims.ZK.FramePlan
import Proofs.Flock.Claims.ZK.FrameLinks
import Proofs.Flock.Claims.ZK.FrameReal
import Proofs.Flock.Claims.ZK.FrameRep
import Proofs.Flock.Claims.ZK.FrameInner
import Proofs.Flock.Claims.ZK.FrameZK
import Proofs.Flock.Claims.ZK.FrameLive
import Proofs.Flock.Claims.ZK.PadsRoot
import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.Claims.ZK.SamplingNonVacuous
import Proofs.Flock.Claims.ZK.Compose
import Proofs.Flock.Claims.ZK.Layout
import Proofs.Flock.Claims.ZK.Cut
import Proofs.Flock.Claims.ZK.Stop
import Proofs.Flock.Claims.ZK.LinkFft
import Proofs.Flock.Claims.ZK.LinkRecheck
import Proofs.Flock.Claims.ZK.FinishRound
import Proofs.Flock.Claims.ZK.FrameTaken
import Proofs.Flock.Claims.ZK.FrameRank
import Proofs.Flock.Claims.ZK.FramePoints
import Proofs.Flock.Claims.ZK.FinishCommit
import Proofs.Flock.Claims.ZK.FinishLevel0
import Proofs.Flock.Claims.ZK.FinishPiop
import Proofs.Flock.Claims.ZK.FinishInner
import Proofs.Flock.Claims.ZK.FinishLig
import Proofs.Flock.Claims.ZK.FinishRep
import Proofs.Flock.Claims.ZK.FinishRank
import Proofs.Flock.Claims.ZK.AdmitsOf
import Proofs.Flock.Claims.ZK.FrameAdmits
import Proofs.Flock.Claims.ZK.FrameBeta
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FrameFree
import Proofs.Flock.Claims.ZK.FrameRanked
import Proofs.Flock.Claims.ZK.FrameRankFault
import Proofs.Flock.Claims.ZK.FrameRho
import Proofs.Flock.Claims.ZK.FrameUnrefused
import Proofs.Flock.Claims.ZK.FinishSrc
import Proofs.Flock.Claims.ZK.FinishBridge
import Proofs.Flock.Claims.ZK.FinishTrunc
import Proofs.Flock.Claims.ZK.FinishLive
import Proofs.Flock.Claims.ZK.FinishStop
import Proofs.Flock.Claims.ZK.FinishStopAt
import Proofs.Flock.Claims.ZK.FinishSwSrc
import Proofs.Flock.Claims.ZK.FinishSwSv
import Proofs.Flock.Claims.ZK.FinishSwMk
import Proofs.Flock.Claims.ZK.FinishSwMkAny
import Proofs.Flock.Claims.ZK.FinishFwL0
import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.FinishSwLig
import Proofs.Flock.Claims.ZK.FinishSwLigE
import Proofs.Flock.Claims.ZK.L0Pin
import Proofs.Flock.Claims.ZK.L0PinCoins
import Proofs.Flock.Claims.ZK.L0PinFrame
import Proofs.Flock.Claims.ZK.L0PinQuiet
import Proofs.Flock.Claims.ZK.L0PinFlags
import Proofs.Flock.Claims.ZK.L0PinSwitch
import Proofs.Flock.Claims.ZK.L0PinRec
import Proofs.Flock.Claims.ZK.L0PinBind
import Proofs.Flock.Claims.ZK.L0PinLive
import Proofs.Flock.Claims.ZK.L0PinChain
import Proofs.Flock.Claims.ZK.L0PinLoop
import Proofs.Flock.Claims.ZK.L0PinCut
import Proofs.Flock.Claims.ZK.FinHmkLive
import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.FinishSwTau
import Proofs.Flock.Claims.ZK.FinishSwRows
import Proofs.Flock.Claims.ZK.FinishSwFw
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.FinishSwBit
import Proofs.Flock.Claims.ZK.FinishSwRep
import Proofs.Flock.Claims.ZK.FinishClaimsX
import Proofs.Flock.Claims.ZK.FinishSwVals
import Proofs.Flock.Claims.ZK.FinishRows
import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinishLane
import Proofs.Flock.Claims.ZK.HelloLean
import Proofs.Flock.Claims.ZK.SessionLean
import Proofs.Flock.Claims.ZK.RepLog
import Proofs.Flock.Claims.ZK.CutPrefix
import Proofs.Flock.Claims.ZK.PiopFree
import Proofs.Flock.Claims.ZK.RepFree
import Proofs.Flock.Claims.ZK.RepReplay
import Proofs.Flock.Claims.ZK.RepInner
import Proofs.Flock.Claims.ZK.RepCutFw

/-!
# Flock's zero knowledge against a malicious verifier

* `Statement`: what the proofs add to the spec's `ZKmalR` and `Claims.Flock.ZK`: ZK bits under `StrictBirthday` at
  `Fork` (`ZKmalR.bits`), the worst-case cost `SimCostLe`, and `Claims.Flock.ZKAt`, the claim's form at a given prover
  and simulator (`ZK_iff_at`: `Claims.Flock.ZK` is it at the record's own).
* `GK`: a committed-coin protocol (`Proto`), its honest prover, its Goldreich–Kahan simulator with `N` rewinds
  (`Proto.simulator`), and the three facts the theorem reads (`PrefixHides`, `SHVZK`, `Binding`).
* `Interact`: what the programs evaluate to: `runStar` and the simulator's runs as one interaction (`runStar_msgs`,
  `spRun_eval`), two provers that agree on a transcript's histories give it to both (`interact_take_congr`), and each
  round's calls of the prover are in the run's `pLog`, so a collision among two runs' calls is `Fork`'s (`Fork_pr_ge`).
* `Coins`: what the coin checks evaluate to: a transcript opens `c` exactly when every opening is answered and passes
  the check (`cls_eq_some_iff`), the forced walk agrees (`cls_eq_some_iff_fcls`), and a mismatch is two checked
  openings within the walk's calls (`fcls_eq_mism`).
* `Congr`: `ZKmalR` carries over to a prover with the same messages and more calls within the rounds
  (`zkmalR_congr`).
* `Pin`: `Claims.Flock.ZK` reads the masked record's own reference prover (`ZkDeploy.prover`) and simulator; the proofs
  prove `Claims.Flock.ZKAt` at a committed-coin protocol's prover. The masked record at a protocol `G` (`zkOf`) has
  `ZkTables.honestPR` behind `G`'s `Hello` and opening checks; where `G`'s messages are `ZkTables.honestPR`'s (`Proto.Honest`, a named
  Prop), `ZKAt` at `G.prover` is `Claims.Flock.ZK` at `zkOf` (`zk_zkOf`).
* `Forced`: the forced run at `c` with the honest prover's messages at `c` opens all of `c` exactly when the real run
  opens `c`, and is then the real run (`fcls_succ_iff`, `forced_eq_of_cls`, `real_eq_of_fcls`); one that stops before a
  binding failure is a real run that opens no coin vector (`cls_none_of_abort`).
* `Rewind`: the rewinding argument over abstract runs: the first run is within `1/(N+1)` plus a binding failure of the
  first successful rewind (`rewind_close`), `N` independent samples are within `N·δ` (`Pr_hybrid`), and an event of
  one of `n` samples is at most `n` times one sample's (`Pr_exists_coord_le`).
* `GKProof`: **`Proto.gk_zk`**, the Goldreich–Kahan simulator meets `ZKmalR` at `1/(N+1) + δ_A + N·δ_B` and `γ = N` from
  `PrefixHides`, `SHVZK` and `Binding`; a binding failure in a rewind is, in the real game, a collision among the honest
  prover's calls in two of its runs on one nonce (`mism_coll`).
* `GKCost`: its cost, `SimCostLe` at `simBound` under `CostLe` (`Proto.gk_cost`).
* `GKExp`: **`Proto.gkE_zk`**, the session analysis's simulator with estimation (`gk_simulate`, threshold `t`, cap `M`)
  as a `Claims.Simulator` meets `ZKmalR` at `4·δ_A + 4t·δ_B + 3^{-t} + t/M` and `γ = 4M` (`2·3^{-t}` for the last two
  and `γ = 4t·3^t` at `M = t·3^t`, `gkE_zk_pow`) from the same three facts, through the old hybrids on the protocol's
  `Model` at each nonce, hybrid 1's binding failure moved to the real run (`rwMism_coll`); its cost is `(1 + 5t)` runs in expectation at every cap (`SimCost`, `gkE_costExp`) and `1 + 5M` at worst
  (`SimCostLe`, `gkE_cost`).
* `CoinCheck`: the coin-tree check (`coin-tree/hm96-sha512/v2`) as an oracle program that logs its hash calls
  (`treeChk`); a protocol whose check it is meets `Binding` with no hypothesis on `H` (`Proto.binding_of_tree`), T6's
  aligned hash inputs being among the two walks' calls (`coinPairs_mem`).
* `CoinBind`: the same where the coin is read from the whole message: two different passing openings of one position
  are a collision among the two walks' calls (`GK.coinPairs_coll_any`), so a protocol whose check is the walk on the
  message read at its round, and whose reading is one coin for two messages that read as one opening, meets `Binding`
  (`Proto.binding_of_open`).
* `Prefix`: runs of a Claims `V*` as runs of the session analysis's model: the committed-coin prover's first `K + 1`
  messages as a causal strategy (`Proto.preStrat`) against `V*`'s responder (`vResp`), whose run is a transcript's
  first `K + 1` messages (`run_prefix`) and determines what the extraction run passes on (`view0_run`); so
  `PrefixHides` follows from a bound on the strategy's runs (`Proto.prefixHides_of_runs`), which Lemma C gives at
  `2·|L|·δ₁` (`Proto.runs_close_hm96`).
* `Shvzk`: `SHVZK` at every coin vector: from Lemma B where the prover goes on (`Proto.shvzk_at_hm96`, `2·n_hid·δ₁`,
  with `InnerHolds`), and from Lemma C at fixed answers where it refuses, the simulator sending the dummy-witness
  prover's messages (`Proto.shvzk_refused_hm96`, `2·|L|·δ₁`).
* `Flock`: **`Flock.zkR_of_proto`**, the spec's `Claims.Flock.ZK` for a masked deployed verifier whose rounds with
  `Hello` are the committed-coin protocol's (`D.V.k + 1 = G.k`), which `FlockDeploy.deployZk`'s are (`deployZk_proto_k`,
  `Flock.zkR_deployZk`); `S_Flock`'s cost (`Flock.simCost_of_proto`) and the expected-cost simulator's; what changes
  for a malicious verifier, how the hypotheses map to the session analysis's Theorem GK (`gk_simulate_hm96`), and what
  the `--zk` instance still needs.
* `Hypotheses`: what the `--zk` instance still needs, as five Props about models of the masked prover (`MsgLink`,
  `SimLink`, `NonDegenerate`, `InnerHolds`, `CoinCheck`), and the claim from them under `Claims.Hm96Hiding`
  (`Flock.zk_of_models`; `Flock.zkAt_deployZk_of_models` at `FlockDeploy.deployZk`; **`Flock.zk_deployZk_of_models`**
  at its masked record `zkOf`, under `Proto.Honest`).
* `Prefinal`: Lemma C from world `W₀` alone, for any family of tables of their own types (`famCard_cut_eq`), and its
  bounds on the runs before the last coin and where the prover refuses (`Proto.runs_close_of_ideal`,
  `Proto.refused_close_of_ideal`).
* `Deployed`: the same at the tables `verify --zk` sets up, each with its own types (`Proto.TPreModel`,
  `Proto.TabModel`, `Proto.TOffModel`), where `NonDegenerate` and `InnerHolds` are `CoinedOK`'s (`SetupTable.ok`,
  `ZkHidden.zkViewAtJ_of_ok`), Lemma C's tables, leaves and world `W₀` are the session's by definition
  (`card_cutAt_ideal`, `exeOf_eq`) and the simulator is the models' (`Proto.withSim`, so `SimLink` holds by
  construction): **`Flock.zk_deployZk_of_tables`** under `Claims.Hm96Hiding` and three Props (`TMsgLink`, `TablesOK`,
  `CoinCheck`, or `Binding` itself in `_bind`), and under those alone at the pinned key's hiding gap
  (`Flock.zk_deployZk_of_tables_gap`).
* `Reference`: the reference prover those three Props are about, `RefSession.refP`, defined from the session's data
  (its round encoding of the cut view, its final message of the tables' real views, refusing where a table doesn't
  admit the coins); `Binding` holds of it with no hypothesis (`Proto.binding_of_open`), `TablesOK` from `Tables` and
  `TMsgLink` from `Complete`, `DummyReg`, `Causal` and `TapeOK` (`viewPre_viewExe`: the real views' pre-final part is
  the cut view the round messages encode): `Flock.zkAt_deployZk_ref` at `refP`'s prover, and `Flock.zk_zkD_ref` at the session's masked record
  `RefSession.zkD` under `RefSession.HonestMsgs` (`refP`'s messages are `ZkTables.honestPR`'s, a named Prop), each also at the
  pinned key's hiding gap (`_gap`).
* `CoinLayer`: the deployed masked record `zkDeployed`, `RefSession.zkD` at `coin-tree/hm96-sha512/v2`'s coin layer
  (the 32-byte nonce, `Hello`'s frame with its hex, `Hello`'s answer of root and hm96 key, the keyed tree, the
  committed schedule's places, the opening read from `Resp::Coins`'s words, the coin handed on its words), and
  **`Flock.zk_deployZk_ref`** at it, at every rep rounds' coin counts; `Claims.Witness.Flock.ZK_meets` states its coin
  layer and `ZK_coins_fails` that the degenerate one (every opening passes, the coin is the opening) is not it.
* `CoinOpen`: the Lean coin check (`flock-firewall coins`, `Flock.Firewall.Coins`), whose verdict `coin_tree::Checked`
  takes, is that coin layer's at SHA-512: one opening (`CoinOpen.treeChk_sha`, `CoinOpen.check_isOk`), and a session's
  openings when its requests follow the committed schedule (`CoinOpen.coinLayer_verdict`), so it refuses the openings
  `LeanRelease.OpeningsRefused` (`Claims/ZK/Refine.lean`) names (`CoinOpen.coinLayer_refuses`). The release
  reads it (`LeanRelease.opens`), and at that coin layer it is the session's check (`LeanRelease.checkIs_coinLayer`).
* `Tape`: `TapeOK` makes the tape onto the leaves' salts (`RefSession.eP_surjective`), so a session has fewer tables than
  tapes at every statement (`RefSession.J_lt_card_of_tapeOK`), and none with statements of every table count meets it
  (`RefSession.not_tapeOK`): the masked scope bounds no table count.
* `Capacity`: the tape asked for on a class `A` of statements only (`RefSession.TapeOKOn`, `RefSession.HonestMsgsOn`,
  the latter at the relation's witnesses, the only ones zero knowledge reads the prover at), as a tape of bounded size
  can meet it where `A` bounds the tables, at the record whose parser refuses the rest
  (`deployOn`): `Flock.zk_zkD_ref_on`, `Claims.Flock.ZK` at `RefSession.zkD` at `Flock.zk_zkD_ref`'s `ε_zk` on `A`
  and `1` outside it (`zkmalR_on`). `RefSession.tapeOKOn_of_tape`: `TapeOKOn A` from the tape read with fibers of one
  size at each statement in `A`, each table's randomness being read from it so (`fibersEq_tabRand`). With it,
  **`Flock.zk_deployZk_ref_on`** (and `_gap`) at `zkDeployed`, the tape asked for only as the tables' tape read with
  fibers of one size on `A`, where `Flock.zk_deployZk_ref`'s `TapeOK` fails at every unbounded table count.
* `Session`: `RefSession.HonestMsgsOn A R` as its three rounds (`HonestPreOn`, `HonestFinOn`, `HonestOffOn`: `enc`,
  `fin` and `off` each `ZkTables.honestPR`'s message at a witness in `R`), and `Flock.zk_deployZk_ref_on_rounds` at them,
  at the relation.
* `MaskedTables`: the deployed masked session's tables (`Masked.tabC`): `setupOf true x`'s, each `coinedOf` at its
  `--zk` table from its statement alone (`ZkHidden.ZkTab.table`), on the statements each of whose tables has one
  (`Masked.Live`, the masked scope at a witness: `live_iff`). `Tables` (`setupTable_tabC`), `DummyReg` at a witness of
  the regions (`dummyReg_tabC`) and `Complete` at each table's share (`complete_tabC`, from `Readable.tableLowers`;
  its one-table case `Readable.oneTableLowers`) hold there.
* `Masked`: the masked session's table data (`Masked.tableData`) at those tables, a round plan (`RoundPlan`: coins,
  round encodings) and a tape plan (`TapePlan`). **`Flock.zk_deployZk_masked`**: `Claims.Flock.ZK` at `zkDeployed`
  there at `ε_zk` on the masked scope, under the round plan's `Causal`, the tape plan read with fibers of one size on
  `Live`, and the three rounds; `Flock.zk_deployZk_masked_at`, at one statement's tape (`TapePlan.at`), without the
  tape premise; each at the pinned key's hiding gap with no hiding premise (`…_gap`).
* `MaskTape`: the tape plan whose masked prover's tape carries the tables' mask words (`Masked.TapePlan.atMask`,
  `TapeZk.mask` the packing of each table's base's slot bits), at which each table's share of `ZkTables.honestPR`'s
  witness is the tables' `zWith` (`Masked.zOf_maskOf`); `Flock.zk_deployZk_masked_atMask_gap` at it, and
  `Flock.zk_deployZk_masked_atMask_stop_gap` at a stopping round plan, its third round `honestPR`'s alone.
* `MaskFamily`: the masked claim on a class of statements (`Flock.zk_deployZk_masked_on`), whose tape bounds the class's
  table count (`Masked.jOf_lt_card_on`; no tape at every `Live` statement once they have every table count:
  `Masked.not_fibersEq_live`), and on statement-indexed tapes: `ZKfam`, `Claims.ZKmalRx`'s body at a family of masked
   records, at the mask tape of each statement with no tape premise (`Flock.zk_deployZk_masked_atMask_rx`, `…_stop_rx`).
* `OneTape`: the masked scope caps the table count at `zkTableCap` (`Masked.jOf_le_cap`), so one tape serves every
  statement: a cell per table (`Masked.TapeLayout`, read as each table's randomness with fibers of one size, its mask
  words in `TapeZk.mask`), the tape `Masked.TapePlan.ofLayout`, read with fibers of one size at every statement
  (`Masked.fibersEq_ofLayout`); `Flock.zk_deployZk_masked_layout_gap` is `Flock.zk_deployZk_masked_gap` with `hT`
  discharged, under the round plan's `Causal` and the three rounds.
* `Round0`: toward the first round, level 0's tree at a `--zk` table places every position at its index at every coin
  vector (`Composed.coinedOf_s0_val`), so its column at tree position `p` is the code at `ω_p` (`Composed.colAt`,
  `Composed.coinedOf_col0At`), the same at every coin vector; such a column's `row0` is `encodeRowsZk`'s row
  (`FlockComplete.row0_eq_encodeRowsZk`), and every `--zk` schedule's level 0 has the sizes this reads
  (`Masked.zkLevel0`). Round `0`'s encoding `Masked.enc0` reads no coin (`Masked.enc0_causal`) and is `honestPR`'s
  first message where `hOf H` is SHA-512, at a layout agreeing with the executable's tape at round `0`
  (`Masked.honest_round0`, `Masked.Agrees0`).
* `ExeLayout`: the layout at the executable's indices (`Masked.TapeLayout.exe`): a cell per table holding each
  `TapeZk` field as an array at its executable indices (`Masked.ExeCell`), each statement reading its tables'
  coordinates at the indices its sizes give, with fibers of one size (`Masked.fibersEq_exeRead`), levels ≥ 1's salts
  by level and position (`Masked.hiIdx`); the masked prover's tape read from the cells (`Masked.exeTape`, PoW nonces
  `0`). The layout agrees with it at round `0` (`Masked.agrees0_exe`), so `HonestPreOn` there is its rounds after the
  first (`Masked.honestPreOn_exe_of_pos`, `RefSession.HonestPreOnPos`), and `Flock.zk_deployZk_masked_exe_gap` is
  `Flock.zk_deployZk_masked_layout_gap` at this layout with round `0` discharged.
* `Refusals`: the last round where a table refuses, at a stopping round plan (`RoundPlan.stop`, `off = []`), refusal by
  refusal (`Masked.honestOffOn_stop_of`): an outer coin `1` is `honestPR_stop`'s at a plan reading each rep's outer
  coins at their slots (`Masked.OuterReads`, met by any plan with its outer coins so read, `Masked.withOuter`;
  `Masked.outerHit_of`); `β = 0`, a rank-deficient mask slot and `ρ₀, ρ₁`
  dependent are one premise each. `Flock.zk_deployZk_masked_exe_stop_gap` is `Flock.zk_deployZk_masked_exe_gap` at a
  stopping plan with them in place of `HonestOffOn`.
* `FirstRefusal`: `β = 0`, `ρ₀, ρ₁` dependent and a rank-deficient mask slot, from the session's first refusal: at a
  plan reading each rep's `β`, pair coins and claim points at the executable's rounds (`Masked.CoinReads`,
  `Masked.PointReads`, met by `Masked.withPoints`), a refused one leaves the masked session not returning
  (`FlockComplete.ZkTables.sessionPZk_pass`), so `honestPR` sends `[]` (`Masked.honestPR_refused`,
  `Masked.honestPR_rank`). `Flock.zk_deployZk_masked_exe_reads_gap` is `Flock.zk_deployZk_masked_exe_stop_gap` without
  the first two premises, `Flock.zk_deployZk_masked_exe_refusals_gap` without all three.
* `Stopped`: the rounds after a refusal the coins show (`Masked.Refused`: an answered outer coin `1`, or a throw whose
  coins are all answered, `Masked.Thrown`), which reads the coins below its round only (`Masked.refused_congr`). A plan
  whose encoding sends `[]` there (`Masked.encStop`) is causal where its encoding is (`Masked.causal_encStop`), and its
  `HonestPreOnPos` is its encoding's at the rounds where no refusal is seen, given that a seen throw stops `honestPR`
  (`Masked.honestPreOnPos_stop`; a seen outer coin `1` is `Masked.outerHit_of_lt`'s).
  `Flock.zk_deployZk_masked_exe_stopped_gap` is `Flock.zk_deployZk_masked_exe_refusals_gap` at such a plan. A seen
  throw does stop `honestPR` (`Masked.honestPR_thrown`: the session stops right after the send whose reply it refused,
  within the openings' answers), so `Flock.zk_deployZk_masked_exe_live_gap` drops that premise: left are the encoding
  at the rounds after the first where no refusal is seen, the plan's `Causal`, its round `0` and the last round.
* `Sent`: at a round after the first where no refusal is seen, `honestPR` sends the masked session's next body at the
  openings' answers, tagged `0` (`Masked.honestPR_live`, from `FlockComplete.ZkTables.honestPR_sent`), so
  `Flock.zk_deployZk_masked_exe_sent_gap` asks the encoding there against that run (`hstep`): that the session sends
  past the openings and the encoding is its next body, or that it stops within them and the encoding is `[]`
  (`Masked.honestPR_stopped`), as where table `0`'s extra claims throw right after the link message.
* `Round1`: round `1` of that run at a live statement (`Masked.liveRun_round1`): where table `0`'s extra claims at the
  link points throw, the session stops after the link message; else its second body is table `0` rep `0`'s first
  zerocheck squeeze, of the domain's transcript with the digest, the level-0 cap and the pads' root absorbed
  (`FlockComplete.sessionPZk_go_round1`).
* `Round1Pads`: toward that body on the model's side, a live table's rep `r` pads leaf at tree position `p` commits
  `padColAt` at `X = p + 1`, each pad at its layout index and padding symbol `j` at `len + j`, at every coin vector
  (`Masked.colPAt_tabC`), so its digest reads no coin (`Masked.digOf_pads`); a cut's pads leaf at tree position `q`
  (`Masked.leafAtP_cutAt`).
* `Round1Root`: and on the executable's side, `innerCommitP`'s root is `layerOf`'s of its leaves
  (`FlockComplete.innerCommitP_root_layerOf`); at the executable's layout each is the executable's pads leaf of the cut's
  view at that tree position (`Masked.exeOf_pads`), so table `j`'s rep `r` pads root is `layerOf`'s of the cut's pads
  leaves by tree position (`Masked.innerRoot_exe`).
* `Round1Enc`: round `1`'s encoding (`Masked.enc1`): that body of the cut's level-0 and pads leaves by tree position, at
  the link points of round `0`'s coin, and `[]` where the extra claims there throw. It reads round `0`'s coin only
  (`Masked.enc1_causal`), and at the executable's layout it is round `1`'s `hstep` (`Masked.honest_round1_exe`), so
  `Flock.zk_deployZk_masked_exe_round1_gap` asks `hstep` after the second round only.
* `Round2`: round `2` of that run (`Masked.liveRun_round2`): where table `0`'s extra claims return, its third body is
  table `0` rep `0`'s outer zerocheck point squeeze, the header of `m − 13` coins of kind `2` alone, which reads no
  value, commitment or coin; its encoding (`Masked.enc2`) reads round `0`'s coin only (`Masked.enc2_causal`) and is
  round `2`'s `hstep` at every tape (`Masked.honest_round2_exe`), so `Flock.zk_deployZk_masked_exe_round2_gap` asks
  `hstep` after the third round only.
* `Round3Vals`: toward round `3` (`SimPiop`'s zerocheck), the clear values a cut masks at `round1_ab`'s and
  `round1_c`'s layout indices are the executable's `round1_ab` and `round1_c` at the draw's zerocheck vector, on a trace
  whose bits are the model's tables (`Composed.honestAt_r1ab_exe`, `Composed.honestAt_r1c_exe`, from
  `Replay.Zc.r1ab_eq`); a rep's outer coin is read at the round after its first (`Masked.outerAt_eq`).
* `SimZerocheck`: `SimPiop`'s zerocheck: two runs in lockstep (`FlockComplete.GoAlike`), `asksCut`'s squeezes, and the
  cut's zerocheck against the session's where the cut's masked values are the plain prover's plus their pads
  (`FlockComplete.zerocheckCut_alike`).
* `SimPiopGo`: the rest of `SimPiop`'s PIOP in lockstep: the cut's lincheck against the session's
  (`FlockComplete.lincheckCut_alike`), its ring switch (`FlockComplete.ringSwitchCut_alike`), and the cut's PIOP
  returning the session's from a source at the cut's coins (`FlockComplete.PiopSrc`, `FlockComplete.piopCut_alike`).
* `SimPiopOf`: `Masked.SimPiop` from the deployed source at the opened coins (`Masked.PiopSrcE`, `Masked.simPiop_of`):
  `repMix 0` and `repMix 1` differ only in the PIOP (`FlockComplete.repMix_zero_one`).
* `PiopSrcVals`: `Masked.PiopSrcE`'s zerocheck fields: the deployed source masks at the zerocheck's layout indices
  the plain prover's messages at the opened coins plus the session's pads (`Masked.src_r1ab`, `Masked.src_r1c`,
  `Masked.src_ml0`, `Masked.src_ml1`, `Masked.src_fa`, `Masked.src_fb`, `Masked.src_fc`), since the model's tables are
  the session's trace's bits (`Masked.trsE_bits`) and the draw's honest zerocheck is the executable's
  (`Masked.zcH_exe`).
* `PiopSrcLc`: `Masked.PiopSrcE`'s lincheck fields: the deployed source masks at the lincheck's layout indices the
  plain prover's round messages and `z_partial` at `lcCoins` plus the session's pads (`Masked.src_lc0`,
  `Masked.src_lc1`, `Masked.src_zp`), since the prover's halved columns are the model's `lcU`, `lcV` folded at the round
  coins (`Replay.Lc.msg_lcMsg`) and the draw's coins are the executable's (`Masked.lcH_exe`).
* `SentE`: every check of the deployed cut up to a seen slot passes where no refusal is seen (`Masked.famPasses_of`):
  each check's round (`Masked.guardSlot`) follows the session's order (`Masked.guardSlot_mono`), and the rank check is
  full rank at the plan's points, the executable's (`Masked.rankOn_of`).
* `PiopSrcSv`: `Masked.PiopSrcE`'s ring-switch field (`Masked.src_sv`): where the rep sends masked claim `2r + k`'s
  `ŝ` (`Masked.famPasses_of`) it is `shat` at `z = (w, u)` at the opened coins (`Masked.sv_masked`), and region claim
  `k − 2`'s is the witness's off the mask slot at the link points of round `0`'s coin (`Masked.sv_region`,
  `Masked.extras_x`), each the prover's switched vector (`Replay.Rs.switchVec_shat`).
* `SimPiop`: `Masked.simPiop`, from `Masked.simPiop_of` at `Masked.piopSrcE`.
* `HiSalts`: levels ≥ 1's salts by level and position (`Masked.hiSalt`, at `Masked.hiIdx`): at the executable's
  layout, the salts the model hands table `j`'s rep `r` are the masked prover's tape's `salt j r l p` on the schedule's
  levels `l ≥ 1` and their positions (`Masked.hiSalt_exe`), so a Ligerito prover of levels ≥ 1 built from one reading
  salts by level and position (`Masked.ofSalts`) sends, there, what it sends on the tape's salts
  (`Masked.ofSalts_exe`, given `Masked.SaltsOn`).
* `SaltProver`: `HiSalts`' definitions, and the deployed levels ≥ 1 prover's salts (`Masked.PcExe`: the tape's salts
  `192` bytes each, by level and position, read back by `Masked.blobSalt`), which read the schedule's levels and
  positions only (`Masked.saltsOn_PcExe`).
* `CutSession`: the masked session reading what it sends from a source beside it (`FlockComplete.sessionCut`,
  `FlockComplete.RepSrc`, `FlockComplete.TabSrc`), stopped after a round (`FlockComplete.asksCut`), and the hybrids
  between a rep from the source and the session's (`FlockComplete.repMix`).
* `ExePlan`: the deployed round plan: every coin at its executable round (`Masked.czExe`, extending
  `Masked.withPoints`), the encoding as `sessionCut`'s body on the cut (`Masked.encExe`, `Masked.tabSrcCut`), the last
  message as its record on the view (`Masked.finExe`, `Masked.tabSrcView`).
* `Split`: the open premises of `Flock.zk_deployZk_masked_exe_sent_gap` at the deployed plan, as eight statements
  (`Masked.PrefixLive`, `Masked.SimPiop`, `Masked.SimLig`, `Masked.SimInner`, `Masked.SimCommit`, `Masked.SimSession`,
  `Masked.CausalExe`, `Masked.FinExe`).
* `SimCommit`: `Masked.simCommit`, the commitments from the cut's leaves (`Masked.SimCommit`).
* `PrefixLive`: `Masked.prefixLive`, the session answered by the openings against the one stopped after the round.
* `SimSession`: `Masked.simSession_of`, `SimSession` from `SimCommit` and the three hybrids.
* `SimInner`: `Masked.simInner_of`, `SimInner` from what the cut's source carries for the inner proof
  (`Masked.InnerSrcE`).
* `SplitInner`: `InnerSrcE` as statements at its hypotheses (`Masked.InnerAt`): the slot count, the triple's openings,
  the batched constraint, `τ`'s value and leaf, and `Y` (`Masked.InnerSlot`, `Masked.InnerTriple`,
  `Masked.InnerBatched`, `Masked.InnerTauVal`, `Masked.InnerTauLeaf`, `Masked.InnerY`), and `SimInner` from them
  (`Masked.simInner_of_split`).
* `InnerSlotTriple`: `Masked.innerSlot_holds` and `Masked.innerTriple_holds`, `InnerSlot` and `InnerTriple` from
  `SimInner`'s `innerPre_slot` and `innerTripleE`.
* `InnerLeaf`: `Masked.innerTriple` and `Masked.innerTauLeaf`, the cut's triple openings and `τ` commitment read off
  the tape, and `y_τ`'s bytes as the honest pads' `τ` salt (`Masked.ytau_saltBytes`).
* `InnerY`: `Masked.innerY`, the cut's `Y` is the session's at the honest pads once `β`'s round is answered.
* `InnerBatched`: `Masked.innerBatched`, the batched constraint over the shared phases' constraints holds at the honest
  pads.
* `InnerTauVal`: `Masked.innerTauVal_of`, `InnerTauVal` from the inner proof's seven rows before the triple
  (`Masked.TauRows`), split as the replay's five (`Masked.TauRep`) and level 0's two (`Masked.TauLig`).
* `InnerTauRep`: `Masked.tauRep`, the replay's three rows and the two ring-switch rows are `rowsOf`'s on the view and
  `czExe`'s coins (the run being the session's, `Masked.innerPre_cutE`); `Masked.innerTauVal_of_tauLig : TauLig →
  InnerTauVal`.
* `InnerTauLig`: `Masked.tauLig`, level 0's two lane rows are `limbRows` of the model's `laneRow` at `czExe`'s coins
  (`Masked.lanesA`), so **`Masked.innerTauVal`**: `InnerTauVal`.
* `SimLig`: `Masked.simLig`, hybrid 1 against hybrid 2: Ligerito from the cut's source is the session's.
* `FinExe`: `FinExe` from `LiveEnd`, `SimCommit`, the three hybrids, `RepTau` and the opened positions' fields
  `RepOpenRest` (`Masked.finExe_of_rest`); the rep count and rep 0's claim points (`Masked.repExact`,
  `Masked.repPoints`); and `FinalRest`'s level-0 half: level 0's opened rows and salts read from the views are the
  session's (`Masked.repOpenRest_l0`).
* `FinPads`: `Masked.repOpenRest_pads`, `FinalRest`'s pads half: the pads' opened columns and salts read from the views
  are the session's.
* `LiveEnd`: `Masked.liveEnd`, the session answered by the openings ends as the one answered by the opened coins.
* `FinTau`: `Masked.repTau_of`, `FinExe`'s `RepTau` from `InnerTauVal`: the session's rep's `τ` is `innerP`'s after the
  shared phases, at the rep's `ε` slot's coins.
* `FinInner`: `Masked.repOpenRest` (both halves of `FinalRest`) and `Masked.finExe_of_inner`, `FinExe` from `LiveEnd`,
  `SimCommit`, the three hybrids and `InnerTauVal`.
* `CausalExe`: `Masked.causalExe`, `CausalExe`: every executable send reads only coins drawn before it.
* `Assemble`: **`Flock.zk_deployZk_masked_exe_split`**, Flock's zero knowledge at the deployed masked session given
  only that `H` is SHA-512 and `0 < t`: `hstep` from `Masked.prefixLive` and `SimSession` (`Masked.hstep_exe`),
  `SimSession` from `Masked.simCommit`, `Masked.simPiop`, `Masked.simLig` and `SimInner` (`Masked.simSession_of`),
  `SimInner` from `SplitInner` (`Masked.simInner_of_split`, `τ`'s value `Masked.innerTauVal`), `Causal` by
  `Masked.causalExe`, and the last message from `Masked.liveEnd` and `Masked.innerTauVal` (`Masked.finExe_of_inner`).
* `Worker`: Lemma B at any clear values a worker supplies, which never read the pads: the switched prover (`p`'s view
  where its rows hold, else the honest prover's on the pads Lemma A moves) has `S_shvzk`'s distribution on the honest
  `InnerHolds` alone (`sw_shvzk`, `worker_shvzk`), and differs from `p`'s view only in `Y₁`, `C_h` and the pads tree
  (`swView_outer`). It is the honest prover at the worker's values rectified, which pass the inner constraint at every
  pad, on a tape moved by a bijection (`swView_sameDist_rect`, `rect_innerHolds`), and Lemma B at such row-satisfying
  values is `table_shvzk`'s (`vals_shvzk`); each also with real hm96 leaves (`worker_shvzk_hm96`, `vals_shvzk_hm96`).
* `PadFree`: the inner constraint at a worker's values reads no pad: each row's value at the masked values and the
  drawn pads is its value at the clear values and the zero pads (`rows_padFree`), so the drawn pads' check passes iff
  `RowsAt` holds (`rowsAt_iff_pass`), and the Finish's stop bit is `1[¬RowsAt (w, v)]`, a function of the witness and
  the values alone.
* `Switched`: the switched prover at the session's `J` tables: Lemma B at the workers' values there (`zkViewAtJ_sw`),
  and `SHVZK` against the honest prover's simulator where the prover goes on (`Proto.TabModel.shvzk_sw`).
* `Server`: the ZK server (`ZkServer`) that releases untrusted workers' values in four kinds (`fixed`, `own`,
  `recomputed`, `masked`), masks added after its checks, on a public schedule. For every worker it releases the
  reference prover's run cut at some round, but with probability the masked checks' errors (`ZkServer.split`), from
  two obligations of its definition (`Ref`, `CheckSound`); so zero knowledge carries over within a factor `k + 1`, the
  stop's leak, of the simulator that guesses the stop round (`ZkServer.zk_server`; **`Flock.zk_server_ref`** at
  the session's masked record), and over a stream with at most `b` stops the leak is at most `log₂ ∑_{i ≤ b} C(m, i)·r^i` bits
  (`stream_leak`). At the worker's own values the split is exact and the checks' errors drop out
  (`ZkServer.split_own`, `ZkServer.zk_server_own`), and a pad-blind worker (`ZkServer.BWorker`) is handed neither the
  server's draws nor its messages (`ZkServer.lift`, `ZkServer.Blind`). The server that defers every stop to the last
  round (`ZkServer.nextD`) releases the reference run or it with its last message the stop (`ZkServer.split_deferred`),
  within a factor 2 of the simulator that guesses which (`ZkServer.zk_deferred`, `leaky1`).
* `Blind`: the switched prover at the reference session (`RefSession.swP`, at values `RefSession.SwVals` indexed by what
  a pad-blind worker sees), with `refP`'s simulator, which never reads the prover's messages (`Proto.simulatorE_msg`):
  **`Flock.zk_deployZk_sw`**, `Claims.Flock.ZKAt` at the same `ε_zk`, `γ` and simulator as `Flock.zkAt_deployZk_ref`
  for any values whose round encoding is causal (`RefSession.SwCausal`), `Flock.zkAt_deployZk_ref` its case at the
  tables' own values (`Flock.zkAt_deployZk_ref_of_sw`); and **`Flock.zk_server_blind`**, the server that defers every
  stop to the session's last round (`ZkServer.zk_deferred`) at every pad-blind worker within a factor 2 of the
  simulator that guesses whether it stops (one bit a session; over a stream, `log₂ ∑_{i ≤ b} C(m, i)` bits,
  `stream_leak_deferred`), with no term for the masked checks, and at a
  statement of a class `A` with `TapeOK` on `A` only (`Flock.zk_server_blind_on`), at the deployed coin layer with the
  tables' tape read with fibers of one size on `A` (`Flock.zk_server_blind_deployed`).
* `Release`: the deployed release (`LeanRelease`), the ZK server whose frame is the compiled Lean frame
  (`Masker.answer`, then `Flock.Firewall.Frame.round`: `LeanRelease.release`, `refMsg_toSrv` by definition) at the
  frame's own configuration from its draws on the server's tape (`LeanRelease.cfg`, `Cfg.ofDraws`: `cfg_pads`,
  `cfg_root`). Its coin stops are the frame's, all four of them (a zerocheck outer coin `1`, a mask slot short of full
  rank, level 0's `ρ` dependent, the inner proof's `β = 0`: `answer_rho_dependent`, `answer_beta_zero`), and where the
  frame refuses the coins the release is the stop (`LeanRelease.release_of_refusesCoins`). Its pass decision is the
  frame's too (`Flock.Firewall.Frame.check`: `LeanRelease.pass`, `check_release`), the device round against the
  scheduled round in its place; Rust's own checks only fail a round more (`LeanRelease.cross`). Any other stop comes at its
  `Finish`, with its scheduled rounds in a failed device's place
  (`LeanRelease.fillOf`), and every way a worker fails is a failed round (`LeanRelease.pass_of_none`, `nextD_of_fail`);
  the server refuses the `Finish` where the firewall's own `Finish` stops on values that fail a constraint at the drawn
  pads (`LeanRelease.finStops`, `refusesFinish`), so the one bit is `1[a round failed ∨ finStops]`.
  Its rounds are the deployed session's (`LeanRelease.kindAt`): the server's own `Hello` at the tape's nonce, the
  firewall's own link (its recheck's level-0 roots) and `Finish` (`Final.encode` of the record and proof files it
  builds, `LeanRelease.ownMsg`), and frame rounds released tagged `0`. Its `RefD` is the item-stream refinement
  `LeanRelease.ItemsRefine` alone, with no assumption on what the firewall computes (`HelloRefine` at the `Hello`,
  `OwnRefine` at the link and the `Finish`, `LiveRefine` at the worker's passed frame rounds and `FillRefine` at the
  scheduled ones, read at the values the firewall's transcript reports, where the frame takes the coins, and
  `StopRefine`, the stop where it refuses them; `LeanRelease.refD`), so
  **`Flock.zk_server_lean`** is `ZkServer.zk_deferred` at it, at any worker: one bit a session.
* `Refine`: toward `ItemsRefine` at the reference session's prover (`strat`). Its `StopRefine` holds wherever the
  session sends the stop where the frame refuses the coins (`LeanRelease.stopRefine_of_sessionStops`), and so wherever
  the session's message is the frame's release (`SessionFrames`, `stopRefine_of_sessionFrames`) and its abort is the
  stop. The frame checks no opening, so `LiveRefine` makes the release the abort after any opening the session refuses
  (`LeanRelease.release_eq_abort_of_liveRefine`); the release reads the transport's coin check at the tape's nonce
  (`LeanRelease.opens`) and is the stop where it refuses, so that holds where the check is the session's
  (`LeanRelease.CheckIs`, `openingsRefused_of_checkIs`). On the histories the check passes, the frame's coin refusal
  reads no tape (`LeanRelease.refusesCoins_tape`), and `RefSession.withStops` is a session with the stop where a rule
  of the coins says, and its frame rounds without a rep's leading domain frame (`dropDom`), keeping the session's six
  Props; where that rule is the frame's refusals up to the protocol's last coin wherever the openings pass
  (`LeanRelease.StopsAt`),
  its `StopRefine` is proved (`LeanRelease.stopRefine_withStops`), and its `LiveRefine` and `FillRefine` follow from the
  check being the session's and the per-round byte equality at the worker's and the stand-in rounds
  (`LeanRelease.liveRefine_withStops`, `fillRefine_withStops`; the deferred server takes no worker value past the
  `Finish`, `LeanRelease.live_lt_last`), its `HelloRefine` and `OwnRefine` from the server's `Hello`, the firewall's
  link and its `Finish` being the session's (`HelloEq`, `LinkRoundEq`, `FinalRoundEq`: `helloRefine_withStops`,
  `ownRefine_withStops`), so `Flock.zk_server_lean_stops` assumes no part of `ItemsRefine`. A transport that reads the
  history only through the decoded coins on the histories its check passes (`LeanRelease.ReadsCoins`) gives the rule
  and `StopsAt` (`LeanRelease.rfOf`, `stopsAt_of_readsCoins`). `Flock.zk_server_lean_coinLayer` is it at the deployed
  coin layer (`CoinLayer.session`) at SHA-512 from the transport's trusts alone, each named for its reason:
  `FinalRoundEq` is a fact about the proof files the session's Lean finish computes (`Flock.Firewall.Finish`), which
  `flock-circuit` sends, `ProgReads` one about the state the session loop's requests leave each rep's frame in
  (`Flock.Firewall.Frames`; a theorem at `LeanRelease.ofLoop`, `ofLoop_progReads`), `LinkRoundEq` one
  about what the transport hands `Flock.Firewall.Link` (`LinkRecheck`), and `LeanRelease.HelloIs` (`HelloLean`;
  `HelloEq` follows, `helloEq_of_helloIs`), `HelloParses`, `ReqsFollow`, `CoinsRead` (`ReadsCoins` is `ProgReads` and
  `CoinsRead`) and the `Finish`'s round facts about the session loop
  (`flock-firewall session`), where they are theorems (`SessionLean`); `CheckIs` is proved from them
  (`LeanRelease.checkIs_coinLayer`). The released bytes are read only through `LiveRoundEq`, `FillRoundEq`,
  `FinalRoundEq` (from `FrameAdmits` and `FinishIs`), `LinkRoundEq` (from `LinkIs`) and `ProgReads`, none of them yet
  proved about the live path, so the deployed ZK server is not yet proved zero-knowledge.
* `SessionLean`: the release at the session loop (`LeanRelease.loop`): its `Hello`, its coin check's reading of it and its
  requests are the loop's pure functions at the statement's setup (`Flock.Firewall.Session.helloOf`, `planOf`, `reqAt`),
  so `HelloIs`, `HelloParses` and `ReqsFollow` hold at every statement and `Finish` is one past the last coin
  (`LeanRelease.loop_helloIs`, `loop_helloParses`, `loop_reqsFollow`, `loop_last`), and
  **`Flock.zk_server_lean_loop`** is `Flock.zk_server_lean_coinLayer` there.
* `HelloLean`: the server's `Hello` at Lean's `flock-firewall hello`. Its `Hello` is the coin layer's
  (`Flock.Firewall.Hello.json_eq`, `CoinLayer.helloD_eq`, by `rfl`), so `HelloEq` follows from the server's `Hello`
  being that command's message (`LeanRelease.HelloIs`, **`LeanRelease.helloEq_of_helloIs`**).
* `LinkRecheck`: the firewall's link at the Lean recheck. `Flock.Firewall.Link`'s root (`Flock.Firewall.Link.root`, its
  tree built in tasks) is `layerOf`'s cap over `hm96-sha512/v1`'s leaves of `encodeRowsZk`'s rows
  (`Flock.Firewall.Link.root_eq`, `cap_eq`), the switched prover's level-0 leaves are the honest prover's
  (`swExeOf_inl`, `Masked.leafAt_swCutAt`), so the recheck's link message at the tape (`Masked.linkMsg`) is round `0`'s
  encoding of the switched cut (`Masked.enc0_swCutAt`), and at the deployed session `LinkRoundEq` follows from the
  firewall's link being that message (`LeanRelease.LinkIs`, **`LeanRelease.linkRoundEq_of_linkIs`**). Where the
  release's link is the loop's own (`Loop.linkOf`, the roots of the level 0 the loop drew, as its `Commit` carries
  them), `LinkIs` holds at a `Live` statement whose drawn level 0 is the tape's (`Loop.LinkLevel0`, from `Loop.Src`,
  **`LeanRelease.linkRoundEq_ofLoop`**, `Proofs.Flock.SessionLoop.Link`).
* `FinishRound`: `FinalRoundEq` split at its branch. On a live history whose last coins the frame takes, every table
  admits the opened coins (`LeanRelease.FrameAdmits`, one of the transport's trusts: the frame refuses each coin fault
  `Admits` excludes, and the state the transport hands it is its own run, the session loop's, `Flock.Firewall.Frames`),
  and where the firewall's own
  `Finish` does not stop either its files encode `fin` of the switched views (`LeanRelease.FinishIs`), so
  `FinalRoundEq` follows (**`LeanRelease.finalRoundEq_of_finishIs`**, a live run's `Finish` not stopping by
  `finStops_of_live`).
* `FrameTaken`: `FrameAdmits`' first piece, at any `LeanRelease`. On a live history up to the `Finish` whose last
  coins the frame takes, at every prefix of at least three rounds the transport's coin check passed and the frame of
  the rep that released the prefix's last round, at its state and configuration, answered its coins without refusal
  (**`LeanRelease.coins_taken_of_live`**, from `refusesCoins_of_live` at the prefixes, `live_take`). `Hello`'s and the
  `Commit`'s coins go to the session (`takeAt`), so it says nothing of them.
* `FrameRank`: the frame's rank check is the prover's. Where the frame's reduction over the mask slot's words fills all
  `128 k` columns (`Flock.Firewall.Frame.maskRank`), the weight rows' GF(2)-span is all of `GF(2^128)^k`
  (**`Rank.maskRankOk_of_maskRank`**: `MaskRank.maskRankOk`, by an invariant of both loops and a downward induction over
  the pivots, `lowBit_spec`, `span_of_pivots`), and the statement's slot (`Drive.maskSlot`) enumerates its mask words
  (`Rank.words_maskSlot`, `Composed.maskWordsOf`).
* `FramePoints`: the frame's claim points are the session's. On the algebraic phase's coins as the frame keeps them,
  one flat array holding round `1`'s outer coins from `K_SKIP`, `ρ`'s from `m − 6` and the lincheck's from `2 m − 10`,
  `Flock.Firewall.Frame.claimPoints` is `MaskRank.claimPoints` at the rounds (**`Rank.claimPoints_frame`**), so a rep
  whose frame takes the opening's label has full rank at its rounds and the earlier reps' points
  (**`Rank.rank_of_label`**, with `maskRankOk_of_maskRank`).
* `LinkFft`: the recheck's rows by the additive FFT in the novel basis (`Flock.Firewall.Link.rowsFft`) are
  `encodeRowsZk`'s where the padding fits below `2^cols` and `cols < d ≤ 64` (`Flock.Firewall.Link.rowsFft_eq`), so
  `rows` is `encodeRowsZk` on both of its paths (`rows_eq`).
* `FinishCommit`: the firewall's own `Finish` (`Flock.Firewall.Finish`) commits as the masked prover does. Its codeword
  by the FFT is `encodeRows`'s (`Flock.Firewall.Finish.rowsFft_eq`, `rowsOf_eq`), so at SHA-512 and `hm96-sha512/v1`, in
  any lawful monad, `FlockComplete.commitP` returns `commitFw`'s commitment and `FlockComplete.commitPZk` of the table
  `Flock.Firewall.Link` reads returns `commit0`'s (**`commitP_eq`**, **`commitPZk_eq`**, through `Commit.toC`), and
  their openings agree (`openP_eq`).
* `FinishLevel0`: the commitment the session loop's finish reads off its `Level0` (`Flock.Firewall.Finish.commitOf`)
  is `commit0`'s at the `Level0`'s own level, `t_pad` and table, wherever `Level0.ofDraws` returned it
  (**`commitOf_eq`**), so the loop's finish proves at the commitment the `table` path commits.
* `FinishPiop`: the firewall's transcript, zerocheck, lincheck and ring switch are the masked session's from its source,
  by unfolding through `Asks.toC`: **`Flock.Firewall.Finish.zerocheckFw_eq`**, **`lincheckFw_eq`** (`zerocheckCut`,
  `lincheckCut` at the masked values) and **`ringSwitchP_eq`** (`ringSwitchCut` at the switched vectors).
* `FinishInner`: the firewall's pads commitment is the masked prover's (`Flock.Firewall.Finish.innerCommitP_eq`, through
  `InnerCommit.toC`), and its inner proof, which never checks the batched constraint, is `FlockComplete.innerP`'s
  wherever that constraint holds at every coin (**`innerP_eq`**), which it does at pads with `h_ab = h_a·h_b` and
  `w = x·y` where `repFw`'s stop does not fire (**`batched_of_failingOn`**: no constraint `failingOn` the pads).
* `FinishLig`: the firewall's Ligerito prover is the masked prover's (`FlockComplete.ligeritoPZk`), its level loop
  carried through `LigState.toC` (`ligConj`), wherever its commitment is `FlockComplete.commitP`'s
  (**`Flock.Firewall.Finish.ligeritoPZk_eq`**), so at SHA-512 and `hm96-sha512/v1` with `commitFw`
  (**`ligeritoPZk_fw`**).
* `FinishRep`: the copies put together. The firewall's rep is `Flock.Firewall.Finish.repFwC`, `piopCut`'s PIOP at the
  masked values with no rank check (the frame's), `FlockComplete.ligeritoPZk` at SHA-512 and `hm96-sha512/v1`, the
  session's replay and constraints, the stop where one fails on the pads, then `FlockComplete.innerP`
  (**`Flock.Firewall.Finish.repFw_eq`**, at pads with `h_ab = h_a·h_b` and `w = x·y`, in a monad where a throw ends the
  run, as `flock-firewall finish`'s does, `throw_bind_sv`). Where no constraint fails on the pads the batched
  constraint holds whatever the constraints' lengths (`batched_of_noFail`, by `evalL_combo_linAt`).
* `FinishBridge`: the firewall's rounds as the session's. Its rep counts rounds from zero in `StateT Nat V`; the session
  logs every send in `FlockComplete.PS`. The state map `Flock.Firewall.Finish.bridge base` relates a rep at round `k` to
  a log of `base + k` sends, and every prover the rep calls is walked once, generic in its `Sim`, so a `repFw` run that
  returns at round `k'` is `repFwC` answered by `asksCut` at its coins from round `base` (`coinsFrom`), returning the
  same with a log of `base + k'` (**`Flock.Firewall.Finish.repFw_cut`**).
* `FinishTrunc`: the same at coins that agree with the rep's only below each round's count, as the session's opened
  coins do. A squeeze below `2^64` coins reads only those (`squeezeP_br`), and Ligerito's recursive levels keep their
  sizes through its loop (`loopG_bridge`), so `repFwC` at the rep's coins and at such coins is related, given `m − 13`,
  the claims, `n + 1` and every level's `q + 1` bounded (**`Flock.Firewall.Finish.repFwC_br`**).
* `FinishLive`: the served rep's live rounds as `repFw`'s. Answered by `liveAsks` at recorded coins `C`, a rep's
  rounds within `C` do no channel IO and are the rounds under `asksOf C` (`liveSim`, `liveAsks_rel`), so `repCons`
  run live is `repFw`'s `repCons` with the same result, transcript and coins (`repCons_live`); where `repFw` returns,
  its `repCons` used exactly the recorded coins and no constraint failed on the pads (`repFw_parts`), and `innerLive`
  continues from that state into `innerP` at `repFw`'s own inputs (**`Flock.Firewall.Finish.innerProg_run`**).
* `FinishSrc`: the firewall's rep against the rep from a source. Answered by `asksCut`, where the source's masked values
  below `zp 64` (the PIOP's) are `vals + h`, its switched vectors the firewall's at the claims, the rank check passes, and the
  session's Ligerito and inner hybrids hold (`LigHyp`, `InnerSrc`, the latter given the batched constraint, which held
  wherever the firewall's own inner proof returned: `Flock.Firewall.Finish.innerP_batched`), `FlockComplete.repCut` at
  level 0's commitment erased returns the firewall's transcript and log, and its proof up to `eraseZ`
  (**`Flock.Firewall.Finish.repCut_fw`**).
* `FinishRank`: the rank check of `FinishSrc`'s rep from a source. Answered by `asksCut`, a rep's zerocheck and lincheck
  leave rounds carrying the coins of their sends whatever the masked values (`FlockComplete.RankAt.growS`), so the check
  reads the claim points at the coins from the rep's first send (`FlockComplete.RankAt.roundPoints_cut`); where every
  table admits the opened coins no refusal shows, so table `j`'s rep `r`'s check passes at coins that are the opened
  ones through it (**`Claims.ZKMal.Masked.rank_at_admits`**).
* `AdmitsOf`: `not_refused_of_admits`' converse. At the executable's coin plan of any prover, where every refusal's round
  is below `n` (`SlotsBelow`) and the coins below `n` show none (`Refused`), every table admits the opened coins
  (**`Claims.ZKMal.Masked.admits_of_not_refused`**). Every refusal's round is a round of the session
  (`slotsBelow_refK`, each within its rep's `repRounds`), so no refusal by `refK x` is enough
  (**`admits_of_not_refused_refK`**), and `LeanRelease.FrameAdmits` at the session loop is the frames refusing each
  fault `Refused` names at its round.
* `FrameAdmits`: `LeanRelease.FrameAdmits` split at `Refused`, into the tables' obligation
  `Flock.Guarantees.TablesAdmitUnrefused` (no fault shown, every table admits; proved at the deployed tables,
  **`LeanRelease.tablesAdmitUnrefused_dataE`**) and the frames' side `LeanRelease.FrameUnrefused` (at a `Live`
  statement, a live history the frame takes shows no fault), so `FinalRoundEq` at `ofLoop` is theirs and `FinishIs`'
  (**`LeanRelease.finalRoundEq_ofLoop`**).
* `FrameBeta`: `FrameUnrefused`' `β` fault. At a `Live` statement the loop serves, on a live history of `refK x` openings
  whose last coins the frame takes, each rep's opening at `betaSlot` is its inner proof's round 2 (the rep's round
  position, `Loop.locate_take_sum` with `Loop.repStartZk_eq`, and its schedule so far complete,
  `Count.schedSoFar_size`), whose frame takes only one nonzero `β` (`Loop.answer_beta`), so the opened `β` is nonzero
  (**`LeanRelease.beta_unrefused`**).
* `FrameOuter`: `FrameUnrefused`' outer-coin fault. The algebraic phase's first two scheduled rounds are the
  zero-check's head and the outer round (`Sched.rounds_two`); the frame takes the head (`roundFree_zcHead`) and owes the
  outer round `m - 13` coins (`Loop.run_outer`), each of which it takes only when its `+ 1` is nonzero
  (`Loop.answer_outer`), so at each rep's `outerSlot` the opened outer coin is never `1`
  (**`LeanRelease.outer_unrefused`**).
* `FrameFree`: the values a round carries freely, read back off its bytes. Inside the algebraic phase the words at the
  offsets `roundFree` frees are the round's frames' values in order (`roundFree_vals`), the zerocheck's first round frees
  none (`roundFree_zcFree`), and the frame takes a round's coins where no outer coin is `1` (`answer_ok`).
* `FrameRanked`: `FrameUnrefused`' rank fault, at the frame. Before the opening's label the frame stays in the
  algebraic phase, and past it the rank check held at its coins (`Ranked.Good`): every label, round and answer keeps
  that (`Ranked.label_good`, `Ranked.step_round`, `Ranked.answer_keep`). Along a rep's requests the coins it holds are
  the rounds' answers up to the label, each the round's count (`Ranked.run_head`), so where the requests up to an inner
  proof's round run on, the rank check held at the rounds' coins after the earlier rep's claim points
  (**`Ranked.ownReqs_rank`**).
* `FrameRankFault`: `FrameUnrefused`' rank fault. At a `Live` statement the loop serves, on a live history of `refK x`
  openings whose last coins the frame takes, each rep's opening at `betaSlot` finds the frame at its inner proof's round
  2, where the rank check held on the rounds' coins after the earlier rep's claim points (`Ranked.repState_rank`).
  Those coins are the opened ones (`Loop.wordsAt_slot`), at the algebraic phase's counts (`Ranked.alg_cnt`,
  `Loop.loopNr_rep`), and the frame's claim points on them are the session's (`Ranked.frame_points`), so no rep's rank
  check refuses (**`LeanRelease.rank_unrefused`**).
* `FrameRho`: `FrameUnrefused`' `ρ` fault. The firewall's masked Ligerito stops at dependent `ρ₀, ρ₁` right after
  `ρ₁`'s squeeze, so a rep's schedule that ran its opening to its end was answered independent ones at Ligerito's rounds
  `k₀ + 2` and `k₀ + 3` (**`Count.openStep_rho`**, through `SPW`, a step program's answers so far). At the rep's inner
  round 2 its schedule so far is complete (**`Count.schedSoFar_rho`**), and those answers are the opened coins at
  `rhoSlot` and the next, so no rep's pair coins are dependent (**`LeanRelease.rho_unrefused`**).
* `FrameUnrefused`: the four faults together. Each fault `Refused` shows by `refK x` at a `Live` statement the loop
  serves is one a rep frame refuses at its round, below `refK x` (`LeanRelease.slots_lt`), so the session loop's own
  frames refuse every coin fault (**`LeanRelease.frameUnrefused_ofLoop`**), and `SessionLoopZK` takes no
  `FrameUnrefused`.
* `FinishStop`: the firewall's stop at a worker's values. At constraints whose forms are rep `r`'s first seven rows on
  the masked values, none fails on the drawn pads exactly where those rows vanish at the clear values at zero pads
  (**`Composed.noFail_iff_rows`**, by `rows_padFree`), so the stop bit reads no pad; and where they vanish for both
  reps the worker's rows hold at every pad (**`Composed.rowsAt_of_rows`**, `Table.RowsAt`).
* `FinishStopAt`: that stop at a `--zk` table's own draw. Where, at each rep, the firewall's constraints are the first
  seven rows on the view's masked values and none fails on the drawn pads, the worker's rows hold at the draw's mask slot
  and extra lanes (**`Composed.rowsAt_of_cons`**), so the switched view is the worker's own.
* `FinishSwSrc`: the switched prover's rep source, `tabSrcCut` of the switched cut at the workers' values
  (`Masked.swSrcE`, `Masked.swCutE`, `RefSession.swCutOf`), the source `FinishSrc`'s premises are to be met at for
  `FinishIs`. Where a table's rows hold at its base, its masked values are the worker's view's
  (**`Masked.swCutE_masked_of_rowsAt`**).
* `FinishSwSv`: that source's switched vectors. The switched view moves only pads, masked values and `Y₁`, so its `ŝ`
  and region claims are the honest view's (`Table.swPre_sHat`, `Table.swPre_sReg`), and so are the switched source's
  switched vectors (`Masked.swSrcE_sv`): `src_sv` holds there at every worker's values (**`Masked.swSrc_sv`**), the
  switched vectors `repCut_fw`'s `hpiop` asks of it.
* `FinishSwMk`: that source's masked values. Where a table's rows hold at its base, rep `r`'s masked value `n` is the
  worker's clear value plus the base's pad, as the firewall's words (**`Masked.swSrcE_masked_of_rowsAt`**); below
  `h_ab` that pad is the session's pad array's (**`Masked.swSrcE_masked_pad`**). At the firewall's clear values by pad
  index (`Masked.valsFw`: the passed values below `zp 64`, level 0's opening's from there on), that is `repCut_fw`'s
  `hmk` below `zp 64` (**`Masked.swSrcE_hmk`**) and the opening's values plus the pads up to `h_ab`
  (**`Masked.swSrcE_l0`**).
* `FinishSwMkAny`: the same masked values with no rows: the shifted view adds `v - vals` to the pad, so rep `r`'s
  masked value is the passed value plus the session's pad whether or not the rows hold
  (**`Masked.swSrcE_masked_pad_any`**).
* `FinishFwL0`: level 0's values from the firewall's own run: where `repFwC` returns, its proof's out-of-domain value,
  lane messages and pair's sums are `l0Clear` at the coins from the round its PIOP ends at, on its packed table and its
  PIOP's basis, plus the pads (**`Finish.repFwC_l0`**).
* `FinishSwFrame`: those at the firewall's own request. Where the passed values are the request's, its draws the
  release's frame draws on the tape, and the session's tape has the pads of the tape of those draws (`tapeOfDraws`), the
  session's pads are the request's below `h_ab` (`Masked.pads_frame`), so the masked values below `zp 64` are the
  request's `vals + padsOf draws` (**`Masked.swSrcE_hmk_frame`**), and up to `h_ab` the opening's values plus those pads
  (**`Masked.swSrcE_l0_frame`**).
* `FinishSwLig`: `repCut_fw`'s `hlig` at that source. Level 0's opening's clear value at each pad index from `zp 64`
  (`LigLock.l0Clear`: the out-of-domain value, each lane message's words, the pair's sums' words) gives `LigHyp`'s
  masked fields (**`LigLock.ligHyp_of_l0`**), so at `valsFw` with `l0Clear` as level 0's part and the request's pads,
  `LigHyp` holds at the switched source's `y₁'`, `ȳ` and salts (**`Masked.swSrcE_ligHyp_frame`**).
* `FinishSwLigE`: those at the coins. The switched view moves only pads, masked values and `Y₁`, so its `y₁'`, `ȳ` and
  levels ≥ 1 are the honest view's (`Table.swPre_y1`, `swPre_ybar`, `swPre_lig`), and so are the switched source's
  (`Masked.swSrcE_restart`); the honest source's are the session's at the coins (`ligY1E`, `ligYbarE`, `ligSaltE`). So
  at the deployed rep's coins, packed table, basis and the tape's extra lanes, lane paddings and salts, `LigHyp` holds at
  the switched source with `l0Clear` as level 0's part and the request's pads (**`Masked.swSrcE_ligHyp_E`**).
* `L0Pin`: level 0's bodies pin the source. Equal bodies at one coins, table and basis have equal pads from `zp 64` to
  `h_ab` (**`LigLock.l0BodyAt_pads`**); a rep from a source, cut in level 0's opening, sends `l0BodyAt` at the pads
  `padsL0` (masked plus `l0Clear`; **`LigLock.repCut_cutL0`**), so where those are a run's at pads `h` the source's
  masked values there are `l0Clear` plus `h` (**`LigLock.masked_of_cutL0`**).
* `L0PinCoins`: the cut PIOP from a source ends where the firewall's does. It splits like `piopFw`
  (**`piopCut_split`**, **`piopCut_at`**), and at a deployed rep's coins its log ends at `ligP0` with basis `ligBasis`
  (**`Masked.piopCut_lig`**), so `repCut_cutL0`'s bodies are at the coins, `p0` and basis that `swSrcE_repCut` reads.
* `PiopFree`: the cut PIOP returns on its coins alone. Where its phases returned at other values, it returns from any
  source, layout, cap and root wherever its rank check passes (**`piopCut_of_phases`**), so two cut PIOPs from one log
  size both return or neither does, with the same log size and switched claims (**`piopCut_src_free`**,
  `piopCut_sws_eq`), and where the firewall's rep returned, the cut PIOP from any source does
  (**`Masked.piopCut_of_repFwC`**).
* `RepFree`: the cut opening returns on its coins alone. Two runs of the opening, the firewall's `ligeritoPZk` and the
  cut's `ligeritoCut` from any source, at logs of one size, both return with equal message, challenge and OOD counts
  and equal log sizes, or the firewall's doesn't return (**`ligeritoPZk_cut_fw`**).
* `RepReplay`: the cut rep's replay returns on its own rounds, its fold at the coins (**`replay_cut_ok`**), and a
  returned replay folds at the coins it read (**`replay_fold_coins`**).
* `RepInner`: the cut's inner proof returns where the firewall's does, from any commitment, value and source
  (**`innerP_cut_fw`**).
* `RepCutFw`: the cut rep returns on its coins alone. Where the firewall's rep returned and the cut's rank check passes,
  the cut rep from any hash, commitments and source returns, with a log of the firewall's size
  (**`CutFree.repCut_of_repFwC`**).
* `L0PinFrame`: the firewall's scheduled opening, fed the stream's coins, is the cut run. Fed coins one round at a
  time (`Fed.feedR`), a run over `Drive.OpenM` is the cut run over `PS` at those coins (`Fed.feedSim`), so the opening
  is (`Fed.asksRel_feed`, `Fed.repOpening_feed`), and past its ring switch it sends level 0's bodies
  (`Fed.repOpening_cutL0`). Where `openStep` sends anything its PIOP returned, and its level-0 rounds are `l0BodyAt`
  at zero pads (**`Fed.openStep_l0`**), which reads only its rounds' coins (`Fed.l0BodyAt_congr`).
* `L0PinQuiet`: the algebraic phase's rounds after the first are quiet. Each is plain frames (no label that moves the
  frame's phase) and then a squeeze (**`Quiet.rounds_quiet`**).
* `L0PinFlags`: the frame's flags along a rep's scheduled rounds. Quiet rounds and their coins keep them, and the
  opening's first round (its label, `LABEL_PCS`) ends the algebraic phase at `c.nPiop` (`Flags.roundFree_pcs`). So
  past it, at the coins, the frame is outside every masked phase at `c.nPiop`, owing nothing (**`Flags.sched_opened`**).
* `L0PinSwitch`: the cut ring switch with room for its asks returns, having sent each claim's round and `γ`'s
  squeeze after the pending bytes, and leaves nothing pending (**`Switch.ringSwitchCut_bodies`**); a rep's PIOP ends
  with nothing pending (`Switch.endE_repPiop`); from empty pending bytes the switch's first round is the opening's
  (`Flags.PcsRound`) and its later rounds and `γ`'s are quiet (`Switch.pcsRound_rs`, `Switch.qRound_rs`,
  `Switch.qRound_gamma`).
* `L0PinRec`: a rep's PIOP on its recorded coins. Answered by the coins the frame recorded, the PIOP is the cut run
  at those coins (`Fed.recSim`, `Fed.recCoin`), so it returns the zerocheck's and lincheck's claims at the recorded
  coins' points (**`Fed.repPiop_rec`**, `Fed.PtsAt`), and the claims' points and switched claims read only their rounds'
  first words (`Fed.ptsAt_congr`, `Fed.sws_coins_congr`).
* `L0PinBind`: the frame at a rep's binding. Past the algebraic phase a rep's schedule is what its opening, fed, sent
  (`Bind.schedSoFar_sentQ`, `Bind.feed_part`); cut at `γ`, its first rounds are the ring switch's
  (`Bind.openStep_switch`), squeezing `7` a claim and then the claims' count (`Bind.sched_count`), so at `γ`'s coins
  the loop's state is outside every masked phase at `c.nPiop`, before level 0, owing nothing
  (**`Bind.repState_opened`**). Fed level 0's first coins too, its next rounds are the binding and level 0's bodies at
  zero pads (`Bind.sched_binding`, **`Bind.sched_l0`**).
* `L0PinLive`: on a live history whose messages are the release's own, a round scheduled with no free value is the
  frame's release of the scheduled round (`LeanRelease.msg_sched`), so a sent level-0 round is its body at the pads
  (**`LeanRelease.msg_l0`**) and the binding is itself (`LeanRelease.msg_openBind`).
* `L0PinChain`: level 0's places in the absorbed vector (`l0Off`, `l0_size`) and its sent bodies as `L0Sent` there
  (`LigLock.l0Sent_at`). On a live history whose rounds from a binding on are masked, at one rep, scheduled as the
  binding and level 0's bodies at zero pads, the sent rounds are the binding and the bodies at the rep's draws
  (**`LeanRelease.l0_chain`**).
* `L0PinLoop`: `l0_chain` at the loop's release. At a `Live` statement the loop serves, on a live history of `refK x`
  openings whose tables are the loop's, a rep's binding round sends the binding at its table's cap and `z`'s squeeze
  of `n`, and its level-0 rounds send level 0's bodies at its table, its draws' pads and the opened coins from the
  binding's on (**`LeanRelease.ofLoop_l0`**). Every `--zk` schedule's `n` is under `64` (`Count.sched_n`).
* `L0PinCut`: the cut session in rep `2j + r`'s level 0 rounds. Where the reps before it return by the firewall's stop
  and its PIOP returns, the session stopped at `n ∈ [ligP0, ligP0 + k₀ + 4)` sent `l0BodyAt`'s body `n − ligP0` at the
  coins from `ligP0`, basis `ligBasis` and pads `padsL0` (**`Masked.runCut_cutL0`**).
* `FinHmkLive`: a rep's masked values below `zp 64` from its proofs. The zerocheck and lincheck return every masked
  value below `zp 64` as their proofs, which `eraseZ` keeps, so where the firewall's rep at `vals` and `pads` and a cut rep
  from a source return proofs equal up to `eraseZ`, the source's masked values there are `vals + pads.h`
  (**`HmkLive.masked_of_eraseZ`**).
* `FinishSwInner`: `repCut_fw`'s `hinner` at that source. The switched view keeps `ρ_in`, `σ_in` (`padShift` moves
  `h_tx`, `h_ty` against the exposed pads; `Table.innerMsg_padShift`) and `Y₂`, and where the rows hold `Y₁`, so the
  switched source's `ρ_in`, `σ_in` and `Y` are the honest source's (`Masked.swSrcE_inner`, `Masked.swSrcE_y`). The
  request's pads are the tape's wherever the tape's pads are its draws' (`Masked.pads_frame_eq`), so at the inner
  proof's slot `InnerSrc` holds at the switched source and the request's pads, given the batched constraint
  (`hinner`'s own premise) and `τ`'s commitment there (**`Masked.swSrcE_innerSrc_frame`**).
* `FinishSwTau`: that `τ` commitment. Where the rows hold, the switched `τ` leaf commits the worker's view's `τ` under
  the honest view's `y_τ` (**`Masked.swSrcE_tauCom`**), and that `τ` is `InZ.tau` over any constraints whose first seven
  rows are the worker's view's (`Masked.SwRows`, **`Masked.swTau_of_rows`**: the triple's rows don't move), so
  `InnerSrc` holds at the switched source from the batched constraint and `SwRows` (**`Masked.swSrcE_innerSrc_rows`**).
* `FinishSwRows`: `SwRows` at `hinner`'s constraints. The replay's rows and values have the layout's length
  (`Composed.replay_asz`), so where level 0's two rows do too the firewall's constraints are seven of that length
  (`Composed.fwCons_asz`), and where their rows are `rowsSwE`'s as whole forms (as `Composed.tauFw` gives them) or by
  coefficient (as `Composed.tauFw_at_coef` does) they are `SwRows` (**`Masked.swRows_of`**, **`Masked.swRows_coef`**).
* `FinishSwFw`: `hinner` at the switched source. `Composed.tauFw_at_coef` at the worker's view reads the deployed
  rep's coins from its base (`czExe`'s), the view's masked values below `zp 64` (`Masked.swViewMasked` with
  `Masked.swSrcE_hmk_frame`), the model's fold and level 0's record, whose coins are the rep's at its slots
  (`ligeritoPZk_lanes`' `record`, `Masked.LaneCoins`), and the inner proof begins at `repSlot … (innerOff 0)`, so
  `InnerSrc` holds at the switched source wherever the ring switch's vectors are the view's (**`Masked.swSrcE_hinner`**,
  its premise `hsw`), which `hpiop`'s switched vectors give (**`Masked.swSrcE_hinner_sv`**).
* `FinishSwPts`: the firewall's PIOP's points at the coins. Answered by `asksCut`, a returned zerocheck's and lincheck's
  points are their coins' (`zerocheckCut_at`, `lincheckCut_at`), and a ring switch's switched claims are `Rs.sws` at its
  coins (`ringSwitchCut_sws`), so the firewall's PIOP switched the closed-form claims at the coins from its base after
  `piopRoundsZk` bodies, all before the cut's stop (**`piopFw_at`**).
* `FinishSwBit`: the firewall's switched vectors read off the packed table, bit `v` of word `y` being the trace's bit
  `128 y + v` (`packed_bit`), are the prover's at a claim with `m − 6` coordinates (**`switchVec_packed`**).
* `FinishSwRep`: **`Masked.swSrcE_repCut`**, `repCut_fw` at the switched source. Its `hmk` is `swSrcE_hmk_frame`;
  `hpiop` is `rank_at_admits` and `swSrc_sv` through `switchVec_packed`; `hlig` is `swSrcE_ligHyp_E` at `piopFw_at`'s
  log and `swsE`; `hinner` is `swSrcE_hinner_sv`. So where the firewall's rep returns at table `j`'s rep `r`, the cut
  from the switched source returns the same transcript and log, and the proof up to `eraseZ`. Its premises: the
  firewall's packed table is the witness's (`hg`, a host-input trust as `LinkIs` is for the link), the coins are
  admitted, and level 0's clear values are `l0Clear` (`hB`). The rep's base being before the stop comes from its PIOP
  having returned (`repFwC_piopFw`), and no refusal from the admitted coins (`not_refused_of_admits`).
* `FinishSwVals`: `swSrcE_repCut` at any clear values (**`Masked.swSrcE_repCut_v`**), with what `valsFw` supplied as
  premises: the masked values below `zp 64` are the request's values plus its pads (`hmkV`), and, where the source's
  PIOP returns, level 0's from `zp 64` to `h_ab` are `l0Clear` plus the pads (`hl0`). So the views' source at the
  workers' values takes the firewall's rep's place.
* `FinishClaimsX`: the firewall's rep reads the extra claims' coordinates alone, through the ring switch
  (**`Flock.Firewall.Finish.repFwC_claims_x`**), so it is one run at `claimsOf xs` and at any extra claims at `xs`.
* `FinishRows`: the replay's rows on a view's masked values, constants included. Where the proof's masked messages are
  the view's, the coins `c`'s and the switched vectors `ŝ`'s, `Zk.replay`'s three constraints and the two ring-switch
  rows are `rowsOf`'s first five as whole forms (**`Composed.replay_rows`**), as the firewall's stop reads them.
* `FinishZc`: the masked zerocheck's and lincheck's proofs. Answered by `asksCut` with room for their asks, the cut's
  zerocheck and lincheck return the masked values `mk` at the layout's indices as their proofs
  (**`Claims.ZKServer.zerocheckCut_proof`**, **`Claims.ZKServer.lincheckCut_proof`**).
* `FinishTau`: those two met at the firewall's own run. Where the firewall's rep under `asksCut` passes, its coins are
  the tape's (`RankAt.cut_coins`, through the ring switch and the opening by `RankAt.ringSwitchCut_grows` and
  `RankAt.ligeritoPZk_grows`), and at the model's coins, switched vectors and a level 0 read at the view
  (`Masked.LaneAt`) its seven constraints are `rowsOf`'s first seven (**`Composed.tauFw`**).
* `FinishLane`: level 0's two lane rows on a view's masked values, constants included. Where level 0's proof reads the
  view's masked values, its target is the opening's `T` and its restart is `T'`, `ligeritoConstraints`' two rows are
  `rowsOf`'s rows 5 and 6 as whole forms (**`Masked.lanesR`**, by `Masked.laneRowR`).
* `CoinPass`: those premises leave a history to be live at. At every tape, past `Hello` and the link, the check passes
  `Hello`'s answer whose root the first opening walks to under the tape's nonce, and that opening, at every key with bit
  2047 clear and every opening that reads back (`LeanRelease.opens_coinPass`; the all-zero one,
  `opens_coinPass_zeros`), so the coins the frame takes there are its own.
* `Os`: the server's tape as the operating system's bytes (`OsTape`: a stream of exactly `drawBytes` bytes for each rep's
  frame and the tap's streams, one for one), the frame's read proved to be the release's draws (`frame_draws`, with a read
  of any other length refused, `drawsWith_refuses`), and the step to a uniform tape beside independent randomness under
  A3 per test, up to its slack `η` (`Pr_os_prod`). `ZkDeferredOs`, `ZkServerBlindOs` and `ZkServerLeanOs` state the
  three theorems at those bytes, the bound plus `η`, proved as `zkDeferredOs`, `zkServerBlindOs` and `zkServerLeanOs`.
* `FrameTape`: the frames' draws read as the masked prover's tape (`tapeOfDraws`: table `i`'s rep `r` reads frame rep
  `rep i r`'s bytes as its `h`, `μ`, their padding, `τ`'s salt and leaf salts), whose pads are the frame's
  (`padsOf_tapeOfDraws`, `padsOf_drawsAt`, with `h_ab` and `w`), whose leaf salts are the frame's (`innerSalt_fit`), and
  which the release's configuration carries at the rep's rounds (`cfg_tapeOfDraws`), at the operating system's bytes
  too (`padsOf_os`).
* `FrameParse`: the frame's round (`roundFree`) at the bytes of a round's frames (`encs fs` then the squeeze) is the fold
  of the frame's step over them (`stepAt`: labels through `Masker.label`, bytes copied, each observed value absorbed and
  free while the algebraic phase masks), then the inner proof's own round and the squeeze (`roundFree_encs`).
* `FrameMask`: what the frame releases of such a round. The frames come out with domains dropped and each value of a
  masked phase as `v + h_i` (`maskFrames`, `maskVal`; `roundFree_on`, `roundFree_off`), the pads' label and root
  following the zerocheck's label (`roundFree_zc`). The pads are taken in the absorbed order: every pad below `h_ab` but
  `final_c`'s, ascending (`absorbed_toList`), the indices the masked prover pads its sends under. A device that sends
  the masked prover's round with each pad added back, its plain values, has that round released but its domain frames
  (`roundFree_honest`, `dropDoms`), under the masked prover's pads (`padAt_ofDraws_absPos`).
* `FrameRounds`: the cut's zerocheck, answered by `asksCut`, sends the binding, `r_skip`'s and `r_outer`'s squeezes,
  `round1`'s and each residual round's masked values with their coins' squeezes (`zerocheckCut_bodies`, `zcBodies`),
  and the frame releases the device's zerocheck rounds as those bodies (`roundFree_zcBind`, `roundFree_zcOuter`,
  `roundFree_zcR1`, `roundFree_zcMl`) where each masked value is the device's value plus the frame's pad at its
  absorbed position (`zcBodies_frame`), the binding without its pending domain. A rep's phase and position carry across
  a round's coins (`answer_alg`), and the Lean frame's release (`LeanRelease.release`) of those rounds is the frame's
  round (`release_zcBind`, `release_zcOuter`, `release_zcR1`, `release_zcMl`). The lincheck's bodies are the same way
  (`lincheckCut_bodies`, `lcBodies_frame`, its values at their absorbed positions `absPos`), released by
  `LeanRelease.release_on`. A frame round's release is the session's message without its leading domain frame
  (`dropDom_dom`, `dropDom_noDom`) wherever the device's frames through the frame's masks are the session's
  (`LeanRelease.RoundFrames`, `release_of_roundFrames`), so wherever the device sends the session's frames with each
  pad added back (`RoundHonest`, `roundFrames_of_honest`); that gives `LiveRoundEq` and `FillRoundEq`
  (`LeanRelease.liveRoundEq_of_frames`, `liveRoundEq_of_honest`, `fillRoundEq_of_frames`).
* `RepLog`: a cut rep's rounds are its log's bodies (`RepLog.repCut_rounds`): round `j` from a log of `L` bodies squeezes
  at most `B` of `coin (L + j)`'s coins, and its bytes are body `L + j`, the rep's domain frame first at round `0` only
  and the squeeze's frame last (`RoundAt`).
* `CutPrefix`: a cut rep stopped at body `n` and one stopped later agree up to the stop (`CutPrefix.repCut_cut_agree`):
  the same end, or the earlier threw right after body `n` and the later's bodies extend it (`eq_of_agree`,
  `prefix_of_agree'`); a rep run from a longer history sends the same bodies after it (`repCut_shift`, `go_of_shift`).
* `FrameOpen`: level 0's masked rounds. The pads enter the masked Ligerito only there, additively (its out-of-domain
  value, its first `k_0` lane messages, the pair's sums), so the firewall's zero-pad round is the rep's round at its pads
  with each pad added back (`oodFr_unmask`, `pairFr_unmask`, characteristic 2), at absorbed positions past the
  algebraic phase where the frame's pads are the draws' (`l0_pad`) and a stopped rep masks as a live one
  (`maskFrames_ge`). The frame releases the zero-pad round as the round at the pads (`L0Round`, `l0Round_ood`,
  `l0Round_msg`, `l0Round_e`, `LeanRelease.release_l0`), which is `RoundHonest` at a live round and `RoundFrames` at a
  stand-in (`roundHonest_l0`, `roundFrames_l0`).
* `FrameOpenCut`: what the masked Ligerito sends at level 0. Cut after its `j`-th level-0 send, it stops past the
  openings having sent exactly bodies `0..j` (`l0Body`, `l0BodyAt`): the opening's prefix with `z`'s squeeze, the
  out-of-domain frame, the `k_0` lane messages, the pair of sums and the last squeeze (`ligeritoPZk_cutL0`).
* `FrameOpenLive`: the cut opening at the pads and at `openLive`'s zero pads. The bodies without pads agree
  (`l0BodyAt_free`), and each masked body is an `L0Round`'s zero-pad round and round at the pads with one squeeze
  (`L0Sent`, `l0Sent_ood`, `l0Sent_lane`, `l0Sent_pair`, `l0Sent_zero`), which the frame releases as the body at the
  pads (`L0Sent.release`, `L0Sent.roundHonest`, `L0Sent.roundFrames`); both openings, cut (`ligeritoPZk_cutL0_live`).
* `FramePhase`: level 0's phase labels. The opening's label starts level 0's masked phase and its round leaves as it
  came (`label_open`, `roundFree_open`; the binding, `openBind`, `roundFree_openBind`, `LeanRelease.release_openBind`);
  level 0's closing label, with every masked value absorbed and the `ρ` coins taken, ends it, the frames after it
  leaving as they came (`label_close`, `roundFree_close`, `LeanRelease.release_close`).
* `FrameState`: the frame's state across level 0's opening. Where the scheduled round carries no free value, the state
  the frame's `check` returns on any device round is the scheduled round's but for `stopped` (`check_l0`, from
  `passes_eq`; the free offsets: `roundFree_on_fr`, `roundFree_openBind`), and under the transport's trust that within
  a rep its state after a released round is its `check`'s (`LeanRelease.ProgChecks`; `answered_of_scheduled`) the
  binding starts level 0's phase at the algebraic phase's position (`answered_openBind`), each masked round moves it
  by its values (`answered_l0`), the pair's sums reach the layout's end and take `ρ_0` (`answered_l0_last`,
  `answer_rho_first`), and `ρ_1`'s round takes `ρ_1` (`answered_rho1`), the state the closing round reads.
* `FrameRun`: the frame's fold over its requests (`Frame.run`): over `qs ++ rs` it is over `qs`, then over `rs` where
  that did not end the rep (`run_append`, `run_snoc`); a round with its scheduled round in its place is `check`
  (`step_sched`), and coins are `Masker.answer` (`step_coins`).
* `FrameShape`: the frame's round reads a round's shape alone: two rounds whose frames are alike (`Alike`: the same
  kinds, sizes, labels and lengths), at configurations alike but for their draws (`CfgAlike`) and states equal but for
  `stopped`, leave the same state but for `stopped` and the same error (`roundFree_alike`), and where both are taken,
  inside the inner proof too, the same state (`roundFree_alike_ok`).
* `FrameOwn`: the frame's check of the inner proof takes the firewall's own rounds (`ownRound`) at every configuration
  and state (`innerOwn_ownRound`).
* `FramePass`: a device round that passes the frame's check at a scheduled round of frames `fs` is the bytes of frames
  alike to `fs` and the same squeeze (`passes_frames`): the words the frame frees are the frames' values
  (`roundFree_free`), and any 16 bytes are a value's (`toBytes_readWord`).
* `FrameLoop`: the release at the session loop's own frames (`LeanRelease.ofLoop`), its transport's state the loop's
  `Frame.run` over `Drive.reqs` at placeholder draws and salts, from the decoded coins alone: the state of the rep that
  released the last round (`Loop.progOf`), which takes its coins, and a rep's first-round state (`Loop.freshOf`, as
  `Frames.roundPlace` starts it), so `ProgReads` holds by definition (`ofLoop_progReads`).
* `FrameSched`: two runs of the firewall's prover at inputs of one shape send rounds of one shape (`Sim`, over any
  monad with a `Lift`): the algebraic phase's schedule at two level-0 caps of one size (`rounds_alike`), a rep's
  opening at two tables of one shape whatever its salts for levels ≥ 1 (`openStep_alike`), and so a rep's scheduled
  rounds so far (`schedSoFar_alike`), which only grow with one more round's coins (`schedSoFar_snoc`).
* `FrameProg`: toward `ProgChecks` at `ofLoop`: where the deployed frame's check of a device round releases a round, the
  placeholder's check of its alike scheduled round against itself releases one from the same state, its state the same
  but for `stopped` (`check_alike_self`), and the loop's frame state one scheduled round on is that check's
  (`repState_step`), so `ProgChecks`' step holds at a scheduled round past a rep's first (`ofLoop_progChecks_step`).
* `FrameCount`: the algebraic phase's scheduled rounds squeeze the masked replay's counts (`rounds_counts`,
  `piop_counts`: `replayCoins` with its zeros dropped), each round's count as its squeeze header holds it (`Tally`, over
  the recording channel `Skeleton.recAsks`).
* `FrameAsks`: the live opening's squeezes ask for their counts (`OT`, over `Drive.stepAsks` past the coins it was
  given): the ring switch's (`ringSwitchP_ot`) and, with `1 ≤ k0`, Ligerito's, `ligeritoCoins` with its zeros dropped
  (`ligeritoPZk_ot`).
* `FramePlan`: a rep's schedule so far squeezes a prefix of the algebraic phase's counts, then the opening's
  (`sched_prefix`): the PIOP on the algebraic phase's coins asks for nothing (`piop_na`), the opening asks for the ring
  switch's and Ligerito's (`openStep_sp`), and fed coins, the rounds sent are a prefix of the asks (`feed_prefix`). So at
  the placeholder at a link of `spec.points` points, each scheduled round squeezes the plan's count (`plan_sched`).
* `FrameLinks`: at a `Live` statement every rep stream's table has the session's link points, nonzero and within
  `Frames.COIN_MAX` at its two points (`Masked.live_links`): `ofLoop_progChecks_of`'s `hP`.
* `FrameReal`: where `real` answers the session loop's own tables (`RealTabs`: `tabOfLevel0` of a level 0 `ofDraws`
  drew), the rep's own table is of the placeholder's shape (`real_tabAlike`) and within the bounds `schedSoFar_alike`
  reads (`real_bounds`, with `sched_bounds` by `decide +kernel`).
* `FrameRep`: the session loop places a rep's rounds as `Drive.reqs` does: at a rep whose kept algebraic phase and
  opening are the schedule so far on its coins (`RepInv`), `Frames.placeOf` places its next round at that schedule's
  round there, the inner proof's own past it (`placeOf_sched`), and the rep after the round's coins keeps `RepInv`.
* `FrameInner`: at the inner proof's rounds the frame takes only the firewall's own round (`innerOwn_inv`,
  `roundFree_own_frames`), and where the deployed frame's check releases one, the placeholder's own round at zero pads
  (`ZeroCfg`) from the same state but for `stopped` leaves the same state but for `stopped`: past the inner proof's
  label (`check_own_ok`), and at it where the scheduled round in its place is the firewall's own round 0
  (`check_own0_ok`). Along the loop the rep's state counts the inner proof's rounds (`repState_own_inner`), refuses
  its round 4 (`check_inner_done`) and owes no coins past it (`repState_past`), so the loop's frame state one round
  past the rep's schedule is the deployed check's but for `stopped` (`progOf_own_step`), and `ProgChecks` holds at
  `ofLoop` (`ofLoop_progChecks`) from `hS`, `hR`, `hP` and two named premises: the placeholder's pad row within a
  `u64`'s count (`hRow`) and `L₀`'s round at the inner proof's round 0 the firewall's own (`hL0`).
* `FrameZK`: **`Flock.zk_server_lean_ofLoop`**, `Flock.zk_server_lean_loop` at `ofLoop` with no `ProgReads`
  (`ofLoop_progReads`) and `ProgChecks` a conclusion (`ofLoop_progChecks_sizes`): `hS`, `hP` and `hRow` from one check
  of the setup's sizes (`Session.sizesOk`, which the loop runs at its `Hello`: `Loop.Serves`), and `hR` (`RealTabs`)
  and `hL0` its premises.
* `FrameLive`: the bound is not vacuous past `Hello`: at `ofLoop` the link's coins are the session's, so the link is the
  firewall's (`ofLoop_ownMsg_link`), and the history of `Hello`, the link and rep 0's first round is live at the honest
  worker through rep 0's second round wherever the deployed frame's check passes those device rounds at the loop's
  states (`ofLoop_live_three`).
* `PadsRoot`: the frame's pads root (`padsRoot`, `treeRoot` of its leaves) is `layerOf`'s (`treeRoot_layerOf`), so at a
  frame rep's draws of `drawBytes` bytes it is the root `innerCommitP` commits the masked prover's pads under on
  `tapeOfDraws` (`padsRoot_tapeOfDraws`).
* `NonVacuous`: the claim is not vacuous. A prover that makes no calls has `Fork.Pr = 0`, so the simulator that hashes a
  collision and outputs the empty transcript fails `ZKmalR` at every `ε_zk < 1` and every `γ` (`not_zkmalR_collSim`),
  a prover that sends its witness has no simulator at `ε_zk < 1/2` (`not_zkmalR_leak`), and both fail `ZKAt` at
  a masked record whose relation holds everywhere (`toyD`). Flock's `ZK` fails at the masked record of `toyD` whose
  reference prover is quiet and whose simulator is colliding (`not_flockZK_toyZD`), so it is not met at every masked
  record (`not_forall_flockZK`).
* `SamplingNonVacuous`: the same for the sampling `ZK`: it fails at a deployment whose reference prover is quiet and
  whose simulator is colliding given the leaf layer (`not_samplingZK_collSim`), at one whose reference prover sends its
  witness at two honest witnesses of one leaf layer (`not_samplingZK_leak`), and is not met at every masked deployment
  where one has an honest pair (`not_forall_samplingZK`).
* `Compose`: sequential composition. Two protocols in turn, the second at a statement the first's history selects
  (`Seq`), are `ZKmalR` under the simulator that runs the first's, then the second's at the unit its transcript selects
  (**`zkmalR_seqU`**, with one coin type for the second at every unit, so units need not be finitely many;
  **`zkmalR_seq`** for finitely many): `ε` and `γ` add up, the hybrid that changes the second keeping the first real
  and giving `V*` its messages as tape, and the composed tape (`forkTape`) guesses which session's `Fork` to charge
  (`Seq.fork_ge`). A first protocol with `γ₀ ≤ c·g`, then `n` sessions in turn (**`zkmalR_iter`**), cost
  `ε₀ + Σ_{j < n} eB j` and `(c + n)·g`: `n·g` after a prefix that rewinds nothing. Both hold at one statement
  (`zkmalAt_seqU`, `zkmalAt_iter`), so the tapes and the number of sessions may be chosen per statement: on the
  spec's statement-indexed tapes (`ZKmalRx`), `m x` sessions in turn (`zkmalRx_iter`).
* `Layout`: a run in parts, a first part then sessions back to back, on a flat tape (the first part's and one session
  tape per slot) is `zkmalAt_iterG`'s nested run (`Layout.iterProver_flat`), so **`zkmalAt_layout`** charges it
  `ε₀ + Σ_{j < m} eB j` and `(c + m)·g`, each session's zero knowledge needed only at the histories that start it.
* `Cut`: a run cut at a round (`cutT`), and the run of a prover stopped there is its own run cut (`interact_stopAt`).
* `Stop`: a run in parts with a public stop, a test of the verifier-visible history read through the first part:
  the run with the stop is the run cut where it fires (`interact_censor`), so a first part that rewinds nothing keeps
  its error (`ZKmalAt.censor`), sessions where it fired are silent and simulated exactly, and **`zkmalAt_layout_stop`**
  charges the parts in turn `ε₀ + Σ_{j < m} eB j` and `m·g` against the prover with the stop.
-/
