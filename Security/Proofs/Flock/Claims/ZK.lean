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
import Proofs.Flock.Claims.ZK.Sent
import Proofs.Flock.Claims.ZK.Round1
import Proofs.Flock.Claims.ZK.Round1Pads
import Proofs.Flock.Claims.ZK.Round1Root
import Proofs.Flock.Claims.ZK.Round1Enc
import Proofs.Flock.Claims.ZK.Round2
import Proofs.Flock.Claims.ZK.Worker
import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server
import Proofs.Flock.Claims.ZK.Blind
import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Claims.ZK.Refine
import Proofs.Flock.Claims.ZK.Os
import Proofs.Flock.Claims.ZK.FrameTape
import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.Claims.ZK.SamplingNonVacuous
import Proofs.Flock.Claims.ZK.Compose
import Proofs.Flock.Claims.ZK.Layout
import Proofs.Flock.Claims.ZK.Cut
import Proofs.Flock.Claims.ZK.Stop

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
* `Worker`: Lemma B at any clear values a worker supplies, which never read the pads: the switched prover (`p`'s view
  where its rows hold, else the honest prover's on the pads Lemma A moves) has `S_shvzk`'s distribution on the honest
  `InnerHolds` alone (`sw_shvzk`, `worker_shvzk`), and differs from `p`'s view only in `Y₁`, `C_h` and the pads tree
  (`swView_outer`). It is the honest prover at the worker's values rectified, which pass the inner constraint at every
  pad, on a tape moved by a bijection (`swView_sameDist_rect`, `rect_innerHolds`), and Lemma B at such row-satisfying
  values is `table_shvzk`'s (`vals_shvzk`); each also with real hm96 leaves (`worker_shvzk_hm96`, `vals_shvzk_hm96`).
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
  (`LeanRelease.fillOf`), and every way a worker fails is a failed round (`LeanRelease.pass_of_none`, `nextD_of_fail`).
  Its `RefD` is the item-stream refinement `LeanRelease.ItemsRefine` alone, with no assumption on what the firewall
  computes (`LiveRefine` at the worker's passed rounds and `FillRefine` at the scheduled rounds, read at the values the
  firewall's transcript reports, where the frame takes the coins, and `StopRefine`, the stop where it refuses them;
  `LeanRelease.refD`), so
  **`Flock.zk_server_lean`** is `ZkServer.zk_deferred` at it, at any worker: one bit a session.
* `Refine`: toward `ItemsRefine` at the reference session's prover (`strat`). Its `StopRefine` holds wherever the
  session sends the stop where the frame refuses the coins (`LeanRelease.stopRefine_of_sessionStops`), and so wherever
  the session's message is the frame's release (`SessionFrames`, `stopRefine_of_sessionFrames`) and its abort is the
  stop. The frame checks no opening, so `LiveRefine` makes the release the abort after any opening the session refuses
  (`LeanRelease.release_eq_abort_of_liveRefine`). The frame's coin refusal reads no tape (`LeanRelease.refusesCoins_tape`),
  and `RefSession.withStops` is a session with the stop where a rule of the coins says, keeping the session's six Props;
  where that rule is the frame's refusals up to the protocol's last coin wherever the openings pass
  (`LeanRelease.StopsAt`), its `StopRefine` is proved (`LeanRelease.stopRefine_withStops`), and its `LiveRefine` and
  `FillRefine` follow from the transport's refusal of a failed opening, the frame's end past the last coin
  (`LeanRelease.EndsAt`) and the per-round byte equality at the worker's and the stand-in rounds
  (`LeanRelease.liveRefine_withStops`, `fillRefine_withStops`), so `Flock.zk_server_lean_stops` assumes no part of
  `ItemsRefine`; `Flock.zk_server_lean_coinLayer` is it at the deployed coin layer (`CoinLayer.session`). A transport
  that reads the history only through the decoded coins (`LeanRelease.ReadsCoins`) gives the rule and `StopsAt`
  (`LeanRelease.stopsAt_of_readsCoins`).
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
