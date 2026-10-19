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
import Proofs.Flock.Claims.ZK.Worker
import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server
import Proofs.Flock.Claims.ZK.Blind
import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.Claims.ZK.SamplingNonVacuous

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
  server's draws nor its messages (`ZkServer.lift`, `ZkServer.Blind`).
* `Blind`: the switched prover at the reference session (`RefSession.swP`, at values `RefSession.SwVals` indexed by what
  a pad-blind worker sees), with `refP`'s simulator, which never reads the prover's messages (`Proto.simulatorE_msg`):
  **`Flock.zk_deployZk_sw`**, `Claims.Flock.ZKAt` at the same `ε_zk`, `γ` and simulator as `Flock.zkAt_deployZk_ref`
  for any values whose round encoding is causal (`RefSession.SwCausal`), `Flock.zkAt_deployZk_ref` its case at the
  tables' own values (`Flock.zkAt_deployZk_ref_of_sw`); and **`Flock.zk_server_blind`**, the server at
  every pad-blind worker within `D.V.k x + 2` of the guessing simulator, with no term for the masked checks, and at a
  statement of a class `A` with `TapeOK` on `A` only (`Flock.zk_server_blind_on`), at the deployed coin layer with the
  tables' tape read with fibers of one size on `A` (`Flock.zk_server_blind_deployed`).
* `Release`: the deployed release (`LeanRelease`), the ZK server whose frame is the compiled `Flock.Firewall.Frame.round`
  (`LeanRelease.refMsg_toSrv`, by definition). Its `Ref` is D8 (a)'s `FirewallComputes` plus the item-stream refinement
  `LeanRelease.ItemsRefine` (`LeanRelease.ref`), so **`Flock.zk_server_lean`** is `Flock.zk_server_ref` at it with
  those two in place of `hRef`.
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
-/
