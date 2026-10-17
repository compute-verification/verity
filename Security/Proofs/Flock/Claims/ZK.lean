import Proofs.Flock.Claims.ZK.Statement
import Proofs.Flock.Claims.ZK.GK
import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Coins
import Proofs.Flock.Claims.ZK.Forced
import Proofs.Flock.Claims.ZK.Rewind
import Proofs.Flock.Claims.ZK.GKProof
import Proofs.Flock.Claims.ZK.GKCost
import Proofs.Flock.Claims.ZK.GKExp
import Proofs.Flock.Claims.ZK.CoinCheck
import Proofs.Flock.Claims.ZK.Prefix
import Proofs.Flock.Claims.ZK.Shvzk
import Proofs.Flock.Claims.ZK.Flock
import Proofs.Flock.Claims.ZK.Hypotheses
import Proofs.Flock.Claims.ZK.Prefinal
import Proofs.Flock.Claims.ZK.Deployed
import Proofs.Flock.Claims.ZK.Reference
import Proofs.Flock.Claims.ZK.Tape
import Proofs.Flock.Claims.ZK.Worker
import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server
import Proofs.Flock.Claims.ZK.Blind
import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.Claims.ZK.SamplingNonVacuous

/-!
# Flock's zero knowledge against a malicious verifier

* `Statement`: what the proofs add to the spec's `ZKmalR` and `Claims.Flock.ZK`: ZK bits under `StrictBirthday` at
  `Fork` (`ZKmalR.bits`) and the worst-case cost `SimCostLe`.
* `GK`: a committed-coin protocol (`Proto`), its honest prover, its Goldreich–Kahan simulator with `N` rewinds
  (`Proto.simulator`), and the three facts the theorem reads (`PrefixHides`, `SHVZK`, `Binding`).
* `Interact`: what the programs evaluate to: `runStar` and the simulator's runs as one interaction (`runStar_msgs`,
  `spRun_eval`), two provers that agree on a transcript's histories give it to both (`interact_take_congr`), and each
  round's calls of the prover are in the run's `pLog`, so a collision among two runs' calls is `Fork`'s (`Fork_pr_ge`).
* `Coins`: what the coin checks evaluate to: a transcript opens `c` exactly when every opening is answered and passes
  the check (`cls_eq_some_iff`), the forced walk agrees (`cls_eq_some_iff_fcls`), and a mismatch is two checked
  openings within the walk's calls (`fcls_eq_mism`).
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
  (`Flock.zk_of_models`; **`Flock.zk_deployZk_of_models`** at `FlockDeploy.deployZk`).
* `Prefinal`: Lemma C from world `W₀` alone, for any family of tables of their own types (`famCard_cut_eq`), and its
  bounds on the runs before the last coin and where the prover refuses (`Proto.runs_close_of_ideal`,
  `Proto.refused_close_of_ideal`).
* `Deployed`: the same at the tables `verify --zk` sets up, each with its own types (`Proto.TPreModel`,
  `Proto.TabModel`, `Proto.TOffModel`), where `NonDegenerate` and `InnerHolds` are `CoinedOK`'s (`SetupTable.ok`,
  `ZkHidden.zkViewAtJ_of_ok`), Lemma C's tables, leaves and world `W₀` are the session's by definition
  (`card_cutAt_ideal`, `exeOf_eq`) and the simulator is the models' (`Proto.withSim`, so `SimLink` holds by
  construction): **`Flock.zk_deployZk_of_tables`** under `Claims.Hm96Hiding` and three Props (`TMsgLink`, `TablesOK`,
  `CoinCheck`), and under those alone at the pinned key's hiding gap (`Flock.zk_deployZk_of_tables_gap`).
* `Reference`: the reference prover those three Props are about, `RefSession.refP`, defined from the session's data
  (its round encoding of the cut view, its final message of the tables' real views, refusing where a table doesn't
  admit the coins); `CoinCheck` holds of it by definition, `TablesOK` from `Tables` and `TMsgLink` from `Complete`,
  `DummyReg`, `Causal` and `TapeOK` (`viewPre_viewExe`: the real views' pre-final part is the cut view the round
  messages encode): **`Flock.zk_deployZk_ref`**, and at the pinned key's hiding gap (`Flock.zk_deployZk_ref_gap`).
* `Tape`: `TapeOK` makes the tape onto the leaves' salts (`RefSession.eP_surjective`), so a session has fewer tables than
  tapes at every statement (`RefSession.J_lt_card_of_tapeOK`), and none with statements of every table count meets it
  (`RefSession.not_tapeOK`): the masked scope bounds no table count.
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
  `refP`), and over a stream with at most `b` stops the leak is at most `log₂ ∑_{i ≤ b} C(m, i)·r^i` bits
  (`stream_leak`). At the worker's own values the split is exact and the checks' errors drop out
  (`ZkServer.split_own`, `ZkServer.zk_server_own`), and a pad-blind worker (`ZkServer.BWorker`) is handed neither the
  server's draws nor its messages (`ZkServer.lift`, `ZkServer.Blind`).
* `Blind`: the switched prover at the reference session (`RefSession.swP`, at values `RefSession.SwVals` indexed by what
  a pad-blind worker sees), with `refP`'s simulator, which never reads the prover's messages (`Proto.simulatorE_msg`):
  **`Flock.zk_deployZk_sw`**, `Claims.Flock.ZK` at the same `ε_zk`, `γ` and simulator as `Flock.zk_deployZk_ref` for
  any values whose round encoding is causal (`RefSession.SwCausal`); and **`Flock.zk_server_blind`**, the server at
  every pad-blind worker within `D.V.k x + 2` of the guessing simulator, with no term for the masked checks.
* `NonVacuous`: the claim is not vacuous. A prover that makes no calls has `Fork.Pr = 0`, so the simulator that hashes a
  collision and outputs the empty transcript fails `ZKmalR` at every `ε_zk < 1` and every `γ` (`not_zkmalR_collSim`),
  a prover that sends its witness has no simulator at `ε_zk < 1/2` (`not_zkmalR_leak`), and both fail Flock's `ZK` at
  a masked record whose relation holds everywhere (`toyD`).
* `SamplingNonVacuous`: the same for the sampling `ZK`: it fails at the colliding simulator given the leaf layer
  (`not_samplingZK_collSim`), and for the witness-sending prover at two honest witnesses of one leaf layer
  (`not_samplingZK_leak`).
-/
