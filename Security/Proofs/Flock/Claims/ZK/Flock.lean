import Proofs.Flock.Claims.ZK.GKProof
import Proofs.Flock.Claims.ZK.GKCost
import Proofs.Flock.Claims.ZK.GKExp
import Proofs.Flock.Claims.ZK.CoinCheck
import Proofs.Flock.Claims.ZK.Prefix
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's zero knowledge against a malicious verifier, from its committed coins

`Flock.zkR_of_proto`: a deployed masked verifier (`D.zk`) whose rounds with `Hello` are a committed-coin protocol's
(`D.V.k + 1 = G.k`: `Hello`, answered by the coin commitment, then `K` rounds each answered by an opened coin, then the
last) meets the spec's claim `Claims.Flock.ZK` at the protocol's honest prover `P_zk = G.prover` and the Goldreich–Kahan
simulator `S_Flock = G.simulator N`, at

  `ε_zk(x) = 1/(N+1) + δ_A(x) + N·δ_B(x)` and `γ(x) = N`,

plus `γ` times the probability that `Fork`, the honest prover's runs on one nonce and two tapes against `V*`, finds a
collision among the prover's SHA-512 calls, with cost `G.simBound N` (`Flock.simCost_of_proto`). The verifier's
decision does not enter: zero knowledge reads only its rounds and that it is the masked one.

`Flock.zkR_gkE_of_proto`: from the same hypotheses, the session analysis's simulator with estimation,
`S_GK = G.simulatorE t (t·3^t)`, meets `Claims.Flock.ZK` at

  `ε_zk(x) = 4·δ_A(x) + 4t·δ_B(x) + 2·3^{-t}` and `γ(x) = 4t·3^t`,

at `(1 + 5t)` runs in expectation (`Flock.simCostExp_of_proto`) and `1 + 5t·3^t` at worst
(`Flock.simCost_gkE_of_proto`); at a cap `M`, `Proto.gkE_zk` gives `4·δ_A + 4t·δ_B + 3^{-t} + t/M` and `γ = 4M`.

**The rounds.** A simulator against a malicious `V*` needs the coins fixed before the prover's first message, which only
the coin commitment at `Hello` does (`verity/ml/flock/live/PROTOCOL.md` §9). `FlockDeploy.V` is the public-coin form,
whose `k = 2 + rounds` messages (`next_acc_length`) are the link, the recorded rounds and the last, each but the last
answered by a live coin in the clear, so `Claims.Flock.ZK` reads `D.V.k + 1` rounds, `Hello` first. The protocol's
coins are `D.V`'s, `K = D.V.k − 1`, and it has `G.k = D.V.k + 1` messages (`deployZk_proto_k`), so at
`FlockDeploy.deployZk` the claim's rounds are the protocol's (`Flock.zkR_deployZk`, `Flock.zkR_gkE_deployZk`).

**What changes for a malicious verifier**, against `Flock.Guarantees.ZeroKnowledgeHidden` (honest-verifier zero
knowledge at each fixed coin vector the prover does not refuse, on runs its self-check does not stop):

* the coins: `V*` chooses them, so the server commits every coin at `Hello` and the prover checks each opening before it
  uses the coin (`Proto.chk`); the simulator rewinds `V*` on the same `Hello` until it opens the same coins again;
* the error gains `1/(N+1)` (the rewinds' truncation, `rewind_close`) and `δ_A` (the extraction run on the dummy
  witness, `PrefixHides`), and `δ_B` (the old result's term) is paid once per rewind;
* the binding of the coin commitment is computational: a rewind that opens another coin is, in the real game, a
  SHA-512 collision among the honest prover's calls in two of its runs on one nonce (`mism_coll`), so the claim carries
  `γ` times `Fork`'s success (`ZKmalR`), a finder built from `V*` and the honest prover alone, and ZK bits follow from
  `StrictBirthday` at `Fork` (`ZKmalR.bits`);
* the claim fixes `V*`'s rounds to the protocol's (`Vs.k = D.V.k + 1`).

**Against the session analysis's Theorem GK** (`FlockSoundness.ZK.GK.gk_simulate_hm96`). It bounds the simulator with
threshold `t` and cap `M` at `3δ_C + 8t·δ_B + 3^{-t} + t/M + q²/2^513`, over a `Model` of one session whose rewinds are
functions of a view. `GKExp` builds that `Model` from the protocol at each nonce, with `V*`'s answers as an oracle
(`Proto.model`), and runs the same simulator as a `Claims.Simulator` on tapes fixed in advance (`Proto.simulatorE`), so
the old hybrids give `ZKmalR` with hybrid 1's binding failure in place of `q²/2^513`, which `PrefixHides` moves to the
real run, where it is `Fork`'s (`Proto.gkE_zk`), and the old `sim_runs_ev` and `sim_runs_le` its expected and
worst-case cost. Its inputs are this module's three
hypotheses, each from the old result's: Lemma C (`Session.adaptive_prefinal_hm96`, `δ_C = 2|L|·δ₁`) is `PrefixHides`
once a `V*` run's first `K + 1` messages are the old model's run (`Proto.prefixHides_of_runs`,
`Proto.runs_close_hm96`); Lemma B (`Session.session_shvzk_hm96`, `δ_B = 2·n_hid·δ₁`) is `SHVZK` where the prover goes
on (`Proto.shvzk_at_hm96`), and Lemma C at fixed answers where it refuses (`Proto.shvzk_refused_hm96`, `2|L|·δ₁`);
T6 (`coinPairs_coll`) is `Binding` once `chk` is the keyed coin tree's walk with its hash calls logged
(`Proto.binding_of_tree`). `δ₁` is `Hm96Hiding` (`hash-derived-key`, `2^-193`). `Flock.zkR_gkE_hm96` is the three
together: `ε_zk = 8·|L|·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` with `n_B ≥ max(n_hid, |L|)`, and `γ = 4t·3^t`.
`S_Flock`'s `N` rewinds cost `1/(N+1)` at `N + 1` runs and `N·δ_B`; `S_GK`'s estimation costs `3^{-t}` and `4t·δ_B`
at `1 + 5t` runs in expectation, and its cap `t/M` at `1 + 5M` at worst. A `Claims.Simulator` has finitely many coins,
so `S_GK` keeps the cap the deployed `gk_simulate` does not have; at `M = t·3^t` it costs another `3^{-t}`, and nothing
in expectation. `S_GK` fails at a binding failure, where `S_Flock` goes on to the next rewind, and both are in the
collision term.

**What the masked verifier's instance still assumes** (`G : Proto Flock.Wit` for the deployed `--zk` protocol), as
hypotheses of `Flock.zkR_gkE_hm96` and of the theorems that discharge them, each named as a Prop in
`Proofs.Flock.Claims.ZK.Hypotheses`:

* `P_zk`'s message function per coin round (`Proto.msg`, `Proto.dmsg`): the masked prover (`flock-circuit --zk`) has no
  Lean model of its messages, so their link to the session analysis's prover is a hypothesis. At each statement and
  nonce, its first `K + 1` messages at fixed answers are the session's strategy on randomness read through a map with
  fibers of one size (`hlink`, `hlink₀` of `Proto.runs_close_hm96`), given by its leaves (`hP`) and in world `W₀` by the
  pre-final view cut where it stops (`hI`); at each coin vector, its messages are a function of the session's real view
  where it goes on (`hR` of `Proto.shvzk_at_hm96`) and of its leaves where it refuses (`hR`, `hD` of
  `Proto.shvzk_refused_hm96`) (`Proto.MsgLink`);
* the SHVZK simulator (`Proto.sim`): the same function of the tables' `S_shvzk` outputs where the prover goes on (`hS`),
  and the dummy-witness prover's messages where it refuses (`Proto.SimLink`);
* where the prover goes on, the tables are non-degenerate (`Rank`, `PadOnto`, each `W^r` a bijection, `PadsOnto`,
  `β ≠ 0`; `Proto.NonDegenerate`) and a valid witness's pads satisfy the inner constraint `⟨λ, h⟩ = θ`
  (`Proto.InnerHolds`, which nothing proves); that it refuses at every other coin vector is its self-check's job;
* `Proto.chk` is the coin-tree walk `treeChk` and the coin is read from the opening's coins (`hchk`, `hdec`;
  `Proto.CoinCheck`), after which `Binding` needs nothing of `H`;
* `G.K = (FlockDeploy.V true).k − 1`: the protocol's coins are the deployed verifier's, the link's and the recorded
  rounds' (`1 + rounds` where the statement sets up, none where it does not). Zero knowledge reads nothing else of
  `FlockDeploy.deployZk` but its `zk`.
-/

namespace Claims.ZKMal

open Claims

/-- **Flock's zero knowledge against a malicious verifier, from a committed-coin protocol**: the spec's claim
`Claims.Flock.ZK` at `P_zk = G.prover` and `S_Flock = G.simulator N`, at `ε_zk = 1/(N+1) + δ_A + N·δ_B` and `γ = N`,
when the deployed verifier's rounds with `Hello` are the protocol's (`Hello`, its `K` coins, the last message). -/
theorem Flock.zkR_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (N : ℕ) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    (hA : G.PrefixHides H (Flock.R D.parse) δA) (hB : G.SHVZK H (Flock.R D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Claims.Flock.ZK H D G.prover (G.simulator N) (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N := by
  refine ⟨hzk, ?_⟩
  rw [show (fun x => D.V.k x + 1) = G.k from funext hk]
  exact G.gk_zk H _ N δA δB hA hB hB0 hC

/-- The deployed verifier has at least one message, the last, which it decides. -/
theorem deployV_k_pos (zk : Bool) (x : Msg) : 0 < (FlockDeploy.V zk).k x := by
  simp only [FlockDeploy.V]
  split <;> omega

/-- The deployed masked verifier's messages are a committed-coin protocol's coin rounds and its last, when the protocol's
coins are the deployed verifier's. -/
theorem deployZk_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (x : Msg) : (FlockDeploy.deployZk rel).V.k x = G.K x + 1 := by
  have := deployV_k_pos true x
  rw [hK x]
  show (FlockDeploy.V true).k x = _
  omega

/-- **The round count at `FlockDeploy.deployZk`**: a committed-coin protocol whose coins are the deployed verifier's has
`G.k = D.V.k + 1` messages, `D.V`'s and `Hello` before them. -/
theorem deployZk_proto_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (x : Msg) : G.k x = (FlockDeploy.deployZk rel).V.k x + 1 := by
  rw [deployZk_k rel G hK x]
  rfl

/-- **Flock's zero knowledge against a malicious verifier, at the deployed masked verifier** `FlockDeploy.deployZk`:
`Claims.Flock.ZK` at `P_zk = G.prover` and `S_Flock = G.simulator N`, at `ε_zk = 1/(N+1) + δ_A + N·δ_B` and `γ = N`,
for a committed-coin protocol whose coins are the deployed verifier's. -/
theorem Flock.zkR_deployZk (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (G : Proto Flock.Wit) (N : ℕ) (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) G.prover (G.simulator N)
      (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N :=
  Flock.zkR_of_proto H _ G N δA δB rfl (fun x => (deployZk_proto_k rel G hK x).symm) hA hB hB0 hC

/-- `S_Flock`'s cost (`SimCost`), at `simBound`: its worst case. -/
theorem Flock.simCost_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (N : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulator N) fun x => (G.simBound N cChk cMsg cSim x : ℝ) :=
  (G.gk_cost H N cChk cMsg cSim hc).simCost

/-! ## The simulator with estimation -/

/-- **Flock's zero knowledge against a malicious verifier, from a committed-coin protocol, at expected cost**: the
spec's claim `Claims.Flock.ZK` at `P_zk = G.prover` and `S_GK = G.simulatorE t (t·3^t)`, the session analysis's simulator,
at `ε_zk = 4·δ_A + 4t·δ_B + 2·3^{-t}` and `γ = 4t·3^t`, when the deployed verifier's rounds with `Hello` are the
protocol's. -/
theorem Flock.zkR_gkE_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    (hA : G.PrefixHides H (Flock.R D.parse) δA) (hB : G.SHVZK H (Flock.R D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Claims.Flock.ZK H D G.prover (G.simulatorE t (t * 3 ^ t)) (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  refine ⟨hzk, ?_⟩
  rw [show (fun x => D.V.k x + 1) = G.k from funext hk]
  exact G.gkE_zk_pow H _ t ht δA δB hA hB hB0 hC

/-- **The same at `FlockDeploy.deployZk`.** -/
theorem Flock.zkR_gkE_deployZk (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkR_gkE_of_proto H _ G t ht δA δB rfl (fun x => (deployZk_proto_k rel G hK x).symm) hA hB hB0 hC

/-! ## From the session analysis -/

/-- **Flock's zero knowledge against a malicious verifier, from the session analysis**: `Claims.Flock.ZK` at
`P_zk = G.prover` and `S_GK = G.simulatorE t (t·3^t)`, at

  `ε_zk(x) = 8·n_L(x)·δ₁ + 8t·n_B(x)·δ₁ + 2·3^{-t}` and `γ(x) = 4t·3^t`,

when the deployed verifier's rounds with `Hello` are the protocol's and

* (Lemma C) the committed-coin prover's runs before its last message, on the witness and on the dummy, are within
  `2·n_L(x)·δ₁` against every responder (`hA`: `Proto.runs_close_hm96` at each statement, witness and nonce, `n_L` the
  pre-final view's leaves);
* (Lemmas B and C) `SHVZK` holds at `2·n_B(x)·δ₁` (`hB`: `Proto.shvzk_at_hm96` where the prover goes on and
  `Proto.shvzk_refused_hm96` where it refuses, `n_B` at least the hidden leaves and the pre-final ones);
* (T6) the opening check is the coin-tree walk and the coin is read from the opening's coins (`hchk`, `hdec`).

`δ₁` is `Hm96Hiding`'s; the binding term is `γ` times `Fork`'s collision probability among the honest prover's calls
(`ZKmalR`). -/
theorem Flock.zkR_gkE_hm96 {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x) (δ₁ : ENNReal) (nL nB : Msg → ℕ)
    (hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : G.Nonce) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
      (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * nL x * δ₁.toReal)
    (hB : G.SHVZK H (Flock.R D.parse) fun x => 2 * nB x * δ₁.toReal) {CoinT Salt : Type}
    (Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT Salt) (nb : G.Nonce → List UInt8)
    (root : Msg → Msg → Dg) (parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT Salt))
    (pos : Msg → Msg → ℕ → List UInt8) (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List CoinT → G.Coin)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (Tr x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m))
    (hdec : ∀ m op, parse m = some op → G.dec m = dc op.cs) :
    Claims.Flock.ZK H D G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have h := Flock.zkR_gkE_of_proto H D G t ht (fun x => 2 * nL x * δ₁.toReal) (fun x => 2 * nB x * δ₁.toReal) hzk hk
    (G.prefixHides_of_runs H _ _ hA) hB (fun x => by positivity)
    (G.binding_of_tree H Tr nb root parse pos idx n depth dc hchk hdec)
  have e : (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3 : ℝ) ^ t) =
      fun x => 4 * (2 * nL x * δ₁.toReal) + 4 * t * (2 * nB x * δ₁.toReal) + 2 * (1 / 3) ^ t :=
    funext fun x => by ring
  rw [e]
  exact h

/-- **The same at `FlockDeploy.deployZk`.** -/
theorem Flock.zkR_gkE_hm96_deployZk (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (δ₁ : ENNReal) (nL nB : Msg → ℕ)
    (hA : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * nL x * δ₁.toReal)
    (hB : G.SHVZK H (Flock.R (FlockDeploy.deployZk rel).parse) fun x => 2 * nB x * δ₁.toReal) {CoinT Salt : Type}
    (Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT Salt) (nb : G.Nonce → List UInt8)
    (root : Msg → Msg → Dg) (parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT Salt))
    (pos : Msg → Msg → ℕ → List UInt8) (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List CoinT → G.Coin)
    (hchk : ∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (Tr x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m))
    (hdec : ∀ m op, parse m = some op → G.dec m = dc op.cs) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zkR_gkE_hm96 H _ G t ht rfl (fun x => (deployZk_proto_k rel G hK x).symm) δ₁ nL nB hA hB Tr nb root parse
    pos idx n depth dc hchk hdec

/-- `S_GK`'s cost: `(1 + 5t)` runs in expectation (`SimCost`) at every cap `M`. -/
theorem Flock.simCostExp_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulatorE t M) fun x => (1 + 5 * t) * G.runBound cChk cMsg cSim x :=
  G.gkE_costExp H t M cChk cMsg cSim hc

/-- `S_GK`'s worst case: `1 + 5M` runs (`SimCostLe`). -/
theorem Flock.simCost_gkE_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCostLe H (G.simulatorE t M) fun x => (1 + 5 * M) * G.runBound cChk cMsg cSim x :=
  G.gkE_cost H t M cChk cMsg cSim hc

end Claims.ZKMal
