import Proofs.Flock.Claims.ZK.GKProof
import Proofs.Flock.Claims.ZK.GKCost
import Proofs.Flock.Claims.ZK.GKExp
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's zero knowledge against a malicious verifier, from its committed coins

`Flock.zkR_of_proto`: a deployed masked verifier (`D.zk`) whose rounds are a committed-coin protocol's (`D.V.k = G.k`:
`Hello`, answered by the coin commitment, then `K` rounds each answered by an opened coin, then the last) meets the
proposed claim `Flock.ZKR` at the protocol's honest prover `P_zk = G.prover` and the Goldreich–Kahan simulator
`S_Flock = G.simulator N`, at

  `ε_zk(x) = 1/(N+1) + 2·δ_A(x) + 2·N·δ_B(x)`

plus the probability of a collision among `S_Flock`'s own SHA-512 calls, with cost `G.simBound N`
(`Flock.simCost_of_proto`). The verifier's decision does not enter: zero knowledge reads only its rounds and that it
is the masked one.

`Flock.zkR_gkE_of_proto`: from the same hypotheses, the session analysis's simulator with estimation,
`S_GK = G.simulatorE t (t·3^t)`, meets `Flock.ZKR` at

  `ε_zk(x) = 3·δ_A(x) + 8t·δ_B(x) + 2·3^{-t}`

plus its own collision probability, at `(1 + 5t)` runs in expectation (`Flock.simCostExp_of_proto`) and `1 + 5t·3^t`
at worst (`Flock.simCost_gkE_of_proto`); at a cap `M`, `Proto.gkE_zk` gives `3·δ_A + 8t·δ_B + 3^{-t} + t/M`.

**The rounds.** A simulator against a malicious `V*` needs the coins fixed before the prover's first message, which only
the coin commitment at `Hello` does (`verity/ml/flock/live/PROTOCOL.md` §9), so the deployed `--zk` verifier's model
`D.V` is to have that round, and `Flock.ZKR` reads `D.V.k`. `FlockDeploy.V` does not have it yet: it is the public-coin
form, whose messages are the link, the recorded rounds and the last, each but the last answered by a live coin in the
clear. Until it does, `Flock.ZKRHello` reads `D.V.k + 1` rounds, `Hello` first, and the protocol's `K` coins are
`D.V`'s `D.V.k − 1` (`Flock.zkRHello_of_proto`, `Flock.zkRHello_deployZk` and `Flock.zkRHello_gkE_deployZk` at
`FlockDeploy.deployZk`).

**What changes for a malicious verifier**, against `Flock.Guarantees.ZeroKnowledgeHidden` (honest-verifier zero
knowledge at each fixed coin vector the prover does not refuse, on runs its self-check does not stop):

* the coins: `V*` chooses them, so the server commits every coin at `Hello` and the prover checks each opening before it
  uses the coin (`Proto.chk`); the simulator rewinds `V*` on the same `Hello` until it opens the same coins again;
* the error gains `1/(N+1)` (the rewinds' truncation, `rewind_close`) and `2·δ_A` (the extraction run on the dummy
  witness, `PrefixHides`), and `δ_B` (the old result's term) is paid once per rewind and twice over (the output and the
  binding failure both move through the hybrids);
* the binding of the coin commitment is computational: a rewind that opens another coin is a SHA-512 collision among
  the simulator's own calls (`mm_coll`), so the claim carries that probability (`ZKmalR`), and ZK bits follow from
  `StrictBirthday` at `Bzk` (`ZKmalR.bits`);
* the claim must fix `V*`'s rounds to the protocol's: as frozen, `ZKmal` holds only at `ε_zk ≥ 1` (`ZKmal_iff`).

**Against the session analysis's Theorem GK** (`FlockSoundness.ZK.GK.gk_simulate_hm96`). It bounds the simulator with
threshold `t` and cap `M` at `3δ_C + 8t·δ_B + 3^{-t} + t/M + q²/2^513`, over a `Model` of one session whose rewinds are
functions of a view. `GKExp` builds that `Model` from the protocol at each nonce, with `V*`'s answers as an oracle
(`Proto.model`), and runs the same simulator as a `Claims.Simulator` on tapes fixed in advance (`Proto.simulatorE`), so
the old `gk_close` gives `ZKmalR` with the simulator's collision probability in place of `q²/2^513` (`Proto.gkE_zk`),
and the old `sim_runs_ev` and `sim_runs_le` its expected and worst-case cost. Its inputs are this module's three
hypotheses: Lemma C (`prefinalClose_hm96`, `δ_C = 2|L|·δ₁`) is `PrefixHides`, Lemma B (`rewindClose_hm96`,
`δ_B = 2·n_hid·δ₁` at a completed first run's coins) is `SHVZK`, and T6 (`bindCollKeyed_isColl`) is `Binding` once
`chk` is the keyed coin tree's walk with its hash inputs logged; `δ₁` is `Hm96Hiding` (`hash-derived-key`, `2^-193`).
`S_Flock`'s `N` rewinds cost `1/(N+1)` at `N + 1` runs and `2N·δ_B`; `S_GK`'s estimation costs `3^{-t}` and `8t·δ_B`
at `1 + 5t` runs in expectation, and its cap `t/M` at `1 + 5M` at worst. A `Claims.Simulator` has finitely many coins,
so `S_GK` keeps the cap the deployed `gk_simulate` does not have; at `M = t·3^t` it costs another `3^{-t}`, and nothing
in expectation. `S_GK` fails at a binding failure, where `S_Flock` goes on to the next rewind, and both are in the
collision term.

**What the masked verifier's instance still needs** (`G : Proto Flock.Wit` for the deployed `--zk` protocol):

* `P_zk`'s message function per coin round (`Proto.msg`, `Proto.dmsg`): the masked prover (`flock-circuit --zk`) has no
  Lean model of its messages. The session analysis takes them as a parameter too (a causal `ZK.Strategy` described by
  the tables, `hP`, `hI`); the old result's real view (`ZkViewAtJ`) is the verifier's joint view of the `J` tables at a
  fixed coin vector, not a next-message function;
* `Proto.chk` as an oracle program walking the coin tree (`coin-tree/hm96-sha512/v2`), so that `Binding` follows from
  T6: `CoinBinding` and `bindCollKeyed_isColl` give a collision of `H`, not one among the check's logged calls;
* `SHVZK` at every coin vector, from Lemma B at the non-degenerate ones, with the refused coin vectors (where the honest
  prover aborts, a function of the coins) added;
* `PrefixHides` from Lemma C, and the honest prover's self-check never stopping at a valid witness (`InnerHolds`, which
  nothing proves); the dummy-witness prover skips the self-check, which would otherwise stop it;
* `G.K = (FlockDeploy.V true).k − 1`: the protocol's coins are the deployed verifier's, the link's and the recorded
  rounds' (`1 + rounds` where the statement sets up, none where it does not). Zero knowledge reads nothing else of
  `FlockDeploy.deployZk` but its `zk`.
-/

namespace Claims.ZKMal

open Claims

/-- **Flock's zero knowledge against a malicious verifier, from a committed-coin protocol**: the proposed claim
`Flock.ZKR` at `P_zk = G.prover` and `S_Flock = G.simulator N`, at `ε_zk = 1/(N+1) + 2·δ_A + 2·N·δ_B`, when the
deployed verifier's rounds are the protocol's (`Hello`, its `K` coins, the last message). -/
theorem Flock.zkR_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (N : ℕ) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x = G.k x)
    (hA : G.PrefixHides H (Flock.R H D.parse) δA) (hB : G.SHVZK H (Flock.R H D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Flock.ZKR H D G.prover (G.simulator N) fun x => 1 / ((N : ℝ) + 1) + 2 * δA x + 2 * N * δB x := by
  refine ⟨hzk, ?_⟩
  rw [show D.V.k = G.k from funext hk]
  exact G.gk_zk H _ N δA δB hA hB hB0 hC

/-- **The same before `D.V` models `Hello`** (`Flock.ZKRHello`): the protocol commits the coins of `D.V`'s public-coin
form (`D.V.k = G.K + 1`). -/
theorem Flock.zkRHello_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T)
    (G : Proto Flock.Wit) (N : ℕ) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x = G.K x + 1)
    (hA : G.PrefixHides H (Flock.R H D.parse) δA) (hB : G.SHVZK H (Flock.R H D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Flock.ZKRHello H D G.prover (G.simulator N) fun x => 1 / ((N : ℝ) + 1) + 2 * δA x + 2 * N * δB x := by
  refine ⟨hzk, ?_⟩
  have hk' : (fun x => D.V.k x + 1) = G.k := funext fun x => by
    show D.V.k x + 1 = G.K x + 2
    rw [hk x]
  rw [hk']
  exact G.gk_zk H _ N δA δB hA hB hB0 hC

/-- The deployed verifier has at least one message, the last, which it decides. -/
theorem deployV_k_pos (zk : Bool) (x : Msg) : 0 < (FlockDeploy.V zk).k x := by
  simp only [FlockDeploy.V]
  split <;> omega

/-- A committed-coin protocol whose coins are the deployed verifier's has one round fewer than `D.V`, `Hello`. -/
theorem deployZk_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) (x : Msg) : (FlockDeploy.deployZk rel).V.k x = G.K x + 1 := by
  have := deployV_k_pos true x
  rw [hK x]
  show (FlockDeploy.V true).k x = _
  omega

/-- **Flock's zero knowledge against a malicious verifier, at the deployed masked verifier** `FlockDeploy.deployZk`: the
bridge `Flock.ZKRHello` at `P_zk = G.prover` and `S_Flock = G.simulator N`, at `ε_zk = 1/(N+1) + 2·δ_A + 2·N·δ_B`,
for a committed-coin protocol whose coins are the deployed verifier's. `FlockDeploy.V` is the public-coin form, without
the `Hello` round; once it has it, `Flock.zkR_of_proto` gives `Flock.ZKR` itself. -/
theorem Flock.zkRHello_deployZk (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (G : Proto Flock.Wit) (N : ℕ) (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R H (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R H (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Flock.ZKRHello H (FlockDeploy.deployZk rel) G.prover (G.simulator N)
      fun x => 1 / ((N : ℝ) + 1) + 2 * δA x + 2 * N * δB x :=
  Flock.zkRHello_of_proto H _ G N δA δB rfl (deployZk_k rel G hK) hA hB hB0 hC

/-- `S_Flock`'s cost (`SimCost`), at `simBound`. -/
theorem Flock.simCost_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (N : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) : SimCost H (G.simulator N) (G.simBound N cChk cMsg cSim) :=
  G.gk_cost H N cChk cMsg cSim hc

/-! ## The simulator with estimation -/

/-- **Flock's zero knowledge against a malicious verifier, from a committed-coin protocol, at expected cost**: the
proposed claim `Flock.ZKR` at `P_zk = G.prover` and `S_GK = G.simulatorE t (t·3^t)`, the session analysis's simulator,
at `ε_zk = 3·δ_A + 8t·δ_B + 2·3^{-t}`, when the deployed verifier's rounds are the protocol's. -/
theorem Flock.zkR_gkE_of_proto {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (δA δB : Msg → ℝ) (hzk : D.zk = true) (hk : ∀ x, D.V.k x = G.k x)
    (hA : G.PrefixHides H (Flock.R H D.parse) δA) (hB : G.SHVZK H (Flock.R H D.parse) δB) (hB0 : ∀ x, 0 ≤ δB x)
    (hC : G.Binding H) :
    Flock.ZKR H D G.prover (G.simulatorE t (t * 3 ^ t)) fun x => 3 * δA x + 8 * t * δB x + 2 * (1 / 3) ^ t := by
  refine ⟨hzk, ?_⟩
  rw [show D.V.k = G.k from funext hk]
  exact G.gkE_zk_pow H _ t ht δA δB hA hB hB0 hC

/-- **The same at `FlockDeploy.deployZk`**, through the bridge `Flock.ZKRHello` until `FlockDeploy.V` has `Hello`. -/
theorem Flock.zkRHello_gkE_deployZk (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (δA δB : Msg → ℝ) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (hA : G.PrefixHides H (Flock.R H (FlockDeploy.deployZk rel).parse) δA)
    (hB : G.SHVZK H (Flock.R H (FlockDeploy.deployZk rel).parse) δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    Flock.ZKRHello H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      fun x => 3 * δA x + 8 * t * δB x + 2 * (1 / 3) ^ t := by
  refine ⟨rfl, ?_⟩
  rw [show (fun x => (FlockDeploy.deployZk rel).V.k x + 1) = G.k from
    funext fun x => by rw [deployZk_k rel G hK x]; rfl]
  exact G.gkE_zk_pow H _ t ht δA δB hA hB hB0 hC

/-- `S_GK`'s cost: `(1 + 5t)` runs in expectation (`SimCostExp`) at every cap `M`. -/
theorem Flock.simCostExp_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCostExp H (G.simulatorE t M) fun x => (1 + 5 * t) * G.runBound cChk cMsg cSim x :=
  G.gkE_costExp H t M cChk cMsg cSim hc

/-- `S_GK`'s worst case: `1 + 5M` runs (`SimCost`). -/
theorem Flock.simCost_gkE_of_proto (H : Msg → Msg) (G : Proto Flock.Wit) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulatorE t M) fun x => (1 + 5 * M) * G.runBound cChk cMsg cSim x :=
  G.gkE_cost H t M cChk cMsg cSim hc

end Claims.ZKMal
