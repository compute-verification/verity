import Proofs.Flock.Claims.ZK.Hypotheses
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# Sampling's zero knowledge from a committed-coin protocol

`Claims.Sampling.ZK` is `ZKmalR` at the sampled verifier's rounds, on the honest pairs whose published leaf layer is
`l x`, with the simulator given `l x`. Its binding term is `γ` times `Fork` of the sampled prover's own two runs on one
nonce, so it is the Goldreich–Kahan theorem at the sampled protocol itself, with every coin it draws (the selection's
and each checked unit's session's) committed at its `Hello`. Flock's zero knowledge at each session does not compose to
it: a hybrid over the sessions charges each session's `Fork` against a verifier that has already simulated the earlier
sessions, and no term of the sampled claim bounds those.

* `Leaky`: a committed-coin protocol (`Claims.ZKMal.Proto`) whose dummy-witness messages and SHVZK simulator also read
  the leakage; `Leaky.proto lx` is the protocol at the leakage `lx`. Its honest prover, coins and opening check read
  none, so `ZK`'s prover is one prover.
* `Leaky.simLeak N`, `Leaky.simLeakE t M`: `S_Flock` and `S_GK` given the leakage, as `SimLeak`s.
* `zk_of_leaky`, `zk_of_leakyE`: `Claims.Sampling.ZK` from `PrefixHides` and `SHVZK` of the protocol at each leakage
  `lx`, on the honest pairs whose leaf layer is `lx` (`RAt`), and `Binding`. `ZKmalR` runs one simulator per statement,
  so each statement takes the protocol at its own leakage.
* `zk_of_models`: the same from the session analysis's models at each leakage (`Proto.MsgLink`, `Proto.SimLink`,
  `Proto.NonDegenerate`, `Proto.InnerHolds`) and `Proto.CoinCheck`, under `Claims.Hm96Hiding`, as `Flock.zk_of_models`
  is Flock's. The models' tables are every checked unit's session tables; the selection's coins are among the committed
  coins, and the reveals are the sessions' witnesses. The leaf layer is the leakage, so the claim reads no hiding of it;
  that it hides the unopened rows is the leaf's own hm96 hiding, outside this claim.
* `zk_deploy_of_models`, `zk_deploy_of_models_gap`: at the deployed record `SamplingDeploy.deploy rel c`, whose rounds
  are the protocol's when `K + 2 = c.V.k` at every statement that parses, the second at the key's hiding gap
  (`CROnly.hm96Hiding_gap`), and `SamplingDeploy.sampling_deployed_models`, `sampling_deployed` with its zero
  knowledge from them.
-/

namespace Claims.Sampling

open Claims Claims.ZKMal

/-- **A committed-coin protocol with leakage**: `G`'s honest prover, coins and opening check, with its dummy-witness
messages (`dmsg`) and SHVZK simulator (`sim`) also given the leakage `lx`, here the published leaf layer. `G`'s own
`dmsg` and `sim` are not read. -/
structure Leaky (W : Type) where
  G : Proto W
  dmsg : List Msg → Msg → G.Nonce → G.Rand → List G.Coin → OC Msg
  sim : List Msg → (x : Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → G.SimRand x → OC (List Msg)

namespace Leaky

variable {W : Type}

/-- The protocol at the leakage `lx`. -/
def proto (L : Leaky W) (lx : List Msg) : Proto W := { L.G with dmsg := L.dmsg lx, sim := L.sim lx }

/-- **`S_Flock` given the leakage**, with `N` rewinds: at `lx`, the protocol's at `lx`. -/
def simLeak (L : Leaky W) (N : ℕ) : SimLeak where
  C x := L.G.Nonce × L.G.Rand × (Fin N → L.G.SimRand x)
  prog x lx c := ((L.proto lx).simulator N).prog x c

/-- **`S_GK` given the leakage**, at `t` and the cap `M`: at `lx`, the protocol's at `lx`. -/
def simLeakE (L : Leaky W) (t M : ℕ) : SimLeak where
  C x := L.G.Nonce × L.G.Rand × (Fin M → L.G.Rand) × (Fin (4 * M) → L.G.SimRand x)
  prog x lx c := ((L.proto lx).simulatorE t M).prog x c

end Leaky

variable {Coin Col T : Type}

/-- **`ZK`'s relation at the leakage `lx`**: the honest pairs whose published leaf layer is `lx`. -/
def RAt (H : Msg → Msg) (D : Deploy Coin Col T) (lx : List Msg) : Rel Wit :=
  fun x w => Rhon H D.commit D.parse D.partition x w ∧ leafLayer H D.commit D.parse x w = lx

/-- **Sampling's zero knowledge from a committed-coin protocol with leakage**, through `S_Flock` with `N` rewinds:
`ZK` at `P = L.G.prover`, at `ε_zk = 1/(N+1) + δ_A + N·δ_B` and `γ = N`, when the sampled verifier's rounds are the
protocol's at every statement that parses (`ZKmalR` reads only the honest pairs, whose statements parse). -/
theorem zk_of_leaky (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (N : ℕ) (δA δB : Msg → ℝ)
    (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x)
    (hA : ∀ lx, (L.proto lx).PrefixHides H (RAt H D lx) δA) (hB : ∀ lx, (L.proto lx).SHVZK H (RAt H D lx) δB)
    (hB0 : ∀ x, 0 ≤ δB x) (hC : L.G.Binding H) :
    ZK H D L.G.prover (L.simLeak N) (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N := by
  refine ⟨hzk, fun l x w hR Θ Vs θ hkV Ev => ?_⟩
  obtain ⟨s, _, hs, _⟩ := hR.1
  exact (L.proto (l x)).gk_zk H (RAt H D (l x)) N δA δB (hA _) (hB _) hB0 hC x w hR Θ Vs θ
    (hkV.trans (hk x s hs).symm) Ev

/-- **The same through `S_GK`** at `t` and the cap `t·3^t`: at `ε_zk = 4δ_A + 4t·δ_B + 2·3^{-t}` and `γ = 4t·3^t`. -/
theorem zk_of_leakyE (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (δA δB : Msg → ℝ) (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x)
    (hA : ∀ lx, (L.proto lx).PrefixHides H (RAt H D lx) δA) (hB : ∀ lx, (L.proto lx).SHVZK H (RAt H D lx) δB)
    (hB0 : ∀ x, 0 ≤ δB x) (hC : L.G.Binding H) :
    ZK H D L.G.prover (L.simLeakE t (t * 3 ^ t)) (fun x => 4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  refine ⟨hzk, fun l x w hR Θ Vs θ hkV Ev => ?_⟩
  obtain ⟨s, _, hs, _⟩ := hR.1
  exact (L.proto (l x)).gkE_zk_pow H (RAt H D (l x)) t ht δA δB (hA _) (hB _) hB0 hC x w hR Θ Vs θ
    (hkV.trans (hk x s hs).symm) Ev

/-- **Sampling's zero knowledge from the models**: `ZK` at `P = L.G.prover` and `S_GK` given the leakage, at
`ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, under `Claims.Hm96Hiding` and, at each leakage `lx`, the
message link, the simulator, the tables' non-degeneracy and `InnerHolds` on the honest pairs whose leaf layer is `lx`,
and the coin check. -/
theorem zk_of_models (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x) {Salt B Cc : Type}
    [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B)
    (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H D lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H D lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZK H D L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have h := zk_of_leakyE H D L t ht (fun x => 2 * nL x * δ₁.toReal) (fun x => 2 * nB x * δ₁.toReal) hzk hk
    (fun lx => (L.proto lx).prefixHides_of_runs H _ _
      ((L.proto lx).prefix_of_models H _ My cs hT1 hδ₁ (Mp lx) (Mc lx) nL (hnL lx) (hL lx)))
    (fun lx => (L.proto lx).shvzk_of_models H _ My cs hT1 hδ₁ (Mp lx) (Mc lx) nB (hnB lx) (hL lx) (hS lx) (hN lx)
      (hI lx))
    (fun x => by positivity) (L.G.binding_of_coinCheck H hC)
  have e : (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3 : ℝ) ^ t) =
      fun x => 4 * (2 * nL x * δ₁.toReal) + 4 * t * (2 * nB x * δ₁.toReal) + 2 * (1 / 3) ^ t :=
    funext fun x => by ring
  rw [e]
  exact h

end Claims.Sampling

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-- **Sampling's zero knowledge at the deployed record, from the models**: `zk_of_models` at `deploy rel c`, whose
sessions are `deployZk`'s, masked, when the sampled verifier's rounds are the protocol's (`K + 2 = c.V.k`) at every
statement that parses. -/
theorem zk_deploy_of_models (H : Msg → Msg) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZK H (deploy rel c) L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_of_models H (deploy rel c) L t ht rfl hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC

/-- **The same at the key's hiding gap** (`CROnly.hm96Hiding_gap`), with `Claims.Hm96Hiding` discharged: what remains is
the protocol's rounds, the four models' Props at each leakage and the coin check. -/
theorem zk_deploy_of_models_gap (H : Msg → Msg) (L : Leaky Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [DecidableEq Cc] (My : Salt → B)
    (cs : Salt → Cc) (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZK H (deploy rel c) L.G.prover (L.simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  zk_deploy_of_models rel c H L t ht hK My cs (FlockSoundness.CROnly.hm96Hiding_gap My cs) ENNReal.ofReal_ne_top Mp
    Mc nL nB hnL hnB hL hS hN hI hC

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record, with zero knowledge from the models**: `sampling_deployed` at
`P_zk = L.G.prover` and `S_GK` given the leaf layer, its `hZK` from `zk_deploy_of_models_gap`. -/
theorem sampling_deployed_models (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hfixed : DrawFixed (deploy rel c)) (hsessions : SessionRuns H (deploy rel c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deploy rel c) iroots) (hfix : InteriorFixed H (deploy rel c) iroots)
    (hprog : ProgramScope (deploy rel c)) (hchecks : ChecksScope H (deploy rel c)) (hreveals : RevealsLe c)
    {Ω : Type} (P : Wit → Prover Ω) (δc : Msg → ℝ) (hComplete : Complete H (deploy rel c) P δc)
    (L : Leaky Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [DecidableEq Cc] (My : Salt → B) (cs : Salt → Cc)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).PreModel x Salt B Cc)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ lx x ν, Fintype.card (Mp lx x ν).L ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hL : ∀ lx, (L.proto lx).MsgLink H (RAt H (deploy rel c) lx) My cs (Mp lx) (Mc lx))
    (hS : ∀ lx, (L.proto lx).SimLink H (Mc lx)) (hN : ∀ lx, (L.proto lx).NonDegenerate (Mc lx))
    (hI : ∀ lx, (L.proto lx).InnerHolds (RAt H (deploy rel c) lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      KS H (deploy rel c) ∧ KSCount H (deploy rel c) ∧ Bits H (deploy rel c) ∧
      Efficient H (deploy rel c) FlockDeploy.zkStepPoly ∧ Complete H (deploy rel c) P δc ∧
      ZK H (deploy rel c) L.G.prover (L.simLeakE t (t * 3 ^ t))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap My cs)).toReal + 2 * (1 / 3) ^ t)
        fun _ => 4 * t * 3 ^ t :=
  sampling_deployed rel c H hKS hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks hreveals P δc hComplete
    L.G.prover (L.simLeakE t (t * 3 ^ t)) _ _
    (zk_deploy_of_models_gap rel c H L t ht hK My cs Mp Mc nL nB hnL hnB hL hS hN hI hC)

end SamplingDeploy
