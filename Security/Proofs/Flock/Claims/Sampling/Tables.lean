import Proofs.Flock.Claims.Sampling.Committed
import Proofs.Flock.Claims.ZK.Deployed

/-!
# Sampling's zero knowledge at the sessions' tables

`Sampling.ZK`'s `zk_of_models` takes four Props of the models at each leaf layer. At models whose tables are tables
the session sets up (`Proto.TPreModel`, `Proto.TCoinModel`), two of them are theorems, as at Flock's
`Flock.zk_deployZk_of_tables`: what `NonDegenerate` and `InnerHolds` ask is in `Proto.TablesOK`, and the SHVZK simulator
is built from the models, so `SimLink` holds by construction. What remains at each leaf layer is `Proto.TMsgLink` and
`Proto.TablesOK`.

The models' simulator tapes are their tables', which may depend on the leaf layer, and a `SimLeak`'s coins do not
(`Claims.Sampling.ZK` runs one simulator at every leaf layer). So the simulator's tape (`Leaky.TabTape`) holds, at each
nonce, coin vector and `m < n_S(x)`, an element of `Fin (m + 1)`; at a leaf layer it reads the entry of its model's tape
size `N` through `Fin N ≃ SimAt` (`pickAt`), whose fibers are all of one size when `N ≤ n_S(x)` (`pickAt_fiber`).

* `Leaky.withTabSim`: `L` with that simulator, at each leaf layer the model's messages there (`TCoinModel.simAt`); its
  prover, coins and opening check are `L`'s. `withTabSim_shvzk` and `withTabSim_prefixHides` are its `SHVZK` and
  `PrefixHides` at each leaf layer, from the models.
* `zk_of_tables`: `ZKAt` at `L.G.prover` and `S_GK` of `withTabSim`, under `Claims.Hm96Hiding` at the pinned key, from
  `TMsgLink` and `TablesOK` at each leaf layer, the models' sizes and `CoinCheck`.
* `SamplingDeploy.zk_deploy_of_tables_gap`: at the deployed record and the key's hiding gap;
  `SamplingDeploy.zk_sampled_tables`, `zk_sampled_tables_mask` and `sampling_deployed_tables`: at the sampled masked
  protocol (`Sampled.leaky`), its rounds from `HelloRounds` and its coin check by definition.

**What `TMsgLink` asks of the sampled prover before its last coin.** `TPreModel.Link`, like `PreModel.Link`, asks every
message before the last coin to be `out` of the tables' pre-final cut views at the executable leaves. A checked
session's final message opens columns: their contents and salts, which those leaves hide, so it is no function of the
cut views. So `TMsgLink` can hold at `Sampled.leaky` only where every checked session's final message comes after the
run's last coin. Where the sessions run one after another, each earlier session's final message is in the prefix, and
`PrefixHides` needs instead each earlier session's whole view hidden against coins chosen after its commitments.
-/

namespace Claims.Sampling

open Claims Claims.ZKMal
open FlockSoundness.Discharge

/-- The tape's entry at `q` and `N - 1`, read in `Fin N`. -/
def pickFin {ι : Type} {B : ℕ} (s : ∀ p : ι × Fin B, Fin (p.2.val + 1)) (q : ι) (N : ℕ) (h0 : 0 < N)
    (hB : N ≤ B) : Fin N :=
  have hN : N - 1 + 1 = N := by omega
  Fin.cast hN (s (q, ⟨N - 1, by omega⟩))

open Classical in
/-- **The tape's element of a finite type `α`** at `q`: its entry of `α`'s size, through `Fin |α| ≃ α`, where
`|α| ≤ B`. -/
noncomputable def pickAt {ι : Type} {B : ℕ} (α : Type) [Fintype α] [Nonempty α]
    (s : ∀ p : ι × Fin B, Fin (p.2.val + 1)) (q : ι) : α :=
  if h : Fintype.card α ≤ B then (Fintype.equivFin α).symm (pickFin s q _ Fintype.card_pos h)
  else Classical.arbitrary α

/-- **`pickAt` has fibers of one size** where `|α| ≤ B`. -/
theorem pickAt_fiber {ι : Type} {B : ℕ} (α : Type) [Fintype α] [Nonempty α] (q : ι) (h : Fintype.card α ≤ B)
    (a a' : α) :
    Nat.card {s : ∀ p : ι × Fin B, Fin (p.2.val + 1) // pickAt α s q = a} =
      Nat.card {s : ∀ p : ι × Fin B, Fin (p.2.val + 1) // pickAt α s q = a'} := by
  have h0 : 0 < Fintype.card α := Fintype.card_pos
  let i : ι × Fin B := (q, ⟨Fintype.card α - 1, by omega⟩)
  have hlt : ∀ b : α, (Fintype.equivFin α b).val < i.2.val + 1 := fun b => by
    have := (Fintype.equivFin α b).isLt
    show _ < Fintype.card α - 1 + 1
    omega
  have hc : ∀ b : α, Nat.card {s : ∀ p : ι × Fin B, Fin (p.2.val + 1) // pickAt α s q = b} =
      Nat.card {s : ∀ p : ι × Fin B, Fin (p.2.val + 1) // s i = ⟨_, hlt b⟩} := fun b =>
    Nat.card_congr (Equiv.subtypeEquivRight fun s => by
      simp only [pickAt, h, ↓reduceDIte, Equiv.symm_apply_eq, Fin.ext_iff]
      rfl)
  rw [hc a, hc a']
  exact Proto.eval_fiber_card (β := fun p : ι × Fin B => Fin (p.2.val + 1)) i _ _

namespace Leaky

variable {W : Type}

/-- **The simulator's tape at the sessions' tables**: at each nonce, coin vector and `m < n_S(x)`, an element of
`Fin (m + 1)`. -/
abbrev TabTape (L : Leaky W) (nS : Msg → ℕ) (x : Msg) : Type :=
  ∀ p : (L.G.Nonce × (Fin (L.G.K x) → L.G.Coin)) × Fin (nS x), Fin (p.2.val + 1)

/-- The SHVZK simulator the models give at the leaf layer `lx`: the model's messages at `(ν, c)` from the tape's element
of its tape (`pickAt`). -/
noncomputable def tabSim (L : Leaky W) [Fintype L.G.Coin]
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nS : Msg → ℕ)
    (lx : List Msg) (x : Msg) (ν : L.G.Nonce) (c : Fin (L.G.K x) → L.G.Coin) (s : L.TabTape nS x) : OC (List Msg) :=
  (Mc lx x ν c).simAt (pickAt _ s (ν, c)) ν c

open Classical in
/-- **`L` with the SHVZK simulator its table models give**, on one tape at every leaf layer (`TabTape`): at `lx`,
`tabSim`. Its prover, coins and opening check are `L`'s. -/
noncomputable def withTabSim (L : Leaky W) [Fintype L.G.Coin]
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nS : Msg → ℕ) : Leaky W where
  G := { L.G with
    SimRand := L.TabTape nS
    finSim := fun _ => inferInstance
    neSim := fun _ => inferInstance
    sim := L.tabSim Mc nS [] }
  dmsg := L.dmsg
  sim := L.tabSim Mc nS

/-- **`SHVZK` of `withTabSim` at the leaf layer `lx`**, within `2·n_B·δ₁`, from the message links and the tables there,
where every model's tape has at most `n_S(x)` elements. -/
theorem withTabSim_shvzk (L : Leaky W) [Fintype L.G.Coin] (H : Msg → Msg) (R : Rel W) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nB nS : Msg → ℕ)
    (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x) (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (lx : List Msg) (hL : ∀ x w, R x w → ∀ ν c, (Mc lx x ν c).Link H ν c w) (hOK : ∀ x ν c, (Mc lx x ν c).OK) :
    ((L.withTabSim Mc nS).proto lx).SHVZK H R fun x => 2 * nB x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := Proto.TCoinModel.instNonemptySimAt (Mc lx x ν c)
  exact Proto.TCoinModel.shvzk (G := L.proto lx) H hT1 hδ₁ ν c w E (Ω := ((L.withTabSim Mc nS).proto lx).SimRand x)
    (fun s => ((L.tabSim Mc nS lx x ν c s).eval H).1) (Mc lx x ν c) (fun s => pickAt _ s (ν, c)) _
    (fun p => pickAt_fiber _ (ν, c) (hnS lx x ν c) p p₀) (fun _ => rfl) (hL x w hR ν c) (hOK x ν c) (hnB lx x ν c)

/-- **`PrefixHides` of `withTabSim` at the leaf layer `lx`**, within `2·n_L·δ₁`, from the models before the last coin
there (`TPreModel.runs_close`). -/
theorem withTabSim_prefixHides (L : Leaky W) [Fintype L.G.Coin] (H : Msg → Msg) (R : Rel W) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).TPreModel x)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nL nS : Msg → ℕ)
    (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (lx : List Msg) (hL : ∀ x w, R x w → ∀ ν, (Mp lx x ν).Link H ν w)
    (hOK : ∀ x ν, (Mp lx x ν).OK) :
    ((L.withTabSim Mc nS).proto lx).PrefixHides H R fun x => 2 * nL x * δ₁.toReal :=
  ((L.withTabSim Mc nS).proto lx).prefixHides_of_runs H R _ fun x w hR ν V E => by
    have : ((Mp lx x ν).size : ℝ) ≤ nL x := by exact_mod_cast hnL lx x ν
    calc _ ≤ 2 * (Mp lx x ν).size * δ₁.toReal := (Mp lx x ν).runs_close H hT1 hδ₁ ν w (hL x w hR ν) (hOK x ν) V E
      _ ≤ 2 * nL x * δ₁.toReal := by gcongr

end Leaky

variable {Coin Col T : Type}

/-- **Sampling's zero knowledge at the sessions' tables**: `ZKAt` at `P = L.G.prover` and `S_GK` of `withTabSim`, at
`ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, under `Claims.Hm96Hiding` at the pinned key, from `TMsgLink`
and `TablesOK` at each leaf layer on the honest pairs whose leaf layer it is, the models' sizes and `CoinCheck`. -/
theorem zk_of_tables (H : Msg → Msg) (D : Deploy Coin Col T) (L : Leaky Wit) [Fintype L.G.Coin] (t : ℕ) (ht : 0 < t)
    (hzk : D.flock.zk = true) (hk : ∀ x s, D.parse x = some s → L.G.k x = D.V.k x) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).TPreModel x)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nL nB nS : Msg → ℕ)
    (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (L.proto lx).TMsgLink H (RAt H D lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (L.proto lx).TablesOK (Mp lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZKAt H D L.G.prover ((L.withTabSim Mc nS).simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have h := zk_of_leakyE H D (L.withTabSim Mc nS) t ht (fun x => 2 * nL x * δ₁.toReal)
    (fun x => 2 * nB x * δ₁.toReal) hzk hk
    (fun lx => L.withTabSim_prefixHides H _ hT1 hδ₁ Mp Mc nL nS hnL lx (fun x w hR ν => (hL lx x w hR ν).1)
      (hOK lx).1)
    (fun lx => L.withTabSim_shvzk H _ hT1 hδ₁ Mc nB nS hnB hnS lx (fun x w hR ν c => (hL lx x w hR ν).2 c)
      (hOK lx).2)
    (fun x => by positivity) ((L.withTabSim Mc nS).G.binding_of_coinCheck H hC)
  have e : (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3 : ℝ) ^ t) =
      fun x => 4 * (2 * nL x * δ₁.toReal) + 4 * t * (2 * nB x * δ₁.toReal) + 2 * (1 / 3) ^ t :=
    funext fun x => by ring
  rw [e]
  exact h

end Claims.Sampling

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal
open FlockSoundness.Discharge

instance Sampled.instFintypeCoin {c : Code} (S : Sampled c) : Fintype S.leaky.G.Coin := FlockDeploy.instFintypeCoin

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-- **Sampling's zero knowledge at the deployed record, at the sessions' tables and the key's hiding gap**
(`CROnly.hm96Hiding_gap`): `zk_of_tables` at `deploy rel c`, when the sampled verifier's rounds are the protocol's
(`K + 2 = c.V.k`) at every statement that parses. -/
theorem zk_deploy_of_tables_gap (H : Msg → Msg) (L : Leaky Wit) [Fintype L.G.Coin] (t : ℕ) (ht : 0 < t)
    (hK : ∀ x s, c.parse x = some s → L.G.K x + 2 = c.V.k x) (Mp : ∀ lx x, L.G.Nonce → (L.proto lx).TPreModel x)
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nL nB nS : Msg → ℕ)
    (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (L.proto lx).TMsgLink H (RAt H (deploy rel c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (L.proto lx).TablesOK (Mp lx) (Mc lx)) (hC : L.G.CoinCheck) :
    ZKAt H (deploy rel c) L.G.prover ((L.withTabSim Mc nS).simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_of_tables H (deploy rel c) L t ht rfl hK (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top Mp Mc nL nB nS hnL hnB hnS hL hOK hC

/-- **Sampling's zero knowledge at the deployed record, of the sampled masked protocol, at the sessions' tables**:
`zk_deploy_of_tables_gap` at `L = Sampled.leaky S`, its rounds from `HelloRounds` and its coin check by definition;
what remains is `TMsgLink` and `TablesOK` at each leaf layer. It is `ZKAt`, of `L`'s prover. -/
theorem zk_sampled_tables (H : Msg → Msg) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).TPreModel x)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).TCoinModel x)
    (nL nB nS : Msg → ℕ) (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (S.leaky.proto lx).TMsgLink H (RAt H (deploy rel c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (S.leaky.proto lx).TablesOK (Mp lx) (Mc lx)) :
    ZKAt H (deploy rel c) S.leaky.G.prover ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_deploy_of_tables_gap rel c H S.leaky t ht (S.rounds hR) Mp Mc nL nB nS hnL hnB hnS hL hOK S.coinCheck

/-- **Sampling's zero knowledge at the deployed record masked at the sampled masked protocol's tapes, at the sessions'
tables**: `Claims.Sampling.ZK` from `zk_sampled_tables`, where the code's reference prover on the masked tape is the
protocol's (`Honest`). -/
theorem zk_sampled_tables_mask (H : Msg → Msg) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → Reference.Tape) (hP : Honest c H S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).TPreModel x)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).TCoinModel x)
    (nL nB nS : Msg → ℕ) (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (S.leaky.proto lx).TMsgLink H (RAt H (deploy rel c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (S.leaky.proto lx).TablesOK (Mp lx) (Mc lx)) :
    ZK H ((deploy rel c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf
        ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_mask_of_zkAt H (deploy rel c) S.leaky.G.prover tapeOf _ _ _ hP
    (zk_sampled_tables rel c H S hR t ht Mp Mc nL nB nS hnL hnB hnS hL hOK)

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record, with zero knowledge of the sampled masked protocol at the sessions'
tables**: `sampling_deployed` at the record masked at `tapeOf` and `S_GK` of `withTabSim`, its `hZK` from
`zk_sampled_tables_mask`. -/
theorem sampling_deployed_tables (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hfixed : DrawFixed (deploy rel c)) (hsessions : SessionRuns H (deploy rel c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deploy rel c) iroots) (hfix : InteriorFixed H (deploy rel c) iroots)
    (hprog : ProgramScope (deploy rel c)) (hchecks : ChecksScope H (deploy rel c)) (hreveals : RevealsLe c)
    (hComplete : Complete H (deploy rel c)) (S : Sampled c) (hR : HelloRounds c) (t : ℕ) (ht : 0 < t)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → Reference.Tape) (hP : Honest c H S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).TPreModel x)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).TCoinModel x)
    (nL nB nS : Msg → ℕ) (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (S.leaky.proto lx).TMsgLink H (RAt H (deploy rel c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (S.leaky.proto lx).TablesOK (Mp lx) (Mc lx)) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      KS H (deploy rel c) ∧ KSCount H (deploy rel c) ∧ Bits H (deploy rel c) ∧
      Efficient H (deploy rel c) FlockDeploy.zkStepPoly ∧ Complete H (deploy rel c) ∧
      ZK H ((deploy rel c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf
          ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  sampling_deployed rel c H hKS hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks hreveals hComplete
    tapeOf ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)) _ _
    (zk_sampled_tables_mask rel c H S hR t ht tapeOf hP Mp Mc nL nB nS hnL hnB hnS hL hOK)

end SamplingDeploy
