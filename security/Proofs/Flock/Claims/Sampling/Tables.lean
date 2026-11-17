module

public import Proofs.Flock.Claims.Sampling.Committed
public import Proofs.Flock.Claims.ZK.Deployed

/-! Sampling's zero knowledge at the sessions' tables, where two of `zk_of_models`' four model Props are theorems. -/

@[expose] public section

namespace Claims.Sampling

open Claims Claims.ZKMal
open FlockSoundness.Discharge

def pickFin {ι : Type} {B : ℕ} (s : ∀ p : ι × Fin B, Fin (p.2.val + 1)) (q : ι) (N : ℕ) (h0 : 0 < N)
    (hB : N ≤ B) : Fin N :=
  have hN : N - 1 + 1 = N := by omega
  Fin.cast hN (s (q, ⟨N - 1, by omega⟩))

open Classical in
noncomputable def pickAt {ι : Type} {B : ℕ} (α : Type) [Fintype α] [Nonempty α]
    (s : ∀ p : ι × Fin B, Fin (p.2.val + 1)) (q : ι) : α :=
  if h : Fintype.card α ≤ B then (Fintype.equivFin α).symm (pickFin s q _ Fintype.card_pos h)
  else Classical.arbitrary α

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

abbrev TabTape (L : Leaky W) (nS : Msg → ℕ) (x : Msg) : Type :=
  ∀ p : (L.G.Nonce × (Fin (L.G.K x) → L.G.Coin)) × Fin (nS x), Fin (p.2.val + 1)

noncomputable def tabSim (L : Leaky W) [Fintype L.G.Coin]
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nS : Msg → ℕ)
    (lx : List Msg) (x : Msg) (ν : L.G.Nonce) (c : Fin (L.G.K x) → L.G.Coin) (s : L.TabTape nS x) : OC (List Msg) :=
  (Mc lx x ν c).simAt (pickAt _ s (ν, c)) ν c

open Classical in
noncomputable def withTabSim (L : Leaky W) [Fintype L.G.Coin]
    (Mc : ∀ lx x, L.G.Nonce → (Fin (L.G.K x) → L.G.Coin) → (L.proto lx).TCoinModel x) (nS : Msg → ℕ) : Leaky W where
  G := { L.G with
    SimRand := L.TabTape nS
    finSim := fun _ => inferInstance
    neSim := fun _ => inferInstance
    sim := L.tabSim Mc nS [] }
  dmsg := L.dmsg
  sim := L.tabSim Mc nS

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

/-- What remains is `TMsgLink` and `TablesOK` at each leaf layer; `TMsgLink` holds at `Sampled.leaky` only where every
checked session's final message comes after the run's last coin. -/
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
theorem sampling_deployed_tables (H : Msg → Msg) (hslots : DrawSlots (deploy rel c))
    (hprog : ProgramScope (deploy rel c)) (hComplete : Complete H (deploy rel c)) (S : Sampled c) (hR : HelloRounds c)
    (t : ℕ) (ht : 0 < t)
    (tapeOf : S.leaky.G.Nonce × S.leaky.G.Rand → Reference.Tape) (hP : Honest c H S.leaky.G tapeOf)
    (Mp : ∀ lx x, S.leaky.G.Nonce → (S.leaky.proto lx).TPreModel x)
    (Mc : ∀ lx x, S.leaky.G.Nonce → (Fin (S.leaky.G.K x) → S.leaky.G.Coin) → (S.leaky.proto lx).TCoinModel x)
    (nL nB nS : Msg → ℕ) (hnL : ∀ lx x ν, (Mp lx x ν).size ≤ nL x) (hnB : ∀ lx x ν c, (Mc lx x ν c).size ≤ nB x)
    (hnS : ∀ lx x ν c, Fintype.card (Mc lx x ν c).SimAt ≤ nS x)
    (hL : ∀ lx, (S.leaky.proto lx).TMsgLink H (RAt H (deploy rel c) lx) (Mp lx) (Mc lx))
    (hOK : ∀ lx, (S.leaky.proto lx).TablesOK (Mp lx) (Mc lx)) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      Complete H (deploy rel c) ∧
      ZK H ((deploy rel c).mask S.leaky.G.Nonce S.leaky.G.Rand tapeOf
          ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)))
        (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
          2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  sampling_deployed rel c H hslots hprog hComplete tapeOf ((S.leaky.withTabSim Mc nS).simLeakE t (t * 3 ^ t)) _ _
    (zk_sampled_tables_mask rel c H S hR t ht tapeOf hP Mp Mc nL nB nS hnL hnB hnS hL hOK)

end SamplingDeploy
