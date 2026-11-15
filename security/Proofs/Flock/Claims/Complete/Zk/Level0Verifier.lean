module

public import Proofs.Flock.Claims.Complete.Zk.Level0Opens
public import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Claims.Complete.Json.Session

/-!
# Level 0's verifier premise: the masked prover's run against the masked model

`ZkTables.Level0.Verifier` asks that wherever the masked prover's Ligerito (`ligeritoPZk`) returns, the masked verifier's
(`Zk.ligeritoWith`) returns its record at its final position. `ligeritoWithZ_of` gives the verifier's return on the
masked model's rounds (`outZ`) from the proof's lengths, its cap, the pair's coins' independence at the model's, its
Merkle openings and the model's final identity; `honestZ` puts the prover's final transcript at the model's and gives
the final identity. This file holds what links the prover's run to the rest of `ligeritoWithZ_of`'s premises:

* `ligeritoPZk_initialCap`: the proof's level-0 cap is the commitment's.
* `preZ_rho`, `preZ_rhoIndependent`: the model's pair coins at the replies (`cfOf resp L`) are the prover's
  (`ZkTables.Level0.laneRho`), so they are independent wherever the prover returns (`ligeritoPZk_exact`).
-/

@[expose] public section

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg

/-- **The masked prover's Ligerito proof carries level 0's commitment's cap.** -/
theorem ligeritoPZk_initialCap (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {t t' : Transcript}
    {l l' : Array Claims.Msg} {P : LigeritoProof} {zl : Zk.Level0Msgs} {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) hs ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    P.initialCap = cm0.cap := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y0⟩, stF⟩, t₁⟩, l₁, -, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP3, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP, -⟩ := Prod.mk.inj hP3
  subst hP
  rfl

/-- **The masked model's pair coins at the replies are the prover's**: at the coins `cfOf resp L` from a transcript at
position `k`, the lane phase's pair `ρ` is `laneRho` at `L + k` sends. -/
theorem preZ_rho (resp : ℕ → Claims.Msg) (L : ℕ) (P : LigeritoProof) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (zl : Zk.Level0) (t : Transcript) (hk0 : 1 ≤ sch.k0) :
    (preZ P (cfOf resp L) sch T cap zl t).1.2 = ZkTables.Level0.laneRho resp sch.k0 (L + t.k) := by
  have hk : (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2.k = t.k + 2 := by
    simp only [head_fields, startZ_k]
  show (laneRun P (cfOf resp L) sch.k0 zl.msgs (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).1
    (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2 sch.k0).1.2 = _
  rw [laneRun_rho _ _ _ _ _ _ hk0, hk]
  show ((⟨coinOf (resp (L + (t.k + 2 + sch.k0))) 0, coinOf (resp (L + (t.k + 2 + sch.k0))) 1⟩ : F256),
      (⟨coinOf (resp (L + (t.k + 2 + sch.k0 + 1))) 0, coinOf (resp (L + (t.k + 2 + sch.k0 + 1))) 1⟩ : F256)) =
    (coinKOf (resp (L + t.k + 2 + sch.k0)), coinKOf (resp (L + t.k + 3 + sch.k0)))
  rw [show L + (t.k + 2 + sch.k0) = L + t.k + 2 + sch.k0 by omega,
    show L + (t.k + 2 + sch.k0 + 1) = L + t.k + 3 + sch.k0 by omega]
  rfl

/-- **The masked model's pair coins are independent wherever the masked prover's Ligerito returns**: run from `L + k`
bodies sent (`k` its position), the prover's pair coins are independent (`ligeritoPZk_exact`), and they are the model's
at the replies (`preZ_rho`). -/
theorem preZ_rhoIndependent (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {L : ℕ} {t t' : Transcript}
    {l l' : Array Claims.Msg} {r : LigeritoProof × Zk.Level0Msgs × Zk.L0Rec} (hl : l.size = L + t.k)
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok (r, t'), l'))
    (P : LigeritoProof) (cap : Array ByteArray) (zl : Zk.Level0) :
    Zk.rhoIndependent (preZ P (cfOf resp L) sch T cap zl t).1.2 = true := by
  rw [preZ_rho resp L P sch T cap zl t hk0, ← hl]
  exact (ligeritoPZk_exact resp h ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0 hL hq hrun).2

end FlockComplete.L0
