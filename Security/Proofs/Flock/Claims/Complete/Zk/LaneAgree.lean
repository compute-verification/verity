import Proofs.Flock.Claims.Complete.Zk.LigeritoPos
import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos

/-!
# Level 0's lane constraints agree

`lanes_agree`: where the verifier's masked Ligerito reads, from round `K`, rounds whose coins are the replies to the
sends at their index past `L` (as the prover's session records them, `Recs`), and the masked prover ran from `L + K`
sends on the same target, the two records give the same lane constraints (`Zk.ligeritoConstraints`) on any proof and
hiding messages: `Inner`'s record agreement, given where the two Ligeritos start. Level 0's out-of-domain value is the
one field the records may differ in, and the constraints do not read it (`ligeritoConstraints_ood`).
-/

namespace FlockComplete

open Flock Transcript Replay FlockSoundness.Discharge.ZkExec

/-- Level 0's lane constraints read the out-of-domain coin, not its value. -/
theorem ligeritoConstraints_ood (lay : Zk.PadLayout) (T c y₁ y₂ : F128) (chs : Array F256) (rho : F256 × F256)
    (lg : LigeritoProof) (msgs : Zk.Level0Msgs) :
    Zk.ligeritoConstraints lay ⟨T, #[(c, y₁)], chs, rho⟩ lg msgs =
      Zk.ligeritoConstraints lay ⟨T, #[(c, y₂)], chs, rho⟩ lg msgs := by
  unfold Zk.ligeritoConstraints
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.size_toArray, List.length_cons,
    List.length_nil]
  rfl

theorem coins_getD0 {r : Round} {m : Claims.Msg} (hc : r.coins = (Array.range r.n).map (coinOf m)) (hn : r.n = 1) :
    r.coins.getD 0 0 = coinOf m 0 := by
  rw [hc, hn]
  simp [Array.getD]

theorem coins_K {r : Round} {m : Claims.Msg} (hc : r.coins = (Array.range r.n).map (coinOf m)) (hn : r.n = 2) :
    coinK r.coins = coinKOf m := by
  rw [hc, hn]
  simp [coinK, coinKOf, Array.getD]

/-- **Level 0's lane constraints agree**: the verifier's masked Ligerito, run from round `K` of rounds whose `i`-th
coins are the reply to send `L + i`, and the masked prover, run from `L + K` sends on the same target, give records with
the same lane constraints. -/
theorem lanes_agree (resp : ℕ → Claims.Msg) (hsV : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zv : Zk.Level0)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {R : Array Round} {K L : ℕ}
    (hR : ∀ i < R.size, (R[i]!).coins = (Array.range (R[i]!).n).map (coinOf (resp (L + i)))) {rcV : Zk.L0Rec}
    {tV : Transcript} (hV : Zk.ligeritoWith hsV ms sch T cap p basisAt zv ⟨R, K, ByteArray.empty⟩ = .ok (rcV, tV))
    (hsP : Hashes PS) (ms' : MerkleScheme) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    {tP tP' : Transcript} {lP lP' : Array Claims.Msg} (hl : lP.size = L + K) {P : LigeritoProof} {zl : Zk.Level0Msgs}
    {rcP : Zk.L0Rec}
    (hP : ((ligeritoPZk (asksS resp) hsP ms' sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run tP).go lP =
      (.ok ((P, zl, rcP), tP'), lP'))
    (lay' : Zk.PadLayout) (lg : LigeritoProof) (msgs : Zk.Level0Msgs) :
    Zk.ligeritoConstraints lay' rcV lg msgs = Zk.ligeritoConstraints lay' rcP lg msgs := by
  obtain ⟨hTV, hoV, hszV, hchV, hρV, -, hn1, hn2, hnρ₀, hnρ₁⟩ :=
    zkLigeritoWith_pos hsV ms sch T cap p basisAt zv hn hk0 h2 hq hV
  obtain ⟨hTP, ⟨y, hoP⟩, hszP, hchP, hρP⟩ :=
    ligeritoPZk_pos resp hsP ms' sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0 hP
  dsimp only at hoV hchV hρV hn1 hn2 hnρ₀ hnρ₁
  have hc : ∀ i, (R[i]!).n ≠ 0 → (R[i]!).coins = (Array.range (R[i]!).n).map (coinOf (resp (L + i))) := by
    intro i h
    refine hR i ?_
    by_contra hge
    rw [getElem!_neg R i hge] at h
    exact h rfl
  have hch : rcV.challenges = rcP.challenges := by
    refine Array.ext (by rw [hszV, hszP]) fun j h₁ h₂ => ?_
    have hj : j < sch.k0 := hszV ▸ h₁
    rw [← getElem!_pos rcV.challenges j h₁, ← getElem!_pos rcP.challenges j h₂, hchV j hj, hchP j hj,
      coins_K (hc _ (by rw [hn2 j hj]; omega)) (hn2 j hj), show L + (K + 2 + j) = lP.size + 2 + j by omega]
  have hρ : rcV.rho = rcP.rho := by
    rw [hρV, hρP, coins_K (hc _ (by rw [hnρ₀]; omega)) hnρ₀, coins_K (hc _ (by rw [hnρ₁]; omega)) hnρ₁,
      show L + (K + 2 + sch.k0) = lP.size + 2 + sch.k0 by omega,
      show L + (K + 3 + sch.k0) = lP.size + 3 + sch.k0 by omega]
  have hb : (R[K + 1]!).coins.getD 0 0 = coinOf (resp (lP.size + 1)) 0 := by
    rw [coins_getD0 (hc _ (by rw [hn1]; omega)) hn1, show L + (K + 1) = lP.size + 1 by omega]
  obtain ⟨TV, oV, cV, ρV⟩ := rcV
  obtain ⟨TP, oP, cP, ρP⟩ := rcP
  dsimp only at hTV hoV hTP hoP hch hρ
  subst hTV hTP hoV hoP hch hρ
  rw [hb]
  exact ligeritoConstraints_ood lay' _ _ _ _ _ _ lg msgs

end FlockComplete
