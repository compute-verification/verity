import Proofs.Flock.Claims.ZK.FinishStopAt

/-!
# The replay's rows on a view's masked values, constants included

`Zk.replay`'s three constraints and the two ring-switch rows after them read only the masked proof, the transcript's
coins and the switched vectors. Where the proof's zerocheck, lincheck rounds and `z_partial` are a view's (`zcOf`,
`lcRoundsOf`, `zPartialOf`), the coins are the model's and the switched vectors are the view's `ŝ`, they are `rowsOf`'s
first five as whole forms (`InnerTauRep.tauRep` identifies their coefficients only, which is all `InnerTauVal` reads;
the firewall's stop, `rowsAt_of_cons`, reads their constants too).

* **`replay_rows`**: the five rows, whatever the Ligerito constraints after them.
-/

namespace FlockSoundness.Discharge.Composed
open Flock Model ZkSession ZkView ZK Refine
local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **The replay's five rows are `rowsOf`'s**: on a run of `Zk.replay` at `layOf` (at `m ≥ 14`) whose proof's masked
messages are the view's `v`, whose transcript's coins are `c`'s, whose fold folds as the model's (`combvOf`), and whose
switched vectors at the `ab` and `c` claims are `ŝ`'s, the replay's three constraints and the two ring-switch rows, as
the firewall and the session append them, are `rowsOf`'s first five, constants included. -/
theorem replay_rows {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (S : Flock.Setup) (p : Flock.Proof) (root : ByteArray) {t t' : Flock.Transcript} {rp : Zk.Replayed}
    (hm : 14 ≤ S.m) (hSm : S.m = st.m) (hSk : S.kLog = st.c.kLog)
    (h : Zk.replay S (layOf Tz st) p root t = .ok (rp, t'))
    (hzc : p.zerocheck = zcOf Tz st v) (hrd : p.lincheck.rounds = lcRoundsOf Tz st v)
    (hzp : p.lincheck.zPartial = zPartialOf Tz st v)
    (ez : ZerocheckMasked.zAt t = zcF Tz st c r) (er : pinned ++ ZerocheckMasked.rOuterAt t = rF Tz st c r)
    (eρ : ZerocheckMasked.rhoAt t (S.m - 6) = rhoF Tz st c r) (eρ' : Lincheck.rhoOf S.m t = rhoF Tz st c r)
    (eα : Lincheck.coinAt t (3 + (S.m - 6)) = c.rest.lcAlpha r)
    (eβ : Lincheck.coinAt t (4 + (S.m - 6)) = c.rest.lcBeta r)
    (eτ : Lincheck.tsOf S.m S.kLog t = tsF Tz st c r)
    (er' : Lincheck.coinAt t (5 + (S.m - 6) + (S.kLog - 6)) = c.rest.rPrime r)
    (hfold : ∀ combv, S.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz st c r))
      ((rhoF Tz st c r).extract 0 (st.c.kLog - 6)) (tsF Tz st c r) = .ok combv → combv = combvOf Tz st c r)
    (ss : Array (Array F128)) (hs0 : ss[0]! = shArr sh (2 * r.val)) (hs1 : ss[1]! = shArr sh (2 * r.val + 1))
    (lgc : Array (String × Zk.Lin)) (k : ℕ) (hk : k < 5) :
    (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
        ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
        ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz st c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩) := by
  obtain ⟨h0, h1, hc1, hcv⟩ := ZerocheckMasked.replay_zerocheck _ _ _ _ hm h
  obtain ⟨combv, hcomb, h2, habv, hc0⟩ := Lincheck.replay_lincheck _ _ _ _ hm h
  have h3 := ZkExec.replay_cons_size _ _ _ _ h
  rw [eα, eβ, show Lincheck.coinAt t 2 = ZerocheckMasked.zAt t from rfl, ez, eρ', eτ, hSk] at hcomb
  have hcv' := hfold combv hcomb
  rw [hzc, ez] at h0
  rw [hzc, ez, er, eρ, hSm] at h1
  rw [hzc, hrd, hzp, eα, eβ, eτ, hcv', hSk] at h2
  rw [hzp, er'] at habv
  rw [eρ', eτ, er', hSm, hSk] at hc0
  rw [ez, er] at hc1
  rw [hzc] at hcv
  have hsz : (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))]).size = 5 := by
    rw [Array.size_append, h3] <;> rfl
  rw [ZkExec.append_get!_left _ _ (lt_of_lt_of_eq hk hsz.symm)]
  simp only [rowsOf, Fin.append_left]
  interval_cases k
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h0]
    rfl
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h1]
    rfl
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h2]
    rfl
  · rw [ZkExec.append_get!_right _ _ (j := 0) (by rw [h3]) (by simp), habv, hc0, hs0]
    rfl
  · rw [ZkExec.append_get!_right _ _ (j := 1) (by rw [h3]) (by simp), hcv, hc1, hs1]
    rfl

end FlockSoundness.Discharge.Composed
