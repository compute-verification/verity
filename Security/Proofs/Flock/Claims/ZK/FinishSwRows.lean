import Proofs.Flock.Claims.ZK.FinishSwTau
import Proofs.Flock.Claims.ZK.FinishTau

/-!
# The firewall's seven constraints as `SwRows`

`FinishSwTau.swSrcE_innerSrc_rows` asks that the firewall's constraints at `repCut_fw`'s `hinner` be the worker's
view's first seven rows (`SwRows`): seven constraints, each of the layout's length, with `rowsSwE`'s coefficients.
`FinishTau.tauFw` gives the coefficients as whole forms (`ofZk`), `FinishTau.tauFw_at_coef` by coefficient with level
0's lengths; the replay's lengths are its own.

* `Composed.replay_asz`: `Zk.replay`'s three constraints and its `ab` and `c` values have the layout's length.
* `Composed.fwCons_asz`: so, where level 0's two rows do too, the firewall's constraints (the replay's, the two
  ring-switch rows and level 0's) are seven, each of the layout's length.
* **`Masked.swRows_of`**: at those, constraints that are `rowsSwE`'s first seven as whole forms are `SwRows`.
* **`Masked.swRows_coef`**: the same by coefficient, level 0's lengths `nE + 4`, the inner layout's.
* `Finish.piopFw_split`: `hinner`'s PIOP run is the zerocheck, lincheck and ring-switch runs that `tauFw_at` reads.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.Discharge.Composed
open Flock Model ZkSession ZkView ZK Refine

/-- **The replay's rows have the layout's length**: `Zk.replay`'s three constraints, and its `ab` and `c` values. -/
theorem replay_asz {st : Setup} {lay : Zk.PadLayout} {p : Flock.Proof} {root : ByteArray} {t t' : Flock.Transcript}
    {rp : Zk.Replayed} (hm : 14 ≤ st.m) (h : Zk.replay st lay p root t = .ok (rp, t')) :
    rp.cons.size = 3 ∧ (rp.cons[0]!).2.a.size = lay.len ∧ (rp.cons[1]!).2.a.size = lay.len ∧
      (rp.cons[2]!).2.a.size = lay.len ∧ rp.abValue.a.size = lay.len ∧ rp.cValue.a.size = lay.len := by
  obtain ⟨h0, h1, -, hcv⟩ := ZerocheckMasked.replay_zerocheck _ _ _ _ hm h
  obtain ⟨_, -, h2, habv, -⟩ := Lincheck.replay_lincheck _ _ _ _ hm h
  refine ⟨ZkExec.replay_cons_size _ _ _ _ h, ?_, ?_, ?_, ?_, ?_⟩
  · rw [h0]
    exact size_rowCEval_form _ _ _
  · rw [h1]
    exact size_rowAB_form _ _ _ _ _ _
  · rw [h2]
    simp [Lincheck.lcFinal, FlockComplete.Replay.lcRunning_size]
  · rw [habv]
    simp [Lincheck.lcAbValue]
  · rw [hcv]
    simp [ZerocheckMasked.fcL, Zk.Lin.masked]

/-- **The firewall's constraints are seven, each of the layout's length**: the replay's three, the two ring-switch
rows on its `ab` and `c` values, and level 0's two, where those have the layout's length. -/
theorem fwCons_asz {st : Setup} {lay : Zk.PadLayout} {p : Flock.Proof} {root : ByteArray} {t t' : Flock.Transcript}
    {rp : Zk.Replayed} (hm : 14 ≤ st.m) (h : Zk.replay st lay p root t = .ok (rp, t')) (ss : Array (Array F128))
    {lgc : Array (String × Zk.Lin)} (hl : lgc.size = 2 ∧ ∀ k : Fin 2, (lgc[k.val]!).2.a.size = lay.len) :
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
        ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc).size = 7 ∧
      ∀ k < 7, ((rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
        ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!).2.a.size =
          lay.len := by
  obtain ⟨h3, h0, h1, h2, hab, hc⟩ := replay_asz hm h
  have hsz5 : (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))]).size = 5 := by
    rw [Array.size_append, h3]
    rfl
  refine ⟨by rw [Array.size_append, hsz5, hl.1], fun k hk => ?_⟩
  by_cases h5 : k < 5
  · rw [ZkExec.append_get!_left _ _ (by rw [hsz5]; exact h5)]
    by_cases h3' : k < 3
    · rw [ZkExec.append_get!_left _ _ (by rw [h3]; exact h3')]
      interval_cases k
      · exact h0
      · exact h1
      · exact h2
    · obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 3 + k'.val := ⟨⟨k - 3, by omega⟩, by simp only; omega⟩
      rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [h3]) (by first | exact k'.isLt | (simp; omega))]
      fin_cases k'
      · exact hab
      · exact hc
  · obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 5 + k'.val := ⟨⟨k - 5, by omega⟩, by simp only; omega⟩
    rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [hsz5]) (by rw [hl.1]; exact k'.isLt)]
    exact hl.2 k'

end FlockSoundness.Discharge.Composed

namespace Flock.Firewall.Finish
open Flock Transcript

/-- **The PIOP's run is its three runs**: where `piopFw` returned, so did its zerocheck, its lincheck and its ring
switch, in turn, as `tauFw_at` reads them. -/
theorem piopFw_split {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {lay : Zk.PadLayout} {g : Array F128}
    {mk : ℕ → F128} {cap root : ByteArray} {extras : Array Claim} {t₀ : Transcript} {l : Array Claims.Msg}
    {zp lp ss T sws t₃ l₃}
    (h : ((piopFw (FlockComplete.asksCut coin n) st lay g mk cap root extras).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) :
    ∃ zc t₁ l₁ lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest cap root lay mk).run t₀).go l =
          (.ok ((zp, zc), t₁), l₁) ∧
        ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ =
          (.ok ((lp, lc), t₂), l₂) ∧
        ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
            (fun k => switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
            (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) := by
  unfold piopFw at h
  obtain ⟨⟨zp', zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp', lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss', T', sws'⟩, t₃', l₃', h₃, h⟩ := run_bind_ok h
  try dsimp only at h
  rw [FlockComplete.Replay.pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩, rfl⟩ := h
  exact ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩

end Flock.Firewall.Finish

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)

namespace Masked

/-- **The firewall's constraints as `SwRows`**: where the replay returned (`hm`, `hrp`) and level 0's two rows have the
layout's length, constraints whose first seven rows are `rowsSwE`'s as whole forms (`ofZk`, as `tauFw` gives them)
are `SwRows`. -/
theorem swRows_of (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    {st : _root_.Flock.Setup} (hm : 14 ≤ st.m) {p : _root_.Flock.Proof} {root : ByteArray}
    {t t' : _root_.Flock.Transcript} {rp : _root_.Flock.Zk.Replayed}
    (hrp : _root_.Flock.Zk.replay st (innerLayE x j) p root t = .ok (rp, t')) (ss : Array (Array _root_.Flock.F128))
    {lgc : Array (String × _root_.Flock.Zk.Lin)}
    (hl : lgc.size = 2 ∧ ∀ k : Fin 2, (lgc[k.val]!).2.a.size = (innerLayE x j).len)
    (hrow : ∀ k (hk : k < 7),
      (Composed.ofZk ((rp.cons ++ #[("ring switch ab", rp.abValue.addConst
          (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)), ("ring switch c", rp.cValue.addConst
          (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!).2 :
        ZkSession.Lin FlockLevel3.GF128 ((tabC PE x j).z.nE + 4)) = rowsSwE x c w v τ j r (Fin.castAdd 3 ⟨k, hk⟩)) :
    SwRows x c w v τ j r (rp.cons ++ #[("ring switch ab", rp.abValue.addConst
      (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)), ("ring switch c", rp.cValue.addConst
      (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) := by
  obtain ⟨hsz, hasz⟩ := Composed.fwCons_asz hm hrp ss hl
  exact ⟨hsz, hasz, fun k i => congrArg (fun l => ZkSession.Lin.coef l i) (hrow k.val k.isLt)⟩

/-- **The firewall's constraints as `SwRows` by coefficient**: `swRows_of` where the first seven rows are `rowsSwE`'s
coefficient by coefficient (as `tauFw_at_coef` gives them), level 0's two of `nE + 4`, the inner layout's length. -/
theorem swRows_coef (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    {st : _root_.Flock.Setup} (hm : 14 ≤ st.m) {p : _root_.Flock.Proof} {root : ByteArray}
    {t t' : _root_.Flock.Transcript} {rp : _root_.Flock.Zk.Replayed}
    (hrp : _root_.Flock.Zk.replay st (innerLayE x j) p root t = .ok (rp, t')) (ss : Array (Array _root_.Flock.F128))
    {lgc : Array (String × _root_.Flock.Zk.Lin)}
    (hl : lgc.size = 2 ∧ ∀ k : Fin 2, (lgc[k.val]!).2.a.size = (tabC PE x j).z.nE + 4)
    (hrow : ∀ k (hk : k < 7) (i : Fin ((tabC PE x j).z.nE + 4)),
      (Composed.ofZk ((rp.cons ++ #[("ring switch ab", rp.abValue.addConst
          (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)), ("ring switch c", rp.cValue.addConst
          (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!).2 :
        ZkSession.Lin FlockLevel3.GF128 ((tabC PE x j).z.nE + 4)).coef i =
        (rowsSwE x c w v τ j r (Fin.castAdd 3 ⟨k, hk⟩)).coef i) :
    SwRows x c w v τ j r (rp.cons ++ #[("ring switch ab", rp.abValue.addConst
      (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)), ("ring switch c", rp.cValue.addConst
      (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) := by
  obtain ⟨hsz, hasz⟩ := Composed.fwCons_asz hm hrp ss ⟨hl.1, fun k => (hl.2 k).trans (innerLayE_len x j).symm⟩
  exact ⟨hsz, hasz, fun k i => hrow k.val k.isLt i⟩

end Masked
end Claims.ZKMal
