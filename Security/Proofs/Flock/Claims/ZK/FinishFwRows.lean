import Proofs.Flock.Claims.ZK.FinishStopAt
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinishSwRows

/-!
# A worker's rows from the firewall's own run

`FinishStopAt.rowsAt_of_cons` decides a table's rows at its own draw. Here they are decided at the firewall's run on a
rep's request (`repFwC`, answered by `asksCut`): where it goes on, its constraints are the first seven rows on its masked
values `vals + h` (`tauFw_vals`) and none fails on the request's pads `h`, and where that holds at both reps, at masked
values that are the worker's clear values plus `h`, the worker's rows hold (`rowsAt_of_fw`).

* `LaneVals`: a Ligerito proof's out-of-domain value, lane messages and pair's sums are the view's masked values
  (`LaneAt`'s value fields).
* **`tauFw_vals`**: `tauFw`, with level 0 read off the returned proof: its record's target is the opening's `T` and its
  coins the model's (`hrec`, at the run's steps), its values are the view's (`LaneVals`) and its restart is `T'`.
* **`rowsAt_of_fw`**: where, at each rep, constraints on the layout's length that are the first seven rows on the
  clear values plus pads `h` (`h_ab = h_a·h_b`, `h_w = h_x·h_y`) don't fail on `h`, the rows hold.
-/

namespace Claims.ZKMal.Masked

open FlockSoundness.Refine (fOf gOf)
open FlockSoundness (execArith)

/-- **Level 0's values read at the view**: its out-of-domain value, lane messages and pair's sums are the view's masked
values (`LaneAt`'s value fields). -/
structure LaneVals {mPts : ℕ} (Tz : FlockSoundness.Discharge.ZkSession.TabZK FlockLevel3.GF128 mPts) (st : _root_.Flock.Stmt) {nE : ℕ}
    (v : Fin nE → FlockLevel3.GF128) (lg : _root_.Flock.LigeritoProof) (zl : _root_.Flock.Zk.Level0Msgs) : Prop where
  oodVal : fOf lg.oodValues[0]! = FlockSoundness.Discharge.Composed.msgAt v
    ((FlockSoundness.Discharge.Composed.layOf Tz st).lgOod 0)
  msg : ∀ j < Tz.sch.k0,
    gOf lg.messages[j]!.1 = execArith.ofLimbs
      (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgMsg j 0))
      (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgMsg j 1)) ∧
    gOf lg.messages[j]!.2 = execArith.ofLimbs
      (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgMsg j 2))
      (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgMsg j 3))
  e0 : gOf zl.e0 = execArith.ofLimbs
    (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgE 0 0))
    (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgE 0 1))
  e1 : gOf zl.e1 = execArith.ofLimbs
    (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgE 1 0))
    (FlockSoundness.Discharge.Composed.msgAt v ((FlockSoundness.Discharge.Composed.layOf Tz st).lgE 1 1))

end Claims.ZKMal.Masked

namespace FlockSoundness.Discharge.Composed
open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **The firewall's seven rows, level 0 read off its proof**: `tauFw`'s, where, at the run's steps, level 0's record's
target is the opening's `T` and its coins are the model's (`hrec`), and the returned proof's level 0 values are the
view's (`LaneVals`) and its restart is `T'`. -/
theorem tauFw_vals {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (stm : Stmt) (c : CoinsZK (RestOf Tz stm)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ : 𝔽 × 𝔽)
    (coin : ℕ → ℕ → Flock.F128) (n : ℕ) (st : Flock.Setup) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array Flock.Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array Flock.F128) (extra lanePad : ℕ → ℕ → Flock.F128)
    (salt : ℕ → ℕ → ByteArray) {t₀ : Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array _root_.Claims.Msg}
    (hm : 14 ≤ st.m) (hk6 : 6 ≤ st.kLog) (hSm : st.m = stm.m) (hSk : st.kLog = stm.c.kLog)
    (hkL : st.fold.kLog = st.kLog) (hlay : layOf Tz stm = Zk.PadLayout.new st.m st.kLog sch.k0)
    (hmk : ∀ k < (layOf Tz stm).zp 64, (msgAt v k).val = vals[k]! + pads.h[k]!)
    (hz : coin (l.size + 2) 0 = zcF Tz stm c r)
    (hr : pinned ++ (Array.range (st.m - 13)).map (coin (l.size + 1)) = rF Tz stm c r)
    (hρ : rhoF Tz stm c r = FlockComplete.Replay.pushes (fun j => coin (l.size + (3 + j)) 0) (st.m - 6))
    (hα : coin (l.size + (3 + (st.m - 6))) 0 = c.rest.lcAlpha r)
    (hβ : coin (l.size + (4 + (st.m - 6))) 0 = c.rest.lcBeta r)
    (hτ : tsF Tz stm c r =
      FlockComplete.Replay.pushes (fun j => coin (l.size + (5 + (st.m - 6) + j)) 0) (st.kLog - 6))
    (hr' : coin (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = c.rest.rPrime r)
    (hfold : ∀ combv, st.fold.folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz stm c r))
      ((rhoF Tz stm c r).extract 0 (stm.c.kLog - 6)) (tsF Tz stm c r) = .ok combv → combv = combvOf Tz stm c r)
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = shArr sh (2 * r.val) ∧ ss[1]! = shArr sh (2 * r.val + 1))
    (hlen : (layOf Tz stm).len = nE + 4)
    (hrec : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut coin n)
          (fun k => Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash)
          Flock.MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt (fun _ _ => 0)).run t₃).go l₃ =
        (.ok ((lg, zl, rec), t₄), l₄) →
      fOf rec.target = openT Tz stm c r sh ∧ _root_.Claims.ZKMal.Masked.LaneCoins Tz stm c r rec)
    {p : Zk.ZkProof} {t : Flock.Transcript} {l' : Array _root_.Claims.Msg}
    (h : ((Flock.Firewall.Finish.repFwC (FlockComplete.asksCut coin n) (Flock.Hashes.ofSha512 Flock.Sha512.hash) st sch
      tPad rep dom g cm0 extras pads ic vals extra lanePad salt).run t₀).go l = (.ok (p, t), l'))
    (hV : _root_.Claims.ZKMal.Masked.LaneVals Tz stm v p.proof.ligerito p.level0) (hTP : gOf p.level0.tPrime = T') :
    ∃ (rp : Zk.Replayed) (ss : Array (Array Flock.F128)) (lgc : Array (String × Zk.Lin)),
      Flock.Firewall.Finish.failingOn (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
          ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) pads.h = #[] ∧
      (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc).size = 7 ∧
      (∀ k < 7, ((rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!).2.a.size =
          (layOf Tz stm).len) ∧
      ∀ k (hk : k < 7), (ofZk (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
          ss[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)[k]!.2 :
          ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz stm c r v sh T' ρσ (Fin.castAdd 3 ⟨k, by omega⟩) := by
  dsimp only [Flock.Firewall.Finish.repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := Flock.Firewall.Finish.run_bind_ok h
  rw [Flock.Firewall.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := Flock.Firewall.Finish.run_bind_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := Flock.Firewall.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := Flock.Firewall.Finish.run_bind_ok hr₆
  obtain ⟨hlgc, ht₇, hl₇⟩ := Flock.Firewall.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  split at hc
  · obtain ⟨_, _, _, h', _⟩ := Flock.Firewall.Finish.run_bind_ok hc
    rw [Flock.Firewall.Finish.throw_run_go] at h'
    cases h'
  rename_i hf
  have hpl : p.proof.ligerito = lg ∧ p.level0 = zl := by
    first
    | (obtain ⟨_, _, _, -, hp⟩ := Flock.Firewall.Finish.run_bind_ok hc
       simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
       obtain ⟨⟨rfl, -⟩, -⟩ := hp
       exact ⟨rfl, rfl⟩)
    | (obtain ⟨_, _, _, -, hc'⟩ := Flock.Firewall.Finish.run_bind_ok hc
       obtain ⟨_, _, _, -, hp⟩ := Flock.Firewall.Finish.run_bind_ok hc'
       simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
       obtain ⟨⟨rfl, -⟩, -⟩ := hp
       exact ⟨rfl, rfl⟩)
  rw [hpl.1, hpl.2] at hV
  rw [hpl.2] at hTP
  have hrp' : Zk.replay st (layOf Tz stm)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := lg } ic.root (Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [hlay]
    exact hrp
  have hlgc' : Zk.ligeritoConstraints (layOf Tz stm) rec lg zl = .ok lgc := by
    rw [hlay]
    exact hlgc
  obtain ⟨hT, hLC⟩ := hrec zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ h₁ h₂ h₃ h₄
  obtain ⟨hsz2, hlA⟩ := _root_.Claims.ZKMal.Masked.lanesA Tz stm c r v sh T' hlen lg rec zl hLC.ood hLC.lane hLC.rho
    hlgc'
  obtain ⟨hc7, hasz⟩ := fwCons_asz hm hrp' ss (lgc := lgc) ⟨hsz2, fun k => by rw [(hlA k).1, hlen]⟩
  refine ⟨rp, ss, lgc, by simpa using hf, hc7, hasz, fun k hk => ?_⟩
  by_cases hk5 : k < 5
  · exact tauFw_at_replay Tz stm c r v sh T' ρσ coin n st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt
      ht₀ hm hk6 hSm hSk hkL hlay hmk hz hr hρ hα hβ hτ hr' hfold hsw h₁ h₂ h₃ h₄ hrp lgc k hk5
  obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 5 + k'.val := ⟨⟨k - 5, by omega⟩, by simp only; omega⟩
  have hL : _root_.Claims.ZKMal.Masked.LaneAt Tz stm c r v sh T' lg rec zl :=
    ⟨hT, hLC.ood, hLC.lane, hLC.rho, hV.oodVal, hV.msg, hV.e0, hV.e1, hTP⟩
  obtain ⟨-, hrow⟩ := _root_.Claims.ZKMal.Masked.lanesAt Tz stm c r v sh T' hlen lg rec zl hL hlgc'
  have h3 := ZkExec.replay_cons_size _ _ _ _ hrp'
  rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [Array.size_append, h3]; rfl) (by rw [hsz2]; exact k'.isLt),
    hrow k']
  fin_cases k' <;> rfl

variable (P : LigProver) {mPts : ℕ}

/-- **Where the firewall goes on at masked values that are a worker's, its rows hold**: where, at each rep, constraints
on the layout's length are the first seven rows on the worker's clear values plus pads `h` of the layout's length (with
`h_ab = h_a·h_b` and `h_w = h_x·h_y`), and none fails on `h`, the worker's rows hold at the mask slot `u` and the extra
lanes `R` (`noFail_iff_rows`, `rowsAt_of_rows`). -/
theorem rowsAt_of_fw (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (u : (C.T c).tab.onVals.Mask) (R : (C.T c).tab.onVals.XLanes),
      (∀ r : Fin 2, ∃ (h : Array Flock.F128) (cons : Array (String × Zk.Lin)),
        h.size = (layOf Tz st).len ∧
        h[(layOf Tz st).hab]! = h[(layOf Tz st).fa]! * h[(layOf Tz st).fb]! ∧
        h[(layOf Tz st).tw]! = h[(layOf Tz st).tx]! * h[(layOf Tz st).ty]! ∧
        cons.size = 7 ∧ (∀ q ∈ cons, q.2.a.size = (layOf Tz st).len) ∧
        (∀ k : Fin 7, (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 ((C.T c).tab.onVals.nE + 4)) =
          rowsOf Tz st c r (fun i => (C.T c).tab.onVals.vals r p u R i + fOf h[i.val]!)
            ((C.T c).tab.onVals.nu p + (C.T c).tab.onVals.maskHom u)
            (∑ col, (C.T c).tab.onVals.y1' p u R r col * bFold Tz st c r col.val) (0, 0) (Fin.castAdd 3 k)) ∧
        Flock.Firewall.Finish.failingOn cons h = #[]) →
      (C.T c).tab.onVals.RowsAt p u R := by
  intro C c p u R hall
  refine rowsAt_of_rows P Tz st c p u R fun r => ?_
  obtain ⟨h, cons, hsz, hab, htw, hcs, hlen, hform, hf⟩ := hall r
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  have hs := len_eq Tz st hE
  obtain ⟨hfa, hfb, htx, hty, htw'⟩ := lay_triple _ _ hE
  have hix : ∀ k, k < (C.T c).tab.onVals.nE + 4 → (ix (C.T c).tab.onVals.nE k).val = k :=
    fun k hk => Nat.mod_eq_of_lt hk
  have harr : arrOf (padsAt hs fun i => fOf h[i.val]!) = h := by
    rw [arrOf_padsAt]
    refine Array.ext (by rw [arrOf, Array.size_ofFn, hsz, hs]) fun i h1 h2 => ?_
    simp only [arrOf, Array.getElem_ofFn, Fin.val_mk]
    rw [getElem!_pos h i h2]
    rfl
  rw [← harr] at hf
  refine (noFail_iff_rows Tz st c r _ _ (fun i => fOf h[i.val]!) hE (fun i => rfl) ?_ ?_ _ _ hs cons hcs hlen
    hform).1 hf
  · show fOf h[(ix _ (layOf Tz st).hab).val]! = fOf h[(ix _ (layOf Tz st).fa).val]! * fOf h[(ix _ (layOf Tz st).fb).val]!
    rw [hix (layOf Tz st).hab (by omega), hix (layOf Tz st).fa (by omega), hix (layOf Tz st).fb (by omega), hab,
      fOf_mul]
  · show fOf h[(ix _ (layOf Tz st).tw).val]! = fOf h[(ix _ (layOf Tz st).tx).val]! * fOf h[(ix _ (layOf Tz st).ty).val]!
    rw [hix (layOf Tz st).tw (by omega), hix (layOf Tz st).tx (by omega), hix (layOf Tz st).ty (by omega), htw,
      fOf_mul]

end FlockSoundness.Discharge.Composed
