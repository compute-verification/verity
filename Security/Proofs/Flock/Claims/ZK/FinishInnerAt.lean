import Proofs.Flock.Claims.ZK.FinishLigEnd
import Proofs.Flock.Claims.ZK.L0PinCoins

/-!
# The rep from a source that misses the firewall at `final_c`, at its inner proof's first round

`repCut_fw_inner` stops the rep from a source that may differ from the firewall at `final_c`'s pad at the inner proof's
first round, after the firewall's PIOP and Ligerito logs. Here that stop is placed: Ligerito left no pending bytes
(`ligeritoCut_end`), and the history then holds `CutEq.innerAt` bodies, the PIOP's (`piopCut_at`) and the opening's
(`ligeritoCut_eqAt`).

* `piopCut_fw_fc`: `piopCut_fw` from a source that may differ at `fc`; its zerocheck proof's `finalC` is the source's.
* `repCut_fw_inner_at`: the rep from the source, cut at the inner proof's first round, sends that round's body there.
-/

namespace Flock.Firewall.Finish

open Flock Transcript
open FlockComplete (PS)

/-- **`piopCut_fw` from a source that may differ at `fc`**: where the firewall's PIOP at `mk` returns, the PIOP from a
source whose masked values below `zp 64` but at `fc` are `mk`'s returns the same, its zerocheck proof's `finalC` the
source's `masked lay.fc`. -/
theorem piopCut_fw_fc (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (lay : Zk.PadLayout) (g : Array F128)
    (mk : ℕ → F128) (src : FlockComplete.RepSrc) (cap root : ByteArray) (extras : Array Claim)
    (prior : Array (List F128)) (words : List ℕ) (hm14 : 14 ≤ st.m) (hm : st.m - 6 ≤ lay.nMlv)
    (hk : st.fold.kLog - 6 ≤ lay.nLc) (hmk : ∀ i < lay.zp 64, i ≠ lay.fc → src.masked i = mk i) {t₀ : Transcript}
    {l : Array Claims.Msg} (hn : l.size ≤ n)
    (hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest cap root lay mk).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds) words =
          true ∧
        ∀ k < (abcClaims st.m st.kLog zc lc ++ extras).size,
          src.sv k = switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    {zp : ZerocheckProof} {lp : LincheckProof} {ss : Array (Array F128)} {T : F128} {sws : Array Switched}
    {t₃ : Transcript} {l₃ : Array Claims.Msg}
    (h : ((piopFw (FlockComplete.asksCut coin n) st lay g mk cap root extras).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) :
    ((FlockComplete.piopCut (FlockComplete.asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok (({ zp with finalC := src.masked lay.fc }, lp, ss, T, sws), t₃), l₃) := by
  have hPC' := piopCut_fw coin n st lay g mk
    { src with masked := fun i => if i = lay.fc then mk lay.fc else src.masked i }
    cap root extras prior words hm14 hm hk
    (fun i hi => by
      try dsimp only
      split_ifs with he
      · rw [he]
      · exact hmk i hi he)
    hn hpiop h
  exact piopCut_fc_go _ st lay src _ cap root extras prior words hm
    (fun i hi => by simp only [if_neg hi]) hPC'

set_option maxHeartbeats 1000000 in
/-- **The rep from a source that misses the firewall at `fc`, at its inner proof's first round**: at `repCut_fw_inner`'s
premises and a schedule the rep's walk takes (`CutEq.SchOk`), the rep from the source, answered by `asksCut` from `l`
with the history cut after `CutEq.innerAt l.size` bodies, stops past its Ligerito opening with no pending bytes and sends
the inner proof's first body there, at the source's `masked lay.fc`. -/
theorem repCut_fw_inner_at (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit)
    (extras : Array Claim) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (src : FlockComplete.RepSrc)
    (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript) (l : Array Claims.Msg)
    (hs : FlockComplete.CutEq.SchOk st sch) (hn : l.size ≤ n) (hkL : st.fold.kLog = st.kLog)
    (hmk : ∀ i < (Zk.PadLayout.new st.m st.kLog sch.k0).zp 64, i ≠ (Zk.PadLayout.new st.m st.kLog sch.k0).fc →
      src.masked i = vals[i]! + pads.h[i]!)
    (hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds) words =
          true ∧
        ∀ k < (abcClaims st.m st.kLog zc lc ++ extras).size,
          src.sv k = switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    (hlig : ∀ zp lp ss T sws t₃ l₃,
      ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
          (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      FlockComplete.LigLock.LigHyp coin n l₃.size st.merkle sch g (FlockComplete.basisVals sws (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt src)
    {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ (t₄ : Transcript) (L : Array Claims.Msg), t₄.pending = ByteArray.empty ∧
      L.size = FlockComplete.CutEq.innerAt l.size st sch extras ∧
      ((FlockComplete.repCut (FlockComplete.asksCut coin L.size) (Hashes.ofSha512 Sha512.hash) st sch rep dom
          (FlockComplete.eraseC cm0) extras ic src prior words).run t₀).go l =
        (.error "past the openings", L.push (Claims.toMsg
          (framed (FlockComplete.Replay.InZ.t₁ (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) t₄) 1 1))) := by
  obtain ⟨zp, lp, ss, T, sws, t₃, l₃, lg, zl, rec, t₄, l₄, hPC, hL, -, hR⟩ :=
    repCut_fw_inner coin n st hm sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt src prior words t₀ l
      hs.m hn hkL hmk hpiop hlig h
  obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopCut_at hs.m hPC default
  have hl₄ := ((FlockComplete.CutEq.ligeritoCut_eqAt coin n st.merkle sch T (FlockComplete.eraseC cm0)
      (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src src hs.n hs.k0 hs.levels
      hs.q t₃ (l := l₃) rfl hn₃ (fun _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl)
      (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ hL).1
  refine ⟨t₄, l₄, ligeritoCut_end coin n st.merkle sch T _ _ _ src t₃ l₃ _ t₄ l₄ hL, ?_, hR⟩
  rw [hl₄, hl₃]
  simp only [FlockComplete.CutEq.innerAt, FlockComplete.CutEq.ligAt]
  omega

end Flock.Firewall.Finish
