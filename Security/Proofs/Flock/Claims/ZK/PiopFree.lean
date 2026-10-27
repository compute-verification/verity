import Proofs.Flock.Claims.ZK.L0PinCoins
import Proofs.Flock.Claims.ZK.L0PinSwitch
import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinishSwRows

/-!
# The cut PIOP returns on its coins alone

Answered by `asksCut`, the cut PIOP throws only where a send reaches the cut's stop `n`, where the lincheck's circuit
fold at its coins fails, or where the rank check at its rounds' points fails. Its absorbs never throw, its sends' count
reads only the claims' count, the fold reads only the zerocheck's coins, and the rounds' points are the coins'. So
whether it returns, and the log it leaves, read the coins, the log's size, the setup and the extra claims' count: never
the masked values, the switched vectors, the layout, the cap or the root.

* **`FlockComplete.piopCut_of_phases`**: where zerocheck, lincheck and the ring switch returned from `l` at some masked
  values, switched vectors and extra claims, the cut PIOP from any source, layout, cap, root and transcript returns from
  a log of `l`'s size at extra claims of the same count, with a log of the same size, wherever its rank check passes.
* **`FlockComplete.piopCut_src_free`**: two cut PIOPs from a log of one size and transcripts with no rounds: where one
  returns, so does the other at any source, layout, cap, root and extra claims of the same count, with a log of the same
  size; at the same extra claims, with the same switched claims (`piopCut_sws_eq`).
* **`Masked.piopCut_of_repFwC`**: at a live statement whose opened coins are admitted, where the firewall's rep `r` of
  table `j` returned from rep `2j + r`'s base, the cut PIOP from any source returns there, its rank check at the earlier
  reps' points and the mask slot's words, with no premise on the masked values.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete

open Flock Transcript Replay

/-- **The cut PIOP returns where its phases did at other values**: where zerocheck, lincheck and the ring switch
returned from `l` at the masked values `mk`, switched vectors `sv` and extra claims `extras`, the cut PIOP from any
source, layout, cap, root and transcript returns from a log of `l`'s size at extra claims of `extras`'s count, wherever
its rank check passes, with a log of the ring switch's size. -/
theorem piopCut_of_phases {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} (hm : 14 ≤ st.m) {lay : Zk.PadLayout}
    {mk : ℕ → F128} {cap root : ByteArray} {sv : ℕ → Array F128} {extras : Array Claim} {t₀ : Transcript}
    {l : Array Claims.Msg} {zp : ZerocheckProof} {zc : ZcOut} {t₁ : Transcript} {l₁ : Array Claims.Msg}
    {lp : LincheckProof} {lc : LcOut} {t₂ : Transcript} {l₂ : Array Claims.Msg}
    {r : Array (Array F128) × F128 × Array Switched} {t₃ : Transcript} {l₃ : Array Claims.Msg}
    (h₁ : ((zerocheckCut (asksCut coin n) st.m st.digest cap root lay mk).run t₀).go l = (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((lincheckCut (asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂))
    (h₃ : ((ringSwitchCut (asksCut coin n) sv (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ =
      (.ok (r, t₃), l₃))
    (lay' : Zk.PadLayout) (src : RepSrc) (cap' root' : ByteArray) {extras' : Array Claim}
    (hes : extras'.size = extras.size) (prior : Array (List F128)) (words : List ℕ) (t₀' : Transcript)
    {l' : Array Claims.Msg} (hl : l'.size = l.size)
    (hrk : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest cap' root' lay' src.masked).run t₀').go l' =
          (.ok ((zp, zc), t₁), l₁) →
        ((lincheckCut (asksCut coin n) st.fold st.m lay' src.masked zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true) :
    ∃ res t₃' l₃', ((piopCut (asksCut coin n) st lay' src cap' root' extras' prior words).run t₀').go l' =
      (.ok (res, t₃'), l₃') ∧ l₃'.size = l₃.size := by
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨-, hl₂⟩ := lincheckCut_at coin n _ _ _ _ _ h₂
  obtain ⟨w, hw⟩ := RankAt.lincheckCut_fold coin n h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size]
  obtain ⟨-, hl₃, hn₃⟩ := ringSwitchCut_sws coin n _ _ (by rw [hsz]; omega) h₃
  rw [hsz] at hl₃
  obtain ⟨zc', t₁', l₁', h₁'⟩ :=
    Claims.ZKServer.zerocheckCut_proof coin n hm st.digest cap' root' lay' src.masked t₀' l' (by omega)
  obtain ⟨hz', hl₁'⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁'
  rw [hl] at hz' hl₁'
  have ez : zc'.z = zc.z := hz'.1.trans hz.1.symm
  have er : zc'.rho = zc.rho := hz'.2.1.trans hz.2.1.symm
  have el₁ : l₁'.size = l₁.size := by omega
  obtain ⟨lc', t₂', l₂', h₂'⟩ :=
    Claims.ZKServer.lincheckCut_proof coin n st.fold st.m lay' src.masked zc' t₁' l₁' (by omega)
      ⟨w, by rw [el₁, ez, er]; exact hw⟩
  obtain ⟨-, hl₂'⟩ := lincheckCut_at coin n _ _ _ _ _ h₂'
  have hrk' := hrk _ _ _ _ _ _ _ _ h₁' h₂'
  have hsz' : (abcClaims st.m st.kLog zc' lc' ++ extras').size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size, hes]
  obtain ⟨⟨ss, T, sws⟩, t₃', l₃', h₃', -, -⟩ :=
    Claims.ZKServer.Switch.ringSwitchCut_bodies coin n src.sv _ (by rw [hsz']; omega) t₂' l₂' (by rw [hsz']; omega)
  obtain ⟨-, hl₃', -⟩ := ringSwitchCut_sws coin n _ _ (by rw [hsz']; omega) h₃'
  rw [hsz'] at hl₃'
  unfold piopCut
  rw [Replay.bind_go h₁']
  try dsimp only
  rw [Replay.bind_go h₂']
  try dsimp only
  rw [Replay.bind_go (Flock.Firewall.Finish.rankCheck_pass_go l₂' hrk')]
  try dsimp only
  rw [Replay.bind_go h₃']
  exact ⟨_, t₃', l₃', rfl, by omega⟩

/-- **A returned cut PIOP ran its four phases and passed its rank check** at the lincheck's rounds. -/
theorem piopCut_split_rank {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {lay : Zk.PadLayout} {src : RepSrc}
    {cap root : ByteArray} {extras : Array Claim} {prior : Array (List F128)} {words : List ℕ} {t₀ : Transcript}
    {l : Array Claims.Msg} {zp lp ss T sws t₃ l₃}
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) :
    ∃ zc t₁ l₁ lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest cap root lay src.masked).run t₀).go l = (.ok ((zp, zc), t₁), l₁) ∧
        ((lincheckCut (asksCut coin n) st.fold st.m lay src.masked zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) ∧
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true ∧
        ((ringSwitchCut (asksCut coin n) src.sv (abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ =
          (.ok ((ss, T, sws), t₃), l₃) := by
  unfold piopCut at h
  obtain ⟨⟨zp', zc⟩, t₁, l₁, h₁, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp', lc⟩, t₂, l₂, h₂, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨u, t₂', l₂', hr, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  have hok := rankCheckP_ok hr
  obtain ⟨ht, hl⟩ := rankCheckP_go hr
  rw [ht, hl] at h
  try dsimp only at h
  obtain ⟨⟨ss', T', sws'⟩, t₃', l₃', h₃, h⟩ := Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  rw [pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩, rfl⟩ := h
  exact ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, hok, h₃⟩

/-- **The cut PIOP's return reads no masked value**: answered by `asksCut`, from transcripts with no rounds and logs of
one size, where the cut PIOP from `src` returned, so does the one from any `src'`, layout, cap, root and extra claims of
the same count, at the same earlier points and words, with a log of the same size. -/
theorem piopCut_src_free {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} (hm : 14 ≤ st.m) (hk : 6 ≤ st.kLog)
    (hfk : st.fold.kLog = st.kLog) {lay : Zk.PadLayout} {src : RepSrc} {cap root : ByteArray} {extras : Array Claim}
    {prior : Array (List F128)} {words : List ℕ} {t₀ : Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Claims.Msg}
    {zp lp ss T sws t₃ l₃}
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃))
    (lay' : Zk.PadLayout) (src' : RepSrc) (cap' root' : ByteArray) {extras' : Array Claim}
    (hes : extras'.size = extras.size) {t₀' : Transcript} (ht₀' : t₀'.rounds = #[]) {l' : Array Claims.Msg}
    (hl : l'.size = l.size) :
    ∃ res t₃' l₃', ((piopCut (asksCut coin n) st lay' src' cap' root' extras' prior words).run t₀').go l' =
      (.ok (res, t₃'), l₃') ∧ l₃'.size = l₃.size := by
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, hok, h₃⟩ := piopCut_split_rank h
  have hpts := RankAt.roundPoints_cut coin n hm hk st.digest cap root lay src.masked st.fold hfk ht₀ h₁ h₂
  refine piopCut_of_phases hm h₁ h₂ h₃ lay' src' cap' root' hes prior words t₀' hl
    fun _ _ _ _ _ _ _ _ h₁' h₂' => ?_
  rw [RankAt.roundPoints_cut coin n hm hk st.digest cap' root' lay' src'.masked st.fold hfk ht₀' h₁' h₂', hl,
    ← hpts]
  exact hok

/-- **Two returned cut PIOPs from logs of one size have the same switched claims** at the same extra claims. -/
theorem piopCut_sws_eq {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} (hm : 14 ≤ st.m) {lay lay' : Zk.PadLayout}
    {src src' : RepSrc} {cap root cap' root' : ByteArray} {extras : Array Claim} {prior prior' : Array (List F128)}
    {words words' : List ℕ} {t₀ t₀' : Transcript} {l l' : Array Claims.Msg} {zp lp ss T sws t₃ l₃}
    {zp' lp' ss' T' sws' t₃' l₃'}
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃))
    (h' : ((piopCut (asksCut coin n) st lay' src' cap' root' extras prior' words').run t₀').go l' =
      (.ok ((zp', lp', ss', T', sws'), t₃'), l₃'))
    (hl : l'.size = l.size) : sws' = sws := by
  rw [(piopCut_at hm h' default).1, (piopCut_at hm h default).1, hl]

end FlockComplete

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The cut PIOP from any source returns where the firewall's rep did**: at a live statement whose opened coins are
admitted, where the firewall's rep of table `j` returned from rep `2j + r`'s base, answered by the opened coins, the cut
PIOP at table `j`'s setup from any source, layout, cap and root, at extra claims of the rep's count, from a transcript
with no rounds, returns from that base, its rank check at the earlier reps' points and the mask slot's words. -/
theorem piopCut_of_repFwC {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) (n : ℕ)
    {st : _root_.Flock.Setup} (hsm : st.m = (stsOf x)[j]!.m) (hsk : st.kLog = (stsOf x)[j]!.c.kLog)
    (hfk : st.fold.kLog = st.kLog) {sch : _root_.Flock.Schedule} {tPad rep : ℕ} {dom : ByteArray}
    {g : Array _root_.Flock.F128} {cm0 : FlockComplete.Commit} {extras : Array _root_.Flock.Claim}
    {pads : FlockComplete.PadsZk} {ic : FlockComplete.InnerCommit} {vals : Array _root_.Flock.F128}
    {extra lanePad : ℕ → ℕ → _root_.Flock.F128} {salt : ℕ → ℕ → ByteArray} {t₀ : _root_.Flock.Transcript}
    {l : Array Msg} {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt).run t₀).go l = (.ok (p, t), l'))
    (hl : l.size = repBase x (2 * j + r)) (lay' : _root_.Flock.Zk.PadLayout) (src : FlockComplete.RepSrc)
    (cap' root' : ByteArray) {extras' : Array _root_.Flock.Claim} (hes : extras'.size = extras.size)
    {t₀' : _root_.Flock.Transcript} (ht₀' : t₀'.rounds = #[]) :
    ∃ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) n) st lay' src cap' root' extras'
      (priorE c j r) (Composed.maskWordsOf (stsOf x)[j]!)).run t₀').go l = (.ok (res, t₃), l₃) := by
  obtain ⟨hm, -⟩ := tab_bounds hx hj
  obtain ⟨_, _, _, _, _, _, _, hfw⟩ := _root_.Flock.Firewall.Finish.repFwC_piopFw h
  obtain ⟨_, _, _, _, _, _, h₁, h₂, h₃⟩ := _root_.Flock.Firewall.Finish.piopFw_split hfw
  obtain ⟨res, t₃, l₃, hrun, -⟩ := FlockComplete.piopCut_of_phases (by omega) h₁ h₂ h₃ lay' src cap' root' hes
    (priorE c j r) (Composed.maskWordsOf (stsOf x)[j]!) t₀' rfl fun _ _ _ _ _ _ _ _ h₁' h₂' =>
      rank_at_admits hx hA hj hr (ligCoin c) n (fun _ _ _ => rfl) st hsm hsk hfk _ _ _ _ ht₀' hl h₁' h₂'
  exact ⟨res, t₃, l₃, hrun⟩

end Masked
end Claims.ZKMal
