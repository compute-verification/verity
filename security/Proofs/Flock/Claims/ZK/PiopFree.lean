module

public import Proofs.Flock.Claims.ZK.FinishRep
public import Proofs.Flock.Claims.ZK.SimLig
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.FinishSwRep
import Proofs.Flock.Claims.ZK.FinishSwRows
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.L0PinCoins
import Proofs.Flock.Claims.ZK.L0PinSwitch

/-! The cut PIOP returns on its coins alone. -/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000  -- as in the ZK files this builds on

namespace FlockComplete

open Flock Transcript Replay

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
  rw [Replay.bind_go (Flock.ZkServer.Finish.rankCheck_pass_go l₂' hrk')]
  try dsimp only
  rw [Replay.bind_go h₃']
  exact ⟨_, t₃', l₃', rfl, by omega⟩

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
  obtain ⟨⟨zp', zc⟩, t₁, l₁, h₁, h⟩ := Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp', lc⟩, t₂, l₂, h₂, h⟩ := Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨u, t₂', l₂', hr, h⟩ := Flock.ZkServer.Finish.run_bind_ok h
  have hok := rankCheckP_ok hr
  obtain ⟨ht, hl⟩ := rankCheckP_go hr
  rw [ht, hl] at h
  try dsimp only at h
  obtain ⟨⟨ss', T', sws'⟩, t₃', l₃', h₃, h⟩ := Flock.ZkServer.Finish.run_bind_ok h
  try dsimp only at h
  rw [pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩, rfl⟩ := h
  exact ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, hok, h₃⟩

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

theorem piopCut_of_repFwC {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) (n : ℕ)
    {st : _root_.Flock.Setup} (hsm : st.m = (stsOf x)[j]!.m) (hsk : st.kLog = (stsOf x)[j]!.c.kLog)
    (hfk : st.fold.kLog = st.kLog) {sch : _root_.Flock.Schedule} {tPad rep : ℕ} {dom : ByteArray}
    {g : Array _root_.Flock.F128} {cm0 : FlockComplete.Commit} {extras : Array _root_.Flock.Claim}
    {pads : FlockComplete.PadsZk} {ic : FlockComplete.InnerCommit} {vals : Array _root_.Flock.F128}
    {extra lanePad : ℕ → ℕ → _root_.Flock.F128} {salt : ℕ → ℕ → ByteArray} {t₀ : _root_.Flock.Transcript}
    {l : Array Msg} {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} {l' : Array Msg}
    (h : ((_root_.Flock.ZkServer.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt).run t₀).go l = (.ok (p, t), l'))
    (hl : l.size = repBase x (2 * j + r)) (lay' : _root_.Flock.Zk.PadLayout) (src : FlockComplete.RepSrc)
    (cap' root' : ByteArray) {extras' : Array _root_.Flock.Claim} (hes : extras'.size = extras.size)
    {t₀' : _root_.Flock.Transcript} (ht₀' : t₀'.rounds = #[]) :
    ∃ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) n) st lay' src cap' root' extras'
      (priorE c j r) (Composed.maskWordsOf (stsOf x)[j]!)).run t₀').go l = (.ok (res, t₃), l₃) := by
  obtain ⟨hm, -⟩ := tab_bounds hx hj
  obtain ⟨_, _, _, _, _, _, _, hfw⟩ := _root_.Flock.ZkServer.Finish.repFwC_piopFw h
  obtain ⟨_, _, _, _, _, _, h₁, h₂, h₃⟩ := _root_.Flock.ZkServer.Finish.piopFw_split hfw
  obtain ⟨res, t₃, l₃, hrun, -⟩ := FlockComplete.piopCut_of_phases (by omega) h₁ h₂ h₃ lay' src cap' root' hes
    (priorE c j r) (Composed.maskWordsOf (stsOf x)[j]!) t₀' rfl fun _ _ _ _ _ _ _ _ h₁' h₂' =>
      rank_at_admits hx hA hj hr (ligCoin c) n (fun _ _ _ => rfl) st hsm hsk hfk _ _ _ _ ht₀' hl h₁' h₂'
  exact ⟨res, t₃, l₃, hrun⟩

end Masked
end Claims.ZKMal
