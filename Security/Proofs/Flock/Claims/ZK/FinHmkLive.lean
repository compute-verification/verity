import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinPads
import Proofs.Flock.Claims.ZK.L0PinCoins

/-!
# `hmk` from the proofs: a rep's masked values below `zp 64` are the request's plus its pads

`tauFw_at`'s `hmk` asks that below `zp 64` the view's masked values be the firewall's request's values plus its pads.
The zerocheck and lincheck send every masked value below `zp 64` and return them as their proofs (`zerocheckCut_proof`,
`lincheckCut_proof`): `round_1_ab`, `round_1_c`, the residual rounds, `final_a`, `final_b`, `final_c`, the lincheck's
rounds and `z_partial`. `eraseZ` keeps both proofs. So where the firewall's rep at `vals` and `pads` (`repFwC`) and a cut
rep from a source (`repCut`) both return, at proofs equal up to `eraseZ`, the source's masked values below `zp 64` are
`vals + pads.h`, whatever answers either run.

* `zcProofOf`, `lcProofOf`: the zerocheck's and lincheck's proofs at the masked values `mk`.
* `mk_eq_of_proofs`: the two proofs at `mk'` and at `mk` agree only where `mk'` and `mk` agree below `zp 64`.
* `piop_proofs`: a zerocheck, lincheck and ring switch answered by `asksCut` that returned have those proofs.
* `repFwC_proofs`, `repCut_proofs`: so do a returned firewall's rep and a returned cut rep.
* **`masked_of_eraseZ`**: the source's masked values below `zp 64` are `vals + pads.h`.
-/

namespace Claims.ZKServer.HmkLive

open _root_.Flock (F128 Transcript TMW Claim)
open FlockComplete (PS asksCut zerocheckCut lincheckCut ringSwitchCut RunPost)
open FlockComplete.Replay (pushes pushes_get)

/-- **The zerocheck's proof at the masked values `mk`**: `mk` at `round_1_ab`'s, `round_1_c`'s, the residual rounds' and
the finals' indices. -/
def zcProofOf (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) : _root_.Flock.ZerocheckProof :=
  ⟨(Array.range 64).map fun i => mk (lay.r1ab i), (Array.range 64).map fun i => mk (lay.r1c i),
    pushes (fun j => (mk (lay.ml j 0), mk (lay.ml j 1))) lay.nMlv, mk lay.fa, mk lay.fb, mk lay.fc⟩

/-- **The lincheck's proof at the masked values `mk`**: `mk` at the rounds' and `z_partial`'s indices. -/
def lcProofOf (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) : _root_.Flock.LincheckProof :=
  ⟨pushes (fun j => (mk (lay.lc j 0), mk (lay.lc j 1))) lay.nLc, (Array.range 64).map fun i => mk (lay.zp i)⟩

theorem map_range_eq {f g : ℕ → F128} (h : (Array.range 64).map f = (Array.range 64).map g) {j : ℕ} (hj : j < 64) :
    f j = g j := by
  have e := congrArg (fun a : Array F128 => a[j]?) h
  simpa [Array.getElem?_map, Array.getElem?_range, hj] using e

theorem pushes_eq {f g : ℕ → F128 × F128} {N : ℕ} (h : pushes f N = pushes g N) {j : ℕ} (hj : j < N) : f j = g j := by
  have e : (pushes f N)[j]! = (pushes g N)[j]! := congrArg (fun a : Array (F128 × F128) => a[j]!) h
  rwa [pushes_get f N j hj, pushes_get g N j hj] at e

/-- **The zerocheck's and lincheck's proofs fix the masked values below `zp 64`**: every index below it is one of
`round_1_ab`'s, `round_1_c`'s, a residual round's, a final's, a lincheck round's or `z_partial`'s. -/
theorem mk_eq_of_proofs {lay : _root_.Flock.Zk.PadLayout} {mk mk' : ℕ → F128}
    (hz : zcProofOf lay mk' = zcProofOf lay mk) (hl : lcProofOf lay mk' = lcProofOf lay mk) :
    ∀ i < lay.zp 64, mk' i = mk i := by
  intro i hi
  simp only [zcProofOf, _root_.Flock.ZerocheckProof.mk.injEq] at hz
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hz
  simp only [lcProofOf, _root_.Flock.LincheckProof.mk.injEq] at hl
  obtain ⟨h7, h8⟩ := hl
  have hz64 : lay.zp 64 = 128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc + 0 + 64 := rfl
  rw [hz64] at hi
  by_cases a1 : i < 64
  · exact map_range_eq h1 a1
  by_cases a2 : i < 128
  · have e := map_range_eq h2 (j := i - 64) (by omega)
    have ei : lay.r1c (i - 64) = i := by
      show 64 + (i - 64) = i
      omega
    rwa [ei] at e
  by_cases a3 : i < 128 + 2 * lay.nMlv
  · have e := pushes_eq h3 (j := (i - 128) / 2) (by omega)
    simp only [Prod.mk.injEq] at e
    rcases Nat.mod_two_eq_zero_or_one (i - 128) with hp | hp
    · have ei : lay.ml ((i - 128) / 2) 0 = i := by
        show 128 + 2 * ((i - 128) / 2) + 0 = i
        omega
      rw [ei] at e
      exact e.1
    · have ei : lay.ml ((i - 128) / 2) 1 = i := by
        show 128 + 2 * ((i - 128) / 2) + 1 = i
        omega
      rw [ei] at e
      exact e.2
  by_cases a4 : i = 128 + 2 * lay.nMlv
  · have ei : lay.fa = i := by
      show 128 + 2 * lay.nMlv = i
      omega
    rwa [ei] at h4
  by_cases a5 : i = 128 + 2 * lay.nMlv + 1
  · have ei : lay.fb = i := by
      show 128 + 2 * lay.nMlv + 1 = i
      omega
    rwa [ei] at h5
  by_cases a6 : i = 128 + 2 * lay.nMlv + 2
  · have ei : lay.fc = i := by
      show 128 + 2 * lay.nMlv + 2 = i
      omega
    rwa [ei] at h6
  by_cases a7 : i < 128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc
  · have e := pushes_eq h7 (j := (i - (128 + 2 * lay.nMlv + 3)) / 2) (by omega)
    simp only [Prod.mk.injEq] at e
    rcases Nat.mod_two_eq_zero_or_one (i - (128 + 2 * lay.nMlv + 3)) with hp | hp
    · have ei : lay.lc ((i - (128 + 2 * lay.nMlv + 3)) / 2) 0 = i := by
        show 128 + 2 * lay.nMlv + 2 + 1 + 2 * ((i - (128 + 2 * lay.nMlv + 3)) / 2) + 0 = i
        omega
      rw [ei] at e
      exact e.1
    · have ei : lay.lc ((i - (128 + 2 * lay.nMlv + 3)) / 2) 1 = i := by
        show 128 + 2 * lay.nMlv + 2 + 1 + 2 * ((i - (128 + 2 * lay.nMlv + 3)) / 2) + 1 = i
        omega
      rw [ei] at e
      exact e.2
  · have e := map_range_eq h8 (j := i - (128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc)) (by omega)
    have ei : lay.zp (i - (128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc)) = i := by
      show 128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc + 0 + (i - (128 + 2 * lay.nMlv + 2 + 1 + 2 * lay.nLc)) = i
      omega
    rwa [ei] at e

theorem zcProofOf_k0 (m kLog k0 k0' : ℕ) (mk : ℕ → F128) :
    zcProofOf (_root_.Flock.Zk.PadLayout.new m kLog k0') mk = zcProofOf (_root_.Flock.Zk.PadLayout.new m kLog k0) mk :=
  rfl

theorem lcProofOf_k0 (m kLog k0 k0' : ℕ) (mk : ℕ → F128) :
    lcProofOf (_root_.Flock.Zk.PadLayout.new m kLog k0') mk = lcProofOf (_root_.Flock.Zk.PadLayout.new m kLog k0) mk :=
  rfl

theorem fst_go {α : Type} {m : TMW PS α} {s : Transcript} {l : Array Claims.Msg} {a : α} {t : Transcript}
    (h : ((m.run s).go l).1 = .ok (a, t)) : (m.run s).go l = (.ok (a, t), ((m.run s).go l).2) :=
  Prod.ext h rfl

theorem runPost_apply {α : Type} {m : TMW PS α} {P : α → Transcript → Array Claims.Msg → Prop} (hm : RunPost m P)
    {s : Transcript} {l : Array Claims.Msg} {a : α} {t : Transcript} {l' : Array Claims.Msg}
    (e : (m.run s).go l = (.ok (a, t), l')) : P a t l' :=
  hm s l a t l' e

/-- **A returned PIOP's zerocheck and lincheck proofs are its masked values**: answered by `asksCut`, a zerocheck and
lincheck at `mk` followed by a ring switch with a claim, all returned, sent their bodies before the stop, so they
returned `zcProofOf` and `lcProofOf` at `mk`. -/
theorem piop_proofs (coin : ℕ → ℕ → F128) (n : ℕ) {st : _root_.Flock.Setup} (hm : 14 ≤ st.m)
    (hkL : st.fold.kLog = st.kLog) (k0 : ℕ) {cap root : ByteArray} {mk : ℕ → F128} {sv : ℕ → Array F128}
    {claims : Array Claim} (hc : claims.size ≠ 0) {t₀ t₁ t₂ t₃ : Transcript} {l l₁ l₂ l₃ : Array Claims.Msg}
    {zp : _root_.Flock.ZerocheckProof} {zc : _root_.Flock.ZcOut} {lp : _root_.Flock.LincheckProof}
    {lc : _root_.Flock.LcOut} {ss : Array (Array F128)} {T : F128} {sws : Array _root_.Flock.Switched}
    (h₁ : ((zerocheckCut (asksCut coin n) st.m st.digest cap root (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0)
      mk).run t₀).go l = (.ok ((zp, zc), t₁), l₁))
    (h₂ : ((lincheckCut (asksCut coin n) st.fold st.m (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) mk zc).run
      t₁).go l₁ = (.ok ((lp, lc), t₂), l₂))
    (h₃ : ((ringSwitchCut (asksCut coin n) sv claims).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃)) :
    zp = zcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) mk ∧
      lp = lcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) mk := by
  obtain ⟨-, hl₁⟩ := FlockComplete.zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨-, hl₂⟩ := FlockComplete.lincheckCut_at coin n _ _ _ _ _ h₂
  obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.ringSwitchCut_sws coin n _ _ hc h₃
  obtain ⟨zc', t', L', e⟩ := Claims.ZKServer.zerocheckCut_proof coin n hm st.digest cap root
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) mk t₀ l (by omega)
  rw [h₁] at e
  simp only [Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨lc', t'', L'', e'⟩ := Claims.ZKServer.lincheckCut_proof coin n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) mk zc t₁ l₁ (by omega)
    (FlockComplete.RankAt.lincheckCut_fold coin n h₂)
  rw [h₂] at e'
  simp only [Prod.mk.injEq, Except.ok.injEq] at e'
  have elp := e'.1.1.1
  rw [hkL] at elp
  exact ⟨e.1.1.1, elp⟩

/-- **A returned firewall's rep has the PIOP proofs of its masked values `vals + pads.h`.** -/
theorem repFwC_proofs (coin : ℕ → ℕ → F128) (n : ℕ) (hs : _root_.Flock.Hashes PS) {st : _root_.Flock.Setup}
    (hm : 14 ≤ st.m) (hkL : st.fold.kLog = st.kLog) {sch : _root_.Flock.Schedule} {tPad rep : ℕ} {dom : ByteArray}
    {g : Array F128} {cm0 : FlockComplete.Commit} {extras : Array Claim} {pads : FlockComplete.PadsZk}
    {ic : FlockComplete.InnerCommit} {vals : Array F128} {extra lanePad : ℕ → ℕ → F128} {salt : ℕ → ℕ → ByteArray}
    {t₀ : Transcript} {l : Array Claims.Msg} {p : _root_.Flock.Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (asksCut coin n) hs st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    p.proof.zerocheck = zcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) ∧
      p.proof.lincheck =
        lcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) := by
  unfold _root_.Flock.Firewall.Finish.repFwC at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  have hp : p.proof.zerocheck = zp ∧ p.proof.lincheck = lp := by
    refine runPost_apply (P := fun q _ _ => q.proof.zerocheck = zp ∧ q.proof.lincheck = lp) ?_ h
    refine RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ => ?_
    dsimp only
    refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
      exact RunPost.bind_pure fun _ _ _ _ _ _ => ⟨rfl, rfl⟩
  rw [hp.1, hp.2]
  exact piop_proofs coin n hm hkL sch.k0
    (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) h₁ h₂ h₃

/-- **A returned cut rep has the PIOP proofs of its source's masked values.** -/
theorem repCut_proofs (coin : ℕ → ℕ → F128) (n : ℕ) (hs : _root_.Flock.Hashes PS) {st : _root_.Flock.Setup}
    (hm : 14 ≤ st.m) (hkL : st.fold.kLog = st.kLog) {sch : _root_.Flock.Schedule} {rep : ℕ} {dom : ByteArray}
    {cm0 : FlockComplete.Commit} {extras : Array Claim} {ic : FlockComplete.InnerCommit} {src : FlockComplete.RepSrc}
    {prior : Array (List F128)} {words : List ℕ} {t₀ : Transcript} {l : Array Claims.Msg}
    {p : _root_.Flock.Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((FlockComplete.repCut (asksCut coin n) hs st sch rep dom cm0 extras ic src prior words).run t₀).go l =
      (.ok (p, t), l')) :
    p.proof.zerocheck = zcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) src.masked ∧
      p.proof.lincheck = lcProofOf (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) src.masked := by
  unfold FlockComplete.repCut at h
  obtain ⟨⟨zp, lp, ss, T, sws⟩, t₃, l₃, h₃, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  have hp : p.proof.zerocheck = zp ∧ p.proof.lincheck = lp := by
    refine runPost_apply (P := fun q _ _ => q.proof.zerocheck = zp ∧ q.proof.lincheck = lp) ?_ h
    exact RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ =>
      RunPost.bind_pure fun _ _ _ _ _ _ => ⟨rfl, rfl⟩
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃'⟩ := FlockComplete.piopCut_split h₃
  rw [hp.1, hp.2]
  exact piop_proofs coin n hm hkL sch.k0
    (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) h₁ h₂ h₃'

/-- **The source's masked values below `zp 64` are the firewall's `vals + pads.h`**: where the firewall's rep at `vals`
and `pads` and a cut rep from `src`, at one setup and any schedules (`k₀` places nothing below `zp 64`), both return,
each answered by `asksCut` at any coins and stop, at proofs equal up to `eraseZ`. -/
theorem masked_of_eraseZ (coin coin' : ℕ → ℕ → F128) (n n' : ℕ) (hs hs' : _root_.Flock.Hashes PS)
    {st st' : _root_.Flock.Setup} (hst : st' = st) (hm : 14 ≤ st.m) (hkL : st.fold.kLog = st.kLog)
    {sch sch' : _root_.Flock.Schedule} {tPad rep rep' : ℕ} {dom dom' : ByteArray} {g : Array F128}
    {cm0 cm0' : FlockComplete.Commit} {extras extras' : Array Claim} {pads : FlockComplete.PadsZk}
    {ic ic' : FlockComplete.InnerCommit} {vals : Array F128} {extra lanePad : ℕ → ℕ → F128}
    {salt : ℕ → ℕ → ByteArray} {src : FlockComplete.RepSrc} {prior : Array (List F128)} {words : List ℕ}
    {t₀ t₀' : Transcript} {l l₀ : Array Claims.Msg} {p p' : _root_.Flock.Zk.ZkProof} {t t' : Transcript}
    {l₀' : Array Claims.Msg}
    (hF : (((_root_.Flock.Firewall.Finish.repFwC (asksCut coin n) hs st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt).run t₀).go l).1 = .ok (p, t))
    (hC : ((FlockComplete.repCut (asksCut coin' n') hs' st' sch' rep' dom' cm0' extras' ic' src prior words).run
      t₀').go l₀ = (.ok (p', t'), l₀'))
    (he : FlockComplete.eraseZ p' = FlockComplete.eraseZ p) :
    ∀ i < (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).zp 64, src.masked i = vals[i]! + pads.h[i]! := by
  subst hst
  obtain ⟨hF1, hF2⟩ := repFwC_proofs coin n hs hm hkL (fst_go hF)
  obtain ⟨hC1, hC2⟩ := repCut_proofs coin' n' hs' hm hkL hC
  rw [zcProofOf_k0 _ _ sch.k0] at hC1
  rw [lcProofOf_k0 _ _ sch.k0] at hC2
  have hz : p'.proof.zerocheck = p.proof.zerocheck := congrArg (fun q => q.proof.zerocheck) he
  have hl : p'.proof.lincheck = p.proof.lincheck := congrArg (fun q => q.proof.lincheck) he
  rw [hC1, hF1] at hz
  rw [hC2, hF2] at hl
  exact mk_eq_of_proofs hz hl

end Claims.ZKServer.HmkLive
