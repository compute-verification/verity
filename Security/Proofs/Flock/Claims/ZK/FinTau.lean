import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.SplitInner

/-!
# The session rep's `τ` is its view's (`RepTau`)

`repTau_of` derives `RepTau` from `InnerTauVal`.

The session's rep's inner proof is what the phases hybrids 2 and 3 share (`innerPre`) leave to `innerP`
(`repPZk_map_inner`). Answered by the opened coins (`asksCut`), an `innerP` that returns has answered `ε` and the
batching coins, and its `τ` is `InZ.tau` at the coins from the round it began (`innerP_cut_tau`). `innerPre_slot` puts
that round at the rep's `ε` slot. With no refusal at the opened coins (`not_refused_of_admits`), the run of the shared
phases is `InnerAt`, and `InnerTauVal` gives `τ` as the view's.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

namespace FinTau

open FlockComplete FlockComplete.SimInner

/-- A returning run of a bind returns at its first step and continues from there. -/
theorem runT_bind_ok {α β : Type} {x : _root_.Flock.TMW FlockComplete.PS α}
    {f : α → _root_.Flock.TMW FlockComplete.PS β} {t t'' : _root_.Flock.Transcript} {l l'' : Array Msg} {b : β}
    (h : ((x >>= f).run t).go l = (.ok (b, t''), l'')) :
    ∃ a t' l', (x.run t).go l = (.ok (a, t'), l') ∧ ((f a).run t').go l' = (.ok (b, t''), l'') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := JsonRT.go_bind_ok h
  exact ⟨a, t', l', h1, h2⟩

/-- A returning run returns its value's image under a map. -/
theorem run_map_ok {α β : Type} {m : _root_.Flock.TMW FlockComplete.PS α} (F : α → β)
    {s t : _root_.Flock.Transcript} {l l' : Array Msg} {p : α} (h : (m.run s).go l = (.ok (p, t), l')) :
    ((F <$> m).run s).go l = (.ok (F p, t), l') := by
  simp only [StateT.run_map, FlockComplete.go_map', h, Except.map]

/-- A returning run has the value of a field its every run has. -/
theorem field_of_map {α β : Type} {m : _root_.Flock.TMW FlockComplete.PS α} (F : α → β) {b : β}
    (hm : F <$> m = (fun _ => b) <$> m) {s t : _root_.Flock.Transcript} {l l' : Array Msg} {p : α}
    (h : (m.run s).go l = (.ok (p, t), l')) : F p = b := by
  have e := congrArg (fun z => ((z.run s).go l).1) hm
  simp only [StateT.run_map, FlockComplete.go_map', h, Except.map, Except.ok.injEq] at e
  exact (Prod.mk.inj e).1

/-- **The session's rep's inner proof is the shared phases' `innerP`**: `repPZk` runs `innerPre`'s phases, level 0's
opening unerased (`replay_eraseL`, `ligeritoConstraints_eraseL`), then `innerP` on their `final_c` and constraints. -/
theorem repPZk_map_inner (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (fun z => z.inner) <$> repPZk io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
        words =
      innerPre io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words >>= fun o =>
        innerP io hs (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) pads ic o.1 o.2.2.2 := by
  unfold repPZk innerPre piopPZk
  simp only [bind_assoc, pure_bind, map_bind, map_pure, bind_pure, FinRep.replay_eraseL,
    FinRep.ligeritoConstraints_eraseL]

/-- **An inner proof answered by the opened coins that returns** has had `ε` and the batching coins answered, and its
`τ` is `InZ.tau` at the coins from the round it began. -/
theorem innerP_cut_tau {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {h : ByteArray → ByteArray}
    {lay : _root_.Flock.Zk.PadLayout} {p : PadsZk} {ic : InnerCommit} {finalC : _root_.Flock.F128}
    {cons : Array (String × _root_.Flock.Zk.Lin)} {t t' : _root_.Flock.Transcript} {l l' : Array Msg}
    {ip : _root_.Flock.Zk.InnerProof}
    (e : ((innerP (asksCut coin n) (_root_.Flock.Hashes.ofSha512 h) lay p ic finalC cons).run t).go l =
      (.ok (ip, t'), l')) :
    l.size + 1 < n ∧ ip.tau = Replay.InZ.tau lay p cons (cutCoins coin l.size) := by
  unfold innerP at e
  dsimp only at e
  obtain ⟨_, t₁, l₁, h₁, e⟩ := runT_bind_ok e
  obtain rfl := liftTM_ok_log h₁
  obtain ⟨_, t₂, l₂, h₂, e⟩ := runT_bind_ok e
  obtain rfl := liftTM_ok_log h₂
  obtain ⟨eps, t₃, l₃, h₃, e⟩ := runT_bind_ok e
  obtain ⟨hn₀, rfl, rfl, rfl⟩ := ite_ok ((sampleCut_go coin n t₂ l).symm.trans h₃)
  obtain ⟨_, t₄, l₄, h₄, e⟩ := runT_bind_ok e
  obtain rfl := liftTM_ok_log h₄
  obtain ⟨gs, t₅, l₅, h₅, e⟩ := runT_bind_ok e
  obtain ⟨hn₁, rfl, rfl, rfl⟩ :=
    ite_ok ((sampleVecCut_go coin n (by simp [Flock.Zk.triple]) t₄ _).symm.trans h₅)
  simp only [Array.size_push] at hn₁
  refine ⟨by omega, field_of_map (fun z : _root_.Flock.Zk.InnerProof => z.tau) ?_ e⟩
  simp only [map_bind, map_pure, FinRep.map_ite', Replay.InZ.tau, Replay.InZ.lin, Replay.InZ.all, Replay.InZ.rho,
    Replay.InZ.sigma, cutCoins, Array.size_push]

end FinTau

/-- **`RepTau` from `InnerTauVal`**: the session's rep's inner proof is `innerP` after the shared phases
(`FinTau.repPZk_map_inner`), whose run at the opened coins is `InnerAt` (no refusal there,
`not_refused_of_admits`) and ends at the rep's `ε` slot (`innerPre_slot`); `innerP` returning there has answered the
batching coins, and its `τ` is `InZ.tau` at the slot's coins (`FinTau.innerP_cut_tau`), the view's by
`InnerTauVal`. -/
theorem repTau_of (hT : InnerTauVal) : RepTau := by
  intro x w τ c spec i hi r cm0 extras ic l hC q t l' e hr
  obtain ⟨hx, hw, hA, ⟨s, hsx⟩, -, -, -, hex, hl⟩ := hC
  unfold repHE at e
  have e1 := FinTau.run_map_ok (fun z => z.inner) e
  rw [FinTau.repPZk_map_inner] at e1
  obtain ⟨o, t₁, l₁, h₁, h₂⟩ := FinTau.runT_bind_ok e1
  have hslot := innerPre_slot hsx w τ c (refK x) ⟨i, hi⟩ ⟨r, hr⟩ (domE spec i r) cm0 ic extras hex hl h₁
  unfold ioE at h₂
  obtain ⟨hlt, htau⟩ := FinTau.innerP_cut_tau h₂
  have hAt : InnerAt x w τ c (refK x) ⟨i, hi⟩ ⟨r, hr⟩ o.2.2.2 l₁.size :=
    ⟨hx, hw, not_refused_of_admits hx hA (refK x), s, spec, hsx, domE spec i r, cm0, ic, extras, hex, l, hl, o, t₁,
      l₁, h₁, rfl, rfl⟩
  have hv := hT x w τ c (refK x) ⟨i, hi⟩ ⟨r, hr⟩ o.2.2.2 l₁.size hAt
    (by rw [innerSlotE, ← hslot]; exact hlt)
  rw [htau, hslot]
  exact hv

end Masked

end Claims.ZKMal
