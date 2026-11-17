module

public import Proofs.Flock.ZkAny.RelaySlot

/-! A frame round as the loop ran it (`RoundRun`), over the frames' pure updates, read into `RelaySlot`'s premises (`envOk_round_run`, `stopped_round_run`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec)
open _root_.Flock.ZkServer.Session (State Placed byte?)
open _root_.Flock.ZkServer.Frame (Masker)
open _root_.Flock.ZkServer.Frames (Sessions Rep)

theorem Kept.trans {sid i r : Nat} {R R' R'' : Sessions} (h₁ : Kept sid i r R R') (h₂ : Kept sid i r R' R'') :
    Kept sid i r R R'' := by
  induction h₂ with
  | refl => exact h₁
  | step _ hs ih => exact .step ih hs

theorem SessKept.trans {sid : Nat} {R R' R'' : Sessions} (h₁ : SessKept sid R R') (h₂ : SessKept sid R' R'') :
    SessKept sid R R'' := by
  induction h₂ with
  | refl => exact h₁
  | step _ hs ih => exact .step ih hs

def Between (sid : Nat) (R R' : Sessions) : Prop := (∀ i r, Kept sid i r R R') ∧ SessKept sid R R'

def streamDom (st : State) (stream : String) : ByteArray :=
  ((st.domains.find? (·.1 == stream)).map (·.2)).getD .empty

def streamRep (st : State) (stream : String) : Nat × Nat :=
  ((st.reps.find? (·.1 == stream)).map (·.2)).getD (0, 0)

structure RoundRun (sid : Nat) (st : State) (pl : Placed) (stream : String) (n i r : Nat) (dom content : ByteArray)
    (e : Env) (R : Sessions) (L0 : _root_.Flock.ZkServer.Level0.Slots) (R₃ : Sessions) : Prop where
  rep_eq : streamRep st stream = (i, r)
  dom_eq : streamDom st stream = dom
  round : st.request pl (.round stream n) = none →
    e.round = (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).out
  coins : ∀ {o : ByteArray} {p₁ : Rep} {a : ByteArray}, st.request pl (.round stream n) = none → e.round = .ok o →
    (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).put = some p₁ → e.ans = some a →
    byte? a 0 = some 2 → (st.answer (.round stream n) a).1.stop = none → e.coins = .ok () →
    ∃ R₁ R₂ R₂' p', Between sid R R₁ ∧ Between sid (R₁.put p₁) R₂ ∧
      _root_.Flock.ZkServer.Frames.coinsStep R₂ sid i r (_root_.Flock.ZkServer.Frames.feedOf n a.data.toList) =
        .ok p' ∧ Between sid R₂ R₂' ∧ Between sid (R₂'.put p') R₃

theorem find_of_frameAt {R : Sessions} {sid i r : Nat} {p : Rep} (h : frameAt R sid i r = some p) :
    ∃ j, R.find sid i r = some (j, p) := by
  have hm := find_frameAt R sid i r
  rw [h] at hm
  rcases hfd : R.find sid i r with _ | ⟨j, q⟩
  · rw [hfd] at hm
    cases hm
  · rw [hfd] at hm
    cases hm
    exact ⟨j, rfl⟩

theorem coinsStep_inv {R : Sessions} {sid i r j : Nat} {p p' : Rep} {cs : Array F128}
    (hf : R.find sid i r = some (j, p)) (h : _root_.Flock.ZkServer.Frames.coinsStep R sid i r cs = .ok p') :
    ∃ st₂, _root_.Flock.ZkServer.Frame.step p.cfg p.st (.coins (.ok cs)) = (.ok, st₂) ∧
      p' = Loop.coinsOf p st₂ cs := by
  unfold _root_.Flock.ZkServer.Frames.coinsStep at h
  simp only [hf] at h
  split at h
  · cases h
    exact ⟨_, by assumption, rfl⟩
  all_goals cases h

theorem roundRun_chain {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    (hfr : (frameAt R sid i r).isSome) {o : ByteArray} {p₁ : Rep}
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) :
    ∃ R₁ R₂ cs st₂, Kept sid i r R R₁ ∧ SessKept sid R R₁ ∧ Kept sid i r (R₁.put p₁) R₂ ∧
      SessKept sid (R₁.put p₁) R₂ ∧ _root_.Flock.ZkServer.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂) ∧
      SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃ := by
  obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ :=
    hr.coins hreq ((hr.round hreq).trans ho) hput ha hb hs hc
  have hfa : frameAt R₂ sid i r = some p₁ := frameAt_own hfr (roundStep_key hput) (h01.1 i r) (h12.1 i r)
  obtain ⟨j, hj⟩ := find_of_frameAt hfa
  obtain ⟨st₂, hst, rfl⟩ := coinsStep_inv hj hcs
  exact ⟨R₁, R₂', _, st₂, h01.1 i r, h01.2, (h12.1 i r).trans (h22.1 i r), h12.2.trans h22.2, hst, h23.2⟩

theorem envOk_round_run {W Rr : Type} (L : LeanRelease W Rr) (nN : Msg → Nat) (wk : L.toSrv.Worker) (x : Msg)
    (w : W) (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hn : h.length + 1 < nN x) (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {sid : Nat} {st : State} {pl : Placed} {stream : String}
    {i r : Nat} {dom : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots}
    {R₃ : Sessions} (hr : RoundRun sid st pl stream (L.coins x h.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (L.coins x h.length)) = none) {q : Rep} {f : Bool} {nd : Option String}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h) {spec : SessionSpec} {seen : List String} {k : Nat}
    (hσ : SetupOk spec st) (hsent : st.sent = true) (hf : f = !seen.contains stream) :
    EnvOk spec seen (.round stream k) e (L.toSrv.nextD L.fillOf nN wk x w ρ κ h) :=
  envOk_round_next L nN wk x w ρ κ h hk hn hlt hv hcr hm hfr hrp hnd hcfg hu hst (hr.round hreq) hf
    (hr.dom_eq.symm.trans (dom_setupOk hσ hsent stream))

theorem stopped_round_run {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {sid : Nat} {st : State} {pl : Placed} {stream : String}
    {i r : Nat} {dom : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots}
    {R₃ : Sessions} (hr : RoundRun sid st pl stream (L.coins x h.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (L.coins x h.length)) = none) {q p₁ : Rep} {f : Bool}
    {nd : Option String} {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h)
    (ho : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out = .ok o)
    (hput : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).put =
      some p₁)
    {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (L.coins x h.length)) a).1.stop = none) (hc : e.coins = .ok ())
    (e' : Msg × Msg) : R₃.stopped sid = !L.toSrv.liveBefore wk x w ρ κ (h ++ [e']) := by
  obtain ⟨R₁, R₂, cs, st₂, hK₀₁, hS₀₁, hK₁₂, hS₁₂, hcs, hS₂₃⟩ := roundRun_chain hr hreq hfr ho hput ha hb hs hc
  exact stopped_round_next L wk x w ρ κ h hk hlt hv hcr hm hfr hrp hnd hcfg hu hst ho hput hK₀₁ hS₀₁ hK₁₂ hS₁₂ hcs
    hS₂₃ e'

end Claims.ZKServer.Relay
