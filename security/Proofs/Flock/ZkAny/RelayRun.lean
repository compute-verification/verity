module

public import Proofs.Flock.ZkAny.RelaySlot

/-!
# A frame round as the loop ran it

`Session.relay` steps a `Round` through its rep's frame (`Frames.round`) and, where the auditor answered coins to a live
session, has the frame take them (`Frames.coins`). Each reads the loop's frames and puts a frame back, and the loop's
other tasks may update the frames between that read and that put. Here that run is stated over the frames' pure updates
(`RegStep`'s, as `RelayReg` reads the code) and read into `RelaySlot`'s premises:

* `Kept.trans`, `SessKept.trans`: runs of updates that keep a rep's frame, or a session's stop, compose. `Between`:
  updates that keep each of a session's frames and its stop, as all of them between two of its requests do.
* `streamDom`, `streamRep`: the domain frame and the rep the relay reads for a stream's round from the session.
* **`RoundRun`**: the round as the loop ran it. Where the session lets it through (`State.request`), what the relay read
  for it is `roundStep`'s release on the frames and level 0s `Frames.round` read, at the stream's rep and domain frame;
  where the auditor then answered coins that the rep's frame took (`Frames.coins`, `Frames.feedOf`), the round's put,
  the coins' read and their put each come after `Between` updates, and so does the session's next round's read.
* `find_of_frameAt`, `coinsStep_inv`: the coins' read finds the frame the round put, and takes the coins as
  `Loop.coinsOf` puts them. `roundRun_chain`: so a `RoundRun` gives `stopped_next`'s run of updates.
* **`envOk_round_run`**, **`stopped_round_run`**: `envOk_round_next` and `stopped_round_next` at a `RoundRun`, the
  round's content the worker's message, its domain frame the setup's.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frame (Masker)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **A run of updates that keep a rep's frame, then another, keeps it.** -/
theorem Kept.trans {sid i r : Nat} {R R' R'' : Sessions} (h₁ : Kept sid i r R R') (h₂ : Kept sid i r R' R'') :
    Kept sid i r R R'' := by
  induction h₂ with
  | refl => exact h₁
  | step _ hs ih => exact .step ih hs

/-- **A run of updates that keep a session's stop, then another, keeps it.** -/
theorem SessKept.trans {sid : Nat} {R R' R'' : Sessions} (h₁ : SessKept sid R R') (h₂ : SessKept sid R' R'') :
    SessKept sid R R'' := by
  induction h₂ with
  | refl => exact h₁
  | step _ hs ih => exact .step ih hs

/-- **Updates that keep each of session `sid`'s frames and its stop**: those between two of the session's requests,
since only its own task puts its frames, one request at a time (`RelayReg`). -/
def Between (sid : Nat) (R R' : Sessions) : Prop := (∀ i r, Kept sid i r R R') ∧ SessKept sid R R'

/-- **The domain frame the relay reads for a round of the stream `stream`** from the session `st` (`Session.relay`'s
`Round`). -/
def streamDom (st : State) (stream : String) : ByteArray :=
  ((st.domains.find? (·.1 == stream)).map (·.2)).getD .empty

/-- **The rep a round of the stream `stream` is stepped through**, its table and rep, as the relay reads it from the
session `st` (`Session.relay`'s `Round`). -/
def streamRep (st : State) (stream : String) : Nat × Nat :=
  ((st.reps.find? (·.1 == stream)).map (·.2)).getD (0, 0)

/-- **A frame round as the loop ran it**: session `sid`'s request `.round stream n` from the session `st` and placement
`pl`, its content `content`, on table `i`'s rep `r` and the domain frame `dom` the relay reads for the stream. Where the
session lets it through (`State.request`), `Frames.round` read the loop's frames `R` and level 0s `L0`, so what the
relay read for the round is `roundStep`'s release there. Where the auditor then answered coins `a` to a session still
going on and the rep's frame took them (`e.coins`), the round's put landed on frames updates between the session's
requests (`Between`) reached from `R`; `Frames.coins` read frames those reached from the put, and took the coins
`Frames.feedOf` reads off `a`; its put landed on frames those reached from what it read; and from that put they reach
`R₃`, the frames the session's next round reads. -/
structure RoundRun (sid : Nat) (st : State) (pl : Placed) (stream : String) (n i r : Nat) (dom content : ByteArray)
    (e : Env) (R : Sessions) (L0 : _root_.Flock.Firewall.Level0.Slots) (R₃ : Sessions) : Prop where
  rep_eq : streamRep st stream = (i, r)
  dom_eq : streamDom st stream = dom
  round : st.request pl (.round stream n) = none →
    e.round = (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out
  coins : ∀ {o : ByteArray} {p₁ : Rep} {a : ByteArray}, st.request pl (.round stream n) = none → e.round = .ok o →
    (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁ → e.ans = some a →
    byte? a 0 = some 2 → (st.answer (.round stream n) a).1.stop = none → e.coins = .ok () →
    ∃ R₁ R₂ R₂' p', Between sid R R₁ ∧ Between sid (R₁.put p₁) R₂ ∧
      _root_.Flock.Firewall.Frames.coinsStep R₂ sid i r (_root_.Flock.Firewall.Frames.feedOf n a.data.toList) =
        .ok p' ∧ Between sid R₂ R₂' ∧ Between sid (R₂'.put p') R₃

/-- **A rep's frame is one `Sessions.find` finds.** -/
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

/-- **The coins a rep's frame took**: where `Frames.coinsStep` puts a frame back, the frame it found took the coins, and
the frame put back is `Loop.coinsOf`'s. -/
theorem coinsStep_inv {R : Sessions} {sid i r j : Nat} {p p' : Rep} {cs : Array F128}
    (hf : R.find sid i r = some (j, p)) (h : _root_.Flock.Firewall.Frames.coinsStep R sid i r cs = .ok p') :
    ∃ st₂, _root_.Flock.Firewall.Frame.step p.cfg p.st (.coins (.ok cs)) = (.ok, st₂) ∧
      p' = Loop.coinsOf p st₂ cs := by
  unfold _root_.Flock.Firewall.Frames.coinsStep at h
  simp only [hf] at h
  split at h
  · cases h
    exact ⟨_, by assumption, rfl⟩
  all_goals cases h

/-- **A frame round as the loop ran it gives `stopped_next`'s run of updates**: where its rep had a frame when the
round read the loop's frames, the round released and put a frame back, and the rep's frame took the auditor's coins, the
updates from the round's read to its put, from its put to the coins' put, and from that to the session's next round
keep the rep's frame and the session's stop, and the coins' put is `Loop.coinsOf`'s of the frame the round put. -/
theorem roundRun_chain {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    (hfr : (frameAt R sid i r).isSome) {o : ByteArray} {p₁ : Rep}
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) :
    ∃ R₁ R₂ cs st₂, Kept sid i r R R₁ ∧ SessKept sid R R₁ ∧ Kept sid i r (R₁.put p₁) R₂ ∧
      SessKept sid (R₁.put p₁) R₂ ∧ _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂) ∧
      SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃ := by
  obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ :=
    hr.coins hreq ((hr.round hreq).trans ho) hput ha hb hs hc
  have hfa : frameAt R₂ sid i r = some p₁ := frameAt_own hfr (roundStep_key hput) (h01.1 i r) (h12.1 i r)
  obtain ⟨j, hj⟩ := find_of_frameAt hfa
  obtain ⟨st₂, hst, rfl⟩ := coinsStep_inv hj hcs
  exact ⟨R₁, R₂', _, st₂, h01.1 i r, h01.2, (h12.1 i r).trans (h22.1 i r), h12.2.trans h22.2, hst, h23.2⟩

/-- **What the relay read for a frame round as the loop ran it is the deferred release's message**: `envOk_round_next`
at a `RoundRun` whose content is the worker's message, in a session whose `Hello` left with the setup's domain
frames. -/
theorem envOk_round_run {W Rr : Type} (L : LeanRelease W Rr) (nN : Msg → Nat) (wk : L.toSrv.Worker) (x : Msg)
    (w : W) (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hn : h.length + 1 < nN x) (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {sid : Nat} {st : State} {pl : Placed} {stream : String}
    {i r : Nat} {dom : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {R₃ : Sessions} (hr : RoundRun sid st pl stream (L.coins x h.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (L.coins x h.length)) = none) {q : Rep} {f : Bool} {nd : Option String}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h) {spec : SessionSpec} {seen : List String} {k : Nat}
    (hσ : SetupOk spec st) (hsent : st.sent = true) (hf : f = !seen.contains stream) :
    EnvOk spec seen (.round stream k) e (L.toSrv.nextD L.fillOf nN wk x w ρ κ h) :=
  envOk_round_next L nN wk x w ρ κ h hk hn hlt hv hcr hm hfr hrp hnd hcfg hu hst (hr.round hreq) hf
    (hr.dom_eq.symm.trans (dom_setupOk hσ hsent stream))

/-- **The stop invariant one round on, at a frame round as the loop ran it**: `stopped_round_next` at a `RoundRun`
whose content is the worker's message, where the round released and the rep's frame took the auditor's coins, the
session going on: at its next round, the session has stopped exactly where a round up to this one failed. -/
theorem stopped_round_run {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {sid : Nat} {st : State} {pl : Placed} {stream : String}
    {i r : Nat} {dom : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {R₃ : Sessions} (hr : RoundRun sid st pl stream (L.coins x h.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (L.coins x h.length)) = none) {q p₁ : Rep} {f : Bool}
    {nd : Option String} {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).put =
      some p₁)
    {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (L.coins x h.length)) a).1.stop = none) (hc : e.coins = .ok ())
    (e' : Msg × Msg) : R₃.stopped sid = !L.toSrv.liveBefore wk x w ρ κ (h ++ [e']) := by
  obtain ⟨R₁, R₂, cs, st₂, hK₀₁, hS₀₁, hK₁₂, hS₁₂, hcs, hS₂₃⟩ := roundRun_chain hr hreq hfr ho hput ha hb hs hc
  exact stopped_round_next L wk x w ρ κ h hk hlt hv hcr hm hfr hrp hnd hcfg hu hst ho hput hK₀₁ hS₀₁ hK₁₂ hS₁₂ hcs
    hS₂₃ e'

end Claims.ZKServer.Relay
