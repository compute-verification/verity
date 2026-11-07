module

public import Proofs.Flock.ZkAny.RelaySched

/-!
# A frame round of the session at the model's place, from one request to the next

At a frame round whose place, coin count, configuration and masking state are the model's, where the session has
stopped exactly where an earlier round failed:

* `scheduled_bytes`: the model's scheduled round at the loop's own frames, as the frame's place holds it, is the round
  the schedule so far places there (`scheduled_place`).
* **`envOk_round_next`**: what the relay read for the round, where it is the loop's round on its frames, is the
  deferred release's message (`EnvOk`), with no premise on the frame `roundPlace` readies (`roundPlace_live`).
* **`stopped_round_next`**: and at the session's next request, after the round's put, its coins and the updates in
  between, the session has stopped exactly where a round up to this one failed: the stop invariant one round on.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Frame (Masker)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **The model's scheduled round, as the frame's place holds it**, is the round the schedule so far places at the
located round (`scheduled_place`'s premises). -/
theorem scheduled_bytes (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (hreal : real x w ρ (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 = .ok (t, hi))
    {sched : Array (ByteArray × Nat)}
    (hs : Loop.schedSoFar t ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 % 2)
      (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 / 2]!)
        ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).1 / 2]!
          (Loop.wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi
      ((Array.range (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2).map
        fun j => Loop.wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2)
          (q - (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched)
    {rr : ByteArray}
    (hpl : (Loop.placeAt sched (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2).1 = .sched rr) :
    (⟨((LeanRelease.ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals).toArray⟩ : ByteArray) = rr := by
  rw [scheduled_place T L₀ real x w ρ e0 h1 vals hx hq hreal hs hpl, Array.toArray_toList]

/-- **What the relay read for a frame round is the deferred release's message**: `roundStep_nextD` and
`envOk_roundStep`, the frame `roundPlace` readies not stopped in a live session (`roundPlace_live`). -/
theorem envOk_round_next {W Rr : Type} (L : LeanRelease W Rr) (nN : Msg → Nat) (wk : L.toSrv.Worker) (x : Msg)
    (w : W) (ρ : Rr) (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hn : h.length + 1 < nN x) (hlt : h.length < L.last x) {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {sid i r : Nat} {dom : ByteArray} {q : Rep} {f : Bool} {nd : Option String}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h)
    {spec : SessionSpec} {seen : List String} {stream : String} {k : Nat} {e : Env}
    (he : e.round = (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out)
    (hf : f = !seen.contains stream) (hd : dom = domOf spec stream) :
    EnvOk spec seen (.round stream k) e (L.toSrv.nextD L.fillOf nN wk x w ρ κ h) := by
  refine envOk_roundStep hrp he (fun o p₁ ho hput => ?_) hf hd
  obtain ⟨out, hoo, hmsg, -⟩ :=
    roundStep_nextD L nN wk x w ρ κ h hk hn hlt hv hcr hm hfr hrp hnd (roundPlace_live hrp) hcfg hu ho hput hst
  exact ⟨out, hoo, hmsg⟩

/-- **The stop invariant one round on**: at a frame round with `envOk_round_next`'s premises that releases, after its
put, its coins and updates that keep the rep's frame and the session's stop in between (`Kept`, `SessKept`), the
session at its next request has stopped exactly where a round up to this one failed. -/
theorem stopped_round_next {W Rr : Type} (L : LeanRelease W Rr) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : Rr)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked) (hlt : h.length < L.last x)
    {v : Msg} (hv : wk x w ρ h 0 = some (v, false))
    (hcr : L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false)
    {m : Masker} (hm : L.answered x ρ h = .ok m) {R R₁ R₂ R₃ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots}
    {sid i r : Nat} {dom : ByteArray} {q p₁ : Rep} {f : Bool} {nd : Option String} {o : ByteArray}
    (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length) =
      .ok (q, .sched ⟨(L.scheduled x w ρ h (computedOf (L.computed x w ρ h))).toArray⟩, f, nd))
    (hnd : nd.isSome = R.stopped sid) (hcfg : q.cfg = L.cfg x ρ h) (hu : unstop q.st.m = unstop m)
    (hst : R.stopped sid = !L.toSrv.liveBefore wk x w ρ κ h)
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (L.coins x h.length)).put =
      some p₁)
    (hK₀₁ : Kept sid i r R R₁) (hS₀₁ : SessKept sid R R₁) (hK₁₂ : Kept sid i r (R₁.put p₁) R₂)
    (hS₁₂ : SessKept sid (R₁.put p₁) R₂) {cs : Array _root_.Flock.F128} {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂))
    (hS₂₃ : SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃) (e' : Msg × Msg) :
    R₃.stopped sid = !L.toSrv.liveBefore wk x w ρ κ (h ++ [e']) := by
  obtain ⟨-, -, -, hst'⟩ :=
    roundStep_nextD L (fun _ => h.length + 2) wk x w ρ κ h hk (by show h.length + 1 < h.length + 2; omega) hlt hv
      hcr hm hfr hrp hnd (roundPlace_live hrp) hcfg hu ho hput hst
  rw [stopped_step hfr hrp ho hput hK₀₁ hS₀₁ hK₁₂ hS₁₂ hcs hS₂₃, hst' e']

end Claims.ZKServer.Relay
