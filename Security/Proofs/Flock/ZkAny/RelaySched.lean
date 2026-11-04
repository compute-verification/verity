module

public import Proofs.Flock.ZkAny.RelaySession
public import Proofs.Flock.ZkAny.RelayParams

/-!
# A round's frame keeps the session's stop, and its scheduled round is the model's

Two of the Round slot's facts at a frame round, from the loop's code:

* `prior_stopped`: rep 1's prior step at its first round (rep 0's algebraic-phase coins) adds only prior claims to the
  masking state. **`roundPlace_stopped`**: so the frame `roundPlace` readies has stopped exactly where the frame it
  found had, and (`roundPlace_live`) not in a session that has not stopped. **`stopped_step`**: `stopped_next` with
  no premise on the readied frame: from a round that releases to the session's next request, the session's stop is
  the round's put's on the frames the round read.
* `placeAt_sched`: where the schedule so far places a round in the scheduled place, it holds that round there.
  **`scheduled_place`**: the model's scheduled round at the loop's own frames (`LeanRelease.ofLoop`), after a history
  whose located rep's own table and schedule so far on the history's coins place round `rr` at the located round, is
  `rr`: the round `Loop.placeOf_sched` places at a frame keeping that schedule on those coins.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Frame (Cfg)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **Rep 1's prior step keeps the masking state's stop**: it adds only prior claims. -/
theorem prior_stopped (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (k : Nat) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.step c st (.prior k b)).2.m.stopped = st.m.stopped := by
  simp only [_root_.Flock.Firewall.Frame.step]
  split
  · rfl
  · split <;> rfl

theorem prior_stopped' {c : Cfg} {st st' : _root_.Flock.Firewall.Frame.State} {k : Nat} {b : ByteArray}
    (h : _root_.Flock.Firewall.Frame.step c st (.prior k b) = (.ok, st')) : st'.m.stopped = st.m.stopped := by
  have e := prior_stopped c st k b
  rw [h] at e
  exact e

theorem placeOf_st' {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : _root_.Flock.Firewall.Frames.Release} {p : Rep} {dom : ByteArray}
    {v : _root_.Flock.Firewall.Frame.Place × Nat × Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) : v.2.2.st = p.st :=
  (placeOf_run (pl := v.1) (n := v.2.1) h).2.1

set_option maxHeartbeats 1000000 in
/-- **The frame `roundPlace` readies has stopped exactly where the frame it found had**: `withAlg`, rep 0's
algebraic-phase coins at rep 1's first round and `placeOf` keep the masking state's stop. -/
theorem roundPlace_stopped {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String} (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    q.st.m.stopped = p.st.m.stopped := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  by_cases hfst : (p.coins.isEmpty && !p.st.begun) = true
  all_goals repeat' (first
    | split at h
    | simp only [eq_true hfst, ↓reduceIte] at h
    | simp only [eq_false hfst, ↓reduceIte] at h
    | simp only at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h
       have e1 := placeOf_st' (by assumption)
       first
         | exact congrArg (fun s : _root_.Flock.Firewall.Frame.State => s.m.stopped) e1
         | (have e2 := (withAlg_run (by assumption)).2.1
            first
              | exact congrArg (fun s : _root_.Flock.Firewall.Frame.State => s.m.stopped) (e1.trans e2)
              | (have e3 := prior_stopped' (by assumption)
                 exact ((congrArg (fun s : _root_.Flock.Firewall.Frame.State => s.m.stopped) e1).trans e3).trans
                   (congrArg (fun s : _root_.Flock.Firewall.Frame.State => s.m.stopped) e2))))

/-- **In a session that has not stopped, the frame `roundPlace` readies has not stopped.** -/
theorem roundPlace_live {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hR : R.stopped sid = false) : q.st.m.stopped = false := by
  rcases hf : R.find sid i r with _ | ⟨_, p⟩
  · unfold _root_.Flock.Firewall.Frames.roundPlace at h
    simp only [hf] at h
    cases h
  · rw [roundPlace_stopped hf h]
    have hfa : frameAt R sid i r = some p := by
      rw [← find_frameAt, hf] <;> rfl
    exact stopped_frameAt hR hfa

/-- **The session's stop from a round that releases to its next request** is the round's put's on the frames the
round read (`stopped_next`, its readied frame's stop from `roundPlace_live`). -/
theorem stopped_step {R R₁ R₂ R₃ : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {rr : ByteArray} {f : Bool} {nd : Option String}
    {o : ByteArray} (hfr : (frameAt R sid i r).isSome)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, .sched rr, f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁)
    (hK₀₁ : Kept sid i r R R₁) (hS₀₁ : SessKept sid R R₁) (hK₁₂ : Kept sid i r (R₁.put p₁) R₂)
    (hS₁₂ : SessKept sid (R₁.put p₁) R₂) {cs : Array _root_.Flock.F128} {st₂ : _root_.Flock.Firewall.Frame.State}
    (hcs : _root_.Flock.Firewall.Frame.step p₁.cfg p₁.st (.coins (.ok cs)) = (.ok, st₂))
    (hS₂₃ : SessKept sid (R₂.put (Loop.coinsOf p₁ st₂ cs)) R₃) :
    R₃.stopped sid = (R.put p₁).stopped sid :=
  stopped_next hfr hrp (roundPlace_live hrp) ho hput hK₀₁ hS₀₁ hK₁₂ hS₁₂ hcs hS₂₃

/-- **Where the schedule so far places a round in the scheduled place, it holds that round there.** -/
theorem placeAt_sched {sched : Array (ByteArray × Nat)} {k : Nat} {rr : ByteArray}
    (h : (Loop.placeAt sched k).1 = .sched rr) : ∃ n, sched[k]? = some (rr, n) := by
  unfold Loop.placeAt at h
  split at h
  · cases h
    exact ⟨_, by assumption⟩
  · cases h

/-- **The model's scheduled round is the round the schedule places in the device's place**: at the loop's own frames,
after a history whose located rep's own table is `(t, hi)` and whose schedule so far on the history's coins (`sched`)
places round `rr` at the located round, the scheduled round is `rr`. -/
theorem scheduled_place (T : ZKMal.CoinLayer.TableData)
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
    (LeanRelease.ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals = rr.data.toList := by
  obtain ⟨n, hn⟩ := placeAt_sched hpl
  have hlt : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2 < sched.size := by
    rcases Nat.lt_or_ge (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2 sched.size with hl | hge
    · exact hl
    · rw [Array.getElem?_eq_none hge] at hn
      cases hn
  have hg : (sched.getD (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q).2 (.empty, 0)).1 = rr := by
    simp only [Array.getD_eq_getD_getElem?, hn, Option.getD_some]
  show (match Loop.schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₀.scheduled x w ρ (e0 :: h1) vals) = _
  rw [Loop.schedOf_eq (real x w ρ) x _ hx (by rw [List.length_map, hq]) hreal hs hlt, hg]

end Claims.ZKServer.Relay
