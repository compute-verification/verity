module

public import Proofs.Flock.ZkAny.RelayLive

/-!
# A frame round's coin count and configuration are the model's

The model's release at the loop's own frames (`LeanRelease.ofLoop`) reads at each frame round the coin count the loop
hands `Frames.round` and the rep's configuration as the loop started its frame. Here the loop's are the model's:

* **`loopNr_round`**: a round the relay's coin check passes (`coinsOf`) asks the loop's count at its opening
  (`Loop.loopNr`), where the relay's schedule is the setup's; `ofLoop_coins_round`: so the model's coin count at the
  round after a history one longer than the openings is the round's.
* `roundPlace_cfg`, `roundStep_cfg`: the frame `roundPlace` readies has the configuration of the frame it found, and so
  has the frame the round puts back, and its coins' put (`Loop.coinsOf`, by definition): a rep's frame keeps its
  start's configuration.
* **`ofLoop_cfg`**: the model's configuration at a round of rep stream `k` is `Cfg.ofDraws` at the stream's start
  (`Drive.start` at table `k / 2`'s statement) of the rep's samples on the tape, with the start's mask slot, as
  `Frames.startRep` starts the rep's frame.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Session (State planOf reqAt)
open _root_.Flock.Firewall.Frame (Cfg Draws)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **A round's coin count is the loop's at its opening**: a round of `n` coins of `stream` that the relay's coin check
passes (`coinsOf`), where the relay's schedule is the setup's (`State.sendHello`'s `planOf`), is at an opening `j ≥ 1`
of the session's, and `n` is the loop's count there (`Loop.loopNr`). -/
theorem loopNr_round {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) {S : State} (hp : S.plan = planOf spec sts)
    {stream : String} {n : Nat} (hc : coinsOf S (.round stream n) = none) :
    ∃ C, S.session = some C ∧ 1 ≤ C.opened.length ∧ Loop.loopNr x C.opened.length = n := by
  obtain ⟨C, hC, h1, hg⟩ := coins_round hc
  refine ⟨C, hC, h1, ?_⟩
  have h0 : C.opened.length ≠ 0 := by omega
  have e : S.plan.getD (C.opened.length - 1) ("", 0) = (stream, n) := by
    rw [Array.getD_eq_getD_getElem?, hg, Option.getD_some]
  unfold Loop.loopNr
  rw [Loop.reqOf_eq hs, ← hp]
  unfold reqAt
  rw [if_neg h0]
  exact congrArg Prod.snd e

/-- **The model's coin count at a round the relay's coin check passes is the round's**: at the loop's own frames, the
round after a history one longer than the session's openings (the `Hello`, then one message an opening) takes the
round's `n` coins. -/
theorem ofLoop_coins_round (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) {S : State} (hp : S.plan = planOf spec sts)
    {stream : String} {n : Nat} (hc : coinsOf S (.round stream n) = none) :
    ∃ C, S.session = some C ∧ 1 ≤ C.opened.length ∧
      (LeanRelease.ofLoop T L₀ real).coins x (C.opened.length + 1) = n := by
  obtain ⟨C, hC, h1, hn⟩ := loopNr_round hs hp hc
  exact ⟨C, hC, h1, hn⟩

theorem placeOf_cfg' {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : _root_.Flock.Firewall.Frames.Release} {p : Rep} {dom : ByteArray}
    {v : _root_.Flock.Firewall.Frame.Place × Nat × Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) : v.2.2.cfg = p.cfg :=
  (placeOf_run (pl := v.1) (n := v.2.1) h).1

set_option maxHeartbeats 1000000 in
/-- **The frame `roundPlace` readies has the configuration of the frame it found**: `withAlg`, rep 0's
algebraic-phase coins at rep 1's first round and `placeOf` keep it. -/
theorem roundPlace_cfg {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String} (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    q.cfg = p.cfg := by
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
       have e1 := placeOf_cfg' (by assumption)
       first
         | exact e1
         | (have e2 := (withAlg_run (by assumption)).1
            exact e1.trans e2))

/-- **The frame a round puts back has the configuration of the frame `roundPlace` readied.** -/
theorem roundStep_cfg {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q p₁ : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) :
    p₁.cfg = q.cfg := by
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl) with ⟨a, st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  rfl

/-- **The model's configuration at a round of rep stream `k`** (`Loop.repOf`) is `Cfg.ofDraws` at the stream's start
(`Drive.start` at table `k / 2`'s statement) of the rep's samples on the tape, with the start's mask slot, as
`Frames.startRep` starts the rep's frame. -/
theorem ofLoop_cfg (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s, spec, sts)) {lay : _root_.Flock.Zk.PadLayout}
    {mask : _root_.Flock.Firewall.Frame.MaskSlot}
    (hst : _root_.Flock.Firewall.Drive.start sts[Loop.repOf x h.length / 2]! = .ok (lay, mask)) :
    (LeanRelease.ofLoop T L₀ real).cfg x ρ h =
      Cfg.ofDraws lay (Draws.ofBytes lay ((LeanRelease.ofLoop T L₀ real).draws ρ (Loop.repOf x h.length)))
        (some mask) := by
  have hS : Loop.startOf x (Loop.repOf x h.length) = .ok (lay, mask) := by
    unfold Loop.startOf
    rw [hx]
    exact hst
  have hr : (LeanRelease.ofLoop T L₀ real).rep x h.length = Loop.repOf x h.length := rfl
  have hl : (LeanRelease.ofLoop T L₀ real).lay x (Loop.repOf x h.length) = lay := by
    show ((Loop.startOf x (Loop.repOf x h.length)).toOption.map (·.1)).getD default = lay
    rw [hS]
    rfl
  have hsl : (LeanRelease.ofLoop T L₀ real).slot x (Loop.repOf x h.length) = some mask := by
    show (Loop.startOf x (Loop.repOf x h.length)).toOption.map (·.2) = some mask
    rw [hS]
    rfl
  unfold LeanRelease.cfg LeanRelease.drawsAt
  rw [hr, hl, hsl]

end Claims.ZKServer.Relay
