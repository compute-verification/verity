module

public import Proofs.Flock.SessionLoop.Statement

/-!
# The session loop's own rounds of the inner proof

Past an opened rep's scheduled rounds the loop's frame releases the inner proof's rounds itself (`Frames.roundPlace`,
at `Frame.innerSched`): its round `j` is `Frame.ownRound`, every word the frame's own (its `ε`, `β`, pads and coin
count) but `final_c'` (round 0) and `τ`'s commitment (round 2), which it reads off the finish's round `j`, as the
finish hands it to the frame (`Sessions.innerRound`, from `Finish.liveAsks`'s `sent`) before Rust relays it. A relayed
round that differs from the frame's fails the frame's check at `.sched`, which asks for the round byte for byte: the
frame's own round stands in and the rep stops (`standIn`). A round the finish has not computed has no place: the loop
refuses it whatever Rust relays (`Frames.roundPlace`), and the session stops there. This file defines that release and
proves `LoopInnerRound0` at it, at every finish:

* `Loop.innerOf`: the round in the device's place past the located rep's schedule, at the inner proof's round `j`
  (`j < INNER_ROUNDS`), `Frame.innerSched` at the rep's configuration, `ε` and `β` the first words of the coins of the
  inner proof's rounds 0 and 2 (as `Loop.ownRounds` reads them), the finish's rounds `fin` and the round's coin count;
  none at a scheduled round, past the inner proof, or where `x` sets up no session.
* `LeanRelease.withInner`: `L₀` with those rounds, at the finish's rounds `fin` of the statement, the witness, the
  tape and the history; elsewhere `L₀`'s.
* `step_sched_self`: a round at `.sched r` with `r` itself in the device's place, an honest relay, leaves the frame's
  state at `.own`'s, `Drive.reqs`' place for the inner proof's rounds, which `Loop.repState` runs. A relay that differs
  stops the rep (`standIn`): the state but for `stopped`, as `ProgChecks` reads it.
* `Flock.SecurityProofs.LoopInnerRound0Own : Flock.Guarantees.LoopInnerRound0Own`: `LoopInnerRound0` holds at
  `withInner`, at every `fin`, `final_c'` the first word past the label frame and its header of the finish's round 0.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws INNER_ROUNDS labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The loop's own round of the inner proof in the device's place** after the decoded coins `cs` at `x`, at the
round's configuration `c` and the located rep's finish rounds `fin`: where rep stream `k`'s schedule so far
(`schedSoFar`, on `real k`) has ended before its round `i` (`k` and `i` as `schedOf` locates them), its inner proof's
round `i - sched.size`, `Frame.innerSched` (`Frames.roundPlace` at an opened rep), `ε` and `β` the first words of the
coins of the inner proof's rounds 0 and 2, squeezing the round's coin count (`loopNr`, its request's). None at a
scheduled round, past the inner proof (whose round the frame refuses), where the rep's table or schedule is refused, or
where `x` sets up no session. -/
noncomputable def Loop.innerOf (real : ℕ → V (Tab × Array ByteArray)) (c : Cfg) (fin : Array ByteArray) (x : Msg)
    (cs : List FlockDeploy.Coin) : Option ByteArray :=
  match FlockDeploy.setupOf true x, cs.length with
  | some (_, spec, sts), q + 1 =>
    let ki := CoinLayer.locate (Loop.roundsOf spec sts) 0 q
    let st := sts[ki.1 / 2]!
    let cr := (Array.range ki.2).map fun j => Loop.wordsAt x cs (q - ki.2 + j + 1)
    match real ki.1 with
    | .ok (t, hi) =>
      match Loop.schedSoFar t (ki.1 % 2) (spec.domain (spec.tables[ki.1 / 2]!) (ki.1 % 2)).toUTF8
          (_root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st (Loop.wordsAt x cs 0))) hi
          cr with
      | .ok sched =>
        if sched.size ≤ ki.2 ∧ ki.2 - sched.size < INNER_ROUNDS then
          let ci := cr.extract sched.size cr.size
          let s : Masker := { eps := (ci.getD 0 #[]).getD 0 0, beta := (ci.getD 2 #[]).getD 0 0 }
          some (_root_.Flock.Firewall.Frame.innerSched c s fin (Loop.loopNr x (q + 1)) (ki.2 - sched.size))
        else none
      | .error _ => none
    | .error _ => none
  | _, _ => none

namespace LeanRelease

/-- **`L₀` with the session loop's own rounds of the inner proof**, at the finish's rounds `fin` (rep stream `k`'s
finish's live inner rounds at `x`, the witness, the tape and the history, as the finish hands them to the frame): past
a rep's scheduled rounds, the loop's own round (`Loop.innerOf`), at the round's configuration as the loop's frames
start its rep (`withFrames`' `cfg`: `L₀`'s samples at `Drive.start`'s layout and mask slot); elsewhere `L₀`'s. -/
noncomputable def withInner (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      List (Msg × Msg) → Array ByteArray) :
    LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand) :=
  { L₀ with
    scheduled := fun x w ρ h vals =>
      match Loop.innerOf (real x w ρ) ((withFrames T L₀ real).cfg x ρ h) (fin x w ρ h) x
          (h.tail.map fun p => CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ h vals }

end LeanRelease

/-- **The session loop's own inner proof's round 0 is the ZK server's**: at every tables `T`, release `L₀`, rep tables
`real` and finish rounds `fin`, `LoopInnerRound0` holds at `withInner T L₀ real fin`, with `final_c'` the finish's. -/
def _root_.Flock.Guarantees.LoopInnerRound0Own : Prop :=
  ∀ (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (x : Msg) (w : Flock.Wit),
    _root_.Flock.Guarantees.LoopInnerRound0 T (LeanRelease.withInner T L₀ real fin) real x w

/-- **An inner round relayed as the frame computes it steps as the frame's own**: where the rep owes no coins and the
frame takes the round `r` inside the inner proof, `Frame.step` at `.sched r` (the place `Frames.roundPlace` gives an
opened rep's round) with `r` in the device's place leaves the state `.own` leaves (`Drive.reqs`' place for the inner
proof's rounds), the rep stopped or not. -/
theorem step_sched_self (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (n : ℕ) (r : ByteArray) {out : ByteArray}
    {s' : Masker} (ho : st.m.owed = none) (hr : _root_.Flock.Firewall.Frame.round c st.m r n = .ok (out, s'))
    (hin : s'.inner.isSome) :
    (_root_.Flock.Firewall.Frame.step c st (.round n r (.sched r))).2 =
      (_root_.Flock.Firewall.Frame.step c st (.round n r .own)).2 := by
  have hn : s'.inner.isNone = false := by cases h : s'.inner <;> simp_all
  have hown : (_root_.Flock.Firewall.Frame.step c st (.round n r .own)).2 = { m := s', begun := true } := by
    simp only [_root_.Flock.Firewall.Frame.step, ho, hr, hn, bind, Except.bind, pure, Except.pure,
      Bool.false_eq_true, ↓reduceIte]
  rw [hown]
  cases hs : st.m.stopped
  · simp only [_root_.Flock.Firewall.Frame.step, ho, _root_.Claims.ZKServer.check_self hs hr]
  · have he : ∀ m : Masker, m.stopped = true → { m with stopped := true } = m := by
      intro m hm
      cases m
      dsimp only at hm
      subst hm
      rfl
    have hc : _root_.Flock.Firewall.Frame.check c st.m r r n = .ok (out, s', some "the rep is stopped") := by
      unfold _root_.Flock.Firewall.Frame.check _root_.Flock.Firewall.Frame.standIn
      simp only [hs, he st.m hs, hr, ↓reduceIte, bind, Except.bind, pure, Except.pure]
    simp only [_root_.Flock.Firewall.Frame.step, ho, hc]

theorem wordBytes_size (v : Array F128) : (wordBytes v).size = 16 * v.size := by
  rw [← ByteArray.size_data, ← Array.length_toList, wordBytes_data, FlockSoundness.Refine.Frame.colBytes_length,
    Array.length_toList]

/-- **The loop's own round past a rep's schedule** after the decoded coins `cs`, where `x` sets up a session, the
located rep's own table and salts are `(t, hi)` and its schedule so far at them, `sched`, ends `i` rounds before the
located round, `i` within the inner proof: the inner proof's round `i`, the frame's own on the finish's rounds. -/
theorem Loop.innerOf_eq (real : ℕ → V (Tab × Array ByteArray)) (c : Cfg) (fin : Array ByteArray) (x : Msg)
    (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    (hloc : CoinLayer.locate (Loop.roundsOf spec sts) 0 q = (k, j)) {t : Tab} {hi : Array ByteArray}
    (hreal : real k = .ok (t, hi)) {sched : Array (ByteArray × ℕ)}
    (hs : Loop.schedSoFar t (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (Loop.wordsAt x cs 0))) hi
      ((Array.range j).map fun i => Loop.wordsAt x cs (q - j + i + 1)) = .ok sched)
    (hle : sched.size ≤ j) (hlt : j - sched.size < INNER_ROUNDS) :
    Loop.innerOf real c fin x cs =
      some (_root_.Flock.Firewall.Frame.innerSched c
        { eps := ((((Array.range j).map fun i => Loop.wordsAt x cs (q - j + i + 1)).extract sched.size j).getD 0
            #[]).getD 0 0,
          beta := ((((Array.range j).map fun i => Loop.wordsAt x cs (q - j + i + 1)).extract sched.size j).getD 2
            #[]).getD 0 0 }
        fin (Loop.loopNr x (q + 1)) (j - sched.size)) := by
  unfold Loop.innerOf
  rw [hx, hq]
  simp only [hloc, hreal, hs, Array.size_map, Array.size_range, hle, hlt, and_self, ↓reduceIte]

end Claims.ZKServer

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open _root_.Flock.Firewall.Frame (labelFrame wordBytes words)

/-- **The session loop's own inner proof's round 0 is the ZK server's**, at every finish: at a rep's first round past
its schedule, `withInner`'s round is `Frame.innerSched`'s round 0, the inner proof's label frame and `final_c'` between
its headers, `final_c'` one word (16 bytes) of the finish's round 0. -/
theorem LoopInnerRound0Own : Flock.Guarantees.LoopInnerRound0Own := by
  intro T L₀ real fin x w ρ h vals s0 spec sts q k j tr hr rsched hx hlen hloc hreal hs hsize
  have hq : (h.tail.map fun p => CoinLayer.rd p.2).length = q + 1 := by
    simp only [List.length_map, List.length_tail]; omega
  refine ⟨wordBytes (words ((fin x w ρ h).getD 0 .empty)
    ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1), ?_, ?_⟩
  · simp only [wordBytes_size, Loop.words_size]
  · have h0 : j + 1 - rsched.size = 0 := by omega
    have e := Loop.innerOf_eq (real x w ρ) ((LeanRelease.withFrames T L₀ real).cfg x ρ h) (fin x w ρ h) x _ hx hq
      hloc hreal hs (by omega) (by rw [h0]; decide)
    rw [h0] at e
    show (match Loop.innerOf (real x w ρ) ((LeanRelease.withFrames T L₀ real).cfg x ρ h) (fin x w ρ h) x
        (h.tail.map fun p => CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ h vals) = _
    rw [e]
    rfl

end Flock.SecurityProofs
