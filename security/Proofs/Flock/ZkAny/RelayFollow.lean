module

public import Proofs.Flock.ZkAny.RelayCarry
public import Proofs.Flock.ZkAny.RelayMark

/-!
# A rep's next round reads the frame its last coins put back

Past a stream's first round, `Frames.roundPlace` readies the frame it finds as it found it: `placeOf` keeps the frame's
configuration, state and coins. So at the rep's next round in the session, after the frame its last round's coins put
back (`roundRun_carry`):

* `roundPlace_run_later`: past the stream's first round, the frame `roundPlace` readies is the found frame's
  configuration, state and coins.
* `repRun_snoc_coins`, `roundPlace_next`: a frame that ran a round has coins, so its next round is not its stream's
  first, and the readied frame is the found one's.
* **`roundRun_follow`**: so at a `RoundRun` whose next round is the same rep's, where only updates that keep the rep's
  frame come before that round's read, the round is not the stream's first, and the frame it readies is its run one
  round on, at the model's masking state for that round but for `stopped`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Masker Req Place)

/-- `placeOf_run` at its result as a whole. -/
theorem placeOf_run' {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : _root_.Flock.Firewall.Frames.Release} {p : Rep} {dom : ByteArray} {v : Place × Nat × Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) :
    v.2.2.cfg = p.cfg ∧ v.2.2.st = p.st ∧ v.2.2.coins = p.coins :=
  placeOf_run (pl := v.1) (n := v.2.1) h

set_option maxHeartbeats 1000000 in
/-- **Past the stream's first round the readied frame is the found one**: its configuration, state and coins. -/
theorem roundPlace_run_later {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hl : f = false) : q.cfg = p.cfg ∧ q.st = p.st ∧ q.coins = p.coins := by
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
       first
         | exact placeOf_run' (by assumption)
         | (rw [hfst] at hl
            cases hl)
         | (cases hl; done))

/-- **A frame that ran a round has coins.** -/
theorem repRun_snoc_coins {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    {e : Nat × ByteArray × Place × Array F128} {p : Rep} (h : RepRun pre (ev ++ [e]) p) :
    p.coins.isEmpty = false := by
  have h2 := congrArg List.length h.2
  simp only [Array.length_toList, List.map_append, List.length_append, List.length_map, List.map_cons,
    List.map_nil, List.length_cons, List.length_nil] at h2
  simp [Array.isEmpty, h2]

/-- **At a frame with coins, the round is past its stream's first**, and the readied frame is the found one. -/
theorem roundPlace_next {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hfa : frameAt R sid i r = some p) (hk : p.coins.isEmpty = false)
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    f = false ∧ q.cfg = p.cfg ∧ q.st = p.st ∧ q.coins = p.coins := by
  obtain ⟨j, hj⟩ := find_of_frameAt hfa
  have hl : f = false := by rw [roundPlace_first hj h, hk, Bool.false_and]
  exact ⟨hl, roundPlace_run_later hj h hl⟩

open Loop LeanRelease in
/-- **The rep's next round reads its frame at the model's state**: at `roundRun_carry`'s round, where only updates
that keep the rep's frame come between the frames the session's next request sees (`R₃`) and the same rep's next
round's read (`R'`), that round is not its stream's first, and the frame `roundPlace` readies there is the run of the
frame this round readied one round on, at the model's state for the history with `(v, a)` before the rank stop
(`answeredR`) but for `stopped`. -/
theorem roundRun_follow (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 h1.length).2 = j + 1) {sid : Nat} {st : State}
    {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Msg} {e : Env} {R : Sessions}
    {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream (loopNr x h1.length) i r dom ⟨v.toArray⟩ e R L0 R₃)
    (hreq : st.request pl (.round stream (loopNr x h1.length)) = none) (hfr : (frameAt R sid i r).isSome)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length) =
      .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length)).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom ⟨v.toArray⟩ (loopNr x h1.length)).put =
      some p₁) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr x h1.length)) a).1.stop = none) (hc : e.coins = .ok ())
    {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q)
    (hcfg : CfgAlike q.cfg ((ofLoop T L₀ real).cfg x ρ (e0 :: (h1 ++ [(v, a.data.toList)]))))
    (hu : unstop (_root_.Flock.Firewall.Frame.step q.cfg q.st
        (.round (loopNr x h1.length) (if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray)) pl')).2.m =
      unstop (progOf x ((h1 ++ [(v, a.data.toList)]).map fun p => ZKMal.CoinLayer.rd p.2)))
    {m : Masker} (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: (h1 ++ [(v, a.data.toList)])) = .ok m)
    {R' : Sessions} (hK : Kept sid i r R₃ R') {L0' : _root_.Flock.Firewall.Level0.Slots}
    {dom' content' : ByteArray} {n' : Nat} {q' : Rep} {pl'' : Place} {f' : Bool} {nd' : Option String}
    (hrp' : _root_.Flock.Firewall.Frames.roundPlace R' L0' sid i r dom' content' n' = .ok (q', pl'', f', nd')) :
    f' = false ∧
      RepRun pre (ev ++ [(loopNr x h1.length, if nd.isSome then ByteArray.empty else (⟨v.toArray⟩ : ByteArray), pl',
        _root_.Flock.Firewall.Frames.feedOf (loopNr x h1.length) a.data.toList)]) q' ∧
      unstop q'.st.m = unstop m := by
  obtain ⟨p, hat, hrun, hu'⟩ :=
    roundRun_carry T L₀ real x ρ e0 h1 hx hj hr hreq hfr hrp ho hput ha hb hs hc hR hcfg hu hA
  have hat' : frameAt R' sid i r = some p := by
    rw [frameAt_kept hK (by rw [hat]; rfl), hat]
  obtain ⟨hl, hcf, hst, hco⟩ := roundPlace_next hat' (repRun_snoc_coins hrun) hrp'
  exact ⟨hl, hrun.of_eq hcf hst hco, by rw [hst]; exact hu'⟩

end Claims.ZKServer.Relay
