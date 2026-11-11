module

public import Proofs.Flock.ZkAny.RelayFollow

/-!
# A rep's first round after another rep's, at the model's state

At the session's first frame round, and at a round after another rep's round, the model checks the round at the next
rep's start (`Loop.freshOf`), which at rep 0 is the frame's start, and the loop's frame of rep 0 starts there:

* `ofLoop_answered_switch`: where the round after a history is the session's first frame round or another rep's than
  the round before it, the model's state for it is `freshOf`'s (`takeAt`'s first or third case);
  `ofLoop_answeredR_switch`: so is its state before the rank stop.
* `freshOf_rep0`: at a rep 0 stream's first round, `freshOf`'s state is the frame's start.
* `roundPlace_st`: at any rep but rep 1, the frame `roundPlace` readies is at the found frame's state: `withAlg` and
  `placeOf` keep it, and only rep 1's first round adds rep 0's algebraic-phase coins.
* **`roundPlace_switch0`**: so at rep 0's first round, on its frame's run on no requests (`RepRun [] []`), the frame
  `roundPlace` readies is at the model's state for a rep 0 stream's first round.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Masker Place)

open Loop LeanRelease in
/-- **At a rep switch the model's state is the next rep's start**: where the round after `e0 :: h1` is the session's
first frame round, or another rep's than the round before it, the model checks it at `freshOf`'s state (past the link,
after the last rep's frame took the coins). -/
theorem ofLoop_answered_switch (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg))
    (hs : 2 ≤ h1.length → repOf x (h1.length + 1) ≠ repOf x h1.length) {m : Masker}
    (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m) :
    freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m := by
  have hF : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold LeanRelease.answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer T L₀ real x ρ e0 h1] at hF
  rcases takeAt_ok hF with ⟨-, hf⟩ | ⟨hn, hb, -⟩ | ⟨-, -, -, hf⟩
  · exact hf
  · simp only [List.length_cons, Nat.add_sub_cancel, decide_eq_true_eq] at hb
    simp only [List.length_cons] at hn
    exact absurd hb (hs (by omega))
  · exact hf

open Loop LeanRelease in
/-- **At a rep switch the model's state before the rank stop is the next rep's start**, as `ofLoop_answered_switch`:
there is no rank stop at a rep's first round. -/
theorem ofLoop_answeredR_switch (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg))
    (hs : 2 ≤ h1.length → repOf x (h1.length + 1) ≠ repOf x h1.length) {m : Masker}
    (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m) :
    freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m :=
  ofLoop_answered_switch T L₀ real x ρ e0 h1 hs (answered_of_answeredR hA fun hn he => by
    simp only [List.length_cons, Nat.add_sub_cancel] at hn he
    exact absurd he (hs (by omega)))

open Loop in
/-- **At a rep 0 stream's first round the loop starts the frame at its start.** -/
theorem freshOf_rep0 (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    {q : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2 = 0) {m : Masker} (hm : freshOf x cs = .ok m) :
    m = {} := by
  obtain ⟨_, _, _, _, _, _, _, _, h | h⟩ := freshOf_ok x cs hx hq hm
  · exact h.2
  · exact absurd hk h.1

set_option maxHeartbeats 1000000 in
/-- **At any rep but rep 1 the readied frame is at the found frame's state**: `withAlg` and `placeOf` keep it. -/
theorem roundPlace_st {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hr : r ≠ 1) : q.st = p.st := by
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
         | exact (placeOf_run' (by assumption)).2.1
         | exact (placeOf_run' (by assumption)).2.1.trans (withAlg_run (by assumption)).2.1
         | exact absurd (by assumption : (r == 1) = true) (by simpa using hr))

/-- **At rep 0's first round the readied frame is at the model's state**: on its frame's run on no requests, where
the model's state for the round is `freshOf`'s at a rep 0 stream's first round. -/
theorem roundPlace_switch0 {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 0 = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 0 dom content n = .ok (q, pl, f, nd))
    (hR : RepRun [] [] p) (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec}
    {sts : Array Stmt} {q₀ : Nat} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q₀ + 1) (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 % 2 = 0)
    {m : Masker} (hm : Loop.freshOf x cs = .ok m) : q.st.m = m := by
  rw [roundPlace_st hf h (by decide), repRun_start_fresh hR, freshOf_rep0 x cs hx hq hk hm]
  all_goals rfl

end Claims.ZKServer.Relay
