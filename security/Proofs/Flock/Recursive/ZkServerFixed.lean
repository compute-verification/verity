module

public import Proofs.Flock.Recursive.ZkServerAdaptive

/-! The live zk_server at a worker whose requests are fixed finishes at one salt list exactly when it finishes at every list of that length. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.ZkServerAdaptive

open Flock Flock.ZkServer ZkServerRun ZkServerAny

def chopReqs : List ByteArray → List ByteArray → List (ByteArray × List ByteArray)
  | [], _ => []
  | b :: rest, ys => (b, ys.take (reads b)) :: chopReqs rest (ys.drop (reads b))

theorem shape_chopReqs : ∀ (reqs ys ys' : List ByteArray), ys.length = ys'.length →
    shape (chopReqs reqs ys) = shape (chopReqs reqs ys')
  | [], _, _, _ => rfl
  | b :: rest, ys, ys', h => by
    have ih := shape_chopReqs rest (ys.drop (reads b)) (ys'.drop (reads b)) (by simp only [List.length_drop, h])
    show (b, (ys.take (reads b)).length) :: shape (chopReqs rest (ys.drop (reads b))) =
      (b, (ys'.take (reads b)).length) :: shape (chopReqs rest (ys'.drop (reads b)))
    rw [ih, List.length_take, List.length_take, h]

theorem adaptOut_fixed (s : Schedule) (reqs : List ByteArray) :
    ∀ (rest : List ByteArray) (st : State) (h : History) (ys : List ByteArray), reqs.drop h.length = rest →
      adaptOut s (fun h => reqs[h.length]?) rest.length st h ys = liveOut s st (chopReqs rest ys)
  | [], _, _, _, _ => rfl
  | b :: rest, st, h, ys, hd => by
    have hW : (fun h : History => reqs[h.length]?) h = some b := by
      show reqs[h.length]? = some b
      rw [← List.head?_drop, hd]
      rfl
    have ih : ∀ h' : History, h'.length = h.length + 1 → ∀ (st' : State) (ys' : List ByteArray),
        adaptOut s (fun h => reqs[h.length]?) rest.length st' h' ys' = liveOut s st' (chopReqs rest ys') :=
      fun h' hl st' ys' => adaptOut_fixed s reqs rest st' h' ys' (by rw [hl]; exact drop_succ_of_drop hd)
    show adaptOut s _ (rest.length + 1) st h ys =
      liveOut s st ((b, ys.take (reads b)) :: chopReqs rest (ys.drop (reads b)))
    rcases hm : (stepOf s b).run { st with salts := st.salts ++ ys.take (reads b), sent := #[] } with e | ⟨⟨⟩, t⟩
    · rw [adaptOut_error hW hm, liveOut_error hm]
    · rw [adaptOut_ok hW hm, liveOut_ok hm, ih]
      simp

theorem adaptFinished_fixed (s : Schedule) (reqs ys : List ByteArray) :
    adaptFinished s (fun h => reqs[h.length]?) reqs.length ys = liveFinished s (chopReqs reqs ys) := by
  unfold adaptFinished
  rw [adaptOut_fixed s reqs reqs { salts := [] } [] ys rfl]
  rfl

theorem liveFinished_chopReqs (s : Schedule) (reqs : List ByteArray) {ys ys' : List ByteArray}
    (hl : ys.length = ys'.length) : liveFinished s (chopReqs reqs ys) = liveFinished s (chopReqs reqs ys') := by
  obtain ⟨_, _, fin, -, -, -, -, hL⟩ := Flock.SecurityProofs.liveOut_any s (shape (chopReqs reqs ys'))
  rw [(hL _ (shape_chopReqs reqs ys ys' hl)).2, (hL _ rfl).2]

end FlockSoundness.Discharge.Recursion.ZkServerAdaptive
