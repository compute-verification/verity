module

public import Proofs.Flock.Recursive.FirewallAdaptive

/-!
# The live firewall at a worker whose requests are fixed

A worker that sends a fixed list of requests, its `i`-th request the list's `i`-th whatever the ZK server answered
(`fun h => reqs[h.length]?`), runs the live session (`FirewallAny.liveOut`) on those requests, each with the next salts
it reads from the list. By `FirewallAny.liveOut_any`, whether that session finishes depends on the requests and how many
salts each reads, never on the salts' values, so one evaluation at one salt list decides it at every list of that
length.

* `chopReqs reqs ys`: the requests `reqs`, each with the salts `liveWith` reads for it from `ys` in turn.
* `adaptOut_fixed`, `adaptFinished_fixed`: the session at the fixed worker is the live session on `chopReqs reqs ys`.
* `liveFinished_chopReqs`: that session finishes at `ys` exactly when it finishes at any `ys'` of the same length.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallAdaptive

open Flock Flock.Firewall FirewallRun FirewallAny

/-- **The requests `reqs`, each with the salts it reads**: `reads b` of them from `ys` in turn, as `adaptOut` takes them. -/
def chopReqs : List ByteArray → List ByteArray → List (ByteArray × List ByteArray)
  | [], _ => []
  | b :: rest, ys => (b, ys.take (reads b)) :: chopReqs rest (ys.drop (reads b))

/-- Salt lists of one length give the requests the same shape. -/
theorem shape_chopReqs : ∀ (reqs ys ys' : List ByteArray), ys.length = ys'.length →
    shape (chopReqs reqs ys) = shape (chopReqs reqs ys')
  | [], _, _, _ => rfl
  | b :: rest, ys, ys', h => by
    have ih := shape_chopReqs rest (ys.drop (reads b)) (ys'.drop (reads b)) (by simp only [List.length_drop, h])
    show (b, (ys.take (reads b)).length) :: shape (chopReqs rest (ys.drop (reads b))) =
      (b, (ys'.take (reads b)).length) :: shape (chopReqs rest (ys'.drop (reads b)))
    rw [ih, List.length_take, List.length_take, h]

/-- **The fixed worker's session is the live session on its requests**: from the requests left, `rest`, on. -/
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

/-- **The fixed worker finishes exactly when the live session on its requests does.** -/
theorem adaptFinished_fixed (s : Schedule) (reqs ys : List ByteArray) :
    adaptFinished s (fun h => reqs[h.length]?) reqs.length ys = liveFinished s (chopReqs reqs ys) := by
  unfold adaptFinished
  rw [adaptOut_fixed s reqs reqs { salts := [] } [] ys rfl]
  rfl

/-- **Whether it finishes reads only how many salts there are** (`Flock.SecurityProofs.liveOut_any`). -/
theorem liveFinished_chopReqs (s : Schedule) (reqs : List ByteArray) {ys ys' : List ByteArray}
    (hl : ys.length = ys'.length) : liveFinished s (chopReqs reqs ys) = liveFinished s (chopReqs reqs ys') := by
  obtain ⟨_, _, fin, -, -, -, -, hL⟩ := Flock.SecurityProofs.liveOut_any s (shape (chopReqs reqs ys'))
  rw [(hL _ (shape_chopReqs reqs ys ys' hl)).2, (hL _ rfl).2]

end FlockSoundness.Discharge.Recursion.FirewallAdaptive
