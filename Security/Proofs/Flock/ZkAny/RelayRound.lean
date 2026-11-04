module

public import Proofs.Flock.ZkAny.RelayStay

/-!
# A rep's later round in the loop's frames is at `placeOf`'s place

At a rep's round past its stream's first, `Frames.roundPlace` reads the rep's frame, its session's entry (`sessAt`),
its table's statement and name, its level 0 and, through `Frames.placeOf`, the loop's salts, and changes the frame only
as `placeOf` does:

* `roundPlace_later`: where `placeOf` places the round in a scheduled round's place, or fails it, the round's place
  is that, the frame `placeOf`'s, and the frame takes the device round.
* `roundPlace_own`: past the opening (`placeOf`'s `own`), the round's place is the inner proof's round `j` the
  frame's own (`Frame.innerSched` on the rounds the loop's finish computed for the rep, `Sessions.innerOf`), and the
  frame takes the device round exactly where the finish computed round `j`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)

/-- **A later round at a scheduled or failed place** is at `placeOf`'s place, on `placeOf`'s frame. -/
theorem roundPlace_later {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {pl : _root_.Flock.Firewall.Frame.Place} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (pl, k, q))
    (hpl : pl ≠ .own) :
    _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, false, none) := by
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  rcases pl with c | _ | _
  all_goals first
    | exact absurd rfl hpl
    | (simp only [hf, hs, hst, hname, hl0, hfst, hp, bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
        ↓reduceIte]
       all_goals rfl)

/-- **A later round past the opening** is at the inner proof's round `j` the frame's own, on the rounds the loop's
finish computed for the rep, and the frame takes the device round exactly where the finish computed round `j`. -/
theorem roundPlace_own {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p : Rep} {s : Release} {st : Stmt} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {k : Nat} {q : Rep}
    (hf : R.find sid i r = some (j, p)) (hs : sessAt R sid = some s) (hst : s.sts[i]? = some st)
    (hname : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hfst : (p.coins.isEmpty && !p.st.begun) = false)
    (hp : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (.own, k, q))
    (hj : q.st.m.inner.getD 0 < _root_.Flock.Firewall.Frame.INNER_ROUNDS) :
    ∃ nd, _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n =
        .ok (q, .sched (_root_.Flock.Firewall.Frame.innerSched q.cfg q.st.m (R.innerOf l0.root r) n
          (q.st.m.inner.getD 0)), false, nd) ∧
      (nd = none ↔ q.st.m.inner.getD 0 < (R.innerOf l0.root r).size) := by
  have hj' : ¬ _root_.Flock.Firewall.Frame.INNER_ROUNDS ≤ q.st.m.inner.getD 0 := Nat.not_le.mpr hj
  unfold sessAt at hs
  unfold _root_.Flock.Firewall.Frames.roundPlace
  simp only [hf, hs, hst, hname, hl0, hfst, hp, hj', bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
    ↓reduceIte]
  refine ⟨_, rfl, ?_⟩
  split
  · rename_i h
    simp [h]
  · rename_i h
    simp [h]

end Claims.ZKServer.Relay
