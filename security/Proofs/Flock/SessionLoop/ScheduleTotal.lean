module

public import Proofs.Flock.Claims.ZK.FrameLoop

/-! `Loop.ScheduleTotal`: a rep's schedule not refused on the coins the frame took is not refused on one more round's, a
fact about the opening program alone. -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (F128)

namespace Loop

def ScheduleTotal (x : Msg) : Prop :=
  ∀ (s0 : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt),
    FlockDeploy.setupOf true x = some (s0, spec, sts) →
  ∀ k < 2 * sts.size, ∀ (link : Array F128), link.size / sts[k / 2]!.mPts = spec.points →
  ∀ (c : _root_.Flock.ZkServer.Frame.Cfg) (t : _root_.Flock.ZkServer.Finish.Tab) (dom : ByteArray)
    (xs : Array (Array F128)) (hs : Array ByteArray),
    placeholder spec sts k link = .ok (c, t, dom, xs, hs) →
  ∀ (cr : Array (Array F128)) (sched : Array (ByteArray × ℕ)),
    schedSoFar t (k % 2) dom xs hs cr.pop = .ok sched →
    (∀ i (h : i < sched.size), i < cr.size → (cr[i]!).size = sched[i].2) →
    (let I := FlockComplete.piopRoundsZk spec sts[k / 2]! + t.sch.k0 + 2;
      I + 1 < cr.size →
        _root_.Flock.Zk.rhoIndependent (⟨(cr[I]!)[0]!, (cr[I]!)[1]!⟩, ⟨(cr[I + 1]!)[0]!, (cr[I + 1]!)[1]!⟩) = true) →
    ∃ s, schedSoFar t (k % 2) dom xs hs cr = .ok s

end Loop

end Claims.ZKServer
