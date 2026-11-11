module

public import Proofs.Flock.Claims.ZK.FrameLoop

/-!
# Schedule totality

`Loop.repState` schedules a rep's rounds on the coins of the rounds the frame has taken (`schedSoFar` on `cr.pop`), and the
round in the device's place after them is the schedule's next on one more round's coins (`Loop.schedOf`, on `cr`). That
the second is not refused where the first is not is a fact about the opening program alone (`Drive.openStep`), not about
the frame:

* `Loop.ScheduleTotal`: at each rep stream's placeholder table and salts (`Loop.placeholder`, at a link of the session's
  `points` link points, `Drive.linkPoints`, as the ring switch's round count reads them), a schedule
  not refused on `cr.pop` is not refused on `cr`, where `cr` holds each scheduled round's coin count (`Drive.stepAsks`'
  check) and level 0's pair `ρ₀, ρ₁` is independent over F128 once `cr` holds both (`Finish.ligeritoPZk`'s stop). Their
  rounds are Ligerito's `k₀ + 2` and the next, after the rep's `piopRoundsZk` rounds (`FlockComplete.rhoSlotsZk`).
  `schedSoFar_alike` carries it to the rep's real table.
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (F128)

namespace Loop

/-- **Schedule totality at `x`**: at each rep stream `k` of `x`'s session, at its placeholder table and salts at a link
of the session's `points` link points (so its ring switch has `piopRoundsZk`'s claims), a rep's schedule not refused on
its rounds' coins `cr.pop` is not refused on `cr`, where each array of `cr` holds its scheduled round's coin count and
level 0's pair coins (words `0` and `1` of rounds `piopRoundsZk + k₀ + 2` and the next) are independent over F128 once
`cr` holds both. -/
def ScheduleTotal (x : Msg) : Prop :=
  ∀ (s0 : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt),
    FlockDeploy.setupOf true x = some (s0, spec, sts) →
  ∀ k < 2 * sts.size, ∀ (link : Array F128), link.size / sts[k / 2]!.mPts = spec.points →
  ∀ (c : _root_.Flock.Firewall.Frame.Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
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
