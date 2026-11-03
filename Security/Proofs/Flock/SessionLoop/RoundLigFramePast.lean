import Proofs.Flock.SessionLoop.RoundLigFrameBind
import Proofs.Flock.SessionLoop.RoundLigFrameShape

/-!
# A rep's schedule past level 0's close

Past level 0's closing round `C` (the rep's algebraic phase, the ring switch's and `γ`'s rounds, then `k0 + 4`), each
round a rep's schedule holds, up to its next, is `ligeritoPZk`'s body there (`Bind.sched_lig`), so it is the shape
`LigShape.ligeritoPZk_logQ` gives Ligerito's bodies from the close on: the closing label at `C`, then good frames
(`LigShape.Good`), then a squeeze of the round's coin count.

* **`LigOff.sched_past`**: fed fewer coins than the opening asks for, each round of the schedule from `C` to the rounds
  it was fed is `PastRound`'s.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.LigOff

open _root_.Flock (F128)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- **A scheduled round past level 0's close at `C`**: at round `p`, its bytes are the closing label if `p = C`, good
frames, then a squeeze of its count `r.2` of coins, `r.2 ≠ 0` and below `2 ^ 64`, as the frame takes it. -/
def PastRound (C p : ℕ) (r : ByteArray × ℕ) : Prop :=
  ∃ (fs : List Frame) (k : UInt8),
    r.1.data.toList = encs (FlockComplete.LigShape.preAt C p ++ fs) ++ (Frame.sq k r.2).enc ∧
    (∀ f ∈ fs, FlockComplete.LigShape.Good f) ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ r.2 ≠ 0 ∧ r.2 < 2 ^ 64

/-- **A rep's schedule past level 0's close**: fed fewer coins than the opening asks for, each round from the close
`C = alg.size + (xs.size + 2) + 1 + k0 + 4` to the rounds fed, the next included, is `PastRound`'s. -/
theorem sched_past {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hm14 : 14 ≤ t.st.m) (hn0 : t.sch.n ≠ 0) (htp : 2 * t.tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64)
    (hc2 : cr.size < alg.size + (Count.openList (xs.size + 2) t.sch).length) {p : ℕ}
    (hp1 : alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 ≤ p) (hp2 : p ≤ cr.size) :
    p < sched.size ∧
      PastRound (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4) p (sched.getD p (.empty, 0)) := by
  sorry

end Claims.ZKServer.LigOff
