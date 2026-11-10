module

public import Proofs.Flock.ZkAny.RelayCount
public import Proofs.Flock.Claims.ZK.FramePlan

/-!
# A rep's algebraic phase as the loop schedules it, beside the placeholder's

At a rep's first round the loop schedules the rep's algebraic phase on its table's statement (`Frames.withAlg`,
`Frames.algOf`); the placeholder (`Loop.placeholder`) schedules it on its stream's table's statement. The rounds' counts
are the PIOP's for the statement (`Count.rounds_counts`), whatever the level-0 root and the domain, so where the two
statements are one, the two phases have as many scheduled rounds:

* `withAlg_alg`: on a frame with no scheduled rounds yet, `withAlg` schedules them by `Skeleton.rounds` on the table's
  statement, at its `--zk` schedule.
* **`withAlg_size`**: so on a frame with none yet, the frame `withAlg` gives has as many scheduled rounds as the
  placeholder's algebraic phase, where the placeholder's stream's table's statement is the frame's table's
  (`roundPlace_switch1`'s count).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Rep Release)
open _root_.Flock.Firewall.Frame (Cfg)

/-- **`withAlg` on a frame with no scheduled rounds yet** schedules the algebraic phase by `Skeleton.rounds` on the
frame's table's statement, at its `--zk` schedule, on the level-0 root and the stream's domain. -/
theorem withAlg_alg {s : Release} {l0 : _root_.Flock.Firewall.Level0.Level0} {name : String} {p q : Rep} {stm : Stmt}
    (h : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok q) (hp : p.alg.isEmpty = true)
    (hst : s.sts[p.table]? = some stm) :
    ∃ sch tp, _root_.Flock.Zk.schedule stm.m = .ok (sch, tp) ∧
      _root_.Flock.Firewall.Skeleton.rounds (_root_.Flock.Setup.ofCircuit stm) sch.k0 l0.root
        (s.spec.domain name p.rep).toUTF8 = .ok q.alg := by
  unfold _root_.Flock.Firewall.Frames.withAlg at h
  simp only [hp, hst, Bool.not_true, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure] at h
  split at h
  · cases h
  · rename_i alg hal
    cases h
    unfold _root_.Flock.Firewall.Frames.algOf at hal
    simp only [bind, Except.bind] at hal
    split at hal
    · cases hal
    · rename_i a ha
      exact ⟨a.1, a.2, ha, hal⟩

/-- **The algebraic phase `withAlg` schedules has the placeholder's count of rounds**: on a frame with no scheduled
rounds yet, whose table's statement is the placeholder's stream's table's, the frame `withAlg` gives has as many
scheduled rounds as the placeholder's algebraic phase. -/
theorem withAlg_size {s : Release} {l0 : _root_.Flock.Firewall.Level0.Level0} {name : String} {p p' : Rep}
    {stm : Stmt} (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') (hp : p.alg.isEmpty = true)
    (hst : s.sts[p.table]? = some stm) {spec : SessionSpec} {sts : Array Stmt} {k : Nat} {link : Array F128}
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} (hph : Loop.placeholder spec sts k link = .ok (c', t, dom', xs, hi))
    (hk : sts[k / 2]! = stm) {alg : Array (ByteArray × Nat)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom' =
      .ok alg) :
    p'.alg.size = alg.size := by
  obtain ⟨sch, tp, hsch, ha⟩ := withAlg_alg hw hp hst
  have e7 : (_root_.Flock.Setup.ofCircuit stm).m = stm.m := rfl
  have hm : (_root_.Flock.Setup.ofCircuit stm).m < 2 ^ 64 := by
    rw [e7]
    exact Nat.lt_trans (Count.sched_facts hsch).1 (by decide)
  have hts : t.st = _root_.Flock.Setup.ofCircuit stm := by rw [(Count.placeholder_tab hph).1, hk]
  have hmt : t.st.m < 2 ^ 64 := by
    rw [hts]
    exact hm
  have e1 := congrArg List.length (Count.rounds_counts hm ha)
  have e2 := congrArg List.length (Count.rounds_counts hmt hal)
  rw [hts] at e2
  simp only [List.length_map, Array.length_toList] at e1 e2
  exact e1.trans e2.symm

end Claims.ZKServer.Relay
