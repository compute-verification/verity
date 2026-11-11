module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# A session's registered reads open its rows

`opens_of_setup`: at a session statement that sets up as a masked Flock statement with registered ports `own`, every
statement of its setup passes Flock's registered check against them (`setup_checkRegistered`), so with
`FlockSoundness.Discharge.ZkReg.checkRegistered_ok`, `check_tables` and `checkPort_at`, each instance's table row opens
its port's root, at the position the port gives for the instance's unit, through the path the session header gives.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A loop that succeeds ran every step to success**, where each step that succeeds continues: whatever `P` each
successful step gives holds at every element. -/
theorem forIn_ok_each {α β ε : Type} {f : α → β → Except ε (ForInStep β)} {P : α → Prop}
    (hf : ∀ a b s, f a b = .ok s → P a ∧ ∃ b', s = .yield b') :
    ∀ (l : List α) (b r : β), forIn l b f = .ok r → ∀ a ∈ l, P a
  | [], _, _, _, a, ha => absurd ha List.not_mem_nil
  | x :: l, b, r, h, a, ha => by
    rw [List.forIn_cons] at h
    cases hx : f x b with
    | error e => rw [hx] at h; cases h
    | ok s =>
      obtain ⟨hP, b', rfl⟩ := hf x b s hx
      rw [hx] at h
      cases ha with
      | head => exact hP
      | tail _ ha => exact forIn_ok_each hf l b' r h a ha

theorem forIn_ok_each_array {α β ε : Type} {f : α → β → Except ε (ForInStep β)} {P : α → Prop} {xs : Array α}
    {b r : β} (h : forIn xs b f = .ok r) (hf : ∀ a b s, f a b = .ok s → P a ∧ ∃ b', s = .yield b') :
    ∀ a ∈ xs, P a := by
  intro a ha
  rw [← Array.forIn_toList] at h
  exact forIn_ok_each hf xs.toList b r h a (Array.mem_toList_iff.2 ha)

/-- **The setup checks each statement's registered reads** against the inputs' registered ports. -/
theorem setup_checkRegistered {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hs : s.setup = .ok (spec, sts)) :
    ∀ st ∈ sts, _root_.Flock.checkRegistered st s.registered = .ok () := by
  unfold FlockDeploy.Ins.setup at hs
  simp only [bind, Except.bind, pure, Except.pure] at hs
  repeat' split at hs
  all_goals first | (cases hs; done) | (cases hs)
  all_goals
    intro st hst
    refine forIn_ok_each_array (P := fun st => _root_.Flock.checkRegistered st s.registered = .ok ()) ‹_› ?_ st hst
    intro a b r hr
    cases hc : _root_.Flock.checkRegistered a s.registered with
    | error e => simp [hc] at hr
    | ok u =>
      cases hn : FlockDeploy.noIdentifier a with
      | error e => simp [hc, hn] at hr
      | ok v =>
        simp [hc, hn] at hr
        cases hr
        exact ⟨rfl, _, rfl⟩

/-- **Each row the session reads from a registered port opens under the port**: at a session statement that sets up as
a masked Flock statement with registered ports `own`, for every statement of its setup and every port of `own`, the
circuit has the port by name, and each instance's table row opens the port's commitment at the position the port reads
for the instance's unit, through the path the session header gives for that row. -/
theorem opens_of_setup {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hown : ins.registered = some own) :
    ∀ st ∈ sts, ∃ tb idx, st.pub.tables = some tb ∧ _root_.Flock.Registered.unitIndices st.pub = .ok idx ∧
      ∀ reg ∈ own, ∃ p positions paths,
        (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
        _root_.Flock.Registered.entry st.pub reg = .ok (positions, paths) ∧ ∀ i < st.pub.n,
          _root_.Flock.Registered.ref tb st.c.ports.size p i < tb.counts.getD p 0 ∧
          reg.reads[idx.getD i 0]? =
            some (some (positions.getD (_root_.Flock.Registered.ref tb st.c.ports.size p i) 0)) ∧
          _root_.Flock.Registered.opens reg
            (_root_.Flock.Registered.tableRow st.pub tb p (_root_.Flock.Registered.ref tb st.c.ports.size p i))
            (positions.getD (_root_.Flock.Registered.ref tb st.c.ports.size p i) 0)
            (paths.getD (_root_.Flock.Registered.ref tb st.c.ports.size p i) ByteArray.empty) = true := by
  intro st hst
  obtain ⟨-, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hck : _root_.Flock.checkRegistered st (some own) = .ok () := by
    rw [← hown]
    exact setup_checkRegistered hsetup st hst
  obtain ⟨tb, idx, ht, hidx, hp⟩ := FlockSoundness.Discharge.ZkReg.check_tables
    (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hck)
  exact ⟨tb, idx, ht, hidx, fun reg hreg => FlockSoundness.Discharge.ZkReg.checkPort_at (hp reg hreg)⟩

end SpotFlock
