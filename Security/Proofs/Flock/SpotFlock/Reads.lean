module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# The session's registered reads are the run's

`checkRegistered_of_stmtOkAt`: at a run's statement and template `t`'s session statement as the verifier's round-2
check passes it (`stmtOkAt`), the session statement sets up as a masked Flock statement whose registered ports are the
ones `h.registered` gives for `t` at the run's roots and the interior registration's, read as Flock's verifier reads
them (`Registered.ofJson`), and every statement of its setup passes Flock's registered check against those ports
(`checkRegistered`).

With `FlockSoundness.Discharge.ZkReg.checkRegistered_ok`, `check_tables` and `checkPort_at`, each instance's table row
then opens its port's root, at the position the port gives for the instance's unit, through the path the session header
gives.
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

theorem ok_of_map_some {ε α : Type} {e : Except ε α} {v : Option α} (h : (some <$> e) = Except.ok v) :
    ∃ a, e = .ok a ∧ v = some a := by
  cases e with
  | error x =>
    have h' : (Except.error x : Except ε (Option α)) = Except.ok v := h
    cases h'
  | ok a =>
    have h' : (Except.ok (some a) : Except ε (Option α)) = Except.ok v := h
    cases h'
    exact ⟨a, rfl, rfl⟩

/-- **The inputs' registered ports** are Flock's reading of the statement's registered reads. -/
theorem ins_registered {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} {b : ByteArray} (hw : w.ins = .ok s)
    (hb : w.registered = some b) :
    ∃ own, s.registered = some own ∧
      (FlockDeploy.jsonOf "registered reads" b >>= _root_.Flock.Registered.ofJson) = .ok own := by
  unfold FlockDeploy.Raw.ins at hw
  rw [hb] at hw
  simp only [bind, Except.bind, pure, Except.pure, Option.mapM] at hw
  repeat' split at hw
  all_goals first | (cases hw; done) | skip
  all_goals
    cases hw
    obtain ⟨own, hj, rfl⟩ := ok_of_map_some ‹_›
    exact ⟨own, rfl, by simpa only [bind, Except.bind] using hj⟩

/-- The statement's inputs are its decoding's parsers' output. -/
theorem insOf_ok {zk : Bool} {x : Claims.Msg} {s : FlockDeploy.Ins} (h : FlockDeploy.insOf zk x = some s) :
    ∃ w, FlockDeploy.decode x = some w ∧ w.ins = .ok s := by
  unfold FlockDeploy.insOf at h
  cases hd : FlockDeploy.decode x with
  | none => simp [hd] at h
  | some w =>
    cases hi : w.ins with
    | error e => simp [hd, hi, Except.toOption] at h
    | ok s' =>
      simp [hd, hi, Except.toOption] at h
      exact ⟨w, rfl, by rw [hi, h.2]⟩

variable (h : Held)

/-- **The session's registered reads are the run's**: at a run's statement and template `t`'s session statement as the
verifier's check passes it, the session statement sets up as a masked Flock statement, its registered ports are Flock's
reading of `h.registered` for `t` at the run's roots and the interior registration's, and every statement of its setup
passes Flock's registered check against them. -/
theorem checkRegistered_of_stmtOkAt {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} {t : ℕ} {s : AuditStatement}
    (hx : parse h x = some s) (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) :
    ∃ it ins spec sts own, interiorsOf m₁ = some it ∧ FlockDeploy.setupOf true xS = some (ins, spec, sts) ∧
      ins.registered = some own ∧
      (FlockDeploy.jsonOf "registered reads" (h.registered t s.roots it.roots) >>=
        _root_.Flock.Registered.ofJson) = .ok own ∧
      ∀ st ∈ sts, _root_.Flock.checkRegistered st (some own) = .ok () := by
  have hr : readsOk h t s m₁ xS c₀ c₁ = true := by
    simp only [stmtOkAt, hx, Bool.and_eq_true] at hok
    exact hok.2
  unfold readsOk at hr
  split at hr
  · rename_i it draw ins spec sts w hit _ hset hdec
    simp only [Bool.and_eq_true, beq_iff_eq] at hr
    obtain ⟨⟨⟨-, -⟩, hreg⟩, -⟩ := hr
    obtain ⟨hins, hsetup⟩ := FlockDeploy.setupOf_some hset
    have hwi : w.ins = .ok ins := by
      obtain ⟨w', hw', hi⟩ := insOf_ok hins
      rw [hdec] at hw'
      cases hw'
      exact hi
    obtain ⟨b, hb, hbd⟩ := Option.map_eq_some_iff.1 hreg
    have hbe : b = h.registered t s.roots it.roots := by
      calc b = ⟨b.data⟩ := rfl
        _ = ⟨(h.registered t s.roots it.roots).data⟩ := by rw [hbd]
        _ = h.registered t s.roots it.roots := rfl
    subst hbe
    obtain ⟨own, hown, hj⟩ := ins_registered hwi hb
    refine ⟨it, ins, spec, sts, own, hit, hset, hown, hj, fun st hst => ?_⟩
    rw [← hown]
    exact setup_checkRegistered hsetup st hst
  · simp at hr

/-- **Each row the session reads from a registered port opens under the run's ports**: at a run's statement and
template `t`'s session statement as the verifier's check passes it, for every statement of the session's setup and every
port of the run's registered reads, the circuit has the port by name, and each instance's table row opens the port's
commitment at the position the port reads for the instance's unit, through the path the session header gives for that
row. -/
theorem opens_of_stmtOkAt {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} {t : ℕ} {s : AuditStatement}
    (hx : parse h x = some s) (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) :
    ∃ it ins spec sts own, interiorsOf m₁ = some it ∧ FlockDeploy.setupOf true xS = some (ins, spec, sts) ∧
      ins.registered = some own ∧
      (FlockDeploy.jsonOf "registered reads" (h.registered t s.roots it.roots) >>=
        _root_.Flock.Registered.ofJson) = .ok own ∧
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
  obtain ⟨it, ins, spec, sts, own, hit, hset, hown, hj, hck⟩ := checkRegistered_of_stmtOkAt h hx hok
  refine ⟨it, ins, spec, sts, own, hit, hset, hown, hj, fun st hst => ?_⟩
  obtain ⟨tb, idx, ht, hidx, hp⟩ := FlockSoundness.Discharge.ZkReg.check_tables
    (FlockSoundness.Discharge.ZkReg.checkRegistered_ok (hck st hst))
  exact ⟨tb, idx, ht, hidx, fun reg hreg => FlockSoundness.Discharge.ZkReg.checkPort_at (hp reg hreg)⟩

end SpotFlock
