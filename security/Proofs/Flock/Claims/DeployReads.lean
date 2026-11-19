module

public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Verifier.Hash
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.Soundness.Discharge.Layout.OutShape
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Soundness.ExecCheck

/-! What the deployed verifier's reading and setup of one statement establish, of any statement it reads. -/

@[expose] public section

set_option Elab.async false

namespace FlockDeploy

theorem Raw.ins_publicInputs {w : Raw} {s : Ins} (h : w.ins = .ok s) : s.read.publicInputs = w.publicInputs := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

open Flock

section Walk

open FlockSoundness.Refine FlockSoundness.Discharge.Exec

-- `walk_step` down `Stmt.setupTables`' unfolded do-block, a goal per branch
set_option maxHeartbeats 4000000 in
/-- **A session has at most `Tags.MAX_TABLES` tables, and its spec names one per statement**, at any public input
words. -/
theorem setupTables_le {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray} {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure publicInputs =
      .ok (spec, sts)) :
    sts.size ≤ Tags.MAX_TABLES ∧ spec.tables.size = sts.size := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals
    first
    | (simp [Stmt.spec, Tags.MAX_TABLES]; done)
    | (obtain ⟨hsz, -⟩ := range_mapM_ok (by assumption)
       refine ⟨by omega, ?_⟩
       simp [Tags.tableNames, show ¬ count ≤ 1 by omega, hsz])

end Walk

open FlockComplete

/-- **SHA-512's outputs are 64 bytes.** -/
theorem out64_sha512 : Claims.Out64 sha512 := fun m => by
  show (Claims.toMsg (Sha512.hash (Claims.ofMsg m))).length = 64
  have h := Sha512.hash_size (Claims.ofMsg m)
  generalize Sha512.hash (Claims.ofMsg m) = b at h ⊢
  cases b
  unfold Claims.toMsg
  rw [Array.length_toList]
  exact h

end FlockDeploy

namespace RecDeploy.ResLower

open Flock

/-- `FlockDeploy.Raw.ins` reads its tags from `--statement`'s name, and C's are them. -/
theorem raw_ins_tags {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    Tags.byName? w.statement = some s.stmtTags ∧ s.read.I.tags = s.stmtTags := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨by assumption, rfl⟩)

/-- **Every output word is returned**: in an untyped circuit whose unit returns the outputs, `HmRow.check`'s `seen`
loop marks each output word its copies' leaf maps return (`check_returned`'s invariant), and the check refuses a word
left unmarked ("an output leaf no unit returns"). -/
theorem check_complete {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) {o : ℕ} (hoW : o < c.outWords) :
    o ∈ FlockSoundness.Discharge.HiddenExec.returned c (List.range c.upv) := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  replace h := FlockSoundness.ite_join h fun ha => (FlockSoundness.Discharge.Layout.ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u2, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨wired, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨wired2, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u3, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨seen, hseen, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨hany, -⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  have hI := FlockSoundness.Discharge.HiddenExec.forIn_range_inv
    (fun pre s => FlockSoundness.Discharge.HiddenExec.SeenInv c
      (FlockSoundness.Discharge.HiddenExec.returned c pre) s)
    (fun pre a s t hs hst => by
      obtain ⟨-, hst⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond hst
      obtain ⟨s', hin, hst⟩ := FlockSoundness.except_bind_ok hst
      cases hst
      refine ⟨_, rfl, ?_⟩
      have e0 : (default : Array ℕ) = #[] := rfl
      rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD', e0, ← Array.forIn_toList] at hin
      have := FlockSoundness.Discharge.HiddenExec.forIn_list_inv
        (fun pre' s => FlockSoundness.Discharge.HiddenExec.SeenInv c
          (FlockSoundness.Discharge.HiddenExec.returned c pre ++ pre') s) (fun pre' o s t hs hst => by
          obtain ⟨hc, hst⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond hst
          cases hst
          refine ⟨_, rfl, ?_⟩
          rw [← List.append_assoc]
          exact FlockSoundness.Discharge.HiddenExec.seenInv_step hs hc) _ [] s s' (by simpa using hs) hin
      simpa [FlockSoundness.Discharge.HiddenExec.returned, List.flatMap_append] using this)
    (by
      refine ⟨Array.size_replicate, fun o h1 => ?_, List.nodup_nil, fun o h => by cases h⟩
      rw [FlockSoundness.Discharge.HiddenExec.bool_get, Array.getElem?_replicate, ho, bne_self_eq_false] at h1
      split at h1 <;> cases h1) hseen
  have hlt : o < seen.size := by rw [hI.1]; exact hoW
  refine hI.2.1 o ?_
  by_contra hf
  rw [getElem!_pos seen o hlt] at hf
  exact hany (Array.any_eq_true.2 ⟨o, hlt, by simpa using hf⟩)

end RecDeploy.ResLower

namespace RecDeploy

open Flock

/-- **A session with no public inputs that sets up is in range**: one to `64` tables, two link points, and each
table's `k_log`, regions and link points within `checkInRange`'s range. -/
theorem read_setup_in {r : Flock.Readable.Read} {spec : SessionSpec} {sts : Array Stmt}
    (h : r.setup = .ok (spec, sts)) (hpi : r.publicInputs = none) :
    0 < sts.size ∧ sts.size ≤ 64 ∧ spec.points = 2 ∧ ∀ st ∈ sts.toList, (FlockDeploy.acct spec st).InRange := by
  have hle := (FlockDeploy.setupTables_le h).1
  have hmax : Flock.Tags.MAX_TABLES = 64 := rfl
  unfold Flock.Readable.Read.setup at h
  rw [hpi] at h
  obtain ⟨h0, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne h
  obtain ⟨hp2, hpts⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny h
  refine ⟨h0, by omega, hp2, fun st hst => ?_⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  obtain ⟨d, hd, -⟩ := hall j hj
  obtain ⟨hr, hwf⟩ := FlockDeploy.KS.setupOne_range hd
  obtain ⟨hk, hreg, h64⟩ := (Flock.checkInRange_ok _ _ _).1 hr
  have hm := hwf.mPts
  have hpj := hpts j hj
  show sts[j].c.kLog ≤ 27 ∧ sts[j].regions.size ≤ 1024 ∧ spec.mPts ≤ 64
  exact ⟨hk, hreg, by omega⟩

end RecDeploy

namespace Verity.Recursive.Numbers

open Flock RecDeploy FlockSoundness.Refine

section Walks

variable {tags : Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray}
  {spec : SessionSpec} {sts : Array Flock.Stmt}

/-- A setup followed by a refusal guard (`setupTables`' refusal of pinned outputs) passes only the setup's own table. -/
theorem guard_ok {α : Type} {x : Flock.V α} {p : α → Bool} {e : α → String} {w : α}
    (h : (x >>= fun st => if p st = true then throw (e st) else pure st) = .ok w) : x = .ok w := by
  cases x with
  | error => cases h
  | ok a =>
    change (if p a = true then throw (e a) else pure a : Flock.V α) = .ok w at h
    split at h
    · first | (simp [throw, throwThe, MonadExceptOf.throw] at h; done) | cases h
    · cases h
      rfl

set_option maxHeartbeats 4000000 in
/-- Under hidden outputs, `setupTables` at any public input words is `setupHidden`'s one table, or several. -/
theorem setupTables_hidden (hh : tags.hiddenOutputs = true)
    (h : Flock.Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure
      publicInputs = .ok (spec, sts)) :
    (∃ st, Flock.Stmt.setupHidden tags cf pf coins tables draw partition program publicInputs = .ok st ∧
        spec = st.spec ∧ sts = #[st]) ∨ (1 < count ∧ sts.size = count) := by
  unfold Flock.Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals
    first
    | exact Or.inl ⟨_, by assumption, rfl, rfl⟩
    | exact Or.inl ⟨_, guard_ok (by assumption), rfl, rfl⟩
    | exact absurd hh ‹_›
    | (obtain ⟨hsz, -⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok (by assumption)
       exact Or.inr ⟨by omega, hsz⟩)

set_option maxHeartbeats 4000000 in
/-- What `setupHidden`'s table is made of, at any public input words: its circuit, its population and sample, the unit
count `n` the shape reads, and its regions. -/
theorem setupHidden_shape {st : Flock.Stmt}
    (h : Flock.Stmt.setupHidden tags cf pf coins tables draw partition program publicInputs = .ok st) :
    ∃ pi pop A B, HmOut.parseCommit tags cf tables = .ok st.c ∧ HmIn.parse st.c = .ok pi ∧
      HmIn.loadPublic tags st.c pi pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
      (∀ d, draw = some d → HmIn.drawn st.c pi pop d = .ok st.pub) ∧ st.n = st.pub.n ∧
      st.m = st.c.kLog + max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) ∧
      st.mPts = PT_LOCAL + max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) ∧
      HmOut.regions st.c st.pub = .ok A ∧ HmIn.regions st.c pi st.pub = .ok B ∧ st.regions = A ++ B := by
  cases draw <;> unfold Flock.Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, _, _, _, by assumption, by assumption, by assumption, fun hd => ?_, fun d' hd => ?_, rfl, rfl, rfl,
      by assumption, by assumption, rfl⟩
    · first | rfl | cases hd
    · first | (cases hd; assumption) | cases hd

/-- `HmIn.loadPublic`'s population is `HmRow.loadPublic`'s, at the circuit with its public input row. -/
theorem hmIn_loadPublic_row {c : Flock.Circuit} {pi : HmIn.Ports} {f : ByteArray} {pop : Public}
    (h : HmIn.loadPublic tags c pi f = .ok pop) : ∃ c', HmRow.loadPublic tags c' f = .ok pop := by
  unfold HmIn.loadPublic at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, by assumption⟩

/-- `HmIn.regions` returns one region per public input block, a count of the circuit's alone. -/
theorem hmIn_regions_size {c : Flock.Circuit} {pi : HmIn.Ports} {pub : Public} {B : Array Region}
    (h : HmIn.regions c pi pub = .ok B) : B.size = if pi.ports.isEmpty then 0 else (HmIn.inRegions c pi).size := by
  unfold HmIn.regions at h
  dsimp only at h
  split at h
  · rename_i he
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp [he]
  · rename_i he
    rw [if_neg he]
    exact (FlockSoundness.Discharge.Exec.range_mapM_ok h).1

end Walks

/-- The verifier's readers keep the statement's circuit, table count and tables. -/
theorem raw_ins_files {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    s.read.I.circuitFile = w.circuit ∧ s.read.I.count = w.count ∧ s.read.I.tables = w.tables := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨rfl, rfl, rfl⟩)

/-- A `--zk` statement's tags are its statement's under the circuit's mask and its table count. -/
theorem read_tags_of {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) (hz : w.zk = true) :
    s.read.tags = Zk.tags s.stmtTags (Zk.maskLogOf w.circuit) w.count := by
  obtain ⟨-, hT⟩ := ResLower.raw_ins_tags h
  obtain ⟨hc, hn, -⟩ := raw_ins_files h
  have hz' : s.read.zk = true := (FlockDeploy.Raw.ins_zk h).trans hz
  unfold Flock.Readable.Read.tags
  rw [if_pos hz', FlockSoundness.Discharge.ZkExec.zkTags_eq, hT, hc, hn]

theorem rounds_one (zk : Bool) (spec : SessionSpec) (t : Flock.Stmt) :
    FlockDeploy.rounds zk spec #[t] = 2 * FlockDeploy.repRounds zk spec t := by
  simp [FlockDeploy.rounds]

end Verity.Recursive.Numbers
