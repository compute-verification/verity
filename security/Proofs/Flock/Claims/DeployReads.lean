module

public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Verifier.Hash
public import Spot.OneStageZkStatement
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
import all Lean.Data.Json.Basic

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

namespace FlockDeploy.BaseSetup

open Lean (Json)

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

theorem forIn_list_ok {α : Type} {l : List α} {f : α → PUnit → Flock.V (ForInStep PUnit)}
    (h : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit)) : forIn l PUnit.unit f = .ok PUnit.unit := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.forIn_cons, h a List.mem_cons_self]
    exact ih fun b hb => h b (List.mem_cons_of_mem _ hb)

/-- A `for` loop over an array whose every step yields is `.ok`. -/
theorem forIn_ok {α : Type} {xs : Array α} {f : α → PUnit → Flock.V (ForInStep PUnit)}
    (h : ∀ a ∈ xs, f a PUnit.unit = .ok (.yield PUnit.unit)) : forIn xs PUnit.unit f = .ok PUnit.unit := by
  rw [← Array.forIn_toList]
  exact forIn_list_ok fun a ha => h a (Array.mem_toList_iff.1 ha)

theorem forIn_list_mem {α : Type} {l : List α} {f : α → PUnit → Flock.V (ForInStep PUnit)}
    (hy : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn l PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  induction l with
  | nil => intro a ha; cases ha
  | cons a l ih =>
    rw [List.forIn_cons] at h
    rcases hf : f a PUnit.unit with e | s
    · rw [hf] at h; cases h
    · have hs := hy a s hf
      subst hs
      rw [hf] at h
      intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hf
      · exact ih h b hb

/-- A `.ok` `for` loop over an array, whose steps only yield, yields at each element. -/
theorem forIn_mem {α : Type} {xs : Array α} {f : α → PUnit → Flock.V (ForInStep PUnit)}
    (hy : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn xs PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ xs, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  rw [← Array.forIn_toList] at h
  exact fun a ha => forIn_list_mem hy h a (Array.mem_toList_iff.2 ha)

/-- A loop step `do A a; B a; pure (.yield ())` yields exactly when both calls are `.ok`. -/
theorem body_ok {α : Type} {A B : α → Flock.V Unit} {a : α} :
    (do A a; B a; pure (ForInStep.yield PUnit.unit) : Flock.V (ForInStep PUnit)) = .ok (.yield PUnit.unit) ↔
      A a = .ok () ∧ B a = .ok () := by
  rcases hA : A a with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
    simp [bind, Except.bind, pure, Except.pure]

/-- A loop step `do C a (← W a); B a; pure (.yield ())` yields exactly when `W a >>= C a` and `B a` are `.ok`. -/
theorem body_bind_ok {α β : Type} {W : α → Flock.V β} {C : α → β → Flock.V Unit} {B : α → Flock.V Unit} {a : α} :
    (do C a (← W a); B a; pure (ForInStep.yield PUnit.unit) : Flock.V (ForInStep PUnit)) = .ok (.yield PUnit.unit) ↔
      (W a >>= C a) = .ok () ∧ B a = .ok () := by
  rcases hW : W a with e | r
  · simp [bind, Except.bind]
  · rcases hC : C a r with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
      simp [bind, Except.bind, pure, Except.pure, hC, hB]

theorem body_bind_yield {α β : Type} {W : α → Flock.V β} {C : α → β → Flock.V Unit} {B : α → Flock.V Unit} :
    ∀ a s, (do C a (← W a); B a; pure (ForInStep.yield PUnit.unit) : Flock.V (ForInStep PUnit)) = .ok s →
      s = .yield PUnit.unit := by
  intro a s h
  rcases hW : W a with e | r
  · simp [hW, bind, Except.bind] at h
  · rcases hC : C a r with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
      simp [hW, hC, hB, bind, Except.bind, pure, Except.pure] at h
    exact h.symm

theorem body_yield {α : Type} {A B : α → Flock.V Unit} :
    ∀ a s, (do A a; B a; pure (ForInStep.yield PUnit.unit) : Flock.V (ForInStep PUnit)) = .ok s →
      s = .yield PUnit.unit := by
  intro a s h
  rcases hA : A a with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
    simp [hA, hB, bind, Except.bind, pure, Except.pure] at h
  exact h.symm

/-- The compiled `checkIdentifier` (no `--identifier`) passing is the model's `noIdentifier` passing: both pass exactly
when the public header names no identifier, and the compiled one refuses every header that names one. -/
theorem noIdentifier_of_check {st : Flock.Stmt} (h : Spot.OneStageZk.checkIdentifier st = .ok ()) :
    FlockDeploy.noIdentifier st = .ok () := by
  unfold Spot.OneStageZk.checkIdentifier at h
  unfold FlockDeploy.noIdentifier
  rcases hj : st.pub.header.getObjVal? "identifier" with e | j
  · rfl
  · exfalso
    rw [hj] at h
    cases j
    all_goals try (simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h; done)
    next kvs =>
      cases hd : Flock.Program.domainError (.obj kvs) "identifier" <;>
        simp [hd, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- A field `Flock.J.arr` reads as an array is that array, as `Spot.getD` reads it. -/
theorem getD_of_arr {j : Json} {k ctx : String} {a : Array Json} (h : Flock.J.arr j k ctx = .ok a) :
    _root_.Spot.getD j k = .arr a := by
  unfold Flock.J.arr at h
  obtain ⟨v, hv, h⟩ := bind_ok h
  have hg : j.getObjVal? k = .ok v := by
    unfold Flock.J.get at hv
    cases hj : j.getObjVal? k with
    | error e => rw [hj] at hv; cases hv
    | ok w => rw [hj] at hv; cases hv; rfl
  have hva : v = .arr a := by
    try dsimp only at h
    unfold Flock.J.asArr at h
    cases v <;> first
      | (cases h; rfl)
      | cases h
  rw [← hva]
  simp only [_root_.Spot.getD, _root_.Spot.field?, hg, Except.toOption, Option.getD_some]

end FlockDeploy.BaseSetup
