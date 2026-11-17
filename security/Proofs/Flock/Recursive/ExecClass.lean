module

public import Proofs.Flock.Recursive.Universal
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Setup

/-! The inner statement's class, regions and schedule as a total function of it (`execClassAt`), so the class's fields are no hypotheses and the compile premise is a lemma. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit

/-- A region with its bytes dropped: its name, free bits, mask and fixed bits. -/
def stripBytes (r : Flock.Region) : Flock.Region := { r with bytes := fun _ => ByteArray.empty }

/-- **A statement's shape**: the fields its class and its regions' places read, every value dropped. -/
def shapeOf (st : Flock.Stmt) : Flock.Stmt :=
  { (default : Flock.Stmt) with
    leaf := st.leaf, c := st.c, nbl := st.nbl, m := st.m, mPts := st.mPts, pin := st.pin,
    regions := st.regions.map stripBytes, da := st.da, db := st.db }

/-- **The executable's checks of a statement's class**: its layout (`StmtWF`), its regions (`RegionsWF`), and
`Zk.schedule` accepting its `m`. -/
def ClassOk (st : Flock.Stmt) : Prop :=
  Refine.StmtWF st ∧ Refine.RegionsWF st ∧ ∃ p, Flock.Zk.schedule st.wideLeaf st.m = .ok p

/-- **Lean's verifier accepts `st` as the inner statement**, set up as `RecVStar.innerOf` and `RecCoins.Inner.load` set
it up: from some circuit and public files, partition object and program (`innerOf` passes the verifier's own, and
`Stmt.setupHidden` refuses a statement whose META has `links` without one; `Inner.load` passes none),
`Flock.Stmt.setupTables` at the inner tags (`RecVStar.innerTags`) and the challenger's coin tree (`RecVStar.innerTree`)
returns one table, `st`, and `Flock.Zk.schedule` accepts its `m`. -/
def AcceptsStmt (st : Flock.Stmt) : Prop :=
  (∃ (cf pf : ByteArray) (partition : Option Lean.Json) (program : Option ByteArray) (spec : Flock.SessionSpec),
    Flock.Stmt.setupTables Flock.RecVStar.innerTags cf pf Flock.RecVStar.innerTree 1 (partition := partition)
      (program := program) = .ok (spec, #[st])) ∧
  ∃ p, Flock.Zk.schedule st.wideLeaf st.m = .ok p

theorem innerTags_hiddenOutputs : Flock.RecVStar.innerTags.hiddenOutputs = true := rfl

theorem innerTags_merkleLeaf : Flock.RecVStar.innerTags.merkleLeaf = "flock-leaf/sha512-unsalted" := rfl

theorem acceptsStmt_merkle {st : Flock.Stmt} (h : AcceptsStmt st) :
    st.merkle = Flock.MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
  obtain ⟨⟨_, _, _, _, _, hs⟩, -⟩ := h
  rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hs with ⟨-, st', hst, -, hsts⟩ | ⟨hc, -⟩
  · obtain rfl : st = st' := by simpa using hsts
    have s := Hidden.setupHidden_spec' hst
    have hf := s.merkle
    rw [Hidden.parse_leafScheme (s.parseF (by decide)), innerTags_merkleLeaf,
      show Flock.MerkleScheme.find? "flock-leaf/sha512-unsalted" =
        some (Flock.MerkleScheme.sha512 "flock-leaf/sha512-unsalted") from rfl] at hf
    exact (Option.some.inj hf).symm
  · omega

/-- **A class, its regions and its schedule**, the compiled table's parameters. -/
structure ExecClass where
  cls : HClass
  regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog))
  sch : Accounting.Schedule

/-- **The fixed class**, a constant: 13 variables, `k_log` 6, zero matrices, no hidden entries. -/
def HClass.fixed : HClass where
  m := 13
  kLog := 6
  hk := by decide
  hk6 := le_rfl
  hm := le_rfl
  pin := ⟨0, by decide⟩
  A₀pub := 0
  B₀pub := 0
  slotLog := 0
  units := ∅
  ptLocal := Flock.PT_LOCAL
  mPts := Flock.PT_LOCAL + 7
  nnz := 0

/-- The fixed class with no regions and the default schedule. -/
def ExecClass.fixed : ExecClass := ⟨HClass.fixed, [], default⟩

/-- `fast100`'s schedule at `m`, as the model's; the default where `fast100` has none. -/
def schedAt (m : ℕ) : Accounting.Schedule :=
  match Flock.fast100 m with
  | some s₀ => Refine.schedOf s₀
  | none => default

theorem hm_of_classOk {st : Flock.Stmt} (h : ClassOk st) : 13 ≤ st.m :=
  h.2.2.elim fun _ hp => hm_of_schedule hp

/-- **The class of a statement whose checks pass**: `HClass.ofStmt` of its shape at no hidden entries, its regions
and its `fast100` schedule. -/
def ExecClass.ofStmt (st : Flock.Stmt) (h : ClassOk (shapeOf st)) : ExecClass where
  cls := HClass.ofStmt (shapeOf st) h.1 (hm_of_classOk h) 0 ∅ 0
  regs := Refine.regionsOf st
  sch := schedAt st.m

open Classical in
/-- **The class, regions and schedule of an executable statement**, a total function of it: `ExecClass.ofStmt` when
the executable's checks of its shape pass, the fixed class otherwise. -/
noncomputable def execClassAt (st : Flock.Stmt) : ExecClass :=
  if h : ClassOk (shapeOf st) then ExecClass.ofStmt st h else ExecClass.fixed

theorem ClassOk.of_shape {st : Flock.Stmt} (h : ClassOk (shapeOf st)) : ClassOk st := by
  obtain ⟨hwf, hr, p, hp⟩ := h
  have hm : 25 ≤ st.m ∧ st.m ≤ 35 := ZkHidden.schedule_range hp
  refine ⟨⟨hwf.hk6, hwf.hk, hwf.pin, hwf.slots⟩, ⟨hr.hm, hr.mPts, fun reg hreg => ?_⟩,
    ZkHidden.schedule_ok_of_range _ hm.1 hm.2⟩
  have hmem : stripBytes reg ∈ (shapeOf st).regions.toList := by
    rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map]
    exact List.mem_map.2 ⟨reg, hreg, rfl⟩
  have w := hr.regions _ hmem
  exact ⟨w.free_lt, w.nodup, w.free_size⟩

theorem ClassOk.to_shape {st : Flock.Stmt} (h : ClassOk st) : ClassOk (shapeOf st) := by
  obtain ⟨hwf, hr, p, hp⟩ := h
  have hm : 25 ≤ st.m ∧ st.m ≤ 35 := ZkHidden.schedule_range hp
  refine ⟨⟨hwf.hk6, hwf.hk, hwf.pin, hwf.slots⟩, ⟨hr.hm, hr.mPts, fun reg hreg => ?_⟩,
    ZkHidden.schedule_ok_of_range _ hm.1 hm.2⟩
  rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map] at hreg
  obtain ⟨r, hr', rfl⟩ := List.mem_map.1 hreg
  have w := hr.regions r hr'
  exact ⟨w.free_lt, w.nodup, w.free_size⟩

theorem classOk_of_accepts {st : Flock.Stmt} (h : AcceptsStmt st) : ClassOk (shapeOf st) := by
  obtain ⟨⟨_, _, _, _, _, hs⟩, hp⟩ := h
  rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hs with ⟨-, st', hst, -, hsts⟩ | ⟨hc, -⟩
  · obtain rfl : st = st' := by simpa using hsts
    obtain ⟨hwf, hr⟩ := Hidden.setupHidden_wf hst
    exact ClassOk.to_shape ⟨hwf, hr, hp⟩
  · omega

theorem execClassAt_of_accepts {st : Flock.Stmt} (h : AcceptsStmt st) :
    execClassAt st = ExecClass.ofStmt st (classOk_of_accepts h) :=
  dif_pos _

theorem compiledForkSound_execClassAt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (h : ClassOk (shapeOf st)) :
    Flock.Premises.CompiledForkSound T CT (execClassAt st).cls (execClassAt st).regs (execClassAt st).sch := by
  obtain ⟨hwf, hr, p, hp⟩ := ClassOk.of_shape h
  obtain ⟨s₀, hs₀⟩ := ZkHidden.fast100_of_schedule hp
  have hsch : Accounting.fast100 st.m = some (schedAt st.m) := by
    have e := Refine.fast100_schedOf st.m
    rw [hs₀] at e
    unfold schedAt
    rw [hs₀]
    exact e.symm
  unfold execClassAt
  rw [dif_pos h]
  exact compiledForkSound_of_fast100 T CT _ _ _ hsch fun _ => Refine.stmtOf_linkLayout st hwf hr

open Classical in
noncomputable def clsOf (sh : Flock.Stmt) : HClass :=
  if h : ClassOk sh then HClass.ofStmt sh h.1 (hm_of_classOk h) 0 ∅ 0 else HClass.fixed

theorem execClassAt_cls (st : Flock.Stmt) : (execClassAt st).cls = clsOf (shapeOf st) := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, clsOf, dif_pos h]
    rfl
  · rw [execClassAt, dif_neg h, clsOf, dif_neg h]
    rfl

open Classical in
theorem execClassAt_regs_length (st : Flock.Stmt) :
    (execClassAt st).regs.length = if ClassOk (shapeOf st) then (shapeOf st).regions.size else 0 := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, if_pos h]
    show (Refine.regionsOf st).length = (st.regions.map stripBytes).size
    simp [Refine.regionsOf]
  · rw [execClassAt, dif_neg h, if_neg h]
    rfl

theorem execClassAt_sch_of_fast100 {st : Flock.Stmt} (h : ClassOk (shapeOf st)) {s₀ : Flock.Schedule}
    (hs : Flock.fast100 st.m = some s₀) : (execClassAt st).sch = Refine.schedOf s₀ := by
  rw [execClassAt, dif_pos h]
  show schedAt st.m = _
  simp only [schedAt, hs]

open Classical in
theorem execClassAt_sch (st : Flock.Stmt) :
    (execClassAt st).sch = if ClassOk (shapeOf st) then schedAt (shapeOf st).m else default := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, if_pos h]
    rfl
  · rw [execClassAt, dif_neg h, if_neg h]
    rfl

theorem execClassAt_shape {st st' : Flock.Stmt} (h : shapeOf st = shapeOf st') :
    (execClassAt st).cls = (execClassAt st').cls ∧ (execClassAt st).regs.length = (execClassAt st').regs.length ∧
      (execClassAt st).sch = (execClassAt st').sch := by
  rw [execClassAt_regs_length, execClassAt_regs_length, execClassAt_sch, execClassAt_sch, execClassAt_cls,
    execClassAt_cls, h]
  exact ⟨rfl, rfl, rfl⟩

theorem regionsOf_layout (st : Flock.Stmt) :
    (Refine.regionsOf st).map (fun R => (R.free, R.fixedBits)) =
      (Refine.regionsOf (shapeOf st)).map (fun R => (R.free, R.fixedBits)) := by
  unfold Refine.regionsOf
  rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map]
  simp only [List.map_map]
  rfl

end FlockSoundness.Discharge.Recursion
