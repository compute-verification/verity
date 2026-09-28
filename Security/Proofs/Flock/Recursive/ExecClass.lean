import Proofs.Flock.Recursive.Universal
import Proofs.Flock.Recursive.VStarU2

/-!
# Recursion: the class, regions and schedule of an executable statement

`RecursiveAudit` takes the inner statement `st : Flock.Stmt` itself, and its class, regions and schedule are a total
function of it (`execClassAt`), so the class's fields are no hypotheses and the compile premise is a lemma.

* `shapeOf st`: the fields the class reads, the statement with its values dropped. Kept: `m`, the circuit, the row
  leaf, the pin, Δ, the link points, the blocks, and each region's name, free bits, mask and fixed bits. Dropped: the
  regions' bytes (the registration's commit strings, instance data), the public file, the digest, σ, `Hello`, the tags,
  the Merkle scheme, the partition finding and the counts `n` and `blocksReal`.
* `ClassOk st`: the executable's checks of the class: `StmtWF`, `RegionsWF` and `Zk.schedule st.m` accepting.
* `execClassAt st`: when `ClassOk (shapeOf st)`, `HClass.ofStmt` at no hidden entries (the universal unit's class,
  the one public statement: `HClass.ofStmt_toStatement`), the statement's regions (`Refine.regionsOf`) and its
  `fast100` schedule (`ExecClass.ofStmt`); otherwise the constant `ExecClass.fixed`.
* `AcceptsStmt st`: Lean's verifier accepts the statement: `Stmt.setupH` returns it and `Zk.schedule` accepts its `m`.
  `classOk_of_accepts`: then its class checks pass, so an accepted statement's class is its own
  (`execClassAt_of_accepts`), never the fixed one.
* `compiledForkSound_execClassAt`: the compile premise at `execClassAt st` when its checks pass
  (`compiledForkSound_of_fast100`, from `flock_inner_sound_compiled`).
* Class-fixed regions: the class (`execClassAt_cls`), the number of regions (`execClassAt_regs_length`) and the
  schedule (`execClassAt_sch`) are functions of `shapeOf st`, and so is each region's place in the block, its free and
  fixed bits (`regionsOf_layout`); two statements of one shape differ only in the regions' bytes (`execClassAt_shape`).
* `VStmt.exec`, `execStmts`: a statement with its model read off its executable strata where they tile its units (its
  live law unchanged), and `VStmt.exec_execStrata`, `_hS` there. `nonempty_reg₂`: any prover of the recursive audit
  sends some registration of V*'s sessions, so there is a session of each of V*'s statements whose draw file
  (`VStmt.tiles_of_session`) gives the tiling.
-/

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
  Refine.StmtWF st ∧ Refine.RegionsWF st ∧ ∃ p, Flock.Zk.schedule st.m = .ok p

/-- **Lean's verifier accepts `st` as the inner statement**: its setup on the hm96 path (`Flock.Stmt.setupH`) returns
`st` from some circuit and public files, and `Flock.Zk.schedule` accepts its `m`. -/
def AcceptsStmt (st : Flock.Stmt) : Prop :=
  (∃ (tags : Flock.Tags) (cf pf : ByteArray) (coins : Flock.Coins) (tables : List (String × ByteArray))
      (draw partition : Option Lean.Json) (program : Option ByteArray),
    Flock.Stmt.setupH tags cf pf coins tables draw partition program = .ok st) ∧
  ∃ p, Flock.Zk.schedule st.m = .ok p

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
  obtain ⟨hwf, hr, hp⟩ := h
  refine ⟨⟨hwf.hk6, hwf.hk, hwf.pin, hwf.slots⟩, ⟨hr.hm, hr.mPts, fun reg hreg => ?_⟩, hp⟩
  have hmem : stripBytes reg ∈ (shapeOf st).regions.toList := by
    rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map]
    exact List.mem_map.2 ⟨reg, hreg, rfl⟩
  have w := hr.regions _ hmem
  exact ⟨w.free_lt, w.nodup, w.free_size⟩

theorem ClassOk.to_shape {st : Flock.Stmt} (h : ClassOk st) : ClassOk (shapeOf st) := by
  obtain ⟨hwf, hr, hp⟩ := h
  refine ⟨⟨hwf.hk6, hwf.hk, hwf.pin, hwf.slots⟩, ⟨hr.hm, hr.mPts, fun reg hreg => ?_⟩, hp⟩
  rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map] at hreg
  obtain ⟨r, hr', rfl⟩ := List.mem_map.1 hreg
  have w := hr.regions r hr'
  exact ⟨w.free_lt, w.nodup, w.free_size⟩

/-- **Lean's verifier accepting a statement passes its class checks**: `setupH`'s statements are well formed
(`Refine.setupH_wf`), and `Zk.schedule` is the acceptance's own. -/
theorem classOk_of_accepts {st : Flock.Stmt} (h : AcceptsStmt st) : ClassOk (shapeOf st) := by
  obtain ⟨⟨_, _, _, _, _, _, _, _, hs⟩, hp⟩ := h
  obtain ⟨hwf, hr⟩ := Refine.setupH_wf hs
  exact ClassOk.to_shape ⟨hwf, hr, hp⟩

/-- **An accepted statement's class is its own**: `execClassAt` takes the `ExecClass.ofStmt` branch. -/
theorem execClassAt_of_accepts {st : Flock.Stmt} (h : AcceptsStmt st) :
    execClassAt st = ExecClass.ofStmt st (classOk_of_accepts h) :=
  dif_pos _

/-- **The compile premise at a statement's class** when its checks pass: `compiledForkSound_of_fast100` at its
`fast100` schedule, with the link layout `Refine.stmtOf_linkLayout`'s. -/
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

/-! ### Class-fixed regions -/

open Classical in
/-- The class `execClassAt` picks, read off a shape. -/
noncomputable def clsOf (sh : Flock.Stmt) : HClass :=
  if h : ClassOk sh then HClass.ofStmt sh h.1 (hm_of_classOk h) 0 ∅ 0 else HClass.fixed

/-- **The class reads only the shape.** -/
theorem execClassAt_cls (st : Flock.Stmt) : (execClassAt st).cls = clsOf (shapeOf st) := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, clsOf, dif_pos h]
    rfl
  · rw [execClassAt, dif_neg h, clsOf, dif_neg h]
    rfl

open Classical in
/-- **The number of regions reads only the shape.** -/
theorem execClassAt_regs_length (st : Flock.Stmt) :
    (execClassAt st).regs.length = if ClassOk (shapeOf st) then (shapeOf st).regions.size else 0 := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, if_pos h]
    show (Refine.regionsOf st).length = (st.regions.map stripBytes).size
    simp [Refine.regionsOf]
  · rw [execClassAt, dif_neg h, if_neg h]
    rfl

open Classical in
/-- **The schedule reads only the shape.** -/
theorem execClassAt_sch (st : Flock.Stmt) :
    (execClassAt st).sch = if ClassOk (shapeOf st) then schedAt (shapeOf st).m else default := by
  by_cases h : ClassOk (shapeOf st)
  · rw [execClassAt, dif_pos h, if_pos h]
    rfl
  · rw [execClassAt, dif_neg h, if_neg h]
    rfl

/-- **Two statements of one shape have one class, as many regions and one schedule.** -/
theorem execClassAt_shape {st st' : Flock.Stmt} (h : shapeOf st = shapeOf st') :
    (execClassAt st).cls = (execClassAt st').cls ∧ (execClassAt st).regs.length = (execClassAt st').regs.length ∧
      (execClassAt st).sch = (execClassAt st').sch := by
  rw [execClassAt_regs_length, execClassAt_regs_length, execClassAt_sch, execClassAt_sch, execClassAt_cls,
    execClassAt_cls, h]
  exact ⟨rfl, rfl, rfl⟩

/-- **Each region's place reads only the shape**: the regions of `st` and of its shape have the same free and fixed
bits, one by one; they differ only in their bytes, the registration's commit strings. -/
theorem regionsOf_layout (st : Flock.Stmt) :
    (Refine.regionsOf st).map (fun R => (R.free, R.fixedBits)) =
      (Refine.regionsOf (shapeOf st)).map (fun R => (R.free, R.fixedBits)) := by
  unfold Refine.regionsOf
  rw [show (shapeOf st).regions = st.regions.map stripBytes from rfl, Array.toList_map]
  simp only [List.map_map]
  rfl

/-! ### The model read off V*'s strata -/

open Classical in
/-- **A law with its model read off its executable strata** where they tile its units (`VLaw.ofExec`'s: each unit's
stratum the one holding it, each stratum's draw count its own), its own model otherwise. Its live law's fields are its
own. -/
noncomputable def VLaw.exec {nv : ℕ} (law : VLaw nv) : VLaw nv :=
  { law with
    σs := if h : Law.Tiles law.St nv then h.stratumOf else law.σs
    ks := if Law.Tiles law.St nv then fun s => (law.St s).k else law.ks
    hks := by
      split_ifs with h
      · intro s
        rw [h.card_stratum]
        exact h.k_le s
      · exact law.hks }

/-- **A statement with its law's model read off its executable strata**: the same circuit and live law. -/
noncomputable def VStmt.exec (Q : VStmt) : VStmt := { Q with law := Q.law.exec }

/-- **V*'s statements with their models read off their executable strata.** -/
noncomputable abbrev execStmts {S : ℕ} (Vs : Fin S → VStmt) : Fin S → VStmt := fun j => (Vs j).exec

open Classical in
/-- **`_hS` at a statement whose strata tile its units**: its model read off them is theirs. -/
theorem VStmt.exec_execStrata (Q : VStmt) (h : Law.Tiles Q.exec.law.St Q.exec.nv) :
    Law.ExecStrata Q.exec.law.σs Q.exec.law.ks Q.exec.law.St := by
  have h' : Law.Tiles Q.law.St Q.nv := h
  refine (VLaw.execStrata_iff Q.exec.law).2 ⟨h, ?_, ?_⟩
  · show (if h : Law.Tiles Q.law.St Q.nv then h.stratumOf else Q.law.σs) = _
    rw [dif_pos h']
    exact rfl
  · show (if Law.Tiles Q.law.St Q.nv then fun s => (Q.law.St s).k else Q.law.ks) = _
    rw [if_pos h']
    exact rfl

/-- A strategy of the rounds reaches a leaf. -/
theorem rounds_leaf {M C β : Type} [Fintype C] [Nonempty C] :
    ∀ (r : ℕ) (f : List (M × C) → Game β), Strategy (rounds M C r f) → ∃ t, Nonempty (Strategy (f t))
  | 0, _, s => ⟨[], ⟨s⟩⟩
  | r + 1, f, s => by
      obtain ⟨t, ht⟩ := rounds_leaf r (fun t => f ((s.1, Classical.arbitrary C) :: t)) (s.2 (Classical.arbitrary C))
      exact ⟨_, ht⟩

/-- **A prover of the recursive audit reaches an outer session**: at some draw and transcript, it has a strategy
there. -/
theorem exists_outer {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm β : Type}
    {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β} (σ : Strategy (recGame L Reg I Cm outer)) :
    ∃ ω t, Nonempty (Strategy (outer σ.1 ω t)) := by
  obtain ⟨t, ⟨s⟩⟩ := rounds_leaf I.r _ (σ.2 (Classical.arbitrary L.Ω))
  exact ⟨_, t, ⟨Strategy.ofMap _ _ s⟩⟩

/-- **A prover of the recursive audit sends a registration of V*'s sessions.** -/
theorem nonempty_reg₂ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N : ℕ}
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Nonempty Reg₂ := by
  obtain ⟨_, _, ⟨s⟩⟩ := exists_outer σ
  exact ⟨s.1⟩

end FlockSoundness.Discharge.Recursion
