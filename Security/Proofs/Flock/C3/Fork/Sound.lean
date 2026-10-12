import Proofs.Flock.C3.Fork.Runs
import Proofs.Flock.Recursive.Universal

/-!
# Piece (a): candidate 3's table is sound, in fork form

`CompiledForkSound`'s analogue at `Model.holoTableCL` (`Recursion.holoInnerCL`): at a statement no table near the
registered word makes true, every inner prover has a fork whose two runs clash (`clashPickHL`: the reps, R's stream, or
R's level 0 opened to a row other than the registered word's), or the table accepts with probability at most
`holoError`. `holoForkSound_holoAt` is that at the executable's parameters (`holoAt st hp`), the form
`RecursiveAuditC3`'s proof reads.

* `holo_inner_sound_compiled`: the general lemma, `flock_inner_sound_compiled`'s analogue, from `holo_runs_sound`
  (`C3/Fork/Runs.lean`) as `flock_inner_sound_compiled` is from `compiled_runs_sound`. `holo_runs_sound` composes the
  reps' count, the sparse phase's (`sparse_sound`, `C3/Fork/Sparse.lean`) and R's opening at unique decoding
  (`regOpen_sound`, `C3/Fork/RegOpen.lean`).
* `holoAt_fast100`, `holoAt_linkLayout`, `holoAt_rsch`, `holoAt_units`: the executable parameters meet its hypotheses,
  as `compiledForkSound_execClassAt` does for `execClassAt`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **Candidate 3's compiled table is sound in fork form**: at the reps' schedule `fast100`'s, a link layout at every
table, R's registered schedule either empty (R rejects) or `fast100`'s at `mReg`, and every unit slot within the table,
the inner protocol `holoInnerCL` is sound with the pick `clashPickHL` and the error `holoError`. -/
theorem holo_inner_sound_compiled (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog)) :
    InnerSoundForkOn H (holoInnerCL T CT A H E Lf e) (clashPickHL T CT A H E Lf e)
      (ENNReal.ofReal (holoError e)) := by
  intro x hno G s
  change {x : HoloReg F D // x.Opens H E Lf e.rsch.sch.level₀} at x
  have hout : ∀ {t o}, PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal
      e.cls.mPts e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.1.cap) (schedSteps T CT 0 t) o →
      outHL T CT A H E Lf e x.1 t = some o := by
    intro t o ho
    have hex : ∃ out, PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal
        e.cls.mPts e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.1.cap) (schedSteps T CT 0 t) out :=
      ⟨o, ho⟩
    rw [outHL, dif_pos hex]
    exact congrArg some (PlaysT.det hex.choose_spec ho)
  rcases holo_runs_sound T CT A hA H E Lf e hsch hlay hR hunits x hno G s with
    ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hc⟩ | hb
  · left
    have hr : 0 < (holoInnerCL T CT A H E Lf e).r := Nat.succ_pos r
    refine Inner.fork_of_outcomes H _ hr _ s ht ht' ?_
    have ep : pickOn G (clashPickHL T CT A H E Lf e x) t t' = clashPickHL T CT A H E Lf e x t t' :=
      pickOn_of hG hG'
    exact (congrArg (StrictCR.IsColl H) ep).mpr (collOf_isColl H _ ⟨o, o', hout ho, hout ho', hc⟩)
  · exact Or.inr hb

/-- **The reps' schedule at the executable's parameters is `fast100`'s**, at a statement whose class checks pass. -/
theorem holoAt_fast100 {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    Accounting.fast100 (holoAt st hp).cls.m = some (holoAt st hp).sch := by
  obtain ⟨-, -, _, hs⟩ := ClassOk.of_shape h
  obtain ⟨s₀, hs₀⟩ := ZkHidden.fast100_of_schedule hs
  have hsch : Accounting.fast100 st.m = some (schedAt st.m) := by
    have e := Refine.fast100_schedOf st.m
    rw [hs₀] at e
    unfold schedAt
    rw [hs₀]
    exact e.symm
  have e : execClassAt st = ExecClass.ofStmt st h := dif_pos h
  show Accounting.fast100 (execClassAt st).cls.m = some (execClassAt st).sch
  rw [e]
  exact hsch

/-- **Every table's statement at the executable's parameters has the link layout**, at a statement whose class checks
pass: the regions and link points are `execClassAt st`'s (`Refine.stmtOf_linkLayout`), and no table changes them. -/
theorem holoAt_linkLayout {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    ∀ R : RegTable (holoAt st hp).cls,
      (R.toStatement (holoAt st hp).τ (holoAt st hp).regs).LinkLayout (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts := by
  obtain ⟨hwf, hr, -⟩ := ClassOk.of_shape h
  have hl : ∀ ec : ExecClass, ec = ExecClass.ofStmt st h →
      (∀ R ∈ ec.regs, R.free.Nodup ∧ R.free.length ≤ ec.cls.ptLocal) ∧
        ec.cls.ptLocal + (ec.cls.m - ec.cls.kLog) ≤ ec.cls.mPts := by
    rintro _ rfl
    exact Refine.stmtOf_linkLayout st hwf hr
  have e : execClassAt st = ExecClass.ofStmt st h := dif_pos h
  exact fun _ => hl _ e

/-- **R's registered schedule at the executable's parameters is empty or `fast100`'s at `mReg`**: `schedAt mReg`, which
is `fast100`'s where it covers `mReg` and the default (empty levels) elsewhere, or the default. -/
theorem holoAt_rsch (st : Flock.Stmt) (hp : HoloParams) :
    (holoAt st hp).rsch.sch.levels = [] ∨
      Accounting.fast100 (holoAt st hp).cls.shape.mReg = some (holoAt st hp).rsch.sch := by
  have key : ∀ (M : ℕ) (s : Accounting.Schedule), (s = default ∨ s = schedAt M) →
      s.levels = [] ∨ Accounting.fast100 M = some s := by
    rintro M s (rfl | rfl)
    · exact Or.inl rfl
    · obtain h | ⟨s₀, h⟩ : Flock.fast100 M = none ∨ ∃ s₀, Flock.fast100 M = some s₀ := by
        cases Flock.fast100 M <;> simp
      · unfold schedAt
        rw [h]
        exact Or.inl rfl
      · have e := Refine.fast100_schedOf M
        rw [h] at e
        unfold schedAt
        rw [h]
        exact Or.inr e.symm
  refine key (holoAt st hp).cls.shape.mReg (holoAt st hp).rsch.sch ?_
  have e : (holoAt st hp).rsch.sch =
      if hp.EllOk (execClassAt st).cls.kLog then schedAt (holoAt st hp).cls.shape.mReg else default := rfl
  by_cases hl : hp.EllOk (execClassAt st).cls.kLog
  · rw [e, if_pos hl]
    exact Or.inr rfl
  · rw [e, if_neg hl]
    exact Or.inl rfl

/-- **Every unit slot at the executable's parameters is within the table**: `HoloClass.ofExec` keeps the units below
`2^(k_log − ℓ)`. -/
theorem holoAt_units (st : Flock.Stmt) (hp : HoloParams) :
    ∀ σ, ∀ u ∈ (holoAt st hp).cls.units σ, u < 2 ^ ((holoAt st hp).cls.kLog - (holoAt st hp).cls.slotLog) := by
  intro σ u hu
  have hu' : u ∈ (hp.units σ).filter fun u =>
      u < 2 ^ ((execClassAt st).cls.kLog - (if hp.EllOk (execClassAt st).cls.kLog then hp.ell else 6)) := hu
  exact (Finset.mem_filter.1 hu').2

/-- **Candidate 3's table at the executable's parameters is sound in fork form**, at a statement whose class checks
pass: `holo_inner_sound_compiled` at `execArith`, SHA-512 with plain leaves and `holoAt st hp`. -/
theorem holoForkSound_holoAt {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    InnerSoundForkOn Refine.H512 (holoInnerAt T CT st hp)
      (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp))
      (ENNReal.ofReal (holoError (holoAt st hp))) := by
  let _ : Inhabited Refine.Dig := ⟨Refine.H512 []⟩
  exact holo_inner_sound_compiled T CT execArith execArith_correct Refine.H512 Refine.enc512 Refine.leafPlain
    (holoAt st hp) (holoAt_fast100 hp h) (holoAt_linkLayout hp h) (holoAt_rsch st hp) (holoAt_units st hp)

end FlockC3.Fork
