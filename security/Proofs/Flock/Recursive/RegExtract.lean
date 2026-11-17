module

public import Proofs.Flock.Recursive.AuditC3
public import Proofs.Flock.Recursive.ForkInnerC
public import Proofs.Flock.C3.Fork.Sound
public import Proofs.Flock.C3.Fork.Extract
public import Proofs.Flock.Recursive.ValueBridge

/-! The recursive audit at the word the prover's own runs open at R's level 0, where a run opening off that word is a fork against the run that defines it (`zk_sessions_recursive_fork_reg`). -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Runs

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Model Accounting Exec in
theorem holo_runs_knowledge (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog))
    (x : HoloReg F D) (hno : ¬ (holoInnerCL T CT A H E Lf e).holds x)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop) (s : Strategy (holoInnerCL T CT A H E Lf e).game) :
    (∃ t t' o o', Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧
      Game.Outcome (holoInnerCL T CT A H E Lf e).game s t' ∧ G t ∧ G t' ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) o ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t') o' ∧
      ClashRunHL H E Lf e.sch e.rsch.sch x.cap o o') ∨
    (∃ t o, Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧ G t ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) o ∧
      RegClash H E Lf e.rsch.sch.level₀ x o) ∨
    prob (fun t => (holoInnerCL T CT A H E Lf e).accepts x t ∧ G t) (holoInnerCL T CT A H E Lf e).game s ≤
      ENNReal.ofReal (holoError e) := by
  rcases FlockC3.Fork.holo_runs_sound T CT A hA H E Lf e hsch hlay hR hunits x hno G s with
    ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hc | hc | hc | hc⟩ | hb
  · exact Or.inl ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', Or.inl hc⟩
  · exact Or.inl ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', Or.inr hc⟩
  · exact Or.inr (Or.inl ⟨t, o, ht, hG, ho, hc⟩)
  · exact Or.inr (Or.inl ⟨t', o', ht', hG', ho', hc⟩)
  · exact Or.inr (Or.inr hb)

end Runs

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : SamplingDesign n} {Reg Cm : Type}

theorem zk_sessions_recursive_fork_reg {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (_hcls : ClassOk (shapeOf st))
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (holoInnerAt T CT st hp).Msg)
    (cap : Cap Refine.Dig)
    (_hbr : Flock.Assumptions.VBridgeC zs dg N k Rw (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg
      Sk cmt msg) :
    (regFinder zs dg N σ).Finds Refine.H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg Refine.H512
        (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAt T CT st hp).holds (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap o.1 o.2.1))
          (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((holoInnerAt T CT st hp).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerL zs dg N _hk _hRw _hM _hρ _hr σ _ wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_reg T CT hp _hcls zs dg N k Rw σ wmsg cap)

theorem zk_sessions_recursive_fork_regV {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (_hcls : ClassOk (shapeOf st))
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ (holoInnerAt T CT st hp) Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (holoInnerAt T CT st hp).Msg)
    (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json) (bcs' : Array ByteArray)
    (hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    (_hbr : Flock.Assumptions.VBridgeCOff (fun R ω t R₂ o => ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ o ∨
      SaltColl (holoAt st hp).rsch.sch.level₀ cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw
      (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg Sk cmt msg) :
    (valueFinder zs vsrc dg N σ).Finds Refine.H512 ∨ (leafFinder zs dg N σ).Finds Refine.H512 ∨
      (saltFinder (holoAt st hp).rsch.sch.level₀ cap salt zs σ).Finds Refine.H512 ∨
      (topFinder wr st reg bcs' zs σ).Finds Refine.H512 ∨
      (regFinder zs dg N σ).Finds Refine.H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg Refine.H512
        (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAt T CT st hp).holds (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap o.1 o.2.1))
          (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((holoInnerAt T CT st hp).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerLV zs vsrc (holoAt st hp).rsch.sch.level₀ cap salt wr st reg bcs' hstk dg N
    _hk _hRw _hM _hρ _hr σ _ wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_reg T CT hp _hcls zs dg N k Rw σ wmsg cap)

/-- `RecursiveAuditHolo` reads it at the registered typing `hp.τ`, which `_hbr`'s `τv` is wherever check 5 accepts the inner
shape. -/
theorem zk_sessions_recursive_fork_regVτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog))
    (_hcls : ClassOk (shapeOf st))
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ (holoInnerAtτ T CT st hp τv) Vs) (lv : FlockSoundness.Accounting.Level)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (holoInnerAtτ T CT st hp τv).Msg)
    (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (wr : List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Lean.Json × String) (reg : Lean.Json) (bcs' : Array ByteArray)
    (hstk : StackedTopAt lv st reg cap salt bcs')
    (_hbr : Flock.Assumptions.VBridgeCOff (fun R ω t R₂ o => ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ o ∨
      SaltColl lv cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw
      (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap) wmsg Sk cmt msg) :
    (valueFinder zs vsrc dg N σ).Finds Refine.H512 ∨ (leafFinder zs dg N σ).Finds Refine.H512 ∨
      (saltFinder lv cap salt zs σ).Finds Refine.H512 ∨
      (topFinder wr st reg bcs' zs σ).Finds Refine.H512 ∨
      (regFinder zs dg N σ).Finds Refine.H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap) wmsg Refine.H512
        (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAtτ st hp τv)) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAtτ T CT st hp τv).holds
            (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap o.1 o.2.1))
          (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((holoInnerAtτ T CT st hp τv).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerLV zs vsrc lv cap salt wr st reg bcs' hstk dg N _hk _hRw _hM _hρ _hr σ _
    wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_regτ T CT hp τv _hcls zs dg N k Rw σ wmsg cap)

end FlockSoundness.Discharge.Recursion
