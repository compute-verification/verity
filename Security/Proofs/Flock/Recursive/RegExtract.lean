import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.ForkInnerC
import Proofs.Flock.C3.Fork.Sound
import Proofs.Flock.C3.Fork.Extract
import Proofs.Flock.Recursive.ValueBridge

/-!
# Recursion: gap 2 at the word the prover's runs open (the knowledge form)

`RecursiveAuditC3`'s hidden statement is the word the transported prover's own runs open at R's level 0 against R's
cap at each draw (`Flock.Guarantees.regStmt`), not a registrant's (`note:41519fce`). No registrant's openings are taken,
so a run that opens R's level 0 off a word (`RegClash`) is no collision by itself. It is one against another of the
prover's runs at the same draw, the one whose opening defines the word at that position (`regWord`): two runs at one
draw that open one position of R's level 0 to different rows against one cap conflict at R's stream's level-0 cap
(`ClashCL`'s `op₀L` conflict at `OutHL.reg`), which `clashPickHL` reads (`ClashRunHL`), and two independent runs at a
draw are the inner fork at round 0 (`forkG pick 0`, `StrictCR.twoRuns`).

* `holo_runs_knowledge`: piece (a)'s runs (`holo_runs_sound`) with the clash split: two runs that clash against each
  other (`ClashRunHL`), a run that opens R's level 0 off the statement's word (`RegClash`), or the bound.
* `zk_sessions_recursive_fork_reg`: `zk_sessions_recursive_fork_innerC` at `regStmt` and
  `clashPickHL`, with candidate 3's runs (`holo_runs_knowledge` at `holoAt st hp`, from the class checks) in place of
  `InnerSoundForkOn`. Its proof is the extraction: `zk_sessions_recursive_fork_innerC`'s, with
  `innerAtLeavesL_of_forkOn`'s argument at each draw reading `holo_runs_knowledge`, where a leaf strategy's run that
  `RegClash`es at `regWord` is a run of the transported prover at that draw, as is the run `regWord` reads at the
  position (`RunOpensReg`), so the two are an `InnerForkRec` at round 0.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Runs

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Model Accounting Exec in
/-- **Candidate 3's table on one prover's runs, the clash split**: at a statement no table near whose word makes the
hidden statement true, for every `G` and every prover `s` of `holoInnerCL`: two transcripts `s` reaches that satisfy `G`
play the table to outcomes that clash against each other (`ClashRunHL`, at the statement's cap), or one does to an
outcome that opens R's level 0 off the statement's word (`RegClash`), or `s` makes the inner verifier accept a
transcript satisfying `G` with probability at most `holoError`. -/
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

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg Cm : Type}

/-- **Gap 2 at the word the prover's runs open** (the extraction, `C3/Fork/Extract.lean`).
`zk_sessions_recursive_fork_innerC` at candidate 3's inner protocol, at a statement whose class checks pass, with the hidden statement the word the
prover's runs open at R's level 0 against `cap` at each draw (`regStmt`) and the pick `clashPickHL`, which reads only
clashes of two runs: a registrant fork, a round fork, a session fork, an inner fork of `σ`'s own runs, or the bound
with candidate 3's error. -/
theorem zk_sessions_recursive_fork_reg {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (_hcls : ClassOk (shapeOf st))
    (_hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
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
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((holoInnerAt T CT st hp).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerL _hA3 zs dg N _hk _hRw _hM _hρ _hr σ _ wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_reg T CT hp _hcls zs dg N k Rw σ wmsg cap)

/-- **Gap 2 at the word the prover's runs open, off a value the verifier registers itself**: `zk_sessions_recursive_fork_reg`
with V*'s bridge off `RegColl ∨ ValueColl zs vsrc ∨ LeafColl zs ∨ SaltColl ∨ TopColl` (`Flock.Assumptions.VBridgeCOff`,
`SaltColl` at R's level 0, `cap` and the held salts `salt`, `TopColl` at the registered top `bcs'`), the value, the table
leaves', the held salts' and the held top's finders finding a collision four more cases
(`zk_sessions_recursive_fork_innerLV`). -/
theorem zk_sessions_recursive_fork_regV {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (_hcls : ClassOk (shapeOf st))
    (_hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
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
    (_hbr : Flock.Assumptions.VBridgeCOff (fun R ω t R₂ => ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ ∨
      SaltColl (holoAt st hp).rsch.sch.level₀ cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw
      (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg Sk cmt msg) :
    (valueFinder zs vsrc dg N σ).Finds Refine.H512 ∨ (leafFinder zs dg N σ).Finds Refine.H512 ∨
      (saltFinder (holoAt st hp).rsch.sch.level₀ cap salt zs σ).Finds Refine.H512 ∨
      (topFinder wr st reg bcs' (holoAt st hp).rsch.sch.level₀ cap salt hstk zs σ).Finds Refine.H512 ∨
      (regFinder zs dg N σ).Finds Refine.H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg Refine.H512
        (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAt T CT st hp).holds (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap o.1 o.2.1))
          (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((holoInnerAt T CT st hp).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerLV _hA3 zs vsrc (holoAt st hp).rsch.sch.level₀ cap salt wr st reg bcs' hstk dg N
    _hk _hRw _hM _hρ _hr σ _ wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_reg T CT hp _hcls zs dg N k Rw σ wmsg cap)

/-- **`zk_sessions_recursive_fork_regV` at the typing `τv`**: everything at `τv`'s protocol
(`holoInnerAtτ`, `regStmtτ`, `clashPickHL` at `holoAtτ`), the salts' level a free parameter `lv` (the audit reads it
at `(holoAt st hp).rsch.sch.level₀`, which no typing changes), the held top's collision `TopColl` off the bridge as in
`zk_sessions_recursive_fork_regV`, and the error `holoAt st hp`'s own (`innerAtLeavesL_regτ`). `RecursiveAuditC3`
reads it at the registered typing `hp.τ`, which `_hbr`'s `τv` is wherever check 5 accepts the inner shape. -/
theorem zk_sessions_recursive_fork_regVτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog))
    (_hcls : ClassOk (shapeOf st))
    (_hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
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
    (_hbr : Flock.Assumptions.VBridgeCOff (fun R ω t R₂ => ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ ∨
      SaltColl lv cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw
      (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap) wmsg Sk cmt msg) :
    (valueFinder zs vsrc dg N σ).Finds Refine.H512 ∨ (leafFinder zs dg N σ).Finds Refine.H512 ∨
      (saltFinder lv cap salt zs σ).Finds Refine.H512 ∨
      (topFinder wr st reg bcs' lv cap salt hstk zs σ).Finds Refine.H512 ∨
      (regFinder zs dg N σ).Finds Refine.H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap) wmsg Refine.H512
        (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAtτ st hp τv)) ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAtτ T CT st hp τv).holds
            (Flock.Guarantees.regStmtτ T CT st hp τv zs dg N k Rw wmsg σ cap o.1 o.2.1))
          (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((holoInnerAtτ T CT st hp τv).r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp)) :=
  zk_sessions_recursive_fork_innerLV _hA3 zs vsrc lv cap salt wr st reg bcs' hstk dg N _hk _hRw _hM _hρ _hr σ _
    wmsg Sk cmt msg _hbr Refine.H512 _
    (Nat.succ_pos r) (innerAtLeavesL_regτ T CT hp τv _hcls zs dg N k Rw σ wmsg cap)

end FlockSoundness.Discharge.Recursion
