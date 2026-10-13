import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.ForkInnerC
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.C3.Fork.Sound

/-!
# Recursion: the extraction at the word the prover's runs open

The pieces of `zk_sessions_recursive_fork_reg` (`Recursive/RegExtract.lean`). A run of the transported prover at a
draw `ω` that opens a position of R's level 0 off `regWord` conflicts there with the run `regWord` reads at that
position (`RunOpensReg`): both are runs of `σ` at `ω`, so the two are an inner fork at round 0.

* `leaf_of_contF`, `contF_of_leaf`: a trial `contF` reaches is one at a leaf the rounds reach, and back.
* `LeafRun`, `rec_of_leafRun`, `leafRun_of_rec`: the trials at `σ`'s leaves at `ω` are the recursive game's outcomes at
  `ω`, decoded.
* `clash₀_of_two`: two runs that open one position of R's level 0 to different rows against one cap clash (`ClashCL`
  at R's part).
* `regClash_fork`, the per-draw fact: a leaf's run that `RegClash`es at `regStmt` is an inner fork at round 0.
* `innerAtLeavesL_reg`: `innerAtLeavesL_of_forkOn`'s argument at each draw on `holo_runs_sound` at that draw's
  `regStmt`, its `RegClash` cases by `regClash_fork`.
* `zk_sessions_recursive_fork_innerL`: `zk_sessions_recursive_fork_innerC` with the inner term a hypothesis.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Leaf

variable {M C β : Type} [Fintype C] [Nonempty C]

/-- **A trial `contF` reaches is a leaf's**: at some leaf `t` of `s`, with its strategy `s'`, whose outcomes are ones
`s` reaches. -/
theorem leaf_of_contF {δ : Type} : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (Tr : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) (s : Strategy (rounds M C r f)) {d : δ},
    Game.Outcome (contF r f Tr s).g (contF r f Tr s).s d →
      ∃ t s', (∀ b, Game.Outcome (f t) s' b → Game.Outcome (rounds M C r f) s b) ∧
        Game.Outcome (Tr t s').g (Tr t s').s d
  | 0, _, _, s, _, hd => ⟨[], s, fun _ hb => hb, hd⟩
  | r + 1, f, Tr, s, _, hd => by
      obtain ⟨ρ, hd⟩ := hd
      obtain ⟨t, s', hl, ht⟩ :=
        leaf_of_contF r (fun t => f ((s.1, ρ) :: t)) (fun t => Tr ((s.1, ρ) :: t)) (s.2 ρ) hd
      exact ⟨(s.1, ρ) :: t, s', fun b hb => ⟨ρ, hl b hb⟩, ht⟩

/-- **An outcome `s` reaches is a leaf's**, at a leaf whose every trial's outcomes `contF` reaches. -/
theorem contF_of_leaf : ∀ (r : ℕ) (f : List (M × C) → Game β) (s : Strategy (rounds M C r f)) {b : β},
    Game.Outcome (rounds M C r f) s b → ∃ t s', Game.Outcome (f t) s' b ∧
      ∀ {δ : Type} (Tr : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) d,
        Game.Outcome (Tr t s').g (Tr t s').s d → Game.Outcome (contF r f Tr s).g (contF r f Tr s).s d
  | 0, _, s, _, hb => ⟨[], s, hb, fun _ _ hd => hd⟩
  | r + 1, f, s, _, hb => by
      obtain ⟨ρ, hb⟩ := hb
      obtain ⟨t, s', hb, hl⟩ := contF_of_leaf r (fun t => f ((s.1, ρ) :: t)) (s.2 ρ) hb
      exact ⟨(s.1, ρ) :: t, s', hb, fun Tr d hd => ⟨ρ, hl (fun t => Tr ((s.1, ρ) :: t)) d hd⟩⟩

theorem forkG_zero_of {D γ : Type} (H : List UInt8 → D) (pick : γ → γ → StrictCR.Pair) (r : ℕ)
    (f : List (M × C) → Game β) (Tr : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ)
    (s : Strategy (rounds M C (r + 1) f)) {d d' : γ}
    (hd : Game.Outcome (contF (r + 1) f Tr s).g (contF (r + 1) f Tr s).s d)
    (hd' : Game.Outcome (contF (r + 1) f Tr s).g (contF (r + 1) f Tr s).s d')
    (hc : StrictCR.IsColl H (pick d d')) : (forkG pick 0 (r + 1) f Tr s).Finds H :=
  ⟨_, outcome_twoRuns _ _ pick hd hd', hc⟩

end Leaf

section Out

open Model Accounting Exec

theorem rowFin_inj {F : Type} [Field F] [Fintype F] [DecidableEq F] {lanes : ℕ} {u v : List F}
    (hu : u.length = 2 ^ lanes) (hv : v.length = 2 ^ lanes) (h : rowFin lanes u = rowFin lanes v) : u = v := by
  refine List.ext_getElem (hu.trans hv.symm) fun i hi hi' => ?_
  have e := congrFun h ⟨i, hu ▸ hi⟩
  simp only [rowFin, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, List.getElem?_eq_getElem hi',
    Option.getD_some] at e
  exact e

/-- **Two runs that open one position of R's level 0 to different rows against one cap clash**: on R's part, a
self-clash of either or a level-0 conflict. -/
theorem clash₀_of_two {F : Type} [Field F] [Fintype F] [DecidableEq F] {D Sl : Type} (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (rsch : Schedule) (cap : Cap D)
    (a b : OutHL F D Sl) {y z : ℕ × List F × Sl × List D}
    (hy : y ∈ slotOpensL (a.reg cap) 0 0 rsch.level₀.queries) (hz : z ∈ slotOpensL (b.reg cap) 0 0 rsch.level₀.queries)
    (hlt : y.1 < 2 ^ rsch.level₀.logLen) (hzy : z.1 = y.1)
    (hvy : VerifiedL H E Lf cap rsch.level₀.lanes (pathLen rsch.level₀) y)
    (hvz : VerifiedL H E Lf cap rsch.level₀.lanes (pathLen rsch.level₀) z)
    (hne : rowFin rsch.level₀.lanes y.2.1 ≠ rowFin rsch.level₀.lanes z.2.1) :
    ClashCL H E Lf rsch (a.reg cap) (b.reg cap) := by
  have hlv : rsch.level₀ = lvOf rsch 0 := by
    rcases rsch with ⟨m, k0, _ | ⟨l, ls⟩, ext⟩ <;> rfl
  rw [hlv] at hy hz hlt hvy hvz hne
  rcases off_or_clashL H E Lf (lvOf rsch 0).lanes (pathLen (lvOf rsch 0)) (lvOf rsch 0).logLen
      (opens₀L rsch (b.reg cap)) cap z (List.mem_append_left _ hz) ⟨y.1, hlt⟩ hzy hvz _ (Ne.symm hne) with
    ⟨hs, hb⟩ | hcl
  · obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hs
    rcases off_or_clashL H E Lf (lvOf rsch 0).lanes (pathLen (lvOf rsch 0)) (lvOf rsch 0).logLen
        (opens₀L rsch (a.reg cap)) cap y (List.mem_append_left _ hy) ⟨y.1, hlt⟩ rfl hvy w
        (fun h => hb (hw.trans (congrArg some h.symm))) with ⟨hs', ha⟩ | hcl
    · exact Or.inr (Or.inr (Or.inl ⟨⟨y.1, hlt⟩, hs', hs, fun h => ha (h.trans hw)⟩))
    · exact Or.inl (Or.inl hcl)
  · exact Or.inr (Or.inl (Or.inl hcl))

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

omit [Fintype K] in
/-- `outHL` reads the statement only through its cap. -/
theorem outHL_cap (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (e : ExecHolo) {w w' : Oracle F} {cap : Cap D} (t : List (List (SMsg T) × ((k : Fin r) → CT k))) :
    outHL T CT A H E Lf e ⟨w, cap⟩ t = outHL T CT A H E Lf e ⟨w', cap⟩ t :=
  rfl

omit [Fintype K] in
theorem outHL_of_plays (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    {t : List (List (SMsg T) × ((k : Fin r) → CT k))} {o : OutHL F D Sl}
    (ho : PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
      e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) o) :
    outHL T CT A H E Lf e x t = some o := by
  have hex : ∃ out, PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal
      e.cls.mPts e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) out :=
    ⟨o, ho⟩
  rw [outHL, dite_eq_left hex]
  exact congrArg some (PlaysT.det hex.choose_spec ho)

omit [Fintype K] in
theorem outHL_lastCoin (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    {t t' : List (List (SMsg T) × ((k : Fin r) → CT k))} (hl : t.length = r + 1) (h : LastCoin t t') :
    outHL T CT A H E Lf e x t = outHL T CT A H E Lf e x t' := by
  unfold outHL
  rw [schedSteps_lastCoin T CT hl h]

end Out

section Chain

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg Cm : Type} {I : Inner}

/-- **A trial at one of `σ`'s leaves at the draw `ω` decodes to `d`**: `InnerForkRec`'s trial, run from the root. -/
def LeafRun (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) (ω : L.Ω) (d : List (I.Msg × I.Coin)) :
    Prop :=
  Game.Outcome
    (contF I.r (fun t => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
      (fun t s => StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) (decodedV zs dg N k Rw wmsg σ.1 ω t))
      (σ.2 ω)).g
    (contF I.r (fun t => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
      (fun t s => StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) (decodedV zs dg N k Rw wmsg σ.1 ω t))
      (σ.2 ω)).s d

theorem rec_of_leafRun (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) {ω : L.Ω} {d : List (I.Msg × I.Coin)}
    (hd : LeafRun zs dg N k Rw wmsg σ ω d) :
    ∃ O, Game.Outcome (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ O ∧ O.2.1 = ω ∧
      decodedV zs dg N k Rw wmsg O.1 O.2.1 O.2.2.1 O.2.2.2 = d := by
  unfold LeafRun at hd
  obtain ⟨t, s', hl, ht⟩ := leaf_of_contF _ _ _ _ hd
  obtain ⟨b, hb, rfl⟩ := outcome_of_map _ _ _ ht
  have hf := Game.Outcome.map_of (fun b => (σ.1, ω, t, b)) _ _ hb
  rw [map_ofMap] at hf
  exact ⟨(σ.1, ω, t, b), ⟨ω, hl _ hf⟩, rfl, rfl⟩

theorem leafRun_of_rec (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) {O} {ω : L.Ω}
    (hO : Game.Outcome (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ O) (hω : O.2.1 = ω) :
    LeafRun zs dg N k Rw wmsg σ ω (decodedV zs dg N k Rw wmsg O.1 O.2.1 O.2.2.1 O.2.2.2) := by
  subst hω
  obtain ⟨ω, hO⟩ := hO
  obtain ⟨t, s', hb, hl⟩ := contF_of_leaf _ _ _ hO
  have hb' : Game.Outcome ((outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
      (Strategy.map (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s')) O := by
    rw [map_ofMap]
    exact hb
  obtain ⟨b, hb₂, rfl⟩ := outcome_of_map _ _ _ hb'
  show LeafRun zs dg N k Rw wmsg σ ω (decodedV zs dg N k Rw wmsg σ.1 ω t b)
  unfold LeafRun
  exact hl _ _ (outcome_oneRun _ _ _ hb₂)

theorem regWord_of {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    {st : Flock.Stmt} (hp : HoloParams)
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) {ω : L.Ω} {pos : ℕ}
    (hex : ∃ col, Flock.Guarantees.RunOpensReg T CT st hp zs dg N k Rw wmsg σ cap ω pos col) (ℓ : ℕ) :
    Flock.Guarantees.regWord T CT st hp zs dg N k Rw wmsg σ cap ω ℓ pos = hex.choose.getD ℓ 0 := by
  simp only [Flock.Guarantees.regWord, dite_eq_left hex]

/-- **The per-draw fact**: a run at one of `σ`'s leaves at `ω` that opens R's level 0 off the word `σ`'s runs open
there (`RegClash` at `regStmt`) is an inner fork at round 0. The position it opens off the word is one some run of `σ`
at `ω` opens (`RunOpensReg`, the run `regWord` reads), to `regWord`'s column, so to another row against one cap: the
two runs are two continuations of `σ` from its root at `ω`, which `clashPickHL` reads as a conflict (`ClashRunHL`). -/
theorem regClash_fork {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams)
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) {ω : L.Ω} {d} {o : OutHL GF128 Refine.Dig (List UInt8)}
    (hd : LeafRun zs dg N k Rw wmsg σ ω d)
    (ho : outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)
      (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap σ.1 ω) d = some o)
    (hc : RegClash Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp).rsch.sch.level₀
      (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap σ.1 ω) o) :
    (forkG (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)
        (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap σ.1 ω)) 0 (holoInnerAt T CT st hp).r
      (fun t => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b))
      (fun t s => StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) (decodedV zs dg N k Rw wmsg σ.1 ω t))
      (σ.2 ω)).Finds Refine.H512 := by
  obtain ⟨y, hy, hlt, hv, hne⟩ := hc
  obtain ⟨O, hO, hOω, rfl⟩ := rec_of_leafRun zs dg N k Rw wmsg σ hd
  have hex : ∃ col, Flock.Guarantees.RunOpensReg T CT st hp zs dg N k Rw wmsg σ cap ω y.1 col :=
    ⟨y.2.1, O, hO, hOω, o, Eq.trans (outHL_cap T CT _ _ _ _ _ _) ho, y, hy, rfl, hv, rfl⟩
  obtain ⟨O₂, hO₂, hO₂ω, o₂, ho₂, y₂, hy₂, hp₂, hv₂, hc₂⟩ := hex.choose_spec
  have hcol : rowFin (holoAt st hp).rsch.sch.level₀.lanes
      ((Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap σ.1 ω).column
        (holoAt st hp).rsch.sch.level₀.lanes y.1) = rowFin (holoAt st hp).rsch.sch.level₀.lanes y₂.2.1 := by
    funext lane
    simp only [rowFin, HoloReg.column]
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range lane.isLt, hc₂]
    exact regWord_of T CT hp zs dg N k Rw wmsg σ cap hex lane
  refine forkG_zero_of Refine.H512 _ r _ _ (σ.2 ω) hd (leafRun_of_rec zs dg N k Rw wmsg σ hO₂ hO₂ω)
    (collOf_isColl Refine.H512 _ ⟨o, o₂, ho, Eq.trans (outHL_cap T CT _ _ _ _ _ _) ho₂, Or.inr ?_⟩)
  refine clash₀_of_two Refine.H512 Refine.enc512 Refine.leafPlain _ cap o o₂ hy hy₂ hlt hp₂ hv hv₂ ?_
  rw [← hcol]
  exact fun h => hne (rowFin_inj hv.1 (by simp only [HoloReg.column, List.length_map, List.length_range]) h)

/-- **The inner term at the word the prover's runs open**: `innerAtLeavesL_of_forkOn`'s argument at each draw, on
`holo_runs_sound` at that draw's `regStmt` in place of `InnerSoundForkOn`. Two of the simulated prover's runs that
clash against each other are a fork of `σ`'s own runs (`finds_forkG_refL`), and one that opens R's level 0 off the word
is one at a leaf of `σ` at that draw but for the last coin (`outcome_contF_refL`), so an inner fork at round 0
(`regClash_fork`). -/
theorem innerAtLeavesL_reg {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (hp : HoloParams) (hcls : ClassOk (shapeOf st))
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg) (cap : Cap Refine.Dig)
    (hno : ¬ InnerForkRec zs dg N k Rw σ (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap) wmsg
      Refine.H512 (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp))) :
    ∀ ref, InnerAtLeavesL σ (fun R ω t o => AcceptsV (zs R ω t) o)
      (Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap)
      (W := ((j : Fin S) → (Vs j).Bits) × (holoInnerAt T CT st hp).Stmt)
      (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2,
        Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap R ω))
      (fun w => wmsg w.1) ref (ENNReal.ofReal (holoError (holoAt st hp))) := by
  classical
  let _ : Inhabited Refine.Dig := ⟨Refine.H512 []⟩
  intro ref ω
  let x := Flock.Guarantees.regStmt T CT st hp zs dg N k Rw wmsg σ cap σ.1 ω
  let pk := clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x
  by_cases hx : (holoInnerAt T CT st hp).holds x
  · refine le_trans (leafAvg_mono (G := fun _ _ => 0) (fun t s => ?_) _) ?_
    · exact le_of_eq (ite_eq_right fun h => h.1 hx)
    · exact (leafAvg_const 0 _).le.trans zero_le
  let f := fun t : List (Cm × (holoInnerAt T CT st hp).Coin) => (outerV (zs σ.1 ω t) dg N).map fun b => (σ.1, ω, t, b)
  let V : (t : List (Cm × (holoInnerAt T CT st hp).Coin)) → Strategy (f t) →
      List ((holoInnerAt T CT st hp).Msg × (holoInnerAt T CT st hp).Coin) → Prop := fun t s u =>
    ∃ b, Game.Outcome (outerV (zs σ.1 ω t) dg N)
      (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) b ∧
      AcceptsV (zs σ.1 ω t) b ∧ decodedV zs dg N k Rw wmsg σ.1 ω t b = u
  let Tr : (t : List (Cm × (holoInnerAt T CT st hp).Coin)) → Strategy (f t) →
      StrictCR.Finder (List ((holoInnerAt T CT st hp).Msg × (holoInnerAt T CT st hp).Coin)) := fun t s =>
    StrictCR.oneRun (outerV (zs σ.1 ω t) dg N)
      (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outerV (zs σ.1 ω t) dg N) s) (decodedV zs dg N k Rw wmsg σ.1 ω t)
  let Q := (holoInnerAt T CT st hp).accepts x
  let G := AtLeafL (holoInnerAt T CT st hp).r f V (σ.2 ω)
  let s := refStratL Q (holoInnerAt T CT st hp).r f (ref ω) id V (σ.2 ω)
  have hVT : ∀ t s' u, V t s' u → Game.Outcome (Tr t s').g (Tr t s').s (id u) := by
    rintro t s' u ⟨b, hb, -, rfl⟩
    exact outcome_oneRun _ _ _ hb
  have hr0 : 0 < (holoInnerAt T CT st hp).r := Nat.succ_pos r
  have hfork : ∀ t t' o o', Game.Outcome (holoInnerAt T CT st hp).game s t →
      Game.Outcome (holoInnerAt T CT st hp).game s t' → G t → G t' →
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x t = some o →
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x t' = some o' →
      ClashRunHL Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp).sch (holoAt st hp).rsch.sch x.cap o o' →
      False := by
    intro t t' o o' ht ht' hG hG' ho ho' hc
    obtain ⟨i, hi, hF⟩ := Inner.fork_of_outcomes Refine.H512 _ hr0 (pickOn G pk) s ht ht'
      ((congrArg (StrictCR.IsColl Refine.H512) (pickOn_of hG hG')).mpr
        (collOf_isColl Refine.H512 _ ⟨o, o', ho, ho', hc⟩))
    exact hno ⟨ω, i, hi, finds_forkG_refL Refine.H512 pk Q G i _ f (ref ω) id V Tr (σ.2 ω) hVT (fun _ hu => hu)
      (clashPickHL_lastFree T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x) hF⟩
  have hreg : ∀ t o, Game.Outcome (holoInnerAt T CT st hp).game s t → G t →
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x t = some o →
      RegClash Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp).rsch.sch.level₀ x o → False := by
    intro t o ht hG ho hc
    obtain ⟨u, u', rfl, hl, hlc, hu'⟩ := outcome_contF_refL Q G _ f (ref ω) id V Tr (σ.2 ω) hVT (fun _ hu => hu) t
      (outcome_contF (holoInnerAt T CT st hp).r id s ht) hG
    exact hno ⟨ω, 0, hr0, regClash_fork T CT hp zs dg N k Rw wmsg σ cap hu'
      ((outHL_lastCoin T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp) x hl hlc).symm.trans
        ho) hc⟩
  rcases FlockC3.Fork.holo_runs_sound T CT execArith execArith_correct Refine.H512 Refine.enc512 Refine.leafPlain
      (holoAt st hp) (FlockC3.Fork.holoAt_fast100 hp hcls) (FlockC3.Fork.holoAt_linkLayout hp hcls)
      (FlockC3.Fork.holoAt_rsch st hp) (FlockC3.Fork.holoAt_units st hp) x hx G s with
    ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hc | hc | hc | hc⟩ | hle
  · exact (hfork t t' o o' ht ht' hG hG' (outHL_of_plays T CT _ _ _ _ _ x ho) (outHL_of_plays T CT _ _ _ _ _ x ho')
      (Or.inl hc)).elim
  · exact (hfork t t' o o' ht ht' hG hG' (outHL_of_plays T CT _ _ _ _ _ x ho) (outHL_of_plays T CT _ _ _ _ _ x ho')
      (Or.inr hc)).elim
  · exact (hreg t o ht hG (outHL_of_plays T CT _ _ _ _ _ x ho) hc).elim
  · exact (hreg t' o' ht' hG' (outHL_of_plays T CT _ _ _ _ _ x ho') hc).elim
  have hlast : (holoInnerAt T CT st hp).LastFree := schedInner_lastFree T CT _ _ _ _
  refine le_trans ?_ hle
  refine le_trans (leafAvg_mono (G := fun t s =>
    if ∃ m, Q (id (relabelL (ref ω) m t)) ∧ V t s (relabelL (ref ω) m t) then 1 else 0)
    (fun t s => ite_one_zero_le ?_) _) ?_
  · rintro ⟨-, m, b, hb, ha, hd, hq⟩
    exact ⟨m, hq, b, hb, ha, hd⟩
  refine (leafAvg_relabelL_le Q (holoInnerAt T CT st hp).r f (ref ω) id V (σ.2 ω) fun u m ρ ρ' hu hq =>
    hlast x u m ρ ρ' hu hq).trans (le_of_eq ?_)
  refine Eq.trans (leafAvg_congr_len (G := fun u s' => prob (fun u => Q u ∧ G u) (.ret (id u)) s')
    (fun u s' _ => ?_) _) ?_
  · exact ite_one_zero_iff Iff.rfl
  exact (prob_rounds _ _ _ _).symm

/-- **Gap 2 at the committed rounds, the inner term given**: `zk_sessions_recursive_fork_innerC`'s argument with the
inner term `InnerAtLeavesL` a hypothesis wherever the recursive prover has no inner fork, in place of the inner
protocol's soundness. -/
theorem zk_sessions_recursive_fork_innerL
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hbr : Flock.Assumptions.VBridgeC zs dg N k Rw x wmsg Sk cmt msg)
    {D : Type} (H : List UInt8 → D) (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair)
    {εin : ℝ≥0∞} (hr0 : 0 < I.r)
    (hleaf : ¬ InnerForkRec zs dg N k Rw σ x wmsg H pick →
      ∀ ref, InnerAtLeavesL σ (fun R ω t o => AcceptsV (zs R ω t) o) x (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt)
        (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω)) (fun w => wmsg w.1) ref εin) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H pick ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases hf : InnerForkRec zs dg N k Rw σ x wmsg H pick
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  have hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (fun a b => k + 1 < I.r → a = b) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      ∃ m, l.zip (t.map Prod.snd) = relabelL (ref ω) m t := by
    intro ref ω t l ht h
    obtain ⟨m, hm⟩ := exists_of_rel (h (t.length - 1) (by omega)) (by
      simp only [ne_eq, Option.map_eq_none_iff, List.getElem?_eq_none_iff, Nat.not_le]; omega)
    exact ⟨m, zip_eq_relabelL (ref ω) m t l hm fun k hk =>
      option_eq_of_rel (R := fun a b => k + 1 < I.r → a = b) (fun _ _ hab => hab (by omega)) (h k (by omega))⟩
  rcases zk_sessions_recursive_fork_on hA3 zs dg N hk hRw hM hρ hr σ x wmsg hbr.1 (fun i _ a b => i + 1 < I.r → a = b)
      (fun ref ω t u => ∃ m, u = relabelL (ref ω) m t) hrd (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt msg
      hbr.2.1 hbr.2.2.1 hbr.2.2.2 (fun ref => InnerAtLeavesL.toZ (hleaf hf ref))
      with h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))

end Chain

end FlockSoundness.Discharge.Recursion
