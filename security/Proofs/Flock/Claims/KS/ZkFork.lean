module

public import Proofs.Flock.Claims.KS.ZkFork0

/-! The masked simulator's fork bound at its caps, from its alignment and its clash; `τ` has no such split, since the
dead branches' `τ` opens under every commitment `c` with `c 0 = Lτ.leaf 0 default`. -/

@[expose] public section

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

def IsCapZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inl _) => True
  | _ => False

def IsTauZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inr (.inr _)) => True
  | _ => False

namespace Gen

variable [S : ZkSetup]

theorem zkSimForkLeOn_capTau {x : Claims.Msg} (hc : ZkSimForkLeOn x fun _ i => IsCapZ i)
    (ht : ZkSimForkLeOn x fun _ i => IsTauZ i) : ZkSimForkLeOn x fun _ i => IsCapTauZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((_ | (_ | _)) | ⟨⟩)
  · exact hc s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact False.elim hi
  · exact ht s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact False.elim hi

end Gen

theorem zkSimForkLeOn_capTau {x : Claims.Msg} (hc : ZkSimForkLeOn x fun _ i => IsCapZ i)
    (ht : ZkSimForkLeOn x fun _ i => IsTauZ i) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  Gen.zkSimForkLeOn_capTau (S := S₀) (x := x) (hc := hc) (ht := ht)

section Event

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

def CapAtZ (r ℓ : ℕ) (out : OutZC 𝔽 D Sl) (d : Model.Cap D) : Prop :=
  (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ]? = some d

def ForkEvZ (Hs : Hashes 𝔽 D Sl) (sch : FlockSoundness.Accounting.Schedule) (j r ℓ : ℕ) (a a' : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d, CapAtZ r ℓ (a.getD j dfltOutZC) d ∧ CapAtZ r ℓ (a'.getD j dfltOutZC) d ∧
    ∃ p, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
      opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d (outs.getD j dfltOutZC)) (a, a') p

def OpensAtZ (Hs : Hashes 𝔽 D Sl) (sch : FlockSoundness.Accounting.Schedule) (j r ℓ : ℕ) (a : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d p, (opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d (a.getD j dfltOutZC) p).isSome

theorem forkEvZ_opens {Hs : Hashes 𝔽 D Sl} {sch : FlockSoundness.Accounting.Schedule} {j r ℓ : ℕ} {a a' : List (OutZC 𝔽 D Sl)}
    (h : ForkEvZ Hs sch j r ℓ a a') : OpensAtZ Hs sch j r ℓ a ∧ OpensAtZ Hs sch j r ℓ a' :=
  let ⟨d, _, _, p, h₁, h₂, _⟩ := h
  ⟨⟨d, p, h₁⟩, ⟨d, p, h₂⟩⟩

theorem opRZ_nul (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (sch : FlockSoundness.Accounting.Schedule) (r ℓ : ℕ) (d : Model.Cap D) {out : OutZC 𝔽 D Sl} (h : out.opens = nulOpZ)
    (p : Fin (2 ^ (lvOf sch ℓ).logLen)) : opRZ H E Lf sch r ℓ d out p = none := by
  unfold opRZ Exec.opOfL
  rw [List.filter_eq_nil_iff.2 fun x hx => ?_]
  · rfl
  · simp only [decide_eq_true_eq, not_and]
    exact fun _ => not_verifiedL_nil H E Lf d _ _ (slotOpensZ_row_nil h hx)

theorem not_opensAtZ_nul {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) (sch : FlockSoundness.Accounting.Schedule)
    {j : ℕ} (hj : j < ts.length) (r ℓ : ℕ) (caps : Fin ts.length → Model.Cap D) (outs : List (OutZC 𝔽 D Sl))
    (h : ∀ i : Fin ts.length, ∃ pend, outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) :
    ¬ OpensAtZ Hs sch j r ℓ outs := by
  rintro ⟨d, p, hs⟩
  obtain ⟨pend, hp⟩ := h ⟨j, hj⟩
  have hp' : outs.getD j dfltOutZC = outJ Hs ts ⟨j, hj⟩ (caps ⟨j, hj⟩) pend nulOpZ := hp
  rw [hp', opRZ_nul Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d rfl] at hs
  exact Bool.false_ne_true hs

end Event

noncomputable def zkSimAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (u : zkTape x) :=
  simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
    (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
    (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
      (fun _ => spec.mPts) zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
      (zkSim P x spec sts) x (ω, u))
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _)

namespace Gen

variable [S : ZkSetup]

noncomputable def zkSimAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (u : zkTape x) :=
  simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
    (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
    (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
      (fun _ => spec.mPts) zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
      (zkSim P x spec sts) x (ω, u))
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _)

end Gen

noncomputable def zkAdvCapAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (ℓ : ℕ) (u : zkTape x) :
    ℝ :=
  advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ + 1)
    (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

namespace Gen

variable [S : ZkSetup]

noncomputable def zkAdvCapAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (ℓ : ℕ) (u : zkTape x) :
    ℝ :=
  advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ + 1)
    (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

theorem zkAdvSim_cap {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inl (r, ℓ))) =
      (∑ u, zkAdvCapAt hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val u) / Fintype.card (zkTape x) :=
  rfl

end Gen

theorem zkAdvSim_cap {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inl (r, ℓ))) =
      (∑ u, zkAdvCapAt hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val u) / Fintype.card (zkTape x) :=
  Gen.zkAdvSim_cap (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (ℓ := ℓ)

open Classical in
def ZkAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (ℓ t : ℕ)
    (u : zkTape x) : Prop :=
  zkAdvCapAt hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ p.1 p.2) (a, a')) t
      _ (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

namespace Gen

variable [S : ZkSetup]

open Classical in
def ZkAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (ℓ t : ℕ)
    (u : zkTape x) : Prop :=
  zkAdvCapAt hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ p.1 p.2) (a, a')) t
      _ (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

end Gen

def ZkLiveFork {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ) (C : Fin (wordLenJ true x j) → List 𝔽)
    (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
    (fun y => ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ y.1 y.2)
    (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
      spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      sha512 (zkSim P x spec sts) x (ω, u) r₁)
    (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
      spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      sha512 (zkSim P x spec sts) x (ω, u) r₂)

namespace Gen

variable [S : ZkSetup]

def ZkLiveFork {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ) (C : Fin (wordLenJ true x j) → List 𝔽)
    (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
    (fun y => ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ y.1 y.2)
    (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
      spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      sha512 (zkSim P x spec sts) x (ω, u) r₁)
    (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
      spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      sha512 (zkSim P x spec sts) x (ω, u) r₂)

end Gen

def ZkForkAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val u

namespace Gen

variable [S : ZkSetup]

def ZkForkAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val u

end Gen

def ZkForkClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2)
      (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length),
      ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).out = .acc →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).out = .acc →
      Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).vLog ++
        (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).vLog)

namespace Gen

variable [S : ZkSetup]

def ZkForkClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2)
      (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length),
      ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).out = .acc →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).out = .acc →
      Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).vLog ++
        (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).vLog)

open Classical in
theorem zkAdvCap_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val t.val u)
    (hpair : ∀ u r₁ r₂, ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂))) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inl (r, ℓ))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val := by
  have hk' := zkN_add hset
  have hj' : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := (finJ _ hj).isLt
  rw [zkAdvSim_cap]
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAtJZ_le execArith ZkExec.hs512 _
    (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C) (AccTxJ ins.read.I d j (zkRoot P x ω) C)
    (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω) _
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _) (Ev := fun p => ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ.val p.1 p.2)
    (Q := OpensAtZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ.val) (fun _ h => forkEvZ_opens h)
    (fun caps outs h => not_opensAtZ_nul ZkExec.hs512 _ _ hj' r.val ℓ.val caps outs h) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStratZ zkN spec.dg
    (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
    (zkSim P x spec sts) x (ω, u) _ ⟨t.val, by have := t.isLt; omega⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_indZ fun h => hpair u _ _ h

end Gen

open Classical in
theorem zkAdvCap_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val t.val u)
    (hpair : ∀ u r₁ r₂, ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂))) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inl (r, ℓ))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val :=
  Gen.zkAdvCap_le_of (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (ℓ := ℓ) (t := t) (halign := halign) (hpair := hpair)

namespace Gen

variable [S : ZkSetup]

theorem pairEvZ_of_forkClash {x : Claims.Msg} (hc : ZkForkClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536} {r : Fin 2}
    {ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length}
    (h : ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) := by
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have hacc : ∀ (r' : Fin (zkN x + 2) → LCoin 65536) a, Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := fun r' _ ha => by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  have h₁ := hacc _ _ ha
  have h₂ := hacc _ _ ha'
  exact ⟨h₁, h₂, hc s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r₁ r₂ r ℓ h h₁ h₂⟩

end Gen

theorem pairEvZ_of_forkClash {x : Claims.Msg} (hc : ZkForkClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536} {r : Fin 2}
    {ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length}
    (h : ZkLiveFork hs hin hset h0 L hz hk hlay d P ω j C u r₁ r₂ r ℓ.val) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) :=
  Gen.pairEvZ_of_forkClash (S := S₀) (x := x) (hc := hc) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec)
    (sts := sts) (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω)
    (P := P) (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r₁ := r₁) (r₂ := r₂) (r := r) (ℓ := ℓ) (h := h)

namespace Gen

variable [S : ZkSetup]

theorem zkSimForkLeCap_of {x : Claims.Msg} (ha : ZkForkAlign x) (hc : ZkForkClash x) :
    ZkSimForkLeOn x fun _ i => IsCapZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((⟨r, ℓ⟩ | _ | _) | ⟨⟩)
  · exact zkAdvCap_le_of hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ _
      (ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ)
      fun _ _ _ hq => pairEvZ_of_forkClash hc hs hin hset h0 L hz hk hlay hd hj hC hq
  all_goals exact False.elim hi

end Gen

theorem zkSimForkLeCap_of {x : Claims.Msg} (ha : ZkForkAlign x) (hc : ZkForkClash x) :
    ZkSimForkLeOn x fun _ i => IsCapZ i :=
  Gen.zkSimForkLeCap_of (S := S₀) (x := x) (ha := ha) (hc := hc)

namespace Gen

variable [S : ZkSetup]

theorem zk_KS_of_align (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_kinds (fun x hno => zkSimForkLeOn_capTau (zkSimForkLeCap_of (ha x hno) (hc x hno)) (ht x hno)) hp hpads

end Gen

theorem zk_KS_of_align (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_align (S := S₀) (ha := ha) (hc := hc) (ht := ht) (hp := hp) (hpads := hpads)

end FlockDeploy.KS
