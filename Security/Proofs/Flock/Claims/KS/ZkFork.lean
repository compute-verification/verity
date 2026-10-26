import Proofs.Flock.Claims.KS.ZkFork0

/-!
# The masked simulator's fork bound at its caps, from its alignment and its clash

`ZkSimForkLeOn x IsCapTauZ` is the masked simulator's fork bound at each rep's caps past level 0 (`IsCapZ`) and at its
`τ` (`IsTauZ`; `zkSimForkLeOn_capTau`). At the caps it splits in two, as plain's `PlainSimForkLe` does (`PlainFork`):

* `ZkForkAlign`: against the simulated strategy after `Commit` at table `j` (`zkSimAt`), the fork at rep `r`'s
  level-`ℓ + 1` table node (`advRSZ`) is at most the fork after `zkRd j i` coins of the event `ForkEvZ`: table `j`'s
  outputs in both runs record one cap `d` for that level and open a position of it to different rows;
* `ZkForkClash`: two accepting runs of `P` at the masked simulator's coins whose decoded live outcomes have `ForkEvZ`
  collide among the verifier's calls in the two.

The rest is the simulation's: a fork after `t` coins against the simulated strategy is at most the live fork after `t`
coins of the decoded event (`forkC_simAtJZ_le`, the dead branches' empty rows opening nothing, `opRZ_nul`), which is
the deployed runs' fork at round `t` (`forkC_liveStratZ`). `zkAdvCap_le_of` and `zkSimForkLeCap_of` put these
together, and `zk_KS_of_align` is masked KS from them.

`τ` has no such split. The dead branches answer rep `r`'s `τ` with `(0, default)` (`nulOpZ`), which opens under every
commitment `c` with `c 0 = Lτ.leaf 0 default`. So `advTauSZ` counts a fork pairing a dead run with a live run that
opens another `τ` there, a collision with an input no run hashed, and `forkC_simAtJZ_le` needs an event that no dead
output has.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

/-! ## The events -/

/-- A rep's cap event at a level past 0. -/
def IsCapZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inl _) => True
  | _ => False

/-- A rep's `τ` event. -/
def IsTauZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inr (.inr _)) => True
  | _ => False

namespace Gen

variable [S : ZkSetup]

/-- **`ZkSimForkLeOn` at the caps and `τ` from each.** -/
theorem zkSimForkLeOn_capTau {x : Claims.Msg} (hc : ZkSimForkLeOn x fun _ i => IsCapZ i)
    (ht : ZkSimForkLeOn x fun _ i => IsTauZ i) : ZkSimForkLeOn x fun _ i => IsCapTauZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((_ | (_ | _)) | ⟨⟩)
  · exact hc s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact False.elim hi
  · exact ht s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact False.elim hi

end Gen

/-- **`ZkSimForkLeOn` at the caps and `τ` from each.** -/
theorem zkSimForkLeOn_capTau {x : Claims.Msg} (hc : ZkSimForkLeOn x fun _ i => IsCapZ i)
    (ht : ZkSimForkLeOn x fun _ i => IsTauZ i) : ZkSimForkLeOn x fun _ i => IsCapTauZ i :=
  Gen.zkSimForkLeOn_capTau (S := S₀) (x := x) (hc := hc) (ht := ht)

section Event

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- Rep `r`'s recorded cap at index `ℓ`, its level-`ℓ + 1` cap, is `d`. -/
def CapAtZ (r ℓ : ℕ) (out : OutZC 𝔽 D Sl) (d : Model.Cap D) : Prop :=
  (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ]? = some d

/-- **The fork event at rep `r`'s level `ℓ + 1` of table `j`**: table `j`'s outputs in both runs record one cap `d` for
it and open a position of it to different rows. -/
def ForkEvZ (Hs : Hashes 𝔽 D Sl) (sch : Accounting.Schedule) (j r ℓ : ℕ) (a a' : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d, CapAtZ r ℓ (a.getD j dfltOutZC) d ∧ CapAtZ r ℓ (a'.getD j dfltOutZC) d ∧
    ∃ p, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
      opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d (outs.getD j dfltOutZC)) (a, a') p

/-- Table `j`'s output opens some position at rep `r`'s level `ℓ + 1`. -/
def OpensAtZ (Hs : Hashes 𝔽 D Sl) (sch : Accounting.Schedule) (j r ℓ : ℕ) (a : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d p, (opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d (a.getD j dfltOutZC) p).isSome

theorem forkEvZ_opens {Hs : Hashes 𝔽 D Sl} {sch : Accounting.Schedule} {j r ℓ : ℕ} {a a' : List (OutZC 𝔽 D Sl)}
    (h : ForkEvZ Hs sch j r ℓ a a') : OpensAtZ Hs sch j r ℓ a ∧ OpensAtZ Hs sch j r ℓ a' :=
  let ⟨d, _, _, p, h₁, h₂, _⟩ := h
  ⟨⟨d, p, h₁⟩, ⟨d, p, h₂⟩⟩

/-- **The dead branches' empty rows open nothing.** -/
theorem opRZ_nul (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (sch : Accounting.Schedule) (r ℓ : ℕ) (d : Model.Cap D) {out : OutZC 𝔽 D Sl} (h : out.opens = nulOpZ)
    (p : Fin (2 ^ (lvOf sch ℓ).logLen)) : opRZ H E Lf sch r ℓ d out p = none := by
  unfold opRZ Exec.opOfL
  rw [List.filter_eq_nil_iff.2 fun x hx => ?_]
  · rfl
  · simp only [decide_eq_true_eq, not_and]
    exact fun _ => not_verifiedL_nil H E Lf d _ _ (slotOpensZ_row_nil h hx)

theorem not_opensAtZ_nul {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) (sch : Accounting.Schedule)
    {j : ℕ} (hj : j < ts.length) (r ℓ : ℕ) (caps : Fin ts.length → Model.Cap D) (outs : List (OutZC 𝔽 D Sl))
    (h : ∀ i : Fin ts.length, ∃ pend, outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) :
    ¬ OpensAtZ Hs sch j r ℓ outs := by
  rintro ⟨d, p, hs⟩
  obtain ⟨pend, hp⟩ := h ⟨j, hj⟩
  have hp' : outs.getD j dfltOutZC = outJ Hs ts ⟨j, hj⟩ (caps ⟨j, hj⟩) pend nulOpZ := hp
  rw [hp', opRZ_nul Hs.H Hs.E Hs.Lf sch r (ℓ + 1) d rfl] at hs
  exact Bool.false_ne_true hs

end Event

/-! ## At the masked simulator -/

/-- **The simulated strategy after `Commit` at table `j` at the masked simulator's tape `u`**: `simAtJZ` at
`zkSim P`'s live strategy and its roots (`zkSim_root`), as `zkAdvSim` reads it. -/
noncomputable def zkSimAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The simulated strategy after `Commit` at table `j` at the masked simulator's tape `u`**: `simAtJZ` at
`zkSim P`'s live strategy and its roots (`zkSim_root`), as `zkAdvSim` reads it. -/
noncomputable def zkSimAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The strategies' fork advantage at a cap at tape `u`**: `zkAdvSim`'s term at rep `r`'s level `ℓ + 1`, `advRSZ`
against `zkSimAt`. -/
noncomputable def zkAdvCapAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (ℓ : ℕ) (u : zkTape x) :
    ℝ :=
  advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ + 1)
    (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

namespace Gen

variable [S : ZkSetup]

/-- **The strategies' fork advantage at a cap at tape `u`**: `zkAdvSim`'s term at rep `r`'s level `ℓ + 1`, `advRSZ`
against `zkSimAt`. -/
noncomputable def zkAdvCapAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
/-- **The simulated fork's alignment at a cap at tape `u`**: against `zkSimAt`, the fork at rep `r`'s level-`ℓ + 1`
table node is at most the fork after `t` coins of `ForkEvZ`. -/
def ZkAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
/-- **The simulated fork's alignment at a cap at tape `u`**: against `zkSimAt`, the fork at rep `r`'s level-`ℓ + 1`
table node is at most the fork after `t` coins of `ForkEvZ`. -/
def ZkAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The live fork event at the masked simulator**: the live outcomes at live coins `r₁` and `r₂` decode, at table
`j`'s verdict, to outputs with `ForkEvZ` at rep `r`'s level `ℓ + 1`. -/
def ZkLiveFork {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The live fork event at the masked simulator**: the live outcomes at live coins `r₁` and `r₂` decode, at table
`j`'s verdict, to outputs with `ForkEvZ` at rep `r`'s level `ℓ + 1`. -/
def ZkLiveFork {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The simulated fork is aligned with the masked fork rounds at the caps**: at every masked statement in scope that
sets up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep `r`'s
level-`ℓ + 1` table node against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of
`ForkEvZ` (`ZkAlignQ`). -/
def ZkForkAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val u

namespace Gen

variable [S : ZkSetup]

/-- **The simulated fork is aligned with the masked fork rounds at the caps**: at every masked statement in scope that
sets up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep `r`'s
level-`ℓ + 1` table node against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of
`ForkEvZ` (`ZkAlignQ`). -/
def ZkForkAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ZkAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ.val (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val u

end Gen

/-- **The live fork event's collision at the caps**: at every masked statement in scope that sets up at a draw, every
prover and tape, every table `j` and table of full-width columns, two accepting runs of `P` at the masked simulator's
coins whose decoded live outcomes have `ForkEvZ` collide among the verifier's calls in the two. -/
def ZkForkClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The live fork event's collision at the caps**: at every masked statement in scope that sets up at a draw, every
prover and tape, every table `j` and table of full-width columns, two accepting runs of `P` at the masked simulator's
coins whose decoded live outcomes have `ForkEvZ` collide among the verifier's calls in the two. -/
def ZkForkClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
/-- **The fork advantage at a cap from the alignment and the live fork's pair**: `zkAdvSim` at rep `r`'s level
`ℓ + 1` is at most the simulated fork at round `t`, where at each tape the fork at the slot's node is at most the fork
after `t` coins of `ForkEvZ` and the live fork event gives the deployed runs' pair event. -/
theorem zkAdvCap_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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
/-- **The fork advantage at a cap from the alignment and the live fork's pair**: `zkAdvSim` at rep `r`'s level
`ℓ + 1` is at most the simulated fork at round `t`, where at each tape the fork at the slot's node is at most the fork
after `t` coins of `ForkEvZ` and the live fork event gives the deployed runs' pair event. -/
theorem zkAdvCap_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **A live fork event's pair**: where `P'` aborts on rejected runs (`zkSimAbortAt`), the live fork event gives two
accepting runs, which collide by `ZkForkClash`. -/
theorem pairEvZ_of_forkClash {x : Claims.Msg} (hc : ZkForkClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **A live fork event's pair**: where `P'` aborts on rejected runs (`zkSimAbortAt`), the live fork event gives two
accepting runs, which collide by `ZkForkClash`. -/
theorem pairEvZ_of_forkClash {x : Claims.Msg} (hc : ZkForkClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The fork bound at the caps from the alignment and the clash**: `ZkSimForkLeOn` at `IsCapZ` from `ZkForkAlign` and
`ZkForkClash`, by `zkAdvCap_le_of` at the abort (`zkSimAbortAt`). -/
theorem zkSimForkLeCap_of {x : Claims.Msg} (ha : ZkForkAlign x) (hc : ZkForkClash x) :
    ZkSimForkLeOn x fun _ i => IsCapZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((⟨r, ℓ⟩ | _ | _) | ⟨⟩)
  · exact zkAdvCap_le_of hs hin hset h0 L hz hk hlay hd P ω j hj C r ℓ _
      (ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ)
      fun _ _ _ hq => pairEvZ_of_forkClash hc hs hin hset h0 L hz hk hlay hd hj hC hq
  all_goals exact False.elim hi

end Gen

/-- **The fork bound at the caps from the alignment and the clash**: `ZkSimForkLeOn` at `IsCapZ` from `ZkForkAlign` and
`ZkForkClash`, by `zkAdvCap_le_of` at the abort (`zkSimAbortAt`). -/
theorem zkSimForkLeCap_of {x : Claims.Msg} (ha : ZkForkAlign x) (hc : ZkForkClash x) :
    ZkSimForkLeOn x fun _ i => IsCapZ i :=
  Gen.zkSimForkLeCap_of (S := S₀) (x := x) (ha := ha) (hc := hc)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked verifier: KS, from the caps' alignment and clash, `τ`'s fork bound, and the pads' premises**,
at SHA-512 at the trimmed record at the deployed relation: `zk_KS_of_kinds` at `zkSimForkLeOn_capTau` and
`zkSimForkLeCap_of`. -/
theorem zk_KS_of_align (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_kinds (fun x hno => zkSimForkLeOn_capTau (zkSimForkLeCap_of (ha x hno) (hc x hno)) (ht x hno)) hp hpads

end Gen

/-- **The deployed masked verifier: KS, from the caps' alignment and clash, `τ`'s fork bound, and the pads' premises**,
at SHA-512 at the trimmed record at the deployed relation: `zk_KS_of_kinds` at `zkSimForkLeOn_capTau` and
`zkSimForkLeCap_of`. -/
theorem zk_KS_of_align (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_align (S := S₀) (ha := ha) (hc := hc) (ht := ht) (hp := hp) (hpads := hpads)

end FlockDeploy.KS
