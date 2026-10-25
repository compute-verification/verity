import Proofs.Flock.Claims.KS.ZkPadsPair

/-!
# The masked simulator's fork bound at `τ`, from its alignment and its clash

`opTauZ` reads accepted outputs only. A dead branch answers the openings with `nulOpZ`, whose pads paths are empty,
and each of its reps' outcomes is an outcome of the rep's game at the session's link points (`Claims.Bridge.DeadAt`),
whose inner check climbs every query's path through the pads tree's `log₂ N ≥ 1` levels (`out_repZCAt_nul`). So no
dead output is accepted, and none opens a `τ` commitment (`not_opensTauZ_nul`); a fork after `t` coins against the
simulated strategy is then at most the live fork after `t` coins of the decoded event (`forkC_simAtJZ_le_out`).

What is left at `τ` splits as at the caps:

* `ZkTauAlign`: against the simulated strategy after `Commit` at table `j`, the fork at rep `r`'s `τ` node
  (`advTauSZ`) is at most the fork after `zkRd j i` coins of `TauForkEvZ`: table `j`'s outputs in both runs are
  accepted and open one cap `d`'s `τ` commitment to different values;
* `ZkTauClash`: two accepting runs of `P` at the masked simulator's coins whose decoded live outcomes have `TauForkEvZ`
  collide among the verifier's calls in the two.

`zkAdvTau_le_of` and `zkSimForkLeTau_of` put these together. `ZkTauPair` proves the clash (`zkTauClash`), so the bound
rests on the alignment alone (`zkSimForkLeTau_of_align`).
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Dead

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Composed (outJ repJ)

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
/-- **A compiled rep's inner check rejects the empty openings**: it climbs every query's path through the pads tree's
`log₂ N ≥ 1` levels, and the empty openings' paths are empty. -/
theorem out_repZCAt_nul [Inhabited D] [Inhabited Sl] (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl)
    (T : TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) {p : PendZ 𝔽 D Sl}
    (h : Out (repZCAt A Hs T pts r) p) (hN : 2 ≤ T.N) (hq : 0 < T.q) (r' : ℕ) :
    ¬ p.inner ((nulOpZ (D := D) (Sl := Sl)).inner r') := by
  unfold repZCAt repZC at h
  obtain ⟨_, -, h⟩ := Out.of_bind h
  obtain ⟨_, -, h⟩ := Out.of_bind h
  obtain ⟨v, hv, h⟩ := Out.of_bind h
  change Out (.ret _) p at h
  cases h
  unfold innerGameC at hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  obtain ⟨_, -, hv⟩ := Out.of_bind hv
  change Out (.ret _) v at hv
  cases hv
  rintro ⟨-, -, hall⟩
  have h1 : ([] : List D).length = Nat.log 2 T.N := (hall ⟨0, hq⟩).2.1
  have h2 : 0 < Nat.log 2 T.N := Nat.log_pos (by norm_num) hN
  simp only [List.length_nil] at h1
  omega

open Classical in
/-- An output whose rep `0` rejects its inner openings is not accepted. -/
theorem accepted_of_inner (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap : Model.Cap D)
    {pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl} {os : OpensZ 𝔽 D Sl} (h : ¬ pend.1.inner (os.inner 0)) :
    (outZC Hs T cap pend os).accepted = false := by
  simp only [outZC, Bool.and_eq_false_iff, decide_eq_false_iff_not, not_and]
  exact Or.inr fun h1 => absurd h1 h

variable [Inhabited D] [Inhabited Sl]

/-- **The `τ` fork event at rep `r` of table `j`**: table `j`'s outputs in both runs are accepted and open one cap
`d`'s `τ` commitment to different values. -/
def TauForkEvZ (Hs : Hashes 𝔽 D Sl) (j r : ℕ) (a a' : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d : Model.Cap D, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
    opTauZ Hs r d (outs.getD j dfltOutZC)) (a, a') ()

/-- Table `j`'s output is accepted and opens some cap's `τ` commitment at rep `r`. -/
def OpensTauZ (Hs : Hashes 𝔽 D Sl) (j r : ℕ) (a : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ d : Model.Cap D, (opTauZ Hs r d (a.getD j dfltOutZC) ()).isSome

theorem tauForkEvZ_opens {Hs : Hashes 𝔽 D Sl} {j r : ℕ} {a a' : List (OutZC 𝔽 D Sl)}
    (h : TauForkEvZ Hs j r a a') : OpensTauZ Hs j r a ∧ OpensTauZ Hs j r a' :=
  let ⟨d, h₁, h₂, _⟩ := h
  ⟨⟨d, h₁⟩, ⟨d, h₂⟩⟩

omit [Fintype K] in
/-- **The dead branches open no `τ` commitment**: their outputs, at the empty openings on their reps' outcomes, are
not accepted (`out_repZCAt_nul`). -/
theorem not_opensTauZ_nul (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) {j : ℕ}
    (hj : j < ts.length) (hN : 2 ≤ (ts.get ⟨j, hj⟩).N) (hq : 0 < (ts.get ⟨j, hj⟩).q) (r : ℕ)
    (caps : Fin ts.length → Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (outs : List (OutZC 𝔽 D Sl))
    (h : ∀ i : Fin ts.length, ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl, Out (repJ A Hs ts i pts 0) pend.1 ∧
      Out (repJ A Hs ts i pts 1) pend.2 ∧ outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) :
    ¬ OpensTauZ Hs j r outs := by
  rintro ⟨d, hs⟩
  obtain ⟨pend, h0, -, hp⟩ := h ⟨j, hj⟩
  have hp' : outs.getD j dfltOutZC = outJ Hs ts ⟨j, hj⟩ (caps ⟨j, hj⟩) pend nulOpZ := hp
  have ha : (outJ Hs ts ⟨j, hj⟩ (caps ⟨j, hj⟩) pend nulOpZ).accepted = false :=
    accepted_of_inner Hs _ _ (out_repZCAt_nul A Hs (ts.get ⟨j, hj⟩) pts 0 h0 hN hq 0)
  rw [hp'] at hs
  unfold opTauZ at hs
  simp [ha] at hs

end Dead

section Live

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

/-- The deployed masked tables make `Zk.Q` queries. -/
theorem zkTabs_q {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (j : ℕ) :
    (zkTabs h0 L hz hk hlay j).q = Zk.Q := by
  unfold zkTabs
  split <;> rfl

open Classical in
/-- **The simulated fork's alignment at rep `r`'s `τ` at tape `u`**: against the simulated strategy after `Commit`
at table `j`, the fork at rep `r`'s `τ` node (`advTauSZ`, `zkAdvSim`'s `τ` term) is at most the fork after `t` coins
of `TauForkEvZ`. -/
def ZkTauAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (t : ℕ)
    (u : zkTape x) : Prop :=
  advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _)) ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) => TauForkEvZ ZkExec.hs512 j r.val p.1 p.2) (a, a')) t
      _ (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The simulated fork's alignment at rep `r`'s `τ` at tape `u`**: against the simulated strategy after `Commit`
at table `j`, the fork at rep `r`'s `τ` node (`advTauSZ`, `zkAdvSim`'s `τ` term) is at most the fork after `t` coins
of `TauForkEvZ`. -/
def ZkTauAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (t : ℕ)
    (u : zkTape x) : Prop :=
  advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _)) ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) => TauForkEvZ ZkExec.hs512 j r.val p.1 p.2) (a, a')) t
      _ (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))

end Gen

/-- **The simulated fork is aligned with the masked fork rounds at `τ`**: at every masked statement in scope that sets
up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep `r`'s `τ` node
against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of `TauForkEvZ`
(`ZkTauAlignQ`). -/
def ZkTauAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (u : zkTape x),
    ZkTauAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r
      (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inr r)))).val u

namespace Gen

variable [S : ZkSetup]

/-- **The simulated fork is aligned with the masked fork rounds at `τ`**: at every masked statement in scope that sets
up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep `r`'s `τ` node
against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of `TauForkEvZ`
(`ZkTauAlignQ`). -/
def ZkTauAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (u : zkTape x),
    ZkTauAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r
      (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inr r)))).val u

end Gen

/-- **The live `τ` fork event's collision**: at every masked statement in scope that sets up at a draw, every prover
and tape, every table `j` and table of full-width columns, two accepting runs of `P` at the masked simulator's coins
whose decoded live outcomes have `TauForkEvZ` collide among the verifier's calls in the two. -/
def ZkTauClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2),
      LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
        (fun y => TauForkEvZ ZkExec.hs512 j r.val y.1 y.2)
        (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
          spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
          sha512 (zkSim P x spec sts) x (ω, u) r₁)
        (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
          spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
          sha512 (zkSim P x spec sts) x (ω, u) r₂) →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).out = .acc →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).out = .acc →
      Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).vLog ++
        (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).vLog)

namespace Gen

variable [S : ZkSetup]

/-- **The live `τ` fork event's collision**: at every masked statement in scope that sets up at a draw, every prover
and tape, every table `j` and table of full-width columns, two accepting runs of `P` at the masked simulator's coins
whose decoded live outcomes have `TauForkEvZ` collide among the verifier's calls in the two. -/
def ZkTauClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ (u : zkTape x) (r₁ r₂ : Fin (zkN x + 2) → LCoin 65536) (r : Fin 2),
      LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
        (fun y => TauForkEvZ ZkExec.hs512 j r.val y.1 y.2)
        (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
          spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
          sha512 (zkSim P x spec sts) x (ω, u) r₁)
        (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
          spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
          sha512 (zkSim P x spec sts) x (ω, u) r₂) →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).out = .acc →
      (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).out = .acc →
      Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).vLog ++
        (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).vLog)

open Classical in
/-- **The fork advantage at rep `r`'s `τ` from the alignment and the clash**: `zkAdvSim` at `τ` is at most the
simulated fork at round `t`, where at each tape the fork at the `τ` node is at most the fork after `t` coins of
`TauForkEvZ`: that fork is at most the live fork of the decoded event (`forkC_simAtJZ_le_out`, the dead branches
opening no `τ`, `not_opensTauZ_nul`), the deployed runs' fork at round `t` (`forkC_liveStratZ`), whose event gives two
accepting runs (`zkSimAbortAt`) that collide by `ZkTauClash`. -/
theorem zkAdvTau_le_of {x : Claims.Msg} (hc : ZkTauClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (hC : ∀ p, wideJ true x j (C p))
    (r : Fin 2) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkTauAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r t.val u) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inr (.inr r))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val := by
  have hk' := zkN_add hset
  have hj' : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := (finJ _ hj).isLt
  have hN : 2 ≤ ((tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).get ⟨j, hj'⟩).N := by
    rw [tabsJ_get]
    exact two_le_zkTabsN h0 L hz hk hlay j
  have hq : 0 < ((tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).get ⟨j, hj'⟩).q := by
    rw [tabsJ_get]
    show 0 < (zkTabs h0 L hz hk hlay j).q
    rw [zkTabs_q]
    decide
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAtJZ_le_out execArith ZkExec.hs512 _
    (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C) (AccTxJ ins.read.I d j (zkRoot P x ω) C)
    (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω) _
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _)
    (Ev := fun p => TauForkEvZ ZkExec.hs512 j r.val p.1 p.2)
    (Q := OpensTauZ ZkExec.hs512 j r.val)
    (fun _ h => tauForkEvZ_opens h)
    (fun caps pts outs h => not_opensTauZ_nul execArith ZkExec.hs512 _ hj' hN hq r.val caps pts outs h) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStratZ zkN spec.dg
    (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
    (zkSim P x spec sts) x (ω, u) _ ⟨t.val, by have := t.isLt; omega⟩)) ?_
  refine Claims.Ex_mono fun rr => ind_le_indZ fun h => ?_
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have hacc : ∀ (r' : Fin (zkN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
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
  exact ⟨h₁, h₂, hc s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u _ _ r h h₁ h₂⟩

end Gen

open Classical in
/-- **The fork advantage at rep `r`'s `τ` from the alignment and the clash**: `zkAdvSim` at `τ` is at most the
simulated fork at round `t`, where at each tape the fork at the `τ` node is at most the fork after `t` coins of
`TauForkEvZ`: that fork is at most the live fork of the decoded event (`forkC_simAtJZ_le_out`, the dead branches
opening no `τ`, `not_opensTauZ_nul`), the deployed runs' fork at round `t` (`forkC_liveStratZ`), whose event gives two
accepting runs (`zkSimAbortAt`) that collide by `ZkTauClash`. -/
theorem zkAdvTau_le_of {x : Claims.Msg} (hc : ZkTauClash x) {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (hC : ∀ p, wideJ true x j (C p))
    (r : Fin 2) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkTauAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r t.val u) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inr (.inr r))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val :=
  Gen.zkAdvTau_le_of (S := S₀) (x := x) (hc := hc) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec)
    (sts := sts) (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω)
    (P := P) (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (r := r) (t := t) (halign := halign)

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at `τ` from the alignment and the clash**: `ZkSimForkLeOn` at the `τ` events from `ZkTauAlign`
and `ZkTauClash` (`zkAdvTau_le_of`). -/
theorem zkSimForkLeTau_of {x : Claims.Msg} (ha : ZkTauAlign x) (hc : ZkTauClash x) :
    ZkSimForkLeOn x fun _ i => ∃ r, i = .inl (.inr (.inr r)) := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  obtain ⟨r, rfl⟩ := hi
  exact zkAdvTau_le_of hc hs hin hset h0 L hz hk hlay hd P ω j hj C hC r _
    (ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r)

end Gen

/-- **The fork bound at `τ` from the alignment and the clash**: `ZkSimForkLeOn` at the `τ` events from `ZkTauAlign`
and `ZkTauClash` (`zkAdvTau_le_of`). -/
theorem zkSimForkLeTau_of {x : Claims.Msg} (ha : ZkTauAlign x) (hc : ZkTauClash x) :
    ZkSimForkLeOn x fun _ i => ∃ r, i = .inl (.inr (.inr r)) :=
  Gen.zkSimForkLeTau_of (S := S₀) (x := x) (ha := ha) (hc := hc)

end Live

end FlockDeploy.KS
