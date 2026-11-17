module

public import Proofs.Flock.Claims.KS.ZkSched
public import Proofs.Flock.Claims.KS.PushSched

/-! The masked slot from a schedule whose tables pause and resume. -/

@[expose] public section

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

def DeadZ (n : ℕ) (outs : List (OutZC 𝔽 D Sl)) : Prop :=
  (outs.getD n dfltOutZC).opens = nulOpZ

theorem ind_conflictAcc_nul (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (sch : FlockSoundness.Accounting.Schedule) (n r ℓ : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)} (h : DeadZ n b) :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (opRAccZ H E Lf sch n r ℓ d) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨p, h₁, -, -⟩ := hc
    unfold opRAccZ at h₁
    split_ifs at h₁
    · rw [opRZ_nul H E Lf sch r ℓ d h p] at h₁
      simp at h₁
    · simp at h₁
  · rfl

def TabFollowZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (pts : Fin 2 → Fin mPts → 𝔽) (k n₀ : ℕ)
    (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (σ' : Strategy (batch (tabAfterZCP A Hs T cap₀ pts ::
      post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) : Prop :=
  ∃ σ₁, Lock.FollowL (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => Dead (x.1 :: x.2))
      (fun x => ∀ rest, Dead (x :: rest)) (some k)
      (tabAfterZCP A Hs T cap₀ pts) (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))
      (Strategy.ofMap (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2)
        ((tabAfterZCP A Hs T cap₀ pts).interleave
          (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) σ') σ₁ ∧
    Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os))
      (fun x => ∀ rest, Dead (x :: rest)) n₀ (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁

def SchedZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (Dead : List (OutZC 𝔽 D Sl) → Prop) (k n₀ n : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    Prop :=
  ∀ pts, Lock.PreSched (tabAfterZCP A Hs T cap₀ pts) (post.map (· pts)) k (pre.map (· pts)) Dead
    (TabFollowZ A Hs T cap₀ post pts k n₀) n (pushStrat A Hs T cap₀ pre post σ pts)

theorem slotRSZ_of_batch (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (ℓ n n₀ c : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (hΦ : ∀ d b b', Dead b → Dead b' → ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (opRAccZ Hs.H Hs.E Hs.Lf T.sch pre.length r (ℓ + 1) d) x p) (b, b') = 0)
    (hD : ∀ pts : Fin 2 → Fin mPts → 𝔽, Lock.Dist (slotOf r (ℓ + 1) + 4) c
      (lockRepZ A Hs T r hr default (Model.extraClaims T.S T.ptLocal mPts pts)))
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (hF : SchedZ A Hs T pre post cap₀ Dead (r * n₀ + c) n₀ n σ) :
    SlotRSZ A Hs T pre post cap₀ r ℓ (n + (r * n₀ + c) + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <| Lock.SlotAtD.slotAt hΦ <|
    Lock.SlotAtD.batchLock_sched dfltOutZC _ (post.map (· pts))
      (fun _ _ ⟨σ₁, hFL, hl⟩ => Lock.SlotAtD.batchLock_nil_followL dfltOutZC _ _ hFL
        (slotAtD_lockAfterZ A Hs T default cap₀ pts r hr ℓ n₀ c _ (hD pts) σ₁ hl))
      (pre.map (· pts)) (hF pts)

def TabFollowC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (pts : Fin 2 → Fin mPts → 𝔽) (k n₀ : ℕ)
    (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (σ' : Strategy (batch (tabAfterZC A Hs T cap₀ pts ::
      post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) : Prop :=
  ∃ σ₁, Lock.FollowL (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => Dead (x.1 :: x.2))
      (fun x => ∀ rest, Dead (x :: rest)) (some k)
      (tabAfterZC A Hs T cap₀ pts) (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))
      (Strategy.ofMap (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2)
        ((tabAfterZC A Hs T cap₀ pts).interleave
          (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) σ') σ₁ ∧
    Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os))
      (fun x => ∀ rest, Dead (x :: rest)) n₀ (repZC A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZC A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁

def SchedC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (Dead : List (OutZC 𝔽 D Sl) → Prop) (k n₀ n : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    Prop :=
  ∀ pts, Lock.PreSched (tabAfterZC A Hs T cap₀ pts) (post.map (· pts)) k (pre.map (· pts)) Dead
    (TabFollowC A Hs T cap₀ post pts k n₀) n (σ pts)

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem pushed_none : Pushed (D := D) (Lock.getPair (getP 0) (getP 1)) ((none, none), (none, none)) :=
  fun s c d h => by simp [Lock.getPair, getP] at h

omit [Fintype K] [Inhabited Sl] in
theorem tabFollowZ_of_c (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (pts : Fin 2 → Fin mPts → 𝔽) (k n₀ : ℕ)
    (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (σ' : Strategy (batch (tabAfterZC A Hs T cap₀ pts ::
      post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts)))
    (h : TabFollowC A Hs T cap₀ post pts k n₀ Dead σ') :
    TabFollowZ A Hs T cap₀ post pts k n₀ Dead (Lock.oracleStrat pushC
      (batchPush (Pushed (Lock.getPair (getP 0) (getP 1))) (pushTab A Hs T cap₀ pts)
        (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts) []) σ') := by
  obtain ⟨σ₁, hF, hL⟩ := h
  refine ⟨Lock.oracleStrat pushC (pushTab A Hs T cap₀ pts) σ₁, ?_,
    lead_push disjoint_getP (fun _ _ _ hr hp => hr hp) hL _ _ pushed_none⟩
  have e := ofMap_oracleStrat_map
    (R₂ := fun i (a b : List (OutZC 𝔽 D Sl)) => Pushed (Lock.getPair (getP 0) (getP 1)) i → a = b)
    (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2)
    (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2) (fun _ _ _ h hp => by rw [h.1 hp, h.2])
    (Lock.leftSame (pushTab A Hs T cap₀ pts)
      (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) σ'
  exact (congrArg (Lock.FollowL _ _ _ _ _ · _) e).mpr
    (followL_push (fun _ _ _ hr hp => hr hp) hF (pushTab A Hs T cap₀ pts) pushed_none)

omit [Fintype K] [Inhabited Sl] in
theorem schedZ_of_schedC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (Dead : List (OutZC 𝔽 D Sl) → Prop) (k n₀ n : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (h : SchedC A Hs T pre post cap₀ Dead k n₀ n σ) : SchedZ A Hs T pre post cap₀ Dead k n₀ n σ := fun pts =>
  preSched_push (pushTab A Hs T cap₀ pts) _ pushed_none
    (fun Dead σ' h => tabFollowZ_of_c A Hs T cap₀ post pts k n₀ Dead σ' h) _ (h pts)

omit [Inhabited Sl] in
theorem noSlot_lockAfterZ_ge (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (ℓ : ℕ)
    (hℓ : T.sch.levels.length ≤ ℓ + 1) :
    Lock.NoSlot (slotOf r (ℓ + 1) + 4) (lockAfterZ A Hs T C₀ cap₀ pts) := by
  unfold lockAfterZ
  refine Lock.NoSlot.bind (fun _ _ _ _ => .send _ fun _ => .ret _) ?_
  match r, hr with
  | 0, _ =>
    exact (noSlot_lockRepZ_ge A Hs T 0 _ C₀ _ ℓ hℓ).interleaveLock disjoint_getZ
      (Lock.noSlot_of_get_none (fun i => by simp [getZ, getRep, slotOf]) _)
  | 1, _ =>
    exact (Lock.noSlot_of_get_none (fun i => by simp [getZ, getRep, slotOf]) _).interleaveLock disjoint_getZ
      (noSlot_lockRepZ_ge A Hs T 1 _ C₀ _ ℓ hℓ)

theorem slotRSZ_of_ge (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (ℓ t : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (hℓ : T.sch.levels.length ≤ ℓ + 1) :
    SlotRSZ A Hs T pre post cap₀ r ℓ t σ := by
  unfold SlotRSZ lockSessZ
  refine Lock.SlotAt.of_noSlot (.coin _ fun pts => ?_) _
  exact Lock.NoSlot.mono _ (Lock.NoSlot.batchLock dfltOutZC
    (Lock.NoSlot.mapL _ _ (noSlot_lockAfterZ_ge A Hs T default cap₀ pts r hr ℓ hℓ)) _ _)

end Session

def ZkBatchAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ℓ.val + 1 < (zkTabs h0 L hz hk hlay j).sch.levels.length →
      SchedC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) (DeadZ j)
        (r.val * repRounds true spec sts[zkTbl sts j]! + zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val)
        (repRounds true spec sts[zkTbl sts j]!) (zkFront spec sts j)
        (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

namespace Gen

variable [S : ZkSetup]

def ZkBatchAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    ℓ.val + 1 < (zkTabs h0 L hz hk hlay j).sch.levels.length →
      SchedC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) (DeadZ j)
        (r.val * repRounds true spec sts[zkTbl sts j]! + zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val)
        (repRounds true spec sts[zkTbl sts j]!) (zkFront spec sts j)
        (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

theorem zkFrontAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkFrontAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u
  by_cases hℓ : ℓ.val + 1 < (zkTabs h0 L hz hk hlay j).sch.levels.length
  · obtain ⟨-, -, -, -, -, hl⟩ := zkTabs_facts h0 L hz hk hlay j
    refine slotRSZ_of_batch _ _ _ _ _ _ r.val r.isLt ℓ.val _ _ _ (DeadZ j) ?_
      (fun pts => dist_lockRepZ_zkCap _ _ _ hl r.val r.isLt default ℓ.val pts) _
      (schedZ_of_schedC _ _ _ _ _ _ _ _ _ _ _ (h s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u hℓ))
    intro e b b' hb _
    rw [take_gamesZC_length]
    exact ind_conflictAcc_nul _ _ _ _ _ _ _ _ hb
  · exact slotRSZ_of_ge _ _ _ _ _ _ r.val r.isLt ℓ.val _ _ (by omega)

end Gen

theorem zkFrontAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkFrontAlign x :=
  Gen.zkFrontAlign_of_batch (S := S₀) (x := x) (h := h)

namespace Gen

variable [S : ZkSetup]

theorem zkSlotAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkSlotAlign x :=
  zkSlotAlign_of_front (zkFrontAlign_of_batch h)

end Gen

theorem zkSlotAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkSlotAlign x :=
  Gen.zkSlotAlign_of_batch (S := S₀) (x := x) (h := h)

end FlockDeploy.KS
