import Proofs.Flock.Claims.KS.ZkSched
import Proofs.Flock.Claims.KS.PushSched

/-!
# The masked slot from a schedule whose tables pause and resume

`slotRSZ_of_sched` asks that the primed prover play each table in front of the session's to its end before the next,
and the session's table to its end before the ones after it. The masked simulator does neither: after its live rounds
it sends every table's final messages and openings (`simEndJ`). Its schedule is `Lock.PreSched`: each table in front
advances until only sends are left in it, after its coins, and then waits until the session's table is past the slot
(`Lock.FollowR`); then the session's table follows its own strategy, rep 0 first, while the tables after it send at any
time; once it is past the slot, the strategy may play anything that keeps the table's dead runs dead (`Lock.FollowL`),
as the simulator does when it sends the table's openings after the later tables' coins. The count is the same:

* `ind_conflictAcc_nul`: `advRSZ`'s event is `0` at a run whose session table opens nothing (`nulOpZ`), as every
  dead branch of the simulator does;
* `TabFollowZ`, `SchedZ`, `slotRSZ_of_batch`: where at every link point the primed prover plays the tables in front by
  `Lock.PreSched` after `n` coins and then the session's table by `TabFollowZ`, rep 0 first after `n₀` coins, `SlotRSZ`
  holds after `n + (r · n₀ + c) + 1` coins, `c` a distance of rep `r`'s cap;
* `noSlot_lockAfterZ_ge`, `slotRSZ_of_ge`: past the last cap there is no such node, and `SlotRSZ` holds at every
  count;
* `TabFollowC`, `SchedC`, `schedZ_of_schedC`: the same schedule played by the compiled prover is played by the primed
  one (`preSched_push`, `followL_push`, `lead_push`);
* `ZkBatchAlign`, `zkFrontAlign_of_batch`: so `ZkFrontAlign`, and with it `ZkSlotAlign`, holds where the simulated
  strategy follows that schedule at the deployed tables' counts.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

/-- **Table `n` opens nothing.** -/
def DeadZ (n : ℕ) (outs : List (OutZC 𝔽 D Sl)) : Prop :=
  (outs.getD n dfltOutZC).opens = nulOpZ

/-- **The filtered conflict is `0` at a run whose table `n` opens nothing.** -/
theorem ind_conflictAcc_nul (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (sch : Accounting.Schedule) (n r ℓ : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)} (h : DeadZ n b) :
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

/-- **The session's table follows its own strategy, rep 0 first, beside the tables after it**: past the tables in
front, the primed prover's strategy `σ'` plays the session's table as `σ₁` does, the tables after it moving only by
sends, for `k` of that table's coins, and from there on keeping its runs `Dead` wherever `σ₁`'s are (`Lock.FollowL`),
and `σ₁` advances rep 0 until it ends after `n₀` coins (`Lock.Lead`), except below parts every reached outcome of which
is `Dead`. -/
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

/-- **The primed prover's schedule**: at every link point it plays the tables in front of the session's by
`Lock.PreSched`, after `n` coins, then the session's table by `TabFollowZ`. -/
def SchedZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (Dead : List (OutZC 𝔽 D Sl) → Prop) (k n₀ n : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    Prop :=
  ∀ pts, Lock.PreSched (tabAfterZCP A Hs T cap₀ pts) (post.map (· pts)) k (pre.map (· pts)) Dead
    (TabFollowZ A Hs T cap₀ post pts k n₀) n (pushStrat A Hs T cap₀ pre post σ pts)

/-- **The slot from a schedule whose tables pause**: `SlotRSZ` after `n + (r · n₀ + c) + 1` coins, `c` a distance of
rep `r`'s cap, where the primed prover follows `SchedZ` with `r · n₀ + c` coins of the session's table before the
tables after it flip any, short of runs `Dead` at which `advRSZ`'s event is `0` (`Lock.SlotAtD.batchLock_sched`,
`Lock.SlotAtD.batchLock_nil_followL`, `slotAtD_lockAfterZ`). -/
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

/-- **`TabFollowZ` played by the compiled prover**: past the tables in front, `σ'` plays the session's table as `σ₁`
does, the tables after it moving only by sends, for `k` of that table's coins, and from there on keeping its runs
`Dead` wherever `σ₁`'s are, and `σ₁` advances rep 0 until it ends after `n₀` coins, except below parts every reached
outcome of which is `Dead`. -/
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

/-- **The compiled prover's schedule**: `SchedZ`'s, played by the compiled prover. -/
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
/-- **The session's table pushed follows the primed prover of `σ₁`.** -/
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
/-- **The compiled prover's schedule is the primed prover's.** -/
theorem schedZ_of_schedC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (Dead : List (OutZC 𝔽 D Sl) → Prop) (k n₀ n : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (h : SchedC A Hs T pre post cap₀ Dead k n₀ n σ) : SchedZ A Hs T pre post cap₀ Dead k n₀ n σ := fun pts =>
  preSched_push (pushTab A Hs T cap₀ pts) _ pushed_none
    (fun Dead σ' h => tabFollowZ_of_c A Hs T cap₀ post pts k n₀ Dead σ' h) _ (h pts)

omit [Inhabited Sl] in
/-- **Past the last cap the session's table has no node of the slot.** -/
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

/-- **Past the last cap `SlotRSZ` holds at every count.** -/
theorem slotRSZ_of_ge (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (ℓ t : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (hℓ : T.sch.levels.length ≤ ℓ + 1) :
    SlotRSZ A Hs T pre post cap₀ r ℓ t σ := by
  unfold SlotRSZ lockSessZ
  refine Lock.SlotAt.of_noSlot (.coin _ fun pts => ?_) _
  exact Lock.NoSlot.mono _ (Lock.NoSlot.batchLock dfltOutZC
    (Lock.NoSlot.mapL _ _ (noSlot_lockAfterZ_ge A Hs T default cap₀ pts r hr ℓ hℓ)) _ _)

end Session

/-! ## At the masked simulator -/

/-- **The simulated strategy follows the deployed schedule**: at every masked statement in scope that sets up at a
draw, every prover and tape, every table `j` and table of columns, every rep `r` and level `ℓ` below the last, the
simulated strategy after `Commit` (`zkSimAt`) follows `SchedC`: the tables in front by `Lock.PreSched` after their
coins (`zkFront`), then table `j`, rep 0 first after its `repRounds`, with `r · repRounds + zkCapDist` of its coins
before the tables after it flip any, short of runs where table `j` opens nothing. -/
def ZkBatchAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The simulated strategy follows the deployed schedule**: at every masked statement in scope that sets up at a
draw, every prover and tape, every table `j` and table of columns, every rep `r` and level `ℓ` below the last, the
simulated strategy after `Commit` (`zkSimAt`) follows `SchedC`: the tables in front by `Lock.PreSched` after their
coins (`zkFront`), then table `j`, rep 0 first after its `repRounds`, with `r · repRounds + zkCapDist` of its coins
before the tables after it flip any, short of runs where table `j` opens nothing. -/
def ZkBatchAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
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

/-- **The slot at the schedule's split count from the schedule**: `ZkFrontAlign` from `ZkBatchAlign` by
`schedZ_of_schedC` and `slotRSZ_of_batch` at `dist_lockRepZ_zkCap`'s distance, and by `slotRSZ_of_ge` at the last
level. -/
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

/-- **The slot at the schedule's split count from the schedule**: `ZkFrontAlign` from `ZkBatchAlign` by
`schedZ_of_schedC` and `slotRSZ_of_batch` at `dist_lockRepZ_zkCap`'s distance, and by `slotRSZ_of_ge` at the last
level. -/
theorem zkFrontAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkFrontAlign x :=
  Gen.zkFrontAlign_of_batch (S := S₀) (x := x) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **The slot alignment from the schedule.** -/
theorem zkSlotAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkSlotAlign x :=
  zkSlotAlign_of_front (zkFrontAlign_of_batch h)

end Gen

/-- **The slot alignment from the schedule.** -/
theorem zkSlotAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkSlotAlign x :=
  Gen.zkSlotAlign_of_batch (S := S₀) (x := x) (h := h)

end FlockDeploy.KS
