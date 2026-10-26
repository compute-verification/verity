import Proofs.Flock.Claims.KS.ZkBatchSim
import Proofs.Flock.Claims.KS.ZkPadsPair

/-!
# The masked simulated fork's alignment at the pads

`ZkPadsAlign` compares the fork at rep `r`'s pads node against the simulated strategy after `Commit` (`advPadsSZ`) with
the fork after `zkRd j i` coins of `PadsForkEvZ`. The pads root is a rep's first table node (`lockRepZ`), so it is
`0` coins into the rep, and its fork round `zkB2` is the tables in front's coins and rep 0's if `r = 1`: the caps'
schedule (`ZkBatch`, `ZkBatchSim`) at distance `0`.

* `dist_lockRepZ_pads`, `slotAtD_lockAfterZ_pads`: the session's table meets rep `r`'s pads after rep 0's `n` coins
  if `r = 1`, where its strategy plays rep 0 first;
* `SlotPadsZ`, `advPadsSZ_le_forkC`: where the primed prover reaches the pads node after `t` coins, `advPadsSZ` is at
  most the fork after `t` coins of `PadsForkEvZ`, whose cap is any;
* `ind_conflictPads_nul`, `slotPadsZ_of_batch`: a run whose session table opens nothing opens no pads position, so
  `SchedZ` at `r · n₀` gives the slot after `n + r · n₀ + 1` coins;
* `zkPadsAlign`: `ZkPadsAlign` holds at every message, the simulated strategy following `SchedC` there
  (`schedC_simAtJZ`), so the pads fork bound holds outright (`zkSimForkLePads`).
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

omit [Inhabited D] [Inhabited Sl] in
/-- **A rep's pads root is its first table node.** -/
theorem dist_lockRepZ_pads (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Model.Oracle 𝔽) (extra : List (Model.Claim 𝔽 T.S.m)) : Lock.Dist r 0 (lockRepZ A Hs T r hr C₀ extra) := by
  unfold lockRepZ
  exact .here

/-- **The session's table meets rep `r`'s pads past rep 0's `n` coins if `r = 1`, at every table choice**, where its
strategy plays rep 0 first, which ends after `n` coins, or reaches only `Dead` outcomes (`Lock.SlotAtD.lead_fst`,
`Lock.SlotAtD.lead_snd`). -/
theorem slotAtD_lockAfterZ_pads (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (n : ℕ) (Dead : OutZC 𝔽 D Sl → Prop)
    (σ₁ : Strategy (tabAfterZCP A Hs T cap₀ pts))
    (hS : Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)) Dead n
      (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    Lock.SlotAtD r Dead (r * n)
      (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
        (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts)) σ₁ := by
  refine Lock.SlotAtD.mapL _ _ ?_
  unfold lockAfterZ
  match r, hr with
  | 0, _ =>
    rw [Nat.zero_mul]
    exact Lock.SlotAtD.lead_fst disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _
      (dist_lockRepZ_pads A Hs T 0 _ C₀ _) (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _)
  | 1, _ =>
    rw [show 1 * n = n + 0 by omega]
    exact Lock.SlotAtD.lead_snd disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _
      (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _) (dist_lockRepZ_pads A Hs T 1 _ C₀ _)

/-- **The primed prover reaches rep `r`'s pads node after `t` coins** in the session after `Commit`: `Lock.SlotAt` at
`advPadsSZ`'s chooser, slot, fork event and lockstep. -/
def SlotPadsZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) r
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^pads_r` in the session is at most the fork after `t` coins of `PadsForkEvZ`** where the primed prover
reaches rep `r`'s pads node after `t` coins: the node's conflict at its cap is `PadsForkEvZ`'s at that cap. -/
theorem advPadsSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotPadsZ A Hs T pre post cap₀ r t σ) :
    advPadsSZ A Hs T pre post cap₀ r σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        PadsForkEvZ Hs T.N T.q pre.length r p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun _ d _ _ _ _ _ _ _ _ _ _ => ind_le_indZ fun ⟨p, hc⟩ => ⟨d, p, hc⟩) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

/-- **The pads conflict is `0` at a run whose table `n` opens nothing**: its pads paths are empty
(`opPadsAtZ_nul`). -/
theorem ind_conflictPads_nul (Hs : Hashes 𝔽 D Sl) {N : ℕ} (hN : 2 ≤ N) (q n r : ℕ) (d : Model.Cap D)
    {b b' : List (OutZC 𝔽 D Sl)} (h : DeadZ n b) :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opPadsAtZ Hs N q r d (outs.getD n dfltOutZC)) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨p, h₁, -, -⟩ := hc
    dsimp only at h₁
    rw [opPadsAtZ_nul Hs hN q r d h p] at h₁
    exact absurd h₁ (by simp)
  · rfl

/-- **The pads slot from a schedule whose tables pause**: `SlotPadsZ` after `n + r · n₀ + 1` coins, where the primed
prover follows `SchedZ` with `r · n₀` coins of the session's table before the tables after it flip any, short of runs
`Dead` at which the pads conflict is `0` (`Lock.SlotAtD.batchLock_sched`, `Lock.SlotAtD.batchLock_nil_followL`,
`slotAtD_lockAfterZ_pads`). -/
theorem slotPadsZ_of_batch (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (n n₀ : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (hΦ : ∀ d b b', Dead b → Dead b' → ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
        opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b') = 0)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (hF : SchedZ A Hs T pre post cap₀ Dead (r * n₀) n₀ n σ) :
    SlotPadsZ A Hs T pre post cap₀ r (n + r * n₀ + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <| Lock.SlotAtD.slotAt hΦ <|
    Lock.SlotAtD.batchLock_sched dfltOutZC _ (post.map (· pts))
      (fun _ _ ⟨σ₁, hFL, hl⟩ => Lock.SlotAtD.batchLock_nil_followL dfltOutZC _ _ hFL
        (slotAtD_lockAfterZ_pads A Hs T default cap₀ pts r hr n₀ _ σ₁ hl))
      (pre.map (· pts)) (hF pts)

end Session

/-! ## At the masked simulator -/

namespace Gen

variable [S : ZkSetup]

/-- **`ZkPadsAlign` holds at every message**: the pads fork round is the tables in front's coins, rep 0's if `r = 1`,
and the link coin (`zkB2`); the simulated strategy follows `SchedC` at `r · repRounds` (`schedC_simAtJZ` on
`zkStreams`, as `zkBatchAlign`), so it reaches the pads node there (`slotPadsZ_of_batch`), and `advPadsSZ_le_forkC`
gives the alignment, table `j`'s position in the session being `j` (`take_gamesZC_length`). -/
theorem zkPadsAlign (x : Claims.Msg) : ZkPadsAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r u
  have hjS : j < sts.size := by rw [← zk_tables_length hset]; exact hj
  have ht : zkTbl sts j = j := by unfold zkTbl; exact ite_eq_left hjS
  have hv : (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inl r)))).val =
      zkFront spec sts j + r.val * repRounds true spec sts[zkTbl sts j]! + 1 := by
    show zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r = _
    rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with
      ⟨a, e⟩ | ⟨a, e⟩ <;> rw [e, a] <;> unfold zkBase zkFront <;> omega
  obtain ⟨-, hR, -, -⟩ := zkRd_facts h0 L hz hk hlay j
  have hkR : r.val * repRounds true spec sts[zkTbl sts j]! <
      repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! := by
    have := Nat.mul_le_mul_right (repRounds true spec sts[zkTbl sts j]!) (show r.val ≤ 1 by omega)
    omega
  have hsched : SchedC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!)
      (DeadZ j) (r.val * repRounds true spec sts[zkTbl sts j]!) (repRounds true spec sts[zkTbl sts j]!)
      (zkFront spec sts j) (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u) := by
    rw [ht] at hkR ⊢
    rw [zkFront_eq hjS]
    have hJ : sts.size ≤ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
      rw [tabsJ_length, zk_tables_length hset]
    refine schedC_simAtJZ execArith ZkExec.hs512 _ _ _ _ spec.dg _ _ _
      (capOf := fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x) _ (fun _ _ _ => rfl) (zkRoot P x ω) _ _ _ _
      (fun t => repRounds true spec sts[t]!) _ _ hkR
      (fromS (fun t => repRounds true spec sts[t]!) (j + 1) (sts.size - (j + 1))) (fun p hp => ?_) fun pts => ?_
    · have := mem_fromS _ _ _ p hp
      dsimp only [finJ]
      omega
    · have h := zkLive_sched hJ h0 (zk_mPts_le hs hin hset) P ω u pts (zkStreams_length hz hset)
      rw [zkStreams_split hz hjS] at h
      exact h
  have ha := advPadsSZ_le_forkC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ _ r.val _ _
    (slotPadsZ_of_batch _ _ _ _ _ _ r.val r.isLt _ _ (DeadZ j)
      (fun e b b' hb _ => by
        rw [take_gamesZC_length]
        exact ind_conflictPads_nul _ (two_le_zkTabsN h0 L hz hk hlay j) _ _ _ _ hb)
      _ (schedZ_of_schedC _ _ _ _ _ _ _ _ _ _ _ hsched))
  rw [take_gamesZC_length] at ha
  unfold ZkPadsAlignQ
  rw [hv]
  exact ha

end Gen

/-- **`ZkPadsAlign` holds at every message**: the pads fork round is the tables in front's coins, rep 0's if `r = 1`,
and the link coin (`zkB2`); the simulated strategy follows `SchedC` at `r · repRounds` (`schedC_simAtJZ` on
`zkStreams`, as `zkBatchAlign`), so it reaches the pads node there (`slotPadsZ_of_batch`), and `advPadsSZ_le_forkC`
gives the alignment, table `j`'s position in the session being `j` (`take_gamesZC_length`). -/
theorem zkPadsAlign (x : Claims.Msg) : ZkPadsAlign x :=
  Gen.zkPadsAlign (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **The masked simulator's fork bound at the pads holds at every message.** -/
theorem zkSimForkLePads (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  zkSimForkLePads_of (zkPadsAlign x)

end Gen

/-- **The masked simulator's fork bound at the pads holds at every message.** -/
theorem zkSimForkLePads (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  Gen.zkSimForkLePads (S := S₀) (x := x)

end FlockDeploy.KS
