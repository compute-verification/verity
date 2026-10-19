import Proofs.Flock.Claims.KS.ZkBatch
import Proofs.Flock.Claims.KS.ZkPadsPair

/-!
# The masked pads slot from the schedule

`ZkPadsAlign` asks that the fork at rep `r`'s pads node against the simulated strategy after `Commit` be at most the
fork after the pads round's coins. Rep `r`'s pads node is its lockstep's first node (`lockRepZ`), so it comes after the
tables in front's coins and rep 0's if `r = 1`, at distance `0` of rep `r`. The schedule that puts each cap's slot at
its count (`ZkBatchAlign`) puts the pads node at its own, with a budget of `r · repRounds` of the session's table's
coins before the tables after it flip any, which is at most a cap's:

* `BudLe`, `FollowL.mono_b`, `FollowR.mono_b`, `FirstS.mono_Q`, `PreSched.mono`: a schedule with a budget of coins up
  to the slot is one with any smaller budget;
* `TabFollowC.mono`, `SchedC.mono`: so is the compiled prover's;
* `SlotPadsSZ`, `advPadsSZ_le_forkC`: where the primed prover reaches the pads node after `t` coins, `advPadsSZ` is at
  most the fork after `t` coins of `PadsForkEvZ`;
* `slotAtD_lockAfterZ_pads`, `slotPadsSZ_of_batch`: the pads node after `n + r · n₀ + 1` coins, where the primed prover
  follows `SchedZ` with `r · n₀` coins of the session's table before the tables after it flip any;
* `zkRdAt_pads`, `zkPadsAlign_of_batch`: at the deployed tables that count is the pads round, so `ZkPadsAlign` holds
  where `ZkBatchAlign` does, at level `0`'s cap (every deployed table has two levels).
-/

namespace FlockSoundness.Game.Lock

/-- One budget of coins still to go up to the slot at most another: `none`, the slot behind, at most every budget. -/
def BudLe : Option ℕ → Option ℕ → Prop
  | none, _ => True
  | some a, some b => a ≤ b
  | some _, none => False

theorem budLe_predS {a b : Option ℕ} (h : BudLe a b) : BudLe (predS a) (predS b) := by
  rcases a with _ | ⟨_ | a⟩ <;> rcases b with _ | ⟨_ | b⟩ <;> simp_all [BudLe, predS]

theorem budLe_none {a : Option ℕ} (h : BudLe a none) : a = none := by
  rcases a with _ | a
  · rfl
  · simp [BudLe] at h

/-- **A lockstep's game followed with a budget is followed with any smaller one.** -/
theorem FollowL.mono_b {β γ : Type} {Dead : β × γ → Prop} {Dead' : β → Prop} {b : Option ℕ} {g' : Game β}
    {h : Game γ} {σ : Strategy (interleave g' h)} {τ : Strategy g'} (hF : FollowL Dead Dead' b g' h σ τ) :
    ∀ {b' : Option ℕ}, BudLe b' b → FollowL Dead Dead' b' g' h σ τ := by
  induction hF with
  | dead hσ => exact fun _ => .dead hσ
  | spent hs =>
      intro b' hb
      rw [budLe_none hb]
      exact .spent hs
  | gret => exact fun _ => .gret
  | hret σ => exact fun _ => .hret σ
  | gcoinS _ ih => exact fun hb => .gcoinS fun c => ih c (budLe_predS hb)
  | gcoinC _ ih => exact fun hb => .gcoinC fun c => ih c (budLe_predS hb)
  | gsendS m _ ih => exact fun hb => .gsendS m (ih hb)
  | gsendC m _ ih => exact fun hb => .gsendC m (ih hb)
  | hsendS m₂ _ ih => exact fun hb => .hsendS m₂ (ih hb)
  | hsendC m₂ _ ih => exact fun hb => .hsendC m₂ (ih hb)

/-- **A game in front waiting with a budget waits with any smaller one.** -/
theorem FollowR.mono_b {γ β : Type} {Dead : γ × β → Prop} {Dead' : β → Prop} {b : Option ℕ} {h : Game γ}
    {g' : Game β} {σ : Strategy (interleave h g')} {τ : Strategy g'} (hF : FollowR Dead Dead' b h g' σ τ) :
    ∀ {b' : Option ℕ}, BudLe b' b → FollowR Dead Dead' b' h g' σ τ := by
  induction hF with
  | dead hσ => exact fun _ => .dead hσ
  | spent hs =>
      intro b' hb
      rw [budLe_none hb]
      exact .spent hs
  | icoinS _ ih => exact fun hb => .icoinS fun c => ih c (budLe_predS hb)
  | isendS m₂ _ ih => exact fun hb => .isendS m₂ (ih hb)

theorem FirstS.mono_Q {γ β : Type} {Dead : γ × β → Prop}
    {Q Q' : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop}
    (hQ : ∀ h g' σ, Q h g' σ → Q' h g' σ) {n : ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')}
    (hS : FirstS Dead Q n h g' σ) : FirstS Dead Q' n h g' σ := by
  induction hS with
  | dead hσ => exact .dead hσ
  | stop hq => exact .stop (hQ _ _ _ hq)
  | last => exact .last
  | coinCoin _ ih => exact .coinCoin ih
  | coinSend _ ih => exact .coinSend ih
  | sendCoin m _ ih => exact .sendCoin m ih
  | sendSend m _ ih => exact .sendSend m ih

/-- **A schedule with a slot `k` coins past where the games in front pause is one with any nearer slot**, where the
lockstep's game's schedule carries over. -/
theorem PreSched.mono {ζ : Type} (g : Game ζ) (post : List (Game ζ)) {k k' : ℕ} (hk : k' ≤ k)
    {Q Q' : (List ζ → Prop) → Strategy (batch (g :: post)) → Prop} (hQ : ∀ Dead σ, Q Dead σ → Q' Dead σ) :
    ∀ (pre : List (Game ζ)) {Dead : List ζ → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g :: post))},
      PreSched g post k pre Dead Q n σ → PreSched g post k' pre Dead Q' n σ
  | [], _, _, _, ⟨hn, hq⟩ => ⟨hn, hQ _ _ hq⟩
  | _ :: pre, _, _, _, ⟨n₀, n₁, hn, hF⟩ => ⟨n₀, n₁, hn, hF.mono_Q fun _ _ _ ⟨e, τ, hR, hP⟩ =>
      ⟨e, τ, hR.mono_b (show n₁ + k' ≤ n₁ + k by omega), PreSched.mono g post hk hQ pre hP⟩⟩

end FlockSoundness.Game.Lock

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem TabFollowC.mono (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (pts : Fin 2 → Fin mPts → 𝔽) {k k' : ℕ}
    (hk : k' ≤ k) (n₀ : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (σ' : Strategy (batch (tabAfterZC A Hs T cap₀ pts ::
      post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts)))
    (h : TabFollowC A Hs T cap₀ post pts k n₀ Dead σ') : TabFollowC A Hs T cap₀ post pts k' n₀ Dead σ' :=
  let ⟨σ₁, hF, hL⟩ := h
  ⟨σ₁, hF.mono_b (show k' ≤ k from hk), hL⟩

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The compiled prover's schedule with a budget is its schedule with any smaller one.** -/
theorem SchedC.mono {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {T : TabZK 𝔽 mPts}
    {pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))} {cap₀ : Model.Cap D}
    {Dead : List (OutZC 𝔽 D Sl) → Prop} {k n₀ n : ℕ} {σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))}
    (h : SchedC A Hs T pre post cap₀ Dead k n₀ n σ) {k' : ℕ} (hk : k' ≤ k) :
    SchedC A Hs T pre post cap₀ Dead k' n₀ n σ := fun pts =>
  Lock.PreSched.mono _ _ hk (fun Dead σ' h => TabFollowC.mono A Hs T cap₀ post pts hk n₀ Dead σ' h) _ (h pts)

/-- **The primed prover reaches rep `r`'s pads node after `t` coins** in the session after `Commit`: `Lock.SlotAt` at
`advPadsSZ`'s chooser, slot, fork event and lockstep. -/
def SlotPadsSZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) r
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^pads_r` in the session is at most the fork after `t` coins of `PadsForkEvZ`** where the primed prover
reaches rep `r`'s pads node after `t` coins. -/
theorem advPadsSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotPadsSZ A Hs T pre post cap₀ r t σ) :
    advPadsSZ A Hs T pre post cap₀ r σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        PadsForkEvZ Hs T.N T.q pre.length r p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun _ d _ _ _ _ _ _ _ _ _ _ => ind_le_indZ fun ⟨p, h⟩ => ⟨d, p, h⟩) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

/-- **The pads fork event is `0` at a run whose table `n` opens nothing** (`opPadsAtZ_nul`). -/
theorem ind_padsAt_nul (Hs : Hashes 𝔽 D Sl) {N : ℕ} (hN : 2 ≤ N) (q n r : ℕ) (d : Model.Cap D)
    {b b' : List (OutZC 𝔽 D Sl)} (h : DeadZ n b) :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opPadsAtZ Hs N q r d (outs.getD n dfltOutZC)) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨p, h₁, -, -⟩ := hc
    dsimp only at h₁
    rw [opPadsAtZ_nul Hs hN q r d h p] at h₁
    simp at h₁
  · rfl

/-- **The session's table meets rep `r`'s pads node after rep 0's `n` coins if `r = 1`, at every table choice**, where
its strategy plays rep 0 first, which ends after `n` coins, or reaches only `Dead` outcomes: the pads node is rep `r`'s
lockstep's first (`Lock.SlotAtD.lead_fst`, `Lock.SlotAtD.lead_snd`). -/
theorem slotAtD_lockAfterZ_pads (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (n : ℕ) (Dead : OutZC 𝔽 D Sl → Prop)
    (σ₁ : Strategy (tabAfterZCP A Hs T cap₀ pts))
    (hS : Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)) Dead n
      (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    Lock.SlotAtD r Dead (r * n + 0)
      (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
        (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts)) σ₁ := by
  refine Lock.SlotAtD.mapL _ _ ?_
  unfold lockAfterZ
  match r, hr with
  | 0, _ =>
    rw [Nat.zero_mul]
    exact Lock.SlotAtD.lead_fst disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _
      (by unfold lockRepZ; exact .here) (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _)
  | 1, _ =>
    rw [Nat.one_mul]
    exact Lock.SlotAtD.lead_snd disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _
      (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _) (by unfold lockRepZ; exact .here)

/-- **The pads slot from a schedule whose tables pause**: `SlotPadsSZ` after `n + r · n₀ + 1` coins, where the primed
prover follows `SchedZ` with `r · n₀` coins of the session's table before the tables after it flip any, short of runs
`Dead` at which `advPadsSZ`'s event is `0` (`Lock.SlotAtD.batchLock_sched`, `Lock.SlotAtD.batchLock_nil_followL`,
`slotAtD_lockAfterZ_pads`). -/
theorem slotPadsSZ_of_batch (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (n n₀ : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (hΦ : ∀ d b b', Dead b → Dead b' → ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
        opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b') = 0)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (hF : SchedZ A Hs T pre post cap₀ Dead (r * n₀ + 0) n₀ n σ) :
    SlotPadsSZ A Hs T pre post cap₀ r (n + (r * n₀ + 0) + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <| Lock.SlotAtD.slotAt hΦ <|
    Lock.SlotAtD.batchLock_sched dfltOutZC _ (post.map (· pts))
      (fun _ _ ⟨σ₁, hFL, hl⟩ => Lock.SlotAtD.batchLock_nil_followL dfltOutZC _ _ hFL
        (slotAtD_lockAfterZ_pads A Hs T default cap₀ pts r hr n₀ _ σ₁ hl))
      (pre.map (· pts)) (hF pts)

end Session

/-! ## At the masked simulator -/

/-- **The pads round, split at the schedule**: the link coin, the tables in front's coins, then rep 0's if `r = 1`. -/
theorem zkRdAt_pads (spec : SessionSpec) (sts : Array Stmt) (j : ℕ) {Lv : ℕ} (r : Fin 2) :
    zkFront spec sts j + (r.val * repRounds true spec sts[zkTbl sts j]! + 0) + 1 =
      zkRdAt spec sts j (Lv := Lv) (.inl (.inr (.inl r))) := by
  show _ = zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r
  rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with ⟨a, e⟩ | ⟨a, e⟩ <;>
    rw [e, a] <;> unfold zkBase zkFront <;> omega

/-- **The pads alignment from the schedule**: `ZkPadsAlign` from `ZkBatchAlign` at level `0`'s cap (every deployed
table has two levels, `TabZK.hshape`), its budget cut to `r · repRounds` (`SchedC.mono`), by `schedZ_of_schedC`,
`slotPadsSZ_of_batch` and `advPadsSZ_le_forkC`, table `j`'s dead runs opening no pads position (`ind_padsAt_nul`). -/
theorem zkPadsAlign_of_batch {x : Claims.Msg} (h : ZkBatchAlign x) : ZkPadsAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r u
  have h2 := (zkTabs h0 L hz hk hlay j).hshape.1
  have hb := h s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ⟨0, by omega⟩ u
    (show 0 + 1 < _ by omega)
  have ha := advPadsSZ_le_forkC _ _ _ _ _ _ _ _ _ <|
    slotPadsSZ_of_batch _ _ _ _ _ _ r.val r.isLt _ _ (DeadZ j)
      (fun e b b' hb _ => by
        rw [take_gamesZC_length]
        exact ind_padsAt_nul _ (two_le_zkTabsN h0 L hz hk hlay j) _ _ _ _ hb) _
      (schedZ_of_schedC _ _ _ _ _ _ _ _ _ _ _ (hb.mono (Nat.add_le_add_left (Nat.zero_le _) _)))
  rw [take_gamesZC_length] at ha
  have hv : (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inl r)))).val =
      zkRdAt spec sts j (Lv := (zkTabs h0 L hz hk hlay j).sch.levels.length) (.inl (.inr (.inl r))) := rfl
  unfold ZkPadsAlignQ
  rw [hv, ← zkRdAt_pads spec sts j (Lv := (zkTabs h0 L hz hk hlay j).sch.levels.length) r]
  exact ha

end FlockDeploy.KS
