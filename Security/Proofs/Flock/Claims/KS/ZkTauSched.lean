import Proofs.Flock.Claims.KS.SlotRej
import Proofs.Flock.Claims.KS.RepDistZ
import Proofs.Flock.Claims.KS.ZkPadsSched
import Proofs.Flock.Claims.KS.ZkTauPair
import Proofs.Flock.Claims.KS.ZkBatchSim

/-!
# The masked `τ` slot from the schedule, and the `τ` and pads alignments

`ZkTauAlign` asks that the fork at rep `r`'s `τ` node against the simulated strategy after `Commit` be at most the fork
after the `τ` round's coins. Rep `r`'s `τ` node is its inner proof's table, after the masked phase and two of the inner
proof's coins (`lockInnerZ`). A run whose masked phase ends well takes exactly its shape (`repMaskedZC_on`); one that
ends badly rejects, and its table's output is not accepted, where `opTauZ` opens nothing. So the node is at a fixed count
on the runs that can open `τ` (`Lock.DistR`, `Lock.SlotAtDR`), and the schedule that puts every node of the session's
table at its count (`SchedC`, which ZKA1's `schedC_simAtJZ` gives at every budget short of both reps' rounds) puts the
`τ` node at its own:

* `noSlot_reindex_off`, `coinsK_recShape_four`, `coinsK_ligPadShape`, `zkTauDist`, `distR_lockRepZ_tau`: a masked rep's
  `τ` node is `zkTauDist` coins in, or its outcomes reject;
* `slotAtDR_lockAfterZ_tau`, `ind_tau_nul`, `ind_tau_rej`, `SlotTauSZ`, `advTauSZ_le_forkC`, `slotTauSZ_of_batch`:
  the `τ` slot after `n + (r · n₀ + zkTauDist) + 1` coins, where the primed prover follows `SchedZ` with that budget,
  the fork event `0` on two dead runs and on two rejected ones;
* `zkRdAt_tau`, `ZkSchedAlign`, `zkSchedAlign`: at the deployed tables that count is the `τ` round, and the simulated
  strategy follows `SchedC` at every budget below `2 · repRounds`;
* `zkTauAlign_of_sched`, `zkTauAlign`, `zkPadsAlign`: so `ZkTauAlign` and `ZkPadsAlign` hold at every message, and with
  them the `τ` and pads fork bounds (`zkSimForkLeTau`, `zkSimForkLePads`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession Exec ZkExec

section Reindex

variable {α β T Dd I J : Type} {getI : I → ℕ → Option (T × Dd)} {getJ : J → ℕ → Option (T × Dd)}
  {R : I → α → β → Prop} (emb : I → J) (sl : ℕ → ℕ) (hget : ∀ i s, getJ (emb i) (sl s) = getI i s)
  (hoff : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (emb i) s = getJ (emb i') s)

/-- **Moving every slot off `slot` leaves no table node of it.** -/
theorem noSlot_reindex_off {slot : ℕ} (hs : ∀ s, sl s ≠ slot) {i : I} {g : Game α} {g' : Game β}
    (L : Lock T Dd getI R i g g') : Lock.NoSlot slot (lockReindex emb sl hget hoff L) := by
  induction L with
  | ret hr => exact .ret _
  | coin C L ih => exact .coin C ih
  | send M L ih => exact .send M ih
  | table s upd hf hs' hk L ih => exact .table (sl s) (hs s) ih

end Reindex

theorem coinsK_recShape_four (F K D : Type) (ext : ℕ) :
    ∀ (lv : Level) (rest : List Level), (∀ l ∈ lv :: rest, l.lanes = 4) →
      Lock.coinsK (recShape F K D ext (lv :: rest)) = 7 + 11 * rest.length
  | lv, [], h => by
      have h4 := h lv (by simp)
      simp only [recShape, Lock.coinsK_append, Lock.coinsK, coinsK_qbShape, coinsK_foldShape, h4,
        List.length_nil]
  | lv, lv' :: rest, h => by
      have h4 := h lv (by simp)
      have ih := coinsK_recShape_four F K D ext lv' rest fun l hl => h l (List.mem_cons_of_mem _ hl)
      simp only [recShape, Lock.coinsK_append, Lock.coinsK, coinsK_qbShape, coinsK_oodShape, coinsK_foldShape, h4, ih,
        List.length_cons]
      omega

/-- **The masked level 0's coins**, with four lanes after level 0: its two first coins, its `k0` lane rounds, its `e`
extra draws, then the level-1 queries' and the recursion's. -/
theorem coinsK_ligPadShape (F K D : Type) (sch : Schedule) (e t : ℕ) (h2 : 2 ≤ sch.levels.length)
    (hl : ∀ lv ∈ sch.levels.tail, lv.lanes = 4) :
    Lock.coinsK (ligPadShape F K D sch e t) = sch.k0 + e + 5 + 11 * (sch.levels.length - 1) := by
  obtain ⟨m, k0, levels, ext⟩ := sch
  rcases levels with _ | ⟨lv₀, _ | ⟨lv₁, rest⟩⟩
  · simp at h2
  · simp at h2
  · simp only [List.tail_cons] at hl
    simp only [ligPadShape, tailShape, Lock.coinsK_append, Lock.coinsK, coinsK_foldShapeZ, coinsK_replicate_coin,
      coinsK_oodShape, coinsK_qbShape, coinsK_recShape_four F K D ext lv₁ rest hl, List.length_cons]
    omega

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

/-- **The inner proof's `τ` node is two coins in**: after `ε`, `ρσ` and `γ`. -/
theorem dist_lockInnerZ_tau (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ)
    (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev : IRep F D) (O : Oracle F)
    (c : Cap D) : Lock.Dist (2 + r) 2 (lockInnerZ Hs r hr dom s nP q pos rows tr lev O c) :=
  .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .here

/-- A masked rep's coins to its `τ` node, on `nExtra` extra claims: the zerocheck's, the lincheck's, the opening's, the
masked level 0's with four lanes after it, then the inner proof's `ε` and `γ`. -/
def zkTauDist (T : TabZK F mPts) (nExtra : ℕ) : ℕ :=
  (T.S.m - 6 + 3) + (T.S.kLog - 6 + 3) + (nExtra + 2 + 1) + (T.sch.k0 + T.e + 5 + 11 * (T.sch.levels.length - 1)) + 2

/-- **A masked rep's `τ` node is `zkTauDist` coins in, or its outcomes reject**: a run whose masked phase ends well
takes its shape (`repMaskedZC_on`) and meets `τ` two coins into the inner proof; one that ends badly rejects. The pads
root and the level caps hold no `τ` slot. -/
theorem distR_lockRepZ_tau (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Oracle F) (extra : List (Claim F T.S.m)) (h2 : 2 ≤ T.sch.levels.length)
    (hl : ∀ lv ∈ T.sch.levels.tail, lv.lanes = 4) :
    Lock.DistR (2 + r) (fun p : PendZ F D Sl => p.lig = PendL.reject) (zkTauDist T extra.length)
      (lockRepZ A Hs T r hr C₀ extra) := by
  have hdist : zkTauDist T extra.length = 2 + Lock.coinsK (zcShape F T.S.m ++ (lcShape F T.S.kLog ++
      (opShape F (extra.length + 2) ++ (ligPadShape F K D T.sch T.e T.t ++ [.msg F])))) := by
    simp only [Lock.coinsK_append, coinsK_zcShape, coinsK_lcShape, coinsK_opShape,
      coinsK_ligPadShape F K D T.sch T.e T.t h2 hl, Lock.coinsK]
    unfold zkTauDist
    omega
  rw [hdist]
  refine .other r (by omega) fun O cp => ?_
  refine Lock.DistR.bind_on (P := fun x => x.2 ≠ PendL.reject) ?_ ?_ ?_ _
    (repMaskedZC_on A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra)
  · intro j o o' h _
    rcases j with ⟨lev, pads, tau⟩
    obtain ⟨hp, ht, h⟩ := h
    simp only at hp ht h
    subst hp ht
    obtain ⟨hrows, hrec⟩ := h
    rcases o with ⟨rows, v⟩
    rcases o' with ⟨rows', p⟩
    simp only at hrows hrec
    subst hrows
    refine Lock.DistR.of_dist (Lock.Dist.bind_noSlot ?_ (dist_lockInnerZ_tau Hs r hr _ _ _ _ _ _ _ _ _ _))
    intro _ _ _ _
    exact .ret _
  · intro b hb c hc
    obtain ⟨v, -, hv⟩ := Out.of_bind hc
    have hv' : Out (Game.ret (⟨b.2, v.1, v.2⟩ : PendZ F D Sl)) c := hv
    cases hv'
    exact not_not.mp hb
  · exact Lock.NoSlot.mono _ (noSlot_reindex_off _ (· + 4) _ _ (fun s => by show s + 4 ≠ 2 + r; omega) _)

end FlockSoundness.Discharge.ZkLink

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

/-- **The session's table meets rep `r`'s `τ` node after rep 0's `n` coins if `r = 1` and `zkTauDist`, at every table
choice**, where its strategy plays rep 0 first, which ends after `n` coins, or reaches only `Dead` outcomes, or ends
only in outputs not accepted: the `τ` node is rep `r`'s, at its distance (`distR_lockRepZ_tau`), and a rejecting rep
makes its table's output not accepted (`Lock.SlotAtDR.lead_fst`, `Lock.SlotAtDR.lead_snd`). -/
theorem slotAtDR_lockAfterZ_tau (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (C₀ : Model.Oracle 𝔽) (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (n : ℕ)
    (Dead : OutZC 𝔽 D Sl → Prop) (h2 : 2 ≤ T.sch.levels.length) (hl : ∀ lv ∈ T.sch.levels.tail, lv.lanes = 4)
    (σ₁ : Strategy (tabAfterZCP A Hs T cap₀ pts))
    (hS : Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)) Dead n
      (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    Lock.SlotAtDR (2 + r) Dead (fun o : OutZC 𝔽 D Sl => o.accepted = false)
      (r * n + zkTauDist T (2 * T.S.regions.length))
      (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
        (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts)) σ₁ := by
  refine Lock.SlotAtDR.mapL _ _ ?_
  unfold lockAfterZ
  have hRj : ∀ (x : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl) c,
      (x.1.lig = Model.PendL.reject ∨ x.2.lig = Model.PendL.reject) →
      Out (Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ x os)) c → c.accepted = false := by
    intro x c hx hc
    cases hc with
    | send os hc =>
      cases hc
      rcases hx with hx | hx <;> simp [outZC, hx, Model.PendL.reject]
  match r, hr with
  | 0, _ =>
    have h := distR_lockRepZ_tau A Hs T 0 (by norm_num) C₀ (Model.extraClaims T.S T.ptLocal mPts pts) h2 hl
    rw [FlockSoundness.extraClaims_length] at h
    rw [Nat.zero_mul, Nat.zero_add]
    exact Lock.SlotAtDR.lead_fst disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _)
      (fun x c hx hc => hRj x c (.inl hx) hc) hS _ _ h (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _)
  | 1, _ =>
    have h := distR_lockRepZ_tau A Hs T 1 (by norm_num) C₀ (Model.extraClaims T.S T.ptLocal mPts pts) h2 hl
    rw [FlockSoundness.extraClaims_length] at h
    rw [Nat.one_mul]
    exact Lock.SlotAtDR.lead_snd disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _)
      (fun x c hx hc => hRj x c (.inr hx) hc) hS _ _ (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _) h

/-- **The `τ` fork event is `0` at two runs whose table `n` opens nothing**: both open the same `τ`. -/
theorem ind_tau_nul (Hs : Hashes 𝔽 D Sl) (n r : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)} (h : DeadZ n b)
    (h' : DeadZ n b') :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opTauZ Hs r d (outs.getD n dfltOutZC)) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨⟨⟩, h₁, h₂, hne⟩ := hc
    dsimp only at h₁ h₂ hne
    unfold DeadZ at h h'
    rw [(opTauZ_some h₁).2.2, (opTauZ_some h₂).2.2, h, h'] at hne
    exact absurd rfl hne
  · rfl

/-- **The `τ` fork event is `0` at a run whose table `n`'s output is not accepted** (`opTauZ_some`). -/
theorem ind_tau_rej (Hs : Hashes 𝔽 D Sl) (n r : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)}
    (h : (b.getD n dfltOutZC).accepted = false) :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opTauZ Hs r d (outs.getD n dfltOutZC)) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨⟨⟩, h₁, -, -⟩ := hc
    dsimp only at h₁
    have ha := (opTauZ_some h₁).1
    rw [h] at ha
    exact absurd ha Bool.false_ne_true
  · rfl

/-- **The primed prover reaches rep `r`'s `τ` node after `t` coins** in the session after `Commit`: `Lock.SlotAt` at
`advTauSZ`'s chooser, slot, fork event and lockstep. -/
def SlotTauSZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) (2 + r)
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^τ_r` in the session is at most the fork after `t` coins of `TauForkEvZ`** where the primed prover reaches
rep `r`'s `τ` node after `t` coins. -/
theorem advTauSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotTauSZ A Hs T pre post cap₀ r t σ) :
    advTauSZ A Hs T pre post cap₀ r σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        TauForkEvZ Hs pre.length r p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun _ d _ _ _ _ _ _ _ _ _ _ => ind_le_indZ fun ⟨⟨⟩, h⟩ => ⟨d, h⟩) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

/-- **The `τ` slot from a schedule whose tables pause**: `SlotTauSZ` after `n + (r · n₀ + zkTauDist) + 1` coins,
where the primed prover follows `SchedZ` with `r · n₀ + zkTauDist` coins of the session's table before the tables after
it flip any, short of runs `Dead` at which `advTauSZ`'s event is `0` (`Lock.SlotAtDR.batchLock_sched`,
`Lock.SlotAtDR.batchLock_nil_followL`, `slotAtDR_lockAfterZ_tau`); the event is `0` on two rejected runs
(`ind_tau_rej`). -/
theorem slotTauSZ_of_batch (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (n n₀ : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop) (h2 : 2 ≤ T.sch.levels.length)
    (hl : ∀ lv ∈ T.sch.levels.tail, lv.lanes = 4)
    (hΦ : ∀ d b b', Dead b → Dead b' → ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
        opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b') = 0)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (hF : SchedZ A Hs T pre post cap₀ Dead (r * n₀ + zkTauDist T (2 * T.S.regions.length)) n₀ n σ) :
    SlotTauSZ A Hs T pre post cap₀ r (n + (r * n₀ + zkTauDist T (2 * T.S.regions.length)) + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <|
    Lock.SlotAtDR.slotAt hΦ
      (fun d b b' hb _ => ind_tau_rej Hs pre.length r d (b' := b') (by simpa only [List.length_map] using hb)) <|
    Lock.SlotAtDR.batchLock_sched (P := fun o : OutZC 𝔽 D Sl => o.accepted = false) dfltOutZC _ (post.map (· pts))
      (fun _ _ ⟨σ₁, hFL, hl'⟩ => Lock.SlotAtDR.batchLock_nil_followL dfltOutZC _ _ hFL
        (slotAtDR_lockAfterZ_tau A Hs T default cap₀ pts r hr n₀ _ h2 hl σ₁ hl'))
      (pre.map (· pts)) (hF pts)

end Session

/-! ## At the masked simulator -/

/-- **The `τ` round, split at the schedule**: the link coin, the tables in front's coins, rep 0's if `r = 1`, then
`zkTauDist` at two link points per region. -/
theorem zkRdAt_tau {spec : SessionSpec} {sts : Array Stmt} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (hp : spec.points = 2) (j : ℕ) (r : Fin 2) :
    zkFront spec sts j + (r.val * repRounds true spec sts[zkTbl sts j]! +
        zkTauDist (zkTabs h0 L hz hk hlay j) (2 * (zkTabs h0 L hz hk hlay j).S.regions.length)) + 1 =
      zkRdAt spec sts j (Lv := (zkTabs h0 L hz hk hlay j).sch.levels.length) (.inl (.inr (.inr r))) := by
  obtain ⟨he, hm, hkl, hreg, hk0, -⟩ := zkTabs_facts h0 L hz hk hlay j
  have hlen := zkTabs_len h0 L hz hk hlay j
  have h13 := (zkTabs h0 L hz hk hlay j).hm
  show _ = zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r +
    zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2
  rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with ⟨a, e⟩ | ⟨a, e⟩ <;>
    rw [e, a] <;> unfold zkTauDist zkBase zkFront zkPre zkLig <;> rw [hp] <;> omega

/-- **The simulated strategy follows the deployed schedule at every budget short of both reps' rounds**: at every
masked statement in scope that sets up at a draw, every prover and tape, every table `j` and table of columns, and
every `k < 2 · repRounds`, the simulated strategy after `Commit` (`zkSimAt`) follows `SchedC` with `k` of table `j`'s
coins before the tables after it flip any. -/
def ZkSchedAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (u : zkTape x) (k : ℕ),
    k < repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! →
      SchedC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) (DeadZ j) k
        (repRounds true spec sts[zkTbl sts j]!) (zkFront spec sts j)
        (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

/-- **`ZkSchedAlign` holds at every message**: the masked simulator's live strategy submits its rounds on
`zkStreams` (`zkLive_sched`), split at table `j` (`zkStreams_split`), so the simulated strategy follows `SchedC` at
every budget short of both reps' rounds (`schedC_simAtJZ`). -/
theorem zkSchedAlign (x : Claims.Msg) : ZkSchedAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C u k hkR
  have hjS : j < sts.size := by rw [← zk_tables_length hset]; exact hj
  have ht : zkTbl sts j = j := by unfold zkTbl; exact ite_eq_left hjS
  rw [ht] at hkR ⊢
  rw [zkFront_eq hjS]
  have hJ : sts.size ≤ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
    rw [tabsJ_length, zk_tables_length hset]
  refine schedC_simAtJZ execArith ZkExec.hs512 _ _ _ _ spec.dg _ _ _ (capOf := fun rs j => Exec.capOfRoot rs[j.val]!)
    (zkN x) _ (fun _ _ _ => rfl) (zkRoot P x ω) _ _ _ _ (fun t => repRounds true spec sts[t]!) _ _ hkR
    (fromS (fun t => repRounds true spec sts[t]!) (j + 1) (sts.size - (j + 1))) (fun p hp => ?_) fun pts => ?_
  · have := mem_fromS _ _ _ p hp
    dsimp only [finJ]
    omega
  · have h := zkLive_sched hJ h0 (zk_mPts_le hs hin hset) P ω u pts (zkStreams_length hz hset)
    rw [zkStreams_split hz hjS] at h
    exact h

/-- **The `τ` alignment from the schedule**: `ZkTauAlign` from `ZkSchedAlign` at the `τ` node's budget
`r · repRounds + zkTauDist`, short of both reps' rounds since `zkTauDist` is two short of a rep's, by
`schedZ_of_schedC`, `slotTauSZ_of_batch` and `advTauSZ_le_forkC`, table `j`'s dead runs opening one `τ`
(`ind_tau_nul`), at the `τ` round (`zkRdAt_tau`). -/
theorem zkTauAlign_of_sched {x : Claims.Msg} (h : ZkSchedAlign x) : ZkTauAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r u
  obtain ⟨he, hm, hkl, hreg, hk0, hl⟩ := zkTabs_facts h0 L hz hk hlay j
  have h2 := (zkTabs h0 L hz hk hlay j).hshape.1
  have hp := zk_points hs hin hset
  obtain ⟨-, hRe, -, -⟩ := zkRd_facts h0 L hz hk hlay j
  have hlen := zkTabs_len h0 L hz hk hlay j
  have h13 := (zkTabs h0 L hz hk hlay j).hm
  have hk' : r.val * repRounds true spec sts[zkTbl sts j]! +
      zkTauDist (zkTabs h0 L hz hk hlay j) (2 * (zkTabs h0 L hz hk hlay j).S.regions.length) <
      repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! := by
    unfold zkPre zkLig at hRe
    rw [hp] at hRe
    unfold zkTauDist
    have := r.isLt
    rcases (show r.val = 0 ∨ r.val = 1 by omega) with e | e <;> rw [e] <;> omega
  have hb := h s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C u _ hk'
  have ha := advTauSZ_le_forkC _ _ _ _ _ _ _ _ _ <|
    slotTauSZ_of_batch _ _ _ _ _ _ r.val r.isLt _ _ (DeadZ j) h2 hl
      (fun e b b' hb hb' => by
        rw [take_gamesZC_length]
        exact ind_tau_nul _ _ _ _ hb hb') _
      (schedZ_of_schedC _ _ _ _ _ _ _ _ _ _ _ hb)
  rw [take_gamesZC_length] at ha
  have hv : (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inr r)))).val =
      zkRdAt spec sts j (Lv := (zkTabs h0 L hz hk hlay j).sch.levels.length) (.inl (.inr (.inr r))) := rfl
  unfold ZkTauAlignQ
  rw [hv, ← zkRdAt_tau h0 L hz hk hlay hp j r]
  exact ha

/-- **`ZkTauAlign` holds at every message.** -/
theorem zkTauAlign (x : Claims.Msg) : ZkTauAlign x :=
  zkTauAlign_of_sched (zkSchedAlign x)

/-- **`ZkPadsAlign` holds at every message** (`zkPadsAlign_of_batch`, `zkBatchAlign`). -/
theorem zkPadsAlign (x : Claims.Msg) : ZkPadsAlign x :=
  zkPadsAlign_of_batch (zkBatchAlign x)

/-- **The fork bound at `τ`, with no hypothesis.** -/
theorem zkSimForkLeTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => ∃ r, i = .inl (.inr (.inr r)) :=
  zkSimForkLeTau_of_align (zkTauAlign x)

/-- **The pads fork bound, with no hypothesis.** -/
theorem zkSimForkLePads (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  zkSimForkLePads_of (zkPadsAlign x)

end FlockDeploy.KS
