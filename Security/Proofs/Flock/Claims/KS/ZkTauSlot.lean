import Proofs.Flock.Claims.KS.SlotDistOn
import Proofs.Flock.Claims.KS.ZkKSAlign

/-!
# The masked simulated fork's alignment at `τ`

`ZkTauAlign` compares the fork at rep `r`'s `τ` node (`advTauSZ`, slot `2 + r`) against the simulated strategy after
`Commit` with the fork after `zkRd j i` coins of `TauForkEvZ`. `τ`'s commitment is the inner proof's second message,
after the masked phase and the inner proof's two first coins. The masked phase's shape is fixed only on the runs whose
opening can end well (`repMaskedZC_on`): a run whose opening rejects returns early, and then meets `τ` sooner. Its
Ligerito part is `PendL.reject`, so the table's output there is not accepted, and an output that is not accepted opens
no `τ` (`opTauZ`). So the distance holds on the paths that can end well (`Lock.DistOn`), the rest `Dead`:

* `SlotAtD.batchLock_sched_inv`: `SlotAtD.batchLock_sched` with an invariant on the dead runs, carried past the tables
  in front, here that a run whose table `j` is not accepted is dead;
* `noSlot_reindex_off`, `dist_lockInnerZ_tau`, `distOn_lockRepZ_tau`: a rep's `τ` is two coins into the inner proof,
  and its masked phase's coins in on the runs that can end well, the others' Ligerito part `PendL.reject`;
* `outZC_accepted_reject₀`, `outZC_accepted_reject₁`, `slotAtD_lockAfterZ_tau`: an output with a rejected rep is not
  accepted, so the session's table meets rep `r`'s `τ` past rep 0's `n` coins if `r = 1`, or reaches only outcomes
  that are not accepted;
* `DeadTZ`, `deadTZ_depAt`, `ind_conflictTau_deadT`: a run whose table opens nothing or is not accepted, read at the
  table alone, at which the `τ` conflict is `0`;
* `SlotTauZ`, `advTauSZ_le_forkC`, `slotTauZ_of_batch`: where the primed prover reaches the `τ` node after `t` coins,
  `advTauSZ` is at most the fork after `t` coins of `TauForkEvZ`, and `SchedZ` at `r · n₀ + c` gives it;
* `coinsK_recShape`, `coinsK_tailShape`, `coinsK_ligPadShape`, `coinsK_maskedZ`, `zkTau_dist`: the masked phase's
  coins at the deployed tables are `zkPre + zkLig`, so `τ` is `zkPre + zkLig + 2` coins into a rep;
* `zkTauAlign`: `ZkTauAlign` holds at every message, the simulated strategy following `SchedC` short of `DeadTZ`
  (`schedC_simAtJZ_dead`), so the `τ` fork bound holds outright (`zkSimForkLeTau`), and with it masked knowledge
  soundness (`zk_KS`).
-/

namespace FlockSoundness.Game.Lock

section Batch

variable {T D I ζ : Type} {get : I → ℕ → Option (T × D)} {R : I → ζ → ζ → Prop}

/-- **`SlotAtD.batchLock_sched` with an invariant on the dead runs**: `Inv m Dead` at the tables in front, kept as each
game in front moves the dead runs one place down, holds of the dead runs `Q` sees. -/
theorem SlotAtD.batchLock_sched_inv (d : ζ) {i : I} {g g' : Game ζ} (L : Lock T D get R i g g')
    (post : List (Game ζ)) {slot k : ℕ} {Q : (List ζ → Prop) → Strategy (batch (g' :: post)) → Prop}
    (Inv : ℕ → (List ζ → Prop) → Prop)
    (hInv : ∀ m Dead, Inv (m + 1) Dead → Inv m fun outs => ∀ c, Dead (c :: outs))
    (hQ : ∀ Dead σ', Inv 0 Dead → Q Dead σ' → SlotAtD slot Dead k (batchLock d L post []) σ') :
    ∀ (pre : List (Game ζ)) {Dead : List ζ → Prop} {n : ℕ} {σ : Strategy (batch (pre ++ g' :: post))},
      Inv pre.length Dead → PreSched g' post k pre Dead Q n σ →
        SlotAtD slot Dead (n + k) (batchLock d L post pre) σ
  | [], Dead, _, σ, hI, ⟨hn, hq⟩ => by
      subst hn
      rw [Nat.zero_add]
      exact hQ Dead σ hI hq
  | h :: pre, Dead, _, σ, hI, ⟨n₀, n₁, hn, hF⟩ => by
      subst hn
      have hs := SlotAtD.rightSame_firstS (k := n₁ + k) hF (batchLock d L post pre) fun h' σ' ⟨_, τ, hR, hP⟩ => by
        have hP' : PreSched g' post k pre (fun outs => ∀ c, Dead (c :: outs)) Q n₁ τ := hP
        exact SlotAtD.rightSame_followR (Dead' := fun outs => ∀ c, Dead (c :: outs)) hR _ rfl
          (SlotAtD.batchLock_sched_inv d L post Inv hInv hQ pre (hInv pre.length Dead hI) hP')
      have key := SlotAtD.map
        (R₂ := fun i (a b : List ζ) => R i (a.getD (h :: pre).length d) (b.getD (h :: pre).length d))
        (Dead₂ := Dead) (fun x : ζ × List ζ => x.1 :: x.2) (fun x : ζ × List ζ => x.1 :: x.2)
        (fun _ _ _ h' => h'.2) (fun _ hx => hx) hs
      rw [Strategy.map_ofMap (fun x : ζ × List ζ => x.1 :: x.2) (h.interleave (batch (pre ++ g' :: post))) σ,
        ← Nat.add_assoc] at key
      exact key

end Batch

end FlockSoundness.Game.Lock

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession Exec ZkExec

section Reindex

variable {α β T Dd I J : Type} {getI : I → ℕ → Option (T × Dd)} {getJ : J → ℕ → Option (T × Dd)}
  {R : I → α → β → Prop} (emb : I → J) (sl : ℕ → ℕ) (hget : ∀ i s, getJ (emb i) (sl s) = getI i s)
  (hoff : ∀ i i' s, (∀ s', sl s' ≠ s) → getJ (emb i) s = getJ (emb i') s)

/-- **A slot off the moved ones has no table node** in a moved lockstep. -/
theorem noSlot_reindex_off {slot : ℕ} (hsl : ∀ s, sl s ≠ slot) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T Dd getI R i g g'),
      Lock.NoSlot slot (lockReindex emb sl hget hoff L)
  | _, _, _, .ret _ => .ret _
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L => .coin C fun c => noSlot_reindex_off hsl (L c)
  | _, _, _, .send M L => .send M fun m => noSlot_reindex_off hsl (L m)
  | _, _, _, .table s _ _ _ _ L => .table (sl s) (hsl s) fun t d => noSlot_reindex_off hsl (L t d)

end Reindex

section Shapes

/-- **The recursive levels' coins**, four lanes at each: seven at the last, eleven at each before it. -/
theorem coinsK_recShape (F K D : Type) (ext : ℕ) :
    ∀ (lv : Level) (levs : List Level), (∀ l ∈ lv :: levs, l.lanes = 4) →
      Lock.coinsK (recShape F K D ext (lv :: levs)) = 7 + 11 * levs.length
  | lv, [], hl => by
      have h4 := hl lv (by simp)
      simp only [recShape, Lock.coinsK_append, Lock.coinsK, coinsK_foldShape, coinsK_qbShape, List.length_nil]
      omega
  | lv, lv' :: rest, hl => by
      have h4 := hl lv (by simp)
      have ih := coinsK_recShape F K D ext lv' rest fun l hl' => hl l (List.mem_cons_of_mem _ hl')
      simp only [recShape, Lock.coinsK_append, Lock.coinsK, coinsK_foldShape, coinsK_qbShape, coinsK_oodShape, ih,
        List.length_cons]
      omega

/-- **Levels 1+ of the masked level 0's coins**, with a further level and four lanes at each. -/
theorem coinsK_tailShape (F K D : Type) (sch : Schedule) (lv₀ lv₁ : Level) (rest : List Level)
    (hl : ∀ l ∈ lv₁ :: rest, l.lanes = 4) :
    Lock.coinsK (tailShape F K D sch lv₀ (lv₁ :: rest)) = 14 + 11 * rest.length := by
  have := coinsK_recShape F K D sch.ext lv₁ rest hl
  simp only [tailShape, Lock.coinsK, Lock.coinsK_append, coinsK_oodShape, coinsK_qbShape, this]
  omega

/-- **The masked level 0's coins**: its two first coins, its `k0` lane rounds, its `e` extra draws and levels 1+. -/
theorem coinsK_ligPadShape (F K D : Type) (sch : Schedule) (e t : ℕ) (lv₀ lv₁ : Level) (rest : List Level)
    (hlevs : sch.levels = lv₀ :: lv₁ :: rest) (hl : ∀ l ∈ lv₁ :: rest, l.lanes = 4) :
    Lock.coinsK (ligPadShape F K D sch e t) = sch.k0 + e + 16 + 11 * rest.length := by
  have := coinsK_tailShape F K D sch lv₀ lv₁ rest hl
  simp only [ligPadShape, hlevs, Lock.coinsK, Lock.coinsK_append, coinsK_foldShapeZ, coinsK_replicate_coin,
    this]
  omega

/-- **A masked rep's masked phase's coins**: the zerocheck's, the lincheck's, the opening's of its `nExtra + 2` claims
and the masked level 0's. -/
theorem coinsK_maskedZ {F : Type} [Field F] [Fintype F] [DecidableEq F] (K D : Type) {mPts : ℕ} (T : TabZK F mPts)
    (nExtra : ℕ) (lv₀ lv₁ : Level) (rest : List Level) (hlevs : T.sch.levels = lv₀ :: lv₁ :: rest)
    (hl : ∀ l ∈ lv₁ :: rest, l.lanes = 4) :
    Lock.coinsK (zcShape F T.S.m ++ (lcShape F T.S.kLog ++ (opShape F (nExtra + 2) ++
      (ligPadShape F K D T.sch T.e T.t ++ [.msg F])))) =
      (T.S.m - 6 + 3) + (T.S.kLog - 6 + 3) + (nExtra + 2 + 1) + (T.sch.k0 + T.e + 16 + 11 * rest.length) := by
  simp only [Lock.coinsK_append, coinsK_zcShape, coinsK_lcShape, coinsK_opShape,
    coinsK_ligPadShape F K D T.sch T.e T.t lv₀ lv₁ rest hlevs hl, Lock.coinsK]
  omega

end Shapes

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-- **`τ`'s commitment is two coins into the inner proof.** -/
theorem dist_lockInnerZ_tau (Hs : Hashes F D Sl) (r : ℕ) (hr : r < 2) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ)
    (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s) (lev : IRep F D) (O : Oracle F)
    (c : Cap D) : Lock.Dist (2 + r) 2 (lockInnerZ Hs r hr dom s nP q pos rows tr lev O c) :=
  .coin _ fun _ => .send _ fun _ => .coin _ fun _ => .here

/-- **A rep's `τ` commitment is the masked phase's coins and two in, on the runs whose opening can end well**; the
others' Ligerito part is `PendL.reject`. -/
theorem distOn_lockRepZ_tau (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Oracle F) (extra : List (Claim F T.S.m)) (c : ℕ)
    (hc : c = 2 + Lock.coinsK (zcShape F T.S.m ++ (lcShape F T.S.kLog ++ (opShape F (extra.length + 2) ++
      (ligPadShape F K D T.sch T.e T.t ++ [.msg F]))))) :
    Lock.DistOn (2 + r) (fun p : PendZ F D Sl => p.lig = PendL.reject) c (lockRepZ A Hs T r hr C₀ extra) := by
  subst hc
  refine .other r (by omega) fun O cap => ?_
  refine Lock.DistOn.bind_on (P := fun x => x.2 ≠ PendL.reject) ?_ ?_
    (Lock.NoSlot.mono _ (noSlot_reindex_off _ (· + 4) _ _ (fun s => by show s + 4 ≠ 2 + r; omega) _)) _
    (repMaskedZC_on A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra)
  · intro j o o' h
    rcases j with ⟨lev, pads, tau⟩
    obtain ⟨hp, ht, h⟩ := h
    simp only at hp ht h
    subst hp ht
    obtain ⟨hrows, hrec⟩ := h
    rcases o with ⟨rows, v⟩
    rcases o' with ⟨rows', p⟩
    simp only at hrows hrec
    subst hrows
    refine Lock.Dist.distOn (Lock.Dist.bind_noSlot ?_ (dist_lockInnerZ_tau Hs r hr _ _ _ _ _ _ _ _ _ _))
    intro _ _ _ _
    exact .ret _
  · intro b y hb hy
    obtain ⟨v, -, hv⟩ := Out.of_bind hy
    cases hv
    exact not_not.mp hb

/-- An output whose rep `0`'s Ligerito part rejects is not accepted. -/
theorem outZC_accepted_reject₀ (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pend : PendZ F D Sl × PendZ F D Sl) (os : OpensZ F D Sl) (h : pend.1.lig = PendL.reject) :
    (outZC Hs T cap₀ pend os).accepted = false := by
  simp [outZC, h, PendL.reject]

/-- An output whose rep `1`'s Ligerito part rejects is not accepted. -/
theorem outZC_accepted_reject₁ (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pend : PendZ F D Sl × PendZ F D Sl) (os : OpensZ F D Sl) (h : pend.2.lig = PendL.reject) :
    (outZC Hs T cap₀ pend os).accepted = false := by
  simp [outZC, h, PendL.reject]

end FlockSoundness.Discharge.ZkLink

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

/-- **The session's table meets rep `r`'s `τ` past rep 0's `n` coins if `r = 1`, at every table choice**, where its
strategy plays rep 0 first, which ends after `n` coins, or reaches only `Dead` outcomes, every output that is not
accepted `Dead` (`Lock.SlotAtD.lead_fst_on`, `Lock.SlotAtD.lead_snd_on`). -/
theorem slotAtD_lockAfterZ_tau (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (n c : ℕ) (Dead : OutZC 𝔽 D Sl → Prop)
    (hacc : ∀ x : OutZC 𝔽 D Sl, x.accepted = false → Dead x)
    (hD : Lock.DistOn (2 + r) (fun p : PendZ 𝔽 D Sl => p.lig = Model.PendL.reject) c
      (lockRepZ A Hs T r hr C₀ (Model.extraClaims T.S T.ptLocal mPts pts)))
    (σ₁ : Strategy (tabAfterZCP A Hs T cap₀ pts))
    (hS : Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)) Dead n
      (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    Lock.SlotAtD (2 + r) Dead (r * n + c)
      (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
        (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts)) σ₁ := by
  refine Lock.SlotAtD.mapL _ _ ?_
  unfold lockAfterZ
  match r, hr, hD with
  | 0, _, hD =>
    rw [Nat.zero_mul, Nat.zero_add]
    exact Lock.SlotAtD.lead_fst_on disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _)
      (fun b y hb hy => by
        cases hy with
        | send os hy =>
          cases hy
          exact hacc _ (outZC_accepted_reject₀ Hs T cap₀ b os hb)) hS _ _ hD
      (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _)
  | 1, _, hD =>
    rw [Nat.one_mul]
    exact Lock.SlotAtD.lead_snd_on disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _)
      (fun b y hb hy => by
        cases hy with
        | send os hy =>
          cases hy
          exact hacc _ (outZC_accepted_reject₁ Hs T cap₀ b os hb)) hS _ _
      (Lock.noSlot_of_get_none (fun i => by simp [getZ]) _) hD

/-- **The primed prover reaches rep `r`'s `τ` node after `t` coins** in the session after `Commit`: `Lock.SlotAt` at
`advTauSZ`'s chooser, slot, fork event and lockstep. -/
def SlotTauZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) (2 + r)
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^τ_r` in the session is at most the fork after `t` coins of `TauForkEvZ`** where the primed prover reaches
rep `r`'s `τ` node after `t` coins: the node's conflict at its cap is `TauForkEvZ`'s at that cap. -/
theorem advTauSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotTauZ A Hs T pre post cap₀ r t σ) :
    advTauSZ A Hs T pre post cap₀ r σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        TauForkEvZ Hs pre.length r p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun _ d _ _ _ _ _ _ _ _ _ _ => ind_le_indZ fun ⟨_, hc⟩ => ⟨d, hc⟩) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

/-- **Table `n` opens nothing or is not accepted.** -/
def DeadTZ (n : ℕ) (outs : List (OutZC 𝔽 D Sl)) : Prop :=
  DeadZ n outs ∨ (outs.getD n dfltOutZC).accepted = false

omit [Inhabited D] [Inhabited Sl] in
/-- **Table `n` opening nothing or not accepted is read at position `n` alone.** -/
theorem deadTZ_depAt {D Sl : Type} [Inhabited D] [Inhabited Sl] (n : ℕ) :
    Lock.DepAt n (DeadTZ (D := D) (Sl := Sl) n) :=
  fun l l' e h => by
    unfold DeadTZ DeadZ at h ⊢
    rw [List.getD_eq_getElem?_getD] at h ⊢
    rwa [← e]

/-- **The `τ` conflict is `0` between runs whose table `n` opens nothing or is not accepted**: an output that is not
accepted opens no `τ`, and two that open nothing open the same `τ` value. -/
theorem ind_conflictTau_deadT (Hs : Hashes 𝔽 D Sl) (n r : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)}
    (hb : DeadTZ n b) (hb' : DeadTZ n b') :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opTauZ Hs r d (outs.getD n dfltOutZC)) x p) (b, b') = 0 := by
  have e : ∀ (o : OutZC 𝔽 D Sl) (p : Unit), (opTauZ Hs r d o p).isSome →
      opTauZ Hs r d o p = some (o.opens.inner r).tau.1 := fun o p h => by
    unfold opTauZ at h ⊢
    split_ifs at h ⊢
    · rfl
    · simp at h
  unfold ind
  split_ifs with hc
  · obtain ⟨p, h₁, h₂, hne⟩ := hc
    dsimp only at h₁ h₂ hne
    rcases hb with hb | hb
    · rcases hb' with hb' | hb'
      · unfold DeadZ at hb hb'
        rw [e _ p h₁, e _ p h₂, hb, hb'] at hne
        exact absurd rfl hne
      · unfold opTauZ at h₂
        rw [hb'] at h₂
        simp at h₂
    · unfold opTauZ at h₁
      rw [hb] at h₁
      simp at h₁
  · rfl

/-- **The `τ` slot from a schedule whose tables pause**: `SlotTauZ` after `n + (r · n₀ + c) + 1` coins, `c` rep `r`'s
`τ` distance on the runs that can end well, where the primed prover follows `SchedZ` with `r · n₀ + c` coins of the
session's table before the tables after it flip any, short of runs `Dead` at which the `τ` conflict is `0`, every run
whose table is not accepted `Dead` (`Lock.SlotAtD.batchLock_sched_inv`, `Lock.SlotAtD.batchLock_nil_followL`,
`slotAtD_lockAfterZ_tau`). -/
theorem slotTauZ_of_batch (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (n n₀ c : ℕ) (Dead : List (OutZC 𝔽 D Sl) → Prop)
    (hΦ : ∀ d b b', Dead b → Dead b' → ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
        opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b') = 0)
    (hI : ∀ l : List (OutZC 𝔽 D Sl), (l.getD pre.length dfltOutZC).accepted = false → Dead l)
    (hD : ∀ pts : Fin 2 → Fin mPts → 𝔽, Lock.DistOn (2 + r) (fun p : PendZ 𝔽 D Sl => p.lig = Model.PendL.reject) c
      (lockRepZ A Hs T r hr default (Model.extraClaims T.S T.ptLocal mPts pts)))
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (hF : SchedZ A Hs T pre post cap₀ Dead (r * n₀ + c) n₀ n σ) :
    SlotTauZ A Hs T pre post cap₀ r (n + (r * n₀ + c) + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <| Lock.SlotAtD.slotAt hΦ <|
    Lock.SlotAtD.batchLock_sched_inv dfltOutZC _ (post.map (· pts))
      (fun m Dead => ∀ l : List (OutZC 𝔽 D Sl), (l.getD m dfltOutZC).accepted = false → Dead l)
      (fun _ _ h l hl c => h (c :: l) hl)
      (fun _ _ hI' ⟨σ₁, hFL, hl⟩ => Lock.SlotAtD.batchLock_nil_followL dfltOutZC _ _ hFL
        (slotAtD_lockAfterZ_tau A Hs T default cap₀ pts r hr n₀ c _ (fun x hx rest => hI' (x :: rest) hx)
          (hD pts) σ₁ hl))
      (pre.map (· pts)) (by rw [List.length_map]; exact hI) (hF pts)

end Session

/-! ## At the masked simulator -/

/-- **A rep's `τ` is `zkPre + zkLig + 2` coins in at the deployed tables**, on the runs whose opening can end well:
the masked phase's coins (`coinsK_maskedZ`) at two link points per region and four lanes after level 0. -/
theorem zkTau_dist {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {spec : SessionSpec} {sts : Array Stmt}
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (hp : spec.points = 2) (j : ℕ) (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (r : ℕ) (hr : r < 2)
    (C₀ : Model.Oracle 𝔽) (pts : Fin 2 → Fin spec.mPts → 𝔽) :
    Lock.DistOn (2 + r) (fun p : PendZ 𝔽 D Sl => p.lig = Model.PendL.reject)
      (zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2)
      (lockRepZ A Hs (zkTabs h0 L hz hk hlay j) r hr C₀
        (Model.extraClaims (zkTabs h0 L hz hk hlay j).S (zkTabs h0 L hz hk hlay j).ptLocal spec.mPts pts)) := by
  obtain ⟨he, hm, hkl, hreg, hk0, hl⟩ := zkTabs_facts h0 L hz hk hlay j
  have h13 := (zkTabs h0 L hz hk hlay j).hm
  have hlen := zkTabs_len h0 L hz hk hlay j
  have h2 := (zkTabs h0 L hz hk hlay j).hshape.1
  obtain ⟨lv₀, lv₁, rest, hlevs⟩ : ∃ a b c, (zkTabs h0 L hz hk hlay j).sch.levels = a :: b :: c := by
    rcases hL : (zkTabs h0 L hz hk hlay j).sch.levels with _ | ⟨a, _ | ⟨b, c⟩⟩
    · rw [hL] at h2
      simp at h2
    · rw [hL] at h2
      simp at h2
    · exact ⟨a, b, c, rfl⟩
  rw [hlevs] at hl hlen
  simp only [List.tail_cons, List.length_cons] at hl hlen
  refine distOn_lockRepZ_tau A Hs _ r hr C₀ _ _ ?_
  rw [coinsK_maskedZ K D _ _ lv₀ lv₁ rest hlevs hl, FlockSoundness.extraClaims_length]
  unfold zkPre zkLig
  rw [hp]
  omega

namespace Gen

variable [S : ZkSetup]

/-- **`ZkTauAlign` holds at every message**: the `τ` fork round is the tables in front's coins, rep 0's if `r = 1`,
the link coin and `zkPre + zkLig + 2` (`zkTau_dist`); the simulated strategy follows `SchedC` at `r · repRounds` and
that count short of `DeadTZ` (`schedC_simAtJZ_dead` on `zkStreams`, as `zkPadsAlign`), so it reaches the `τ` node
there (`slotTauZ_of_batch`), and `advTauSZ_le_forkC` gives the alignment, table `j`'s position in the session being
`j` (`take_gamesZC_length`). -/
theorem zkTauAlign (x : Claims.Msg) : ZkTauAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r u
  have hjS : j < sts.size := by rw [← zk_tables_length hset]; exact hj
  have ht : zkTbl sts j = j := by unfold zkTbl; exact ite_eq_left hjS
  have hv : (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inr r)))).val =
      zkFront spec sts j + (r.val * repRounds true spec sts[zkTbl sts j]! +
        (zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2)) + 1 := by
    show zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r +
      zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2 = _
    rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with
      ⟨a, e⟩ | ⟨a, e⟩ <;> rw [e, a] <;> unfold zkBase zkFront <;> omega
  obtain ⟨-, hR, -, -⟩ := zkRd_facts h0 L hz hk hlay j
  have hkR : r.val * repRounds true spec sts[zkTbl sts j]! +
      (zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2) <
      repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! := by
    have := Nat.mul_le_mul_right (repRounds true spec sts[zkTbl sts j]!) (show r.val ≤ 1 by omega)
    omega
  have hsched : SchedC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!)
      (DeadTZ j) (r.val * repRounds true spec sts[zkTbl sts j]! +
        (zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 2))
      (repRounds true spec sts[zkTbl sts j]!) (zkFront spec sts j)
      (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u) := by
    rw [ht] at hkR ⊢
    rw [zkFront_eq hjS]
    have hJ : sts.size ≤ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
      rw [tabsJ_length, zk_tables_length hset]
    refine schedC_simAtJZ_dead execArith ZkExec.hs512 _ _ _ _ spec.dg _ _ _
      (capOf := fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x) _ (fun _ _ _ => rfl) (zkRoot P x ω) _ _ _ _
      (fun _ h => Or.inl h) (deadTZ_depAt _) (fun t => repRounds true spec sts[t]!) _ _ hkR
      (fromS (fun t => repRounds true spec sts[t]!) (j + 1) (sts.size - (j + 1))) (fun p hp => ?_) fun pts => ?_
    · have := mem_fromS _ _ _ p hp
      dsimp only [finJ]
      omega
    · have h := zkLive_sched hJ h0 (zk_mPts_le hs hin hset) P ω u pts (zkStreams_length hz hset)
      rw [zkStreams_split hz hjS] at h
      exact h
  have ha := advTauSZ_le_forkC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ _ r.val _ _
    (slotTauZ_of_batch _ _ _ _ _ _ r.val r.isLt _ _ _ (DeadTZ j)
      (fun e b b' hb hb' => by
        rw [take_gamesZC_length]
        exact ind_conflictTau_deadT _ _ _ _ hb hb')
      (fun l hl => Or.inr (by rw [take_gamesZC_length] at hl; exact hl))
      (fun pts => zkTau_dist h0 L hz hk hlay (zk_points hs hin hset) j _ _ r.val r.isLt _ pts)
      _ (schedZ_of_schedC _ _ _ _ _ _ _ _ _ _ _ hsched))
  rw [take_gamesZC_length] at ha
  unfold ZkTauAlignQ
  rw [hv]
  exact ha

end Gen

/-- **`ZkTauAlign` holds at every message**: the `τ` fork round is the tables in front's coins, rep 0's if `r = 1`,
the link coin and `zkPre + zkLig + 2` (`zkTau_dist`); the simulated strategy follows `SchedC` at `r · repRounds` and
that count short of `DeadTZ` (`schedC_simAtJZ_dead` on `zkStreams`, as `zkPadsAlign`), so it reaches the `τ` node
there (`slotTauZ_of_batch`), and `advTauSZ_le_forkC` gives the alignment, table `j`'s position in the session being
`j` (`take_gamesZC_length`). -/
theorem zkTauAlign (x : Claims.Msg) : ZkTauAlign x :=
  Gen.zkTauAlign (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **The masked simulator's fork bound at `τ` holds at every message.** -/
theorem zkSimForkLeTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsTauZ i :=
  zkSimForkLeTauZ_of_align (zkTauAlign x)

end Gen

/-- **The masked simulator's fork bound at `τ` holds at every message.** -/
theorem zkSimForkLeTau (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsTauZ i :=
  Gen.zkSimForkLeTau (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **Masked knowledge soundness**, at every message: `zk_KS_of_tauAlign` with `τ`'s alignment (`zkTauAlign`). -/
theorem zk_KS : Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_tauAlign fun x _ => zkTauAlign x

end Gen

/-- **Masked knowledge soundness**, at every message: `zk_KS_of_tauAlign` with `τ`'s alignment (`zkTauAlign`). -/
theorem zk_KS : Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS (S := S₀)

end FlockDeploy.KS
