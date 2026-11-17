module

public import Proofs.Flock.Claims.KS.ForkAlign
public import Proofs.Flock.Claims.KS.RepDistZ
public import Proofs.Flock.Claims.KS.SlotPost
public import Proofs.Flock.Claims.KS.ZkFork

/-! The masked simulated fork's alignment at the caps, from the slot's coin count. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {D : Type}

theorem forkC_push {α I : Type} {get : I → ℕ → Option (Cap D × D)} (F : α → α → ℝ) :
    ∀ (t : ℕ) {i : I} {g g' : Game α} (L : Lock (Cap D) D get (fun i a b => Pushed get i → a = b) i g g')
      (σ' : Strategy g'), Pushed get i → forkC F t g' σ' = forkC F t g (Lock.oracleStrat pushC L σ')
  | 0, _, g, g', L, σ', hi => by
      rw [forkC_zero, forkC_zero]
      unfold expect₂
      rw [show (fun a => expect (F a) g' σ') = fun a => expect (F a) g (Lock.oracleStrat pushC L σ') from
        funext fun a => expect_push (F a) L σ' hi]
      exact expect_push _ L σ' hi
  | _ + 1, _, _, _, .ret h, _, hi => by
      rw [forkC_ret, forkC_ret, h hi]
  | t + 1, _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      rw [forkC_coin, forkC_coin]
      congr 1
      exact Finset.sum_congr rfl fun c _ => forkC_push F t (L c) (σ' c) hi
  | t + 1, _, _, _, .send _ L, σ', hi => by
      rw [forkC_send, forkC_send]
      exact forkC_push F (t + 1) (L σ'.1) σ'.2 hi
  | t + 1, _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ _ _ hset hkeep L, σ', hi => by
      rw [forkC_send, forkC_send]
      refine forkC_push F (t + 1) _ σ'.2 ?_
      intro s' c d h
      by_cases hs : s' = s
      · subst hs
        rw [hset] at h
        cases h
        rfl
      · rw [hkeep _ _ _ hs] at h
        exact hi s' c d h

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {Sl : Type}

omit [Fintype K] in
theorem forkC_pushStrat [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (Φ : List (OutZC F D Sl) → List (OutZC F D Sl) → ℝ) (t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    forkC Φ t _ (pushStrat A Hs T cap₀ pre post σ) = forkC Φ t _ σ :=
  (forkC_push Φ t (pushSess A Hs T cap₀ pre post) σ fun s c d h => by simp [Lock.getPair, getP] at h).symm

end FlockSoundness.Discharge.ZkLink

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

omit [Inhabited D] [Inhabited Sl] in
open Classical in
theorem opRZ_mem {H : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D H}
    {sch : FlockSoundness.Accounting.Schedule} {r ℓ : ℕ} {cap : Model.Cap D} {out : OutZC 𝔽 D Sl}
    {p : Fin (2 ^ (lvOf sch ℓ).logLen)} (h : (opRZ H E Lf sch r ℓ cap out p).isSome) :
    ∃ x ∈ slotOpensZ out r ℓ (lvOf sch ℓ).queries, x.1 = p.val := by
  unfold opRZ Exec.opOfL at h
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1⟩

omit [Fintype K] [DecidableEq K] in
theorem capAtZ_of_sessRelZ {A : Model.Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {T : TabZK 𝔽 mPts} {C₀ : Model.Oracle 𝔽}
    {cap₀ : Model.Cap D} {n : ℕ} {i : IZ 𝔽 D × IZ 𝔽 D} {a b : List (OutZC 𝔽 D Sl)} {r : Fin 2} {ℓ : ℕ}
    {t : Model.Oracle 𝔽} {d : Model.Cap D} (hR : SessRelZ A Hs T C₀ cap₀ n i a b)
    (hg : Lock.getPair (getZ 0) (getZ 1) i (slotOf r.val (ℓ + 1) + 4) = some (t, d))
    {p : Fin (2 ^ (lvOf T.sch (ℓ + 1)).logLen)}
    (hp : (opRZ Hs.H Hs.E Hs.Lf T.sch r.val (ℓ + 1) d (b.getD n dfltOutZC) p).isSome) :
    CapAtZ r.val ℓ (b.getD n dfltOutZC) d := by
  obtain ⟨_, _, htop⟩ := hR
  obtain ⟨-, -, -, -, hs₀, hs₁⟩ := htop
  obtain ⟨y, hy, -⟩ := opRZ_mem hp
  have hreq : ∃ req, ((if r.val = 0 then (b.getD n dfltOutZC).pend.1 else (b.getD n dfltOutZC).pend.2).lig.reqs.find?
      fun req => req.level == ℓ + 1) = some req := by
    unfold slotOpensZ at hy
    split at hy
    · simp at hy
    · exact ⟨_, ‹_›⟩
  obtain ⟨req, hfind⟩ := hreq
  have hmem := List.mem_of_find?_eq_some hfind
  have hlev : req.level = ℓ + 1 := by simpa using List.find?_some hfind
  have key : ∀ (ir : IRep 𝔽 D) (pr : Model.PendL 𝔽 D), RecStruct T.sch ir pr → req ∈ pr.reqs →
      ir (ℓ + 1) = some (t, d) → pr.caps[ℓ]? = some d := by
    intro ir pr hst hm hi
    have hlt : ℓ + 1 < pr.reqs.length := by
      have h1 : req.level ∈ pr.reqs.map Model.Req.level := List.mem_map_of_mem hm
      rw [hst.1, List.mem_range, hlev] at h1
      exact h1
    obtain ⟨t', d', hi', hc⟩ := hst.2.2.1 (ℓ + 1) (by omega) hlt
    rw [hi] at hi'
    cases hi'
    simpa using hc
  have hs4 : ∀ r' : ℕ, r' < 2 → ¬ slotOf r' (ℓ + 1) + 4 < 4 := fun r' _ => by omega
  have hsub : ∀ r' : ℕ, slotOf r' (ℓ + 1) + 4 - 4 = 2 * (ℓ + 1) + r' := fun r' => by unfold slotOf; omega
  obtain ⟨r, hr⟩ := r
  interval_cases r
  · have hi : i.1.lev (ℓ + 1) = some (t, d) := by
      have hg₀ : getZ 0 i.1 (slotOf 0 (ℓ + 1) + 4) = i.1.lev (ℓ + 1) := by
        unfold getZ getRep
        rw [ite_eq_right (hs4 0 (by omega)), hsub, ite_eq_left (by omega),
          show (2 * (ℓ + 1) + 0) / 2 = ℓ + 1 by omega]
      have e : Lock.getPair (getZ 0) (getZ 1) i (slotOf 0 (ℓ + 1) + 4) =
          (getZ 0 i.1 (slotOf 0 (ℓ + 1) + 4)).orElse fun _ => getZ 1 i.2 (slotOf 0 (ℓ + 1) + 4) := rfl
      rw [e, hg₀] at hg
      cases h : i.1.lev (ℓ + 1) with
      | none =>
          have hg₁ : getZ 1 i.2 (slotOf 0 (ℓ + 1) + 4) = none := by
            unfold getZ getRep
            rw [ite_eq_right (hs4 0 (by omega)), hsub, ite_eq_right (by omega)]
          rw [h, hg₁] at hg
          cases hg
      | some v =>
          rw [h] at hg
          exact hg
    simp only [CapAtZ, ↓reduceIte]
    simp only [↓reduceIte] at hmem
    exact key _ _ hs₀ hmem hi
  · have hi : i.2.lev (ℓ + 1) = some (t, d) := by
      have hg₀ : getZ 0 i.1 (slotOf 1 (ℓ + 1) + 4) = none := by
        unfold getZ getRep
        rw [ite_eq_right (hs4 1 (by omega)), hsub, ite_eq_right (by omega)]
      have hg₁ : getZ 1 i.2 (slotOf 1 (ℓ + 1) + 4) = i.2.lev (ℓ + 1) := by
        unfold getZ getRep
        rw [ite_eq_right (hs4 1 (by omega)), hsub, ite_eq_left (by omega),
          show (2 * (ℓ + 1) + 1) / 2 = ℓ + 1 by omega]
      have e : Lock.getPair (getZ 0) (getZ 1) i (slotOf 1 (ℓ + 1) + 4) =
          (getZ 0 i.1 (slotOf 1 (ℓ + 1) + 4)).orElse fun _ => getZ 1 i.2 (slotOf 1 (ℓ + 1) + 4) := rfl
      rw [e, hg₀, hg₁] at hg
      exact hg
    simp only [CapAtZ, one_ne_zero, ↓reduceIte]
    simp only [one_ne_zero, ↓reduceIte] at hmem
    exact key _ _ hs₁ hmem hi

omit [Fintype K] [DecidableEq K] in
theorem sessForkZ_le (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (n : ℕ) (r : Fin 2) (ℓ : ℕ) (t : Model.Oracle 𝔽) (d : Model.Cap D)
    (i₁ : IZ 𝔽 D × IZ 𝔽 D) (a₁ b₁ : List (OutZC 𝔽 D Sl)) (i₂ : IZ 𝔽 D × IZ 𝔽 D) (a₂ b₂ : List (OutZC 𝔽 D Sl))
    (h₁ : SessRelZ A Hs T C₀ cap₀ n i₁ a₁ b₁) (h₂ : SessRelZ A Hs T C₀ cap₀ n i₂ a₂ b₂)
    (hg₁ : Lock.getPair (getZ 0) (getZ 1) i₁ (slotOf r.val (ℓ + 1) + 4) = some (t, d))
    (hg₂ : Lock.getPair (getZ 0) (getZ 1) i₂ (slotOf r.val (ℓ + 1) + 4) = some (t, d)) :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
        (fun outs : List (OutZC 𝔽 D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r.val (ℓ + 1) d (outs.getD n dfltOutZC)) x p)
        (b₁, b₂) ≤
      ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ForkEvZ Hs T.sch n r.val ℓ x.1 x.2) (b₁, b₂) :=
  ind_le_indZ fun ⟨p, hs₁, hs₂, hne⟩ =>
    ⟨d, capAtZ_of_sessRelZ h₁ hg₁ hs₁, capAtZ_of_sessRelZ h₂ hg₂ hs₂, p, hs₁, hs₂, hne⟩

def SlotRSZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r ℓ t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) (slotOf r (ℓ + 1) + 4)
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (opRAccZ Hs.H Hs.E Hs.Lf T.sch pre.length r (ℓ + 1) d) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

theorem advRSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : Fin 2) (ℓ t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotRSZ A Hs T pre post cap₀ r.val ℓ t σ) :
    advRSZ A Hs T pre post cap₀ r.val (ℓ + 1) σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        ForkEvZ Hs T.sch pre.length r.val ℓ p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun t d i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hg₁ hg₂ =>
      (ind_conflictAcc_le Hs.H Hs.E Hs.Lf T.sch pre.length r.val (ℓ + 1) d b₁ b₂).trans
        (sessForkZ_le A Hs T default cap₀ pre.length r ℓ t d i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hg₁ hg₂)) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

theorem ind_conflictAcc_dead (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (sch : FlockSoundness.Accounting.Schedule) (n r ℓ : ℕ) (d : Model.Cap D) {b b' : List (OutZC 𝔽 D Sl)}
    (h : ¬ AccUpTo n b ∨ ¬ AccUpTo n b') :
    ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
      Rewinding.ConflictAt (opRAccZ H E Lf sch n r ℓ d) x p) (b, b') = 0 := by
  unfold ind
  split_ifs with hc
  · obtain ⟨p, h₁, h₂, _⟩ := hc
    rcases h with h | h
    · exact absurd (acc_of_opRAccZ H E Lf sch n r ℓ d h₁) h
    · exact absurd (acc_of_opRAccZ H E Lf sch n r ℓ d h₂) h
  · rfl

theorem slotRSZ_of_first (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r ℓ n j : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (Q : (pts : Fin 2 → Fin mPts → 𝔽) → List (OutZC 𝔽 D Sl) →
      Strategy (batch (tabAfterZCP A Hs T cap₀ pts :: post.map (· pts))) → Prop)
    (hF : ∀ pts, Lock.PreFirst (tabAfterZCP A Hs T cap₀ pts) (post.map (· pts)) (pre.map (· pts))
      (fun outs => ¬ AccUpTo pre.length outs) (Q pts) n (pushStrat A Hs T cap₀ pre post σ pts))
    (hQ : ∀ (pts : Fin 2 → Fin mPts → 𝔽) (cs : List (OutZC 𝔽 D Sl))
      (σ' : Strategy (batch (tabAfterZCP A Hs T cap₀ pts :: post.map (· pts)))), Q pts cs σ' →
      ∀ (plur' : Lock.Chooser (Model.Oracle 𝔽) (Model.Cap D) (List (OutZC 𝔽 D Sl)))
        (Φ' : Model.Cap D → List (OutZC 𝔽 D Sl) → List (OutZC 𝔽 D Sl) → ℝ),
        (∀ e b b', ¬ AccUpTo pre.length (cs ++ b) → ¬ AccUpTo pre.length (cs ++ b') → Φ' e b b' = 0) →
          Lock.SlotAt plur' (slotOf r (ℓ + 1) + 4) Φ' j
            (Lock.batchLock dfltOutZC
              (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ A T cap₀ x ∧ TopRelZ Hs T default cap₀ i x b)
                (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T default cap₀ pts))
              (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts) []) σ') :
    SlotRSZ A Hs T pre post cap₀ r ℓ (n + j + 1) σ :=
  Lock.SlotAt.coin _ fun pts => Lock.SlotAt.mono _ <|
    Lock.SlotAt.batchLock_first dfltOutZC _ (post.map fun g => g pts) (pre.map fun g => g pts)
      (plur := plurSZ Hs T pre.length)
      (Φ := fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p,
        Rewinding.ConflictAt (opRAccZ Hs.H Hs.E Hs.Lf T.sch pre.length r (ℓ + 1) d) x p) (b, b'))
      (fun e b b' hb _ => ind_conflictAcc_dead Hs.H Hs.E Hs.Lf T.sch pre.length r (ℓ + 1) e (b := b) (b' := b')
        (.inl hb)) (hF pts) (hQ pts)

theorem slotRSZ_of_lead (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r ℓ n j : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (Q : (pts : Fin 2 → Fin mPts → 𝔽) → List (OutZC 𝔽 D Sl) →
      Strategy (batch (tabAfterZCP A Hs T cap₀ pts :: post.map (· pts))) → Prop)
    (hF : ∀ pts, Lock.PreFirst (tabAfterZCP A Hs T cap₀ pts) (post.map (· pts)) (pre.map (· pts))
      (fun outs => ¬ AccUpTo pre.length outs) (Q pts) n (pushStrat A Hs T cap₀ pre post σ pts))
    (hL : ∀ (pts : Fin 2 → Fin mPts → 𝔽) (cs : List (OutZC 𝔽 D Sl))
      (σ' : Strategy (batch (tabAfterZCP A Hs T cap₀ pts ::
        post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))), Q pts cs σ' →
        ∃ σ₁, Lock.Leads (tabAfterZCP A Hs T cap₀ pts)
            (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts)) σ₁
            (Strategy.ofMap (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2)
              ((tabAfterZCP A Hs T cap₀ pts).interleave
                (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) σ') ∧
          Lock.SlotAtD (slotOf r (ℓ + 1) + 4) (fun x => ∀ rest, ¬ AccUpTo pre.length (cs ++ x :: rest)) j
            (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T default cap₀ i x b)
              (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T default cap₀ pts)) σ₁) :
    SlotRSZ A Hs T pre post cap₀ r ℓ (n + j + 1) σ :=
  slotRSZ_of_first A Hs T pre post cap₀ r ℓ n j σ Q hF fun pts cs σ' hq _ _ hΦ' => by
    obtain ⟨σ₁, hS, hD⟩ := hL pts cs σ' hq
    exact Lock.SlotAtD.batchLock_nil (fun e x x' r r' hx hx' => hΦ' e _ _ (hx r) (hx' r')) dfltOutZC _ _ σ₁ σ'
      hS hD

theorem slotAtD_lockAfterZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (C₀ : Model.Oracle 𝔽)
    (cap₀ : Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) (r : ℕ) (hr : r < 2) (ℓ n c : ℕ) (Dead : OutZC 𝔽 D Sl → Prop)
    (hD : Lock.Dist (slotOf r (ℓ + 1) + 4) c (lockRepZ A Hs T r hr C₀ (Model.extraClaims T.S T.ptLocal mPts pts)))
    (σ₁ : Strategy (tabAfterZCP A Hs T cap₀ pts))
    (hS : Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)) Dead n
      (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
      (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    Lock.SlotAtD (slotOf r (ℓ + 1) + 4) Dead (r * n + c)
      (Lock.mapL (R₂ := fun i o b => ∃ x, o = orcOutZ (Sl := Sl) A T cap₀ x ∧ TopRelZ Hs T C₀ cap₀ i x b)
        (orcOutZ A T cap₀) (fun _ a _ hr => ⟨a, rfl, hr⟩) (lockAfterZ A Hs T C₀ cap₀ pts)) σ₁ := by
  refine Lock.SlotAtD.mapL _ _ ?_
  unfold lockAfterZ
  match r, hr, hD with
  | 0, _, hD =>
    rw [Nat.zero_mul, Nat.zero_add]
    exact Lock.SlotAtD.lead_fst disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _ hD
      (Lock.noSlot_of_get_none (fun i => by simp [getZ, getRep, slotOf]) _)
  | 1, _, hD =>
    rw [Nat.one_mul]
    exact Lock.SlotAtD.lead_snd disjoint_getZ (fun _ _ _ _ => .send _ fun _ => .ret _) hS _ _
      (Lock.noSlot_of_get_none (fun i => by simp [getZ, getRep, slotOf]) _) hD

/-- At the masked simulator these premises hold at one table only: it sends every table's final messages and openings
after the last live message (`simEndJ`). -/
theorem slotRSZ_of_sched (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r : ℕ) (hr : r < 2)
    (ℓ n n₀ c : ℕ)
    (hD : ∀ pts : Fin 2 → Fin mPts → 𝔽, Lock.Dist (slotOf r (ℓ + 1) + 4) c
      (lockRepZ A Hs T r hr default (Model.extraClaims T.S T.ptLocal mPts pts)))
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (Q : (pts : Fin 2 → Fin mPts → 𝔽) → List (OutZC 𝔽 D Sl) →
      Strategy (batch (tabAfterZCP A Hs T cap₀ pts :: post.map (· pts))) → Prop)
    (hF : ∀ pts, Lock.PreFirst (tabAfterZCP A Hs T cap₀ pts) (post.map (· pts)) (pre.map (· pts))
      (fun outs => ¬ AccUpTo pre.length outs) (Q pts) n (pushStrat A Hs T cap₀ pre post σ pts))
    (hL : ∀ (pts : Fin 2 → Fin mPts → 𝔽) (cs : List (OutZC 𝔽 D Sl))
      (σ' : Strategy (batch (tabAfterZCP A Hs T cap₀ pts ::
        post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))), Q pts cs σ' →
        ∃ σ₁, Lock.Leads (tabAfterZCP A Hs T cap₀ pts)
            (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts)) σ₁
            (Strategy.ofMap (fun x : OutZC 𝔽 D Sl × List (OutZC 𝔽 D Sl) => x.1 :: x.2)
              ((tabAfterZCP A Hs T cap₀ pts).interleave
                (batch (post.map fun g : (Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl) => g pts))) σ') ∧
          Lock.Lead (fun pend => Game.send (OpensZ 𝔽 D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os))
            (fun x => ∀ rest, ¬ AccUpTo pre.length (cs ++ x :: rest)) n₀
            (repZCP A Hs T 0 (Model.extraClaims T.S T.ptLocal mPts pts))
            (repZCP A Hs T 1 (Model.extraClaims T.S T.ptLocal mPts pts)) σ₁) :
    SlotRSZ A Hs T pre post cap₀ r ℓ (n + (r * n₀ + c) + 1) σ :=
  slotRSZ_of_lead A Hs T pre post cap₀ r ℓ n _ σ Q hF fun pts cs σ' hq => by
    obtain ⟨σ₁, hS, hl⟩ := hL pts cs σ' hq
    exact ⟨σ₁, hS, slotAtD_lockAfterZ A Hs T default cap₀ pts r hr ℓ n₀ c _ (hD pts) σ₁ hl⟩

end Session

def ZkSlotAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    SlotRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val ℓ.val
      (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)


namespace Gen

variable [S : ZkSetup]

def ZkSlotAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    SlotRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val ℓ.val
      (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

end Gen

namespace Gen

variable [S : ZkSetup]

theorem zkForkAlign_of_slot {x : Claims.Msg} (h : ZkSlotAlign x) : ZkForkAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u
  have ha := advRSZ_le_forkC execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ _ r ℓ.val _ _
    (h s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u)
  rw [take_gamesZC_length] at ha
  exact ha

end Gen

theorem zkForkAlign_of_slot {x : Claims.Msg} (h : ZkSlotAlign x) : ZkForkAlign x :=
  Gen.zkForkAlign_of_slot (S := S₀) (x := x) (h := h)

namespace Gen

variable [S : ZkSetup]

theorem zk_KS_of_slot (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSlotAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_align (fun x hno => zkForkAlign_of_slot (ha x hno)) hc ht hp hpads

end Gen

theorem zk_KS_of_slot (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSlotAlign x)
    (hc : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkClash x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_slot (S := S₀) (ha := ha) (hc := hc) (ht := ht) (hp := hp) (hpads := hpads)

end FlockDeploy.KS
