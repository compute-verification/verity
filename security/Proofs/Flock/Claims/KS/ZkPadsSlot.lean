module

public import Proofs.Flock.Claims.KS.ZkBatch
public import Proofs.Flock.Claims.KS.ZkBatchSim
public import Proofs.Flock.Claims.KS.ZkPadsPair
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.KS.ZkLiveSched
import Proofs.Flock.Soundness.Game

/-! The masked simulated fork's alignment at the pads. -/

@[expose] public section

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}

omit [Inhabited D] [Inhabited Sl] in
theorem dist_lockRepZ_pads (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Model.Oracle 𝔽) (extra : List (Model.Claim 𝔽 T.S.m)) : Lock.Dist r 0 (lockRepZ A Hs T r hr C₀ extra) := by
  unfold lockRepZ
  exact .here

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

def SlotPadsZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Prop :=
  Lock.SlotAt (plurSZ Hs T pre.length) r
    (fun d b b' => ind (fun x : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC 𝔽 D Sl) => opPadsAtZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    t (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

theorem advPadsSZ_le_forkC (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D) (r t : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (h : SlotPadsZ A Hs T pre post cap₀ r t σ) :
    advPadsSZ A Hs T pre post cap₀ r σ ≤
      forkC (fun a a' => ind (fun p : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) =>
        PadsForkEvZ Hs T.N T.q pre.length r p.1 p.2) (a, a')) t _ σ :=
  (Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun _ d _ _ _ _ _ _ _ _ _ _ => ind_le_indZ fun ⟨p, hc⟩ => ⟨d, p, hc⟩) h).trans
    (le_of_eq (forkC_pushStrat A Hs T cap₀ pre post _ t σ))

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

namespace Gen

variable [S : ZkSetup]

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

theorem zkPadsAlign (x : Claims.Msg) : ZkPadsAlign x :=
  Gen.zkPadsAlign (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

theorem zkSimForkLePads (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  zkSimForkLePads_of (zkPadsAlign x)

end Gen

theorem zkSimForkLePads (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  Gen.zkSimForkLePads (S := S₀) (x := x)

end FlockDeploy.KS
