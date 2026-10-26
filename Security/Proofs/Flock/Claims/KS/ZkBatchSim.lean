import Proofs.Flock.Claims.KS.ZkLiveSched
import Proofs.Flock.Claims.KS.ZkBatch

/-!
# The masked simulated strategy follows the batch's schedule at the deployed tables

After `Commit`, table `j` of the masked session is at position `j` of the batch (`sessAfterZC_split`), and the
simulated strategy there (`simAtJZ`) is the session simulation's (`simRJ`) at the link points, every dead branch
answering the openings with `nulOpZ` (`simAtJZ_heq`). Where the live strategy submits its rounds stream by stream, the
tables in front of `j` first, the simulated strategy advances them by `Lock.Front` and then table `j` by `Lock.Sess`
(`front_simRJ`), which is `SchedC` (`Lock.preSched_of_front`). The masked simulator's live strategy submits its rounds
on `zkStreams` (`zkLive_sched`), split at table `j` as `front_simRJ` reads them (`zkStreams_split`), and the slot's
count is short of both of table `j`'s reps' rounds (`zk_slot_lt`):

* `simSessZV_heq`, `simAtJZ_heq`: the simulated strategy after `Commit` at table `j` is the simulation's there;
* `deadZ_depAt`, `front_simAtJZ`, `schedC_simAtJZ_dead`, `schedC_simAtJZ`: so it follows `SchedC` wherever the live
  strategy follows the streams, short of runs read at table `j` alone that hold wherever it opens nothing;
* `zkFront_eq`, `zk_slot_lt`: the tables in front's coins are `nsFrom`'s at `repRounds`, and the slot's count
  `r · repRounds + zkCapDist` is below `2 · repRounds`;
* `zkBatchAlign`, `zkSlotAlign`: `ZkBatchAlign` holds at every message, and with it `ZkSlotAlign`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)
open Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-- **Table `n` opening nothing is read at position `n` alone.** -/
theorem deadZ_depAt {D Sl : Type} [Inhabited D] [Inhabited Sl] (n : ℕ) : Lock.DepAt n (DeadZ (D := D) (Sl := Sl) n) :=
  fun l l' e h => by
    unfold DeadZ at h ⊢
    rw [List.getD_eq_getElem?_getD] at h ⊢
    rwa [← e]

section Gen

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts : ℕ}
  (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
  (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
  (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
  (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
  (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)

/-- The masked session's simulated strategy is the session simulation's, carried along
`sessionZC_eq_modelSession`. -/
theorem simSessZV_heq (Pr : Strategy (liveSession dg ts.length mPts N)) :
    HEq (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr)
      ((Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
        (simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
          (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)).strategy Pr) := by
  unfold simSessZV
  exact eqRec_heq_self _ _

/-- **`simAtJZ` at the link points `pts`** is the session simulation's strategy there, its dead branches answered
with empty openings. -/
theorem simAtJZ_heq (Pr : Strategy (liveSession dg ts.length mPts N)) (j : Fin ts.length)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    HEq (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec Pr.1 Pr rfl j rfl pts)
      (((Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
        (simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
          (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)).strategy Pr).2 pts) := by
  have h₁ := cast_coin_heq ((congrArg (fun c => sessAfterZC (gamesZC A Hs ts c))
      (simSessZV_fst A Hs ts ok pL opn dg dec one fin capOf N hdec Pr)).trans
      (sessAfterZC_split A Hs ts (capOf Pr.1) j))
    (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr).2 pts
  have h₂ := heq_snd_coin (sessionZC_eq_modelSession A Hs ts)
    (simSessZV_heq A Hs ts ok pL opn dg dec one fin capOf N hdec Pr).symm pts
  exact h₁.trans h₂.symm

/-- **The simulated strategy after `Commit` at table `j` advances the tables in front by `Lock.Front`, then table `j`
by `Lock.Sess`**, where the live strategy submits the tables in front's rounds stream by stream, then table `j`'s, then
only later tables', short of runs `Dead` read at table `j` alone wherever it opens nothing. -/
theorem front_simAtJZ (hfin : ∀ t r proofs, fin t r proofs = []) (Pr : Strategy (liveSession dg ts.length mPts N))
    (j : Fin ts.length) {Dead : List (OutZC 𝔽 D Sl) → Prop} (hnul : ∀ outs, DeadZ j outs → Dead outs)
    (hDep : Lock.DepAt j.val Dead) (Rf : ℕ → ℕ) (k R : ℕ) (hk : k < R + R) (rest : List ℕ)
    (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p) (pts : Fin 2 → Fin mPts → 𝔽)
    (hS : SchedLJ dg Pr.1 pts N (fun _ _ => [])
      (fromS Rf 0 j ++ (List.replicate R (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest))) (Pr.2 pts)) :
    Lock.Front k (Lock.Sess (some k) R (Tables.KgJ (outJ Hs ts) j (capOf Pr.1 j)) (repJ A Hs ts j pts 0)
        (repJ A Hs ts j pts 1)) (nsFrom Rf 0 j) Dead
      (((gamesZC A Hs ts (capOf Pr.1)).take j).map (· pts) ++ tabAfterZC A Hs (ts.get j) (capOf Pr.1 j) pts ::
        ((gamesZC A Hs ts (capOf Pr.1)).drop (j + 1)).map (· pts))
      (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec Pr.1 Pr rfl j rfl pts) := by
  have hA := Claims.Bridge.Sim.All.mono (R' := fun b _ => pL b)
    (D' := Claims.Bridge.DeadAt (outJ Hs ts) (repJ A Hs ts)) (fun _ _ h => h.1) (fun _ _ h => h) _
    (Claims.Bridge.all_simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
      (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec Pr.1 pts)
  refine tr_list (fun l σ => Lock.Front k (Lock.Sess (some k) R (Tables.KgJ (outJ Hs ts) j (capOf Pr.1 j))
      (repJ A Hs ts j pts 0) (repJ A Hs ts j pts 1)) (nsFrom Rf 0 j) Dead l σ) ?_
    (simAtJZ_heq A Hs ts ok pL opn dg dec one fin capOf N hdec Pr j pts)
    (front_simRJ (out := outJ Hs ts) (nul := nulOpZ) (Dead := Dead) hfin j dfltOutZC
      (fun outs cap pend h => hnul outs (by unfold DeadZ; rw [h]; rfl)) hDep Rf k R hk rest hrest N _ _ _ _
      (Pr.2 pts) hS hA)
  have e : (gamesZC A Hs ts (capOf Pr.1)).map (· pts) =
      List.ofFn (Tables.tabG (outJ Hs ts) (capOf Pr.1) fun j => repJ A Hs ts j pts) := by
    simp only [gamesZC, List.map_ofFn]
    rfl
  have hs : gamesZC A Hs ts (capOf Pr.1) = (gamesZC A Hs ts (capOf Pr.1)).take j ++
      tabAfterZC A Hs (ts.get j) (capOf Pr.1 j) :: (gamesZC A Hs ts (capOf Pr.1)).drop (j + 1) :=
    FlockSoundness.ofFn_split (fun i => tabAfterZC A Hs (ts.get i) (capOf Pr.1 i)) j
  show _ = List.ofFn (Tables.tabG (outJ Hs ts) (capOf Pr.1) fun j => repJ A Hs ts j pts)
  rw [← e]
  conv_rhs => rw [hs]
  simp only [List.map_append, List.map_cons]

/-- **The simulated strategy after `Commit` at table `j` follows `SchedC`** after the tables in front's coins, where
at every link point the live strategy submits the tables in front's rounds stream by stream, `Rf t` on each of table
`t`'s reps, then table `j`'s `R` on rep 0 and `R` on rep 1, then only later tables', short of runs `Dead` read at
table `j` alone wherever it opens nothing. -/
theorem schedC_simAtJZ_dead (hfin : ∀ t r proofs, fin t r proofs = []) (rootB₀ : Array ByteArray)
    (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀) (j : Fin ts.length) {T : TabZK 𝔽 mPts}
    (hT : ts.get j = T) {Dead : List (OutZC 𝔽 D Sl) → Prop} (hnul : ∀ outs, DeadZ j outs → Dead outs)
    (hDep : Lock.DepAt j.val Dead) (Rf : ℕ → ℕ) (k R : ℕ) (hk : k < R + R) (rest : List ℕ)
    (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p)
    (hS : ∀ pts, SchedLJ dg Pr.1 pts N (fun _ _ => [])
      (fromS Rf 0 j ++ (List.replicate R (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest))) (Pr.2 pts)) :
    SchedC A Hs T ((gamesZC A Hs ts (capOf rootB₀)).take j) ((gamesZC A Hs ts (capOf rootB₀)).drop (j + 1))
      (capOf rootB₀ j) Dead k R (nsFrom Rf 0 j).sum
      (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) := by
  subst hP hT
  intro pts
  refine Lock.preSched_of_front _ _ k ?_ _ (nsFrom Rf 0 j) ?_
    (front_simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec hfin Pr j hnul hDep Rf k R hk rest hrest pts
      (hS pts))
  · intro Dead σ hσ
    obtain ⟨hF, hL⟩ := hσ
    exact ⟨_, hF.followL fun x h => h x.2, hL rfl⟩
  · rw [length_nsFrom, List.length_map, take_gamesZC_length]

/-- **`schedC_simAtJZ_dead` at the runs whose table `j` opens nothing.** -/
theorem schedC_simAtJZ (hfin : ∀ t r proofs, fin t r proofs = []) (rootB₀ : Array ByteArray)
    (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀) (j : Fin ts.length) {T : TabZK 𝔽 mPts}
    (hT : ts.get j = T) (Rf : ℕ → ℕ) (k R : ℕ) (hk : k < R + R) (rest : List ℕ)
    (hrest : ∀ p ∈ rest, 2 * j.val + 2 ≤ p)
    (hS : ∀ pts, SchedLJ dg Pr.1 pts N (fun _ _ => [])
      (fromS Rf 0 j ++ (List.replicate R (2 * j.val) ++ (List.replicate R (2 * j.val + 1) ++ rest))) (Pr.2 pts)) :
    SchedC A Hs T ((gamesZC A Hs ts (capOf rootB₀)).take j) ((gamesZC A Hs ts (capOf rootB₀)).drop (j + 1))
      (capOf rootB₀ j) (DeadZ j) k R (nsFrom Rf 0 j).sum
      (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) :=
  schedC_simAtJZ_dead A Hs ts ok pL opn dg dec one fin capOf N hdec hfin rootB₀ Pr hP j hT (fun _ h => h)
    (deadZ_depAt j) Rf k R hk rest hrest hS

end Gen

/-! ## At the deployed tables -/

/-- **The tables in front's coins** are `nsFrom`'s at `repRounds`. -/
theorem zkFront_eq {spec : SessionSpec} {sts : Array Stmt} {j : ℕ} (hj : j < sts.size) :
    zkFront spec sts j = (nsFrom (fun t => repRounds true spec sts[t]!) 0 j).sum := by
  rw [sum_nsFrom, zkFront, zkTbl, ite_eq_left hj, List.range_eq_range']

namespace Gen

variable [S : ZkSetup]

/-- **The slot's count is short of both of table `j`'s reps' rounds**: `zkCapDist` is the rep's rounds before
Ligerito and the cap's round inside it (`zkRdAt_cap`), below the rep's rounds (`zkRd_facts`). -/
theorem zk_slot_lt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (r : Fin 2) (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    r.val * repRounds true spec sts[zkTbl sts j]! + zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val <
      repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! := by
  have hcap := zkRdAt_cap h0 L hz hk hlay (zk_points hs hin hset) j r ℓ
  obtain ⟨-, hRe, hlt, -⟩ := zkRd_facts h0 L hz hk hlay j
  have hl := hlt ℓ
  change _ = zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r +
    zkPre spec sts[zkTbl sts j]! + capRound (zkSchT sts j).k0 (ℓ.val + 1) + 2 at hcap
  rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with ⟨a, e⟩ | ⟨a, e⟩ <;>
    rw [e, a] at hcap <;> rw [a] <;> unfold zkFront zkBase at hcap <;> omega

end Gen

/-- **The slot's count is short of both of table `j`'s reps' rounds**: `zkCapDist` is the rep's rounds before
Ligerito and the cap's round inside it (`zkRdAt_cap`), below the rep's rounds (`zkRd_facts`). -/
theorem zk_slot_lt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (r : Fin 2) (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    r.val * repRounds true spec sts[zkTbl sts j]! + zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val <
      repRounds true spec sts[zkTbl sts j]! + repRounds true spec sts[zkTbl sts j]! :=
  Gen.zk_slot_lt (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (j := j) (r := r) (ℓ := ℓ)

namespace Gen

variable [S : ZkSetup]

/-- **`ZkBatchAlign` holds at every message**: the masked simulator's live strategy submits its rounds on
`zkStreams` (`zkLive_sched`), split at table `j` (`zkStreams_split`), so the simulated strategy follows `SchedC`
(`schedC_simAtJZ`) at the slot's count (`zk_slot_lt`). -/
theorem zkBatchAlign (x : Claims.Msg) : ZkBatchAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u _
  have hkR := zk_slot_lt hs hin hset h0 L hz hk hlay j r ℓ
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

end Gen

/-- **`ZkBatchAlign` holds at every message**: the masked simulator's live strategy submits its rounds on
`zkStreams` (`zkLive_sched`), split at table `j` (`zkStreams_split`), so the simulated strategy follows `SchedC`
(`schedC_simAtJZ`) at the slot's count (`zk_slot_lt`). -/
theorem zkBatchAlign (x : Claims.Msg) : ZkBatchAlign x :=
  Gen.zkBatchAlign (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **`ZkSlotAlign` holds at every message.** -/
theorem zkSlotAlign (x : Claims.Msg) : ZkSlotAlign x :=
  zkSlotAlign_of_batch (zkBatchAlign x)

end Gen

/-- **`ZkSlotAlign` holds at every message.** -/
theorem zkSlotAlign (x : Claims.Msg) : ZkSlotAlign x :=
  Gen.zkSlotAlign (S := S₀) (x := x)

end FlockDeploy.KS
