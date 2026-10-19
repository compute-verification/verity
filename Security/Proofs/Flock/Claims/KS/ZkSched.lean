import Proofs.Flock.Claims.KS.ZkAlign
import Proofs.Flock.Claims.KS.RepDistDeploy
import Proofs.Flock.Claims.KS.ZkStep

/-!
# The masked fork round of a cap, split at the schedule

`ZkSlotAlign` asks that the primed simulated strategy reach rep `r`'s level-`ℓ + 1` table node after `zkRd j i` coins.
The masked simulator submits the rounds stream by stream, table by table and in each rep 0 first (`zkStream`), so that
node comes after the tables in front's coins, rep 0's if `r = 1`, and a distance of rep `r`'s cap (as
`slotRSZ_of_sched` counts it). With four lanes after level 0, as `Zk.schedule` has (`DimSeam`), that distance is
`capRound`'s count (as at plain, `ligSlotDist_capRound`) and the three add up to `zkRd`'s:

* `dist_lockRepZ_capRound`, `dist_lockRepZ_zkCap`: a masked rep's level-`ℓ + 1` cap is `zkCapDist` coins in, at every
  level (past the last there is no cap, `noSlot_lockRepZ_ge`);
* `schedOf_lanes`, `zkTabs_facts`, `zkRdAt_cap`: at the deployed tables, the tables in front's coins (`zkFront`), rep
  0's (`repRounds`) and `zkCapDist` add up to `zkRdAt`;
* `ZkFrontAlign`, `zkSlotAlign_of_front`: so `ZkSlotAlign` is the slot at that split count.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open Claims.Bridge.Live

/-! ## A masked rep's caps at `capRound` -/

/-- A masked rep's coins to its level-`ℓ + 1` cap, with four lanes after level 0: the zerocheck's, the lincheck's, the
opening's of its `2 · regions + 2` claims, level 0's two first coins and `e` extra draws, then `capRound`. -/
def zkCapDist {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ} (T : TabZK F mPts) (ℓ : ℕ) : ℕ :=
  capRound T.sch.k0 (ℓ + 1) + T.e + (2 * T.S.regions.length + 2 + 1) + (T.S.kLog - 6 + 3) + (T.S.m - 6 + 3)

/-- **A masked rep's level-`ℓ + 1` cap is `capRound`'s count in**, with four lanes after level 0, at every level
(`dist_lockRepZ`, `ligSlotDist_capRound`; past the last cap there is no such node, `noSlot_lockRepZ_ge`). -/
theorem dist_lockRepZ_capRound {F K D Sl : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] {mPts : ℕ} (A : Model.Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ) (hr : r < 2)
    (C₀ : Model.Oracle F) (extra : List (Model.Claim F T.S.m)) (hl : ∀ lv ∈ T.sch.levels.tail, lv.lanes = 4)
    (ℓ : ℕ) :
    Lock.Dist (slotOf r (ℓ + 1) + 4)
      (capRound T.sch.k0 (ℓ + 1) + T.e + (extra.length + 2 + 1) + (T.S.kLog - 6 + 3) + (T.S.m - 6 + 3))
      (lockRepZ A Hs T r hr C₀ extra) := by
  by_cases hlt : ℓ < T.sch.levels.tail.length
  · have h := dist_lockRepZ A Hs T r hr C₀ extra ℓ
    have e := ligSlotDist_capRound T.sch.k0 T.sch.levels.tail hl ℓ hlt
    unfold maskedSlotDist ligPadSlotDist at h
    have e' : ligSlotDist T.sch.levels.tail ℓ + T.e + T.sch.k0 + 1 + 1 + (extra.length + 2 + 1) +
        (T.S.kLog - 6 + 3) + (T.S.m - 6 + 3) =
        capRound T.sch.k0 (ℓ + 1) + T.e + (extra.length + 2 + 1) + (T.S.kLog - 6 + 3) + (T.S.m - 6 + 3) := by
      omega
    rw [e'] at h
    exact h
  · exact .noSlot (noSlot_lockRepZ_ge A Hs T r hr C₀ extra ℓ (by rw [List.length_tail] at hlt; omega))

/-- **Every rep's level-`ℓ + 1` cap is `zkCapDist` coins in**, at every link point, with four lanes after level 0. -/
theorem dist_lockRepZ_zkCap {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (hl : ∀ lv ∈ T.sch.levels.tail, lv.lanes = 4) (r : ℕ) (hr : r < 2) (C₀ : Model.Oracle 𝔽) (ℓ : ℕ)
    (pts : Fin 2 → Fin mPts → 𝔽) :
    Lock.Dist (slotOf r (ℓ + 1) + 4) (zkCapDist T ℓ)
      (lockRepZ A Hs T r hr C₀ (Model.extraClaims T.S T.ptLocal mPts pts)) := by
  have h := dist_lockRepZ_capRound A Hs T r hr C₀ (Model.extraClaims T.S T.ptLocal mPts pts) hl ℓ
  rwa [FlockSoundness.extraClaims_length] at h

/-! ## At the deployed tables -/

/-- **`DimSeam`'s schedule folds four lanes at every level after level 0**, read as the model's. -/
theorem schedOf_lanes {s : Flock.Schedule} (hd : Refine.DimSeam s) :
    ∀ lv ∈ (Refine.schedOf s).levels.tail, lv.lanes = 4 := by
  obtain ⟨-, -, -, -, hl4, -, -⟩ := hd
  intro lv hlv
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hlv
  simp only [Refine.schedOf, List.length_tail, List.length_map, Array.length_toList] at hi
  have h4 := hl4 (i + 1) (by omega) (by omega)
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some] at h4
  simp only [Refine.schedOf, List.getElem_tail, List.getElem_map, Array.getElem_toList, Refine.levelOf]
  exact h4

/-- **A deployed table's numbers**: its extra draws, its statement's `m`, `kLog` and regions, its schedule's `k0`, and
four lanes after level 0. -/
theorem zkTabs_facts {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (j : ℕ) :
    (zkTabs h0 L hz hk hlay j).e = 2 ∧ (zkTabs h0 L hz hk hlay j).S.m = sts[zkTbl sts j]!.m ∧
      (zkTabs h0 L hz hk hlay j).S.kLog = sts[zkTbl sts j]!.c.kLog ∧
      (zkTabs h0 L hz hk hlay j).S.regions.length = sts[zkTbl sts j]!.regions.size ∧
      (zkTabs h0 L hz hk hlay j).sch.k0 = (zkSchT sts j).k0 ∧
      ∀ lv ∈ (zkTabs h0 L hz hk hlay j).sch.levels.tail, lv.lanes = 4 := by
  unfold zkTabs zkSchT zkTbl
  split_ifs with h
  · rw [getElem!_pos sts j h]
    obtain ⟨-, -, -, -, -, hdim, -⟩ := ZkExec.zk_schedule_seams _ _ _ (hz j h)
    refine ⟨rfl, rfl, rfl, ?_, rfl, schedOf_lanes hdim⟩
    rw [dite_eq_left h]
    show (Refine.regionsOf sts[j]).length = sts[j].regions.size
    simp [Refine.regionsOf]
  · rw [getElem!_pos sts 0 h0]
    obtain ⟨-, -, -, -, -, hdim, -⟩ := ZkExec.zk_schedule_seams _ _ _ (hz 0 h0)
    refine ⟨rfl, rfl, rfl, ?_, rfl, schedOf_lanes hdim⟩
    rw [dite_eq_right h]
    show (Refine.regionsOf sts[0]).length = sts[0].regions.size
    simp [Refine.regionsOf]

/-- The coins of the tables in front of table `j`'s place: two reps of each. -/
def zkFront (spec : SessionSpec) (sts : Array Stmt) (j : ℕ) : ℕ :=
  ((List.range (zkTbl sts j)).map fun i => 2 * repRounds true spec sts[i]!).sum

/-- **The masked fork round of a cap, split at the schedule**: the link coin, the tables in front's coins, rep 0's if
`r = 1`, then `zkCapDist`, at two link points per region. -/
theorem zkRdAt_cap {spec : SessionSpec} {sts : Array Stmt} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (hp : spec.points = 2) (j : ℕ) (r : Fin 2) (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    zkFront spec sts j + (r.val * repRounds true spec sts[zkTbl sts j]! + zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val) +
        1 = zkRdAt spec sts j (.inl (.inl (r, ℓ))) := by
  obtain ⟨he, hm, hkl, hreg, hk0, -⟩ := zkTabs_facts h0 L hz hk hlay j
  have h13 := (zkTabs h0 L hz hk hlay j).hm
  rw [hm] at h13
  show _ = zkB2 (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r +
    zkPre spec sts[zkTbl sts j]! + capRound (zkSchT sts j).k0 (ℓ.val + 1) + 2
  rcases zkB2_cases (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) r with ⟨a, e⟩ | ⟨a, e⟩ <;>
    rw [e, a] <;> unfold zkBase zkFront zkCapDist zkPre <;> rw [hk0, hp] <;> omega

/-- **The simulated fork reaches each cap's slot at the schedule's count**: `ZkSlotAlign` with `zkRd j i` split as
`zkRdAt_cap` splits it, the tables in front's coins (`zkFront`), rep 0's if `r = 1` (`repRounds`), and `zkCapDist`. -/
def ZkFrontAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length) (u : zkTape x),
    SlotRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val ℓ.val
      (zkFront spec sts j + (r.val * repRounds true spec sts[zkTbl sts j]! +
        zkCapDist (zkTabs h0 L hz hk hlay j) ℓ.val) + 1)
      (zkSimAt hs hin hset h0 L hz hk hlay hd P ω j hj C u)

/-- **The slot alignment at the schedule's count** (`zkRdAt_cap`, two link points per region by `zk_points`). -/
theorem zkSlotAlign_of_front {x : Claims.Msg} (h : ZkFrontAlign x) : ZkSlotAlign x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u
  have hv : (zkRd hset h0 L hz hk hlay j (.inl (.inl (r, ℓ)))).val = zkRdAt spec sts j (.inl (.inl (r, ℓ))) := rfl
  rw [hv, ← zkRdAt_cap h0 L hz hk hlay (zk_points hs hin hset) j r ℓ]
  exact h s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r ℓ u

end FlockDeploy.KS
