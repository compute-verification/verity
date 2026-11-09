module

public import Proofs.Flock.SessionLoop.RoundsPassInner0
public import Proofs.Flock.SessionLoop.RoundsPassZc
public import Proofs.Flock.SessionLoop.RoundsPassPiop
public import Proofs.Flock.SessionLoop.RoundsPassOpen
public import Proofs.Flock.SessionLoop.RoundsPassL0
public import Proofs.Flock.SessionLoop.RoundsPassSwitch
public import Proofs.Flock.SessionLoop.SchedSlot
public import Proofs.Flock.SessionLoop.SchedTotalHolds
public import Proofs.Flock.SessionLoop.Fill

/-!
# Every round of the honest session loop passes the frame's checks at `valsDev honestVals`

`Loop.RoundsPass` at `withInner`'s session loop on `Loop.finHonest`, at its honest device
(**`LeanRelease.roundsPass_valsDev`**), wherever `cross` passes the device's rounds: after every history of
`2 ≤ g.length ≤ refK x` whose coins the frame takes, the round there passes. The rep's schedule on its own table is not
refused (`ofLoop_sched_unrefused`, at `Loop.scheduleTotal`), so the round is one of the rep's schedule or one of the
inner proof's:

* **`LeanRelease.schedTaken_slot`**: before the rep's inner proof (`Loop.sched_slot_at`) the round is the rep's first
  (`schedTaken_zcBind`), in its PIOP (`schedTaken_piop`), in the ring switch from its rank check's (`schedTaken_switch`),
  the opening's binding (`schedTaken_openBind`), level 0's (`schedTaken_l0`) or from level 0's close on
  (`schedTaken_lig`), and the frame takes it (`SchedTaken`, so `roundOk_of_schedTaken`);
* at the end of the rep's schedule, the inner proof's round 0 (`roundOk_inner0`), and past it its rounds 1 to 3
  (`roundOk_innerAt`).

**`LeanRelease.fillRoundEq_finHonest`**: so `FillRoundEq` holds there at the honest worker
(`fillRoundEq_honestVals`), at every `vs`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open Loop in
set_option maxHeartbeats 1000000 in
/-- **The frame takes a rep's round before its inner proof where it takes its coins**: at a history whose next round
is rep stream `2j + r`'s, from its first (`Masked.repBase`) to the last before its inner proof (`Masked.innerOff 0`),
whose coins the loop's frame takes, on the loop's own tables where `src` samples the tape's level 0: the round is the
rep's first, in its PIOP, in the ring switch from its rank check's, the opening's binding, level 0's or from level 0's
close on, and the frame takes it in each. -/
theorem schedTaken_slot (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) (h1 : Masked.repBase x (2 * j.val + r.val) ≤ ops.length)
    (h2 : ops.length < Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0)) :
    (ofLoop Masked.dataE L₁ (Loop.realOf src)).SchedTaken x w (ν, t) ((m₀, ρ₀) :: ops) := by
  have hST := Loop.scheduleTotal x
  have hj := Masked.lt_of_jOf j
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, -, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have eS : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j) +
        f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, ← Masked.repBase_eq hsx, hst]
  have eR : Masked.rankSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    unfold Masked.rankSlot
    rw [getElem!_pos (Masked.stsOf x) j.val hj]
    try rfl
  simp only [eS, Masked.innerOff] at h2
  rcases Nat.lt_or_ge ops.length (Masked.rankSlot x j.val r.val) with hA | hA
  · rcases Nat.eq_or_lt_of_le h1 with h0 | h0
    · exact schedTaken_zcBind L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r h0.symm
    · exact schedTaken_piop L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r
        (jj := ops.length - Masked.repBase x (2 * j.val + r.val)) (by omega) (by omega)
        (by rw [eR] at hA; unfold piopNR; omega)
  · rcases Nat.lt_or_ge ops.length (Masked.repSlot x (2 * j.val + r.val) fun _ => 0) with hB | hB
    · exact schedTaken_switch L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r
        (k := ops.length - Masked.rankSlot x j.val r.val) (by omega) hB
    · simp only [eS] at hB
      obtain ⟨i, hi⟩ : ∃ i, ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i :=
        ⟨ops.length - (Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stAt x j)),
          by simp only [eS]; omega⟩
      have hi' := hi
      simp only [eS] at hi'
      by_cases hl0 : i ≤ (Masked.zkTab x j).sch.k0 + 3
      · rcases Nat.eq_zero_or_pos i with rfl | hi0
        · exact schedTaken_openBind L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r hi
        · exact schedTaken_l0 L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r hi hi0 hl0
      · exact schedTaken_lig L₁ src hx hSv hS0 hST ν t m₀ ρ₀ ops hlt href j r hi (by omega) (by omega)

open Loop in
set_option maxHeartbeats 1000000 in
/-- **Every round of `withInner`'s session loop on `finHonest` passes the frame's checks at `valsDev honestVals`**,
wherever `cross` passes the device's rounds: at a `Live` statement the loop serves, where `src` samples the tape's
level 0, after every history of `2 ≤ g.length ≤ refK x` whose coins the frame takes. The rep's schedule on its own
table is not refused there (`ofLoop_sched_unrefused`); a round before the end of it is the rep's own, which the frame
takes (`schedTaken_slot`), the round at its end is the inner proof's round 0 (`roundOk_inner0`), and one past it is
among its rounds 1 to 3 (`roundOk_innerAt`). -/
theorem roundsPass_valsDev (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Serves x) (hS0 : Level0Src src x w)
    (hcr : ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
      (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src
        (realOf src)).refusesCoins x (ν, t) g = false →
      (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src
        (realOf src)).cross x g.length w (ν, t) g
        (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src)
          (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src)) x w (ν, t) g
          (computedOf ((ofLoopLink (withInner Masked.dataE L₁ (realOf src)
            (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src)).computed x w (ν, t) g))) = true) :
    RoundsPass ((ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src))))
      src (realOf src)).withDevice (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src)
        (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src)))) x w := by
  intro ν t g h2 hK href
  rw [refusesCoins_withDevice] at href
  have hc := hcr ν t g h2 hK href
  obtain ⟨⟨m₀, ρ₀⟩, ops, rfl⟩ : ∃ p ops, g = p :: ops := by
    cases g with
    | nil => simp at h2
    | cons p ops => exact ⟨p, ops, rfl⟩
  have hlt : ops.length < refK x := by
    simp only [List.length_cons] at hK
    omega
  have hpos : 0 < ops.length := by
    simp only [List.length_cons] at h2
    omega
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · obtain ⟨_, hp, -⟩ := hx.1
    exact absurd (refK_of_none hs (decodeRec_none_of_parse hp)) (by omega)
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hlast : (ofLoop Masked.dataE (withFiles (withLink (withInner Masked.dataE L₁ (realOf src)
      (finHonest (ofLoopLink L₁ src (realOf src)))) src) src) (realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE _ x
  have hk := kindAt_masked_of (ofLoop Masked.dataE (withFiles (withLink (withInner Masked.dataE L₁ (realOf src)
      (finHonest (ofLoopLink L₁ src (realOf src)))) src) src) (realOf src)) (x := x) (j := ops.length + 1)
    (by omega) (by rw [hlast]; omega)
  obtain ⟨tr, hir, rsched, hre, hrs⟩ := ofLoop_sched_unrefused (withFiles (withLink (withInner Masked.dataE L₁
    (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src) src) src x w hx hSv hS0 (scheduleTotal x) ν t
    m₀ ρ₀ ops href hk hs hq
  rcases hloc : CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, i⟩
  rw [hloc] at hre hrs
  dsimp only at hre hrs
  have hcs : (ops.map fun p => CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  rcases Nat.lt_or_ge i rsched.size with hi | hi
  · obtain ⟨j, r, -, h1, h2⟩ := sched_slot_at hx hSv hs hcs (by rw [List.length_map]; exact hlt) hloc
      (realTabs_of src x w (ν, t) hs) hre hrs hi
    exact roundOk_of_schedTaken finS.honestVals
      (schedTaken_slot (withFiles (withLink (withInner Masked.dataE L₁ (realOf src)
        (finHonest (ofLoopLink L₁ src (realOf src)))) src) src) src hx hSv hS0 ν t m₀ ρ₀ ops hlt href j r
        (by omega) (by omega)) hc
  · rcases Nat.eq_or_lt_of_le hi with he | hgt
    · exact roundOk_inner0 L₁ src (realOf src) _ rfl hx hSv (fun ρ _ _ _ hs' => realTabs_of src x w ρ hs') ν t m₀ ρ₀
        ops hlt href hs hcs hloc hre hrs he hc
    · exact roundOk_innerAt L₁ src (realOf src) _ rfl hx hSv (fun ρ _ _ _ hs' => realTabs_of src x w ρ hs') ν t m₀ ρ₀
        ops hlt href hs hcs hloc hre hrs hgt hc

open Loop in
/-- **`FillRoundEq` at the honest worker of `withInner`'s session loop on `finHonest`**, at every `vs`, wherever `cross`
passes the device's rounds: `fillRoundEq_honestVals` at `roundsPass_valsDev`, the loop's `Finish` being round
`refK x + 1` (`loop_last`). -/
theorem fillRoundEq_finHonest (vs : finS.SwVals) (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Serves x) (hS0 : Level0Src src x w)
    (hcr : ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
      (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src
        (realOf src)).refusesCoins x (ν, t) g = false →
      (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src
        (realOf src)).cross x g.length w (ν, t) g
        (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src)
          (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src)) x w (ν, t) g
          (computedOf ((ofLoopLink (withInner Masked.dataE L₁ (realOf src)
            (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src)).computed x w (ν, t) g))) = true) :
    FillRoundEq finS vs (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src))))
      src (realOf src)) CoinOpen.shaH (valsWk finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ (realOf src)
        (finHonest (ofLoopLink L₁ src (realOf src)))) src (realOf src))) x w := by
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ (realOf src) (finHonest (ofLoopLink L₁ src (realOf src)))) src
      (realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE _ x
  exact fillRoundEq_honestVals vs _ x w (by omega) (roundsPass_valsDev L₁ src hx hSv hS0 hcr)

end LeanRelease

end Claims.ZKServer
