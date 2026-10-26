import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.Recursion.VerdictRowL
import Proofs.Flock.Claims.Recursion.RoundCounts

/-!
# Row L's target

`TargetL` (row L at each rep, where the rows are asked) holds (`targetL_holds`). Row L's `params`, `initialCap` and `cap`
hold by the reader's pin and the `after` base check (`after_of_base`, `DecAt.rootB_link`); its `fc` by row C's `c` and
the reader's pin once `z` is tied (`targetL`, from `TargetZ`): each rep stream's round-2 coin, as the session records
it, is the live coin the reader read rep `2 i + r`'s `z` from. That holds because the record `decR` hands over is the
reader's (`recJ`), whose rep streams have the schedule's round counts stream by stream (`DecAt.hz`, `RoundCounts`), so
`TargetZ` is proved (`targetZ`). The same per-stream counts give row R's `hsz` where the schedule's coin counts are the
shape's `roundSizes` (`hsz_of_sched`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP ReadOkR VerdictR holdsR)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

namespace Top

/-- **Row L's coin tie**: where the rows are asked, each rep stream's round-2 coin, as the session records it, is the
live coin the reader took the rep's zerocheck `z` from. -/
def TargetZ (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size → ∀ r : Fin 2,
      (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}

/-- **The `after` check at the decision**, from the base checks. -/
theorem after_of_base {x : Msg} {ms : List Msg} {cs : List Coin} (hr : ReadOkR rd x ms cs)
    (hb : ∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) (D : DecAt rd x ms cs) :
    VerdictR.At FlockDeploy.sha512 .after D.d.s x D.d.link D.d.c₁ D.d.mids D.d.f := by
  have h := hb .after (by simp [VerdictP.bases])
  unfold holdsR at h
  rw [actR_of_dec hr D] at h
  exact h

/-- **Row L's target from the coin tie.** -/
theorem targetL (acc : Msg → List Msg → List Coin → ℕ → Array F128) (hZ : TargetZ rd P) : TargetL rd P acc := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact D.rowL_at cls (acc x ms cs) i r hst hf hx hi hti D.points_eq (idx_lt D hi r)
    (shaped_of_base hr hb D _ (idx_lt D hi r)) (after_of_base hr hb D) (hZ x ms cs hA D cls i hi r)

/-- **Row L's coin tie holds** where the rows are asked: the rep streams' names are distinct at a dense inner
statement (`DecAt.names_nodup`), and each stream has the schedule's rounds (`DecAt.hz`). -/
theorem targetZ : TargetZ rd P := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact D.hz cls i (D.names_nodup hst hf hx) hi hti r

/-- **Row L's target holds.** -/
theorem targetL_holds (acc : Msg → List Msg → List Coin → ℕ → Array F128) : TargetL rd P acc :=
  targetL acc targetZ

/-- **Row R's `hsz` where the rows are asked**, in `TargetR`'s form, where the schedule's coin counts at table `i` are
`roundSizes` at its shape (`hsched`). -/
theorem hsz_of_sched (acc : Msg → List Msg → List Coin → ℕ → Array F128) {x : Msg} {ms : List Msg} {cs : List Coin}
    (hA : At rd P x ms cs) (D : DecAt rd x ms cs) (cls : HClass) {i : ℕ} (hi : i < D.sts.size)
    (hsched : (RecDeploy.repSchedAt D.spec D.sts[i]!).map (·.1) = FlockVBridge.Family.roundSizes (D.sh cls i)) :
    ∀ r : Fin 2, (D.rep cls (acc x ms cs) i r).rounds.toList.map Array.size =
      FlockVBridge.Family.roundSizes (D.sh cls i) := by
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact fun r => D.hsz_of_sched cls i (acc x ms cs) (D.names_nodup hst hf hx) hi hti r hsched

end Top

end RecDeploy
