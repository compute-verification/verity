import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.Recursion.VerdictRowL

/-!
# Row L's target, from the coin tie

`TargetL` (row L at each rep, where the rows are asked) holds wherever `TargetZ` does (`targetL`). Row L's `params`,
`initialCap` and `cap` hold outright there: the first two by the reader's pin, `cap` by the `after` base check
(`after_of_base`, `DecAt.rootB_link`). Its `fc` holds by row C's `c` and the reader's pin once `z` is tied: `TargetZ`
says each rep stream's round-2 coin, as the session records it, is the live coin the reader read rep `2 i + r`'s `z`
from. The two can differ only where the record's per-stream round counts differ from the schedule's (`repLens`): custody
pairs the flattened rounds with the round messages, so the offsets agree when each stream has its schedule's rounds.
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

end Top

end RecDeploy
