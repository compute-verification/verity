import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's KS at the deployed instances, from one wait's bound

`Flock.KS_of_oneWait` at the deployed records `FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel`, their
columns' zero the empty column (`instZeroListCol`): `K = 2^256 > 0` (`cap_pos`) and the tables' errors are nonnegative
(`εtable_nonneg`). What stays a hypothesis: one wait's bound (`Flock.OneWait`), and `k ≥ 8` wherever the verifier
accepts some run.
-/

namespace FlockDeploy.KS

open Flock
open FlockSoundness

theorem cap_pos : 0 < Claims.Flock.Numbers.cap := by
  unfold Claims.Flock.Numbers.cap
  positivity

theorem tableErr_nonneg (zk : Bool) (spec : SessionSpec) (st : Stmt) : 0 ≤ tableErr zk spec st := by
  unfold tableErr
  split
  · unfold Accounting.padLinkError
    positivity
  · exact tableError_nonneg _ _

/-- The tables' errors, summed, are nonnegative. -/
theorem εtable_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ εtable zk x := by
  unfold εtable
  refine List.sum_nonneg fun a ha => ?_
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
  exact tableErr_nonneg zk p.1 p.2

/-- **The deployed plain verifier: KS from one wait's bound.** -/
theorem plain_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, 8 ≤ (deployPlain rel).k x ∨ Claims.Flock.NoAcc H (deployPlain rel).V x)
    (h : Claims.Flock.OneWait H (deployPlain rel)) : Claims.Flock.KS H (deployPlain rel) :=
  Claims.Flock.KS_of_oneWait H (deployPlain rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg false x) h

/-- **The deployed masked (`--zk`) verifier: KS from one wait's bound.** -/
theorem zk_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, 8 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc H (deployZk rel).V x)
    (h : Claims.Flock.OneWait H (deployZk rel)) : Claims.Flock.KS H (deployZk rel) :=
  Claims.Flock.KS_of_oneWait H (deployZk rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg true x) h

end FlockDeploy.KS
