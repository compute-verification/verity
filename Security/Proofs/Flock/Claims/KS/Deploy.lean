import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's KS at the deployed instances, from one wait's bound

`Flock.KS_of_oneWait_dedup` at the deployed records `FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel`, at
their decoders (`plainDecodeD`, `zkDecodeD`) reading each opened position once: `K = 2^256 > 0` (`cap_pos`), the
tables' errors are nonnegative (`εtable_nonneg`), and the decoders meet `ListComplete` (`listComplete_plain`,
`listComplete_zk`). What stays a hypothesis: one wait's bound (`Flock.OneWait`), and `k ≥ 8` wherever the verifier
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

/-- **The deployed plain verifier: KS from one wait's bound**, at `E^rep` with the plain decoder reading each opened
position once. -/
theorem plain_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, 8 ≤ (deployPlain rel).k x ∨ Claims.Flock.NoAcc H (deployPlain rel).V x)
    (h : Claims.Flock.OneWait H (deployPlain rel)) :
    Claims.KS H (deployPlain rel).V (Claims.Flock.R H (deployPlain rel).parse)
      (Claims.Flock.E H (deployPlain rel)
        (Claims.Flock.plainDecodeD Claims.Flock.Numbers.agree₀ ∘ Claims.Flock.dedupPos))
      (Claims.Flock.κ (deployPlain rel)) (Claims.Flock.γ (deployPlain rel)) :=
  Claims.Flock.KS_of_oneWait_dedup H (deployPlain rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg false x) h _
    (listComplete_plain rel)

/-- **The deployed masked (`--zk`) verifier: KS from one wait's bound**, at `E^rep` with the masked decoder reading
each opened position once. -/
theorem zk_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, 8 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc H (deployZk rel).V x)
    (h : Claims.Flock.OneWait H (deployZk rel)) :
    Claims.KS H (deployZk rel).V (Claims.Flock.R H (deployZk rel).parse)
      (Claims.Flock.E H (deployZk rel) (Claims.Flock.zkDecodeD Claims.Flock.zkAgree ∘ Claims.Flock.dedupPos))
      (Claims.Flock.κ (deployZk rel)) (Claims.Flock.γ (deployZk rel)) :=
  Claims.Flock.KS_of_oneWait_dedup H (deployZk rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg true x) h _
    (listComplete_zk rel)

end FlockDeploy.KS
