import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's KS at the deployed instances, from one wait's bound

`Flock.KS_of_oneWait` at the deployed records `FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel`, their
columns' zero the empty column (`instZeroListCol`): `K = 2^256 > 0` (`cap_pos`) and the tables' errors are nonnegative
(`εtable_nonneg`). What stays a hypothesis: one wait's bound (`Flock.OneWait`), and `k ≥ 8` and the relation splitting
by table wherever the verifier accepts some run. At an in-scope statement the relation splits (`splits_of_inScope`, from
`relS_split`).
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

/-- **The deployed relation splits by table** at an in-scope statement, for a record that reads the claims' parser,
its tables' own statements, and the tables' traces interleaved (`relS_split`). -/
theorem splits_of_inScope {Coin Col Tr : Type} {zk : Bool} (r : Tr → Claims.Flock.Wit)
    (D : Claims.Flock.Deploy Coin Col Tr) (hp : D.parse = parse zk Readable.relation)
    (ht : D.table = table zk Readable.relation) (hr : D.read = Claims.Flock.readTables r) {x : Claims.Msg}
    {s : Claims.Flock.Stmt} (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) :
    Claims.Flock.Splits D x := by
  intro s' hs' Ms hlen hall
  rw [hp, hs, Option.some.injEq] at hs'
  subst hs'
  rw [hr, relS_split hs hin]
  intro j hj
  obtain ⟨t, ht', hR⟩ := hall j (by omega)
  rw [ht] at ht'
  rw [hr] at hR
  refine ⟨t, ht', (Readable.relS_congr t fun i => ?_).2 hR⟩
  rw [Readable.tableWit_getD hj, ← hlen, Claims.Flock.readTables_getD r Ms (by omega),
    Claims.Flock.readTables_singleton]

/-- **The deployed plain verifier: KS from one wait's bound.** -/
theorem plain_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, (8 ≤ (deployPlain rel).k x ∧ Claims.Flock.Splits (deployPlain rel) x) ∨
      Claims.Flock.NoAcc H (deployPlain rel).V x)
    (h : Claims.Flock.OneWait H (deployPlain rel)) : Claims.Flock.KS H (deployPlain rel) :=
  Claims.Flock.KS_of_oneWait H (deployPlain rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg false x) h

/-- **The deployed masked (`--zk`) verifier: KS from one wait's bound.** -/
theorem zk_of_oneWait (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hk : ∀ x, (8 ≤ (deployZk rel).k x ∧ Claims.Flock.Splits (deployZk rel) x) ∨
      Claims.Flock.NoAcc H (deployZk rel).V x)
    (h : Claims.Flock.OneWait H (deployZk rel)) : Claims.Flock.KS H (deployZk rel) :=
  Claims.Flock.KS_of_oneWait H (deployZk rel) (fun _ => cap_pos) hk (fun x => εtable_nonneg true x) h

end FlockDeploy.KS
