import Proofs.Flock.Bridge.Rows
import Flock.PortRoots

/-!
# `SampledKSLiveV2`'s file preconditions, refused by the verifiers

`SampledKSLiveV2` takes four things of the files as hypotheses: every port a row/v1, row/v2 or row-seg/v1 row
(`Bridge.RowsOk`, which the scope check `scopeOkH` already gives: `Bridge.rowsOk_of_scopeOkH`), each of its own name
(`Registered.dupName`), no two ports' registered roots one root under different domains (`PortRoots.rootsColl = none`),
and the registration's population the game's `n` units in order (`Bridge.PopMatches`).
The verifiers of record now refuse every file that fails the last three: `verify --zk` runs `Flock.PortRoots.check`
last in its verdict (`Main.verifyCmd`, with or without `--ks-form`, the check's old flag), and the sampled-proofs run
refuses the registration (`SampledProofs.Run.run`,
`population-order`, `SampledProofs.Slots.popOk`). So the guarantee reads them off acceptance.

* `Bridge.LiveAcceptsKS I o`: `verify --zk` accepts: the live audit's transcript (`LiveAcceptsZK`), and the files pass
  `Flock.PortRoots.check`, whose roots are framed at each port's schema, as `Bridge.rootsOf`'s. `Bridge.LiveAcceptsKSR`
  (`RowsReg`) is the same under `verify --zk --registered F`.
* `Bridge.RunPopOk desc qr n`: the descriptor parses and its population under the query is `n` units in order
  (`Slots.popOk`), which the run checks at registration.
* `rowsOk_of_scopeOkH`, `rootsColl_none_of_collides_false`, `popMatches_of_popOk`: the checks give V2's hypotheses. The
  roots check is computed at the public file's population, so the game's `n` is identified with it (`pub.n = n`).
* `SampledKSLiveV3` (`RowsReg`) reads them off acceptance: files that pass give V2's hypotheses, and at files that fail
  its event is empty.
-/

namespace Flock.Bridge

open Properties.NetworkCertifier.Bytes (bs_inj)

/-- **No colliding pair in the verifier's roots fixes each root's domain**: the roots `Flock.PortRoots.collides`
computes, each port's at its schema, are `rootsOf`'s. -/
theorem fixes_of_collides_false {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ}
    (h : Flock.PortRoots.collides c pub nl = false) : PortRoots.RootFixesDomain c pub nl := by
  intro q q' hlt hq he
  rw [PortRoots.rootsOf_getD c pub nl (hlt.trans hq), PortRoots.rootsOf_getD c pub nl hq] at he
  exact Flock.PortRoots.domOf_eq_of_collides_false h hlt hq (bs_inj he)

/-- **The verifier's roots check is V2's port-roots guard.** -/
theorem rootsColl_none_of_collides_false {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ}
    (h : Flock.PortRoots.collides c pub nl = false) : PortRoots.rootsColl c pub nl = none :=
  PortRoots.rootsColl_none_iff.2 (fixes_of_collides_false h)

/-- **The run's population check is `PopMatches`.** -/
theorem popMatches_of_popOk {desc : Claims.Msg} {qr : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (h : SampledProofs.Slots.popOk p qr n = true) :
    PopMatches desc qr n := by
  show popCheck desc qr n = true
  unfold popCheck
  rw [hp]
  exact h

open FlockSoundness.Discharge.ZkExec in
/-- **`verify --zk` accepts, files and transcript**: the live audit's transcript (`LiveAcceptsZK`), and the files' rows,
names and roots (`Flock.PortRoots.check`, the last refusal of `verify --zk`'s verdict). -/
def LiveAcceptsKS {n : ℕ} {Reg : Type} (I : FlockSoundness.Discharge.Exec.Inputs)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsZK I o ∧ Flock.PortRoots.check (zkTags I) I.circuitFile I.publicFile I.tables = .ok ()

/-- **The sampled-proofs run passes the registration's population** (`SampledProofs.Run.run`, `population-order`):
the descriptor `desc` parses and its population under the query `qr` is `n` units in order (`Slots.popOk`). -/
def RunPopOk (desc : Claims.Msg) (qr : Lean.Json) (n : ℕ) : Prop :=
  ∃ p, SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p ∧ SampledProofs.Slots.popOk p qr n = true

end Flock.Bridge
