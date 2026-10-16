import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.DeployInputs

/-!
# The deployed relation at the deployed parser

The deployed record's parser is `FlockDeploy.parse` (`parseOf` at the verifier's reading of the bytes,
`FlockDeploy.readOf`), and `ScopeMatches` at both deployed records is `FlockDeploy.scopeMatches_zk` and
`FlockDeploy.scopeMatches_plain_iff` (`Matches`' `scopeMatches_zk` and `scopeMatches_iff`). At the deployed relation
(`relation`):

* `relS_deployed`: `RelS` of a statement the deployed parser accepts, untyped and without public input ports, is every
  table's `Layout.model` satisfied (`Fit.relS_layout`).
-/

namespace Flock.Readable

/-- **The deployed relation at the deployed parser**: a statement it parses is the readable form of a statement the
verifier parses, and when that statement is untyped and has no public input ports, `RelS` is every table's
`Layout.model` at `regionsOf` satisfied by its share of the hidden values. -/
theorem relS_deployed (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : FlockDeploy.parse zk relation x = some s) (ht : s.form.typed = false) (hpi : s.form.publicInputs = false)
    (w : Claims.Flock.Wit) :
    ∃ p : Parsed, parseBytes (FlockDeploy.readOf zk) x = some p ∧ s = toReadable relation p ∧
      (Claims.Flock.RelS H s w ↔ ∀ c (hc : c < p.sts.size), ∃ L : FlockSoundness.Layout p.sts[c],
        (L.model (FlockSoundness.Refine.regionsOf p.sts[c])).Satisfies (zOf (2 ^ s.m) w c p.sts[c].m)) := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simpa [toReadable, Read.form] using hpi)
  exact ⟨p, hp, rfl, relS_layout H (parses_of_parse hrp) ht hpi' w⟩

end Flock.Readable
