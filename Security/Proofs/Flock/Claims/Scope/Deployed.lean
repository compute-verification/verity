import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.DeployInputs

/-!
# `Claims.Flock.ScopeMatches` at the deployed parser and scope check

The deployed record's parser is `FlockDeploy.parse` (the statement's bytes decoded, the verifier's own parsers run on
them, `Read.parse`, `toReadable`) and its scope check `FlockDeploy.scopeCheck` (that parser accepts and the verifier's
own `ProvedScope.check` passes). Both are `Matches`' `parseOf` and `scopeCheckOf` at the decoder `readOf`
(`parse_eq`, `scopeCheck_eq`), whose reading is the masked verifier's exactly when the statement says `--zk`
(`readOf_zk`). So:

* `scopeMatches_deployed_iff`: `ScopeMatches` holds exactly when no statement the deployed parser accepts and
  `ProvedScope.check` passes has `m < 22`;
* `scopeMatches_deployed_zk`: under `--zk` it holds, at every relation, the deployed one (`relation`) among them;
* `relS_deployed`: at the deployed relation, `RelS` of a statement the deployed parser accepts, untyped and without
  public input ports, is every table's `Layout.model` satisfied (`Fit.relS_layout`).
-/

namespace Flock.Readable

/-- What the verifier reads of a statement's bytes, for the verifier masked exactly when `zk`. -/
def readOf (zk : Bool) (x : Claims.Msg) : Option Read := (FlockDeploy.insOf zk x).map (·.read)

theorem parse_eq (zk : Bool) (rel : Parsed → Relation) : FlockDeploy.parse zk rel = parseOf rel (readOf zk) := by
  funext x
  unfold FlockDeploy.parse parseOf parseBytes readOf
  cases FlockDeploy.insOf zk x with
  | none => rfl
  | some s =>
    show Option.bind s.read.parse (fun p => some (toReadable rel p)) = Option.map (toReadable rel) s.read.parse
    cases s.read.parse <;> rfl

theorem scopeCheck_eq (zk : Bool) : FlockDeploy.scopeCheck zk = scopeCheckOf (readOf zk) := by
  funext x
  simp only [FlockDeploy.scopeCheck, scopeCheckOf, parseBytes, readOf]
  cases FlockDeploy.insOf zk x with
  | none => rfl
  | some s =>
    simp only [Option.map_some, Option.bind_some]
    cases hp : s.read.parse with
    | none => rfl
    | some p =>
      rw [← toRead_of_parse hp]
      rfl

theorem ins_zk {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) : s.read.zk = w.zk := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals (cases h <;> rfl)

/-- The deployed decoder's reading is masked exactly when its verifier is. -/
theorem readOf_zk {zk : Bool} {x : Claims.Msg} {r : Read} (h : readOf zk x = some r) : r.zk = zk := by
  unfold readOf FlockDeploy.insOf at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨w, -, hs⟩ := Option.bind_eq_some_iff.1 hs
  by_cases hz : w.zk = zk
  · simp only [hz, ne_eq, not_true_eq_false, ↓reduceIte] at hs
    cases hi : w.ins with
    | error e => rw [hi] at hs; cases hs
    | ok s' =>
      rw [hi] at hs
      cases hs
      rw [ins_zk hi, hz]
  · simp only [hz, ne_eq, not_false_eq_true, ↓reduceIte] at hs
    cases hs

variable {Coin Col T : Type}

/-- **`ScopeMatches` at the deployed parser and scope check**, exactly when every statement the deployed parser accepts
and `ProvedScope.check` passes has `22 ≤ m`. -/
theorem scopeMatches_deployed_iff (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (hparse : D.parse = FlockDeploy.parse D.zk rel) (hcheck : D.scopeCheck = FlockDeploy.scopeCheck D.zk) :
    Claims.Flock.ScopeMatches D ↔
      ∀ x p, parseBytes (readOf D.zk) x = some p → p.check = .ok () → 22 ≤ (toReadable rel p).m :=
  scopeMatches_iff D rel (readOf D.zk) (hparse.trans (parse_eq _ _)) (hcheck.trans (scopeCheck_eq _))
    fun _ _ h => readOf_zk h

/-- **`ScopeMatches` holds at the deployed `--zk` verifier**: its parser and scope check, at any relation. -/
theorem scopeMatches_deployed_zk (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (hparse : D.parse = FlockDeploy.parse D.zk rel) (hcheck : D.scopeCheck = FlockDeploy.scopeCheck D.zk)
    (hz : D.zk = true) : Claims.Flock.ScopeMatches D :=
  scopeMatches_zk D rel (readOf D.zk) (hparse.trans (parse_eq _ _)) (hcheck.trans (scopeCheck_eq _))
    (fun _ _ h => readOf_zk h) hz

/-- **The deployed relation at the deployed parser**: a statement it parses is the readable form of a statement the
verifier parses, and when that statement is untyped and has no public input ports, `RelS` is every table's
`Layout.model` at `regionsOf` satisfied by its share of the hidden values. -/
theorem relS_deployed (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : FlockDeploy.parse zk relation x = some s) (ht : s.form.typed = false) (hpi : s.form.publicInputs = false)
    (w : Claims.Flock.Wit) :
    ∃ p : Parsed, parseBytes (readOf zk) x = some p ∧ s = toReadable relation p ∧
      (Claims.Flock.RelS H s w ↔ ∀ c (hc : c < p.sts.size), ∃ L : FlockSoundness.Layout p.sts[c],
        (L.model (FlockSoundness.Refine.regionsOf p.sts[c])).Satisfies (zOf (2 ^ s.m) w c p.sts[c].m)) := by
  rw [parse_eq] at hs
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simpa [toReadable, Read.form] using hpi)
  exact ⟨p, hp, rfl, relS_layout H (parses_of_parse hrp) ht hpi' w⟩

end Flock.Readable
