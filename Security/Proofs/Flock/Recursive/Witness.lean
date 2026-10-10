import Flock.Witness
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Recursive.Description
import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements

/-!
# `flock-witness`'s checks are the recursive audit's predicates

`Flock.Witness` evaluates, on recorded files, the predicates `RecursiveAudit`'s event and premises state. Each check
passing is the model's predicate holding, at `toExec I`, the model's inputs of the witness's:

* `acceptsZKR_ok`: `ZkReg.AcceptsZKR`, the verifier of record's verdict on one of V*'s sessions;
* `scopeZR_ok`, `scopeZ_ok`: `FailClosed.ScopeZR`, the event's statement check, and `FailClosed.ScopeZ`, `Typable`'s;
* `tagsOk_ok`: `DecodeZL.TagsOk`, `ZkOuterR.hTags`;
* `scopeOk_ok`: the session's circuit as `ZkOuterR` parses it, and `Layout.scopeOk` of it, `VStmt.hscope`;
* `fullDraw_ok`: the record's draw is the full `subset` draw of every unit of the population, `RecordCustody`'s;
* `acceptsStmt_ok`: `Recursion.AcceptsStmt`;
* `acceptsDescription_ok`: `Recursion.AcceptsDescription`, at each of the verifier's own registered reads.
-/

namespace FlockSoundness.Discharge.Recursion.Witness

open Flock
open FlockSoundness.Discharge

/-- The model's inputs of the witness's (`Exec.Inputs`): no work, stratified or closure inputs, and the coins `--zk`
sets, which `ZkExec.zkInputs` sets again. -/
def toExec (I : Flock.Witness.Inputs) : Exec.Inputs where
  tags := I.tags
  circuitFile := I.circuitFile
  publicFile := I.publicFile
  coins := Flock.Witness.zkCoins I
  count := I.count
  tables := I.tables
  partition := I.partition
  program := I.program
  work := none
  stratified := none
  closure := none

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem setupOf_eq (I : Flock.Witness.Inputs) (r : String) :
    Flock.Witness.setupOf I r = Exec.setupOf (ZkExec.zkInputs (toExec I)) r := rfl

/-- **The witness's verdict is the verifier of record's** (`ZkReg.AcceptsZKR`). -/
theorem acceptsZKR_ok {I : Flock.Witness.Inputs} {own : Array Flock.Registered.Port} {r : String} {p : Array ByteArray}
    (h : Flock.Witness.acceptsZKRAt (Flock.Witness.setupOf I r) own r p = .ok ()) :
    ZkReg.AcceptsZKR (toExec I) own ⟨r, p⟩ := by
  unfold Flock.Witness.acceptsZKRAt at h
  obtain ⟨⟨spec, sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨_, h2, h⟩ := bind_ok h
  obtain ⟨_, h3, h4⟩ := bind_ok h
  exact ⟨spec, sts, (setupOf_eq I r).symm.trans h1, h2, h3, h4⟩

/-- **The witness's statement check with the registered reads is `ScopeZR`.** -/
theorem scopeZR_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.scope I true = .ok ()) :
    FailClosed.ScopeZR (toExec I) := h

/-- **The witness's statement check without them is `ScopeZ`.** -/
theorem scopeZ_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.scope I false = .ok ()) :
    FailClosed.ScopeZ (toExec I) := h

/-- **The witness's tags check is `TagsOk`.** -/
theorem tagsOk_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.tagsOk I = .ok ()) : DecodeZL.TagsOk (toExec I) := by
  unfold Flock.Witness.tagsOk at h
  by_cases hc : I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧
      I.count ≤ 1
  · exact hc
  · rw [if_neg hc] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

/-- **The witness's circuit is the session's as `ZkOuterR` parses it, and `Layout.scopeOk` accepts it.** -/
theorem scopeOk_ok {I : Flock.Witness.Inputs} {c : Flock.Circuit} (h : Flock.Witness.scopeOk I = .ok c) :
    Flock.HmRow.parse (ZkExec.zkTags (toExec I)) (toExec I).circuitFile (toExec I).tables = .ok c ∧
      Layout.scopeOk c = true := by
  unfold Flock.Witness.scopeOk at h
  obtain ⟨c', h1, h⟩ := bind_ok h
  by_cases hs : Flock.ProvedScope.scopeOk c' = true
  · rw [if_pos hs] at h
    have hc : c' = c := Except.ok.inj h
    refine ⟨by rw [← hc]; exact h1, ?_⟩
    rw [← hc, ← FailClosed.scopeOk_eq]
    exact hs
  · rw [if_neg hs] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

/-- **The witness's draw check**: the record's draw decodes to the full `subset n` draw over `[0, n)`, every unit and
no closure. -/
theorem fullDraw_ok {n : ℕ} {r : String} (h : Flock.Witness.fullDraw n r = .ok ()) :
    ∃ d D, Exec.recordDraw r = some d ∧ Flock.Draw.UnitDraw.ofJson d = .ok D ∧ D.law = .subset n ∧ D.population = n ∧
      D.units = Array.range n ∧ D.closure = #[] := by
  unfold Flock.Witness.fullDraw at h
  cases hd : Flock.Witness.recordDraw r with
  | none =>
    rw [hd] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h
  | some d =>
    rw [hd] at h
    dsimp only at h
    obtain ⟨D, hD, h⟩ := bind_ok h
    cases hl : D.law with
    | subset k =>
      rw [hl] at h
      dsimp only at h
      by_cases hc : k = n ∧ D.population = n ∧ D.units = Array.range n ∧ D.closure = #[]
      · obtain ⟨rfl, h2, h3, h4⟩ := hc
        exact ⟨d, D, hd, hD, hl, h2, h3, h4⟩
      · rw [if_neg hc] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
    | _ =>
      rw [hl] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h

theorem one_ok {sts : Array Flock.Stmt} {st : Flock.Stmt} (h : Flock.Witness.one sts = .ok st) : sts = #[st] := by
  unfold Flock.Witness.one at h
  split at h
  · rename_i x hx
    cases h
    exact Array.ext' (by simp [hx])
  · cases h

/-- **The witness's inner check is `AcceptsStmt`.** -/
theorem acceptsStmt_ok {cf pf : ByteArray} {p : Option Lean.Json} {g : Option ByteArray} {st : Flock.Stmt}
    (h : Flock.Witness.acceptsStmt cf pf p g = .ok st) : AcceptsStmt st := by
  unfold Flock.Witness.acceptsStmt Flock.Witness.acceptsStmtAt Flock.Witness.innerSetup at h
  obtain ⟨⟨spec, sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨st', h2, h⟩ := bind_ok h
  obtain ⟨q, h3, h⟩ := bind_ok h
  have hst : st' = st := Except.ok.inj h
  rw [← hst]
  have h2' : Flock.Witness.one sts = .ok st' := h2
  exact ⟨⟨cf, pf, p, g, spec, by rw [h1, one_ok h2']⟩, q, h3⟩

/-- **The witness's description check is `AcceptsDescription` at each of the verifier's own registered reads.** -/
theorem acceptsDescription_ok {st : Flock.Stmt} {own : Array Flock.Registered.Port}
    (h : Flock.Witness.acceptsDescription st own = .ok ()) : ∀ rt ∈ own, AcceptsDescription rt st := by
  unfold Flock.Witness.acceptsDescription at h
  obtain ⟨_, h1, _⟩ := bind_ok h
  exact fun rt hrt => ⟨own, hrt, h1⟩

end FlockSoundness.Discharge.Recursion.Witness
