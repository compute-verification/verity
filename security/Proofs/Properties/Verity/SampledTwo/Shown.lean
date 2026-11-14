module

public import Proofs.Properties.Verity.SampledTwo.Ordered
public import Proofs.Properties.Verity.SampledTwo.Deployed

/-!
# What the two-stage form's refusals show of an accepted play

* `evidenceOf_coins`: the form's evidence parses only with the first stage's coin evidence not `null`
  (`FIRST_COINS_NULL`).
* `paramsOfJson_apart`, `paramsOfJson_families`: the form's parameters parse only with the two stages' coin families,
  as the check reads them (`inputOf`), named and apart (`SAME_FAMILY`).
* `accepted_order`: where the check accepts on two windows, the first stage's session ends before the second stage's
  is reserved, on one ledger (`accepted_ordered`, `stagesOrdered_order`).
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

/-- **The form's evidence names the first stage's coin evidence**: `evidenceOf` refuses it `null`. -/
theorem evidenceOf_coins {b : ByteArray} {e : Evidence} (h : evidenceOf b = .ok e) : e.coins ≠ Json.null := by
  replace h := (evidenceOf_ok h).1
  unfold evidenceParsed at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
  all_goals (simp only [pure, Except.pure, Except.ok.injEq] at h; subst h; intro hc; simp_all)

/-- **The form's parameters name the two stages' coin families apart** (`familiesApart`, `SAME_FAMILY`). -/
theorem paramsOfJson_apart {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    (!familiesApart (p.boundary.challenger.map (·.2)) (p.audit.base.challenger.map (·.2))) ≠ true :=
  (paramsOfJson_audit h).2

/-- **The first stage's coin family is named, and the second stage's is another**. -/
theorem paramsOfJson_families {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    ∃ l, p.boundary.challenger.map (·.2) = some l ∧ p.audit.base.challenger.map (·.2) ≠ some l := by
  have hb := paramsOfJson_apart h
  revert hb
  generalize p.boundary.challenger.map (·.2) = b
  generalize p.audit.base.challenger.map (·.2) = i
  intro hb
  cases b with
  | none => simp [familiesApart] at hb
  | some l => exact ⟨l, rfl, fun hi => by subst hi; simp [familiesApart] at hb⟩

/-- **An accepted check on two windows found the stages' sessions in order on one ledger** (`stagesOrdered_order` at
the accepted check's `stagesOrdered`). -/
theorem accepted_order {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof)
    {k : Spot.Challenger.ChallengerKey} {label w₁ w₂ : String} (hk : p.audit.base.challenger = some (k, label))
    (h1 : p.boundary.verifier.windowRoot = some w₁) (h2 : p.audit.base.verifier.windowRoot = some w₂)
    (hne : w₁ ≠ w₂) :
    ∃ first second, Spot.getD e.coins "ledger" = .arr first ∧
      Spot.getD e.interior.base.coins "ledger" = .arr second ∧ Spot.Challenger.extendsHead first second = true ∧
      ∃ wins ms, Spot.Challenger.checked (.arr second) k = .ok (wins, ms) ∧
        ∃ en r, (wins.find? (·.root == w₁)).bind (·.ended) = some en ∧
          ((wins.find? (·.root == w₂)).bind fun w => (second.filterMap Spot.Challenger.ledgerEntry?).find? fun l =>
            l.kind == "reserve" && l.session == w.opened.session) = some r ∧ en.seq < r.seq :=
  stagesOrdered_order (accepted_ordered h) hk h1 h2 hne

end VStar.Recursive.SampledTwo
