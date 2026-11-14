module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# The last step's law is over the run of record's own second stage's cells and counts

The two-stage form's last step holds the interior audit's sample to a sampling design it computes from its parameters
(`secondDesignOf`): the replay block's cells under the interior's query, at the rule's counts. The run of record's second
stage samples over the cells and counts it computes from its input's interior. The check builds that input from the same
parameters (`inputOf`), so the two are one:

* `inputOf_first`: the input's rule, boundary parameters, record, closure and bytes are the form's rule, boundary
  parameters, the commitment's record, none, and the first stage's bytes.
* `inputOf_interior`: the input holds an interior whose parameters, replay block and bytes are the form's interior
  parameters, its replay block, and the second stage's bytes.
* `secondLawOf_ok`: `secondDesignOf` returns `Select.secondLaw` of the proved units of the run's `draw`, over the cells
  the replay block's program has under the interior's query, at the rule's counts.
* `accepted_law_run`: where the two-stage check accepts, the sampling design of its last step is `Select.secondLaw` of the proved
  units of the run of record's `draw`, over the cells of that run's interior's replay block under its query, at the
  counts of that run's rule.

Nothing imports this module but `FirstSample`.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The run of record's first-stage input is the form's**: its rule, boundary parameters, record, closure and
bytes. -/
theorem inputOf_first (p : Params) (c : OneStage.Commitment) (ξ ξ₂ : ByteArray) (e : Evidence)
    (received₂ : String) (vs : Array (String × Json)) :
    (inputOf p c ξ ξ₂ e received₂ vs).rule = p.rule ∧ (inputOf p c ξ ξ₂ e received₂ vs).params = p.boundary ∧
      (inputOf p c ξ ξ₂ e received₂ vs).record = c.record ∧ (inputOf p c ξ ξ₂ e received₂ vs).closure = none ∧
      (inputOf p c ξ ξ₂ e received₂ vs).source = ξ := by
  first
    | exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    | (simp only [inputOf]
       generalize rootsSplit c.record e.interior.base.sessionRoots = rs
       obtain ⟨a, b⟩ := rs
       exact ⟨rfl, rfl, rfl, rfl, rfl⟩)
    | (unfold inputOf
       split
       exact ⟨rfl, rfl, rfl, rfl, rfl⟩)
    | (unfold inputOf
       cases rootsSplit c.record e.interior.base.sessionRoots
       exact ⟨rfl, rfl, rfl, rfl, rfl⟩)

/-- **The run of record's interior is the form's**: its parameters, replay block and bytes. -/
theorem inputOf_interior (p : Params) (c : OneStage.Commitment) (ξ ξ₂ : ByteArray) (e : Evidence)
    (received₂ : String) (vs : Array (String × Json)) :
    ∃ it, (inputOf p c ξ ξ₂ e received₂ vs).interior = some it ∧ it.params = p.interior ∧ it.replay = p.replay ∧
      it.source = ξ₂ := by
  first
    | exact ⟨_, rfl, rfl, rfl, rfl⟩
    | (simp only [inputOf]
       generalize rootsSplit c.record e.interior.base.sessionRoots = rs
       obtain ⟨a, b⟩ := rs
       exact ⟨_, rfl, rfl, rfl, rfl⟩)
    | (unfold inputOf
       split
       exact ⟨_, rfl, rfl, rfl, rfl⟩)
    | (unfold inputOf
       cases rootsSplit c.record e.interior.base.sessionRoots
       exact ⟨_, rfl, rfl, rfl, rfl⟩)

/-- **What `secondDesignOf` computed** where it returned: under a two-stage rule, `Select.secondLaw` of the proved units
of the run's `draw`, over the replay block's cells under the interior's query, at the rule's counts over them. -/
theorem secondLawOf_ok {p : Params} {run law : Json} (h : secondDesignOf p run = .ok law) :
    ∃ grain replay k ns ids ks, p.rule = .twoStage grain replay k ∧
      Select.replayCells p.replay p.interior.verifier.query = .ok ns ∧
      Select.templateIds p.interior.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      law = Select.secondLaw ((Draw.proved (getD run "draw")).map Int.toNat) ns ks := by
  unfold secondDesignOf at h
  split at h
  · try dsimp only at h
    obtain ⟨ns, hns, h⟩ := bind_ok h
    obtain ⟨ids, hids, h⟩ := bind_ok h
    obtain ⟨ks, hks, h⟩ := bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨_, _, _, ns, ids, ks, by assumption, hns, hids, hks, h.symm⟩
  · cases h

/-- **The last step's law, at the run of record's own second stage**: where the two-stage check accepts, the run of
record `run` of the input `i` names `accept`; `i` holds an interior `it` on the second stage's bytes `ξ₂` under a
two-stage rule; and the sampling design the last step holds the interior audit's sample to is `Select.secondLaw` of the proved
units of `run`'s `draw`, over the cells of `it`'s replay block under `it`'s query, at the rule's counts over them:
the cells and counts the run's own second stage samples over. -/
theorem accepted_law_run {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i run law it grain replay k ns ids ks,
      inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ Spot.Run.run i true = .ok run ∧
      Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧
      secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂ { e.interior.base with verdicts := r.out } =
        .accept pub prof ∧
      i.interior = some it ∧ it.source = ξ₂ ∧ i.closure = none ∧ i.source = ξ ∧
      i.rule = .twoStage grain replay k ∧
      Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      law = Select.secondLaw ((Draw.proved (Spot.getD run "draw")).map Int.toNat) ns ks := by
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, hr, -, -, hj, law, hl, htwo⟩ := accepted_steps h
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, hio, -, -⟩ := inputAt_ok hi
  obtain ⟨grain, replay, k, ns, ids, ks, hk, hns, hids, hks, hlaw⟩ := secondLawOf_ok hl
  obtain ⟨it, hit, hp, hrep, hsrc⟩ :=
    inputOf_interior p { c with record := j₁ } ξ ξ₂ { e with record := j₂ } received₂ vs
  obtain ⟨h1, -, -, h4, h5⟩ := inputOf_first p { c with record := j₁ } ξ ξ₂ { e with record := j₂ } received₂ vs
  refine ⟨c₂, ξ₂, r, i, run, law, it, grain, replay, k, ns, ids, ks, hi, hr, hj, htwo, ?_, hsrc, ?_, ?_, ?_, ?_, ?_,
    hks, hlaw⟩
  · rw [hio]
    exact hit
  · rw [hio]
    exact h4
  · rw [hio]
    exact h5
  · rw [hio, h1]
    exact hk
  · rw [hrep, hp]
    exact hns
  · rw [hp]
    exact hids

end VStar.Recursive.SampledTwo
