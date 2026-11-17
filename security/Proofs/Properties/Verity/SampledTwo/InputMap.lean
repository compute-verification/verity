module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-! The last step's sampling design is over the run of record's own second stage's cells and counts. -/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

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

theorem secondSamplingDesignOf_ok {p : Params} {run samplingDesign : Json} (h : secondDesignOf p run = .ok samplingDesign) :
    ∃ grain replay k ns ids ks, p.rule = .twoStage grain replay k ∧
      Select.replayCells p.replay p.interior.verifier.query = .ok ns ∧
      Select.templateIds p.interior.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      samplingDesign = Select.secondSamplingDesign ((Draw.proved (getD run "draw")).map Int.toNat) ns ks := by
  unfold secondDesignOf at h
  split at h
  · try dsimp only at h
    obtain ⟨ns, hns, h⟩ := bind_ok h
    obtain ⟨ids, hids, h⟩ := bind_ok h
    obtain ⟨ks, hks, h⟩ := bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨_, _, _, ns, ids, ks, by assumption, hns, hids, hks, h.symm⟩
  · cases h

theorem accepted_samplingDesign_run {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i run samplingDesign it grain replay k ns ids ks,
      inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ Spot.Run.run i true = .ok run ∧
      Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧
      secondCheck p.audit.base.verifier p.audit.base.delta c₂ samplingDesign ξ₂ { e.interior.base with verdicts := r.out } =
        .accept pub prof ∧
      i.interior = some it ∧ it.source = ξ₂ ∧ i.closure = none ∧ i.source = ξ ∧
      i.rule = .twoStage grain replay k ∧
      Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      samplingDesign = Select.secondSamplingDesign ((Draw.proved (Spot.getD run "draw")).map Int.toNat) ns ks := by
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, hr, -, -, hj, samplingDesign, hl, htwo⟩ := accepted_steps h
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, hio, -, -⟩ := inputAt_ok hi
  obtain ⟨grain, replay, k, ns, ids, ks, hk, hns, hids, hks, hlaw⟩ := secondSamplingDesignOf_ok hl
  obtain ⟨it, hit, hp, hrep, hsrc⟩ :=
    inputOf_interior p { c with record := j₁ } ξ ξ₂
      { e with record := j₂, stmts := VStar.Recursive.SampledTwo.stmtsOwn p e bs } received₂ vs
  obtain ⟨h1, -, -, h4, h5⟩ := inputOf_first p { c with record := j₁ } ξ ξ₂
    { e with record := j₂, stmts := VStar.Recursive.SampledTwo.stmtsOwn p e bs } received₂ vs
  refine ⟨c₂, ξ₂, r, i, run, samplingDesign, it, grain, replay, k, ns, ids, ks, hi, hr, hj, htwo, ?_, hsrc, ?_, ?_, ?_, ?_, ?_,
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
