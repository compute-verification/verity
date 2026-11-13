module

public import Proofs.Flock.SpotFlock.RunsAs.Top
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The run's `draw` is the first stage's sample, and the second stage is sampled over it

At a two-stage run `Run.run` accepts, the answer's `draw` is the sample object the first stage's select returned
(`answer_draw`), the units it proves are the select's units, and the second stage's `selectSecond` is handed exactly
those units, at the replay block's cells and the rule's counts (`firstSample`). `selectSecond` samples under the sampling design
`Select.secondLaw` computes from them (`selectSecond_law`). So a second-stage law computed from the answer's `draw` is
the sampling design the run's second stage drew under.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)
open Properties.Spot.SecurityProofs (bind_eq_ok throw_bind_ne)

/-- **The first stage's sample, as the run's answer holds it.** At a two-stage run `Run.run` accepts, with the form
allowed or not: the first stage's select, with no closure map, returned the answer's `draw` with the units `rest`; the
units `draw` proves are `rest`; and the second stage's `selectSecond` ran over those units, at the replay block's cells
`ns` and the rule's counts `ks` over the check's templates. -/
theorem firstSample {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    ∃ (grain : Json) (replay : String) (k : Spot.Select.Counts) (resolved : Spot.Select.Rule)
      (workTable : Option Json) (rest : Array ℕ) (read : ℕ) (ns : Array ℕ) (ids : Array String) (ks : Array ℕ)
      (draw₂ : Json) (read₂ : ℕ),
      i.rule = .twoStage grain replay k ∧ i.closure = none ∧
      Spot.Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable =
        .ok workTable ∧
      Spot.Select.select resolved i.params.verifier.program i.params.verifier.query workTable none i.source =
        .ok (Spot.getD j "draw", rest, read) ∧
      (Spot.Draw.proved (Spot.getD j "draw")).map Int.toNat = rest ∧
      Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Spot.Select.templateIds it.params.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
      Spot.Select.selectSecond ((Spot.Draw.proved (Spot.getD j "draw")).map Int.toNat) ns ks it.source =
        .ok (draw₂, read₂) := by
  obtain ⟨hcl, workTable, resolved, draw, rest, read, ht, hres, hsel, -, grain, replay, k, ns, ids, ks, draw₂,
    read₂, -, hrule, hns, hids, hks, hss, -, -, -⟩ := accepted_steps ha
  obtain ⟨workTable', resolved', rest', read', ht', hres', hsel'⟩ := answer_draw ha
  have hwt : workTable' = workTable := Except.ok.inj (ht'.symm.trans ht)
  have hrs : resolved' = resolved := Except.ok.inj (hres'.symm.trans hres)
  subst hwt hrs
  have hd : Spot.getD j "draw" = draw := congrArg Prod.fst (Except.ok.inj (hsel'.symm.trans hsel))
  rw [hcl] at hsel
  rw [hd]
  exact ⟨_, _, _, _, _, _, _, _, _, _, _, _, hrule, hcl, hres, ht, hsel,
    proved_select (resolve_ne_all hrule hres) hsel, hns, hids, hks, hss⟩

/-- **The second stage samples under `secondLaw`**: where `selectSecond` accepts a replay sample, its sample and the bytes
it read are `drawFirstRead`'s under `Select.secondLaw` of that sample, the cells and the counts, over the sample's
cells. -/
theorem selectSecond_law {sample ns ks : Array ℕ} {src : ByteArray} {d : Json} {r : ℕ}
    (h : Spot.Select.selectSecond sample ns ks src = .ok (d, r)) :
    Spot.Draw.drawFirstRead (Spot.Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) src =
      .ok (d, r) := by
  unfold Spot.Select.selectSecond at h
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e =>
    try rw [hq] at h
    dsimp only at h
    exact absurd h (throw_bind_ne _ _ _)
  | none =>
    try rw [hq] at h
    dsimp only at h
    obtain ⟨dr, hd, h⟩ := bind_eq_ok.1 h
    obtain ⟨d', r'⟩ := dr
    have he : (d', r') = (d, r) := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact Except.ok.inj h
    rw [hd, he]

end SpotFlock.RunsAs
