module

public import Proofs.Properties.Verity.SampledTwo.CutRoots
public import Proofs.Properties.Verity.SampledTwo.FirstCoins
public import Proofs.Properties.Verity.SampledTwo.SecondCoins
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.CoinStages

/-!
# Each stage's sample is its family after one root, from `OtherRoots`

`CoinMap.StageDraws` asks, of each accepted play and its stage's sample, two things: the sample names the stage's family,
and the play's log shows that family after the sample's digest alone (`CoinMap.OneRoot`). The second is CUT's
(`CutRoots.lean`), on a play whose log the check read and found the device's (`ReadIsDevice`), whose sent message is
the case's (`caseOf o.sent`) and that meets `OtherRoots`: `oneRoot_draws` for the first stage, whose sample is the cut's
first sample (`firstDraw_of`, the boundary parameters' family), and `oneRoot_second` for the second. So `StageDraws`
takes the family alone (`stageDraws_first`, `stageDraws_second`).
-/

@[expose] public section

namespace Verity.SampledTwo

/-- **The second stage's sample, read back**: the family the interior audit's parameters name for its challenger and the
interior registration's digest. -/
theorem secondDraw_parts {k : Case} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} (hd : Decoded k p c e) {label : String} {after : ByteArray}
    (h : secondDraw k = some (label, after)) :
    ∃ key c₂, p.audit.base.challenger = some (key, label) ∧ interiorOf e = .ok c₂ ∧ after = c₂.digest := by
  obtain ⟨hp, -, he⟩ := hd
  unfold secondDraw at h
  rw [hp, he] at h
  rcases hch : p.audit.base.challenger with _ | ⟨key, label'⟩
  · simp [hch, bind, Except.bind, Except.toOption] at h
  · cases hs : Flock.canon e.record with
    | error err => simp [hs, bind, Except.bind, Except.toOption] at h
    | ok s =>
      cases hc : Spot.OneStage.commitmentOf s.toUTF8 with
      | error err =>
        have hc' : Spot.OneStage.commitmentOf s.toByteArray = .error err := by
          first
            | exact hc
            | simpa using hc
        simp [hs, hc', bind, Except.bind, Except.toOption] at h
      | ok c₂ =>
        have hc' : Spot.OneStage.commitmentOf s.toByteArray = .ok c₂ := by
          first
            | exact hc
            | simpa using hc
        simp [hch, hs, hc', bind, Except.bind, Except.toOption] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨key, c₂, rfl, ?_, rfl⟩
        unfold interiorOf
        rw [hs]
        exact hc

/-- **The first stage's sample is its family after one root**: `StageDraws` from the family alone, on plays whose log the
check read and found the device's, whose sent message is the case's and that meet `OtherRoots`. -/
theorem stageDraws_first {C : CaseOf} {Acc : Verity.Exchange → Prop} {ℓ : String}
    (hcase : ∀ o k p c e, C o = some k → Decoded k p c e → caseOf o.sent = .ok (p, c, e))
    (hA : ∀ o, Acc o → FormAccepts C o ∧ ReadIsDevice o ∧ OtherRoots o.sent)
    (hfam : ∀ o, Acc o → ∀ label after, (C o).bind firstDraw = some (label, after) → label = ℓ) :
    CoinMap.StageDraws Acc (fun o => (C o).bind firstDraw) ℓ := by
  intro o ho label after hd
  have hℓ := hfam o ho label after hd
  obtain ⟨⟨k, hk, p, c, e, -, -, hdec, -⟩, hg, hx⟩ := hA o ho
  have hd' : firstDraw k = some (label, after) := by
    beta_reduce at hd
    rw [hk] at hd
    exact hd
  obtain ⟨p', c', e', _, d₁, _, _, _, h1, -, h3, -, -, -, ho₁, -⟩ := oneRoot_draws hg hx
  have hpce := Except.ok.inj (h1.symm.trans (hcase o k p c e hk hdec))
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  have hpair := Option.some.inj (hd'.symm.trans (firstDraw_of hdec h3))
  simp only [Prod.mk.injEq] at hpair
  obtain ⟨rfl, rfl⟩ := hpair
  subst hℓ
  exact ⟨rfl, ho₁⟩

/-- **The second stage's sample is its family after one root**: `StageDraws` from the family alone, on the same plays. -/
theorem stageDraws_second {C : CaseOf} {Acc : Verity.Exchange → Prop} {ℓ : String}
    (hcase : ∀ o k p c e, C o = some k → Decoded k p c e → caseOf o.sent = .ok (p, c, e))
    (hA : ∀ o, Acc o → FormAccepts C o ∧ ReadIsDevice o ∧ OtherRoots o.sent)
    (hfam : ∀ o, Acc o → ∀ label after, (C o).bind secondDraw = some (label, after) → label = ℓ) :
    CoinMap.StageDraws Acc (fun o => (C o).bind secondDraw) ℓ := by
  intro o ho label after hd
  have hℓ := hfam o ho label after hd
  obtain ⟨⟨k, hk, p, c, e, -, -, hdec, -⟩, hg, hx⟩ := hA o ho
  have hd' : secondDraw k = some (label, after) := by
    beta_reduce at hd
    rw [hk] at hd
    exact hd
  obtain ⟨key, c₂, hch, h5, rfl⟩ := secondDraw_parts hdec hd'
  subst hℓ
  exact ⟨rfl, oneRoot_second hg hx (hcase o k p c e hk hdec) h5 hch⟩

end Verity.SampledTwo
