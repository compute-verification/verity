module

public import Proofs.Properties.Verity.SampledTwo.StageLog
public import Proofs.Properties.Verity.SampledTwo.SecondCoins
public import Proofs.Properties.Verity.SampledTwo.CutHeld

/-! The bytes each stage read start the coins the device's log shows released in its family; the run's coin
check passes a signature from an earlier monitoring period. -/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled (freshCoins readLog)

def CaseIsSent (C : CaseOf) : Prop :=
  ∀ o k p c e, C o = some k → Decoded k p c e →
    caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs

/-- The cut reads a log of every accepted play; where it reads none, `LogIsDevice` holds of any log and says
nothing. -/
def CutReads (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → readsLogs o.sent ≠ []

def FirstSampled (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ i read, inputOn C o = some i → FirstRead i read →
    ∀ p c e key d ξ, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key → firstSamples key p c e = some d →
      sampleCoins d.key d.window d.label d.after (firstCoins e) = .ok ξ →
      bitsOf (i.source.extract 0 read) <+: bitsOf ξ

def SecondAlone (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e key d₁ ξ r₁ d₂, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key →
    firstSamples key p c e = some d₁ → sampleCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) = .ok ξ →
    firstRead key p c e ξ = some r₁ → secondSamples p e = some d₂ → d₁.window ≠ d₂.window →
    freshCoins d₂.label d₂.after r₁ = []

theorem firstRead_some {key : ChallengerKey} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {ξ : ByteArray} {d : Draws} {r : List Release}
    (hd : firstSamples key p c e = some d) (h : firstRead key p c e ξ = some r) :
    ∃ lines, firstLines e = .ok lines ∧ readLog d.label d.after [] ξ lines = some r := by
  unfold firstRead at h
  rw [hd] at h
  cases hl : firstLines e with
  | error _ => simp [hl, Except.toOption] at h
  | ok lines => exact ⟨lines, rfl, by simpa [hl, Except.toOption] using h⟩

theorem secondRead_some {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {bs : _root_.Core.Blobs} {ξ : ByteArray} {d : Draws} {r : List Release}
    (hd : secondSamples p e = some d) (h : secondRead p e bs ξ = some r) :
    ∃ recs lines, secondRecords bs e = some recs ∧ Verity.Sampled.linesOf e.interior = .ok lines ∧
      readLog d.label d.after recs ξ lines = some r := by
  unfold secondRead at h
  rw [hd] at h
  cases hr : secondRecords bs e with
  | none => simp [hr] at h
  | some recs =>
    cases hl : Verity.Sampled.linesOf e.interior with
    | error _ => simp [hr, hl, Except.toOption] at h
    | ok lines => exact ⟨recs, lines, rfl, rfl, by simpa [hr, hl, Except.toOption] using h⟩

theorem firstLines_arr {e : VStar.Recursive.SampledTwo.Evidence} {l : List String} (h : firstLines e = .ok l) :
    ∃ ls : Array Json, Spot.getD (Spot.getD (firstCoins e) "session") "log" = .arr ls ∧ ls.toList = l.map Json.str := by
  unfold firstLines at h
  split at h
  · rename_i ls hls
    exact ⟨ls, hls, mapM_str_inv (fun j x hj => by cases j <;> cases hj <;> rfl) _ _ h⟩
  · first
      | cases h
      | simp [throw, throwThe, MonadExceptOf.throw] at h

theorem firstSample_of {k : Case} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} {d : Draws} (hd : Decoded k p c e)
    (h : firstSamples key p c e = some d) : firstDraw k = some (d.label, d.after) := by
  obtain ⟨hp, hc, -⟩ := hd
  unfold firstSamples at h
  cases hb : boundaryOf p with
  | error _ => simp [hb, Except.toOption] at h
  | ok bp =>
    cases hw : bp.verifier.windowRoot with
    | none => simp [hb, hw, Except.toOption] at h
    | some w =>
      cases hl : firstLabel p with
      | none => simp [hb, hw, hl, Except.toOption] at h
      | some lab =>
        have h' : some (⟨key, w, lab, c.digest⟩ : Draws) = some d := by
          rw [← h]
          simp [hb, hw, hl, Except.toOption]
        obtain rfl := Option.some.inj h'
        unfold firstLabel at hl
        unfold firstDraw
        rw [hc, hp]
        show (p.boundary.challenger.map (·.2)).map (fun l => (l, c.digest)) = some (lab, c.digest)
        rw [hl]
        rfl

theorem secondSample_of {k : Case} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {s : String} {c₂ : Spot.OneStage.Commitment} {d : Draws}
    (hd : Decoded k p c e) (hs : Flock.canon e.record = .ok s) (hc : Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂)
    (h : oneStageZkSource p.audit c₂ = some d) : secondDraw k = some (d.label, d.after) := by
  obtain ⟨hp, -, he⟩ := hd
  unfold oneStageZkSource oneStageSource at h
  unfold secondDraw
  rw [hp, he]
  rcases hch : p.audit.base.challenger with _ | ⟨key, label⟩ <;>
    rcases hw : p.audit.base.verifier.windowRoot with _ | w <;> simp only [hch, hw] at h <;> cases h
  have hc' : Spot.OneStage.commitmentOf s.toByteArray = .ok c₂ := by
    first
      | exact hc
      | simpa using hc
  simp [hch, hs, hc', bind, Except.bind, Except.toOption]

theorem interiorOf_of {e : VStar.Recursive.SampledTwo.Evidence} {s : String} {c₂ : Spot.OneStage.Commitment}
    (hs : Flock.canon e.record = .ok s) (hc : Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂) : interiorOf e = .ok c₂ := by
  unfold interiorOf
  rw [hs]
  exact hc

theorem inputOf_interior_source (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (ξ ξ₂ : ByteArray) (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json))
    {it : Spot.Run.Interior} (h : (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.source = ξ₂ := by
  unfold VStar.Recursive.SampledTwo.inputOf at h
  first
    | (cases h; rfl)
    | (split at h; cases h; rfl)

theorem firstShown_of (C : CaseOf) (hC : CaseIsSent C) (hR : CutReads C) (hD : FirstSampled C) :
    FirstShown C LogIsDevice := by
  intro o ha hg i read hi hread _
  obtain ⟨k, p, c, e, _, _, _, _, hk, hd, _, _, _, _⟩ := inputOn_accepts ha hi
  obtain ⟨hcase, -⟩ := hC o k p c e hk hd
  obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ (hR o ha)
  obtain ⟨_, _, _, key, d₁, ξ, _, d₂, _, r₁, r₂, hcase', hkey, hd₁, hξ, -, -, -, hr₁, -, hlogs⟩ :=
    readsLogs_cases hl
  have hpce := Except.ok.inj (hcase.symm.trans hcase')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain ⟨lines, hlines, hrl⟩ := firstRead_some hd₁ hr₁
  obtain ⟨ls, hls, hlsl⟩ := firstLines_arr hlines
  have hpre := hD o ha i read hi hread _ _ _ key d₁ ξ hcase hkey hd₁ hξ
  refine ⟨d₁.label, d₁.after, ?_, ?_⟩
  · rw [hk, Option.bind_some]
    exact firstSample_of hd hd₁
  unfold logsOf at hlogs
  by_cases hw : d₁.window = d₂.window
  · rw [if_pos hw] at hlogs
    have hO := hg r₁ (by rw [hlogs]; simp)
    rw [sampled_shown hξ hls hlsl hrl hO]
    exact hpre
  · rw [if_neg hw] at hlogs
    have hO := hg (r₁ ++ r₂) (by rw [hlogs]; simp)
    exact hpre.trans (sampled_shown_append hξ hls hlsl hrl hO)

theorem secondShown_of (C : CaseOf) (hC : CaseIsSent C) (hR : CutReads C) (hA : SecondAlone C) :
    SecondShown C LogIsDevice := by
  intro o ha hg i it hi hit
  obtain ⟨k, p, c, e, c₂, ξ₂, _, _, hk, hd, hin, _, _, _⟩ := inputOn_accepts ha hi
  obtain ⟨hcase, hblobs⟩ := hC o k p c e hk hd
  obtain ⟨⟨s, hs, hc₂⟩, hξ₂, -, -, vs, received₂, j₁, j₂, -, hieq, -, -, -⟩ :=
    VStar.Recursive.SampledTwo.inputAt_ok hin
  rw [hieq] at hit
  have hsrc := inputOf_interior_source _ _ _ _ _ _ _ hit
  have hi₂ := interiorOf_of hs hc₂
  obtain ⟨l, hl⟩ := List.exists_mem_of_ne_nil _ (hR o ha)
  obtain ⟨_, _, _, key, d₁, ξ, _, d₂, _, r₁, r₂, hcase', hkey, hd₁, hξ, hc₂', hd₂, hξ₂', hr₁, hr₂, hlogs⟩ :=
    readsLogs_cases hl
  have hpce := Except.ok.inj (hcase.symm.trans hcase')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain rfl := Except.ok.inj (hi₂.symm.trans hc₂')
  obtain rfl := Except.ok.inj (hξ₂.symm.trans hξ₂')
  have hsrcd := secondSamples_eq hi₂ hd₂
  obtain ⟨raw, hraw, rfl⟩ := oneStageZk_reads _ _ _ hsrcd _ _ hξ₂
  obtain ⟨recs, lines, -, hlines, hrl⟩ := secondRead_some hd₂ hr₂
  obtain ⟨ls, hls, hlsl⟩ := linesOf_arr hlines
  refine ⟨d₂.label, d₂.after, ?_, ?_⟩
  · rw [hk, Option.bind_some]
    exact secondSample_of hd hs hc₂ hsrcd
  rw [hsrc]
  unfold logsOf at hlogs
  by_cases hw : d₁.window = d₂.window
  · rw [if_pos hw] at hlogs
    have hO := hg r₂ (by rw [hlogs]; simp)
    rw [sampled_shown hraw hls hlsl hrl hO]
    all_goals exact List.prefix_refl _
  · rw [if_neg hw] at hlogs
    have hO := hg (r₁ ++ r₂) (by rw [hlogs]; simp)
    have h₀ := hA o ha _ _ _ key d₁ ξ r₁ d₂ hcase hkey hd₁ hξ hr₁ hd₂ hw
    rw [sampled_shown_after hraw hls hlsl hrl h₀ hO]
    all_goals exact List.prefix_refl _

end Verity.SampledTwo
