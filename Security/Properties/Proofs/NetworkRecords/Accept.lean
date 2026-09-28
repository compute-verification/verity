import Properties.Proofs.NetworkRecords.Check

/-!
# What the records check's acceptance means: the proof

The check accepts exactly when the evidence fits, the parameters have their digests, and no link-window has a violation
(`check_accept`); a link-window without one has its one complete record, the Program's (`egress_none`,
`ingress_none`), so acceptance means `ProgramRecords` (`acceptMeans`, `answerAcceptMeans`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Certified Link Evidence Grid Profile LinkWindow manifestSet)
open Properties.NetworkCertifier (EvidenceFits ParamsDigests)
open Properties.NetworkCertifier.CheckProofs (fits_iff any_ne zip_left)
open NetworkCheck.RecordsBytes (frameBytes_toEvidence)

namespace Properties.NetworkRecords

open CheckProofs

/-- `e`'s egress links read as v2 reads them, paired with their names, are `e`'s. -/
theorem zip_toLinks {egress : List String} {e : Records.ByteEvidence} {l : String} {L : Link}
    (hm : (l, L) ∈ egress.zip e.toEvidence.links.toList) :
    ∃ link, (l, link) ∈ egress.zip e.links.toList ∧ L = link.toLink := by
  simp only [Records.ByteEvidence.toEvidence, Array.toList_map, List.zip_map_right, List.mem_map] at hm
  obtain ⟨⟨l', link⟩, hm, he⟩ := hm
  simp only [Prod.map, id, Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  exact ⟨link, hm, rfl⟩

theorem toLinks_zip {egress : List String} {e : Records.ByteEvidence} {l : String} {link : Records.ByteLink}
    (hm : (l, link) ∈ egress.zip e.links.toList) : (l, link.toLink) ∈ egress.zip e.toEvidence.links.toList := by
  simp only [Records.ByteEvidence.toEvidence, Array.toList_map, List.zip_map_right, List.mem_map]
  exact ⟨(l, link), hm, rfl⟩

/-- `evaluate`'s three refusals of the evidence are v2's of the evidence read as bits. -/
theorem fits_bytes {p : Params} {e : Records.ByteEvidence} : EvidenceFits p e.toEvidence ↔
    ¬ e.links.size ≠ p.egress.size ∧ ¬ (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true ∧
      ¬ (e.links.any fun x => x.frames.any fun x => decide (x.payload.size ≠ p.grid.frameBytes)) = true := by
  rw [fits_iff, frameBytes_toEvidence]
  simp only [Records.ByteEvidence.toEvidence, Array.any_map, Function.comp_def]
  simp only [Array.size_map, Records.ByteLink.toLink]
  exact Iff.rfl

theorem verdicts_all {p : Params} {c : Records.Commitment} {e : Records.ByteEvidence} {eg ig : ByteArray}
    (P : LinkWindow → Prop) :
    (∀ v ∈ (Records.verdicts p c e eg ig).toList, P v) ↔
      (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
        P ⟨.egress, l, w, Records.egressVerdict p c e eg l link
          ((Records.expectedGridsOfBytes p.grid p.windows link).getD w #[]) w⟩) ∧
      ∀ l ∈ p.ingress.toList, ∀ w < p.windows, P ⟨.ingress, l, w, Records.ingressVerdict p c ig l w⟩ := by
  simp only [Records.verdicts, List.toList_toArray, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  constructor
  · intro h
    exact ⟨fun l link hm w hw => h _ (Or.inl ⟨(l, link), hm, w, hw, rfl⟩),
      fun l hl w hw => h _ (Or.inr ⟨l, hl, w, hw, rfl⟩)⟩
  · rintro ⟨he, hi⟩ v (⟨⟨l, link⟩, hm, w, hw, rfl⟩ | ⟨l, hl, w, hw, rfl⟩)
    · exact he l link hm w hw
    · exact hi l hl w hw

/-- With evidence that fits and the parameters' digests, the check is the verdicts' cascade. -/
theorem check_digested {p : Params} {c : Records.Commitment} {e : Records.ByteEvidence} {eg ig : ByteArray}
    (hfit : EvidenceFits p e.toEvidence) (hpd : ParamsDigests p eg ig) :
    ((Records.verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = none →
      (Records.verdicts p c e eg ig).find? (·.violation.isSome) = none →
        ∃ pub prof, Records.check p c e = .accept pub prof) ∧
    (∀ v, (Records.verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = some v →
      ∃ why, Records.check p c e = .reject why) ∧
    ((Records.verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = none →
      ∀ v, (Records.verdicts p c e eg ig).find? (·.violation.isSome) = some v →
        ∃ why, Records.check p c e = .undecided why) := by
  obtain ⟨h1, h2, h3⟩ := fits_bytes.mp hfit
  refine ⟨fun hf1 hf2 => ?_, fun v hf1 => ?_, fun hf1 v hf2 => ?_⟩ <;>
  · unfold Records.check Records.evaluate
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, hpd.1, hpd.2]
    dsimp only
    first
    | (rw [hf1, hf2]; exact ⟨_, _, rfl⟩)
    | (rw [hf1]; exact ⟨_, rfl⟩)
    | (rw [hf1, hf2]; exact ⟨_, rfl⟩)

/-- The check accepts only evidence that fits, under parameters with their digests, and with no violation. -/
theorem check_accept {p : Params} {c : Records.Commitment} {e : Records.ByteEvidence} {pub : Unit} {prof : Profile}
    (h : Records.check p c e = .accept pub prof) :
    EvidenceFits p e.toEvidence ∧ ∃ eg ig, ParamsDigests p eg ig ∧
      ∀ v ∈ (Records.verdicts p c e eg ig).toList, v.violation = none := by
  have hfit : EvidenceFits p e.toEvidence := by
    rw [fits_bytes]
    unfold Records.check Records.evaluate at h
    refine ⟨fun h1 => ?_, fun h2 => ?_, fun h3 => ?_⟩
    · rw [ite_eq_left h1] at h; cases h
    · by_cases h1 : e.links.size ≠ p.egress.size
      · rw [ite_eq_left h1] at h; cases h
      rw [ite_eq_right h1, ite_eq_left h2] at h; cases h
    · by_cases h1 : e.links.size ≠ p.egress.size
      · rw [ite_eq_left h1] at h; cases h
      by_cases h2 : (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true
      · rw [ite_eq_right h1, ite_eq_left h2] at h; cases h
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_left h3] at h; cases h
  obtain ⟨h1, h2, h3⟩ := fits_bytes.mp hfit
  refine ⟨hfit, ?_⟩
  cases hd1 : parametersDigest p.grid (some (manifestSet p.syncSet)) with
  | error _ =>
    unfold Records.check Records.evaluate at h
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, hd1] at h
    cases h
  | ok eg =>
    cases hd2 : parametersDigest p.ingressGrid none with
    | error _ =>
      unfold Records.check Records.evaluate at h
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, hd1, hd2] at h
      cases h
    | ok ig =>
      have hpd : ParamsDigests p eg ig := ⟨hd1, hd2⟩
      refine ⟨eg, ig, hpd, ?_⟩
      obtain ⟨-, hR, hU⟩ := check_digested (c := c) hfit hpd
      cases hf1 : (Records.verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) with
      | some v => obtain ⟨_, h'⟩ := hR v hf1; rw [h] at h'; cases h'
      | none =>
        cases hf2 : (Records.verdicts p c e eg ig).find? (·.violation.isSome) with
        | some v => obtain ⟨_, h'⟩ := hU hf1 v hf2; rw [h] at h'; cases h'
        | none =>
          intro v hv
          have := Array.find?_eq_none.mp hf2 v (Array.mem_def.mpr hv)
          simpa using this

/-- **Acceptance means the unit's records are the Program's.** -/
theorem acceptMeans : AcceptMeans := by
  intro p c e pub prof h
  obtain ⟨hfit, eg, ig, hpd, hall⟩ := check_accept h
  obtain ⟨he, hi⟩ := (verdicts_all (fun v => v.violation = none)).mp hall
  refine ⟨hfit, eg, ig, hpd, fun l L hm w hw => ?_, fun l hl w hw => ingress_none (hi l hl w hw)⟩
  obtain ⟨link, hm', rfl⟩ := zip_toLinks hm
  exact egress_none hfit (List.of_mem_zip hm').2 hw (he l link hm' w hw)

/-- **The same, from the bytes.** -/
theorem answerAcceptMeans : AnswerAcceptMeans := by
  intro a b v h
  unfold Core.Protocol.answer at h
  cases hp : (Core.canonicalJson a >>= Records.protocol.params) with
  | error _ => rw [hp] at h; cases h
  | ok p =>
    rw [hp] at h
    cases hc : Records.protocol.commitment b with
    | error _ => rw [hc] at h; cases h
    | ok cm =>
      rw [hc] at h
      cases he : Records.protocol.evidence v with
      | error _ => simp only [he] at h; cases h
      | ok ev =>
        simp only [he] at h
        cases hch : Records.protocol.check p cm ev with
        | accept pub prof => exact ⟨p, cm, ev, rfl, rfl, rfl, acceptMeans p cm ev pub prof hch⟩
        | reject _ => rw [hch] at h; cases h
        | undecided _ => rw [hch] at h; cases h

/-! ## Completeness -/

/-- **The check accepts every unit whose records are the Program's.** -/
theorem programRecords_accepted {p : Params} {c : Records.Commitment} {e : Records.ByteEvidence}
    (h : ProgramRecords p c e.toEvidence) :
    ∃ pub prof, Records.check p c e = .accept pub prof := by
  obtain ⟨hfit, eg, ig, hpd, he, hi⟩ := h
  obtain ⟨hA, -, -⟩ := check_digested (c := c) hfit hpd
  have hall : ∀ v ∈ (Records.verdicts p c e eg ig).toList, v.violation = none :=
    (verdicts_all (fun v => v.violation = none)).mpr ⟨fun l link hm w hw =>
      egress_none_of hfit (List.of_mem_zip hm).2 hw (he l link.toLink (toLinks_zip hm) w hw),
      fun l hl w hw => ingress_none_of (hi l hl w hw)⟩
  refine hA (Array.find?_eq_none.mpr fun v hv => ?_) (Array.find?_eq_none.mpr fun v hv => ?_)
  · rw [hall v (Array.mem_def.mp hv)]; simp
  · rw [hall v (Array.mem_def.mp hv)]; simp

/-- The check accepts exactly the units whose records are the Program's. -/
theorem acceptMeans_iff {p : Params} {c : Records.Commitment} {e : Records.ByteEvidence} :
    (∃ pub prof, Records.protocol.check p c e = .accept pub prof) ↔ ProgramRecords p c e.toEvidence :=
  ⟨fun ⟨pub, prof, h⟩ => acceptMeans p c e pub prof h, programRecords_accepted⟩

end Properties.NetworkRecords
