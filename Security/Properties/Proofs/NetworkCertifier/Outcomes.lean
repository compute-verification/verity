import Properties.Proofs.NetworkCertifier.Check

/-!
# The check's outcomes: refusals, acceptance, completeness and undecided

The proofs of properties (1) to (4) of `Properties.NetworkCertifier`.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check

namespace Properties.NetworkCertifier

open CheckProofs GridProofs

theorem toOption_none {ε α : Type} {x : Except ε α} (h : x.toOption = none) : ∃ e, x = .error e := by
  cases x with
  | error e => exact ⟨e, rfl⟩
  | ok _ => cases h

theorem decodeRefusals : DecodeRefusals := by
  intro a b c h
  unfold Core.Protocol.answer
  rcases h with h | h | h
  · obtain ⟨err, he⟩ := toOption_none h
    rw [he]
  · cases hp : (Core.canonicalJson a >>= protocol.params) with
    | error _ => rfl
    | ok p =>
      obtain ⟨err, he⟩ := toOption_none h
      simp only [he]
  · cases hp : (Core.canonicalJson a >>= protocol.params) with
    | error _ => rfl
    | ok p =>
      cases hc : protocol.commitment b with
      | error _ => rfl
      | ok cm =>
        obtain ⟨err, he⟩ := toOption_none h
        simp only [he]

theorem answerAccepts : AnswerAccepts := by
  intro a b c
  unfold Core.Protocol.answer
  constructor
  · intro h
    cases hp : (Core.canonicalJson a >>= protocol.params) with
    | error _ => rw [hp] at h; cases h
    | ok p =>
      rw [hp] at h
      cases hc : protocol.commitment b with
      | error _ => rw [hc] at h; cases h
      | ok cm =>
        rw [hc] at h
        cases he : protocol.evidence c with
        | error _ => simp only [he] at h; cases h
        | ok ev =>
          simp only [he] at h
          cases hch : protocol.check p cm ev with
          | accept pub prof => exact ⟨p, cm, ev, rfl, rfl, rfl, pub, prof, hch⟩
          | reject _ => rw [hch] at h; cases h
          | undecided _ => rw [hch] at h; cases h
  · rintro ⟨p, cm, ev, hp, hc, he, pub, prof, hch⟩
    simp only [hp, hc, he, hch]

theorem malformedEvidenceRejects : MalformedEvidenceRejects := fun _ _ _ h => check_unfit h

theorem undigestedParamsReject : UndigestedParamsReject := by
  intro p c e h
  by_cases hfit : EvidenceFits p e
  · exact check_undigested hfit h
  · exact check_unfit hfit

theorem acceptsIff : AcceptsIff := by
  intro p c e
  show (∃ pub prof, check p c e = .accept pub prof) ↔ _
  constructor
  · rintro ⟨pub, prof, h⟩
    by_cases hfit : EvidenceFits p e
    swap
    · obtain ⟨_, h'⟩ := check_unfit (c := c) hfit
      rw [h] at h'
      cases h'
    by_cases hd : ∃ eg ig, ParamsDigests p eg ig
    swap
    · obtain ⟨_, h'⟩ := check_undigested (c := c) hfit (fun eg ig h => hd ⟨eg, ig, h⟩)
      rw [h] at h'
      cases h'
    obtain ⟨eg, ig, hpd⟩ := hd
    obtain ⟨he, hi⟩ := (verdicts_all (fun v => v.violation = none)).mp ((check_accept hfit hpd).mp ⟨pub, prof, h⟩)
    exact ⟨hfit, eg, ig, hpd, fun l link hm w hw => (egress_none hfit (List.of_mem_zip hm).2 hw).mp (he l link hm w hw),
      fun l hl w hw => ingress_none.mp (hi l hl w hw)⟩
  · rintro ⟨hfit, eg, ig, hpd, he, hi⟩
    exact (check_accept hfit hpd).mpr ((verdicts_all (fun v => v.violation = none)).mpr
      ⟨fun l link hm w hw => (egress_none hfit (List.of_mem_zip hm).2 hw).mpr (he l link hm w hw),
        fun l hl w hw => ingress_none.mpr (hi l hl w hw)⟩)

theorem acceptsOneRecordEach : AcceptsOneRecordEach := by
  intro p c e h d l hl w hw
  obtain ⟨hfit, eg, ig, hpd, he, hi⟩ := (acceptsIff p c e).mp h
  cases d with
  | egress =>
    obtain ⟨link, hm⟩ := zip_left hfit hl
    obtain ⟨r, dg, hset, -⟩ := he l link hm w hw
    rw [hset.1.1]
    rfl
  | ingress =>
    obtain ⟨r, dg, hset, -⟩ := hi l hl w hw
    rw [hset.1.1]
    rfl

theorem honestAccepted : HonestAccepted := by
  intro p c e eg ig hfit hpd hsync heg hing
  refine (acceptsIff p c e).mpr ⟨hfit, eg, ig, hpd, fun l link hm w hw => ?_, fun l hl w hw => ?_⟩
  · obtain ⟨syncs, hdecl, h, dg, σ, hh, hdg, hrec, m, hm', hσ⟩ := heg l link hm w hw
    obtain ⟨-, -, -, hst, hdl, -⟩ := recordHead_head hh
    refine ⟨honestRecord p .egress l w h dg σ, dg, ⟨⟨hrec, hdg, rfl, rfl, rfl, rfl, rfl, m, hm', hσ⟩, hst, hdl⟩,
      hsync link (List.of_mem_zip hm).2 w hw, fun syncs' hdecl' => ?_⟩
    rw [← declares_unique hdecl hdecl']
    exact ⟨h, hh, hdg⟩
  · obtain ⟨h, dg, σ, hb, ha, hf, hbm, hpar, hst, hdl, hdg, hrec, m, hm', hσ⟩ := hing l hl w hw
    exact ⟨honestRecord p .ingress l w h dg σ, dg, ⟨⟨hrec, hdg, rfl, rfl, rfl, rfl, rfl, m, hm', hσ⟩, hst, hdl⟩,
      hb, ha, hf, hbm, hpar⟩

theorem undecidedIff : UndecidedIff := by
  intro p c e
  show (∃ why, check p c e = .undecided why) ↔ _
  constructor
  · rintro ⟨why, h⟩
    by_cases hfit : EvidenceFits p e
    swap
    · obtain ⟨_, h'⟩ := check_unfit (c := c) hfit
      rw [h] at h'
      cases h'
    by_cases hd : ∃ eg ig, ParamsDigests p eg ig
    swap
    · obtain ⟨_, h'⟩ := check_undigested (c := c) hfit (fun eg ig h => hd ⟨eg, ig, h⟩)
      rw [h] at h'
      cases h'
    obtain ⟨eg, ig, hpd⟩ := hd
    obtain ⟨hall, hex⟩ := (check_undecided hfit hpd).mp ⟨why, h⟩
    obtain ⟨he, hi⟩ := (verdicts_all (fun v => v.violation = none ∨ v.violation = some "no-verdict")).mp hall
    refine ⟨hfit, eg, ig, hpd, fun l link hm w hw => ?_, fun l hl w hw => ?_, ?_⟩
    · rcases he l link hm w hw with h0 | h0
      · exact Or.inl ((egress_none hfit (List.of_mem_zip hm).2 hw).mp h0)
      · exact Or.inr (egress_noVerdict.mp h0)
    · rcases hi l hl w hw with h0 | h0
      · exact Or.inl (ingress_none.mp h0)
      · exact Or.inr (ingress_noVerdict.mp h0)
    · rcases (verdicts_exists (fun v => v.violation = some "no-verdict")).mp hex with
        ⟨l, link, hm, w, hw, h0⟩ | ⟨l, hl, w, hw, h0⟩
      · exact ⟨.egress, l, (List.of_mem_zip hm).1, w, hw, egress_noVerdict.mp h0⟩
      · exact ⟨.ingress, l, hl, w, hw, ingress_noVerdict.mp h0⟩
  · rintro ⟨hfit, eg, ig, hpd, he, hi, d, l, hl, w, hw, hpend⟩
    refine (check_undecided hfit hpd).mpr ⟨(verdicts_all _).mpr ⟨fun l link hm w hw => ?_, fun l hl w hw => ?_⟩,
      (verdicts_exists _).mpr ?_⟩
    · rcases he l link hm w hw with h0 | h0
      · exact Or.inl ((egress_none hfit (List.of_mem_zip hm).2 hw).mpr h0)
      · exact Or.inr (egress_noVerdict.mpr h0)
    · rcases hi l hl w hw with h0 | h0
      · exact Or.inl (ingress_none.mpr h0)
      · exact Or.inr (ingress_noVerdict.mpr h0)
    · cases d with
      | egress =>
        obtain ⟨link, hm⟩ := zip_left hfit hl
        exact Or.inl ⟨l, link, hm, w, hw, egress_noVerdict.mpr hpend⟩
      | ingress => exact Or.inr ⟨l, hl, w, hw, ingress_noVerdict.mpr hpend⟩

end Properties.NetworkCertifier
