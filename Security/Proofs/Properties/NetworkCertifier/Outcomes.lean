import Proofs.Properties.NetworkCertifier.Check

/-!
# The check's outcomes: refusals, acceptance and undecided

The proofs of properties (1) to (3) of `Properties.NetworkCertifier`, from the iffs they are the forward halves of
(`answer_accepts_iff`, `accepts_iff`, `undecided_iff`). The converse halves, and the honest transcript's acceptance
(`HonestAccepted`), are completeness, at the end.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync)

namespace Properties.NetworkCertifier

open CheckProofs GridProofs

theorem toOption_none {ε α : Type} {x : Except ε α} (h : x.toOption = none) : ∃ e, x = .error e := by
  cases x with
  | error e => exact ⟨e, rfl⟩
  | ok _ => cases h

theorem decodeRefusals : DecodeRefusals := by
  intro a b c h
  unfold Core.Protocol.answer at h
  cases hp : (Core.canonicalJson a >>= protocol.params) with
  | error _ => rw [hp] at h; exact (h rfl).elim
  | ok p =>
    rw [hp] at h
    cases hc : protocol.commitment b with
    | error _ => rw [hc] at h; exact (h rfl).elim
    | ok cm =>
      rw [hc] at h
      cases he : protocol.evidence c with
      | error _ => simp only [he] at h; exact (h rfl).elim
      | ok ev => exact ⟨⟨p, hp⟩, ⟨cm, hc⟩, ⟨ev, he⟩⟩

/-- `answer` accepts exactly when the three decode and the check accepts what they decode to. -/
theorem answer_accepts_iff (a b c : ByteArray) :
    (protocol.answer a b c).verdict = .accept ↔ DecodedAccepted a b c := by
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
          | accept pub prof => exact ⟨p, cm, ev, hp, hc, he, pub, prof, hch⟩
          | reject _ => rw [hch] at h; cases h
          | undecided _ => rw [hch] at h; cases h
  · rintro ⟨p, cm, ev, hp, hc, he, pub, prof, hch⟩
    simp only [hp, hc, he, hch]

theorem answerAccepts : AnswerAccepts := fun a b c => (answer_accepts_iff a b c).mp

theorem malformedEvidenceRejects : MalformedEvidenceRejects := fun _ _ _ h =>
  Classical.byContradiction fun hfit => let ⟨why, h'⟩ := check_unfit hfit; h why h'

theorem undigestedParamsReject : UndigestedParamsReject := by
  intro p c e h
  refine Classical.byContradiction fun hd => ?_
  by_cases hfit : EvidenceFits p e
  · obtain ⟨why, h'⟩ := check_undigested (c := c) hfit (fun eg ig h => hd ⟨eg, ig, h⟩)
    exact h why h'
  · obtain ⟨why, h'⟩ := check_unfit (c := c) hfit
    exact h why h'

/-- The check accepts exactly the units every sampling block of which is accepted. -/
theorem accepts_iff (p : Params) (c : Commitment) (e : Evidence) : protocol.Accepts p c e ↔ LinkWindowsAccepted p c e := by
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

theorem acceptsLinkWindows : AcceptsLinkWindows := fun p c e => (accepts_iff p c e).mp

theorem acceptsOneRecordEach : AcceptsOneRecordEach := by
  intro p c e h d l hl w hw
  obtain ⟨hfit, eg, ig, hpd, he, hi⟩ := (accepts_iff p c e).mp h
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

/-- The check is undecided exactly when no sampling block is refused and some awaits its record. -/
theorem undecided_iff (p : Params) (c : Commitment) (e : Evidence) :
    (∃ why, protocol.check p c e = .undecided why) ↔ SomePending p c e := by
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

theorem undecidedPending : UndecidedPending := fun p c e => (undecided_iff p c e).mp

/-! ## Completeness

Not properties: each says the check accepts (or is undecided on) what an honest unit sends, the converse of a property
above, or the honest transcript. `Proofs.Properties.NetworkCertifier.Instance` gives each one instance whose
hypotheses hold. -/

/-- **`answer` accepts bytes that decode to a unit the check accepts** (the converse of `AnswerAccepts`). -/
theorem answer_of_decodedAccepted (a b c : ByteArray) (h : DecodedAccepted a b c) :
    (protocol.answer a b c).verdict = .accept :=
  (answer_accepts_iff a b c).mpr h

/-- **The check accepts a unit every sampling block of which is accepted** (the converse of `AcceptsLinkWindows`). -/
theorem accepts_of_linkWindowsAccepted (p : Params) (c : Commitment) (e : Evidence) (h : LinkWindowsAccepted p c e) :
    protocol.Accepts p c e :=
  (accepts_iff p c e).mpr h

/-- **The check is undecided on a unit with no sampling block refused and some awaiting its record** (the converse of
`UndecidedPending`). -/
theorem undecided_of_somePending (p : Params) (c : Commitment) (e : Evidence) (h : SomePending p c e) :
    ∃ why, protocol.check p c e = .undecided why :=
  (undecided_iff p c e).mpr h

/-- The record an honest certifier issues for sampling block `(d, l, w)`: head `h`, the salt `s` it drew for it, and a
certificate of the key's identifier over `(d, l, w)` and `h`'s digest `dg` under `s` with signature `σ`. -/
def honestRecord (p : Params) (d : Direction) (l : String) (w : Nat) (h : RecordHead) (s dg σ : ByteArray) :
    Certified :=
  ⟨h, s, ⟨p.key.extract 0 16, l, d, w, certificateDigest s dg, σ⟩⟩

/-- **The honest transcript is accepted.** Evidence that fits, with every declared sync admitted; parameters with their
digests; and a commitment holding exactly one record of every sampling block, the honest one: on egress the head
`record_head` makes of the Program's grid under the declared syncs, on ingress a complete head settled by the deadline
of the ingress grid and its digest, each digest under the record's salt signed by the key over its
`certificateMessage` in the parameters' session. -/
def HonestAccepted : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (eg ig : ByteArray),
    EvidenceFits p e → ParamsDigests p eg ig →
    (∀ link ∈ e.links.toList, ∀ w < p.windows, SyncAdmitted p e link w) →
    (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
      ∃ syncs : Fin p.windows → ClockSync, Declares link syncs ∧ ∃ (h : RecordHead) (s dg σ : ByteArray),
        recordHead l .egress w (programGrid p syncs link.frames w) none p.grid eg true = .ok h ∧ h.digest = .ok dg ∧
        recordsOf c .egress l w = [honestRecord p .egress l w h s dg σ] ∧
        ∃ m, certificateMessage (p.key.extract 0 16) p.session l .egress w (certificateDigest s dg) = .ok m ∧
          LmsVerifies p m σ) →
    (∀ l ∈ p.ingress.toList, ∀ w < p.windows, ∃ (h : RecordHead) (s dg σ : ByteArray),
      h.buckets = (p.ingressGrid.buckets : Int) ∧
      h.allocation = (p.ingressGrid.allocation : Int) ∧ h.frameBytes = (p.ingressGrid.frameBytes : Int) ∧
      h.bucketMs = (p.ingressGrid.bucketMs : Int) ∧ h.parameters = ig ∧ h.status = none ∧ h.deadline = true ∧
      h.digest = .ok dg ∧ recordsOf c .ingress l w = [honestRecord p .ingress l w h s dg σ] ∧
      ∃ m, certificateMessage (p.key.extract 0 16) p.session l .ingress w (certificateDigest s dg) = .ok m ∧
        LmsVerifies p m σ) →
    protocol.Accepts p c e

theorem honestAccepted : HonestAccepted := by
  intro p c e eg ig hfit hpd hsync heg hing
  refine (accepts_iff p c e).mpr ⟨hfit, eg, ig, hpd, fun l link hm w hw => ?_, fun l hl w hw => ?_⟩
  · obtain ⟨syncs, hdecl, h, s, dg, σ, hh, hdg, hrec, m, hm', hσ⟩ := heg l link hm w hw
    obtain ⟨-, -, -, hst, hdl, -⟩ := recordHead_head hh
    refine ⟨honestRecord p .egress l w h s dg σ, dg, ⟨⟨hrec, hdg, rfl, rfl, rfl, rfl, rfl, m, hm', hσ⟩, hst, hdl⟩,
      hsync link (List.of_mem_zip hm).2 w hw, fun syncs' hdecl' => ?_⟩
    rw [← declares_unique hdecl hdecl']
    exact ⟨h, hh, hdg⟩
  · obtain ⟨h, s, dg, σ, hb, ha, hf, hbm, hpar, hst, hdl, hdg, hrec, m, hm', hσ⟩ := hing l hl w hw
    exact ⟨honestRecord p .ingress l w h s dg σ, dg, ⟨⟨hrec, hdg, rfl, rfl, rfl, rfl, rfl, m, hm', hσ⟩, hst, hdl⟩,
      hb, ha, hf, hbm, hpar⟩

end Properties.NetworkCertifier
