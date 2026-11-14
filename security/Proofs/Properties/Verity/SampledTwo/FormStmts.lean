module

public import Proofs.Properties.Verity.SampledTwo.StmtsRound
public import Proofs.Properties.Verity.SampledTwo.RefsOfSent

/-!
# A template's session statement, field by field

`StmtAtFixed` (`StmtsRound.lean`) says two accepted plays with one parameters' bytes and one log before the
statements' round state one session statement a template, or sent two blobs with one SHA-512. Template `t`'s statement
on a case (`SessionMsgs.stmtAtCase`) is its member's, at the circuit and public files the parameters name on the
blobs, the blobs' lookup tables and the session's record's sample as printed. Here `StmtAtFixed` is a theorem from the
two fields no blob's digest fixes:

* **The member** (`MemberAtFixed`): the two plays' `t`-th statements name one member of the parameters.
* **The sample** (`DrawAtFixed`): the two plays' `t`-th sessions' records print one sample.

Between two plays with one parameters' bytes the parameters are one (`SessionMsgs.params_eq_of_sent`), so one member
has one partition, one program, one registered reads and one pair of digests. Its circuit file and its public file are
each the blob under one digest on either play's blobs: one file, or two blobs that collide (`get_same`). The lookup
tables are the pinned tables its circuit file names, which every accepted play's blobs hold (`accepted_tables`): one
list, or two blobs that collide (`names_same`, `tables_same`). A collision is the pair `Messages.blobsAlt` returns
(`Messages.blobsAlt_of`).

**Not established here**: `MemberAtFixed` and `DrawAtFixed`, hypotheses. The check holds the statements to the
members' order and gives a member a statement only where the sample draws one of its units, and it holds each
session's sample to the verifier's own at the stage coins (`SpotFlock.sample_of_stmtOk`); that the stage coins of two
plays with one log before the round are one, and what follows of the members named and the samples printed, is not
shown.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **Each template's statement names one member**: at two plays the check accepts whose logs are the device's, with
one parameters and one log before round `stage + 1`, the `t`-th of the statements' sessions' statements name the same
member of the parameters, template by template. -/
def MemberAtFixed (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') →
      ∀ t < A.h.ms.length, SessionMsgs.memberAt p e t = SessionMsgs.memberAt p e' t

/-- **Each template's session's record prints one sample**: at two plays the check accepts whose logs are the
device's, with one parameters and one log before round `stage + 1`, the records of the `t`-th of the statements'
sessions hold samples of one printed text (`SessionMsgs.drawBytes`), template by template. -/
def DrawAtFixed (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') →
      ∀ t < A.h.ms.length, ∀ fs fs' record record', SessionMsgs.filesAt e t = some fs →
        SessionMsgs.filesAt e' t = some fs' → SessionMsgs.recordAt k.blobs fs = some record →
          SessionMsgs.recordAt k'.blobs fs' = some record' →
            SessionMsgs.drawBytes record = SessionMsgs.drawBytes record'

/-- **The member a statement names is one of the parameters' members.** -/
theorem memberAt_mem {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence} {t : ℕ}
    {m : Spot.OneStageZk.Member} (h : SessionMsgs.memberAt p e t = some m) : m ∈ p.members := by
  unfold SessionMsgs.memberAt at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨st, -, name, -, hf⟩ := h
  exact Array.mem_of_find?_eq_some hf

/-- **An accepted good play states each template's session statement**: the reading has a statements' message
(`StmtsRead`), which states one statement an entry of the evidence's `stmts`, as many as the held copies have
templates (`StmtsCount`). -/
theorem stmtAtCase_of_read {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S) (hcount : StmtsCount A C Good)
    {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} (hA : FormAccepts C o) (hG : Good o) (hk : C o = some k)
    (hf : SessionMsgs.formOf k = some (p, e)) {t : ℕ} (ht : t < A.h.ms.length) :
    ∃ x, SessionMsgs.stmtAtCase k t = some x := by
  have hA₀ := hA
  obtain ⟨k₁, hk₁, hacc⟩ := hA₀
  obtain rfl : k = k₁ := Option.some.inj (hk.symm.trans hk₁)
  obtain ⟨cm, i, -, hi, -, it, m₁, hit, -⟩ := input_of_accepts hacc
  have hin : inputOn C o = some i := by simp only [inputOn, hk, Option.bind_some, hi]
  obtain ⟨m₂, hm₂, -⟩ := hst o i it hA hin hit
  have hmsg : SessionMsgs.stmtsMsg k = some m₂ := by
    have h := hS o
    rw [hk] at h
    exact h.symm.trans hm₂
  have hte : t < e.stmts.size := by
    rw [hcount o k p e hA hG hk hf]
    exact ht
  exact SessionMsgs.stmtAtCase_of_stmtsMsg hmsg hf hte

/-- **`StmtAtFixed` from the member each statement names and the sample each record prints**
(`MemberAtFixed`, `DrawAtFixed`), where each case is the one sent and the reading's statements are the case's: the
circuit file, the public file and the lookup tables are the blobs', one on the two plays or two blobs with one
SHA-512. -/
theorem stmtAtFixed_of (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hsent : CaseIsSent C)
    (hst : StmtsRead A C S) (hcount : StmtsCount A C Good) (hmemb : MemberAtFixed A C Good L)
    (hdraw : DrawAtFixed A C Good L) : StmtAtFixed A C Good L := by
  intro o o' k k' hA hG hA' hG' hk hk' hp hlog
  have hA₀ := hA
  have hA₀' := hA'
  obtain ⟨k₁, hk₁, pp, c, e, _, _, hdec, -⟩ := hA₀
  obtain ⟨k₁', hk₁', pp', c', e', _, _, hdec', -⟩ := hA₀'
  obtain rfl : k = k₁ := Option.some.inj (hk.symm.trans hk₁)
  obtain rfl : k' = k₁' := Option.some.inj (hk'.symm.trans hk₁')
  obtain rfl : pp = pp' := SessionMsgs.params_eq_of_sent hsent hk hk' hdec hdec' hp
  have hf := SessionMsgs.formOf_of_decoded hdec
  have hf' := SessionMsgs.formOf_of_decoded hdec'
  have hbs : k.blobs = Core.Blobs.ofArray o.sent.blobs := (hsent o k pp c e hk hdec).2
  have hbs' : k'.blobs = Core.Blobs.ofArray o'.sent.blobs := (hsent o' k' pp c' e' hk' hdec').2
  by_cases hcol : BlobsCollide o.sent.blobs o'.sent.blobs
  · obtain ⟨b, hb, b', hb', hne, hh⟩ := hcol
    exact Or.inr (Messages.blobsAlt_of hb hb' hne hh _)
  · refine Or.inl fun t ht => ?_
    obtain ⟨x, hx⟩ := stmtAtCase_of_read hS hst hcount hA hG hk hf ht
    obtain ⟨x', hx'⟩ := stmtAtCase_of_read hS hst hcount hA' hG' hk' hf' ht
    obtain ⟨p₁, e₁, m, s, record, circuit, pub, hf₁, hm, hs, hrec, hc, hpb, hxe⟩ := SessionMsgs.stmtAtCase_some hx
    obtain ⟨p₂, e₂, m', s', record', circuit', pub', hf₂, hm', hs', hrec', hc', hpb', hxe'⟩ :=
      SessionMsgs.stmtAtCase_some hx'
    rw [hf] at hf₁
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf₁)
    rw [hf'] at hf₂
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf₂)
    have hmm := hmemb o o' k k' pp e e' hA hG hA' hG' hk hk' hlog hf hf' t ht
    rw [hm, hm'] at hmm
    obtain rfl : m = m' := Option.some.inj hmm
    have hdd : SessionMsgs.drawBytes record = SessionMsgs.drawBytes record' :=
      hdraw o o' k k' pp e e' hA hG hA' hG' hk hk' hlog hf hf' t ht s s' record record' hs hs' hrec hrec'
    have hin : m ∈ pp.members := memberAt_mem hm
    have ht₁ := accepted_tables hA hk hdec m hin
    have ht₂ := accepted_tables hA' hk' hdec' m hin
    rw [hbs] at hc hpb ht₁
    rw [hbs'] at hc' hpb' ht₂
    obtain rfl : circuit = circuit' :=
      (get_same (SpotFlock.HeldOf.ok_of_toOption hc) (SpotFlock.HeldOf.ok_of_toOption hc')).resolve_right hcol
    obtain rfl : pub = pub' :=
      (get_same (SpotFlock.HeldOf.ok_of_toOption hpb) (SpotFlock.HeldOf.ok_of_toOption hpb')).resolve_right hcol
    have htab := (tables_same ((names_same ht₁ ht₂).resolve_right hcol)).resolve_right hcol
    rw [hx, hx', hxe, hxe', hbs, hbs', htab, hdd]

end Verity.SampledTwo
