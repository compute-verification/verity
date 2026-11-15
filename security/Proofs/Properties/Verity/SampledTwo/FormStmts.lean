module

public import Proofs.Properties.Verity.SampledTwo.StmtsRound
public import Proofs.Properties.Verity.SampledTwo.RefsOfSent

/-!
# A template's session statement, field by field

`StmtAtFixed` (`StmtsRound.lean`) says two accepted plays with one parameters' bytes and one log before the
statements' round state one session statement a template, or sent two blobs with one SHA-512. Template `t`'s statement
on a case (`SessionMsgs.stmtAtCase`) is its member's, at the circuit and public files the parameters name on the
blobs, the blobs' lookup tables and the session's record's sample as printed. Here are the pieces `StmtAtFixed` is
made of, two of them fields no blob's digest fixes:

* **The member** (`MemberAtFixed`): the two plays' `t`-th statements name one member of the parameters.
* **The sample** (`SampleAtFixed`): the two plays' `t`-th sessions' records print one sample.

Between two plays with one parameters' bytes the parameters are one (`SessionMsgs.params_eq_of_sent`), so one member
has one partition, one program, one registered reads and one pair of digests. Its circuit file and its public file are
each the blob under one digest on either play's blobs: one file, or two blobs that collide (`get_same`). Two lists of
pinned tables with one list of names are one list, or two blobs that collide (`tables_same`). A collision is the pair
`Messages.blobsAlt` returns (`Messages.blobsAlt_of`).

**Not established here**: `StmtAtFixed`. The statement reads every pinned table the blobs hold
(`Spot.OneStageZk.tablesOf`), and the per-member form hands each member only the tables its circuit looks up
(`Spot.OneStageZk.ownTables`), so no accepted play fixes which pinned tables its blobs hold. `MemberAtFixed` and
`SampleAtFixed` are not shown either: `FormDraw.lean` and `FormMember.lean` hold their pieces, the model's check holding
each session's sample to the verifier's own at the stage coins (`SpotFlock.sample_of_stmtOk`) and each statement's
public file to its template's part, and the form's check holding each member's public file to the member's units.
`FormBytes.lean` gives the pair of stage coins from the coins the log shows (`playCoinsOne_of_shown`).
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
sessions hold samples of one printed text (`SessionMsgs.sampleBytes`), template by template. -/
def SampleAtFixed (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') →
      ∀ t < A.h.ms.length, ∀ fs fs' record record', SessionMsgs.filesAt e t = some fs →
        SessionMsgs.filesAt e' t = some fs' → SessionMsgs.recordAt k.blobs fs = some record →
          SessionMsgs.recordAt k'.blobs fs' = some record' →
            SessionMsgs.sampleBytes record = SessionMsgs.sampleBytes record'

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

end Verity.SampledTwo
