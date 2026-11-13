module

public import Proofs.Properties.Verity.SampledTwo.FormRounds

/-!
# The sessions' records the check read, against the files the messages are read from

`reached_form` (`FormRounds.lean`) takes two facts about the sessions' records: the check read a record for each
template (`SessionCoins.RecordsCover`), and the messages the session reading reads of template `t`'s session are those
of the `t`-th record (`SessionCoins.MsgsOfRecords`). Template `t`'s session on a case is the `t`-th of the statements'
sessions (`SessionMsgs.filesAt`), the order `SessionCoins.sessionRecords` reads the records in, so both facts are about
one session, and this module proves the parts of them that are about which record is read.

* **`RecordsCover`** (`SessionCoins.recordsCover_of`): where the reading's statements are the case's
  (`SessionMsgs.stmtsMsg`), from `StmtsRead`, `StmtsCount` and `SessionCoins.HeldSize`. An accepted play's check read a
  record for each of the statements' sessions (`SessionCoins.sessionRecords_read`). The reading has a statements'
  message, so each template below the statements' count has a session statement
  (`SessionMsgs.stmtAtCase_of_stmtsMsg`), and so a `t`-th session (`SessionMsgs.filesAt_of_stmt`).
* **The record clause of `MsgsOfRecords`** (`SessionCoins.record_of_parts`): where template `t`'s session has
  messages (`SessionMsgs.SessionParts`), the `t`-th record the check read is the parse of the text those messages are
  read from. The check reads the record's blob under one label and `SessionMsgs.recordAt` under another, and a label
  only names a refusal (`Verity.Sampled.Record.get_what`). The check reads a text that doesn't parse as `null`; a
  session with messages has a text that parses (`FlockDeploy.liveOf_parts`).
* **`MsgsOfRecords`** (`SessionCoins.msgsOfRecords_of`): where the reading's messages are the case's
  (`SessionMsgs.msgAtCase`), from `SessionCoins.VerdictsAt` and the record clause.

**Not established here**: `SessionCoins.VerdictsAt`, that the masked verdict accepts each template's session's record
with its proof files at the setup of the session's statement, a setup of one table under SHA-512 digests with the
template's rounds, and that the session has messages. `StmtsCount` and `StmtsRead` are taken as hypotheses.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)

/-! ## The records, one a session -/

/-- **An accepted play's check read a record for each of the statements' sessions**: the case decodes, and
`sessionRecords` is the records of the statements' sessions, in order. -/
theorem sessionRecords_read (C : CaseOf) (hsz : HeldSize) {o : Verity.Exchange} (hacc : FormAccepts C o) :
    ∃ k e recs, C o = some k ∧ VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧
      sessionRecords C o = some recs ∧
      (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs) = some recs := by
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨k, p, e, s, c₂, d, ξ₂, recs, lines, hk, -, he, -, -, -, -, hrecs, -⟩ := interiorRead_some hl
  exact ⟨k, e, recs, hk, he, sessionRecords_of hk he hrecs, hrecs⟩

/-- **The records `sessionRecords` reads are the records of the statements' sessions.** -/
theorem sessionRecords_some {C : CaseOf} {o : Verity.Exchange} {k : Case}
    {e : VStar.Recursive.SampledTwo.Evidence} {recs : List Json} (hk : C o = some k)
    (he : VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e) (h : sessionRecords C o = some recs) :
    (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs) = some recs := by
  unfold sessionRecords at h
  simp only [hk, he, Option.bind_eq_bind, Option.bind_some, Except.toOption] at h
  exact h

/-- **The `t`-th record read is the `t`-th session's**, where the statements have a `t`-th session. -/
theorem record_at {bs : Core.Blobs} {e : VStar.Recursive.SampledTwo.Evidence} {t : ℕ}
    {fs : Spot.OneStageZk.RunFiles} {recs : List Json} (hs : SessionMsgs.filesAt e t = some fs)
    (hmap : (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn bs) = some recs) :
    ∃ rc, recs[t]? = some rc ∧ Verity.Sampled.recordOn bs fs = some rc := by
  unfold SessionMsgs.filesAt at hs
  obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.1 hs
  obtain ⟨hlen, hall⟩ := SessionMsgs.mapM_some _ hmap
  have ht : t < recs.length := by omega
  refine ⟨recs[t], List.getElem?_eq_getElem ht, ?_⟩
  rw [← hget]
  exact hall t hlt ht

/-- **`RecordsCover`, from the statements' count**: where the reading's statements are the case's, an accepted play
whose log the check read is the device's has a session statement for each template (`StmtsRead`, `StmtsCount`), so a
session, and the check read each session's record (`sessionRecords_read`). -/
theorem recordsCover_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hsz : HeldSize) (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S)
    (hcount : StmtsCount A C Good) : RecordsCover A C Good := by
  intro o hacc hgood t ht
  obtain ⟨k, e, recs, hk, he, hrecs, hmap⟩ := sessionRecords_read C hsz hacc
  have hacc₀ := hacc
  obtain ⟨k', hk', hon⟩ := hacc₀
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  have hon₀ := hon
  obtain ⟨pp, c, e', pub, prof, hdec, -⟩ := hon₀
  obtain rfl : e = e' := Except.ok.inj (he.symm.trans hdec.2.2)
  have hf := SessionMsgs.formOf_of_decoded hdec
  obtain ⟨cm, i, -, hi, -, it, m₁, hit, -⟩ := input_of_accepts hon
  have hin : inputOn C o = some i := by simp only [inputOn, hk, Option.bind_some, hi]
  obtain ⟨m₂, hm₂, -⟩ := hst o i it hacc hin hit
  have hmsg : SessionMsgs.stmtsMsg k = some m₂ := by
    rw [hS o, hk] at hm₂
    exact hm₂
  have hn := hcount o k pp e hacc hgood hk hf
  obtain ⟨x, hx⟩ := SessionMsgs.stmtAtCase_of_stmtsMsg hmsg hf (lt_of_lt_of_eq ht hn.symm)
  obtain ⟨fs, hs⟩ := SessionMsgs.filesAt_of_stmt hx hf
  obtain ⟨rc, hrc, -⟩ := record_at hs hmap
  exact ⟨recs, rc, hrecs, hrc⟩

/-! ## The record the messages are read from -/

/-- **The record the check read of a session is its text's parse, or `null`**: `Verity.Sampled.recordOn` and
`SessionMsgs.recordAt` read one blob, under two labels. -/
theorem recordOn_of_recordAt {bs : Core.Blobs} {fs : Spot.OneStageZk.RunFiles} {record : String} {rc : Json}
    (hat : SessionMsgs.recordAt bs fs = some record) (hon : Verity.Sampled.recordOn bs fs = some rc) :
    rc = (Flock.JsonParse.parse record).toOption.getD Json.null := by
  unfold SessionMsgs.recordAt at hat
  unfold Verity.Sampled.recordOn at hon
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hat hon
  obtain ⟨b, hb, hrec⟩ := hat
  obtain ⟨b', hb', txt, htxt, rfl⟩ := hon
  have e1 : b' = b :=
    Except.ok.inj ((Verity.Sampled.Record.get_what (toOption_ok hb')).symm.trans (toOption_ok hb))
  rw [e1, hrec] at htxt
  obtain rfl := Option.some.inj htxt
  rfl

/-- **The record clause of `MsgsOfRecords`**: where template `t`'s session on the case has messages
(`SessionMsgs.SessionParts`, `SessionMsgs.sessionMsgs`), the `t`-th of the sessions' records the check read is the
parse of the session's record's text. -/
theorem record_of_parts {C : CaseOf} {o : Verity.Exchange} {k : Case} (hk : C o = some k) {t : ℕ} {x : Msg}
    {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} {record : String}
    {proofs : Array ByteArray} {ms : List Msg} (hparts : SessionMsgs.SessionParts k t x s spec sts record proofs)
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) {recs : List Json}
    (hrecs : sessionRecords C o = some recs) :
    ∃ rc, recs[t]? = some rc ∧ Flock.JsonParse.parse record = .ok rc := by
  obtain ⟨p, e, fs, hf, hs, hr, -, -, -⟩ := hparts
  have hmap := sessionRecords_some hk (SessionMsgs.formOf_evidence hf) hrecs
  obtain ⟨rc, hrc, hon⟩ := record_at hs hmap
  obtain ⟨lv, mids, hlv, -, -⟩ := SessionMsgs.sessionMsgs_some hm
  obtain ⟨j, _, _, _, hj, -⟩ := FlockDeploy.liveOf_parts (SpotFlock.ok_of_toOption hlv)
  have e1 := recordOn_of_recordAt hr hon
  rw [hj] at e1
  have e2 : rc = j := e1
  exact ⟨j, hrc.trans (congrArg some e2), hj⟩

/-! ## `MsgsOfRecords`, from the verdicts -/

/-- **The masked verdict accepts each template's session at its statement**: on a play the form accepts whose log
the check read is the device's, template `t`'s session on the case (`SessionMsgs.SessionParts`) has a statement that
sets up one table, `tab t`, under SHA-512 digests, with `2 + 2 · repRounds` rounds; the masked verdict accepts its
record's text with its proof files at that setup; and the session has messages (`SessionMsgs.sessionMsgs`). -/
def VerdictsAt (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o k, FormAccepts C o → Good o → C o = some k → ∀ t, t < A.h.ms.length →
    ∃ (x : Msg) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (st : Flock.Stmt) (record : String)
      (proofs : Array ByteArray) (ms : List Msg),
      SessionMsgs.SessionParts k t x s spec #[st] record proofs ∧
      Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok () ∧
      spec.sha512Digests = true ∧ spec.tables = #[tab t] ∧
      SpotFlock.k₀ A.h t = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionMsgs.sessionMsgs spec x record proofs = some ms

/-- **`MsgsOfRecords`, from the verdicts**: where the reading's messages are the case's (`SessionMsgs.msgAtCase`),
the `t`-th record the check read is the parse of the text the verdict accepted (`record_of_parts`), and the reading's
messages of template `t`'s session are that session's (`SessionMsgs.msgsAtCase_of_parts`). -/
theorem msgsOfRecords_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    {tab : ℕ → String} (hS : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hv : VerdictsAt A C Good tab) : MsgsOfRecords A C S Good tab := by
  intro o hacc hgood recs rc t ht hrecs hrc
  have hacc₀ := hacc
  obtain ⟨k, hk, -⟩ := hacc₀
  obtain ⟨x, s, spec, st, record, proofs, ms, hparts, hver, hsha, htab, hk₀, hm⟩ := hv o k hacc hgood hk t ht
  obtain ⟨rc', hrc', hparse⟩ := record_of_parts hk hparts hm hrecs
  obtain rfl : rc' = rc := Option.some.inj (hrc'.symm.trans hrc)
  have hms := SessionMsgs.msgsAtCase_of_parts hparts hm
  have hsu : FlockDeploy.setupOf true x = some (s, spec, #[st]) := by
    obtain ⟨_, _, _, -, -, -, -, hsu, -⟩ := hparts
    exact hsu
  refine ⟨x, s, spec, st, record, proofs, ms, hsu, hver, hsha, htab, hk₀, hparse, hm, fun j => ?_⟩
  rw [hS o t j, hk]
  show (SessionMsgs.msgsAtCase k t).bind (fun ms => ms[j]?) = ms[j]?
  rw [hms]
  rfl

end Verity.SampledTwo.SessionCoins
