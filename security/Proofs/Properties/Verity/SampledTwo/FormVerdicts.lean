module

public import Proofs.Properties.Verity.SampledTwo.FormRecords
public import Proofs.Properties.Verity.SampledTwo.RoundsRise

/-!
# The verdict the form's check gave template `t`'s session

Template `t`'s session on a case is the `t`-th of the statements' sessions (`SessionMsgs.filesAt`), and its member is
the one that session's statement names (`SessionMsgs.memberAt`). This module proves that an accepted play's check
verified that session at that member: the first part of `SessionCoins.VerdictsAt` (`FormRecords.lean`).

* **The pairing by position** (`SessionCoins.paired_at`): `Spot.OneStageZk.membersHeld` holds each statement to the
  member it names, the first of that name among the parameters', and to one session (`heldStep_paired`,
  `membersHeld_paired`). So the `t`-th session is the one session of the `t`-th statement, and `SessionMsgs.memberAt`
  is the member the check ran that statement at.
* **Each statement's sessions, with its member's statement kept** (`SessionCoins.runMembers_stmt`): where the
  per-member form's run accepts every session, each statement's member's statement reads
  (`Spot.OneStageZk.Member.statement`) and each of the statement's sessions was accepted at it
  (`MemberChecks.SessionAccepted`).
* **The verdict at template `t`** (`SessionCoins.verified_at`): on an accepted play, Flock's masked verdict accepts
  the text of template `t`'s session's record with the proof files `SessionMsgs.SessionParts` names, at the session
  the form set up (`Spot.OneStageZk.setupAt`) on the member's circuit and public files on the blobs and its registered
  reads, at the record's sample. Those files, with the record's sample, are the fields of the statement
  `SessionMsgs.stmtAtCase` gives the template (`SessionCoins.stmtAtCase_of_member`).

**Not established here**: that the session the form set up is the one the statement sets up
(`FlockDeploy.setupOf`), which needs the statement's printed files read back (`MemberChecks.ReadsBack`); that the
session has messages; and the template's count of rounds. `FormSetup.lean` takes these up and proves
`SessionCoins.VerdictsAt` from them (`SessionCoins.verdictsAt_of`).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)

private theorem throw_ne {α : Type} (w : String) (a : α) : (throw w : Flock.V α) ≠ .ok a := fun h => by cases h

private theorem mapError_ok {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error w => cases h
  | ok a' => cases h; rfl

/-! ## The statements paired with their members, by position -/

/-- **A statement and the member it names**: the first of that name among the parameters' `ms`, the statement of one
session. -/
def Paired (ms : Array Spot.OneStageZk.Member) (st : Spot.OneStageZk.StatementFiles)
    (m : Spot.OneStageZk.Member) : Prop :=
  ∃ name, st.member = some name ∧ ms.find? (·.name == name) = some m ∧ st.sessions.size = 1

/-- `heldStep` adds the member its statement names, a statement of one session. -/
theorem heldStep_paired {ms held h' : Array Spot.OneStageZk.Member} {st : Spot.OneStageZk.StatementFiles}
    (h : Spot.OneStageZk.heldStep ms held st = .ok h') : ∃ m, Paired ms st m ∧ h' = held.push m := by
  unfold Spot.OneStageZk.heldStep at h
  split at h
  · exact absurd h (throw_ne _ _)
  next name hn =>
    split at h
    · exact absurd h (throw_ne _ _)
    next m hm =>
      split at h
      · exact absurd h (throw_ne _ _)
      · split at h
        · exact absurd h (throw_ne _ _)
        · split at h
          · exact absurd h (throw_ne _ _)
          next hsz =>
            exact ⟨m, ⟨name, hn, hm, bne_eq_false_iff_eq.1 hsz⟩, (Except.ok.inj h).symm⟩

theorem foldlM_paired {ms : Array Spot.OneStageZk.Member} :
    ∀ (l : List Spot.OneStageZk.StatementFiles) (acc out : Array Spot.OneStageZk.Member),
      l.foldlM (Spot.OneStageZk.heldStep ms) acc = .ok out →
        ∃ ms', out.toList = acc.toList ++ ms' ∧ List.Forall₂ (Paired ms) l ms'
  | [], acc, out, h => by
    rw [List.foldlM_nil] at h
    obtain rfl : acc = out := Except.ok.inj h
    exact ⟨[], (List.append_nil _).symm, .nil⟩
  | st :: l, acc, out, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨a, ha, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
    obtain ⟨m, hp, rfl⟩ := heldStep_paired ha
    obtain ⟨ms', hout, hF⟩ := foldlM_paired l _ out h
    refine ⟨m :: ms', ?_, .cons hp hF⟩
    rw [hout, Array.toList_push, List.append_assoc]
    rfl

/-- **`membersHeld` pairs the statements with their members in order**: each statement with the member it names. -/
theorem membersHeld_paired {ms held : Array Spot.OneStageZk.Member} {sts : Array Spot.OneStageZk.StatementFiles}
    (h : Spot.OneStageZk.membersHeld ms sts = .ok held) : List.Forall₂ (Paired ms) sts.toList held.toList := by
  unfold Spot.OneStageZk.membersHeld at h
  rw [← Array.foldlM_toList] at h
  obtain ⟨ms', hout, hF⟩ := foldlM_paired sts.toList #[] held h
  have e : held.toList = ms' := by
    rw [hout]
    rfl
  rw [e]
  exact hF

/-- Where each of a list's entries gives one element, the `t`-th of them all is the `t`-th entry's. -/
theorem getElem?_flatMap_one {α β : Type} {f : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).length = 1) → ∀ t : ℕ,
      (l.flatMap f)[t]? = l[t]?.bind fun a => (f a)[0]?
  | [], _, t => by simp
  | a :: l, h, t => by
    obtain ⟨b, hb⟩ := List.length_eq_one_iff.1 (h a (List.mem_cons_self ..))
    rw [List.flatMap_cons, hb]
    cases t with
    | zero => simp [hb]
    | succ t =>
      have ih := getElem?_flatMap_one (f := f) (l := l) (fun a' ha' => h a' (List.mem_cons_of_mem _ ha')) t
      simpa using ih

/-- **Template `t`'s session, its statement and its member are a pair the check ran**: where the statements'
structure held (`Spot.OneStageZk.membersHeld`), the `t`-th of the statements' sessions is the one session of the
`t`-th statement, and `SessionMsgs.memberAt` is the member the check held that statement to. -/
theorem paired_at {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {held : Array Spot.OneStageZk.Member}
    (h : Spot.OneStageZk.membersHeld p.members e.statements = .ok held) {t : ℕ} {fs : Spot.OneStageZk.RunFiles}
    (hs : SessionMsgs.filesAt e t = some fs) :
    ∃ st m, SessionMsgs.stmtFilesAt e t = some st ∧ SessionMsgs.memberAt p e t = some m ∧
      (st, m) ∈ (e.statements.zip held).toList ∧ fs ∈ st.sessions.toList := by
  have hF := membersHeld_paired h
  have h1 : ∀ st ∈ e.statements.toList, st.sessions.toList.length = 1 := by
    intro st hst
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hst
    obtain ⟨m, -, -, -, -, hsz⟩ := forall₂_getElem? hF (List.getElem?_eq_getElem hi)
    simpa using hsz
  unfold SessionMsgs.filesAt at hs
  rw [getElem?_flatMap_one h1 t] at hs
  obtain ⟨st, hst, hfs⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨m, hm, name, hn, hfind, hsz⟩ := forall₂_getElem? hF hst
  have hstmt : SessionMsgs.stmtFilesAt e t = some st := by
    unfold SessionMsgs.stmtFilesAt
    rw [getElem?_flatMap_one (f := fun st : Spot.OneStageZk.StatementFiles => st.sessions.toList.map fun _ => st)
      (fun a ha => by simpa using h1 a ha) t, hst]
    obtain ⟨b, hb⟩ := List.length_eq_one_iff.1 (h1 st (List.mem_of_getElem? hst))
    rw [Option.bind_some, hb]
    rfl
  refine ⟨st, m, hstmt, ?_, ?_, List.mem_of_getElem? hfs⟩
  · unfold SessionMsgs.memberAt
    rw [hstmt]
    simp [hn, hfind]
  · rw [Array.toList_zip]
    refine List.mem_of_getElem? (i := t) ?_
    rw [List.getElem?_zip_eq_some]
    exact ⟨hst, hm⟩

/-! ## Each statement's sessions, accepted at its member's statement -/

/-- **One statement's step, with its member's statement kept**: where the run still accepts every session after
the step, the run before it accepted every session, the member's statement reads at its slice, and each of the
statement's sessions was accepted at it (`MemberChecks.SessionAccepted`), where the statement `st` the step reads is
the member's own at its slice (`hst`). -/
theorem stepMember_stmt {slice : Spot.OneStageZk.Member → Json}
    {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs} {r r' : Spot.OneStageZk.Run}
    {z : Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member}
    (hst : ∀ a, st z.2 = .ok a → z.2.statement (slice z.2) bs = .ok a)
    (h : Spot.OneStageZk.stepMember st bs r z = .ok r') (ha : r'.all = true) :
    r.all = true ∧ ∃ a, z.2.statement (slice z.2) bs = .ok a ∧
      ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted a bs s := by
  unfold Spot.OneStageZk.stepMember at h
  have h := mapError_ok h
  unfold Spot.OneStageZk.runFrom at h
  obtain ⟨a, ha', h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  rw [← Array.foldlM_toList] at h
  obtain ⟨har, hs⟩ := MemberChecks.foldl_all _ r r' h ha
  exact ⟨har, a, hst a ha', hs⟩

theorem foldl_stmt {slice : Spot.OneStageZk.Member → Json}
    {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs} :
    ∀ (l : List (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)) (r r' : Spot.OneStageZk.Run),
      (∀ z ∈ l, ∀ a, st z.2 = .ok a → z.2.statement (slice z.2) bs = .ok a) →
      l.foldlM (Spot.OneStageZk.stepMember st bs) r = .ok r' → r'.all = true →
        r.all = true ∧ ∀ z ∈ l, ∃ a, z.2.statement (slice z.2) bs = .ok a ∧
          ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted a bs s
  | [], r, r', _, h, ha => by
    obtain rfl : r = r' := Except.ok.inj h
    exact ⟨ha, fun _ hz => by simp at hz⟩
  | z :: l, r, r', hst, h, ha => by
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
    obtain ⟨ha₁, hl⟩ := foldl_stmt l r₁ r' (fun z' hz' => hst z' (List.mem_cons_of_mem _ hz')) h ha
    obtain ⟨har, hz⟩ := stepMember_stmt (hst z (by simp)) h₁ ha₁
    refine ⟨har, fun z' hz' => ?_⟩
    rcases List.mem_cons.1 hz' with rfl | hz'
    · exact hz
    · exact hl z' hz'

/-- **The per-member form's run, accepting every session, with each member's statement kept**: where `runMembers`
returns a run that accepts every session, the statements' structure held, and for each statement paired with its
member, in order, the member's statement reads at its slice and each of the statement's sessions was accepted at
it (`MemberChecks.SessionAccepted`). -/
theorem runMembers_stmt {slice : Spot.OneStageZk.Member → Json} {ms : Array Spot.OneStageZk.Member}
    {sts : Array Spot.OneStageZk.StatementFiles} {want : Array ℕ} {bs : Core.Blobs} {r : Spot.OneStageZk.Run}
    (h : Spot.OneStageZk.runMembers slice ms sts want bs = .ok r) (ha : r.all = true) :
    ∃ held, Spot.OneStageZk.membersHeld ms sts = .ok held ∧ ∀ z ∈ (sts.zip held).toList,
      ∃ a, z.2.statement (slice z.2) bs = .ok a ∧
        ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted a bs s := by
  unfold Spot.OneStageZk.runMembers at h
  obtain ⟨held, hheld, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨files, hfiles, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨r₁, hfold, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨_, -, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  have e : r₁ = r := Except.ok.inj h
  rw [e] at hfold
  rw [← Array.foldlM_toList] at hfold
  have hst : ∀ z ∈ (sts.zip held).toList, ∀ a, Spot.OneStageZk.statementAt files z.2 = .ok a →
      z.2.statement (slice z.2) bs = .ok a := by
    rintro ⟨sf, m⟩ hz a hza
    rw [Array.toList_zip] at hz
    have hm : m ∈ held := Array.mem_def.2 (List.of_mem_zip hz).2
    exact MemberChecks.statementAt_statement hfiles (MemberChecks.membersHeld_find hheld m hm) hza
  exact ⟨held, hheld,
    (foldl_stmt (slice := slice) (st := Spot.OneStageZk.statementAt files) _ _ _ hst hfold ha).2⟩

/-! ## The verdict at template `t` -/

/-- **The verdict the check gave template `t`'s session**: on a play the form accepts, at the `t`-th of the
statements' sessions `fs`, the member `SessionMsgs.memberAt` names has its circuit and public files on the blobs and
its registered reads read, the session's record is a text on the blobs, the form set a session up
(`Spot.OneStageZk.setupAt`) on those files at the record's sample, and Flock's masked verdict accepts the record's
text with the proof files `fs`'s reps name at that session (`Spot.OneStageZk.filesOf`). The member's statement
passed, so the circuit statement's leaves are not plain (`MemberChecks.statement_ok`): no computation settles that,
and the statement's setup reads it. -/
theorem verified_at {C : CaseOf} {o : Verity.Exchange} (hacc : FormAccepts C o) {k : Case} (hk : C o = some k)
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    (hf : SessionMsgs.formOf k = some (p, e)) {t : ℕ} {fs : Spot.OneStageZk.RunFiles}
    (hs : SessionMsgs.filesAt e t = some fs) :
    ∃ (m : Spot.OneStageZk.Member) (circuitB pubB : ByteArray) (rd : Spot.OneStageZk.Reads) (record : String)
      (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json) (proofs : Array ByteArray),
      Flock.Tags.circuit.plainLeaves = false ∧ SessionMsgs.memberAt p e t = some m ∧
        k.blobs.get m.own.circuit "circuit" = .ok circuitB ∧
        k.blobs.get m.own.pub "public file" = .ok pubB ∧ Spot.OneStageZk.readsOf m.own.registered = .ok rd ∧
        SessionMsgs.recordAt k.blobs fs = some record ∧
        Spot.OneStageZk.setupAt (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB k.blobs) rd
          (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest) ∧
        (Spot.OneStageZk.filesOf k.blobs fs (.ok (spec, #[], Json.null, Json.null))).toOption = some proofs ∧
        Flock.Zk.verify spec tabs record proofs = .ok () := by
  obtain ⟨k', hk', pp, c, e', pub, prof, hdec, hck⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  have hpe : (pp, e') = (p, e) := Option.some.inj ((SessionMsgs.formOf_of_decoded hdec).symm.trans hf)
  obtain rfl : p = pp := (congrArg Prod.fst hpe).symm
  obtain rfl : e = e' := (congrArg Prod.snd hpe).symm
  obtain ⟨-, c₂, ξ₂, r, i, run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨-, -, hrun, hall, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain ⟨held, ts, L, roots, hheld, -, -, -, -, hrm⟩ := membersRun_ran hrun
  obtain ⟨held', hheld', hz⟩ := runMembers_stmt hrm hall
  obtain rfl : held' = held := Except.ok.inj (hheld'.symm.trans hheld)
  obtain ⟨st, m, -, hm, hmem, hfs⟩ := paired_at hheld hs
  obtain ⟨a, ha, hsess⟩ := hz (st, m) hmem
  obtain ⟨circuitB, pubB, rd, hpl, hc, hp, hrd, rfl⟩ := MemberChecks.statement_ok ha
  obtain ⟨coinKeys, b, record, hb, hu, hv⟩ := hsess fs hfs
  dsimp only [Spot.OneStageZk.StepAt.ofOwn] at hv
  obtain ⟨proofs, spec, tabs, rest, hfiles, hst, hzk⟩ := Verity.Sampled.verdictOf_ok hv
  refine ⟨m, circuitB, pubB, rd, record, spec, tabs, rest, proofs, hpl, hm, hc, hp, hrd, ?_, hst, ?_, hzk⟩
  · have hb' : k.blobs.get fs.record "record" = .ok b := Verity.Sampled.Record.get_what hb
    unfold SessionMsgs.recordAt
    rw [hb']
    exact hu
  · rw [hst] at hfiles
    have hfiles' : Spot.OneStageZk.filesOf k.blobs fs (.ok (spec, #[], Json.null, Json.null)) = .ok proofs :=
      hfiles
    rw [hfiles']
    rfl

/-- **Template `t`'s session statement carries the files the check verified its session on**: where the template's
member, its session's record's text and the member's circuit and public files on the blobs are read,
`SessionMsgs.stmtAtCase` is the member's statement (`SessionMsgs.stmtRaw`) at those files, the blobs' lookup tables,
the record's sample and the member's registered reads. -/
theorem stmtAtCase_of_member {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} (hf : SessionMsgs.formOf k = some (p, e)) {t : ℕ}
    {m : Spot.OneStageZk.Member} (hm : SessionMsgs.memberAt p e t = some m) {fs : Spot.OneStageZk.RunFiles}
    (hs : SessionMsgs.filesAt e t = some fs) {record : String} (hr : SessionMsgs.recordAt k.blobs fs = some record)
    {circuitB pubB : ByteArray} (hc : k.blobs.get m.own.circuit "circuit" = .ok circuitB)
    (hp : k.blobs.get m.own.pub "public file" = .ok pubB) :
    SessionMsgs.stmtAtCase k t =
      some (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).encode := by
  unfold SessionMsgs.stmtAtCase
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq]
  exact ⟨(p, e), hf, m, hm, fs, hs, record, hr, circuitB, congrArg Except.toOption hc, pubB,
    congrArg Except.toOption hp, rfl⟩

end Verity.SampledTwo.SessionCoins
