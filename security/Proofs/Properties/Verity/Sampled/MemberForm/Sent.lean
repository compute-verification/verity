module

public import Proofs.Properties.Verity.Sampled.MemberForm.Read
public import Proofs.Properties.Verity.Sampled.MemberForm.Sessions
public import Proofs.Properties.Verity.Sampled.Window.Steps
public import Proofs.Properties.Verity.Sampled.SpecTie

/-!
# Each session of the per-member form in what the prover sent

A window's sessions (`Verity.Sampled.WindowSessions`) name, for the `u`-th session of a play, the record sent for it
(`sentRecordAt`), the bytes its closing round is opened after the SHA-512 of, and what the verifier's read holds the
device's log to at it. This module reads those off what the per-member form's check names its sessions
(`NamedAt`, `NamedOf`), one statement at a time:

* **The evidence sent for a session** (`sessionEvidenceAt`, `sentEvidenceAt`): the bytes the `u`-th session's closing
  round is opened after the SHA-512 of, as the verifier reads them on the blobs sent
  (`Spot.OneStageZk.sessionEvidence`); and the one table every statement's session is of (`statementTable`).
* **The session at a place** (`sessions_at`, `named_at`): at statements of one session each, the `j`-th session of
  the evidence is the `j`-th statement's one session, and the check names it at the `j`-th place.
* **Its record and evidence in what was sent** (`sentRecordAt_session`, `sentEvidenceAt_session`).
* **Its tables** (`ownRootBTables_one`): one table, the statement's.
* **Its coin tree** (`Member.spec_coinTree`, `Member.coinSpec_one`): the session `--zk` sets up at a member of one
  table is on the coin tree of that table.
* **The order on the device's log** (`opensLast_congr`): a log of the same frames as one whose openings of a tree
  opened before are the last tree's has them so too.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge

/-- **The session `--zk` sets up at a member of one table is on the member's coin tree**, for any record
(`SetsUp.spec_coinTree`, at a member of a window). -/
theorem Member.spec_coinTree {nJ : ℕ} (m : Member nJ) (hc : m.I.count = 1) {record : String}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts)) :
    spec.coinTree = some m.coinSpec := by
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs m.I).trans m.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · exact (setupHidden_coinTreeP hst).trans rfl
  · exact absurd h1 (by show ¬ 1 < m.I.count; omega)

/-- A member of one table's coin tree is the coin tree of that table. -/
theorem Member.coinSpec_one {nJ : ℕ} (m : Member nJ) (hc : m.I.count = 1) :
    m.coinSpec = Flock.Zk.coinSpec #[m.I.tags.table] := by
  unfold Member.coinSpec
  rw [hc]
  rfl

/-- **A log of the same frames has its openings of a tree opened before as the last tree's too**: `OpensLast`
reads only each release's frame. -/
theorem opensLast_congr {l l' : List Verity.Release}
    (h : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l') (hO : OpensLast l) :
    OpensLast l' := by
  intro L₁' x' L₂' i s n he hx u hu
  have h' : List.Forall₂ (fun x y : Verity.Release => x.frame = y.frame ∧ x.answers = y.answers) l
      (L₁' ++ x' :: L₂') := by
    rw [← he]
    exact h
  obtain ⟨L₁, x, L₂, hl, hp1, hxx, -⟩ := forall₂_split h'
  have e := MemberForm.openedTrees_congr hp1
  rw [← e] at hu ⊢
  exact hO L₁ x L₂ i s n hl (hxx.1.trans hx) u hu

namespace MemberForm

/-! ## The evidence sent for a session -/

/-- **The `u`-th session's evidence as the verifier reads it** on the blobs `bs`: the bytes the evidence's `u`-th
session's closing round is opened after the SHA-512 of (`Spot.OneStageZk.sessionEvidence`). None where the evidence
has no such session or the verifier refuses its evidence. -/
def sessionEvidenceAt (u : ℕ) (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) :
    Option Claims.Msg :=
  e.sessions[u]?.bind fun s => (_root_.Spot.OneStageZk.sessionEvidence bs s).toOption.map fun b => b.data.toList

/-- **The `u`-th session's evidence the verifier reads in the play that ended with `x`**, on the blobs sent
(`sentRecordAt`, for the evidence). -/
def sentEvidenceAt (u : ℕ) (x : Verity.Exchange) : Option Claims.Msg :=
  (caseOf x.sent).toOption.bind fun y => sessionEvidenceAt u y.2.2 (_root_.Core.Blobs.ofArray x.sent.blobs)

/-- **The one table of a statement's session**: the table of the tags of the statement every member's session is
of (`Spot.OneStageZk.STATEMENT`), or the empty string where the statement is unknown. -/
def statementTable : String :=
  ((Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).map fun t => t.table).getD ""

/-- The statement's table is the table of the statement's tags. -/
theorem statementTable_eq {tags0 : Flock.Tags}
    (ht : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0) : statementTable = tags0.table := by
  simp only [statementTable, ht, Option.map_some, Option.getD_some]

/-! ## The session at a place -/

/-- **The session at a place among statements of one session each**: where the check names the statements `zs`'
sessions `l`, one a statement (`NamedAt`), the statement at place `j` has one session, the `j`-th of the statements'
sessions in order, and the check names it at place `j`. -/
theorem sessions_at {bs : _root_.Core.Blobs} :
    ∀ {zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)}
      {l : List (String × Array (String × Option ℕ) × Json × String)}, List.Forall₂ (NamedAt bs) zs l →
      ∀ {j : ℕ} {z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member}, zs[j]? = some z →
        ∃ s x, z.1.sessions = #[s] ∧ ((zs.map (·.1)).flatMap (·.sessions.toList))[j]? = some s ∧
          l[j]? = some x ∧ NamedOf bs (z.2, s) x
  | _, _, .nil, j, z, h => by
    rw [List.getElem?_nil] at h
    cases h
  | _, _, .cons (a := z₀) (b := x₀) ⟨s₀, hs₀, hn₀⟩ _, 0, z, h => by
    rw [List.getElem?_cons_zero] at h
    obtain rfl : z₀ = z := Option.some.inj h
    refine ⟨s₀, x₀, hs₀, ?_, List.getElem?_cons_zero, hn₀⟩
    rw [List.map_cons, List.flatMap_cons, hs₀, List.toList_toArray, List.singleton_append,
      List.getElem?_cons_zero]
  | _, _, .cons (a := z₀) ⟨s₀, hs₀, _⟩ t, j + 1, z, h => by
    rw [List.getElem?_cons_succ] at h
    obtain ⟨s, x, hs, hf, hl, hn⟩ := sessions_at t h
    refine ⟨s, x, hs, ?_, ?_, hn⟩
    · rw [List.map_cons, List.flatMap_cons, hs₀, List.toList_toArray, List.singleton_append,
        List.getElem?_cons_succ]
      exact hf
    · rw [List.getElem?_cons_succ]
      exact hl

/-- **The `j`-th statement's session is the evidence's `j`-th, named at place `j`**: at as many statements as
members, where the check names the statements' sessions `named`, one a statement, and the `j`-th statement `st`,
with its member `vm`, has the one session `s`, the statements' sessions in order have `s` at place `j`, and the
check names it there. -/
theorem named_at {bs : _root_.Core.Blobs} {sts : Array _root_.Spot.OneStageZk.StatementFiles}
    {held : Array _root_.Spot.OneStageZk.Member}
    {named : Array (String × Array (String × Option ℕ) × Json × String)} (hsz : sts.size = held.size)
    (hn : List.Forall₂ (NamedAt bs) (sts.zip held).toList named.toList) {j : ℕ}
    {st : _root_.Spot.OneStageZk.StatementFiles} {vm : _root_.Spot.OneStageZk.Member}
    (hz : (sts.zip held).toList[j]? = some (st, vm)) {s : _root_.Spot.OneStageZk.RunFiles}
    (hs : st.sessions = #[s]) :
    (sts.flatMap (·.sessions))[j]? = some s ∧ ∃ x, named[j]? = some x ∧ NamedOf bs (vm, s) x := by
  obtain ⟨s', x, hs', hf, hl, hno⟩ := sessions_at hn hz
  have e : s' = s := by
    have h1 : #[s'] = #[s] := hs'.symm.trans hs
    simpa using h1
  subst e
  have hfst : (sts.zip held).toList.map (·.1) = sts.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by rw [Array.length_toList, Array.length_toList]; exact Nat.le_of_eq hsz)
  refine ⟨?_, x, ?_, hno⟩
  · rw [← Array.getElem?_toList, Array.toList_flatMap, ← hfst]
    exact hf
  · rw [← Array.getElem?_toList]
    exact hl

/-! ## The session's record and evidence in what was sent -/

/-- **The record sent for the `u`-th session is its record on the blobs**: where v2's case in what was sent decodes
to the evidence `e`, whose `u`-th session is `s`, and the blob `s` names as its record is the text `record`. -/
theorem sentRecordAt_session {S : Verity.Sent} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence} (hcs : caseOf S = .ok (p, c, e))
    {u : ℕ} {s : _root_.Spot.OneStageZk.RunFiles} (hs : e.sessions[u]? = some s) {B : ByteArray}
    {record : String}
    (hb : (_root_.Core.Blobs.ofArray S.blobs).get s.record s!"session {s.name}'s record" = .ok B)
    (hr : String.fromUTF8? B = some record) : sentRecordAt u S = some record := by
  unfold sentRecordAt
  rw [hcs]
  show (e.sessions[u]?.bind fun s =>
    ((_root_.Core.Blobs.ofArray S.blobs).get s.record s!"session {s.name}'s record").toOption.bind
      String.fromUTF8?) = _
  rw [hs]
  show (((_root_.Core.Blobs.ofArray S.blobs).get s.record s!"session {s.name}'s record").toOption.bind
    String.fromUTF8?) = _
  rw [hb]
  exact hr

/-- **The evidence sent for the `u`-th session is its evidence on the blobs**: where v2's case in what was sent
decodes to the evidence `e`, whose `u`-th session is `s`, and the verifier reads `s`'s evidence as `evidence`. -/
theorem sentEvidenceAt_session {x : Verity.Exchange} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    (hcs : caseOf x.sent = .ok (p, c, e)) {u : ℕ} {s : _root_.Spot.OneStageZk.RunFiles}
    (hs : e.sessions[u]? = some s) {evidence : ByteArray}
    (he : _root_.Spot.OneStageZk.sessionEvidence (_root_.Core.Blobs.ofArray x.sent.blobs) s = .ok evidence) :
    sentEvidenceAt u x = some evidence.data.toList := by
  unfold sentEvidenceAt
  rw [hcs]
  show (e.sessions[u]?.bind fun s =>
    (_root_.Spot.OneStageZk.sessionEvidence (_root_.Core.Blobs.ofArray x.sent.blobs) s).toOption.map
      fun b => b.data.toList) = _
  rw [hs]
  show ((_root_.Spot.OneStageZk.sessionEvidence (_root_.Core.Blobs.ofArray x.sent.blobs) s).toOption.map
    fun b => b.data.toList) = _
  rw [he]
  rfl

/-! ## The session's tables -/

/-- **A member's session's tables with their `root_B` lengths are one table's, the statement's**: `ownRootBTables`
pairs each of the session's tables (`Spot.OneStageZk.sessionTables`, the statement's one) with a length. -/
theorem ownRootBTables_one {tags0 : Flock.Tags}
    (ht : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0) (p : _root_.Spot.OneStageZk.Own)
    (bs : _root_.Core.Blobs) (record : String) :
    ∃ n?, _root_.Spot.OneStageZk.ownRootBTables p bs record = #[(tags0.table, n?)] := by
  have htab : _root_.Spot.OneStageZk.sessionTables = #[tags0.table] := by
    simp only [_root_.Spot.OneStageZk.sessionTables, ht, Option.map_some, Option.getD_some]
    rfl
  unfold _root_.Spot.OneStageZk.ownRootBTables _root_.Spot.OneStageZk.rootBTablesAt
  simp only [htab, Array.mapIdx_singleton]
  exact ⟨_, rfl⟩

end MemberForm

end Verity.Sampled
