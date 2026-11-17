module

public import Proofs.Properties.Verity.Sampled.MemberForm.Read
public import Proofs.Properties.Verity.Sampled.MemberForm.Sessions
public import Proofs.Properties.Verity.Sampled.Window.Steps
public import Proofs.Properties.Verity.Sampled.SpecTie

/-! Each session of the per-member form in what the prover sent. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge

theorem Member.spec_coinTree {nJ : ℕ} (m : Member nJ) (hc : m.I.count = 1) {record : String}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts)) :
    spec.coinTree = some m.coinSpec := by
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs m.I).trans m.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · exact (setupHidden_coinTreeP hst).trans rfl
  · exact absurd h1 (by show ¬ 1 < m.I.count; omega)

theorem Member.coinSpec_one {nJ : ℕ} (m : Member nJ) (hc : m.I.count = 1) :
    m.coinSpec = Flock.Zk.coinSpec #[m.I.tags.table] := by
  unfold Member.coinSpec
  rw [hc]
  rfl

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

def sessionEvidenceAt (u : ℕ) (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) :
    Option Claims.Msg :=
  e.sessions[u]?.bind fun s => (_root_.Spot.OneStageZk.sessionEvidence bs s).toOption.map fun b => b.data.toList

def sentEvidenceAt (u : ℕ) (x : Verity.Exchange) : Option Claims.Msg :=
  (caseOf x.sent).toOption.bind fun y => sessionEvidenceAt u y.2.2 (_root_.Core.Blobs.ofArray x.sent.blobs)

def statementTable : String :=
  ((Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).map fun t => t.table).getD ""

theorem statementTable_eq {tags0 : Flock.Tags}
    (ht : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0) : statementTable = tags0.table := by
  simp only [statementTable, ht, Option.map_some, Option.getD_some]

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
