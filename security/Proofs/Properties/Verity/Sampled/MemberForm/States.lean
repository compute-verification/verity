module

public import Proofs.Properties.Verity.Sampled.MemberForm.Accept
public import Proofs.Properties.Verity.Sampled.MemberForm.SetsUp
public import Proofs.Properties.Verity.Sampled.Window.Writes
public import Proofs.Properties.Spot.Tables

/-!
# A member's statement of the per-member form is its window's member's

The per-member form reads one statement for each member of the parameters, off the blobs and before any session
(`Spot.OneStageZk.Member.statement`), and judges the member's session at it. A window's member is that member on
the registered files (`MemberOn`). This module ties the two, so that a session's verdict at the statement the form
read is flock-verify `--zk`'s at the window's member (`SessionVerdict.accepts_member`):

* **What the statement read** (`statementOf_reads`, `statement_reads`): the statement's tags, the blobs' circuit and
  public files under the member's two digests, the circuit file's lookups the pinned tables the blobs hold, and the
  inputs `--zk` is handed at them (`Spot.OneStageZk.inputsOf`).
* **v2 reads a member's files among some files** (`ReadsFiles`): what v2 reads on the blobs under the member's two
  digests are the files those digests name among those files.
* **The statement is the window's member's** (`statesMember`): on blobs where v2 reads the member's files among
  the files the window's member is on, and each pinned table the blobs hold is its registered file (`TablesSent`),
  the statement the form read for a member is `StatesMember` of the window's member.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-- The digest a hex digest names is its decoded bytes'. -/
theorem digestOf_unhex (s : String) (dg : ByteArray) (h : Flock.unhex s = .ok dg) :
    digestOf s = some fun i => dg.data.getD i.val 0 := by
  simp only [digestOf, h]

/-- **What `statementOf` read**: the statement's tags, the blobs' circuit and public files under the inputs' two
digests, the circuit file's lookups the pinned tables the blobs hold, and `--zk`'s inputs at them. -/
theorem statementOf_reads {p : _root_.Spot.OneStageZk.Own} {population : ByteArray → Flock.V Unit}
    {bs : _root_.Core.Blobs} {a : _root_.Spot.OneStageZk.StepAt}
    (h : _root_.Spot.OneStageZk.statementOf p population bs = .ok a) :
    ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray),
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.lookupsOf circuitB = .ok ((_root_.Spot.OneStageZk.tablesOf bs).map (·.1)) ∧
      a.I = _root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs := by
  unfold _root_.Spot.OneStageZk.statementOf at h
  obtain ⟨⟨⟩, htab, h⟩ := ebind_ok h
  obtain ⟨circuitB', hc', hl⟩ := Properties.Spot.SecurityProofs.tablesHeld_ok htab
  split at h
  · cases h
  · rename_i tags0 htags
    split at h
    · cases h
    · split at h
      · cases h
      · obtain ⟨circuitB, hc, h⟩ := ebind_ok h
        obtain ⟨pubB, hp, h⟩ := ebind_ok h
        split at h
        · cases h
        · obtain ⟨rd, -, h⟩ := ebind_ok h
          have hcc : circuitB' = circuitB := Except.ok.inj (hc'.symm.trans hc)
          subst hcc
          have ha : _root_.Spot.OneStageZk.StepAt.ofOwn p tags0 circuitB' pubB bs rd (population pubB) = a :=
            Except.ok.inj h
          subst ha
          exact ⟨tags0, circuitB', pubB, htags, hc, hp, hl, rfl⟩

/-- **What a member's statement read** (`statementOf_reads`, at the member's own inputs). -/
theorem statement_reads {vm : _root_.Spot.OneStageZk.Member} {slice : Json} {bs : _root_.Core.Blobs}
    {a : _root_.Spot.OneStageZk.StepAt} (h : vm.statement slice bs = .ok a) :
    ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray),
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get vm.own.circuit "circuit" = .ok circuitB ∧ bs.get vm.own.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.lookupsOf circuitB = .ok ((_root_.Spot.OneStageZk.tablesOf bs).map (·.1)) ∧
      a.I = _root_.Spot.OneStageZk.inputsOf tags0 vm.own circuitB pubB bs := by
  unfold _root_.Spot.OneStageZk.Member.statement at h
  obtain ⟨a', hs, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  have ha : a' = a := Except.ok.inj h
  subst ha
  exact statementOf_reads hs

/-- **v2 reads the member `vm`'s files among `R` on the blobs `bs`**: the circuit file and the public file it reads
there under the member's two digests are the files those digests name among `R` (`Verity.namedFile`). -/
def ReadsFiles (R : Array ByteArray) (vm : _root_.Spot.OneStageZk.Member) (bs : _root_.Core.Blobs) : Prop :=
  (∀ b, bs.get vm.own.circuit "circuit" = .ok b → b = Verity.namedFile (digestOf vm.own.circuit) R) ∧
    ∀ b, bs.get vm.own.pub "public file" = .ok b → b = Verity.namedFile (digestOf vm.own.pub) R

/-- **The statement the per-member form read for a member is the window's member's**: on the blobs sent `s`, where
v2 reads the member's files among the files `R` and each pinned table the blobs hold is its registered file, the
statement read for the parameters' member `vm` is `StatesMember` of the window's member `m` that is `vm` on `R`. -/
theorem statesMember {R : Array ByteArray} {s : Verity.Sent} (ht : TablesSent s)
    {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ} (hon : MemberOn R vm m)
    (hr : ReadsFiles R vm (_root_.Core.Blobs.ofArray s.blobs)) {slice : Json}
    {a : _root_.Spot.OneStageZk.StepAt}
    (ha : vm.statement slice (_root_.Core.Blobs.ofArray s.blobs) = .ok a) : StatesMember (stepAt a) m := by
  obtain ⟨tags0, circuitB, pubB, htags, hgc, hgp, hl, hI⟩ := statement_reads ha
  have hcf : circuitB = m.I.circuitFile := by
    rw [hon.circuit]
    exact hr.1 _ hgc
  have hpf : pubB = m.I.publicFile := by
    rw [hon.publicFile]
    exact hr.2 _ hgp
  have hI' : (stepAt a).I = _root_.Spot.OneStageZk.inputsOf tags0 vm.own circuitB pubB
      (_root_.Core.Blobs.ofArray s.blobs) := hI
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, hon.work, hon.stratified, hon.closure, hon.own, ?_⟩
  · rw [hI']
    exact Option.some.inj (htags.symm.trans hon.tags)
  · rw [hI']
    exact hcf
  · rw [hI']
    exact hpf
  · rw [hI']
    exact hon.count.symm
  · rw [hI']
    exact hon.partition.symm
  · rw [hI']
    exact hon.program.symm
  · rw [hI', hon.tables, ← hcf]
    exact pinnedTables_extend hl rfl ht

end Verity.Sampled.MemberForm
