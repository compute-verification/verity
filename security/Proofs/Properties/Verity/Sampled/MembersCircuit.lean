module

public import Proofs.Properties.Verity.Sampled.Record.Session
public import Spot.OneStageZk

/-!
# A member's session checks the circuit its statement parses

At an accept of the per-member check (`Spot.OneStageZk.checkMembersAt`), each member's statement (`Member.statement`
at its slice of the registration) has inputs `a.I` whose circuit `Flock.ProvedScope.circuitOf` parses at `--zk`'s tags,
and every table of every session set up at those inputs (`sessionAt`, which `setupAt` and so each session's `step` runs
at `a.I`) has that circuit as its `Stmt.c`: the statement is one table's (`inputsAt`), its files pass
`Flock.KsScope.filesOf`, whose tags have hidden outputs, and a one-table session with hidden outputs sets up on the
parse it is handed (`Record.sessionAt_circuitOf`).
-/

@[expose] public section

namespace Verity.Sampled.MembersCircuit

open Lean (Json)
open Flock (V Stmt SessionSpec)
open Spot.OneStageZk

/-- A member's statement's inputs are one table's (`inputsAt`). -/
theorem statement_count {m : Member} {slice : Json} {bs : Core.Blobs} {a : StepAt}
    (h : m.statement slice bs = .ok a) : a.I.count = 1 := by
  unfold Member.statement at h
  obtain ⟨a', hs, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
  obtain rfl : a' = a := Except.ok.inj h
  obtain ⟨tables, -, hs⟩ := Flock.HmNets.bind_ok hs
  obtain ⟨tags0, circuitB, pubB, rd, rfl⟩ := statementOf_ofOwn hs
  rfl

/-- The tags whose files `Flock.KsScope.filesOf` accepts have hidden outputs (`Flock.KsScope.tagsWhy_none`). -/
theorem filesOf_hidden {parsed : V Flock.Circuit} {tags : Flock.Tags} {pf : ByteArray} {count : Nat}
    (h : Flock.KsScope.filesOf parsed tags pf count = .ok ()) : tags.hiddenOutputs = true := by
  unfold Flock.KsScope.filesOf at h
  split at h
  · cases h
  · exact (Flock.KsScope.tagsWhy_none ‹Flock.KsScope.tagsWhy tags = none›).2.1

/-- **At an accept of the per-member check, every session of a member checks the circuit its statement parses**: at the
roots `membersRootsOf` gives, each member's statement `a` (`Member.statement` at its slice) has, on every table of every
session `sessionAt` sets up at `a.I`, whatever its registered reads and sample, `Stmt.c` the circuit
`Flock.ProvedScope.circuitOf` parses at `a.I`'s `--zk` tags. -/
theorem session_circuitOf {p : Params} {ms : Array Member} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} {pub : Spot.OneStage.Public}
    {prof : Core.Integrity} (h : checkMembersAt p ms c ξ e sts bs = .accept pub prof) :
    ∃ roots, membersRootsOf p ms c.record = .ok roots ∧ ∀ m ∈ ms, ∀ a,
      m.statement (m.slice c.record roots) bs = .ok a →
        ∀ (rd : Reads) (draw : Option Json) (spec : SessionSpec) (tabs : Array Stmt),
          sessionAt a.I rd draw = .ok (spec, tabs) →
            ∀ st ∈ tabs, Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables = .ok st.c := by
  obtain ⟨r, hr, -, -⟩ := checkMembersAt_accept h
  obtain ⟨roots, hroots, hf⟩ := membersRun_files hr
  refine ⟨roots, hroots, fun m hm a ha rd draw spec tabs hsa => ?_⟩
  obtain ⟨a', hs, -, -, hks, -⟩ := hf m hm
  obtain rfl : a' = a := Except.ok.inj (hs.symm.trans ha)
  exact Record.sessionAt_circuitOf hsa (filesOf_hidden hks) (statement_count hs).le

end Verity.Sampled.MembersCircuit
