module

public import Flock.Canon
public import Flock.Circuit
public import Flock.Draw
public import Flock.Firewall
public import Flock.HmRow
public import Flock.Public
public import Flock.Registered
public import Mathlib
public import Proofs.Flock.EndToEndRegistered.Inputs
public import Proofs.Flock.Recursive.FirewallJson
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.Soundness.BindSound
public import Proofs.Flock.Soundness.Binding.Layout
public import Proofs.Flock.Soundness.Binding.RowSeg
public import Proofs.Flock.Soundness.Binding.RowsX
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
public import Proofs.Flock.Soundness.Registered.ReadsX
public import Proofs.Flock.Soundness.SessionBatch
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Proofs.Flock.SessionLoop.Concrete.Inputs
import Proofs.Flock.Soundness.Refine.Walk
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# `EndToEndRegistered`'s conclusion fails for a verifier that accepts everything (its nontrivial witness)

`Flock.SecurityProofs.EndToEndRegistered_nontrivial`: at a model-level instance, a read a sampled unit's row makes of a
registered port, committed under the row's own hm96 leaf and listed by `OwnReadsHJ`, is not `Checked` against the
port's registered root through the path the public file names. An accepted session's every such read is: at a
verifier that accepts every transcript, `EndToEndRegistered`'s conclusion would be false.

The instance: no gate (`circ0`), one unit and one table (`tabW`); the circuit file's one port `x` (`circW`) reads row 0
of `portW`, a registered value of two rows whose root the row's leaf climbs to through `pathW`; the record draws unit
0 (`tr0`), and the public file (`pubW`) names no registered tables, so the read's registered path is empty and
`opensLeaf` refuses it on its length. `Flock.SecurityProofs.EndToEndRegisteredDrawn_nontrivial` is the same for
`EndToEndRegisteredDrawn`, whose conclusion is `EndToEndRegistered`'s.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg.Witness

open Lean (Json)
open Exec Binding Registered Audit Refine
open FlockSoundness.Discharge.Composed (drawJ partOf)

/-- **`regPathsHJ` is empty** at a read whose public file, drawn at the record's sample, names no registered tables. -/
theorem regPaths_none {n : ℕ} {Gate : Type} {I : Inputs} {c : Flock.Circuit} {pub : Flock.Public}
    {t : Exec.Transcript} {port : (Fin n × Fin c.ports.size) ⊕ Gate → Flock.Registered.Port} {u : Fin n}
    {q : Fin c.ports.size} {d : Json} {D : Flock.Draw.UnitDraw} {dp : Flock.Public}
    (hd : recordDraw t.record = some d) (hD : Flock.Draw.UnitDraw.ofJson (drawJ I d (partOf I d u.val)) = .ok D)
    (hdp : Flock.HmOut.drawn c pub (drawJ I d (partOf I d u.val)) = .ok dp) (ht : dp.tables = none) :
    regPathsHJ I c pub t port (.inl (u, q)) = ByteArray.empty := by
  simp [regPathsHJ, hd, hD, hdp, ht, Except.toOption]

/-! ## The model -/

/-- The witness's model circuit: no gate. -/
abbrev circ0 : Verity.Boolean.Circuit where
  N := 0
  op g := g.elim0
  topo g := g.elim0
  outputs := ∅

abbrev part0 : Audit.Partition circ0 1 where
  unit g := g.elim0
  input_iff g := g.elim0

/-- The table's statement: `2^22` witness bits in blocks of `2^6`, no region. -/
abbrev stmtW : Model.Statement where
  m := 22
  kLog := 6
  hk := by decide
  hk6 := le_rfl
  pin := ⟨0, by decide⟩
  A₀ := 0
  B₀ := 0
  regions := []

/-- The table: `stmtW` at `fast100`'s schedule for `m = 22`, its link points the session's 16. -/
abbrev tabW : TabSpec 16 where
  S := stmtW
  m := 22
  k0 := 6
  l₀ := ⟨1, 218, 9, 6⟩
  l₁ := ⟨2, 106, 6, 4⟩
  rest := [⟨4, 53, 3, 4⟩]
  ext := 3
  hsch := by decide +kernel
  hm := by decide
  ptLocal := 0
  hlay := by
    unfold Model.Statement.LinkLayout
    exact ⟨fun _ hR => by simp at hR, by decide⟩

abbrev plan0 : Finset (Fin 1) → Unit → List (TabSpec 16) := fun _ _ => [tabW]

abbrev tab0 : (S : Finset (Fin 1)) → (R : Unit) → Fin 1 → Fin (plan0 S R).length := fun _ _ _ => ⟨0, by simp⟩

abbrev decode0 : Audit.Partition.DecoderB (C := circ0) (Reg := Unit) plan0 tab0 := fun _ _ _ _ g => g.elim0

/-! ## The circuit file, the layout and the read -/

/-- The witness's circuit file: one input port, `x`, of one word. -/
def circW : Flock.Circuit := { (default : Flock.Circuit) with ports := #[{ name := "x", words := 1, chunks := 1, nb := 0 }] }

def q0 : Fin circW.ports.size := ⟨0, by decide⟩

/-- Each sampled unit's ports and one other position; one value, the empty row, under a constant registered leaf. -/
abbrev lay0 : FlockSoundness.Binding.Layout part0 plan0 tab0 decode0 where
  Pos := (Fin 1 × Fin circW.ports.size) ⊕ Unit
  inhPos := ⟨Sum.inr ()⟩
  Val := Unit
  pos g := g.elim0
  bit _ g := g.elim0
  regLeaf _ _ := H512 []
  rowBits _ := 0
  rowOf _ _ := ()
  rowAt _ _ _ _ i := i.elim0
  saltAt _ _ _ _ _ _ := ⟨0, Nat.two_pow_pos _⟩
  decode_row _ _ _ _ _ _ g := g.elim0

/-- Every position is a row, of no bytes. -/
abbrev rs0 : HmRowsX lay0 where
  isRow _ := True
  enc _ _ := []
  ext _ _ _ := ([], [])
  ext_spec _ v v' h _ := absurd (Subsingleton.elim (α := Unit) v v') h
  other _ := ExplicitLeaf.hm96X (fun _ => []) (fun _ _ => ([], [])) fun v v' h _ =>
    absurd (Subsingleton.elim (α := Unit) v v') h
  bcAt _ _ _ _ _ _ := ⟨0, Nat.two_pow_pos _⟩
  bc_registered _ _ _ _ _ _ g := g.elim0
  other_registered _ _ _ _ _ _ g := g.elim0

/-- The read's committed leaf: the empty row's hm96 leaf under the empty salt. -/
def payloadW : ByteArray :=
  ((ExplicitLeaf.hm96X (rs0.enc (Sum.inl (0, q0))) (rs0.ext (Sum.inl (0, q0)))
    (rs0.ext_spec (Sum.inl (0, q0)))).leaf () []).1

/-- One 64-byte sibling: a path through a tree of two leaves. -/
def pathW : ByteArray := ⟨Array.replicate 64 0⟩

/-- **The registered port**: two rows of `x`, whose root is row 0's leaf climbed through `pathW`. -/
def portW : Flock.Registered.Port where
  name := "x"
  value := ""
  root := Flock.FV3.Sha512.climb ByteArray.empty (Flock.FV3.Sha512.leafAs Flock.HmRow.ROW_SCHEMA ByteArray.empty 0 payloadW)
    0 (Flock.log2ceil 2) pathW
  leaves := 2
  words := none
  wordBits := 16
  domain := ByteArray.empty
  readsRel := #[some 0]

def own0 : Array Flock.Registered.Port := #[portW]

/-- **The read**: unit 0's port `x` reads `portW`'s row 0, committed under its own leaf through `pathW`. -/
def rd0 : ReadsX lay0 rs0 where
  isRead p := p = Sum.inl (0, q0)
  read_row _ _ := trivial
  port _ := portW
  readAt _ := 0
  wv _ := ()
  ws _ := []
  wpath _ := pathW
  committed := by
    intro p hp
    change p = Sum.inl (0, q0) at hp
    subst hp
    have hlog : Flock.log2ceil 2 = 1 := rfl
    simp only [Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨⟨by decide, ?_⟩, rfl⟩
    show pathW.size = 64 * Flock.log2ceil 2
    rw [hlog]
    rfl

/-! ## The record and the public file -/

def jn (n : Nat) : Json := .num ⟨n, 0⟩

/-- The record's sample: a subset draw of unit 0 of a population of 1. -/
def drawW : Json := Json.mkObj [("k", jn 1), ("law", .str "subset"), ("population", jn 1), ("units", .arr #[jn 0])]

def recW : Json := Json.mkObj [("unit_draw", drawW)]

def tr0 : Exec.Transcript := { record := Flock.Firewall.printJson recW, proofs := #[] }

/-- The public file: one instance, at population index 0, with no digests, outputs or registered tables. -/
def pubW : Flock.Public :=
  { header := Json.mkObj [("units", Json.mkObj [("indices", .arr #[jn 0])])], n := 1, digests := #[], outputs := #[],
    sha := ByteArray.empty }

def drawU : Flock.Draw.UnitDraw := { law := .subset 1, population := 1, units := #[0] }

theorem drawW_obj : drawW = .obj ⟨⟨FlockDeploy.Concrete.objT drawW⟩⟩ := by kernel_rfl

theorem drawW_model : (FlockDeploy.Concrete.objT drawW).toListModel =
    [⟨"k", jn 1⟩, ⟨"law", .str "subset"⟩, ⟨"population", jn 1⟩, ⟨"units", .arr #[jn 0]⟩] := by kernel_rfl

theorem drawW_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT drawW) = true := by kernel_rfl

/-- `plainStr` reads its strings through `String.all`, which the kernel doesn't evaluate, so `simp` decides them. -/
theorem plain_drawW : Flock.Firewall.plain drawW = true := by
  rw [drawW_obj]
  refine FlockDeploy.Concrete.plain_obj_of drawW_model drawW_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  simp [Flock.Firewall.plain, Flock.Firewall.plainList, Flock.Firewall.plainStr, String.all_bool_eq, jn]

theorem recW_obj : recW = .obj ⟨⟨FlockDeploy.Concrete.objT recW⟩⟩ := by kernel_rfl

theorem recW_model : (FlockDeploy.Concrete.objT recW).toListModel = [⟨"unit_draw", drawW⟩] := by kernel_rfl

theorem recW_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT recW) = true := by kernel_rfl

theorem plain_recW : Flock.Firewall.plain recW = true := by
  rw [recW_obj]
  refine FlockDeploy.Concrete.plain_obj_of recW_model recW_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq]
  exact ⟨by simp [Flock.Firewall.plainStr, String.all_bool_eq], plain_drawW⟩

theorem norm_recW : Recursion.FirewallJson.norm recW = recW := by kernel_rfl

theorem recordDraw_tr0 : recordDraw tr0.record = some drawW := by
  show recordDraw (Flock.Firewall.printJson recW) = some drawW
  unfold recordDraw
  rw [Recursion.FirewallJson.parse_printJson plain_recW, norm_recW]
  kernel_rfl

theorem ofJson_drawW : Flock.Draw.UnitDraw.ofJson drawW = .ok drawU := by kernel_rfl

/-- The sampled statement's header: `pubW`'s, with the sample, its size, the population's and the proved units. -/
def hdrW : Json :=
  (((pubW.header.setObjVal! "unit_draw" drawW).setObjVal! "instances" (Lean.toJson drawU.proved.size)).setObjVal!
    "population" (Lean.toJson pubW.n)).setObjVal! "units" (Flock.HmRow.unitsEntry { circW with outPorts := #[] } #[0])

def hdrS : String := (Flock.canonUtf8 hdrW).toOption.getD ""

/-- **The header's canonical string**, the walk's rule for it: closed, since `canonUtf8_eq`'s right side reads
`canonUtf8` again, so as a rule it would rewrite its own result. -/
theorem hdr_of : Flock.canonUtf8 hdrW = .ok hdrS := FlockDeploy.Concrete.canonUtf8_eq (by kernel_rfl)

set_option maxHeartbeats 2000000 in -- the walk steps through `HmRow.drawn` and leaves closed checks to the kernel
fwd_def drawnPub drawn_pubW : Flock.HmOut.drawn circW pubW drawW by
  [hdr_of, ofJson_drawW]
  block [Flock.canonUtf8, Flock.Sha512.hash]
  shallow []

theorem drawnPub_tables : drawnPub.tables = none := by kernel_rfl

theorem unitIndices_drawnPub : Flock.Registered.unitIndices drawnPub = .ok #[0] := by kernel_rfl

theorem regPaths0 : regPathsHJ ins circW pubW tr0 rd0.port (Sum.inl (0, q0)) = ByteArray.empty :=
  regPaths_none recordDraw_tr0 (by rw [drawJ_one ins_count]; exact ofJson_drawW)
    (by rw [drawJ_one ins_count]; exact drawn_pubW) drawnPub_tables

/-- **The read is the record's own**: `OwnReadsHJ` lists it, at the record's sample of unit 0. -/
theorem ownReads0 : OwnReadsHJ ins own0 circW pubW tr0 rd0.isRead rd0.port rd0.readAt := by
  intro p hp
  change p = Sum.inl (0, q0) at hp
  subst hp
  refine ⟨0, q0, drawW, drawU, drawnPub, #[0], rfl, by simp [own0, rd0], by kernel_rfl, recordDraw_tr0,
    by rw [drawJ_one ins_count]; exact ofJson_drawW, by rw [drawJ_one ins_count]; exact drawn_pubW,
    unitIndices_drawnPub, by decide +kernel, by kernel_rfl⟩

/-- **The read is not `Checked`**: the public file names no path to `portW`'s root, and an empty path is no path through
a tree of two leaves. -/
theorem notChecked0 : ¬ rd0.Checked () (regPathsHJ ins circW pubW tr0 rd0.port) := by
  intro h
  have h1 := h (Sum.inl (0, q0)) rfl
  rw [regPaths0] at h1
  simp only [Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h1
  have h2 : ByteArray.empty.size = 64 * Flock.log2ceil 2 := h1.1.2
  have h3 : Flock.log2ceil 2 = 1 := rfl
  have h4 : ByteArray.empty.size = 0 := rfl
  omega

end FlockSoundness.Discharge.ZkReg.Witness

namespace FlockSoundness.Discharge.ZkReg

open Witness

/-- **`EndToEndRegistered` rules out a verifier that accepts everything**: at the witness's instance, a read `OwnReadsHJ`
lists, of a record such a verifier accepts, is not `Checked` against its port's registered root. -/
theorem _root_.Flock.SecurityProofs.EndToEndRegistered_nontrivial :
    ∃ (o : Unit × Finset (Fin 1) × Exec.Transcript) (rd : Registered.ReadsX lay0 rs0)
      (_ : OwnReadsHJ ins own0 circW pubW o.2.2 rd.isRead rd.port rd.readAt),
      ¬ rd.Checked o.1 (regPathsHJ ins circW pubW o.2.2 rd.port) :=
  ⟨((), ∅, tr0), rd0, ownReads0, notChecked0⟩

/-- **`EndToEndRegisteredDrawn` rules out a verifier that accepts everything**, at `EndToEndRegistered`'s instance. -/
theorem _root_.Flock.SecurityProofs.EndToEndRegisteredDrawn_nontrivial :
    ∃ (o : Unit × Finset (Fin 1) × Exec.Transcript) (rd : Registered.ReadsX lay0 rs0)
      (_ : OwnReadsHJ ins own0 circW pubW o.2.2 rd.isRead rd.port rd.readAt),
      ¬ rd.Checked o.1 (regPathsHJ ins circW pubW o.2.2 rd.port) :=
  Flock.SecurityProofs.EndToEndRegistered_nontrivial

end FlockSoundness.Discharge.ZkReg
