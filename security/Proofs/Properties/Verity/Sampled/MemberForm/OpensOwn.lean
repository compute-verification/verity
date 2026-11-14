module

public import Proofs.Properties.Verity.Sampled.MemberForm.Opens
public import Proofs.Properties.Verity.Sampled.MemberForm.Passes
public import Proofs.Properties.Verity.Sampled.OpensAt

/-!
# Each member's deployed commitment opens only at its own files' places

The claim at a window asks of each member that the openings at its gates hold wherever its deployed commitment opens
(`MemberV5.ReadAt.opens`), and `opensWith_read` gives that of the vocabulary's openings at members whose commitments
open only at their own files' places (`OpensOwn`). This module proves `OpensOwn` of a window's members:

* **An opening the commitment accepts opens the gate at the files' place** (`opensPlace_of_ready`): at a commitment
  with no public input port whose files load to its data (`Commit.Loads`) and are ready for its openings
  (`Commit.Ready`), an opening it accepts at a gate's claims position is a `Hash.RowLeaf` opening at the place the
  committed files register for the gate (`Commit.opens_rowLeaf`, `Commit.placeOfRead_eq`), of every value of the
  gate's one bit.
* **A member** (`opensOwn_of_ready`): a member whose files the verifier checked (`MemberV5.Checked`) and are ready
  for its commitment's openings opens only at its own files' places. Its files' circuit is its parsed circuit
  file's (`checked_circuit`), of one bit a gate.
* **A window** (`MembersSetUp.opensOwn`, `MembersSetUp.opensWith`): at a window the verifier sets up, every member's
  files pass the files check, so each member of Flock's window opens only at its own files' places and the
  vocabulary's openings hold wherever each member's deployed commitment opens, where the vocabulary places each gate
  at the committed files' place.

None takes the members' files ready for their openings as given: `Commit.Ready` is that the files pass the ports and
roots check, which the files check's pass holds (`FilesPass.ready`).

Nothing here reads a play, a verdict or a session.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An opening the commitment accepts opens the gate at the files' place**: at a commitment whose circuit names no
public input port and whose files `I` load to its data and are ready for its openings, where the committed files
`r` parse to its descriptor and load to its public file, of its population, an opening it accepts under its roots
at the claims position of a gate of one bit opens, at the place `r` registers for the gate (`placeOfRead`), every
value whose first bit it opens. -/
theorem opensPlace_of_ready {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {n : ℕ}
    {pi : Flock.HmIn.Ports} {pub : Flock.Public} {I : FlockSoundness.Discharge.Exec.Inputs} (hpi : pi = {})
    (hL : Commit.Loads D n pi pub I) (hA : Commit.Ready I)
    {r : Verity.FileRead} (hD : Flock.Desc.parse r.tags r.circuitFile r.tables = .ok D)
    (hpub : Flock.HmIn.loadPublic r.tags D.unroll pi r.publicFile = .ok pub) (hn : pub.n = n)
    (hN : r.circuit.N = (Commit.Cc D hU n pi).N) (g : Fin r.circuit.N) (hw : r.circuit.w g = 1) {b : Bool}
    {dg : Claims.Msg}
    (hop : (Commit.Cd D hU n pi pub).Opens Claims.Sampling.sha512Msg (Commit.roots D n pub)
      ⟨Flock.Bridge.posOf (Commit.Cc D hU n pi) (Fin.cast hN g), b, dg⟩ = true)
    {v : BitVec (r.circuit.w g)} (hv : v.getLsbD 0 = b) :
    OpensPlace (placeOfRead r g) (r.circuit.w g) v := by
  subst hpi
  obtain ⟨p, hp, o, hacc, hrow⟩ := Commit.opens_rowLeaf hL hA hop
  refine ⟨p, (Commit.placeOfRead_eq hD hpub hn hN g).trans hp, o, hacc, fun k => ?_⟩
  have hk : k.val = 0 := by
    have h1 := k.isLt
    omega
  rw [hk, Nat.add_zero, hrow]
  exact congrArg some hv.symm

/-- **A checked member's committed files' circuit is its parsed circuit file's**, at its population. -/
theorem checked_circuit {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) (hc : m.Checked) :
    m.fileRead.circuit = Flock.Bridge.circuitOf m.D m.hU m.n :=
  Flock.Bridge.circuitOfFile_eq hc.parsed

/-- **A checked member whose files are ready for its commitment's openings opens only at its own files' places**:
its files parse to its descriptor and load, over no public input port, to its public file, of its population and
of pairwise distinct published domains (`MemberV5.Checked`, so `Commit.Loads`), so an opening its deployed
commitment accepts is one at the place those files register (`opensPlace_of_ready`). -/
theorem opensOwn_of_ready {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) (hc : m.Checked)
    (hr : Commit.Ready m.I) : OpensOwn m := by
  intro hN g b dg _ hop v hv
  obtain ⟨-, pubO, hO, hnO⟩ := Flock.Bridge.parse_of_parsedOf hc.parsed
  have hn : m.pub.n = m.n := (Flock.Bridge.loadPublic_inOut m.hpub hO).1.symm.trans hnO
  have hw : m.fileRead.circuit.w g = 1 :=
    w_ofAudit ((checked_circuit m hc).trans (Flock.Bridge.circuitOf_eq m.D m.hU m.n)) g
  have hL : Commit.Loads m.D m.n m.pi m.pub m.I := ⟨m.ht, m.hTags, m.hD, m.hpub, hn, hc.distinct⟩
  exact opensPlace_of_ready (r := m.fileRead) hc.noPublic hL hr m.hD m.hpub hn hN g hw hop hv

/-- **A member whose files pass has files ready for its commitment's openings**: `Commit.Ready` is the ports and
roots check's pass, which the files check holds (`FilesPass.ports`). -/
theorem FilesPass.ready {nJ : ℕ} {m : Member nJ} (h : FilesPass m) : Commit.Ready m.I :=
  ⟨h.ports⟩

/-- **Flock's window's members' reads are the vocabulary's read of the committed files, in order**, at a window on
the files, at every cost. -/
theorem WindowOn.reads_toV5 {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) (t : ℕ) :
    (W.ms.map fun m => m.toV5 t).map (·.fileRead) = (V.read θ F).first :: (V.read θ F).rest := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have e₁ : (W.ms.map fun m => m.toV5 t).map (·.fileRead) = W.ms.map fileRead :=
    List.map_map.trans (List.map_congr_left fun _ _ => rfl)
  have e₂ : W.ms.map fileRead = fileRead m₀ :: rest.map fileRead := congrArg (List.map fileRead) hW
  have e₃ : (V.read θ F).first :: (V.read θ F).rest = fileRead m₀ :: rest.map fileRead :=
    congrArg (fun R : Verity.WindowFiles => R.first :: R.rest) hread
  exact (e₁.trans e₂).trans e₃.symm

/-- **At a window the verifier sets up, every member of Flock's window opens only at its own files' places**, at
every cost: each member's files pass the files check, so it is `Checked` (`FilesPass.checked_all`) and its files
are ready for its commitment's openings (`FilesPass.ready`). -/
theorem MembersSetUp.opensOwn {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) (t : ℕ)
    {M : MemberV5 ByteArray W.nJ} (hM : M ∈ W.ms.map fun m => m.toV5 t) : OpensOwn M := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact opensOwn_of_ready (m.toV5 t) ((h.passes m hm).checked_all t) (h.passes m hm).ready

/-- **At a window the verifier sets up, the vocabulary's openings hold wherever each member's deployed commitment
opens**, at every cost: where the vocabulary places each gate at the committed files' place, at the members of
Flock's window in order, each at its own gates of the members' circuits side by side (`opensWith_read`,
`MembersSetUp.opensOwn`). -/
theorem MembersSetUp.opensWith {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) (c : ByteArray)
    (hplace : ∀ g, V.place θ F c g = placeOfFiles (V.read θ F) g) (t : ℕ) :
    OpensWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) (W.ms.map fun m => m.toV5 t) :=
  opensWith_read V θ c F hplace _ (h.toWindowOn.reads_toV5 t) fun _ hM => h.opensOwn t hM

end Verity.Sampled.MemberForm
