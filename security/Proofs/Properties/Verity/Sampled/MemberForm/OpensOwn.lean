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
  ready for its openings (`Commit.Ready`) with no public input port, an opening it accepts at a gate's claims
  position is a `Hash.RowLeaf` opening at the place the committed files register for the gate
  (`Commit.opens_rowLeaf`, `Commit.placeOfRead_eq`), of every value of the gate's one bit.
* **A member** (`opensOwn_of_ready`): a member whose files the verifier checked (`MemberV5.Checked`) and whose
  commitment is ready for its openings opens only at its own files' places. Its files' circuit is its parsed
  circuit file's (`checked_circuit`), of one bit a gate.
* **A window** (`MembersSetUp.opensOwn`, `MembersSetUp.opensWith`): at a window the verifier sets up, every member's
  files pass the files check, so each member of Flock's window opens only at its own files' places and the
  vocabulary's openings hold wherever each member's deployed commitment opens, where the vocabulary places each gate
  at the committed files' place.

Each takes the members' commitments ready for their openings as given. Of `Commit.Ready`'s clauses the files check's
pass gives the ports' published domains (`FilesPass.ready`); the rows' format, two leaves a port, distinct roots and
rows shorter than `2^64` bits are its other four.

Nothing here reads a play, a verdict or a session.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An opening the commitment accepts opens the gate at the files' place**: at a commitment ready for its
openings whose circuit names no public input port, where the committed files `r` parse to its descriptor and load
to its public file, of its population, an opening it accepts under its roots at the claims position of a gate of
one bit opens, at the place `r` registers for the gate (`placeOfRead`), every value whose first bit it opens. -/
theorem opensPlace_of_ready {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {n : ℕ}
    {pi : Flock.HmIn.Ports} {pub : Flock.Public} (hpi : pi = {}) (hA : Commit.Ready D hU n pi pub)
    {r : Verity.FileRead} (hD : Flock.Desc.parse r.tags r.circuitFile r.tables = .ok D)
    (hpub : Flock.HmIn.loadPublic r.tags D.unroll pi r.publicFile = .ok pub) (hn : pub.n = n)
    (hN : r.circuit.N = (Commit.Cc D hU n pi).N) (g : Fin r.circuit.N) (hw : r.circuit.w g = 1) {b : Bool}
    {dg : Claims.Msg}
    (hop : (Commit.Cd D hU n pi pub).Opens Claims.Sampling.sha512Msg (Commit.roots D n pub)
      ⟨Flock.Bridge.posOf (Commit.Cc D hU n pi) (Fin.cast hN g), b, dg⟩ = true)
    {v : BitVec (r.circuit.w g)} (hv : v.getLsbD 0 = b) :
    OpensPlace (placeOfRead r g) (r.circuit.w g) v := by
  subst hpi
  obtain ⟨p, hp, o, hacc, hrow⟩ := Commit.opens_rowLeaf hA hop
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

/-- **A checked member whose commitment is ready for its openings opens only at its own files' places**: its files
parse to its descriptor and load, over no public input port, to its public file, of its population
(`MemberV5.Checked`), so an opening its deployed commitment accepts is one at the place those files register
(`opensPlace_of_ready`). -/
theorem opensOwn_of_ready {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) (hc : m.Checked)
    (hr : Commit.Ready m.D m.hU m.n m.pi m.pub) : OpensOwn m := by
  intro hN g b dg _ hop v hv
  obtain ⟨-, pubO, hO, hnO⟩ := Flock.Bridge.parse_of_parsedOf hc.parsed
  have hn : m.pub.n = m.n := (Flock.Bridge.loadPublic_inOut m.hpub hO).1.symm.trans hnO
  have hw : m.fileRead.circuit.w g = 1 :=
    w_ofAudit ((checked_circuit m hc).trans (Flock.Bridge.circuitOf_eq m.D m.hU m.n)) g
  exact opensPlace_of_ready (r := m.fileRead) hc.noPublic hr m.hD m.hpub hn hN g hw hop hv

/-- **A member whose files pass has a commitment ready for its openings**, at rows of the `sha512/row/v2` or
`sha512/row-seg/v1` format, at least two leaves a port, distinct roots and rows shorter than `2^64` bits: the ports
check's pass gives each port's published domain 64 bytes (`Commit.domain_of_ports`). -/
theorem FilesPass.ready {nJ : ℕ} {m : Member nJ} (h : FilesPass m)
    (hrows : m.D.unroll.ports.all (fun p => p.v2 || p.segs.isSome) = true)
    (hleaves : ∀ q < m.D.unroll.ports.size, 2 ≤ (Commit.Lm m.D m.n m.pub).cnt q)
    (hroots : ∀ q q', q' < q → q < m.D.unroll.ports.size →
      (Commit.roots m.D m.n m.pub).getD q' [] ≠ (Commit.roots m.D m.n m.pub).getD q [])
    (hwidth : ∀ i,
      (Flock.Bridge.placeOf (Commit.Cc m.D m.hU m.n {}) (Commit.rows m.D m.hU m.n {} m.pub) i).1 < 2 ^ 64) :
    Commit.Ready m.D m.hU m.n {} m.pub :=
  ⟨hrows, Commit.domain_of_ports m.I m.ht m.hTags m.hD m.hpub h.ports, hleaves, hroots, hwidth⟩

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
every cost, where each member's commitment is ready for its openings: each member's files pass the files check, so
it is `Checked` (`FilesPass.checked_all`). -/
theorem MembersSetUp.opensOwn {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) (hready : ∀ m ∈ W.ms, Commit.Ready m.D m.hU m.n {} m.pub) (t : ℕ)
    {M : MemberV5 ByteArray W.nJ} (hM : M ∈ W.ms.map fun m => m.toV5 t) : OpensOwn M := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact opensOwn_of_ready (m.toV5 t) ((h.passes m hm).checked_all t) (hready m hm)

/-- **At a window the verifier sets up, the vocabulary's openings hold wherever each member's deployed commitment
opens**, at every cost: where the vocabulary places each gate at the committed files' place and each member's
commitment is ready for its openings, at the members of Flock's window in order, each at its own gates of the
members' circuits side by side (`opensWith_read`, `MembersSetUp.opensOwn`). -/
theorem MembersSetUp.opensWith {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) (c : ByteArray)
    (hplace : ∀ g, V.place θ F c g = placeOfFiles (V.read θ F) g)
    (hready : ∀ m ∈ W.ms, Commit.Ready m.D m.hU m.n {} m.pub) (t : ℕ) :
    OpensWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) (W.ms.map fun m => m.toV5 t) :=
  opensWith_read V θ c F hplace _ (h.toWindowOn.reads_toV5 t) fun _ hM => h.opensOwn hready t hM

end Verity.Sampled.MemberForm
