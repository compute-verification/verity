module

public import Proofs.Properties.Verity.Sampled.MemberForm.Opens
public import Proofs.Properties.Verity.Sampled.MemberForm.Passes
public import Proofs.Properties.Verity.Sampled.OpensAt

/-! Each member's deployed commitment opens only at its own files' places. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

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

theorem checked_circuit {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) (hc : m.Checked) :
    m.fileRead.circuit = Flock.Bridge.circuitOf m.D m.hU m.n :=
  Flock.Bridge.circuitOfFile_eq hc.parsed

theorem opensOwn_of_ready {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ) (hc : m.Checked)
    (hr : Commit.Ready m.I) : OpensOwn m := by
  intro hN g b dg _ hop v hv
  obtain ⟨-, pubO, hO, hnO⟩ := Flock.Bridge.parse_of_parsedOf hc.parsed
  have hn : m.pub.n = m.n := (Flock.Bridge.loadPublic_inOut m.hpub hO).1.symm.trans hnO
  have hw : m.fileRead.circuit.w g = 1 :=
    w_ofAudit ((checked_circuit m hc).trans (Flock.Bridge.circuitOf_eq m.D m.hU m.n)) g
  have hL : Commit.Loads m.D m.n m.pi m.pub m.I := ⟨m.ht, m.hTags, m.hD, m.hpub, hn, hc.distinct⟩
  exact opensPlace_of_ready (r := m.fileRead) hc.noPublic hL hr m.hD m.hpub hn hN g hw hop hv

theorem FilesPass.ready {nJ : ℕ} {m : Member nJ} (h : FilesPass m) : Commit.Ready m.I :=
  ⟨h.ports⟩

theorem ExecutionPhaseOn.reads_toV5 {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : ExecutionPhaseOn V T θ F W) (t : ℕ) :
    (W.ms.map fun m => m.toV5 t).map (·.fileRead) = (V.read θ F).first :: (V.read θ F).rest := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have e₁ : (W.ms.map fun m => m.toV5 t).map (·.fileRead) = W.ms.map fileRead :=
    List.map_map.trans (List.map_congr_left fun _ _ => rfl)
  have e₂ : W.ms.map fileRead = fileRead m₀ :: rest.map fileRead := congrArg (List.map fileRead) hW
  have e₃ : (V.read θ F).first :: (V.read θ F).rest = fileRead m₀ :: rest.map fileRead :=
    congrArg (fun R : Verity.ExecutionPhaseFiles => R.first :: R.rest) hread
  exact (e₁.trans e₂).trans e₃.symm

theorem MembersSetUp.opensOwn {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : MembersSetUp V T θ F W) (t : ℕ)
    {M : MemberV5 ByteArray W.nJ} (hM : M ∈ W.ms.map fun m => m.toV5 t) : OpensOwn M := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact opensOwn_of_ready (m.toV5 t) ((h.passes m hm).checked_all t) (h.passes m hm).ready

theorem MembersSetUp.opensWith {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : MembersSetUp V T θ F W) (c : ByteArray)
    (hplace : ∀ g, V.place θ F c g = placeOfFiles (V.read θ F) g) (t : ℕ) :
    OpensWith (V.read θ F).first (V.read θ F).rest (V.Opens θ F c) (W.ms.map fun m => m.toV5 t) :=
  opensWith_read V θ c F hplace _ (h.toExecutionPhaseOn.reads_toV5 t) fun _ hM => h.opensOwn t hM

end Verity.Sampled.MemberForm
