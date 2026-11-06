module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.SampledProofs.Tables

/-!
# v2's session at the registered files (`Instance.SessionAt`)

`SessionAt` asks, at each parameters `θ`, for one session of the deployment's (`Instance.session θ`) that v2 sets up on
every accepted play's committed files naming the registered circuit and public files. It splits in two:

* `SessionRegistered`: the deployment's session at `θ` is one v2 sets up on the registered files, wherever some accepted
  play names them. `Instance.session` is a field of the deployment, and a session (`Live`) carries facts no accepted play
  proves (the circuit's `Built`, the rate bound), so this is the deployment's to supply, not a consequence of the
  instance's construction.
* `TablesCommitted`: an accepted play's committed files hold the registered files' lookup tables. v2's reading of
  committed files (`readOf`) reads the tables off the committed files `F` alone, so files naming the same circuit and
  public files are read alike only where they hold the same tables (`read_files`, `setsUp_files`).

`sessionAt_of_registered` gives `SessionAt` from the two.

`TablesCommitted` does not hold for a circuit that looks up a table. v2 reads the tables off the committed files, then
the evidence's blobs (`F ++ B`), and holds their names there to the registered circuit's lookups (`tables_registered`,
from `Properties.SampledProofs.TablesRegistered`). It doesn't see which of the blobs were committed. A prover that sends
a looked-up table among the evidence's blobs rather than among its committed files is accepted alike, so two accepted
plays at one `θ` have committed files with different tables, and no one session is set up on both. What closes it is a
`readOf` whose tables are the ones v2 reads:
* their names are the registered circuit's lookups, fixed by `θ` (`tables_registered`);
* each is taken at its pinned digest, with the registered files holding those tables (`registeredOf`);
* the tables' finder compares the blobs sent with the registered files.
With that reading, `read_files` needs no table premise and `SessionAt` follows from `SessionRegistered` alone.
-/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An accepted play's blobs hold the registered circuit's lookup tables and no other pinned table**: v2's case in it
accepts on the blobs sent, the committed files then the evidence's (`Properties.SampledProofs.TablesRegistered`). -/
theorem tables_registered {V : Vocab d α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    {p₀ : _root_.SampledProofs.OneStageZk.Params} (hθ : ParamsAt o.params p₀) (hv : V.Verdict k .sampled o) :
    Properties.SampledProofs.HoldsRegisteredTables p₀.own (_root_.Core.Blobs.ofArray (o.files ++ o.blobs)) := by
  obtain ⟨_, _, _, p, _, _, _, _, hcase, hp, -, -, -, -, pub, prof, hck⟩ := case_drew hv.2.1 hθ (verdict_case hv)
  cases params_of_case hv.2.1 hcase hp hθ
  exact Properties.SampledProofs.SecurityProofs.tablesRegistered.2 _ _ _ _ _ _ _ hck

namespace Instance

variable (I : Instance d α)

/-- **v2's reading of committed files is their circuit and public files' and their tables'**: files naming the same
circuit and public files, with the same tables, are read alike (`readOf`). -/
theorem read_files {θ : ByteArray} {F F' : Array ByteArray} (hc : I.base.circuitFile θ F = I.base.circuitFile θ F')
    (hp : I.base.publicFile θ F = I.base.publicFile θ F')
    (ht : _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) =
      _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F')) :
    I.base.read θ F = I.base.read θ F' := by
  simp only [Vocab.circuitFile, Vocab.publicFile, I.programDigest, I.publicDigest] at hc hp
  rw [I.read, I.read, readOf, readOf, hc, hp, ht]

/-- **A session v2 sets up on some files it sets up on files naming the same circuit and public files, read alike.** -/
theorem setsUp_files {θ : ByteArray} {F F' : Array ByteArray} {A : Live}
    (hc : I.base.circuitFile θ F = I.base.circuitFile θ F') (hp : I.base.publicFile θ F = I.base.publicFile θ F')
    (hr : I.base.read θ F' = I.base.read θ F) (h : SetsUp I.base θ F A) : SetsUp I.base θ F' A :=
  { h with circuit := h.circuit.trans hc, publicFile := h.publicFile.trans hp, read := hr.trans h.read }

/-- **The deployment's session at the parameters `θ` is one v2 sets up on the registered files**, wherever some play's
committed files name the registered circuit and public files and its sampled verdict accepts: a statement of
`I.session θ` and the registered files alone. -/
def SessionRegistered (k : Verity.Registration.Composite) : Prop :=
  ∀ θ, (∃ p, ParamsAt θ p) →
    (∃ F c o, I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) ∧
      I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) ∧
      Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    ∃ A, I.session θ = some A ∧ SetsUp I.base θ (I.base.registered θ) A

/-- **An accepted play's committed files hold the registered files' lookup tables**: at `SessionAt`'s premises, the
pinned tables the committed files hold (`SampledProofs.OneStageZk.tablesOf`, which `readOf` reads) are the registered
files'. -/
def TablesCommitted (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F, (∃ p, ParamsAt θ p) → I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) →
    I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) → TablesAgree F (I.base.registered θ) →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) =
      _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray (I.base.registered θ))

/-- **`SessionAt` from the deployment's session at the registered files**: the session the deployment sets up on the
registered files (`SessionRegistered`) it sets up on an accepted play's committed files, which name the same circuit
and public files and, under `TablesCommitted`, hold the same tables (`read_files`, `setsUp_files`). -/
theorem sessionAt_of_registered (k : Verity.Registration.Composite) (h : I.SessionRegistered k)
    (ht : I.TablesCommitted k) : I.SessionAt k := by
  intro θ F hP hc hp hT ⟨c, o, hreach, hv⟩
  obtain ⟨A, hA, hs⟩ := h θ hP ⟨F, c, o, hc, hp, hreach, hv⟩
  exact ⟨A, hA, I.setsUp_files hc.symm hp.symm (I.read_files hc hp (ht θ F hP hc hp hT ⟨c, o, hreach, hv⟩)) hs⟩

end Instance

end Verity.Sampled
