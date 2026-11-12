module

public import Proofs.Properties.Verity.Sampled.Instance

/-!
# v2's session at the registered files (`Instance.SessionAt`)

`SessionAt` asks, at each parameters `θ` where the deployment runs a session (`Instance.session θ`), that v2 set it up
on every accepted play's committed files naming the registered circuit and public files; where it runs none, SPOT's
published δ is `⊤` (`Instance.deltaOf`). v2's reading of committed files (`readOf`) is their circuit and public files'
and the lookup tables the circuit file registers (`pinnedTables`), so files naming the same circuit and public files are
read alike (`read_files`, `setsUp_files`), whichever tables they hold. That v2 reads the registered tables on the blobs sent is the coupling's (`TablesSent`, whose failure is a SHA-512
collision the read finder outputs, `Binding.finder_of_tables`).

What is left is `SessionRegistered`: the deployment's session at `θ`, where it runs one, is one v2 sets up on the
registered files and whose files pass `verify --zk`'s ports check (`Live.OpensReady`), wherever some accepted play
names them. `Instance.session` is a field of the deployment, and a session (`Live`) carries
facts no accepted play proves (the circuit's `Built`, the rate bound), so this is the deployment's to supply, not a
consequence of the instance's construction. `sessionAt_of_registered` gives `SessionAt` from it.
-/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

namespace Instance

variable (I : Instance d α)

/-- **v2's reading of committed files is their circuit and public files'**: files naming the same circuit and public
files are read alike (`readOf`, whose tables are the ones the circuit file registers). -/
theorem read_files {θ : ByteArray} {F F' : Array ByteArray} (hc : I.base.circuitFile θ F = I.base.circuitFile θ F')
    (hp : I.base.publicFile θ F = I.base.publicFile θ F') : I.base.read θ F = I.base.read θ F' := by
  simp only [Vocab.circuitFile, Vocab.publicFile, I.programDigest, I.publicDigest] at hc hp
  rw [I.read, I.read, readOf, readOf, hc, hp]

/-- **A session v2 sets up on some files it sets up on files naming the same circuit and public files, read alike.** -/
theorem setsUp_files {θ : ByteArray} {F F' : Array ByteArray} {A : Live}
    (hc : I.base.circuitFile θ F = I.base.circuitFile θ F') (hp : I.base.publicFile θ F = I.base.publicFile θ F')
    (hr : I.base.read θ F' = I.base.read θ F) (h : SetsUp I.base θ F A) : SetsUp I.base θ F' A :=
  { h with circuit := h.circuit.trans hc, publicFile := h.publicFile.trans hp, read := hr.trans h.read }

/-- **The deployment's session at the parameters `θ`, where it runs one, is one v2 sets up on the registered files**,
whose files pass `verify --zk`'s ports check (`Live.OpensReady`), wherever some play's committed files name the
registered circuit and public files and its sampled verdict accepts: a statement of `I.session θ` and the registered
files alone. -/
def SessionRegistered (k : Verity.Registration.Composite) : Prop :=
  ∀ θ A, I.session θ = some A → (∃ p, ParamsAt θ p) →
    (∃ F c o, I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) ∧
      I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) ∧
      Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    SetsUp I.base θ (I.base.registered θ) A ∧ A.OpensReady

/-- **`SessionAt` from the deployment's session at the registered files**: the session the deployment sets up on the
registered files (`SessionRegistered`) it sets up on an accepted play's committed files, which name the same circuit
and public files (`read_files`, `setsUp_files`). -/
theorem sessionAt_of_registered (k : Verity.Registration.Composite) (h : I.SessionRegistered k) : I.SessionAt k := by
  intro θ F A hA hP hc hp ⟨c, o, hreach, hv⟩
  obtain ⟨hs, hr⟩ := h θ A hA hP ⟨F, c, o, hc, hp, hreach, hv⟩
  exact ⟨I.setsUp_files hc.symm hp.symm (I.read_files hc hp) hs, hr⟩

end Instance

open scoped ENNReal in
/-- At no session v2's session at the registered files holds of every session, there being none. -/
theorem Instance.Honest.sessionRegistered_none (δ : ℝ≥0∞) (k : Verity.Registration.Composite) :
    (Instance.Honest.inst δ fun _ => none).SessionRegistered k := fun _ _ h => by simp [Instance.Honest.inst] at h

end Verity.Sampled
