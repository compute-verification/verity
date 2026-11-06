module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.SampledDraw
public import Proofs.Properties.SampledProofs.Tables

/-!
# v2's session at the registered files, from v2's refusal of a lookup table (`Instance.SessionAt`)

v2's parameters register its circuit and public files and no lookup table, and v2 refuses blobs holding a pinned MUFU
table before it sets up a session (`Properties.SampledProofs.TablesRefused`). So on an accepted play the committed files
hold none (`tablesOf_files_nil`), and v2's reading of them (`readOf`) is the one of any files naming the same circuit and
public files and holding no table (`read_files`): a session v2 sets up on such files is set up on the play's
(`setsUp_files`). `SessionAt` then follows from a statement of the deployment's session at each parameters alone
(`SessionRegistered`, `sessionAt_of_registered`), and the tables' finder `covers` reads (`TablesAgree`) is not needed for it.
-/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An accepted play's committed files hold no lookup table**: v2's case in it accepts on the blobs sent, the
committed files then the evidence's, which hold none (`TablesRefused`), and every table the committed files hold the
blobs hold (`tablesOf_extends`). -/
theorem tablesOf_files_nil {V : Vocab d α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    {p₀ : _root_.SampledProofs.OneStageZk.Params} (hθ : ParamsAt o.params p₀) (hv : V.Verdict k .sampled o) :
    _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.files) = [] := by
  obtain ⟨_, _, _, _, _, _, _, _, -, -, -, -, -, -, pub, prof, hck⟩ := case_drew hv.2.1 hθ (verdict_case hv)
  have h0 : _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray (o.files ++ o.blobs)) = [] :=
    Properties.SampledProofs.SecurityProofs.tablesRefused.2 _ _ _ _ _ _ _ hck
  cases hT : _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.files) with
  | nil => rfl
  | cons x xs =>
    have h := tablesOf_extends o.files o.blobs x.1 x (by rw [hT]; simp)
    rw [h0] at h
    cases h

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

/-- **The deployment's session at the parameters `θ` is one v2 sets up on the registered circuit and public files with
no lookup table**, wherever some play's committed files name those files and its sampled verdict accepts: a statement
of `I.session θ` alone, which a deployment's session fills at files of its own choosing, the registered files where
they hold no pinned table. -/
def SessionRegistered (k : Verity.Registration.Composite) : Prop :=
  ∀ θ, (∃ p, ParamsAt θ p) →
    (∃ F c o, I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) ∧
      I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) ∧
      Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    ∃ A, I.session θ = some A ∧ ∃ F₀, I.base.circuitFile θ F₀ = I.base.circuitFile θ (I.base.registered θ) ∧
      I.base.publicFile θ F₀ = I.base.publicFile θ (I.base.registered θ) ∧
      _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F₀) = [] ∧ SetsUp I.base θ F₀ A

/-- **`SessionAt` from the deployment's session at the registered files**: an accepted play's committed files hold no
lookup table (`tablesOf_files_nil`), so the session the deployment sets up at `θ` is set up on them
(`setsUp_files`), whatever tables the registered files hold. -/
theorem sessionAt_of_registered (k : Verity.Registration.Composite) (h : I.SessionRegistered k) : I.SessionAt k := by
  intro θ F hP hc hp _ ⟨c, o, hreach, hv⟩
  obtain ⟨A, hA, F₀, hc₀, hp₀, ht₀, hs₀⟩ := h θ hP ⟨F, c, o, hc, hp, hreach, hv⟩
  obtain ⟨hθ, hF, -⟩ := Binding.play_reaches hreach
  obtain ⟨p, hpθ⟩ := hP
  have ht : _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) = [] := by
    rw [← hF]
    exact tablesOf_files_nil (hθ ▸ hpθ) hv
  exact ⟨A, hA, I.setsUp_files (hc₀.trans hc.symm) (hp₀.trans hp.symm)
    (I.read_files (hc.trans hc₀.symm) (hp.trans hp₀.symm) (ht.trans ht₀.symm)) hs₀⟩

end Instance

end Verity.Sampled
