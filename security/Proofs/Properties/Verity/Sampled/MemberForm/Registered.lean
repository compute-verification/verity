module

public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Verity.Sampled.Window.FilesSent

/-!
# The members' files v2 reads on the blobs sent are the registered ones

v2 looks each member's circuit file and public file up on the blobs sent by its digest (`Core.Blobs.get`); the
window's members are the parameters' members on the registered files (`MemberOn`, by `Verity.namedFile`). This file
ties the two at a play whose blobs sent hold each member's registered file (`FilesSent`).

* **`Blobs.get` returns the first blob with the digest** (`get_blobWith`), the one `blobWith` finds, and so the
  registered file the digest names, where the first blob with it is the first registered file with it and the
  registered files hold one (`get_registered`).
* **The registered files hold each member's file** (`RegisteredHolds`): at each member's digest that some file has,
  the registered files hold a file with it. No check of the verifier reads the registered files: this is the
  registration's, whose parameters hold the SHA-512s of the files it registers.
* **v2 reads each member's files at its registered files** (`readsFiles_of_sent`): `ReadsFiles` on the blobs sent,
  for every member of the parameters.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open FlockSoundness.Refine (ebind_ok)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

open Classical in
/-- **`Blobs.get` returns the first blob with the digest**: the first whose SHA-512 is the digest the hex names
(`blobWith`). -/
theorem get_blobWith {S : Array ByteArray} {s what : String} {b : ByteArray}
    (hg : (_root_.Core.Blobs.ofArray S).get s what = .ok b) : blobWith (digestOf s) S = some b := by
  unfold _root_.Core.Blobs.get at hg
  obtain ⟨dg, hu, hg⟩ := ebind_ok hg
  dsimp only at hg
  by_cases c0 : (dg.size != 64 || Flock.hexOf dg != s) = true
  · rw [ite_eq_left c0] at hg
    obtain ⟨_, h0, _⟩ := ebind_ok hg
    cases h0
  rw [ite_eq_right c0] at hg
  have h64 : dg.size = 64 := by
    by_contra hne
    exact c0 (by simp [hne])
  have hq : ∀ f : ByteArray,
      (Flock.Sha512.hash f == dg) = decide (digestOf s = some (Verity.sha512 f.data.toList)) := by
    intro f
    rw [digestOf_unhex s dg hu, sha512_data, Bool.eq_iff_iff, beq_bytes, decide_eq_true_iff, Option.some.injEq]
    constructor
    · rintro rfl
      rfl
    · intro e
      exact (getD_inj h64 (Properties.NetworkCertifier.Bytes.hash_size f) e).symm
  have hfind : (_root_.Core.Blobs.ofArray S).entries.find? (fun x => x.1 == dg) =
      (blobWith (digestOf s) S).map fun f => (Flock.Sha512.hash f, f) := by
    show (S.map fun b => (Flock.Sha512.hash b, b)).find? (fun x => x.1 == dg) = _
    unfold blobWith
    simp only [Array.find?_map, Function.comp_def, hq]
  simp only [hfind] at hg
  cases hb : blobWith (digestOf s) S with
  | none =>
    rw [hb, Option.map_none] at hg
    cases hg
  | some f =>
    rw [hb, Option.map_some] at hg
    cases hg
    rfl

open Classical in
/-- **A blob v2 finds by a digest is the registered file the digest names**, where the first blob with the digest
is the first registered file with it and the registered files hold one. -/
theorem get_registered {S R : Array ByteArray} {s what : String} {b : ByteArray}
    (hs : ∀ a r, blobWith (digestOf s) S = some a → blobWith (digestOf s) R = some r → a = r)
    (hR : (∃ f : ByteArray, digestOf s = some (Verity.sha512 f.data.toList)) →
      ∃ f ∈ R, digestOf s = some (Verity.sha512 f.data.toList))
    (hg : (_root_.Core.Blobs.ofArray S).get s what = .ok b) : b = Verity.namedFile (digestOf s) R := by
  have hb := get_blobWith hg
  obtain ⟨f, hf, hfd⟩ := hR ⟨b, blobWith_digest hb⟩
  obtain ⟨r, hr⟩ : ∃ r, blobWith (digestOf s) R = some r :=
    Option.isSome_iff_exists.1 (Array.find?_isSome.2 ⟨f, hf, decide_eq_true hfd⟩)
  rw [hs b r hb hr]
  unfold Verity.namedFile
  unfold blobWith at hr
  rw [hr, Option.getD_some]

/-- **The registered files hold each member's file**: at each member's digest in the parameters `θ`
(`memberDigests`) that some file has, the registered files hold a file with it. A statement of the registered files
alone, which no check of the verifier reads: the registration's parameters hold the SHA-512s of the files it
registers. -/
def RegisteredHolds (V : Vocab d α) : Prop :=
  ∀ θ, ∀ h ∈ memberDigests θ, (∃ f : ByteArray, h = some (Verity.sha512 f.data.toList)) →
    ∃ f ∈ V.registered θ, h = some (Verity.sha512 f.data.toList)

/-- A member of the parameters has its circuit file's digest and its public file's among the members' digests. -/
theorem mem_memberDigests {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {ms : Array _root_.Spot.OneStageZk.Member} (hp : ParamsAt θ p) (hms : p.members = some ms)
    {vm : _root_.Spot.OneStageZk.Member} (hvm : vm ∈ ms) :
    digestOf vm.own.circuit ∈ memberDigests θ ∧ digestOf vm.own.pub ∈ memberDigests θ := by
  unfold memberDigests
  simp only [paramsOf_eq hp, Option.bind_some, hms]
  exact ⟨List.mem_flatMap.2 ⟨vm, Array.mem_toList_iff.2 hvm, List.mem_cons_self⟩,
    List.mem_flatMap.2 ⟨vm, Array.mem_toList_iff.2 hvm, List.mem_cons_of_mem _ List.mem_cons_self⟩⟩

/-- **v2 reads each member's files at its registered files**, at a play whose blobs sent hold each member's
registered file (`FilesSent`), where the registered files hold the members' files (`RegisteredHolds`): the blob v2
finds by a member's digest is the registered file the digest names. -/
theorem readsFiles_of_sent {V : Vocab d α} {o : Exchange} (hR : RegisteredHolds V) (hf : FilesSent V o)
    {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    (hp : ParamsAt o.params p) (hms : p.members = some ms) {vm : _root_.Spot.OneStageZk.Member}
    (hvm : vm ∈ ms) :
    ReadsFiles (V.registered o.params) vm (_root_.Core.Blobs.ofArray o.sent.blobs) := by
  obtain ⟨hc, hpb⟩ := mem_memberDigests hp hms hvm
  exact ⟨fun _ hg => get_registered (hf _ hc) (hR _ _ hc) hg,
    fun _ hg => get_registered (hf _ hpb) (hR _ _ hpb) hg⟩

end Verity.Sampled.MemberForm
