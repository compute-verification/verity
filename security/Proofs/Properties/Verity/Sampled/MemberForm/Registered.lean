module

public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Verity.Sampled.Window.FilesSent

/-! The members' files v2 reads on the blobs sent are the registered ones. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open FlockSoundness.Refine (ebind_ok)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

open Classical in
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
      exact (getD_inj h64 (Properties.Certifier.Bytes.hash_size f) e).symm
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

def RegisteredHolds (V : Vocab d α) : Prop :=
  ∀ θ, ∀ h ∈ memberDigests θ, (∃ f : ByteArray, h = some (Verity.sha512 f.data.toList)) →
    ∃ f ∈ V.registered θ, h = some (Verity.sha512 f.data.toList)

theorem mem_memberDigests {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {ms : Array _root_.Spot.OneStageZk.Member} (hp : ParamsAt θ p) (hms : p.members = some ms)
    {vm : _root_.Spot.OneStageZk.Member} (hvm : vm ∈ ms) :
    digestOf vm.own.circuit ∈ memberDigests θ ∧ digestOf vm.own.pub ∈ memberDigests θ := by
  unfold memberDigests
  simp only [paramsOf_eq hp, Option.bind_some, hms]
  exact ⟨List.mem_flatMap.2 ⟨vm, Array.mem_toList_iff.2 hvm, List.mem_cons_self⟩,
    List.mem_flatMap.2 ⟨vm, Array.mem_toList_iff.2 hvm, List.mem_cons_of_mem _ List.mem_cons_self⟩⟩

theorem readsFiles_of_sent {V : Vocab d α} {o : Exchange} (hR : RegisteredHolds V) (hf : FilesSent V o)
    {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    (hp : ParamsAt o.params p) (hms : p.members = some ms) {vm : _root_.Spot.OneStageZk.Member}
    (hvm : vm ∈ ms) :
    ReadsFiles (V.registered o.params) vm (_root_.Core.Blobs.ofArray o.sent.blobs) := by
  obtain ⟨hc, hpb⟩ := mem_memberDigests hp hms hvm
  exact ⟨fun _ hg => get_registered (hf _ hc) (hR _ _ hc) hg,
    fun _ hg => get_registered (hf _ hpb) (hR _ _ hpb) hg⟩

end Verity.Sampled.MemberForm
