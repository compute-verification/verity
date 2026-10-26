import Properties.Composed
import Proofs.Properties.Composition
import Proofs.Properties.SampledProofs.OneStageZk

/-!
# Proof that a composition with one-stage v2 among its components gives `Properties.Composed`'s conclusions

A `productB` on the caller's blobs is `product` of its components on them (`productB_on`), so `Properties.Composition`'s
`answerAccepts` and `acceptMeans` reduce its acceptance to each component's own, one-stage v2's among them, which
`Properties.SampledProofs.OneStageZk.acceptMeans` reads (`accepted`). The cases line up with the components by their
ids (`zip_ids`). `registered` reads the registration check off `Accepted`.
-/

open SampledProofs.OneStageZk
open Core Core.Composition

namespace Properties.Composed

/-- Two lists whose ids are a third's line up with it in `zip`: each triple's three ids are one. -/
theorem zip_ids {α : Type _} {β : Type _} {γ : Type _} (f : α → String) (g : β → String) (k : γ → String) :
    ∀ (l1 : List α) (l2 : List β) (l3 : List γ), l2.map g = l1.map f → l3.map k = l1.map f →
      ∀ x ∈ l1.zip (l2.zip l3), g x.2.1 = f x.1 ∧ k x.2.2 = f x.1
  | [], _, _, _, _, x, hx => by simp at hx
  | _ :: _, [], _, _, _, x, hx => by simp at hx
  | _ :: _, _ :: _, [], _, _, x, hx => by simp at hx
  | _ :: l1, _ :: l2, _ :: l3, h2, h3, x, hx => by
    simp only [List.map_cons, List.cons.injEq] at h2 h3
    simp only [List.zip_cons_cons, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ⟨h2.1, h3.1⟩
    · exact zip_ids f g k l1 l2 l3 h2.2 h3.2 x hx

/-- **A composition with v2 among its components accepts only v2's case `Accepted`.** -/
theorem accepted (id : String) (cs0 : List BlobProtocol) (hv2 : SampledProofs.OneStageZk.protocol ∈ cs0)
    (a b e : ByteArray) (blobs : Array ByteArray) (h : ((productB id cs0).answer a b e blobs).verdict = .accept) :
    SampledProofsAccepted a e blobs := by
  let bs := Core.Blobs.ofArray blobs
  let cs := cs0.map (·.onBlobs bs)
  have h' : ((product id cs).answer a b e).verdict = .accept := h
  obtain ⟨ps, c, ev, hp, -, he, pub, prof, hacc⟩ := Properties.Composition.answerAccepts _ a b e h'
  obtain ⟨⟨hpid, heid, hfst, hparams⟩, hdec, -⟩ := Properties.Composition.acceptMeans id cs ps c ev pub prof hacc
  have hv2' : SampledProofs.OneStageZk.protocol.onBlobs bs ∈ cs := List.mem_map_of_mem hv2
  rw [← hfst, List.mem_map] at hv2'
  obtain ⟨x, hx, hx1⟩ := hv2'
  have hids := zip_ids (fun P : Protocol => P.id) (fun n : Named => n.id) (fun x : Case => x.id) cs ps ev hpid heid
    x hx
  have hd := hdec x hx
  rw [hx1] at hd
  obtain ⟨p, c', e', hp', hc', he', pub', prof', hchk⟩ := hd
  rw [hx1] at hids
  exact ⟨ps, ev, x.2.1, x.2.2, hp, he, (List.of_mem_zip hx).2, hids.1, hids.2, hparams x hx, p, c', e', pub', prof',
    hp', hc', he', Properties.SampledProofs.OneStageZk.acceptMeans p c' e' blobs pub' prof' hchk⟩

/-- **v2's case `Accepted` checks the registration**: its circuit file is one of the blobs, named by the parameters'
digest. -/
theorem registered {a e : ByteArray} {blobs : Array ByteArray} (h : SampledProofsAccepted a e blobs) :
    SampledProofsRegistered a blobs := by
  obtain ⟨ps, ev, n, x, hp, _, hx, hn, _, _, p, _, _, _, _, hp', _, _, ⟨_, circuitB, _, _, _, _, hcirc, _⟩⟩ := h
  exact ⟨ps, n, p, circuitB, hp, (List.of_mem_zip hx).1, hn, hp', hcirc⟩

end Properties.Composed
