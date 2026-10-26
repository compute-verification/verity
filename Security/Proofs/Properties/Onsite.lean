import Properties.Onsite
import Proofs.Properties.Composition
import Proofs.Properties.SampledProofs.OneStageZk

/-! # Proof of onsite's integrity property (`Properties.Onsite`)

Onsite on the caller's blobs is `product` of its components on them (`productB_on`), so `Properties.Composition`'s
`answerAccepts` and `acceptMeans` reduce its acceptance to each component's own, one-stage v2's among them, which
`Properties.SampledProofs.OneStageZk.acceptMeans` reads. The cases line up with the components by their ids
(`zip_ids`).
-/

open SampledProofs.OneStageZk
open Core Core.Composition

namespace Properties.Onsite

/-- Two lists whose ids are a third's line up with it in `zip`: each triple's three ids are one. -/
theorem zip_ids {α β γ : Type} (f : α → String) (g : β → String) (k : γ → String) :
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

theorem v2_mem : SampledProofs.OneStageZk.protocol ∈ Core.Onsite.components := by
  unfold Core.Onsite.components
  repeat (first | exact List.Mem.head _ | apply List.Mem.tail)

theorem integrityAccepts : IntegrityAccepts := by
  intro a b e blobs h
  let bs := Core.Blobs.ofArray blobs
  let cs := Core.Onsite.components.map (·.onBlobs bs)
  have h' : ((product "verity/onsite/v0" cs).answer a b e).verdict = .accept := h
  obtain ⟨ps, c, ev, hp, -, he, pub, prof, hacc⟩ := Properties.Composition.answerAccepts _ a b e h'
  obtain ⟨⟨hpid, heid, hfst, hparams⟩, hdec, -⟩ :=
    Properties.Composition.acceptMeans "verity/onsite/v0" cs ps c ev pub prof hacc
  have hv2 : SampledProofs.OneStageZk.protocol.onBlobs bs ∈ cs := List.mem_map_of_mem v2_mem
  rw [← hfst, List.mem_map] at hv2
  obtain ⟨x, hx, hx1⟩ := hv2
  have hids := zip_ids (·.id) (·.id) (·.id) cs ps ev hpid heid x hx
  have hd := hdec x hx
  rw [hx1] at hd
  obtain ⟨p, c', e', hp', hc', he', pub', prof', hchk⟩ := hd
  rw [hx1] at hids
  exact ⟨ps, ev, x.2.1, x.2.2, hp, he, (List.of_mem_zip hx).2, hids.1, hids.2, hparams x hx, p, c', e', pub', prof',
    hp', hc', he', Properties.SampledProofs.OneStageZk.acceptMeans p c' e' blobs pub' prof' hchk⟩

end Properties.Onsite
