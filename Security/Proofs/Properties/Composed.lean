import Properties.Composed
import CoreComposition.Blobs
import Proofs.Properties.Composition
import Proofs.Properties.SampledProofs.OneStageZk

/-!
# Proof that a composition with one-stage v2 among its components gives `Properties.Composed`'s conclusions

A `productB` on the caller's blobs is `product` of its components on them (`productB_on`), so `Properties.Composition`'s
`answerAccepts` and `acceptMeans` reduce its acceptance to each component's own, one-stage v2's among them, which
`Properties.SampledProofs.OneStageZk.acceptMeans` reads (`accepted`). The cases line up with the components by their
ids (`zip_ids`). A `registeredB` accepts only what `product` of its components accepts (`checkRegistered_check`), so
`registeredAccepted` reduces to the same step (`caseAccepted`). `caseRegistered` reads the registration check off
`Accepted`.
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

/-- **A composition's check with v2 among its components on blobs accepts only v2's case `Accepted`.** -/
theorem caseAccepted (id : String) (cs0 : List BlobProtocol) (hv2 : SampledProofs.OneStageZk.protocol ∈ cs0)
    (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed)
    (hacc : Core.Composition.check id (cs0.map (·.onBlobs (Core.Blobs.ofArray blobs))) ps c ev = .accept pub prof) :
    CaseAccepted ps ev blobs := by
  let bs := Core.Blobs.ofArray blobs
  let cs := cs0.map (·.onBlobs bs)
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
  exact ⟨x.2.1, x.2.2, (List.of_mem_zip hx).2, hids.1, hids.2, hparams x hx, p, c', e', pub', prof', hp', hc', he',
    Properties.SampledProofs.OneStageZk.acceptMeans p c' e' blobs pub' prof' hchk⟩

/-- **A composition with v2 among its components accepts only v2's case `Accepted`.** -/
theorem accepted (id : String) (cs0 : List BlobProtocol) (hv2 : SampledProofs.OneStageZk.protocol ∈ cs0)
    (a b e : ByteArray) (blobs : Array ByteArray) (h : ((productB id cs0).answer a b e blobs).verdict = .accept) :
    SampledProofsAccepted a e blobs := by
  let cs := cs0.map (·.onBlobs (Core.Blobs.ofArray blobs))
  have h' : ((product id cs).answer a b e).verdict = .accept := h
  obtain ⟨ps, c, ev, hp, -, he, pub, prof, hacc⟩ := Properties.Composition.answerAccepts _ a b e h'
  exact ⟨ps, ev, hp, he, caseAccepted id cs0 hv2 blobs ps c ev pub prof hacc⟩

/-- **A registered composition accepts only what `product` of its components accepts**, on the components' parameters
and cases, under the binding of their digests. -/
theorem checkRegistered_check {id : String} {R : Protocol} {s : String} {cs : List Protocol} {n : Named}
    {ps : List Named} {b d : ByteArray} {k : Case} {ev : List Case} {pub prof : Keyed}
    (h : checkRegistered id R s cs (n, ps) (b, d) (k, ev) = .accept pub prof) :
    Core.Composition.check id cs ps b ev = .accept pub prof := by
  unfold checkRegistered at h
  unfold Core.Composition.check
  split at h
  rename_i h1 h2 h3
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h2
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h3
  repeat' (split at h)
  all_goals first | (cases h; done) | skip
  all_goals
    have hrun : Core.Composition.run cs _ _ = _ := by assumption
    rw [hrun]
    exact h

/-- **A registered composition with v2 among its components accepts only v2's case `Accepted`.** -/
theorem registeredAccepted (id : String) (R : Protocol) (s : String) (cs0 : List BlobProtocol)
    (hv2 : SampledProofs.OneStageZk.protocol ∈ cs0) (a b e : ByteArray) (blobs : Array ByteArray)
    (h : ((registeredB id R s cs0).answer a b e blobs).verdict = .accept) :
    RegisteredSampledProofsAccepted a e blobs := by
  let cs := cs0.map (·.onBlobs (Core.Blobs.ofArray blobs))
  have h' : ((registered id R s cs).answer a b e).verdict = .accept := h
  obtain ⟨⟨r, ps⟩, ⟨c, d⟩, ⟨k, ev⟩, hp, -, he, pub, prof, hacc⟩ := Properties.Composition.answerAccepts _ a b e h'
  exact ⟨r, ps, k, ev, hp, he,
    caseAccepted id cs0 hv2 blobs ps c ev pub prof (checkRegistered_check (id := id) (R := R) (s := s) (cs := cs) hacc)⟩

/-- **v2's case `Accepted` checks the registration**: its circuit file is one of the blobs, named by the parameters'
digest. -/
theorem caseRegistered {ps : List Named} {ev : List Case} {blobs : Array ByteArray} (h : CaseAccepted ps ev blobs) :
    CaseRegistered ps blobs := by
  obtain ⟨n, x, hx, hn, _, _, p, _, _, _, _, hp', _, _, ⟨_, circuitB, _, _, _, _, hcirc, _⟩⟩ := h
  exact ⟨n, p, circuitB, (List.of_mem_zip hx).1, hn, hp', hcirc⟩

theorem registered {a e : ByteArray} {blobs : Array ByteArray} (h : SampledProofsAccepted a e blobs) :
    SampledProofsRegistered a blobs := by
  obtain ⟨ps, _, hp, _, hc⟩ := h
  exact ⟨ps, hp, caseRegistered hc⟩

theorem registeredRegistered {a e : ByteArray} {blobs : Array ByteArray} (h : RegisteredSampledProofsAccepted a e blobs) :
    RegisteredSampledProofsRegistered a blobs := by
  obtain ⟨r, ps, _, _, hp, _, hc⟩ := h
  exact ⟨r, ps, hp, caseRegistered hc⟩

end Properties.Composed
