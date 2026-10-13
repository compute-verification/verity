import Properties.Proofs.NetworkCertifier.Outcomes
import Properties.Proofs.NetworkCertifier.Extract

/-!
# What a certificate's digest and message pin down

Where SHA-512 has no collision on the strings both hash, two heads with one digest are one (`head_inj`), and a
certificate message's input names one certifier session, link-window and digest (`messageInput_inj`, with
`link_short` for the link's length). `Properties.Proofs.CertifierNetwork.Runs` uses them to match each record an
accepted unit holds to the one the certifier device issued.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier

open Bytes Tree GridProofs CheckProofs

theorem digest_ok {h : RecordHead} {dg : ByteArray} (e : h.digest = .ok dg) :
    ∃ enc x, h.encode = .ok enc ∧ Properties.Identity.taggedInput recordTag enc = .ok x ∧ dg = Sha512.hash x := by
  unfold RecordHead.digest at e
  cases he : h.encode with
  | error _ => rw [he] at e; exact (error_ne_ok e).elim
  | ok enc =>
    rw [he] at e
    have e' : Identity.taggedSha512 recordTag enc = .ok dg := e
    obtain ⟨x, hx, rfl⟩ := Properties.Identity.taggedSha512_ok.mp e'
    exact ⟨enc, x, rfl, hx, rfl⟩

theorem certificateMessage_ok {k s : ByteArray} {l : String} {d : Direction} {w : Int} {g m : ByteArray}
    (e : certificateMessage k s l d w g = .ok m) : ∃ x, messageInput k s l d w g = .ok x ∧ m = Sha512.hash x := by
  unfold certificateMessage at e
  cases hx : messageInput k s l d w g with
  | error _ => rw [hx] at e; exact (error_ne_ok e).elim
  | ok x => rw [hx] at e; exact ⟨x, rfl, (Except.ok.inj e).symm⟩

/-- Two heads with one digest, where SHA-512 has no collision on the strings both hash, are one. -/
theorem head_inj {X : ByteArray → Prop} (hX : CF X) {h h' : RecordHead} {dg : ByteArray}
    (hs : ∀ x, HeadString h x → X x) (hs' : ∀ x, HeadString h' x → X x) (e : h.digest = .ok dg)
    (e' : h'.digest = .ok dg) : h = h' := by
  obtain ⟨enc, x, he, hx, rfl⟩ := digest_ok e
  obtain ⟨enc', x', he', hx', hd⟩ := digest_ok e'
  have hxx := hX _ _ (hs _ ⟨enc, he, hx⟩) (hs' _ ⟨enc', he', hx'⟩) hd
  subst hxx
  have := taggedInput_inj hx hx'
  subst this
  exact encode_inj he he'

theorem link_short {h : RecordHead} {dg : ByteArray} (e : h.digest = .ok dg) : h.link.toUTF8.size < 2 ^ 16 := by
  obtain ⟨enc, -, he, -⟩ := digest_ok e
  exact encode_link he

theorem zip_index {p : Params} {e : Evidence} (hfit : EvidenceFits p e) (ℓ : Fin p.egress.size) {link : Link}
    (hl : e.links[ℓ.1]? = some link) : (p.egress[ℓ], link) ∈ p.egress.toList.zip e.links.toList := by
  have h1 : ℓ.1 < e.links.size := by rw [hfit.1]; exact ℓ.2
  have hz : ℓ.1 < (p.egress.toList.zip e.links.toList).length := by
    rw [List.length_zip, Array.length_toList, Array.length_toList]; omega
  have := List.getElem_mem hz
  rw [List.getElem_zip] at this
  rw [Array.getElem?_eq_getElem h1, Option.some.injEq] at hl
  simpa [← hl] using this

end Properties.NetworkCertifier
