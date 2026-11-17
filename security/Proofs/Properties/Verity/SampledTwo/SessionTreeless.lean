module

public import Proofs.Properties.Verity.SampledTwo.SessionCut

/-! On two windows the first session samples no coin tree (`FirstTreeless`), from the form's own refusal. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Properties.Challenger Spot.Challenger

theorem formLines_of {e : VStar.Recursive.SampledTwo.Evidence} {l : List String}
    (hs : Spot.getD (firstCoins e) "session" = Spot.getD e.coins "session") (h : firstLines e = .ok l) :
    VStar.Recursive.SampledTwo.firstLines e = .ok l := by
  unfold firstLines at h
  split at h
  · rename_i ls hls
    have hls' : Spot.getD (Spot.getD e.coins "session") "log" = Json.arr ls := by
      rw [← hs]
      exact hls
    have hl : ls.toList = l.map Json.str :=
      mapM_str_inv (fun j x hj => by cases j <;> cases hj; rfl) _ _ h
    unfold VStar.Recursive.SampledTwo.firstLines
    rw [hls']
    show ls.toList.mapM _ = _
    rw [hl]
    exact Verity.Sampled.Binding.mapM_map_str (fun _ => rfl) l
  · cases h

theorem firstTreeless_of_check (C : CaseOf)
    (hC : ∀ o k p c e, C o = some k → Decoded k p c e →
      caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs)
    (hS : ∀ o, FormAccepts C o → ∀ p c e c₂, caseOf o.sent = .ok (p, c, e) → interiorOf e = .ok c₂ →
      Spot.getD (firstCoins e) "session" = Spot.getD e.coins "session") :
    FirstTreeless C := by
  refine firstTreeless_of_plain C ?_
  intro o hacc p c e key d₁ d₂ ls h1 _ h3 h6 hw hls
  have h6' := h6
  unfold secondSamples at h6'
  obtain ⟨c₂, hc₂, -⟩ := Verity.Sampled.Binding.obind_some h6'
  have h5 : interiorOf e = .ok c₂ := toOption_ok hc₂
  have hs := hS o hacc p c e c₂ h1 h5
  obtain ⟨k, hk, p', c', e', pub, prof, hdec, hchk⟩ := hacc
  obtain ⟨hcase, -⟩ := hC o k p' c' e' hk hdec
  have hpce : (p, c, e) = (p', c', e') := Except.ok.inj (h1.symm.trans hcase)
  obtain ⟨rfl, hce⟩ := Prod.mk.inj hpce
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hce
  obtain ⟨bp, hb, hw₁, -⟩ := firstSamples_executionPhase h3
  have hbp : bp = p.boundary := (Except.ok.inj hb).symm
  rw [hbp] at hw₁
  have hw₂ := source_executionPhase (secondSamples_eq h5 h6)
  exact VStar.Recursive.SampledTwo.firstTreeless_lines (VStar.Recursive.SampledTwo.accepted_treeless hchk) hw₁ hw₂ hw
    (formLines_of hs hls)

end Verity.SampledTwo.SessionCoins
