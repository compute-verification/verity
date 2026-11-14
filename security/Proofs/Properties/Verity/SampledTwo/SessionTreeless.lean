module

public import Proofs.Properties.Verity.SampledTwo.SessionCut

/-!
# On two windows the first session samples no challenge tree, from the form's check

`SessionCut` proves the sessions' challenges at the cut's read from one fact about the first session,
`FirstTreeless`: on a play the two-stage form accepts whose two sessions are on two windows, the first session's
releases as the cut reads them hold no tree's `Hello`. This module proves it from the form's own refusal.

The form's check refuses, on two windows, a first session's log that holds a tree's line
(`VStar.Recursive.SampledTwo.firstTreeless`, `Spot.Challenger.treeLine`), so an accepted play's first session's log
holds none (`accepted_treeless`, `firstTreeless_lines`). The cut reads the same lines:

* `formLines_of`: the lines the form reads of the first session's log (`VStar.Recursive.SampledTwo.firstLines`, from
  the evidence's `coins`) are the lines the cut reads (`firstLines`, from `firstCoins`), where both name one session.
* `firstTreeless_of_check`: `FirstTreeless C`, where the play's case is the one sent (`hC`, in the words of
  `StageCut`'s `CaseIsSent C`) and the cut's evidence for the first stage names the session the form's does (`hS`,
  the `"session"` part of `StageAlone`'s `SessionMatches C`).

With it the premise `hfree` of `SessionCut.interiorLogged_read` and `SessionCut.coinsRead_read` follows from `hC` and
`hS`. Where `firstCoins e` is the evidence's `coins` itself, `hS` is `rfl`.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Properties.Challenger Spot.Challenger

/-- **The form reads the lines the cut reads of the first session's log**, where the cut's evidence for the first
stage names the session the form's does: the two reads are one function of that session's log. -/
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

/-- **`FirstTreeless`, from the form's check**: the check refuses, on two windows, a first session's log that holds a
tree's line (`VStar.Recursive.SampledTwo.firstTreeless`), so on an accepted play no line the cut reads of that log is
a tree's, and its read shows only fresh challenges (`firstTreeless_of_plain`). The play's case is the one sent (`hC`,
as in `secondRead_interior`), and the cut's evidence for the first stage names the session the form's does (`hS`). -/
theorem firstTreeless_of_check (C : CaseOf)
    (hC : ∀ o k p c e, C o = some k → Decoded k p c e →
      caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs)
    (hS : ∀ o, FormAccepts C o → ∀ p c e c₂, caseOf o.sent = .ok (p, c, e) → interiorOf e = .ok c₂ →
      Spot.getD (firstCoins e) "session" = Spot.getD e.coins "session") :
    FirstTreeless C := by
  refine firstTreeless_of_plain C ?_
  intro o hacc p c e key d₁ d₂ ls h1 _ h3 h6 hw hls
  have h6' := h6
  unfold secondDraws at h6'
  obtain ⟨c₂, hc₂, -⟩ := Verity.Sampled.Binding.obind_some h6'
  have h5 : interiorOf e = .ok c₂ := toOption_ok hc₂
  have hs := hS o hacc p c e c₂ h1 h5
  obtain ⟨k, hk, p', c', e', pub, prof, hdec, hchk⟩ := hacc
  obtain ⟨hcase, -⟩ := hC o k p' c' e' hk hdec
  have hpce : (p, c, e) = (p', c', e') := Except.ok.inj (h1.symm.trans hcase)
  obtain ⟨rfl, hce⟩ := Prod.mk.inj hpce
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hce
  obtain ⟨bp, hb, hw₁, -⟩ := firstDraws_window h3
  have hbp : bp = p.boundary := (Except.ok.inj hb).symm
  rw [hbp] at hw₁
  have hw₂ := source_window (secondDraws_eq h5 h6)
  exact VStar.Recursive.SampledTwo.firstTreeless_lines (VStar.Recursive.SampledTwo.accepted_treeless hchk) hw₁ hw₂ hw
    (formLines_of hs hls)

end Verity.SampledTwo.SessionCoins
