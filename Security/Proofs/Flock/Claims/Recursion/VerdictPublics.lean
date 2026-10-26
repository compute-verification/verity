import Proofs.Flock.Claims.Recursion.VerdictTargetL
import Proofs.Flock.Claims.Recursion.CallsAgree

/-!
# `NonVacuous`'s public-words target

`Top.targetPublics`: wherever the `after` check passes at a decision, custody makes the record's public words the link's
(`custody_ok`), which are empty at every decision (`decR_publics`), and the decoder reads the record's public words as
`liveOfJ` does (`decodeJWithK_publicsJ`). So `TargetPublics` holds at every `P`.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine
open Lean (Json)

/-- `decodeWithK_pubs` at the parsed value. -/
theorem decodeJWithK_pubs (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∃ l pubs, Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList sess.publics.toList := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption, ?_⟩
    refine List.Forall₂.imp (fun t p h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
    try dsimp only at h
    refine ⟨fun hn => ?_, fun hn => ?_⟩
    · rw [ite_eq_left hn] at h
      obtain ⟨_, hw, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at hw
      subst hw
      split at h
      · repeat' obtain ⟨_, _, h⟩ := ebind_ok h
        exact (Except.ok.inj h).symm
      · exact (Except.ok.inj h).symm
    · rw [ite_eq_right (by simp [hn])] at h
      obtain ⟨s, hs', h⟩ := ebind_ok h
      obtain ⟨p', hp', h⟩ := ebind_ok h
      refine ⟨s, hs', ?_⟩
      split at h
      · repeat' obtain ⟨_, _, h⟩ := ebind_ok h
        rw [← Except.ok.inj h]
        exact hp'
      · rw [← Except.ok.inj h]
        exact hp'

/-- `liveOf_pubs` at the parsed value. -/
theorem liveOfJ_pubs {spec : SessionSpec} {j : Json} {lv : Live} (h : FlockDeploy.liveOfJ spec j = .ok lv) :
    ∃ l pubs, Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList lv.publics.toList := by
  unfold FlockDeploy.liveOfJ at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    exact ⟨_, _, by assumption, by assumption,
      List.Forall₂.imp (fun t p h => pubField_of h) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))⟩

/-- **The decoded session's public words are `liveOfJ`'s**, at the parsed value. -/
theorem decodeJWithK_publicsJ (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} {lv : Live} (hd : Record.decodeJWithK keep hs spec j = .ok sess)
    (hl : FlockDeploy.liveOfJ spec j = .ok lv) : sess.publics = lv.publics := by
  obtain ⟨l, pubs, hl1, hp1, hf⟩ := decodeJWithK_pubs keep hs hd
  obtain ⟨l', pubs', hl1', hp1', hf'⟩ := liveOfJ_pubs hl
  rw [hl1] at hl1'
  cases hl1'
  rw [hp1] at hp1'
  cases hp1'
  exact Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := PubField pubs) (fun _ _ _ => pubField_unique) hf hf')

end FlockDeploy.KS

namespace RecDeploy.Top

open Claims (Msg)
open FlockDeploy (Coin FinalP)

/-- **`TargetPublics` holds**: the `after` check's custody makes the decoded public words the link's, which are empty
at a decision. -/
theorem targetPublics (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop) : TargetPublics rd P := by
  intro x ms cs hA D
  obtain ⟨-, hr, hb, -⟩ := hA
  obtain ⟨-, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checksAtJ_ok (after_of_base hr hb D D.spec D.sts D.setup)
  have hd := D.decode
  rw [Flock.Record.decodeJWith_eq_decodeJWithK] at hd
  have e := (Claims.Bridge.Custody.custody_ok hcu).2.1
  rw [FlockDeploy.decR_publics D.dec] at e
  rw [FlockDeploy.KS.decodeJWithK_publicsJ id _ hd hlv, ← Array.all_toList, e, List.all_eq_true]
  intro p hp
  obtain ⟨_, _, rfl⟩ := List.mem_map.1 hp
  rfl

end RecDeploy.Top
