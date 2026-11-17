module

public import Proofs.Properties.Challenger.CanonRead
public import Proofs.Flock.Claims.CanonLine
public import Proofs.Properties.Composition
public import Proofs.Flock.Claims.CanonEq
public import Spot.ChallengerSession
import all Lean.Data.Json.Basic

/-! A release log line that is a sorted value's `canon` text reads as a value equivalent to it. -/

@[expose] public section

namespace Properties.Challenger.CanonLine

open Lean (Json JsonNumber)
open Spot (sortedKeys getD field?)
open FlockDeploy (JsonEquiv jsonForm)
open Properties.Challenger.CanonRead (sortedKeys_arr sortedKeys_obj increasing_iff mapM_forall₂ forall₂_map_eq
  forall₂_imp qsort_keys)

theorem equiv_num {x : Json} {n : JsonNumber} (h : JsonEquiv x (.num n)) : x = .num n := by
  cases x with
  | num m => cases h; rfl
  | arr a => rw [JsonEquiv, CanonRead.jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, FlockDeploy.jsonForm_obj] at h; cases h
  | _ => cases h

theorem jsonStrIs_equiv {a b : Json} (h : JsonEquiv a b) (s : String) :
    Spot.jsonStrIs a s = Spot.jsonStrIs b s := by
  by_cases hb : ∃ t, b = .str t
  · obtain ⟨t, rfl⟩ := hb
    rw [CanonRead.equiv_str h]
  · have ha : ¬ ∃ t, a = .str t := fun ⟨t, ht⟩ => hb ⟨t, CanonRead.equiv_str (ht ▸ h.symm)⟩
    cases a <;> cases b <;> first | rfl | exact absurd ⟨_, rfl⟩ ha | exact absurd ⟨_, rfl⟩ hb

theorem int?_equiv {a b : Json} (h : JsonEquiv a b) : Spot.int? a = Spot.int? b := by
  by_cases hb : ∃ n, b = .num n
  · obtain ⟨n, rfl⟩ := hb
    rw [equiv_num h]
  · have ha : ¬ ∃ n, a = .num n := fun ⟨n, hn⟩ => hb ⟨n, equiv_num (hn ▸ h.symm)⟩
    cases a <;> cases b <;> first | rfl | exact absurd ⟨_, rfl⟩ ha | exact absurd ⟨_, rfl⟩ hb

theorem nat?_equiv {a b : Json} (h : JsonEquiv a b) : Spot.nat? a = Spot.nat? b := by
  unfold Spot.nat?
  rw [int?_equiv h]

theorem event_ok {line : String} {j : Json} (h : Spot.Challenger.event line = .ok j) :
    Flock.canon j = .ok line ∧ sortedKeys j = true := by
  unfold Spot.Challenger.event at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw, beq_iff_eq] at h
  split at h
  · split at h
    · rename_i j' _
      split at h
      · rename_i text htext
        split at h
        · rename_i heq
          split at h
          · rename_i hs
            cases h
            exact ⟨heq ▸ htext, hs⟩
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

theorem event_equiv {line : String} {j d : Json} (h : Spot.Challenger.event line = .ok j)
    (hd : sortedKeys d = true) (hc : Flock.canon d = .ok line) : JsonEquiv j d :=
  canon_inj (event_ok h).2 hd (event_ok h).1 hc

theorem event_getD {line : String} {j d : Json} (h : Spot.Challenger.event line = .ok j)
    (hd : sortedKeys d = true) (hc : Flock.canon d = .ok line) (k : String) : JsonEquiv (getD j k) (getD d k) :=
  CanonRead.getD_equiv (event_ok h).2 hd (event_equiv h hd hc) k

end Properties.Challenger.CanonLine
