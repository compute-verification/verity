module

public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.AddressFacts.Row

/-! Flock's parse of a registered-reads file keeps each port's reads (`portsOf_reads`). -/

@[expose] public section

namespace SpotFlock

open Lean (Json)
open FlockSoundness.Refine

namespace AddressRow

theorem nat_num (n : ℕ) : Spot.nat? (Json.num ⟨Int.ofNat n, 0⟩) = some n := by
  simp [Spot.nat?, Spot.int?]

theorem nat_of_asNat {x : Json} {c : String} {pos : ℕ} (h : _root_.Flock.J.asNat x c = .ok pos) :
    Spot.nat? x = some pos :=
  (congrArg Spot.nat? (FlockComplete.JsonRT.asNat_ok h)).trans (nat_num pos)

theorem getD_of_not_has {j : Json} {k : String} (h : _root_.Flock.J.has j k = false) : Spot.getD j k = .null := by
  cases hg : j.getObjVal? k with
  | ok v => simp [_root_.Flock.J.has, hg] at h
  | error err => simp only [Spot.getD, Spot.field?, hg, Except.toOption, Option.getD_none]

theorem ret_guard {γ : Type} (c : Prop) [Decidable c] (msg : String) (k : Unit → _root_.Flock.V γ) (Q : γ → Prop)
    (h : Ret (k ()) Q) : Ret (if c then (throw msg : _root_.Flock.V Unit) >>= fun r => k r else k ()) Q :=
  ret_ite c _ _ Q (ret_bind' _ _ Q fun _ => h) h

theorem ret_join {α γ : Type} (c : Prop) [Decidable c] (a b : _root_.Flock.V α) (k : α → _root_.Flock.V γ)
    (Q : γ → Prop) (h : ∀ x, (if c then a else b) = .ok x → Ret (k x) Q) :
    Ret (if c then a >>= fun y => k y else b >>= fun y => k y) Q := by
  by_cases hc : c
  · rw [ite_eq_left hc]
    exact ret_bind a _ Q fun x hx => h x (by rw [ite_eq_left hc]; exact hx)
  · rw [ite_eq_right hc]
    exact ret_bind b _ Q fun x hx => h x (by rw [ite_eq_right hc]; exact hx)

theorem readsAt_of_fields {e : Json} {rs : Array Json} {first u pos : ℕ} {x : Json}
    (hF : (Spot.getD e "reads_from" = .null ∧ first = 0) ∨
      Spot.getD e "reads_from" = .num ⟨Int.ofNat first, 0⟩)
    (hR : Spot.getD e "reads" = .arr rs) (hle : first ≤ u) (hx : rs[u - first]? = some x)
    (hn : Spot.nat? x = some pos) : readsAt e u = some pos := by
  rcases hF with ⟨hF, h0⟩ | hF
  · subst h0
    have hx' : rs[u]? = some x := by simpa using hx
    simp only [readsAt, hF, hR]
    simp [hx', hn]
  · have hfst : Spot.nat? (Json.num ⟨Int.ofNat first, 0⟩) = some first := nat_num first
    simp only [readsAt, hF, hR, hfst]
    simp [hle, hx, hn]

theorem readsAt_of_parse {e : Json} {c : String} {rs : Array Json} {reg : _root_.Flock.Registered.Port}
    {f : Json → _root_.Flock.V (Option ℕ)}
    (hfirst : (if _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true then
        _root_.Flock.J.nat e _root_.Flock.Registered.READS_FROM_KEY c else pure 0) = .ok reg.readsFrom)
    (harr : _root_.Flock.J.arr e "reads" c = .ok rs) (hmap : rs.mapM f = .ok reg.readsRel) {u pos : ℕ}
    (h : reg.readAt u = some (some pos))
    (hf : ∀ x pos, f x = .ok (some pos) → Spot.nat? x = some pos) : readsAt e u = some pos := by
  have hF : (Spot.getD e "reads_from" = .null ∧ reg.readsFrom = 0) ∨
      Spot.getD e "reads_from" = .num ⟨Int.ofNat reg.readsFrom, 0⟩ := by
    by_cases hh : _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true
    · rw [ite_eq_left hh] at hfirst
      exact Or.inr (getD_of_ok (FlockComplete.JsonRT.nat_ok hfirst))
    · rw [ite_eq_right hh] at hfirst
      have h0 : reg.readsFrom = 0 := (Except.ok.inj hfirst).symm
      have hh' : _root_.Flock.J.has e "reads_from" = false := by
        cases hb : _root_.Flock.J.has e "reads_from" with
        | false => rfl
        | true => exact absurd hb hh
      exact Or.inl ⟨getD_of_not_has hh', h0⟩
  have hR := getD_of_ok (FlockComplete.JsonRT.arr_ok harr)
  unfold _root_.Flock.Registered.Port.readAt at h
  by_cases hlt : u < reg.readsFrom
  · rw [ite_eq_left hlt] at h
    cases h
  · rw [ite_eq_right hlt] at h
    obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hmap
    obtain ⟨hk, hget⟩ := Array.getElem?_eq_some_iff.1 h
    have hk' : u - reg.readsFrom < rs.size := by omega
    have hfx := hall (u - reg.readsFrom) hk' hk
    rw [hget] at hfx
    exact readsAt_of_fields hF hR (Nat.le_of_not_lt hlt) (Array.getElem?_eq_getElem hk') (hf _ _ hfx)

theorem portsOf_reads {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.portsOf j = .ok own) :
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
      ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg ∧
        ∀ u pos, reg.readAt u = some (some pos) → readsAt e u = some pos := by
  revert own
  show Ret (_root_.Flock.Registered.portsOf j) fun own => ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
    ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg ∧
      ∀ u pos, reg.readAt u = some (some pos) → readsAt e u = some pos
  rw [_root_.Flock.Registered.portsOf.eq_def]
  ret_walk [Array.mapM]
  all_goals
    obtain ⟨hall, -⟩ := mapM_ret (by assumption)
        (fun (kv : String × Json) (reg : _root_.Flock.Registered.Port) => PortFields kv.1 kv.2 reg ∧
          ∀ u pos, reg.readAt u = some (some pos) → readsAt kv.2 u = some pos) fun kv => by
      obtain ⟨nm, e⟩ := kv
      dsimp (config := { zeta := false }) only
      refine ret_guard _ _ _ _ ?_
      refine ret_guard _ _ _ _ ?_
      refine ret_join _ _ _ _ _ fun rf hrf => ?_
      ret_walk [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.unhex, _root_.Flock.J.arr, Array.mapM]
      all_goals
        refine ⟨?_, fun u pos hu => ?_⟩
        · exact ⟨rfl, ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "root" _ = .ok _›, ‹_›⟩,
            ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "domain" _ = .ok _›, ‹_›⟩,
            FlockComplete.JsonRT.nat_ok ‹_root_.Flock.J.nat e "leaves" _ = .ok _›,
            FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "value" _ = .ok _›⟩
        · refine readsAt_of_parse hrf ‹_root_.Flock.J.arr e "reads" _ = .ok _› (by assumption) hu
            fun x pos hx => ?_
          split at hx
          · cases hx
          · exact nat_of_asNat (map_some_ok hx)
    refine ⟨_, by assumption, fun reg hreg => ?_⟩
    obtain ⟨⟨nm, e⟩, hkv, hP, hrd⟩ := hall reg hreg
    obtain rfl : reg.name = nm := hP.1
    exact ⟨e, hkv, hP, hrd⟩

end AddressRow

end SpotFlock
