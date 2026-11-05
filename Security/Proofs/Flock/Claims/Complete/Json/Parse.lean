import Flock.JsonParse
import Std.Data.TreeMap.Raw.WF
import Proofs.Flock.Claims.Complete.Json.Spec

/-!
# The verifier's parser builds well-formed trees

`Flock.JsonParse.parse` is total, so what completeness reads of its trees is a theorem (`parse_wf`): an object is
`∅` with its members inserted one by one (`Std.TreeMap.Raw.WF.insert`), by induction on the parser's fuel through its
three mutual readers (`wf_all`).
-/

namespace FlockComplete.JsonRT

open Lean (Json)

open Flock.JsonParse

/-- A literal reads as its value. -/
theorem word_ok {w : String} {j : Json} {cs : List Char} {o : Nat} {rest : List Char} {o' : Nat} {v : Json}
    (h : word w j cs o = .ok rest o' v) : v = j := by
  unfold word at h
  split at h
  · injection h with _ _ h; exact h.symm
  · cases h

/-- A member inserted into a well-formed object of well-formed values leaves it so. -/
theorem ins_wf {m : Std.TreeMap.Raw String Json} (hm : m.WF) (h : ∀ k v, m.get? k = some v → Wf v)
    (key : String) {v : Json} (hv : Wf v) :
    (m.insert key v).WF ∧ ∀ k w, (m.insert key v).get? k = some w → Wf w := by
  refine ⟨hm.insert, fun k w hw => ?_⟩
  rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm] at hw
  split at hw
  · injection hw with hw; exact hw ▸ hv
  · exact h k w (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hw)

/-- At every fuel, a value read is well formed, and so is every element an array adds and every object built from a
well-formed one. -/
theorem wf_all : ∀ k,
    (∀ cs o rest o' v, value k cs o = .ok rest o' v → Wf v) ∧
    (∀ acc cs o rest o' a, array k acc cs o = .ok rest o' a → (∀ x ∈ acc, Wf x) → ∀ x ∈ a, Wf x) ∧
    (∀ m cs o rest o' m', object k m cs o = .ok rest o' m' → m.WF → (∀ k v, m.get? k = some v → Wf v) →
      m'.WF ∧ ∀ k v, m'.get? k = some v → Wf v)
  | 0 => by
    refine ⟨?_, ?_, ?_⟩
    · intro cs o rest o' v h; simp [value] at h
    · intro acc cs o rest o' a h; simp [array] at h
    · intro m cs o rest o' m' h; simp [object] at h
  | k + 1 => by
    obtain ⟨iv, ia, io⟩ := wf_all k
    refine ⟨?_, ?_, ?_⟩
    · intro cs o rest o' v h
      cases cs with
      | nil => simp [value] at h
      | cons c cs =>
        simp only [value] at h
        split at h
        · split at h
          · cases h
          · injection h with _ _ h; subst h; exact .arr (by simp)
          · split at h
            · cases h; exact .arr (ia _ _ _ _ _ _ (by assumption) (by simp))
            · cases h
        split at h
        · split at h
          · cases h
          · injection h with _ _ h; subst h
            exact .obj Std.TreeMap.Raw.WF.emptyc (by simp)
          · split at h
            · cases h
              obtain ⟨h1, h2⟩ := io _ _ _ _ _ _ (by assumption) Std.TreeMap.Raw.WF.emptyc (by simp)
              exact .obj h1 h2
            · cases h
        split at h
        · split at h
          · injection h with _ _ h; subst h; exact .str _
          · cases h
        split at h
        · rw [word_ok h]; exact .bool _
        split at h
        · rw [word_ok h]; exact .bool _
        split at h
        · rw [word_ok h]; exact .null
        split at h
        · split at h
          · injection h with _ _ h; subst h; exact .num _
          · cases h
        · cases h
    · intro acc cs o rest o' a h hacc
      simp only [array] at h
      split at h
      · cases h
      · cases h
      · rename_i hd hv
        have hw := iv _ _ _ _ _ hv
        have hacc' : ∀ x ∈ acc.push hd, Wf x := by
          intro x hx
          rcases Array.mem_push.1 hx with hx | hx
          · exact hacc x hx
          · exact hx ▸ hw
        split at h
        · injection h with _ _ h; subst h; exact hacc'
        split at h
        · exact ia _ _ _ _ _ _ h hacc'
        · cases h
    · intro m cs o rest o' m' h hm hmv
      unfold object at h
      split at h
      · cases h
      · cases h
      · injections; subst_vars
        split at h
        · cases h
        · split at h
          · cases h
          · split at h
            split at h
            · cases h
            · cases h
            · rename_i v hv
              have hw := ins_wf (key := ‹String›) hm hmv (iv _ _ _ _ _ hv)
              split at h
              · injection h with _ _ h; subst h; exact hw
              split at h
              · exact io _ _ _ _ _ _ h hw.1 hw.2
              · cases h
          · cases h
      · cases h

/-- **`Flock.JsonParse.parse` builds only well-formed trees**: every
object it returns is `∅` with its members inserted, and every value in it is well formed. -/
theorem parse_wf (s : String) (j : Json) (h : parse s = .ok j) : Wf j := by
  rw [parse_eq] at h
  unfold parseList at h
  split at h
  split at h
  · injection h with h; exact h ▸ (wf_all _).1 _ _ _ _ _ ‹_›
  · cases h
  · cases h

end FlockComplete.JsonRT
