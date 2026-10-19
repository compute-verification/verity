import Proofs.Flock.Claims.CanonEq
import Proofs.Flock.Claims.Complete.Json.Spec

/-!
# `JsonSpec`'s two fields on `canon`, proved

`Flock.canon` is total, so the two fields of `JsonSpec` about it are theorems: `canon` is the same
on alike values (`canon_same`: alike values are equivalent, `same_jsonEquiv`, and `canon` prints equivalent values
alike, `FlockDeploy.canon_jsonEquiv`), and returns on printable ones (`canon_ok`: every number an integer, so the one
refusal never fires). The fields left are the ones about `Json.parse` (the printer's and `canon`'s round trips, and
well-formed trees) and `Json`'s `==` (`beq_same`), both `partial` in Lean's core.
-/

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock

/-- **Alike values are equivalent** (`FlockDeploy.JsonEquiv`): one form, whatever their objects' key order. -/
theorem same_jsonEquiv : ∀ {a b : Json}, Same a b → FlockDeploy.JsonEquiv a b
  | _, _, .null => rfl
  | _, _, .bool _ => rfl
  | _, _, .num _ => rfl
  | _, _, .str _ => rfl
  | _, _, .arr hs he => (FlockDeploy.jsonEquiv_arr_iff _ _).2 ⟨hs, fun i h₁ h₂ => same_jsonEquiv (he i h₁ h₂)⟩
  | _, _, .obj hm hn hk hv =>
    (FlockDeploy.jsonEquiv_obj_iff hm hn).2 ⟨hk, fun k a b ha hb => same_jsonEquiv (hv k a b ha hb)⟩

/-- **`canon` is the same on alike values** (`JsonSpec.canon_same`). -/
theorem canon_same (a b : Json) (h : Same a b) : canon a = canon b :=
  FlockDeploy.canon_jsonEquiv (same_jsonEquiv h)

/-- **`canon` returns on printable values** (`JsonSpec.canon_ok`). -/
theorem canon_ok : ∀ j : Json, Good j → ∃ s, canon j = .ok s
  | _, .null => ⟨_, by rw [canon]; rfl⟩
  | _, .bool _ => ⟨_, by rw [canon]; rfl⟩
  | _, .num _ => ⟨_, by rw [canon]; rfl⟩
  | _, .str _ => ⟨_, by rw [canon]; rfl⟩
  | .arr a, .arr h => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok (f := fun x : {x // x ∈ a} => canon x.1) (a := a.attach)
      fun x _ => canon_ok x.1 (h x.1 x.2)
    exact ⟨_, by rw [canon, hr]; rfl⟩
  | .obj m, .obj hm h => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok
      (f := fun p : {p // p ∈ m.toArray} => do pure (p.1.1, ← canon p.1.2)) (a := m.toArray.attach)
      fun p _ => by
        have hp : (p.1.1, p.1.2) ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 p.2
        have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
        obtain ⟨s, hs⟩ := canon_ok p.1.2 (h _ _ (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hg))
        exact ⟨_, by rw [hs]; rfl⟩
    exact ⟨_, by rw [canon, hr]; rfl⟩

end FlockComplete.JsonRT
