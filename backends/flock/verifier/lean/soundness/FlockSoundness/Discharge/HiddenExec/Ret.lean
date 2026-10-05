import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Discharge.Layout.OutShape
import FlockSoundness.Protocol.Discharge.HiddenExec.Ret

set_option Elab.async false

/-!
# Where the circuit computes each output word

`HmOut.delta` copies an output row's message bits from `HmOut.outCol c g o`, the first column of output word `o` at VU
`g`. In an untyped circuit whose unit returns the outputs, that is the unit copy that returns `o`, at the column of its
returned-output port: `outCol` reads which copy and which port from `ret`, the last write of each output word over the
copies' leaf maps. That write is the only one, because `HmRow.check` throws on an output word some copy returns twice:

* `check_returned`: `HmRow.check`'s `seen` loop, so the copies' leaf maps, in order, are distinct (`returned`);
* `ret_leaf`: `ret` at copy `u'`'s output `j` is `(u', j)`;
* `outCol_leaf`: `HmOut.outCol` there is the unit slot `g · upv + u'`, column `16 j` of the returned-output group;
* `outPortOf_spec`: output word `o` is word `o - outAt jo` of output port `jo`;
* `checkRows_words`: each output row's words hold its port's (`HmOut.checkRows`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 FlockSoundness.Discharge.Layout

/-! ## Loop invariants -/

/-! ## `HmRow.check`'s `seen` loop -/

/-! ## `ret` -/

theorem retStep_size (u : ℕ) : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), (L.foldl (retStep u) (r, q)).1.size = r.size
  | [], _, _ => rfl
  | o :: L, r, q => by
    rw [List.foldl_cons]
    exact (retStep_size u L _ _).trans (by simp)

theorem retStep_not_mem (u : ℕ) {x : ℕ} : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), x ∉ L →
    (L.foldl (retStep u) (r, q)).1[x]? = r[x]?
  | [], _, _, _ => rfl
  | o :: L, r, q, hx => by
    rw [List.foldl_cons]
    rw [List.mem_cons, not_or] at hx
    exact (retStep_not_mem u L _ _ hx.2).trans (by simp [Array.getElem?_setIfInBounds_ne (Ne.symm hx.1)])

theorem retStep_mem (u : ℕ) : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), L.Nodup → ∀ i (hi : i < L.length),
    L[i] < r.size → (L.foldl (retStep u) (r, q)).1[L[i]]? = some (u, q + i)
  | [], _, _, _, _, hi, _ => absurd hi (by simp)
  | o :: L, r, q, hnd, i, hi, hlt => by
    rw [List.foldl_cons]
    rw [List.nodup_cons] at hnd
    rcases i with _ | i
    · simp only [List.getElem_cons_zero] at hlt ⊢
      rw [retStep_not_mem u L _ _ hnd.1]
      simp [retStep, hlt]
    · simp only [List.getElem_cons_succ] at hlt ⊢
      have hi' : i < L.length := by simpa using hi
      show (L.foldl (retStep u) (r.set! o (u, q), q + 1)).1[L[i]'hi']? = some (u, q + (i + 1))
      rw [retStep_mem u L _ _ hnd.2 i hi' (by simpa using hlt)]
      simp only [Option.some.injEq, Prod.mk.injEq, true_and]
      omega

theorem retOf_size (c : Circuit) : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), (retOf c us r).size = r.size
  | [], _ => rfl
  | u :: us, r => by
    unfold retOf
    rw [List.foldl_cons]
    exact (retOf_size c us _).trans (by rw [← Array.foldl_toList, retStep_size])

theorem retOf_not_mem (c : Circuit) {x : ℕ} : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), x ∉ returned c us →
    (retOf c us r)[x]? = r[x]?
  | [], _, _ => rfl
  | u :: us, r, hx => by
    unfold retOf
    rw [List.foldl_cons]
    simp only [returned, List.flatMap_cons, List.mem_append, not_or] at hx
    exact (retOf_not_mem c us _ hx.2).trans (by rw [← Array.foldl_toList, retStep_not_mem u _ _ _ hx.1])

/-! ## `HmOut.outCol` at a returned output -/

/-! ## The output port of an output word -/

/-! ## An output row's words -/

end FlockSoundness.Discharge.HiddenExec
