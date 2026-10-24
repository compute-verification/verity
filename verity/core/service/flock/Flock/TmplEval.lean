/-
A templated unit's rows evaluate as the composition of its Definitions (PRF-133): any assignment of the unrolled rows,
each column the AND of its A and B rows' XORs, satisfies at every placed copy its Definition's own rows on that copy's
columns and bound inputs, and its calls' copies recursively (`rows_sat`, `SatAt`). So a fact about a Definition's rows on
any copy (a piece computing its specification) is a fact about the unit's columns at every copy of it, and a session's
circuit is the composition of its pieces as the template places them. No verifier path reads it.

The proof is an induction over the copies `place` writes: each copy's rows are written once (`write` refuses a written
column) and so are the unit's at the end (`Keeps`), and `canon` keeps a row's XOR (`xorL_canon`).
-/
import Flock.TmplOrder

namespace Flock.TmplEval

open Flock.Tmpl Flock.TmplOrder

/-! ## A row's value -/

/-- The XOR of an assignment `v` over a row's columns. -/
def xorList (v : Nat → Bool) : List Nat → Bool
  | [] => false
  | x :: xs => xor (v x) (xorList v xs)

/-- The XOR of an assignment `v` over a row's columns. -/
def xorL (v : Nat → Bool) (xs : Array Nat) : Bool := xorList v xs.toList

theorem xorList_dropPairs (v : Nat → Bool) : ∀ l : List Nat, xorList v (dropPairs l) = xorList v l
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
    simp only [dropPairs]
    split
    · next h =>
      rw [beq_iff_eq] at h
      subst h
      rw [xorList_dropPairs v rest]
      simp only [xorList]
      cases v a <;> simp
    · simp only [xorList]
      rw [xorList_dropPairs v (b :: rest)]
      rfl

theorem xorList_perm (v : Nat → Bool) {l l' : List Nat} (h : l.Perm l') : xorList v l = xorList v l' := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [xorList, ih]
  | swap x y l => simp only [xorList]; cases v x <;> cases v y <;> simp
  | trans _ _ ih1 ih2 => exact ih1.trans ih2

theorem xorList_map (v : Nat → Bool) (g : Nat → Nat) : ∀ l : List Nat, xorList v (l.map g) = xorList (fun x => v (g x)) l
  | [] => rfl
  | x :: xs => by simp only [List.map_cons, xorList, xorList_map v g xs]

/-- **`canon` keeps a row's XOR**: a row's canonical form, its columns through `g`, XORs as the row does through `g`. -/
theorem xorL_canon (v : Nat → Bool) (g : Nat → Nat) (xs : Array Nat) :
    xorL v (Flock.Tmpl.canon (xs.map g)) = xorL (fun x => v (g x)) xs := by
  unfold xorL Flock.Tmpl.canon
  rw [List.toList_toArray, xorList_dropPairs, xorList_perm v (List.mergeSort_perm _ _), Array.toList_map, xorList_map]

/-! ## Rows written once -/

/-- Column `x` is written, with rows `a` and `b`. -/
def Has (x : Nat) (a b : Array Nat) (acc : Acc) : Prop :=
  acc.set.getD x false = true ∧ acc.ra.getD x #[] = a ∧ acc.rb.getD x #[] = b

/-- Every column written in `acc` is written in `acc'`, with the same rows. -/
def Keeps (acc acc' : Acc) : Prop :=
  ∀ x, acc.set.getD x false = true → Has x (acc.ra.getD x #[]) (acc.rb.getD x #[]) acc'

theorem keeps_refl (acc : Acc) : Keeps acc acc := fun _ h => ⟨h, rfl, rfl⟩

theorem keeps_trans {a b c : Acc} (h1 : Keeps a b) (h2 : Keeps b c) : Keeps a c := fun x hx => by
  obtain ⟨hb, ea, eb⟩ := h1 x hx
  obtain ⟨hc, ea', eb'⟩ := h2 x hb
  exact ⟨hc, ea'.trans ea, eb'.trans eb⟩

theorem has_keeps {x : Nat} {a b : Array Nat} {acc acc' : Acc} (h : Has x a b acc) (hk : Keeps acc acc') :
    Has x a b acc' := by
  obtain ⟨hs, ea, eb⟩ := h
  obtain ⟨hs', ea', eb'⟩ := hk x hs
  exact ⟨hs', ea'.trans ea, eb'.trans eb⟩

theorem getD_lt {xs : Array Bool} {x : Nat} (h : xs.getD x true = false) : x < xs.size := by
  rcases Nat.lt_or_ge x xs.size with hx | hx
  · exact hx
  · simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hx] at h

theorem getD_set!_ne {α : Type} (xs : Array α) {c x : Nat} (v d : α) (h : x ≠ c) : (xs.set! c v).getD x d = xs.getD x d := by
  simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds, Ne.symm h,
    ite_false]

theorem getD_set!_eq {α : Type} (xs : Array α) {c : Nat} (v d : α) (h : c < xs.size) : (xs.set! c v).getD c d = v := by
  simp [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, h]

/-- The rows' arrays as long as the written flags, as `Acc.empty` makes them and `write` keeps them. -/
def Sized (acc : Acc) : Prop := acc.ra.size = acc.set.size ∧ acc.rb.size = acc.set.size

/-- **`write` keeps every written column, and writes its own.** -/
theorem write_keeps {acc acc' : Acc} {c : Nat} {a b : Array Nat} (hz : Sized acc) (h : write acc c a b = .ok acc') :
    Keeps acc acc' ∧ Sized acc' ∧ Has c a b acc' := by
  unfold write at h
  split at h
  · cases h
  · next hs =>
    cases h
    have hs' : acc.set.getD c true = false := by simpa using hs
    have hc := getD_lt hs'
    refine ⟨fun x hx => ?_, ?_, ?_⟩
    · have hxc : x ≠ c := by
        rintro rfl
        have : acc.set.getD x true = acc.set.getD x false := by
          simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hc]
        rw [this, hx] at hs'
        cases hs'
      exact ⟨by rw [getD_set!_ne _ _ _ hxc]; exact hx, getD_set!_ne _ _ _ hxc, getD_set!_ne _ _ _ hxc⟩
    · unfold Sized at hz ⊢
      simp only [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]
      exact hz
    · exact ⟨getD_set!_eq _ _ _ hc, getD_set!_eq _ _ _ (hz.1 ▸ hc), getD_set!_eq _ _ _ (hz.2 ▸ hc)⟩

/-- A fact each step of a fold of writes establishes about its element, kept by later writes, holds at the end. -/
theorem foldlM_each {α : Type} {f : Acc → α → Except String Acc} (Q : α → Acc → Prop)
    (hf : ∀ a u u', Sized u → f u a = .ok u' → Keeps u u' ∧ Sized u' ∧ Q a u')
    (hq : ∀ a u w, Q a u → Keeps u w → Q a w) :
    ∀ {l : List α} {s s' : Acc}, Sized s → l.foldlM f s = .ok s' → Keeps s s' ∧ Sized s' ∧ ∀ a ∈ l, Q a s'
  | [], _, _, hs, h => by cases h; exact ⟨keeps_refl _, hs, by simp⟩
  | a :: l, s, _, hs, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s1, h1, h2⟩ := bind_ok h
    obtain ⟨k1, z1, q1⟩ := hf a s s1 hs h1
    obtain ⟨k2, z2, q2⟩ := foldlM_each Q hf hq z1 h2
    refine ⟨keeps_trans k1 k2, z2, fun b hb => ?_⟩
    rcases List.mem_cons.1 hb with rfl | hb
    · exact hq _ _ _ q1 k2
    · exact q2 b hb

/-! ## Copies -/

/-- Every written row of `acc` holds under `v`: its column's value is the AND of its A and B rows' XORs. -/
def RowsHold (v : Nat → Bool) (acc : Acc) : Prop :=
  ∀ x, acc.set.getD x false = true → v x = (xorL v (acc.ra.getD x #[]) && xorL v (acc.rb.getD x #[]))

/-- **Definition `d`'s copy at offset `o`, its inputs bound to `inp`, holds under `v`**: each of its own rows, at its own
column `o + col`, is the AND of its A and B rows' XORs over its references (`Def.ref`: own columns `o + x`, inputs `inp`,
the constant `k`), and each of its calls' copies holds, at its offset and its bound inputs (`Call.bound`). `fuel` bounds
the call depth as `place`'s does. -/
def SatAt (defs : Array Def) (k : Nat) (v : Nat → Bool) : Nat → Nat → Nat → List Nat → Prop
  | 0, _, _, _ => False
  | fuel + 1, d, o, inp => ∃ df, defs[d]? = some df ∧
      (∀ r ∈ df.rows.toList, v (o + r.1) =
        (xorL (fun x => v (df.ref o inp k x)) r.2.1 && xorL (fun x => v (df.ref o inp k x)) r.2.2)) ∧
      ∀ c ∈ df.calls.toList, ∀ j < c.count,
        SatAt defs k v fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j)

/-- **A copy's rows hold wherever the unit's do.** `place` keeps every column already written and writes its copy's rows
once; under any assignment that satisfies the rows of an accumulator keeping them, the copy holds (`SatAt`). -/
theorem place_sat {defs : Array Def} {k : Nat} (v : Nat → Bool) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Sized acc →
      Keeps acc acc' ∧ Sized acc' ∧ ∀ fin, Keeps acc' fin → RowsHold v fin → SatAt defs k v fuel d o inp := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hz
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      obtain ⟨k1, z1, q1⟩ := foldlM_each
        (fun (r : Nat × Array Nat × Array Nat) w => Has (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k)))
          (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))) w)
        (fun _ _ _ hu hw => by obtain ⟨a, b, c⟩ := write_keeps hu hw; exact ⟨a, b, c⟩)
        (fun _ _ _ hq hk => has_keeps hq hk) hz h1
      obtain ⟨k2, z2, q2⟩ := foldlM_each
        (fun (c : Call) w => ∀ j < c.count, ∀ fin, Keeps w fin → RowsHold v fin →
          SatAt defs k v fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j))
        (fun c _ _ hu hw => by
          obtain ⟨kk, zz, qq⟩ := foldlM_each
            (fun (j : Nat) w => ∀ fin, Keeps w fin → RowsHold v fin →
              SatAt defs k v fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j))
            (fun _ _ _ hu' hp => ih hp hu') (fun _ _ _ hq hk fin hk' hr => hq fin (keeps_trans hk hk') hr) hu hw
          exact ⟨kk, zz, fun j hj => qq j (List.mem_range.2 hj)⟩)
        (fun _ _ _ hq hk j hj fin hk' hr => hq j hj fin (keeps_trans hk hk') hr) z1 h2
      refine ⟨keeps_trans k1 k2, z2, fun fin hk hr => ⟨df, hdf, fun r hrow => ?_, fun c hc j hj => q2 c hc j hj fin hk hr⟩⟩
      obtain ⟨hs, ea, eb⟩ := has_keeps (has_keeps (q1 r hrow) k2) hk
      rw [hr (o + r.1) hs, ea, eb, xorL_canon, xorL_canon]

/-! ## The unit -/

/-- **A templated unit's rows evaluate as its Definitions' composition**: under any assignment that satisfies the rows
`Tmpl.rows` unrolls to, the root's copy holds (`SatAt`), so every placed copy of every Definition holds on its columns and
bound inputs. -/
theorem rows_sat {t : Tmpl} {acc : Acc} (v : Nat → Bool) (h : t.rows = .ok acc) (hv : RowsHold v acc) :
    SatAt t.defs t.constPos v (t.defs.size + 1) t.root 0 [] := by
  unfold Tmpl.rows at h
  obtain ⟨acc1, h1, h2⟩ := bind_ok h
  have hz : Sized (Acc.empty (t.defs.getD t.root default).cols) := by simp [Sized, Acc.empty]
  obtain ⟨-, z1, s1⟩ := place_sat v _ h1 hz
  exact s1 acc (write_keeps z1 h2).1 hv

end Flock.TmplEval
