module

public import Circuits.Program
public import Proofs.Flock.Claims.JsonEq
import Base.Sizes
import all Circuits.Program

/-!
# What the decoder reads from a Definition's entry, on equivalent entries

Two descriptors that give a key one closure digest have, for each Definition it reaches, equivalent canonical entries
(`entries_agree`). The decoder reads the raw entry, not the canonical one; this file holds the reads that give the same
answer on equivalent values. A type is the first: what `Ty.ofJson` accepts is the type's own text (`ofJson_ok`), which
holds no object, and a value with no object is equal to every value equivalent to it (`eq_of_jsonEquiv_objFree`), so
`Ty.ofJson` reads one type from equivalent values (`ofJson_jsonEquiv`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json JsonNumber)
open Flock Flock.Program
open FlockDeploy (JsonEquiv jsonEquiv_arr_iff)

theorem natJ_eq {x : Json} {w : Nat} (h : natJ? x = some w) : x = Lean.toJson w := by
  unfold natJ? at h
  split at h
  · rename_i n
    split at h
    · rename_i hc
      cases h
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
      obtain ⟨m, e⟩ := n
      simp only at hc
      obtain ⟨rfl, hm⟩ := hc
      show Json.num ⟨m, 0⟩ = Json.num ⟨((m.toNat : Nat) : Int), 0⟩
      rw [Int.toNat_of_nonneg hm]
    · cases h
  · cases h

theorem arr2 {a : Array Json} {x y : Json} (hs : a.size = 2) (h0 : a[0]? = some x) (h1 : a[1]? = some y) :
    a = #[x, y] := by
  rcases a with ⟨l⟩
  rcases l with _ | ⟨p, _ | ⟨q, _ | ⟨r, l⟩⟩⟩ <;> simp_all

theorem arr3 {a : Array Json} {x y z : Json} (hs : a.size = 3) (h0 : a[0]? = some x) (h1 : a[1]? = some y)
    (h2 : a[2]? = some z) : a = #[x, y, z] := by
  rcases a with ⟨l⟩
  rcases l with _ | ⟨p, _ | ⟨q, _ | ⟨r, _ | ⟨u, l⟩⟩⟩⟩ <;> simp_all

theorem list_mapM_ok {β γ δ ε : Type} (f : β → Except ε γ) (g : γ → δ) (p : β → δ) :
    ∀ (l : List β) (r : List γ), l.mapM f = .ok r → (∀ x ∈ l, ∀ t, f x = .ok t → g t = p x) → l.map p = r.map g
  | [], r, h, _ => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; rfl
  | x :: l, r, h, H => by
    rw [List.mapM_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i t ht
    split at h
    · cases h
    rename_i r' hr
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [List.map_cons, List.cons.injEq]
    exact ⟨(H x (by simp) t ht).symm, list_mapM_ok f g p l r' hr fun y hy => H y (by simp [hy])⟩

/-- `mapM` over `attach` returns, in order, a value for each element, which `g` maps back to it as `p` does. -/
theorem array_mapM_attach_ok {β γ δ ε : Type} {xs : Array β} (f : {x // x ∈ xs} → Except ε γ) (g : γ → δ)
    (p : β → δ) {r : Array γ} (h : xs.attach.mapM f = .ok r) (H : ∀ x : {x // x ∈ xs}, ∀ t, f x = .ok t → g t = p x.1) :
    xs.map p = r.map g := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, Except.map] at h
  split at h
  · cases h
  rename_i l hl
  simp only [Except.ok.injEq] at h
  subst h
  have := list_mapM_ok f g (fun x => p x.1) _ l hl fun x _ t ht => H x t ht
  apply Array.toList_inj.1
  simp only [Array.toList_map] at this ⊢
  rw [← this, Array.toList_attach, List.map_attachWith]
  simp

/-- What `Ty.ofJson` accepts is the text of the type it returns. -/
theorem ofJson_ok : ∀ (n : Nat) (j : Json) (t : Ty), sizeOf j < n → Ty.ofJson j = .ok t → j = t.toJson := by
  intro n
  induction n with
  | zero => intro j t hs; omega
  | succ n ih =>
    intro j t hs h
    cases j with
    | arr a =>
      unfold Ty.ofJson at h
      dsimp only at h
      split at h
      · rename_i h0
        split at h
        · rename_i w h1 hsz
          split at h
          · cases h
            obtain ⟨y, hy, hw⟩ := Option.bind_eq_some_iff.1 h1
            rw [arr2 hsz h0 hy, natJ_eq hw, Ty.toJson]
          · cases h
        · cases h
      · rename_i h0
        split at h
        · rename_i nn e h1 h2 hsz
          cases he : Ty.ofJson e with
          | error => rw [he] at h; cases h
          | ok te =>
            rw [he] at h
            cases h
            have := sizeOf_lt_of_getElem?_arr h2
            obtain ⟨y, hy, hw⟩ := Option.bind_eq_some_iff.1 h1
            rw [arr3 hsz h0 hy h2, natJ_eq hw, ih e te (by omega) he, Ty.toJson]
        · cases h
      · rename_i h0
        split at h
        · rename_i fs h1 hsz
          cases hm : fs.attach.mapM (fun f => Ty.ofJson f.1) with
          | error => simp only [bind, Except.bind, hm] at h; cases h
          | ok ts =>
            simp only [bind, Except.bind, hm, pure, Except.pure, Except.ok.injEq] at h
            subst h
            have hlt := sizeOf_lt_of_getElem?_arr h1
            have hfs := array_mapM_attach_ok _ Ty.toJson id hm fun x t ht =>
              (ih x.1 t (Nat.lt_of_lt_of_le (Nat.lt_trans (sizeOf_lt_of_mem_arr x.2) hlt) (by omega)) ht).symm
            rw [Array.map_id] at hfs
            rw [arr2 hsz h0 h1, hfs, Ty.toJson]
            done
        · cases h
      · rename_i h0
        split at h
        · rename_i fs h1 hsz
          simp only [bind, Except.bind] at h
          split at h
          · cases h
          rename_i rs hm
          split at h
          · cases h
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          have hlt := sizeOf_lt_of_getElem?_arr h1
          have hfs := array_mapM_attach_ok _ (fun f : String × Ty => Json.arr #[.str f.1, f.2.toJson]) id hm
            fun x t ht => by
              obtain ⟨y, hyx⟩ := x
              dsimp only at ht ⊢
              split at ht
              · rename_i _ ys
                split at ht
                · rename_i h2
                  split at ht
                  · rename_i k hk
                    split at ht
                    · cases ht
                    · rename_i v hv
                      simp only [pure, Except.pure, Except.ok.injEq] at ht
                      subst ht
                      have hl1 : sizeOf ys[1] < n := by
                        have := sizeOf_lt_of_mem_arr (a := ys) (Array.getElem_mem (i := 1) (by omega))
                        have := sizeOf_lt_of_mem_arr hyx
                        omega
                      show Json.arr #[.str k, v.toJson] = Json.arr ys
                      rw [← ih _ v hl1 hv]
                      exact congrArg Json.arr (arr2 h2 (by rw [Array.getElem?_eq_getElem (by omega), hk])
                        (Array.getElem?_eq_getElem (by omega))).symm
                  · cases ht
                · cases ht
              · cases ht
          rw [Array.map_id] at hfs
          rw [arr2 hsz h0 h1, hfs, Ty.toJson]
        · cases h
      · cases h
    | _ => unfold Ty.ofJson at h; cases h

/-- A value with no object anywhere in it. -/
inductive ObjFree : Json → Prop
  | null : ObjFree .null
  | bool (b : Bool) : ObjFree (.bool b)
  | num (n : JsonNumber) : ObjFree (.num n)
  | str (s : String) : ObjFree (.str s)
  | arr {a : Array Json} : (∀ x ∈ a, ObjFree x) → ObjFree (.arr a)

/-- A value with no object is the only value equivalent to it. -/
theorem eq_of_jsonEquiv_objFree {a : Json} (ha : ObjFree a) : ∀ {b : Json}, JsonEquiv a b → a = b := by
  induction ha with
  | null => intro b h; cases b <;> first | rfl | cases h
  | bool x => intro b h; cases b <;> first | cases h; rfl | cases h
  | num x => intro b h; cases b <;> first | cases h; rfl | cases h
  | str x => intro b h; cases b <;> first | cases h; rfl | cases h
  | @arr a _ ih =>
    intro b h
    have hb : ∃ b', b = .arr b' := by cases b <;> first | exact ⟨_, rfl⟩ | cases h
    obtain ⟨b', rfl⟩ := hb
    obtain ⟨hs, he⟩ := (jsonEquiv_arr_iff a b').1 h
    exact congrArg Json.arr (Array.ext hs fun i h1 h2 => ih _ (Array.getElem_mem h1) (he i h1 h2))

theorem objFree_toJson : ∀ (n : Nat) (t : Ty), sizeOf t < n → ObjFree t.toJson := by
  intro n
  induction n with
  | zero => intro t h; omega
  | succ n ih =>
    intro t ht
    cases t with
    | value w =>
      rw [Ty.toJson]
      refine .arr fun x hx => ?_
      simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · exact .str _
      · exact .num _
    | array m e =>
      rw [Ty.toJson]
      refine .arr fun x hx => ?_
      simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl | rfl
      · exact .str _
      · exact .num _
      · exact ih e (by simp only [Ty.array.sizeOf_spec] at ht; omega)
    | tuple fs =>
      rw [Ty.toJson]
      refine .arr fun x hx => ?_
      simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · exact .str _
      · refine .arr fun y hy => ?_
        obtain ⟨u, hu, rfl⟩ := Array.mem_map.1 hy
        have := Array.sizeOf_lt_of_mem hu
        simp only [Ty.tuple.sizeOf_spec] at ht
        exact ih u (by omega)
    | record fs =>
      rw [Ty.toJson]
      refine .arr fun x hx => ?_
      simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · exact .str _
      · refine .arr fun y hy => ?_
        obtain ⟨u, hu, rfl⟩ := Array.mem_map.1 hy
        have := Array.sizeOf_lt_of_mem hu
        obtain ⟨k, v⟩ := u
        simp only [Ty.record.sizeOf_spec, Prod.mk.sizeOf_spec] at ht this
        refine .arr fun z hz => ?_
        simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hz
        rcases hz with rfl | rfl
        · exact .str _
        · exact ih v (by omega)

/-- **A type reads the same from equivalent values**: what `Ty.ofJson` accepts is the type's own text, so any value
equivalent to it is that text. -/
theorem ofJson_jsonEquiv {j j' : Json} {t : Ty} (h : Ty.ofJson j = .ok t) (he : JsonEquiv j j') :
    Ty.ofJson j' = .ok t := by
  have hj := ofJson_ok _ j t (Nat.lt_succ_self _) h
  subst hj
  rw [← eq_of_jsonEquiv_objFree (objFree_toJson _ t (Nat.lt_succ_self _)) he, h]

end NciCheck.Properties.Record.SecurityProofs
