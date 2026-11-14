module

public import Circuits.Program
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Flock.Claims.Complete.Json.Value
import Base.Sizes
import Batteries.Tactic.OpenPrivate
import all Circuits.Program
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# What the decoder reads from a Definition's entry, on equivalent entries

Two descriptors that give a key one closure digest have, for each Definition it reaches, equivalent canonical entries
(`entries_agree`). The decoder reads the raw entry, not the canonical one; this file holds the reads that give the same
answer on equivalent values.

* A field of two equivalent well-formed values is absent from both or present in both, equivalent
  (`field_jsonEquiv`); so is a field read with a default (`fieldD_jsonEquiv`), and a string field reads alike
  (`fieldStr_eq`), as do the integer reads (`natJ_jsonEquiv`, `intJ_jsonEquiv`).
* What `Ty.ofJson` accepts is the type's own text (`ofJson_ok`), which holds no object, and a value with no object is
  the only value equivalent to it (`eq_of_jsonEquiv_objFree`), so `Ty.ofJson` reads one type from equivalent values
  (`ofJson_jsonEquiv`); likewise a parameter list (`paramsOf_jsonEquiv`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json JsonNumber)
open Flock Flock.Program
open FlockDeploy
open Claims.Json (Wf)
open private natJ? intJ? paramsOf from Circuits.Program

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

theorem array_mapM_mem' {α β ε : Type} (f : α → Except ε β) (xs : Array α) (r : Array β)
    (h : xs.mapM f = .ok r) : ∀ x ∈ xs, ∃ y ∈ r, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, Except.map] at h
  split at h
  · cases h
  rename_i l hl
  simp only [Except.ok.injEq] at h
  subst h
  intro x hx
  have key : ∀ (xs : List α) (l : List β), xs.mapM f = .ok l → ∀ x ∈ xs, ∃ y ∈ l, f x = .ok y := by
    intro xs
    induction xs with
    | nil => intro _ _ x hx; cases hx
    | cons a as ih =>
      intro l hl x hx
      rw [List.mapM_cons] at hl
      simp only [bind, Except.bind] at hl
      split at hl
      · cases hl
      rename_i b hb
      split at hl
      · cases hl
      rename_i bs hbs
      simp only [pure, Except.pure, Except.ok.injEq] at hl
      subst hl
      rcases List.mem_cons.1 hx with rfl | hx
      · exact ⟨b, List.mem_cons_self, hb⟩
      · obtain ⟨y, hy, hf⟩ := ih bs hbs x hx
        exact ⟨y, List.mem_cons_of_mem _ hy, hf⟩
  obtain ⟨y, hy, hf⟩ := key xs.toList l hl x (Array.mem_toList_iff.2 hx)
  exact ⟨y, by simpa using hy, hf⟩

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

theorem natJ_jsonEquiv {a b : Json} (h : JsonEquiv a b) : natJ? a = natJ? b := by
  cases a <;> cases b <;> simp_all [JsonEquiv, jsonForm, natJ?]

theorem intJ_jsonEquiv {a b : Json} (h : JsonEquiv a b) : intJ? a = intJ? b := by
  cases a <;> cases b <;> simp_all [JsonEquiv, jsonForm, intJ?]

/-- What `paramsOf` accepts holds no object. -/
theorem paramsOf_objFree {j : Json} {what : String} {ps : Array (String × Ty)} (h : paramsOf j what = .ok ps) :
    ObjFree j := by
  unfold paramsOf at h
  split at h
  · rename_i xs
    refine .arr fun x hx => ?_
    obtain ⟨q, -, hq⟩ := array_mapM_mem' _ xs ps h x hx
    rcases x with _ | _ | _ | _ | ⟨⟨_ | ⟨a, _ | ⟨t, _ | ⟨c, l⟩⟩⟩⟩⟩ | _
    all_goals first | cases hq | skip
    cases a with
    | str k =>
      have hq' : (do pure (k, ← Ty.ofJson t) : V (String × Ty)) = .ok q := hq
      simp only [bind, Except.bind] at hq'
      split at hq'
      · cases hq'
      rename_i u hu
      refine .arr fun y hy => ?_
      simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hy
      rcases hy with rfl | rfl
      · exact .str _
      · rw [ofJson_ok _ _ u (Nat.lt_succ_self _) hu]
        exact objFree_toJson _ u (Nat.lt_succ_self _)
    | _ => cases hq
  · cases h

theorem paramsOf_jsonEquiv {j j' : Json} {what : String} {ps : Array (String × Ty)} (h : paramsOf j what = .ok ps)
    (he : JsonEquiv j j') : paramsOf j' what = .ok ps := by
  rw [← eq_of_jsonEquiv_objFree (paramsOf_objFree h) he, h]

/-- Two optional values: both absent, or both present and equivalent. -/
def OptEquiv : Option Json → Option Json → Prop
  | some a, some b => JsonEquiv a b
  | none, none => True
  | _, _ => False

theorem obj_of_jsonEquiv_obj {m : Std.TreeMap.Raw String Json compare} {b : Json} (h : JsonEquiv (.obj m) b) :
    ∃ n, b = .obj n := by
  rw [JsonEquiv, jsonForm_obj] at h
  cases b with
  | obj n => exact ⟨n, rfl⟩
  | _ => simp [jsonForm] at h

theorem field_jsonEquiv {j j' : Json} (hj : Wf j) (hj' : Wf j') (h : JsonEquiv j j') (k : String) :
    OptEquiv (j.getObjVal? k).toOption (j'.getObjVal? k).toOption := by
  cases hj with
  | @obj m hm _ =>
    obtain ⟨n, rfl⟩ := obj_of_jsonEquiv_obj h
    cases hj' with
    | @obj _ hn _ =>
      obtain ⟨hk, hv⟩ := (jsonEquiv_obj_iff hm hn).1 h
      simp only [Json.getObjVal?]
      have := hk k
      cases ha : m.get? k <;> cases hb : n.get? k <;> rw [ha, hb] at this <;>
        simp only [Option.isSome_none, Option.isSome_some, Bool.false_eq_true, Bool.true_eq_false] at this <;>
        simp only [OptEquiv, Except.toOption, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
      exact hv k _ _ ha hb
  | _ =>
    cases j' with
    | obj n => exact absurd (obj_of_jsonEquiv_obj h.symm) (by simp)
    | _ => simp [Json.getObjVal?, OptEquiv, Except.toOption, throw, throwThe, MonadExceptOf.throw]

theorem getD_optEquiv {x y : Option Json} (h : OptEquiv x y) : JsonEquiv (x.getD .null) (y.getD .null) := by
  cases x <;> cases y <;> first | exact h | exact JsonEquiv.refl _ | exact h.elim

theorem fieldD_jsonEquiv {j j' : Json} (hj : Wf j) (hj' : Wf j') (h : JsonEquiv j j') (k : String) :
    JsonEquiv (j.getObjValD k) (j'.getObjValD k) :=
  getD_optEquiv (field_jsonEquiv hj hj' h k)

theorem str_of_jsonEquiv {s : String} {b : Json} (h : JsonEquiv (.str s) b) : b = .str s := by
  cases b <;> simp_all [JsonEquiv, jsonForm]

theorem getStr?_jsonEquiv {a b : Json} (h : JsonEquiv a b) : a.getStr?.toOption = b.getStr?.toOption := by
  cases a with
  | str s => rw [str_of_jsonEquiv h]
  | obj m =>
    obtain ⟨n, rfl⟩ := obj_of_jsonEquiv_obj h
    rfl
  | _ => cases b <;> first | rfl | simp_all [JsonEquiv, jsonForm]

theorem fieldStr_eq {j j' : Json} (hj : Wf j) (hj' : Wf j') (h : JsonEquiv j j') (k : String) :
    (j.getObjValAs? String k).toOption = (j'.getObjValAs? String k).toOption :=
  getStr?_jsonEquiv (fieldD_jsonEquiv hj hj' h k)

end NciCheck.Properties.Record.SecurityProofs
