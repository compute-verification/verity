import Flock.Canon
import Proofs.Flock.Claims.JsonEq

/-!
# `canon` prints equivalent values alike

`Flock.canon` is total (well-founded on `sizeOf`), so its equations unfold. It prints an array element by element, and
an object's values in the order of its tree before it sorts the pairs by key; `JsonEquiv` fixes exactly those: the
elements' forms in order, and the tree's keys and values' forms in order (`jsonForm_obj`). So equivalent values print
alike (`canon_jsonEquiv`), and two draws `drawEq` passes on print alike (`canon_drawEq`).
-/

namespace FlockDeploy

open Lean (Json JsonNumber)
open Flock

/-- `mapM` over two lists of one length, with equal results position by position. -/
theorem mapM_congr_getElem {m : Type → Type} [Monad m] [LawfulMonad m] {α β γ : Type} {f : α → m γ}
    {g : β → m γ} : ∀ {l₁ : List α} {l₂ : List β}, l₁.length = l₂.length →
      (∀ i (h₁ : i < l₁.length) (h₂ : i < l₂.length), f l₁[i] = g l₂[i]) → l₁.mapM f = l₂.mapM g
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: as, b :: bs, h, he => by
    have h0 : f a = g b := he 0 (by simp) (by simp)
    simp only [List.mapM_cons]
    rw [h0, mapM_congr_getElem (by simpa using h) fun i h₁ h₂ => he (i + 1) (by simpa using h₁) (by simpa using h₂)]

/-- `mapM` in `Except` returns where `f` returns on every element. -/
theorem mapM_ok {α γ ε : Type} {f : α → Except ε γ} :
    ∀ {l : List α}, (∀ x ∈ l, ∃ s, f x = .ok s) → ∃ r, l.mapM f = .ok r
  | [], _ => ⟨[], rfl⟩
  | a :: as, h => by
    obtain ⟨s, hs⟩ := h a (by simp)
    obtain ⟨r, hr⟩ := mapM_ok (l := as) fun x hx => h x (List.mem_cons_of_mem _ hx)
    exact ⟨s :: r, by rw [List.mapM_cons, hs, hr]; rfl⟩

/-- `mapM` over an array in `Except` returns where `f` returns on every element. -/
theorem array_mapM_ok {α γ ε : Type} {f : α → Except ε γ} {a : Array α} (h : ∀ x ∈ a, ∃ s, f x = .ok s) :
    ∃ r, a.mapM f = .ok r := by
  obtain ⟨r, hr⟩ := mapM_ok (l := a.toList) fun x hx => h x (Array.mem_toList_iff.1 hx)
  exact ⟨r.toArray, by rw [Array.mapM_eq_mapM_toList, hr]; rfl⟩

theorem jsonEquiv_null {b : Json} (h : JsonEquiv .null b) : b = .null := by
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> first | rfl | simp [JsonEquiv, jsonForm] at h

theorem jsonEquiv_bool {x : Bool} {b : Json} (h : JsonEquiv (.bool x) b) : b = .bool x := by
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> simp_all [JsonEquiv, jsonForm]

theorem jsonEquiv_num {n : JsonNumber} {b : Json} (h : JsonEquiv (.num n) b) : b = .num n := by
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> simp_all [JsonEquiv, jsonForm]

theorem jsonEquiv_str {s : String} {b : Json} (h : JsonEquiv (.str s) b) : b = .str s := by
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> simp_all [JsonEquiv, jsonForm]

theorem jsonEquiv_arr {a : Array Json} {b : Json} (h : JsonEquiv (.arr a) b) : ∃ b', b = .arr b' := by
  obtain ⟨a⟩ := a
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> first | exact ⟨_, rfl⟩ | simp [JsonEquiv, jsonForm] at h

theorem jsonEquiv_obj {m : Std.TreeMap.Raw String Json} {b : Json} (h : JsonEquiv (.obj m) b) :
    ∃ n, b = .obj n := by
  obtain ⟨⟨t⟩⟩ := m
  rcases b with _ | _ | _ | _ | ⟨b⟩ | ⟨⟨b⟩⟩ <;> first | exact ⟨_, rfl⟩ | simp [JsonEquiv, jsonForm] at h

/-- **Equivalent objects' fields**: one length, and at each position one key and equivalent values. -/
theorem jsonEquiv_obj_toList {m n : Std.TreeMap.Raw String Json} (h : JsonEquiv (.obj m) (.obj n)) :
    m.toList.length = n.toList.length ∧ ∀ i (h₁ : i < m.toList.length) (h₂ : i < n.toList.length),
      m.toList[i].1 = n.toList[i].1 ∧ JsonEquiv m.toList[i].2 n.toList[i].2 := by
  rw [JsonEquiv, jsonForm_obj, jsonForm_obj, JsonForm.obj.injEq, FieldForms.ofList_inj] at h
  have hl : m.toList.length = n.toList.length := by simpa using congrArg List.length h
  refine ⟨hl, fun i h₁ h₂ => ?_⟩
  have := congrArg (·[i]?) h
  simp only [List.getElem?_map, List.getElem?_eq_getElem h₁, List.getElem?_eq_getElem h₂, Option.map_some,
    Option.some.injEq, Prod.mk.injEq] at this
  exact this

theorem canon_jsonEquiv_lt : ∀ (k : Nat) (a b : Json), sizeOf a < k → JsonEquiv a b → canon a = canon b
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, a, b, hk, h => by
    match a, h with
    | .null, h => rw [jsonEquiv_null h]
    | .bool x, h => rw [jsonEquiv_bool h]
    | .num n, h => rw [jsonEquiv_num h]
    | .str s, h => rw [jsonEquiv_str h]
    | .arr a, h =>
      obtain ⟨b, rfl⟩ := jsonEquiv_arr h
      obtain ⟨hs, he⟩ := (jsonEquiv_arr_iff a b).1 h
      rw [canon, canon]
      congr 1
      rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
      congr 1
      refine mapM_congr_getElem (by simpa using hs) fun i h₁ h₂ => ?_
      simp only [Array.toList_attach, List.getElem_attachWith]
      have hi₁ : i < a.size := by simpa using h₁
      have hi₂ : i < b.size := by simpa using h₂
      have hlt := Array.sizeOf_lt_of_mem (Array.getElem_mem hi₁)
      simp only [Json.arr.sizeOf_spec] at hk
      exact canon_jsonEquiv_lt k a[i] b[i] (by omega) (he i hi₁ hi₂)
    | .obj m, h =>
      obtain ⟨n, rfl⟩ := jsonEquiv_obj h
      obtain ⟨hl, he⟩ := jsonEquiv_obj_toList h
      rw [canon, canon]
      congr 1
      rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
      congr 1
      refine mapM_congr_getElem (by simpa [Std.TreeMap.Raw.toList_toArray] using hl) fun i h₁ h₂ => ?_
      simp only [Array.toList_attach, List.getElem_attachWith]
      have hi₁ : i < m.toList.length := by simpa [Std.TreeMap.Raw.toList_toArray] using h₁
      have hi₂ : i < n.toList.length := by simpa [Std.TreeMap.Raw.toList_toArray] using h₂
      have hi₀ : i < m.toArray.toList.length := by simpa using h₁
      have hmem : m.toArray.toList[i] ∈ m.toArray := Array.mem_toList_iff.1 (List.getElem_mem hi₀)
      have hlt := sizeOf_lt_of_mem_toArray hmem
      obtain ⟨hk', hv⟩ := he i hi₁ hi₂
      simp only [Std.TreeMap.Raw.toList_toArray] at hlt ⊢
      rw [hk', canon_jsonEquiv_lt k _ _ (by omega) hv]

/-- **`canon` prints equivalent values alike.** -/
theorem canon_jsonEquiv {a b : Json} (h : JsonEquiv a b) : canon a = canon b :=
  canon_jsonEquiv_lt (sizeOf a + 1) a b (Nat.lt_succ_self _) h

/-- **Two draws `drawEq` passes on print alike.** -/
theorem canon_drawEq {a b : Option Json} (h : drawEq a b = true) : a.map canon = b.map canon := by
  cases (drawEq_iff a b).1 h with
  | none => rfl
  | some he => exact congrArg some (canon_jsonEquiv he)

end FlockDeploy
