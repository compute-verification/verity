import Batteries.Tactic.OpenPrivate
import Proofs.Flock.Claims.CanonEq
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Claims.Complete.Json.Spec

/-!
# `JsonSpec`'s two fields on `canon`, proved

`Flock.canon` is total, so the two fields of `JsonSpec` about it are theorems: `canon` is the same
on alike values (`canon_same`: alike values are equivalent, `same_jsonEquiv`, and `canon` prints equivalent values
alike, `FlockDeploy.canon_jsonEquiv`), and returns on printable ones a text of ASCII without DEL (`canon_ok`: every
number an integer, so the one refusal never fires, and `canon_ascii`: `canonStr` writes every character outside printable
ASCII as a `\u` escape, `escU`). The fields left are the ones about `Json.parse` (the printer's and `canon`'s round trips, and
well-formed trees) and `Json`'s `==` (`beq_same`), both `partial` in Lean's core.
-/

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock
open private hex4 from Flock.Canon

namespace Ascii

theorem ascii_append {s t : String} (hs : Claims.Json.Ascii s) (ht : Claims.Json.Ascii t) : Claims.Json.Ascii (s ++ t) := by
  intro c hc
  rw [String.toList_append, List.mem_append] at hc
  rcases hc with h | h
  · exact hs c h
  · exact ht c h

theorem ascii_lit {s : String} (h : s.toList.all (·.toNat < 0x7f) = true) : Claims.Json.Ascii s := by
  intro c hc
  simpa using List.all_eq_true.1 h c hc

theorem ofNat_lt {n : Nat} (h : n < 0x7f) : (Char.ofNat n).toNat < 0x7f := by
  have hv : n.isValidChar := Or.inl (by omega)
  rw [Char.ofNat, dite_eq_left_of_eq_true (eq_true hv)]
  simp [Char.ofNatAux, Char.toNat]
  omega

theorem hex4_ascii (n : Nat) : Claims.Json.Ascii (hex4 n) := by
  unfold hex4
  intro c hc
  simp only [String.toList_ofList, List.mem_cons, List.not_mem_nil, or_false] at hc
  have hx : ∀ k, n >>> (4 * k) % 16 < 16 := fun _ => Nat.mod_lt _ (by decide)
  rcases hc with rfl | rfl | rfl | rfl
  all_goals (split <;> apply ofNat_lt <;> have := hx 0 <;> have := hx 1 <;> have := hx 2 <;> have := hx 3 <;> omega)

theorem escU_ascii (n : Nat) : Claims.Json.Ascii (Flock.escU n) := by
  unfold Flock.escU
  split
  · exact ascii_append (ascii_lit (by decide)) (hex4_ascii _)
  · exact ascii_append (ascii_append (ascii_append (ascii_lit (by decide)) (hex4_ascii _)) (ascii_lit (by decide)))
      (hex4_ascii _)

theorem foldl_ascii (f : String → Char → String) (hf : ∀ b a, Claims.Json.Ascii b → Claims.Json.Ascii (f b a)) :
    ∀ (cs : List Char) (out : String), Claims.Json.Ascii out → Claims.Json.Ascii (cs.foldl f out) := by
  intro cs
  induction cs with
  | nil => intro out h; simpa using h
  | cons c cs ih => intro out h; rw [List.foldl_cons]; exact ih _ (hf _ _ h)

/-- What `canonStr` writes for one character. -/
def esc (c : Char) : List Char :=
  match c with
  | '"' => ['\\', '"']
  | '\\' => ['\\', '\\']
  | '\n' => ['\\', 'n']
  | '\r' => ['\\', 'r']
  | '\t' => ['\\', 't']
  | c => if c.toNat == 8 then ['\\', 'b'] else if c.toNat == 12 then ['\\', 'f']
    else if c.toNat < 0x20 || 0x7f ≤ c.toNat then (Flock.escU c.toNat).toList else [c]

theorem foldl_esc (f : String → Char → String) (hf : ∀ b a, (f b a).toList = b.toList ++ esc a) :
    ∀ (cs : List Char) (out : String), (cs.foldl f out).toList = out.toList ++ cs.flatMap esc := by
  intro cs
  induction cs with
  | nil => intro out; simp
  | cons c cs ih => intro out; rw [List.foldl_cons, ih, hf, List.flatMap_cons, List.append_assoc]

theorem canonStr_toList (s : String) : (Flock.canonStr s).toList = '"' :: s.toList.flatMap esc ++ ['"'] := by
  unfold Flock.canonStr
  simp only [Id.run, List.forIn_pure_yield_eq_foldl]
  show (List.foldl _ "\"" s.toList ++ "\"").toList = _
  rw [String.toList_append, foldl_esc]
  · simp
  · intro b a
    unfold esc
    split
    all_goals first
      | (simp; done)
      | ((repeat' split) <;> simp_all)

theorem list_ascii {l : List Char} (h : l.all (·.toNat < 0x7f) = true) : ∀ x ∈ l, x.toNat < 0x7f := by
  intro x hx
  simpa using List.all_eq_true.1 h x hx

theorem esc_ascii (c : Char) : ∀ x ∈ esc c, x.toNat < 0x7f := by
  unfold esc
  split
  any_goals exact list_ascii rfl
  split
  · exact list_ascii rfl
  split
  · exact list_ascii rfl
  split
  · exact escU_ascii _
  · rename_i h
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_le] at h
    simp only [List.mem_singleton]
    rintro x rfl
    exact h.2

/-- `canonStr` writes ASCII without DEL. -/
theorem canonStr_ascii (s : String) : Claims.Json.Ascii (Flock.canonStr s) := by
  intro x hx
  rw [canonStr_toList] at hx
  simp only [List.mem_cons, List.mem_append, List.mem_flatMap] at hx
  rcases hx with (rfl | ⟨c, -, hc⟩) | (rfl | hx)
  · decide
  · exact esc_ascii c x hc
  · decide
  · simp at hx

theorem mem_intercalate {α} (sep : List α) (c : α) : ∀ (l : List (List α)), c ∈ sep.intercalate l → c ∈ sep ∨ ∃ x ∈ l, c ∈ x
  | [], h => by simp at h
  | [a], h => Or.inr ⟨a, by simp, by simpa using h⟩
  | a :: b :: l, h => by
    rw [List.intercalate_cons_cons] at h
    simp only [List.mem_append] at h
    rcases h with (h | h) | h
    · exact Or.inr ⟨a, by simp, h⟩
    · exact Or.inl h
    · rcases mem_intercalate sep c (b :: l) h with h | ⟨x, hx, hc⟩
      · exact Or.inl h
      · exact Or.inr ⟨x, List.mem_cons_of_mem _ hx, hc⟩

theorem ascii_intercalate {sep : String} (hs : Claims.Json.Ascii sep) {l : List String} (h : ∀ x ∈ l, Claims.Json.Ascii x) :
    Claims.Json.Ascii (sep.intercalate l) := by
  intro c hc
  rw [String.toList_intercalate] at hc
  rcases mem_intercalate _ _ _ hc with hc | ⟨x, hx, hc⟩
  · exact hs c hc
  · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    exact h y hy c hc

theorem ascii_int (m : Int) : Claims.Json.Ascii (toString m) := by
  have hn : ∀ n : Nat, Claims.Json.Ascii (toString n) := by
    intro n c hc
    have hd := Nat.isDigit_of_mem_toDigits (b := 10) (n := n) (by decide) (by decide)
      (by simpa [toString, Nat.repr] using hc)
    simp only [Char.isDigit, Bool.and_eq_true, decide_eq_true_eq] at hd
    have : c.val.toNat ≤ 57 := by
      have := hd.2
      simpa [UInt32.le_iff_toNat_le] using this
    simp only [Char.toNat]
    omega
  cases m with
  | ofNat n => simpa [toString, Int.repr] using hn n
  | negSucc n =>
    show Claims.Json.Ascii ("-" ++ toString (n + 1))
    exact ascii_append (ascii_lit (by decide)) (hn _)

theorem mem_of_mapM_ok {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → ∀ y ∈ r, ∃ x ∈ l, f x = .ok y
  | [], r, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; intro y hy; simp at hy
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | error e => rw [hb] at h; cases h
    | ok b =>
      cases hl : l.mapM f with
      | error e => rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h; cases h
      | ok bs =>
        rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        intro y hy
        rcases List.mem_cons.1 hy with rfl | hy
        · exact ⟨a, by simp, hb⟩
        · obtain ⟨x, hx, hfx⟩ := mem_of_mapM_ok hl y hy
          exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩

theorem array_mem_of_mapM_ok {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β}
    (h : a.mapM f = .ok r) : ∀ y ∈ r, ∃ x ∈ a, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    intro y hy
    obtain ⟨x, hx, hfx⟩ := mem_of_mapM_ok hl y (by simpa using hy)
    exact ⟨x, Array.mem_toList_iff.1 hx, hfx⟩

theorem canon_ascii_lt
    (hq : ∀ (a : Array (String × String)) x, x ∈ a.qsort (fun a b => a.1 < b.1) → x ∈ a) :
    ∀ (k : Nat) (j : Json) {s : String}, sizeOf j < k → Flock.canon j = .ok s → Claims.Json.Ascii s
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, s, hk, h => by
    match j, h with
    | .null, h => rw [Flock.canon] at h; cases h; exact ascii_lit (by decide)
    | .bool b, h => rw [Flock.canon] at h; cases h; cases b <;> exact ascii_lit (by decide)
    | .num n, h =>
      rw [Flock.canon] at h
      split at h
      · cases h; exact ascii_int _
      · cases h
    | .str t, h => rw [Flock.canon] at h; cases h; exact canonStr_ascii t
    | .arr a, h =>
      rw [Flock.canon] at h
      cases hp : a.attach.mapM (fun x => Flock.canon x.1) with
      | error e => rw [hp] at h; cases h
      | ok parts =>
        rw [hp] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ascii_append (ascii_append (ascii_lit (by decide)) (ascii_intercalate (ascii_lit (by decide)) ?_))
          (ascii_lit (by decide))
        intro y hy
        obtain ⟨x, -, hx⟩ := array_mem_of_mapM_ok hp y (Array.mem_toList_iff.1 hy)
        have hlt := Array.sizeOf_lt_of_mem x.2
        simp only [Json.arr.sizeOf_spec] at hk
        exact canon_ascii_lt hq k x.1 (by omega) hx
    | .obj m, h =>
      rw [Flock.canon] at h
      cases hp : m.toArray.attach.mapM (fun p => do pure (p.1.1, ← Flock.canon p.1.2) : _ → Flock.V (String × String)) with
      | error e => rw [hp] at h; cases h
      | ok vals =>
        rw [hp] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ascii_append (ascii_append (ascii_lit (by decide)) (ascii_intercalate (ascii_lit (by decide)) ?_))
          (ascii_lit (by decide))
        intro y hy
        obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hy
        obtain ⟨p, -, hpv⟩ := array_mem_of_mapM_ok hp kv (hq _ _ (Array.mem_toList_iff.1 hkv))
        cases hc : Flock.canon p.1.2 with
        | error e => rw [hc] at hpv; cases hpv
        | ok v =>
          rw [hc] at hpv
          simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hpv
          subst hpv
          have hlt := Flock.sizeOf_lt_of_mem_toArray p.2
          refine ascii_append (ascii_append (canonStr_ascii _) (ascii_lit (by decide))) ?_
          exact canon_ascii_lt hq k p.1.2 (by omega) hc

end Ascii

/-- **`canon` writes ASCII without DEL.** -/
theorem canon_ascii (j : Json) {s : String} (h : canon j = .ok s) : Claims.Json.Ascii s :=
  Ascii.canon_ascii_lt (fun a x hx => (FlockSoundness.Audit.DrawExec.mem_qsort a _ x).1 hx) _ j (Nat.lt_succ_self _) h

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

/-- **`canon` returns on printable values.** -/
theorem canon_ok_some : ∀ j : Json, Good j → ∃ s, canon j = .ok s
  | _, .null => ⟨_, by rw [canon]; rfl⟩
  | _, .bool _ => ⟨_, by rw [canon]; rfl⟩
  | _, .num _ => ⟨_, by rw [canon]; rfl⟩
  | _, .str _ => ⟨_, by rw [canon]; rfl⟩
  | .arr a, .arr h => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok (f := fun x : {x // x ∈ a} => canon x.1) (a := a.attach)
      fun x _ => canon_ok_some x.1 (h x.1 x.2)
    exact ⟨_, by rw [canon, hr]; rfl⟩
  | .obj m, .obj hm h => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok
      (f := fun p : {p // p ∈ m.toArray} => do pure (p.1.1, ← canon p.1.2)) (a := m.toArray.attach)
      fun p _ => by
        have hp : (p.1.1, p.1.2) ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 p.2
        have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
        obtain ⟨s, hs⟩ := canon_ok_some p.1.2 (h _ _ (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hg))
        exact ⟨_, by rw [hs]; rfl⟩
    exact ⟨_, by rw [canon, hr]; rfl⟩

/-- **`canon` returns on printable values a text of ASCII without DEL** (`JsonSpec.canon_ok`). -/
theorem canon_ok (j : Json) (h : Good j) : ∃ s, canon j = .ok s ∧ Claims.Json.Ascii s :=
  let ⟨s, hs⟩ := canon_ok_some j h
  ⟨s, hs, canon_ascii j hs⟩

end FlockComplete.JsonRT
