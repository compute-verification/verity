module

public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.SpotFlock.Code
public import Std.Data.String.ToInt
import all Init.Data.Int.Repr

/-! What `jeq`, which compares canonical texts, tells of a value it holds to a string or a number. -/

@[expose] public section

namespace SpotFlock

open Lean (Json JsonNumber)

theorem int_head (m : Int) : ∃ c, (toString m).toList.head? = some c ∧ (c.isDigit = true ∨ c = '-') := by
  cases m with
  | ofNat k =>
    have hk : (toString (Int.ofNat k)).toList = Nat.toDigits 10 k := Nat.toList_repr
    obtain ⟨c, r, hcr⟩ := List.exists_cons_of_ne_nil (Nat.toDigits_ne_nil (n := k) (b := 10))
    refine ⟨c, by rw [hk, hcr]; rfl,
      .inl (Nat.isDigit_of_mem_toDigits (b := 10) (n := k) (by decide) (by decide) ?_)⟩
    rw [hcr]
    exact List.mem_cons_self
  | negSucc k =>
    refine ⟨'-', ?_, .inr rfl⟩
    show ("-" ++ Nat.repr (k + 1)).toList.head? = _
    simp

theorem ok_bind_pure {α : Type} {x : Except String α} {f : α → String} {t : String}
    (h : (x >>= fun a => pure (f a)) = .ok t) : ∃ a, t = f a := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, (Except.ok.inj h).symm⟩

theorem canon_head {b : Json} {t : String} (h : Flock.canon b = .ok t) :
    ∃ c, t.toList.head? = some c ∧ (c = '"' → ∃ s, b = .str s) ∧ (c.isDigit = true → ∃ n, b = .num n) := by
  cases b with
  | null =>
    rw [Flock.canon] at h
    obtain rfl := (Except.ok.inj h).symm
    exact ⟨'n', by simp, fun hc => absurd hc (by decide), fun hc => absurd hc (by decide)⟩
  | bool v =>
    rw [Flock.canon] at h
    obtain rfl := (Except.ok.inj h).symm
    cases v
    · exact ⟨'f', by simp, fun hc => absurd hc (by decide), fun hc => absurd hc (by decide)⟩
    · exact ⟨'t', by simp, fun hc => absurd hc (by decide), fun hc => absurd hc (by decide)⟩
  | num n =>
    rw [Flock.canon] at h
    split at h
    · obtain rfl := (Except.ok.inj h).symm
      obtain ⟨c, hcr, hc⟩ := int_head n.mantissa
      refine ⟨c, hcr, fun hq => ?_, fun _ => ⟨n, rfl⟩⟩
      subst hq
      rcases hc with hc | hc <;> exact absurd hc (by decide)
    · cases h
  | str s =>
    rw [Flock.canon] at h
    obtain rfl := (Except.ok.inj h).symm
    refine ⟨'"', ?_, fun _ => ⟨s, rfl⟩, fun hc => absurd hc (by decide)⟩
    rw [FlockComplete.JsonRT.Ascii.canonStr_toList]
    rfl
  | arr a =>
    rw [Flock.canon] at h
    obtain ⟨parts, rfl⟩ := ok_bind_pure h
    exact ⟨'[', by simp, fun hc => absurd hc (by decide), fun hc => absurd hc (by decide)⟩
  | obj kvs =>
    rw [Flock.canon] at h
    obtain ⟨vals, rfl⟩ := ok_bind_pure h
    exact ⟨'{', by simp, fun hc => absurd hc (by decide), fun hc => absurd hc (by decide)⟩

theorem str_of_jeq {s : String} {b : Json} (h : Spot.jeq (.str s) b = true) : b = .str s := by
  have hs : Flock.canon (.str s) = .ok (Flock.canonStr s) := by rw [Flock.canon]; rfl
  unfold Spot.jeq at h
  rw [hs] at h
  cases hb : Flock.canon b with
  | error e => rw [hb] at h; simp at h
  | ok t =>
    rw [hb] at h
    have ht : Flock.canonStr s = t := by simpa using h
    obtain ⟨c, hcr, hq, -⟩ := canon_head hb
    have hc : c = '"' := by
      rw [← ht, FlockComplete.JsonRT.Ascii.canonStr_toList, List.cons_append, List.head?_cons] at hcr
      exact (Option.some.inj hcr).symm
    obtain ⟨s', rfl⟩ := hq hc
    have hs' : Flock.canon (.str s') = .ok (Flock.canonStr s') := by rw [Flock.canon]; rfl
    obtain ⟨j, hj, hsj⟩ := FlockComplete.JsonRT.parse_canon (.str s) _ (.str s) hs
    obtain ⟨j', hj', hsj'⟩ := FlockComplete.JsonRT.parse_canon (.str s') _ (.str s') hs'
    rw [hs', Except.ok.injEq] at hb
    rw [hb, ← ht, hj] at hj'
    cases Except.ok.inj hj'
    cases hsj
    cases hsj'
    rfl

theorem nat?_of_jeq {n : ℕ} {b : Json} (h : Spot.jeq (.num ⟨Int.ofNat n, 0⟩) b = true) : Spot.nat? b = some n := by
  have hn : Flock.canon (.num ⟨Int.ofNat n, 0⟩) = .ok (Nat.repr n) := by rw [Flock.canon]; rfl
  unfold Spot.jeq at h
  rw [hn] at h
  cases hb : Flock.canon b with
  | error e => rw [hb] at h; simp at h
  | ok t =>
    rw [hb] at h
    have ht : Nat.repr n = t := by simpa using h
    obtain ⟨c, hcr, -, hd⟩ := canon_head hb
    have hc : c.isDigit = true := by
      rw [← ht, Nat.toList_repr] at hcr
      obtain ⟨tl, htl⟩ := List.head?_eq_some_iff.1 hcr
      refine Nat.isDigit_of_mem_toDigits (b := 10) (n := n) (by decide) (by decide) ?_
      rw [htl]
      exact List.mem_cons_self
    obtain ⟨⟨m, e⟩, rfl⟩ := hd hc
    rw [Flock.canon] at hb
    split at hb
    · rename_i he
      have he : e = 0 := by simpa using he
      subst he
      have hm : m = Int.ofNat n :=
        Int.repr_injective (a := m) (b := Int.ofNat n) ((Except.ok.inj hb).trans ht.symm)
      subst hm
      simp [Spot.nat?, Spot.int?]
    · cases hb

theorem hexMsg_str {s : String} {r : ByteArray} (h : Flock.unhex s = .ok r) :
    Claims.Sampling.bytes (hexMsg (.str s)) = r := by
  simp only [hexMsg, h, Except.toOption, Option.map_some, Option.getD_some]
  exact Claims.Sampling.bytes_bs r

theorem jeq_symm {a b : Json} (h : Spot.jeq a b = true) : Spot.jeq b a = true := by
  unfold Spot.jeq at h ⊢
  cases ha : Flock.canon a <;> cases hb : Flock.canon b <;> rw [ha, hb] at h <;> simp_all

theorem hexMsg_nil {j : Json} (hj : ¬ ∃ s, j = .str s) : hexMsg j = [] := by
  cases j <;> first | rfl | exact absurd ⟨_, rfl⟩ hj

theorem hexMsg_of_jeq {a b : Json} (h : Spot.jeq a b = true) : hexMsg a = hexMsg b := by
  by_cases ha : ∃ s, a = .str s
  · obtain ⟨s, rfl⟩ := ha
    rw [str_of_jeq h]
  · have hb : ¬ ∃ s, b = .str s := fun ⟨s, hs⟩ => ha ⟨s, str_of_jeq (by rw [← hs]; exact jeq_symm h)⟩
    rw [hexMsg_nil ha, hexMsg_nil hb]

end SpotFlock
