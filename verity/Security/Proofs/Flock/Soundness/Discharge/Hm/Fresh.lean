/-
Where the builder's rows put their outputs: after `nIn` input bits only `fresh` adds rows, each outputting the next bit,
so the `k`-th row in creation order outputs bit `nIn + k` (`Good`), the column `Flock.HmNets.layout` gives it
(`col_and`). `Fresh m`: running `m` keeps that. Both nets keep it from the end of their inputs (`good_compression3`,
`good_hm96Row`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Top

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- After `nIn` input bits, the rows (creation order) output bits `nIn, nIn + 1, …`, and the next bit follows them. -/
def Good (nIn : ℕ) (s : St) : Prop :=
  s.next = nIn + s.rows.length ∧ ∀ (k : ℕ) (r : Row), s.rows.reverse[k]? = some r → r.out = nIn + k

/-- `m` adds rows only as `fresh` does. -/
def Fresh {α : Type} (m : M α) : Prop := ∀ nIn s, Good nIn s → Good nIn (m.run s).2

namespace Fresh

variable {α β : Type}

theorem pure' (a : α) : Fresh (pure a : M α) := fun _ _ h => h

theorem bind' {m : M α} {f : α → M β} (hm : Fresh m) (hf : ∀ a, Fresh (f a)) : Fresh (m >>= f) :=
  fun nIn s h => hf _ nIn _ (hm nIn s h)

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} (h₁ : Fresh m₁) (h₂ : Fresh m₂) :
    Fresh (if c then m₁ else m₂) := by
  split_ifs
  · exact h₁
  · exact h₂

theorem mapM' (f : β → M α) (hf : ∀ b, Fresh (f b)) : ∀ l : List β, Fresh (l.mapM f)
  | [] => by rw [List.mapM_nil]; exact pure' _
  | b :: l => by
      rw [List.mapM_cons]
      exact bind' (hf b) fun _ => bind' (mapM' f hf l) fun _ => pure' _

end Fresh

theorem fresh_fresh (a b : Lin) : Fresh (fresh a b) := by
  intro nIn s ⟨hn, hr⟩
  show Good nIn ⟨s.next + 1, ⟨s.next, a, b⟩ :: s.rows⟩
  refine ⟨by simp only [List.length_cons]; omega, fun k r hk => ?_⟩
  simp only [List.reverse_cons] at hk
  by_cases hlt : k < s.rows.length
  · rw [List.getElem?_append_left (by simpa using hlt)] at hk
    exact hr k r hk
  · rw [List.getElem?_append_right (by simpa using hlt), List.length_reverse] at hk
    obtain hk0 | hk0 := Nat.eq_zero_or_pos (k - s.rows.length)
    · rw [hk0] at hk
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hk
      rw [← hk]
      show s.next = nIn + k
      omega
    · rw [List.getElem?_eq_none (by simp; omega)] at hk
      cases hk

theorem fresh_and (a b : Lin) : Fresh (Flock.HmNets.and a b) := by
  unfold Flock.HmNets.and
  exact Fresh.ite' (Fresh.pure' _) (Fresh.ite' (Fresh.pure' _) (Fresh.ite' (Fresh.pure' _) (fresh_fresh a b)))

theorem fresh_maj (x y w : Lin) : Fresh (maj x y w) := by
  unfold maj
  dsimp only
  exact Fresh.ite' (Fresh.ite' (Fresh.pure' _) (Fresh.ite' (Fresh.pure' _) (Fresh.pure' _)))
    (Fresh.bind' (fresh_and _ _) fun _ => Fresh.pure' _)

theorem fresh_commit (f : Lin) : Fresh (commit f) := by
  unfold commit
  exact Fresh.ite' (Fresh.pure' _) (fresh_fresh _ _)

theorem fresh_chW (e f g : Word) : Fresh (chW e f g) :=
  Fresh.mapM' _ (fun _ => Fresh.bind' (fresh_and _ _) fun _ => Fresh.pure' _) _

theorem fresh_majW (a b c : Word) : Fresh (majW a b c) := Fresh.mapM' _ (fun _ => fresh_maj _ _ _) _

theorem fresh_commitW (w : Word) : Fresh (commitW w) := Fresh.mapM' _ fresh_commit _

theorem fresh_carryCommit (i : ℕ) (c : Lin) : Fresh (carryCommit i c) := by
  unfold carryCommit
  exact Fresh.ite' (fresh_commit c) (Fresh.pure' _)

theorem fresh_add2Go : ∀ (a b : Word) (c : Lin) (i : ℕ), Fresh (add2Go a b c i)
  | x :: a, y :: b, c, i => by
      unfold add2Go
      exact Fresh.ite' (Fresh.pure' _) (Fresh.bind' (fresh_maj x y c) fun c₁ =>
        Fresh.bind' (fresh_carryCommit i c₁) fun c₂ =>
          Fresh.bind' (fresh_add2Go a b c₂ (i + 1)) fun _ => Fresh.pure' _)
  | [], _, _, _ => by unfold add2Go; exact Fresh.pure' _
  | _ :: _, [], _, _ => by unfold add2Go; exact Fresh.pure' _

theorem fresh_add2 (a b : Word) : Fresh (add2 a b) := fresh_add2Go a b _ 0

theorem fresh_addN : ∀ (acc : Word) (xs : List Word), Fresh (addN acc xs)
  | _, [] => Fresh.pure' _
  | acc, x :: xs => Fresh.bind' (fresh_add2 acc x) fun acc' => fresh_addN acc' xs

theorem fresh_schedStep (w : List Word) : Fresh (schedStep w) :=
  Fresh.bind' (fresh_addN _ _) fun _ => Fresh.bind' (fresh_commitW _) fun _ => Fresh.pure' _

theorem fresh_schedule : ∀ (n : ℕ) (w : List Word), Fresh (schedule n w)
  | 0, _ => Fresh.pure' _
  | n + 1, w => Fresh.bind' (fresh_schedStep w) fun w' => fresh_schedule n w'

theorem fresh_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : Fresh (roundW k wt r) :=
  Fresh.bind' (fresh_chW _ _ _) fun _ => Fresh.bind' (fresh_addN _ _) fun _ =>
    Fresh.bind' (fresh_majW _ _ _) fun _ => Fresh.bind' (fresh_add2 _ _) fun _ =>
      Fresh.bind' (fresh_add2 _ _) fun _ => Fresh.bind' (fresh_commitW _) fun _ =>
        Fresh.bind' (fresh_add2 _ _) fun _ => Fresh.bind' (fresh_commitW _) fun _ => Fresh.pure' _

theorem fresh_rounds (w : List Word) : ∀ (n t : ℕ) (r : Regs Word), Fresh (rounds w n t r)
  | 0, _, _ => Fresh.pure' _
  | n + 1, t, r => Fresh.bind' (fresh_roundW _ _ r) fun r' => fresh_rounds w n (t + 1) r'

theorem fresh_compress (cv msg : List Word) : Fresh (compress cv msg) :=
  Fresh.bind' (fresh_schedule _ _) fun _ => Fresh.bind' (fresh_rounds _ _ _ _) fun _ =>
    Fresh.mapM' _ (fun _ => fresh_add2 _ _) _

theorem fresh_shaOuts (ins : List (List Lin × List Lin)) : Fresh (shaOuts ins) :=
  Fresh.mapM' _ (fun _ => Fresh.bind' (fresh_compress _ _) fun _ => Fresh.pure' _) ins

theorem fresh_hmTail (key : ByteArray) (cv pad y : List Lin) : Fresh (hmTail key cv pad y) :=
  Fresh.bind' (fresh_compress _ _) fun _ => Fresh.bind' (fresh_compress _ _) fun _ =>
    Fresh.bind' (fresh_compress _ _) fun _ => Fresh.pure' _

/-- **The sha512x3 net's rows output bits `4608, 4609, …`** in creation order, after its 4608 input bits. -/
theorem good_compression3 : Good 4608 (compression3.run St.empty).2 := by
  rw [compression3_run, snd_pair]
  exact fresh_shaOuts sha3Ins 4608 ⟨4608, []⟩ ⟨rfl, fun k r h => by simp at h⟩

/-- **The hm96 net's rows output bits `3072, 3073, …`** in creation order, after its 3072 input bits. -/
theorem good_hm96Row (key : ByteArray) : Good 3072 ((hm96Row key).run St.empty).2 := by
  rw [hm96Row_run]
  exact fresh_hmTail key _ _ _ 3072 ⟨0 + 512 + 1024 + 1536, []⟩ ⟨rfl, fun k r h => by simp at h⟩

end FlockSoundness.Discharge.Hm
