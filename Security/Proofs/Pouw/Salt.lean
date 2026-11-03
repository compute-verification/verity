import Definitions.Pouw.Guarantees.KeyedLeaf
import Proofs.Pouw.Prob
import Proofs.Hash.RowBinding

/-!
# `Pouw.Salt.keyed_leaf_hides : Pouw.Guarantees.KeyedLeaf.Hides` and `Pouw.Salt.keyed_leaf_binds`

**Hiding.** A window's view is the hiding game's first side at `A`'s hiding distinguisher, by definition. Its second
side is the keyed digest's game's first side at `A`'s keyed-digest distinguisher, by the change of coins
`(ks, b, y₀) ↦ ((ks ⊕ b, y₀), ks)` (`swap`), a bijection since xor with `ks` is an involution. So the view's probability
is the two advantages plus the keyed digest's ideal side (`real_eq`). That ideal side reads the tops only through which
of them are equal (`rep`), and distinct domain ids make the tops distinct (`tops_injective`: the id sits at one offset of
every top message, a node's and a frame leaf's tags being of one length). At two windows of distinct tops the ideal
sides are one number, and the difference of the views' probabilities is the difference of their advantages.

**Binding.** Both openings' keys are the key leaf's, so they are one key or the key leaf's messages collide
(`hm96_bind`). Under one key, the two keyed messages have one digest, so they are one message, and their top messages
are equal, or they collide. Two climbs of one length to one top message start from one message, or two of the messages
they hashed collide (`keyed_climb_bind`, by induction up the path as `climb_bind`), and two frame leaves at one rank of one
domain under one schema have the same value (`leafMsg_value`).
-/

namespace Pouw.Salt

open Hash Hash.BindingProofs Pouw.KeyedLeaf

/-! ## Hiding -/

theorem xor_xor (a b : UInt8) : a ^^^ (a ^^^ b) = b := by
  rw [← UInt8.xor_assoc, UInt8.xor_self, UInt8.zero_xor]

/-- The change of coins from the hiding game's second side to the keyed digest's game's first side. -/
def swap : Digest × (Digest × KeySalt) ≃ (Digest × KeySalt) × Digest where
  toFun p := ((fun i => p.1 i ^^^ p.2.1 i, p.2.2), p.1)
  invFun q := (q.2, (fun i => q.2 i ^^^ q.1.1 i, q.1.2))
  left_inv _ := Prod.ext rfl (Prod.ext (funext fun _ => xor_xor _ _) rfl)
  right_inv _ := Prod.ext (Prod.ext (funext fun _ => xor_xor _ _) rfl) rfl

/-- The view's probability is the two advantages plus the keyed digest's ideal side. -/
theorem real_eq {n : ℕ} (H : Msg → Digest) (key : Msg) (salt : Digest) (top : Fin n → Msg)
    (A : Digest × (Fin n → Digest) → Prop) :
    real H key salt top A = hideAdvantage H key (hideDistinguisher H key salt top A) +
      prfAdvantage H salt top (prfDistinguisher H key A) +
      pr (fun p : (Digest × KeySalt) × (Fin n → Digest) => prfDistinguisher H key A (p.1, fun i => p.2 (rep top i))) := by
  have h₁ : real H key salt top A = pr (fun p : Digest × KeySalt =>
      hideDistinguisher H key salt top A (p.1, (mask key p.2, saltDigest H p.2))) := rfl
  have h₂ : pr (fun p : Digest × (Digest × KeySalt) =>
        hideDistinguisher H key salt top A (p.1, (p.2.1, saltDigest H p.2.2))) =
      pr (fun p : (Digest × KeySalt) × Digest =>
        prfDistinguisher H key A (p.1, fun i => Keyed.digest H p.2 salt (top i))) :=
    pr_equiv swap (fun p : (Digest × KeySalt) × Digest =>
      prfDistinguisher H key A (p.1, fun i => Keyed.digest H p.2 salt (top i)))
  unfold hideAdvantage prfAdvantage
  rw [h₁, h₂]
  ring

theorem rep_of_injective {n : ℕ} {top : Fin n → Msg} (h : Function.Injective top) (i : Fin n) : rep top i = i :=
  h (Finset.mem_filter.1 (Finset.min'_mem (Finset.univ.filter fun j => top j = top i)
    ⟨i, Finset.mem_filter.2 ⟨Finset.mem_univ i, rfl⟩⟩)).2

/-- A frame whose first part is a digest, under a tag of one length, determines that digest. -/
theorem frame_id {tag tag' : String} {d d' : Digest} {ps ps' : List Msg} (hl : (utf8 tag).length = (utf8 tag').length)
    (h : Tree.frame tag (d.bytes :: ps) = Tree.frame tag' (d'.bytes :: ps')) : d = d' := by
  simp only [Tree.frame, List.flatMap_cons, bytes_length, List.append_assoc, hl, List.append_cancel_left_eq] at h
  obtain ⟨-, h⟩ := List.append_inj h hl
  have h := List.append_cancel_left h
  exact bytes_injective (List.append_inj h (by simp)).1

/-- Two top messages are equal only under one domain id. -/
theorem top_id {D D' : Tree.Domain} {t t' : Keyed.Top} (h : t.msg D = t'.msg D') : D.id = D'.id := by
  cases t <;> cases t' <;> exact frame_id (by decide) h

theorem tops_injective {n : ℕ} {dom : Fin n → Tree.Domain} (hd : ∀ i j, i ≠ j → (dom i).id ≠ (dom j).id)
    (t : Fin n → Keyed.Top) : Function.Injective (tops dom t) := fun i j h =>
  by_contra fun hij => hd i j hij (top_id h)

theorem ideal_eq {n : ℕ} (H : Msg → Digest) (key : Msg) {top : Fin n → Msg} (h : Function.Injective top)
    (A : Digest × (Fin n → Digest) → Prop) :
    pr (fun p : (Digest × KeySalt) × (Fin n → Digest) => prfDistinguisher H key A (p.1, fun i => p.2 (rep top i))) =
      pr (fun p : (Digest × KeySalt) × (Fin n → Digest) => prfDistinguisher H key A (p.1, p.2)) := by
  simp only [rep_of_injective h]

/-- **A keyed window hides its tops** (`Pouw.Guarantees.KeyedLeaf.Hides`). -/
theorem keyed_leaf_hides : Pouw.Guarantees.KeyedLeaf.Hides := by
  intro H key salt n dom t t' A
  by_cases hd : ∃ i j, i ≠ j ∧ (dom i).id = (dom j).id
  · exact Or.inl hd
  right
  push Not at hd
  rw [real_eq, real_eq, ideal_eq H key (tops_injective hd t), ideal_eq H key (tops_injective hd t')]
  unfold advantage
  set a := hideAdvantage H key (hideDistinguisher H key salt (tops dom t) A)
  set b := prfAdvantage H salt (tops dom t) (prfDistinguisher H key A)
  set a' := hideAdvantage H key (hideDistinguisher H key salt (tops dom t') A)
  set b' := prfAdvantage H salt (tops dom t') (prfDistinguisher H key A)
  set w := pr (fun p : (Digest × KeySalt) × (Fin n → Digest) => prfDistinguisher H key A (p.1, p.2))
  rw [show a + b + w - (a' + b' + w) = (a + b) + -(a' + b') by ring]
  calc |(a + b) + -(a' + b')| ≤ |a + b| + |-(a' + b')| := abs_add_le _ _
    _ = |a + b| + |a' + b'| := by rw [abs_neg]
    _ ≤ (|a| + |b|) + (|a'| + |b'|) := add_le_add (abs_add_le _ _) (abs_add_le _ _)

/-! ## Binding -/

/-- **Keyed climbs bind their start.** -/
theorem keyed_climb_bind (H : Msg → Digest) (D : Tree.Domain) :
    ∀ (s₁ s₂ : List Digest) (lv cur : ℕ) (m₁ m₂ : Msg), s₁.length = s₂.length →
      (Keyed.climb H D lv cur m₁ s₁).2 = (Keyed.climb H D lv cur m₂ s₂).2 →
      m₁ = m₂ ∨ CollisionIn H ((Keyed.climb H D lv cur m₁ s₁).1 ++ (Keyed.climb H D lv cur m₂ s₂).1)
  | [], [], _, _, _, _, _, h => Or.inl (by simpa [Keyed.climb] using h)
  | [], _ :: _, _, _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, _, _, hl, _ => by simp at hl
  | a :: s₁, b :: s₂, lv, cur, m₁, m₂, hl, h => by
    let n₁ := if cur % 2 = 0 then Tree.nodeMsg D lv (cur / 2) (H m₁) a else Tree.nodeMsg D lv (cur / 2) a (H m₁)
    let n₂ := if cur % 2 = 0 then Tree.nodeMsg D lv (cur / 2) (H m₂) b else Tree.nodeMsg D lv (cur / 2) b (H m₂)
    have hc₁ : Keyed.climb H D lv cur m₁ (a :: s₁) =
        (m₁ :: (Keyed.climb H D (lv + 1) (cur / 2) n₁ s₁).1, (Keyed.climb H D (lv + 1) (cur / 2) n₁ s₁).2) := rfl
    have hc₂ : Keyed.climb H D lv cur m₂ (b :: s₂) =
        (m₂ :: (Keyed.climb H D (lv + 1) (cur / 2) n₂ s₂).1, (Keyed.climb H D (lv + 1) (cur / 2) n₂ s₂).2) := rfl
    rw [hc₁, hc₂] at h ⊢
    rcases keyed_climb_bind H D s₁ s₂ (lv + 1) (cur / 2) n₁ n₂ (by simpa using hl) h with hn | hc
    · have hH : H m₁ = H m₂ := by
        by_cases hp : cur % 2 = 0
        · simp only [n₁, n₂, hp, ↓reduceIte] at hn
          exact (nodeMsg_inj hn).1
        · simp only [n₁, n₂, hp, ↓reduceIte] at hn
          exact (nodeMsg_inj hn).2
      exact eq_or_collisionIn (List.mem_append_left _ (List.mem_cons_self ..))
        (List.mem_append_right _ (List.mem_cons_self ..)) hH
    · refine Or.inr (collisionIn_mono ?_ hc)
      intro z hz
      simp only [List.mem_append, List.mem_cons] at hz ⊢
      tauto

/-- Two frame leaves at one rank, in one domain and under one schema, have the same value. -/
theorem leafMsg_value {D : Tree.Domain} {rank : ℕ} {schema : String} {v w : Msg}
    (h : Tree.leafMsg D rank schema v = Tree.leafMsg D rank schema w) : v = w := by
  have h := frame_parts h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, bytes_length, List.append_assoc,
    List.append_cancel_left_eq] at h
  exact (List.append_inj h (by simp)).2

/-- **A keyed leaf binds its values** (`Pouw.Guarantees.KeyedLeaf.Binds`). -/
theorem keyed_leaf_binds : Pouw.Guarantees.KeyedLeaf.Binds := by
  intro H key keyLeaf salt dom schema root Opens hL hS pos rank v₁ v₂ ks₁ ks₂ y₁ y₂ sibs₁ sibs₂ h₁ h₂
  have a₁ := hL _ _ _ _ _ _ h₁
  have a₂ := hL _ _ _ _ _ _ h₂
  rcases hm96_bind H key ((hS _ _ _ _ _ _ h₁).symm.trans (hS _ _ _ _ _ _ h₂)) with hks | hc
  swap
  · refine Or.inr (collisionIn_mono ?_ hc)
    intro z hz
    simp only [List.mem_append] at hz ⊢
    tauto
  subst hks
  set c₁ := Keyed.climb H (dom pos) 0 rank (Tree.leafMsg (dom pos) rank schema v₁) sibs₁
  set c₂ := Keyed.climb H (dom pos) 0 rank (Tree.leafMsg (dom pos) rank schema v₂) sibs₂
  set xs := Hm96.hashed H key ks₁ y₁ ++ Hm96.hashed H key ks₁ y₂ ++
      Keyed.hashed H ks₁ salt (dom pos) rank (Tree.leafMsg (dom pos) rank schema v₁) sibs₁ ++
      Keyed.hashed H ks₁ salt (dom pos) rank (Tree.leafMsg (dom pos) rank schema v₂) sibs₂
  have hsub : c₁.1 ++ c₂.1 ⊆ xs := by
    intro z hz
    simp only [xs, Keyed.hashed, List.mem_append] at hz ⊢
    tauto
  have hm₁ : Keyed.msg ks₁ salt c₁.2 ∈ xs := by simp [xs, Keyed.hashed, c₁]
  have hm₂ : Keyed.msg ks₁ salt c₂.2 ∈ xs := by simp [xs, Keyed.hashed, c₂]
  rcases eq_or_collisionIn hm₁ hm₂ (a₁.2.2.trans a₂.2.2.symm) with ht | hc
  swap
  · exact Or.inr hc
  have ht : ks₁.bytes ++ (salt.bytes ++ c₁.2) = ks₁.bytes ++ (salt.bytes ++ c₂.2) := by
    simpa only [Keyed.msg, List.append_assoc] using ht
  have ht : c₁.2 = c₂.2 := List.append_cancel_left (List.append_cancel_left ht)
  rcases keyed_climb_bind H (dom pos) sibs₁ sibs₂ 0 rank _ _ (a₁.2.1.trans a₂.2.1.symm) ht with hl | hc
  · exact Or.inl (leafMsg_value hl)
  · exact Or.inr (collisionIn_mono hsub hc)

end Pouw.Salt
