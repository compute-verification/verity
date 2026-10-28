import Proofs.Hash.Bytes

/-!
# `Hash.SecurityProofs.TreeBinding : Hash.Guarantees.TreeBinding`

Two climbs of one length from one index that reach one digest start from one digest, or two of the node messages they
hashed collide (`climb_bind`), by induction up the path. At each level both climbs hash a node message at the same level
and index, with the climbing digest on the same side. The next digests are equal, by induction, or the levels above
collided. Equal next digests come from one node message, whose children are then equal (`nodeMsg_inj`), or from two
different node messages, which collide.
-/

namespace Hash.BindingProofs

open Hash.Tree

/-- **Climbs bind their start.** -/
theorem climb_bind (H : Msg → Digest) (D : Domain) :
    ∀ (s₁ s₂ : List Digest) (lv cur : ℕ) (x₁ x₂ : Digest), s₁.length = s₂.length →
      (climb H D lv cur x₁ s₁).2 = (climb H D lv cur x₂ s₂).2 →
      x₁ = x₂ ∨ CollisionIn H ((climb H D lv cur x₁ s₁).1 ++ (climb H D lv cur x₂ s₂).1)
  | [], [], _, _, _, _, _, h => Or.inl (by simpa [climb] using h)
  | [], _ :: _, _, _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, _, _, hl, _ => by simp at hl
  | a :: s₁, b :: s₂, lv, cur, x₁, x₂, hl, h => by
    let m₁ := if cur % 2 = 0 then nodeMsg D lv (cur / 2) x₁ a else nodeMsg D lv (cur / 2) a x₁
    let m₂ := if cur % 2 = 0 then nodeMsg D lv (cur / 2) x₂ b else nodeMsg D lv (cur / 2) b x₂
    have hc₁ : climb H D lv cur x₁ (a :: s₁) =
        (m₁ :: (climb H D (lv + 1) (cur / 2) (H m₁) s₁).1, (climb H D (lv + 1) (cur / 2) (H m₁) s₁).2) := rfl
    have hc₂ : climb H D lv cur x₂ (b :: s₂) =
        (m₂ :: (climb H D (lv + 1) (cur / 2) (H m₂) s₂).1, (climb H D (lv + 1) (cur / 2) (H m₂) s₂).2) := rfl
    rw [hc₁, hc₂] at h ⊢
    rcases climb_bind H D s₁ s₂ (lv + 1) (cur / 2) (H m₁) (H m₂) (by simpa using hl) h with he | hc
    · rcases eq_or_collisionIn (List.mem_append_left _ (List.mem_cons_self ..))
          (List.mem_append_right _ (List.mem_cons_self ..)) he with hm | hc
      · left
        by_cases hp : cur % 2 = 0
        · simp only [m₁, m₂, hp, ↓reduceIte] at hm
          exact (nodeMsg_inj hm).1
        · simp only [m₁, m₂, hp, ↓reduceIte] at hm
          exact (nodeMsg_inj hm).2
      · exact Or.inr hc
    · refine Or.inr (collisionIn_mono ?_ hc)
      intro z hz
      simp only [List.mem_append, List.mem_cons] at hz ⊢
      tauto

end Hash.BindingProofs

/-- **Frame-v3 tree binding** (`Hash.Guarantees.TreeBinding`). -/
theorem Hash.SecurityProofs.TreeBinding : Hash.Guarantees.TreeBinding := by
  intro H D root rank l₁ l₂ sibs₁ sibs₂ h₁ h₂
  exact Hash.BindingProofs.climb_bind H D sibs₁ sibs₂ 0 rank l₁ l₂ (h₁.2.1.trans h₂.2.1.symm)
    (h₁.2.2.trans h₂.2.2.symm)
