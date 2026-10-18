import Definitions.Accounting.TS.Receipt
import Proofs.Accounting.TS.ReceiptLi

/-!
# duo and fx1 checkpoint units bind their words a checkpoint a segment

`right_of_same` (Lemma B for `DuoCPUnits`): segment by segment, the earlier segments' words are honest, so each chain
starts the segment from the honest state (`state_congr`); every chain but the deficient word's absorbs the honest words
and ends the segment in the honest state, so the checkpoint's two XORs cancel them and leave that chain's full state
(`pair_eq`, by `LiLemmas.ofFn_split` over the lanes and then the lane's chains), and
`Pouw.ReceiptProofs.duoRun_update_injective` (`DuoRunWordInjective`) gives the word right. fx1's units are duo's on the
words times their odd IMAD multipliers, which `mul_odd_injective` undoes (`fx1_right_of_same`).
-/

namespace Accounting.SecurityProofs.TS.DuoCPLemmas

open Accounting.TS Pouw.Receipt Pouw.ReceiptProofs

theorem xor_cancel {n : ℕ} {a c x y : BitVec n} (h : a ^^^ x ^^^ c = a ^^^ y ^^^ c) : x = y := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have e := congrArg (fun v => v.getLsbD i) h
  simp only [BitVec.getLsbD_xor] at e
  revert e
  cases a.getLsbD i <;> cases c.getLsbD i <;> cases x.getLsbD i <;> cases y.getLsbD i <;> decide

/-- Two XORs of `m` words that agree but at `i` are equal only if the words at `i` are. -/
theorem xorAll_eq {n m : ℕ} (F G : Fin m → BitVec n) (i : Fin m) (h : ∀ j, j ≠ i → G j = F j)
    (he : xorAll G = xorAll F) : G i = F i := by
  obtain ⟨B, A, hF, hG⟩ := LiLemmas.ofFn_split F G i h
  unfold xorAll at he
  rw [hF, hG] at he
  simp only [List.foldl_append, List.foldl_cons] at he
  rw [LiLemmas.foldl_xor_self A (_ ^^^ G i), LiLemmas.foldl_xor_self A (_ ^^^ F i)] at he
  exact xor_cancel he

/-- **The checkpoint of chains that agree but one gives that chain's full state**: two families of chain states that
agree off `(ℓ₀, c₀)` and have the same XOR of `y` and the same XOR of `z` over the lanes and their chains agree at
`(ℓ₀, c₀)`. -/
theorem pair_eq {n W C : ℕ} (F G : Fin W → Fin C → BitVec n × BitVec n) (ℓ₀ : Fin W) (c₀ : Fin C)
    (h : ∀ ℓ c, (ℓ, c) ≠ (ℓ₀, c₀) → G ℓ c = F ℓ c)
    (he : ((xorAll fun ℓ => xorAll fun c => (G ℓ c).1), (xorAll fun ℓ => xorAll fun c => (G ℓ c).2)) =
      ((xorAll fun ℓ => xorAll fun c => (F ℓ c).1), (xorAll fun ℓ => xorAll fun c => (F ℓ c).2))) :
    G ℓ₀ c₀ = F ℓ₀ c₀ := by
  have half : ∀ π : BitVec n × BitVec n → BitVec n,
      (xorAll fun ℓ => xorAll fun c => π (G ℓ c)) = (xorAll fun ℓ => xorAll fun c => π (F ℓ c)) →
        π (G ℓ₀ c₀) = π (F ℓ₀ c₀) := by
    intro π hπ
    have hl := xorAll_eq (fun ℓ => xorAll fun c => π (F ℓ c)) (fun ℓ => xorAll fun c => π (G ℓ c)) ℓ₀
      (fun ℓ hℓ => congrArg xorAll (funext fun c => congrArg π (h ℓ c fun e => hℓ (congrArg Prod.fst e)))) hπ
    exact xorAll_eq (fun c => π (F ℓ₀ c)) (fun c => π (G ℓ₀ c)) c₀
      (fun c hc => congrArg π (h ℓ₀ c fun e => hc (congrArg Prod.snd e))) hl
  have e := Prod.mk.inj he
  exact Prod.ext (half Prod.fst e.1) (half Prod.snd e.2)

theorem state_succ {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U)
    (w : ℕ → CPPlace W C H → BitVec n) (s : ℕ) (ℓ : Fin W) (c : Fin C) :
    D.state ω u w (s + 1) ℓ c =
      duoRun (fun j => D.k ω u s (ℓ, c, j)) (fun j => (D.a ω u s (ℓ, c, j)).1) (fun j => D.r ω u s (ℓ, c, j))
        (fun j => w s (ℓ, c, j)) (D.state ω u w s ℓ c) := by
  first
  | rfl
  | simp only [DuoCPUnits.state]

/-- The chains' states at segment `s` read only the words of the segments before it. -/
theorem state_congr {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U)
    (w w' : ℕ → CPPlace W C H → BitVec n) :
    ∀ s, (∀ t < s, w' t = w t) → D.state ω u w' s = D.state ω u w s
  | 0, _ => rfl
  | s + 1, h => by
    funext ℓ c
    rw [state_succ, state_succ, h s (Nat.lt_succ_self s),
      state_congr D ω u w w' s fun t ht => h t (Nat.lt_succ_of_lt ht)]

/-- **A segment's checkpoint is injective in each word**: from the honest states, the honest words with the word at
`p` replaced by `v` keep the segment's checkpoint only if `v` is the honest word. -/
theorem segment_update {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U)
    (w w' : ℕ → CPPlace W C H → BitVec n) (s : ℕ) (hy : D.state ω u w' s = D.state ω u w s) (p : CPPlace W C H)
    (v : BitVec n) (hw : w' s = Function.update (w s) p v) (h : D.checkpoint ω u w' s = D.checkpoint ω u w s) :
    v = w s p := by
  obtain ⟨ℓ₀, c₀, j₀⟩ := p
  have hoff : ∀ ℓ c, (ℓ, c) ≠ (ℓ₀, c₀) → D.state ω u w' (s + 1) ℓ c = D.state ω u w (s + 1) ℓ c := by
    intro ℓ c hne
    rw [state_succ, state_succ, hy, hw]
    congr 1
    funext j
    have hne' : ((ℓ, c, j) : CPPlace W C H) ≠ (ℓ₀, c₀, j₀) :=
      fun e => hne (congrArg (fun q : CPPlace W C H => (q.1, q.2.1)) e)
    exact Function.update_of_ne hne' _ _
  have hend := pair_eq (D.state ω u w (s + 1)) (D.state ω u w' (s + 1)) ℓ₀ c₀ hoff h
  rw [state_succ, state_succ, hy, hw] at hend
  have hu : (fun j => Function.update (w s) (ℓ₀, c₀, j₀) v (ℓ₀, c₀, j)) =
      Function.update (fun j => w s (ℓ₀, c₀, j)) j₀ v := by
    funext j
    by_cases hj : j = j₀
    · subst hj
      simp only [Function.update_self]
    · have hne : ((ℓ₀, c₀, j) : CPPlace W C H) ≠ (ℓ₀, c₀, j₀) :=
        fun e => hj (congrArg (fun q : CPPlace W C H => q.2.2) e)
      simp only [Function.update_of_ne hj, Function.update_of_ne hne]
  rw [hu] at hend
  exact duoRun_update_injective (fun j => D.k ω u s (ℓ₀, c₀, j)) (fun j => D.a ω u s (ℓ₀, c₀, j))
    (fun j => D.r ω u s (ℓ₀, c₀, j)) (D.state ω u w s ℓ₀ c₀) (fun j => w s (ℓ₀, c₀, j)) j₀
    (by simpa only [Function.update_eq_self] using hend)

/-- Segment by segment: runs `w` that are the honest words `h` with `v s` at `p s`, whose every checkpoint before `S`
is the honest one, have every replaced word right. -/
theorem right_of_checkpoints {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (u : U)
    (w h : ℕ → CPPlace W C H → BitVec n) (p : ℕ → CPPlace W C H) (v : ℕ → BitVec n)
    (hw : ∀ s, w s = Function.update (h s) (p s) (v s))
    (hc : ∀ s < S, D.checkpoint ω u w s = D.checkpoint ω u h s) : ∀ s < S, v s = h s (p s) := by
  have key : ∀ s, s ≤ S → ∀ t < s, v t = h t (p t) := by
    intro s
    induction s with
    | zero => intro _ t ht; omega
    | succ s ih =>
      intro hs t ht
      have ih' := ih (by omega)
      rcases Nat.lt_succ_iff_lt_or_eq.mp ht with hlt | rfl
      · exact ih' t hlt
      · have hwt : ∀ t' < t, w t' = h t' := fun t' ht' => by
          rw [hw t', ih' t' ht']
          exact Function.update_eq_self _ _
        exact segment_update D ω u h w t (state_congr D ω u h w t hwt) (p t) (v t) (hw t) (hc t (by omega))
  exact fun s hs => key S le_rfl s hs

/-- **Lemma B for duo checkpoint units**: every segment's checkpoint the honest one gives every segment's deficient
word right. -/
theorem right_of_same {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω) (h : D.same ω) :
    D.right ω := by
  intro u
  exact right_of_checkpoints D ω u (D.word ω u) (D.honest ω u) (D.pos ω u) (D.value ω u) (fun _ => rfl) (h u)

/-- **Lemma B for fx1 checkpoint units**: the words times their odd IMAD multipliers are duo's checkpoint units' words,
and multiplying by an odd word is injective. -/
theorem fx1_right_of_same {Ω U : Type} {n S W C H : ℕ} (D : Fx1CPUnits Ω U n S W C H) (ω : Ω) (h : D.same ω) :
    D.right ω := by
  intro u s hs
  have hw : ∀ t, D.imad ω u (D.word ω u) t =
      Function.update (D.imad ω u (D.honest ω u) t) (D.pos ω u t)
        (D.value ω u t * (D.m ω u t (D.pos ω u t)).1) := by
    intro t
    funext q
    simp only [Fx1CPUnits.imad, DuoCPUnits.word]
    by_cases hq : q = D.pos ω u t
    · subst hq
      simp only [Function.update_self]
    · simp only [Function.update_of_ne hq]
  have e := right_of_checkpoints D.toDuoCPUnits ω u (D.imad ω u (D.word ω u)) (D.imad ω u (D.honest ω u))
    (D.pos ω u) (fun t => D.value ω u t * (D.m ω u t (D.pos ω u t)).1) hw (h u) s hs
  exact mul_odd_injective (D.m ω u s (D.pos ω u s)) (a₁ := D.value ω u s) (a₂ := D.honest ω u s (D.pos ω u s)) e

end Accounting.SecurityProofs.TS.DuoCPLemmas
