import Definitions.Accounting.TS.Receipt
import Proofs.Pouw.Receipt

/-!
# li units bind their words a checkpoint a segment, and one word a window does not

`right_of_same` (Lemma B for `LiUnits`): segment by segment, the earlier segments' words are honest, so each chain
starts the segment from the honest state (`state_congr`); the warp's checkpoint is the checkpoint of all its chains
(`redux_eq`, XOR is associative), and the run's chains differ from the honest ones in one word of one half step of one
chain (`ofFn_split`, three times), so `Pouw.SecurityProofs.Receipt.LiCheckpointWordInjective` gives the word right.

`windowUnbound`: at width `n + 1` the top bit `T` passes through a half step with an odd multiplier as XOR
(`(x ⊕ T)·a = x·a ⊕ T`, `Pouw.ReceiptProofs.add_twoPow_top`, `twoPow_mul_odd`), so a flip of `d₀`'s top bit moves the
state by `T`, and a second flip of the next segment's `d₀` takes it back.
-/

namespace Accounting.SecurityProofs.TS.LiLemmas

open Accounting.TS Pouw.Receipt Pouw.ReceiptProofs

theorem ofFn_eq_set {α : Type} {m : ℕ} (F G : Fin m → α) (i : Fin m) (h : ∀ j, j ≠ i → G j = F j) :
    List.ofFn G = (List.ofFn F).set i (G i) := by
  apply List.ext_getElem (by simp)
  intro j h1 h2
  rw [List.getElem_set]
  simp only [List.getElem_ofFn]
  split_ifs with hij
  · subst hij; rfl
  · exact h ⟨j, by simpa using h1⟩ fun e => hij (by rw [← e])

/-- Two functions on `Fin m` that agree but at `i` give lists that share what lies before and after `i`. -/
theorem ofFn_split {α : Type} {m : ℕ} (F G : Fin m → α) (i : Fin m) (h : ∀ j, j ≠ i → G j = F j) :
    ∃ B A : List α, List.ofFn F = B ++ F i :: A ∧ List.ofFn G = B ++ G i :: A := by
  have hlen : (i : ℕ) < (List.ofFn F).length := by simp
  have hset : ∀ x, (List.ofFn F).set i x = (List.ofFn F).take i ++ x :: (List.ofFn F).drop (i + 1) := fun x => by
    rw [List.set_eq_take_append_cons_drop, ite_eq_left_iff.mpr fun h => absurd hlen h]
  refine ⟨(List.ofFn F).take i, (List.ofFn F).drop (i + 1), ?_, ?_⟩
  · rw [← hset, ← ofFn_eq_set F F i fun _ _ => rfl]
  · rw [← hset, ← ofFn_eq_set F G i h]

theorem foldl_xor_self {n : ℕ} (l : List (BitVec n)) (z : BitVec n) :
    l.foldl (· ^^^ ·) z = z ^^^ l.foldl (· ^^^ ·) 0 :=
  foldl_xor (fun c => c) l z

theorem liCheckpoint_append {n : ℕ} (l₁ l₂ : List (List (LiHalfStep n) × BitVec n)) :
    liCheckpoint (l₁ ++ l₂) = liCheckpoint l₁ ^^^ liCheckpoint l₂ := by
  unfold liCheckpoint
  rw [List.foldl_append, foldl_xor _ l₂]

/-- The warp's XOR of its lanes' checkpoints is the checkpoint of all its chains. -/
theorem redux_eq {n : ℕ} (ls : List (List (List (LiHalfStep n) × BitVec n))) :
    (ls.map liCheckpoint).foldl (· ^^^ ·) 0 = liCheckpoint ls.flatten := by
  induction ls with
  | nil => rfl
  | cons l ls ih =>
    rw [List.map_cons, List.foldl_cons, foldl_xor_self, ih, List.flatten_cons, liCheckpoint_append]
    exact congrArg (· ^^^ liCheckpoint ls.flatten) BitVec.zero_xor

/-- A segment's warp checkpoint from the chain states `y`, absorbing the words `w`. -/
def segment {n W C H : ℕ} (k : Fin W × Fin C × Fin H → BitVec n) (a : Fin W × Fin C × Fin H → OddKey n)
    (y : Fin W → Fin C → BitVec n) (w : LiPlace W C H → BitVec n) : BitVec n :=
  (List.ofFn fun ℓ => liCheckpoint (List.ofFn fun c =>
    ((List.ofFn fun j => (k (ℓ, c, j), (a (ℓ, c, j)).1, (w (ℓ, c, j, 0), w (ℓ, c, j, 1)))), y ℓ c))).foldl
    (· ^^^ ·) 0

/-- **A segment's warp checkpoint is injective in each word**: the honest words with the word at `p` replaced by `v`
keep the checkpoint only if `v` is the honest word. -/
theorem segment_update {n W C H : ℕ} (k : Fin W × Fin C × Fin H → BitVec n) (a : Fin W × Fin C × Fin H → OddKey n)
    (y : Fin W → Fin C → BitVec n) (w : LiPlace W C H → BitVec n) (p : LiPlace W C H) (v : BitVec n)
    (h : segment k a y (Function.update w p v) = segment k a y w) : v = w p := by
  obtain ⟨ℓ₀, c₀, j₀, i₀⟩ := p
  set w' := Function.update w (ℓ₀, c₀, j₀, i₀) v with hw'
  have hoff : ∀ q, q ≠ (ℓ₀, c₀, j₀, i₀) → w' q = w q := fun q hq => Function.update_of_ne hq _ _
  -- the half steps of a chain, as odd half steps
  let odd (x : LiPlace W C H → BitVec n) (ℓ : Fin W) (c : Fin C) : Fin H → LiOddHalfStep n :=
    fun j => (k (ℓ, c, j), a (ℓ, c, j), (x (ℓ, c, j, 0), x (ℓ, c, j, 1)))
  let chain (x : LiPlace W C H → BitVec n) (ℓ : Fin W) (c : Fin C) : List (LiHalfStep n) × BitVec n :=
    ((List.ofFn (odd x ℓ c)).map LiOddHalfStep.val, y ℓ c)
  have hchain : ∀ x : LiPlace W C H → BitVec n, segment k a y x =
      liCheckpoint (List.ofFn fun ℓ => List.ofFn fun c => chain x ℓ c).flatten := by
    intro x
    rw [← redux_eq, List.map_ofFn]
    unfold segment
    congr 2
    funext ℓ
    simp only [Function.comp, chain, List.map_ofFn, odd]
    rfl
  rw [hchain, hchain] at h
  -- the lanes but `ℓ₀`, the chains of `ℓ₀` but `c₀`, the half steps of chain `c₀` but `j₀` agree
  obtain ⟨B, A, hF, hG⟩ := ofFn_split (fun ℓ => List.ofFn fun c => chain w ℓ c)
    (fun ℓ => List.ofFn fun c => chain w' ℓ c) ℓ₀ fun ℓ hℓ => by
      have e : ∀ c j i, w' (ℓ, c, j, i) = w (ℓ, c, j, i) := fun c j i => hoff _ (by simp [hℓ])
      simp only [chain, odd, e]
  obtain ⟨Bc, Ac, hFc, hGc⟩ := ofFn_split (chain w ℓ₀) (chain w' ℓ₀) c₀ fun c hc => by
      have e : ∀ j i, w' (ℓ₀, c, j, i) = w (ℓ₀, c, j, i) := fun j i => hoff _ (by simp [hc])
      simp only [chain, odd, e]
  obtain ⟨Bs, As, hFs, hGs⟩ := ofFn_split (odd w ℓ₀ c₀) (odd w' ℓ₀ c₀) j₀ fun j hj => by
      have e : ∀ i, w' (ℓ₀, c₀, j, i) = w (ℓ₀, c₀, j, i) := fun i => hoff _ (by simp [hj])
      simp only [odd, e]
  rw [hF, hFc, hG, hGc] at h
  simp only [chain, hFs, hGs, List.map_append, List.map_cons, List.flatten_append, List.flatten_cons,
    List.append_assoc, List.cons_append] at h
  have e := Pouw.SecurityProofs.Receipt.LiCheckpointWordInjective n (B.flatten ++ Bc) (Ac ++ A.flatten)
    (Bs.map LiOddHalfStep.val) (k (ℓ₀, c₀, j₀)) (a (ℓ₀, c₀, j₀)) (w (ℓ₀, c₀, j₀, 0), w (ℓ₀, c₀, j₀, 1)) As
    (y ℓ₀ c₀)
  simp only [List.append_assoc] at e
  fin_cases i₀
  · have hv0 : w' (ℓ₀, c₀, j₀, 0) = v := Function.update_self _ _ _
    have hv1 : w' (ℓ₀, c₀, j₀, 1) = w (ℓ₀, c₀, j₀, 1) := hoff _ (by simp)
    simp only [odd, LiOddHalfStep.val, hv0, hv1] at h
    exact e.1 h
  · have hv0 : w' (ℓ₀, c₀, j₀, 0) = w (ℓ₀, c₀, j₀, 0) := hoff _ (by simp)
    have hv1 : w' (ℓ₀, c₀, j₀, 1) = v := Function.update_self _ _ _
    simp only [odd, LiOddHalfStep.val, hv0, hv1] at h
    exact e.2 h

/-- The chains' states at segment `s` read only the words of the segments before it. -/
theorem state_congr {Ω U : Type} {n S W C H : ℕ} (D : LiUnits Ω U n S W C H) (ω : Ω) (u : U)
    (w w' : ℕ → LiPlace W C H → BitVec n) :
    ∀ s, (∀ t < s, w' t = w t) → D.state ω u w' s = D.state ω u w s
  | 0, _ => rfl
  | s + 1, h => by
    funext ℓ c
    simp only [LiUnits.state]
    rw [h s (Nat.lt_succ_self s), state_congr D ω u w w' s fun t ht => h t (Nat.lt_succ_of_lt ht)]

/-- **Lemma B for li units**: every segment's checkpoint the honest one gives every segment's deficient word right. -/
theorem right_of_same {Ω U : Type} {n S W C H : ℕ} (D : LiUnits Ω U n S W C H) (ω : Ω) (h : D.same ω) :
    D.right ω := by
  intro u
  have key : ∀ s, s ≤ S → ∀ t < s, D.value ω u t = D.honest ω u t (D.pos ω u t) := by
    intro s
    induction s with
    | zero => intro _ t ht; omega
    | succ s ih =>
      intro hs t ht
      have ih' := ih (by omega)
      rcases Nat.lt_succ_iff_lt_or_eq.mp ht with hlt | rfl
      · exact ih' t hlt
      · have hw : ∀ t' < t, D.word ω u t' = D.honest ω u t' := fun t' ht' => by
          unfold LiUnits.word
          rw [ih' t' ht']
          exact Function.update_eq_self _ _
        have hy := state_congr D ω u (D.honest ω u) (D.word ω u) t hw
        have hc := h u t (by omega)
        unfold LiUnits.checkpoint at hc
        rw [hy] at hc
        exact segment_update (D.k ω u t) (D.a ω u t) (D.state ω u (D.honest ω u) t) (D.honest ω u t)
          (D.pos ω u t) (D.value ω u t) hc
  exact fun s hs => key S le_rfl s hs

theorem xor_top_mul_odd {n : ℕ} (x : BitVec (n + 1)) {a : BitVec (n + 1)} (ha : a.toNat % 2 = 1) :
    (x ^^^ BitVec.twoPow (n + 1) n) * a = x * a ^^^ BitVec.twoPow (n + 1) n := by
  rw [← add_twoPow_top, ← add_twoPow_top, add_mul, twoPow_mul_odd ha]

/-- A flip of `d₀`'s top bit moves a half step with an odd multiplier by the top bit. -/
theorem liHalf_top₀ {n : ℕ} (k : BitVec (n + 1)) {a : BitVec (n + 1)} (ha : a.toNat % 2 = 1)
    (y d₀ d₁ : BitVec (n + 1)) :
    liHalf k a y (d₀ ^^^ BitVec.twoPow (n + 1) n, d₁) = liHalf k a y (d₀, d₁) ^^^ BitVec.twoPow (n + 1) n := by
  simp only [liHalf]
  have e : y ^^^ (d₀ ^^^ BitVec.twoPow (n + 1) n) ^^^ k = (y ^^^ d₀ ^^^ k) ^^^ BitVec.twoPow (n + 1) n := by
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simp only [BitVec.getLsbD_xor]
    cases y.getLsbD i <;> cases d₀.getLsbD i <;> cases k.getLsbD i <;>
      cases (BitVec.twoPow (n + 1) n).getLsbD i <;> rfl
  rw [e, xor_top_mul_odd _ ha, ← add_twoPow_top, ← add_twoPow_top]
  ring

/-- A state moved by the top bit and a flip of `d₀`'s top bit cancel in a half step. -/
theorem liHalf_top_cancel {n : ℕ} (k a z e₀ e₁ : BitVec (n + 1)) :
    liHalf k a (z ^^^ BitVec.twoPow (n + 1) n) (e₀ ^^^ BitVec.twoPow (n + 1) n, e₁) = liHalf k a z (e₀, e₁) := by
  simp only [liHalf]
  congr 2
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases z.getLsbD i <;> cases e₀.getLsbD i <;> cases k.getLsbD i <;>
    cases (BitVec.twoPow (n + 1) n).getLsbD i <;> rfl

theorem twoPow_top_ne_zero (n : ℕ) : BitVec.twoPow (n + 1) n ≠ 0 := by
  intro h
  have h1 := congrArg BitVec.toNat h
  rw [twoPow_top_toNat] at h1
  simp at h1

theorem xor_top_ne {n : ℕ} (x : BitVec (n + 1)) : x ^^^ BitVec.twoPow (n + 1) n ≠ x := by
  intro h
  apply twoPow_top_ne_zero n
  calc BitVec.twoPow (n + 1) n = x ^^^ (x ^^^ BitVec.twoPow (n + 1) n) := (xor_xor_self_left _ _).symm
    _ = 0 := by rw [h, BitVec.xor_self]; rfl

/-- One chain, one half step a segment: the checkpoint after segment `s` is the half step from the state. -/
theorem checkpoint_one {Ω U : Type} {n S : ℕ} (D : LiUnits Ω U n S 1 1 1) (ω : Ω) (u : U)
    (w : ℕ → LiPlace 1 1 1 → BitVec n) (s : ℕ) :
    D.checkpoint ω u w s = D.state ω u w (s + 1) 0 0 := by
  simp [LiUnits.checkpoint, LiUnits.state, LiUnits.steps, liCheckpoint, liChain, List.ofFn_succ]

theorem state_one {Ω U : Type} {n S : ℕ} (D : LiUnits Ω U n S 1 1 1) (ω : Ω) (u : U)
    (w : ℕ → LiPlace 1 1 1 → BitVec n) (s : ℕ) :
    D.state ω u w (s + 1) 0 0 =
      liHalf (D.k ω u s (0, 0, 0)) (D.a ω u s (0, 0, 0)).1 (D.state ω u w s 0 0) (w s (0, 0, 0, 0), w s (0, 0, 0, 1)) := by
  simp [LiUnits.state, LiUnits.steps, liChain, List.ofFn_succ]

/-- **One receipt word a window does not bind li's words.** -/
theorem windowUnbound {Ω U : Type} (n : ℕ) (D : LiUnits Ω U (n + 1) 2 1 1 1) (ω : Ω)
    (h : ∀ u, ∀ s < 2, D.pos ω u s = 0 ∧ D.value ω u s = D.honest ω u s 0 ^^^ BitVec.twoPow (n + 1) n) :
    D.sameWindow ω ∧ ∀ u, (∀ s < 2, D.value ω u s ≠ D.honest ω u s (D.pos ω u s)) ∧
      D.checkpoint ω u (D.word ω u) 0 ≠ D.checkpoint ω u (D.honest ω u) 0 := by
  have hw : ∀ u, ∀ s < 2, D.word ω u s (0, 0, 0, 0) = D.honest ω u s (0, 0, 0, 0) ^^^ BitVec.twoPow (n + 1) n ∧
      D.word ω u s (0, 0, 0, 1) = D.honest ω u s (0, 0, 0, 1) := by
    intro u s hs
    obtain ⟨hp, hv⟩ := h u s hs
    unfold LiUnits.word
    rw [hp]
    refine ⟨?_, Function.update_of_ne (by decide) _ _⟩
    rw [show ((0, 0, 0, 0) : LiPlace 1 1 1) = 0 from rfl, Function.update_self, hv]
  have h1 : ∀ u, D.state ω u (D.word ω u) 1 0 0 = D.state ω u (D.honest ω u) 1 0 0 ^^^ BitVec.twoPow (n + 1) n := by
    intro u
    rw [state_one D ω u (D.word ω u) 0, state_one D ω u (D.honest ω u) 0, (hw u 0 (by omega)).1,
      (hw u 0 (by omega)).2]
    exact liHalf_top₀ _ (D.a ω u 0 (0, 0, 0)).2 _ _ _
  refine ⟨fun u => ?_, fun u => ⟨fun s hs => ?_, ?_⟩⟩
  · show D.checkpoint ω u (D.word ω u) 1 = D.checkpoint ω u (D.honest ω u) 1
    rw [checkpoint_one D ω u (D.word ω u) 1, checkpoint_one D ω u (D.honest ω u) 1, state_one D ω u (D.word ω u) 1,
      state_one D ω u (D.honest ω u) 1, h1 u, (hw u 1 (by omega)).1, (hw u 1 (by omega)).2]
    exact liHalf_top_cancel _ _ _ _ _
  · rw [(h u s hs).2, (h u s hs).1]
    exact xor_top_ne _
  · rw [checkpoint_one D ω u (D.word ω u) 0, checkpoint_one D ω u (D.honest ω u) 0, h1 u]
    exact xor_top_ne _

end Accounting.SecurityProofs.TS.LiLemmas
