import Proofs.Flock.Soundness.ExecPlacement
-- level3 already realizes `Lookup.row`'s and `Lookup.decode`'s equation lemmas, so this file reuses them rather than
-- declaring them a second time, which the audit's kernel replay refuses
import Proofs.Flock.Level3.LookupRows

/-!
# What `Lookup.build` builds

A lookup slot's expanded circuit (`Flock.Lookup.build`): one input word whose port is the `n` index bits, the constant
the last row, reading itself in `A` and in `B`, and the product rows (whose `B` side the fold reads from the table)
before it.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## The builder's loops push rows, to `A` and `B` together -/

theorem foldl_count {α σ : Type} (m : σ → ℕ) {g : σ → α → σ} {k : ℕ} (hg : ∀ s x, m (g s x) = m s + k) :
    ∀ (l : List α) (s : σ), m (l.foldl g s) = m s + l.length * k := by
  intro l
  induction l with
  | nil => intro s; simp
  | cons x l ih => intro s; rw [List.foldl_cons, ih, hg, List.length_cons]; ring

theorem foldl_inv {α σ : Type} (P : σ → Prop) {g : σ → α → σ} (hg : ∀ s x, P s → P (g s x)) :
    ∀ {l : List α} {s t : σ}, P s → l.foldl g s = t → P t := by
  intro l
  induction l with
  | nil => intro s t hs h; cases h; exact hs
  | cons x l ih => intro s t hs h; exact ih (hg s x hs) h

/-- A `for i in [0:k]` loop whose body adds one to `m`, run in `V`, adds `k`. -/
theorem forIn_range_pure {σ : Type} (m : σ → ℕ) {k : ℕ} {s s' : σ} {g : ℕ → σ → σ}
    (hg : ∀ i r, m (g i r) = m r + 1)
    (h : (forIn [:k] s fun i r => pure (ForInStep.yield (g i r)) : V σ) = .ok s') : m s' = m s + k := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl] at h
  cases h
  rw [foldl_count m (fun r i => hg i r)]
  simp [Std.Legacy.Range.size]

/-- Two nested range loops whose inner body adds one to `m` add `k · j`. -/
theorem forIn_range_nested {σ : Type} (m : σ → ℕ) {k j : ℕ} {s s' : σ} {g : ℕ → σ → σ}
    (hg : ∀ i r, m (g i r) = m r + 1)
    (h : (forIn [:k] s fun _ r => do
      let r ← forIn [:j] r fun i r => pure (ForInStep.yield (g i r))
      pure (ForInStep.yield r) : V σ) = .ok s') : m s' = m s + k * j := by
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind] at h
  cases h
  rw [foldl_count m (k := j) (fun r i => by rw [foldl_count m (fun r i => hg i r)]; simp [Std.Legacy.Range.size])]
  simp [Std.Legacy.Range.size]

/-- **A `while` loop's invariant**: `P` holds at the end of a loop that returned, if every step keeps it and every
`yield` lowers the measure `μ`. -/
private theorem loop_inv {σ ε : Type} (P : σ → Prop) (μ : σ → ℕ) {f : Unit → σ → Except ε (ForInStep σ)}
    (hf : ∀ s v, P s → f () s = .ok v → P v.value ∧ ∀ s', v = .yield s' → μ s' < μ s) :
    ∀ (s r : σ), P s → Lean.Loop.forIn Lean.Loop.mk s f = .ok r → P r := by
  intro s
  induction hμ : μ s using Nat.strong_induction_on generalizing s with
  | _ N ih =>
    intro r hs h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    rcases hv : f () s with e | v
    · rw [hv] at h; cases h
    · rw [hv] at h
      obtain ⟨hP, hμ'⟩ := hf s v hs hv
      cases v with
      | done val => cases h; exact hP
      | yield val => exact ih (μ val) (hμ ▸ hμ' val rfl) val rfl r hP h

/-- `Lookup.build`'s padding to a word only pushes rows, to `A` and `B` together. -/
theorem while_pad {X : ℕ} {s s' : Lookup.Builder} (hs : X ≤ s.a.size ∧ s.a.size = s.b.size)
    (h : (forIn Lean.Loop.mk s fun _ r =>
      if (r.a.size % 128 != 0) = true then
        pure (ForInStep.yield ({ a := r.a.push #[], b := r.b.push #[] } : Lookup.Builder))
      else pure (ForInStep.done r) : V Lookup.Builder) = .ok s') :
    X ≤ s'.a.size ∧ s'.a.size = s'.b.size := by
  refine loop_inv (fun r : Lookup.Builder => X ≤ r.a.size ∧ r.a.size = r.b.size)
    (fun r => (128 - r.a.size % 128) % 128) (fun r v hr hv => ?_) s s' hs h
  by_cases hc : (r.a.size % 128 != 0) = true
  · rw [ite_eq_left hc] at hv
    cases hv
    simp only [bne_iff_ne, ne_eq] at hc
    refine ⟨?_, ?_⟩
    · simp only [ForInStep.value, Array.size_push]; omega
    · intro s' hs'
      cases hs'
      simp only [Array.size_push]
      omega
  · rw [ite_eq_right hc] at hv
    cases hv
    exact ⟨hr, fun s' hs' => by cases hs'⟩

/-- `Lookup.decode` pushes its rows to `A` and `B` together. -/
theorem decode_sizes (fuel : ℕ) : ∀ (bd : Lookup.Builder) (bits : List ℕ), bd.a.size = bd.b.size →
    (Lookup.decode bd bits fuel).1.a.size = (Lookup.decode bd bits fuel).1.b.size := by
  induction fuel with
  | zero => intro bd bits h; unfold Lookup.decode; split <;> first | exact h | omega
  | succ fuel ih =>
    intro bd bits h
    unfold Lookup.decode
    split
    · exact h
    · exact h
    · rename_i heq _
      obtain rfl := Nat.succ.inj heq
      dsimp only
      exact foldl_inv (fun s : Lookup.Builder × Array Lookup.Form => s.1.a.size = s.1.b.size)
        (fun s x hs => by simp only [Lookup.row, Array.size_push]; omega) (ih _ _ (ih _ _ h)) rfl

/-! ## The constant row -/

theorem ite_throw_ok_not {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬ c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rw [ite_eq_right hc] at h; exact ⟨hc, h⟩

/-- The row pushed at `k = rs.size`, with `KONST` resolved to `k`, reads `k`. -/
theorem fix_push_getD {rs : Array (Array ℕ)} {k : ℕ} (hk : rs.size = k) :
    (Array.map (fun r => Array.map (fun c => if (c == Lookup.KONST) = true then k else c) r) (rs.push #[k])).getD
      k #[] = #[k] := by
  subst hk
  simp [Array.getD_eq_getD_getElem?, Array.getElem_push]

theorem count_singleton_zmod (j k : ℕ) : (((#[k] : Array ℕ).toList.count j : ℕ) : ZMod 2) = if j = k then 1 else 0 := by
  by_cases hj : j = k
  · subst hj; simp
  · simp [hj, Ne.symm hj]

/-- **What `Lookup.build` builds**: a lookup slot, one input word, its constant the last useful row, reading itself. -/
theorem build_facts {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build name table n lo bits = .ok net) :
    net.lookup.isSome = true ∧ net.inGroups = #[⟨0, 1, #[n]⟩] ∧ net.inWords = 1 ∧ n ≤ 128 ∧
      net.useful = net.constPos + 1 ∧ net.unitLog = max 7 (log2ceil net.useful) ∧
      (net.unitLog ≤ 32 → ConstRow net) := by
  unfold Lookup.build at h
  obtain ⟨hc, h⟩ := ite_throw_ok_not h
  have hn : n ≤ 128 := by simp at hc; exact hc.2
  obtain ⟨s1, hs1, h⟩ := except_bind_ok h
  dsimp only at h
  rcases hd1 : Lookup.decode s1 (List.range lo) lo with ⟨bd1, loM, loTop⟩
  rw [hd1] at h
  rcases hd2 : Lookup.decode bd1 ((List.range (n - lo)).map (lo + ·)) (n - lo) with ⟨bd2, hiM, x⟩
  rw [hd2] at h
  dsimp only at h
  obtain ⟨s2, hs2, h⟩ := except_bind_ok h
  obtain ⟨s3, hs3, h⟩ := except_bind_ok h
  obtain ⟨s4, hs4, h⟩ := except_bind_ok h
  obtain ⟨hc2, h⟩ := ite_throw_ok_not h
  have hnh : hiM.size = 2 ^ (n - lo) := by simp at hc2; exact hc2.2
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  have e1a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs1
  have e1b := forIn_range_pure (fun r : Lookup.Builder => r.b.size) (fun i r => by split <;> simp) hs1
  simp only [Array.size_empty] at e1a e1b
  have hb1 : bd1.a.size = bd1.b.size := by
    have := decode_sizes lo s1 (List.range lo) (by omega); rwa [hd1] at this
  have hb2 : bd2.a.size = bd2.b.size := by
    have := decode_sizes (n - lo) bd1 ((List.range (n - lo)).map (lo + ·)) hb1; rwa [hd2] at this
  have e2a := forIn_range_nested (fun r : Lookup.Builder => r.a.size) (fun i r => by simp) hs2
  have e2b := forIn_range_nested (fun r : Lookup.Builder => r.b.size) (fun i r => by simp) hs2
  have e3 := while_pad (X := s2.a.size) ⟨le_refl _, by omega⟩ hs3
  have e4a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs4
  have e4b := forIn_range_pure (fun r : Lookup.Builder => r.b.size) (fun i r => by split <;> simp) hs4
  refine ⟨rfl, rfl, rfl, hn, Array.size_push _, rfl, fun hu => ⟨?_, fun j => ?_, fun j => ?_⟩⟩
  · show 1 * 128 ≤ s4.a.size
    omega
  · rw [sparseEntry_ofRows ha, ofRows_row ha (Nat.pow_le_pow_right (by norm_num) hu), fix_push_getD rfl,
      count_singleton_zmod]
  · rw [sparseEntry_ofRows hb, ofRows_row hb (Nat.pow_le_pow_right (by norm_num) hu)]
    have hlt : ¬ s4.a.size < bd2.a.size + bits * 2 ^ (n - lo) := by rw [← hnh]; omega
    simp only [lookupEntry, hlt, false_and, and_false, ↓reduceIte, add_zero]
    rw [fix_push_getD (by omega), count_singleton_zmod]

end FlockSoundness
