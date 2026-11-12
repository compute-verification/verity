module

public import Proofs.Properties.Nci.LeafWidthsPart
public import Mathlib.Tactic

/-!
# A body the width check passes reads every leaf at its width

`bodyBad_sound`: where `Widths.bodyBad` finds nothing to refuse in a body, the body is `BodyWide`: every reference of
each node's arguments and of the return names a leaf as wide as the leaf of the type it is read as. `seqBad_sound` is
the same of one sequence of parts, each part checked from the leaf it starts at (`partOk_sound`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-! ## A sequence's parts -/

/-- **Reference `starts[i] + x` of a sequence** is part `i`'s element `x`, for `x` within the part. -/
theorem get_at (ps : Array Part) {i x : ℕ} (hi : i < ps.size) (hx : x < ps[i]!.count) :
    (Seq.ofParts ps).starts[i]! + ps[i]!.count ≤ (Seq.ofParts ps).count ∧
      (Seq.ofParts ps).get ((Seq.ofParts ps).starts[i]! + x) = ps[i]!.get x := by
  obtain ⟨-, -, hst, hcnt⟩ := Walk.ofParts_spec ps
  have hst' : ∀ i < ps.size, (Seq.ofParts ps).starts[i]! + ps[i]!.count =
      ((ps.toList.take (i + 1)).map Part.count).sum := fun i hi => by rw [hst i hi, Walk.sum_take_succ_map ps hi]
  have hmono : ∀ a b, a ≤ b →
      ((ps.toList.take a).map Part.count).sum ≤ ((ps.toList.take b).map Part.count).sum := by
    intro a b hab
    obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le hab
    rw [List.take_add, List.map_append, List.sum_append]
    omega
  have hle : ((ps.toList.take (i + 1)).map Part.count).sum ≤ (Seq.ofParts ps).count := by
    rw [hcnt]
    have := hmono (i + 1) ps.toList.length (by simp only [Array.length_toList]; omega)
    rwa [List.take_length] at this
  have hlt : (Seq.ofParts ps).starts[i]! + x < (Seq.ofParts ps).count := by
    have := hst' i hi
    omega
  refine ⟨by have := hst' i hi; omega, ?_⟩
  obtain ⟨i', hi', h1, h2, hget⟩ := Walk.seq_get ps hlt
  have e : i' = i := by
    rcases Nat.lt_trichotomy i' i with h | h | h
    · have := hst' i' hi'
      have := hmono (i' + 1) i (by omega)
      have := hst i hi
      omega
    · exact h
    · have := hst' i hi
      have := hmono (i + 1) i' (by omega)
      have := hst i' hi'
      omega
  subst e
  rw [hget, Nat.add_sub_cancel_left]

/-! ## A loop over an array, index by index -/

/-- An invariant of a loop of the check's monad over an array, by index: `J i` holds of the state before element `i`
is read; the loop ends with `J` at the array's size, or early with `D`. -/
theorem forIn_m_idx {α β : Type} [Inhabited α] (f : α → β → Widths.M (ForInStep β)) (J : ℕ → β → Prop)
    (D : β → Prop) (xs : Array α)
    (hstep : ∀ i < xs.size, ∀ b s t s', J i b → (f xs[i]! b).run s = .ok t s' → match t with
      | .yield b' => J (i + 1) b'
      | .done b' => D b') :
    ∀ (k i : ℕ) (b r : β) (s s' : ℕ), i + k = xs.size → J i b →
      (forIn (xs.toList.drop i) b f).run s = .ok r s' → J xs.size r ∨ D r := by
  intro k
  induction k with
  | zero =>
    intro i b r s s' hik hJ h
    rw [List.drop_of_length_le (by rw [Array.length_toList]; omega)] at h
    obtain ⟨rfl, -⟩ := m_pure h
    have e : i = xs.size := by omega
    rw [← e]
    exact Or.inl hJ
  | succ k ih =>
    intro i b r s s' hik hJ h
    have hi : i < xs.size := by omega
    rw [List.drop_eq_getElem_cons (by rw [Array.length_toList]; exact hi), List.forIn_cons] at h
    obtain ⟨t, s₁, hf, h⟩ := m_bind h
    rw [Array.getElem_toList, ← getElem!_pos xs i hi] at hf
    have hs := hstep i hi b s t s₁ hJ hf
    cases t with
    | done b' =>
      obtain ⟨rfl, -⟩ := m_pure h
      exact Or.inr hs
    | yield b' => exact ih (i + 1) b' r s₁ s' (by omega) hs h

/-! ## A sequence -/

/-- **A sequence the width check passes names leaves of the widths its type gives them.** -/
theorem seqBad_sound (B : Body) (below : ℕ) (t : Ty) (ps : Array Part) (w w' : ℕ)
    (hq : ∀ q ∈ ps, BodiesOk.PartNonneg q) (hi : ∀ q ∈ ps, PartInner q) (hc : (Seq.ofParts ps).count = t.leaves)
    (hin : ∀ j < (Seq.ofParts ps).count, BodiesOk.RefIn B ((Seq.ofParts ps).get j))
    (h : (Widths.seqBad (srcOf B below) t (Seq.ofParts ps)).run w = .ok none w') :
    SeqWide (srcOf B below) t (Seq.ofParts ps) := by
  obtain ⟨hparts, hsz, -, -⟩ := Walk.ofParts_spec ps
  obtain ⟨hsa, hss, hse⟩ := BodiesOk.toStream_array (Seq.ofParts ps).starts
  unfold Widths.seqBad at h
  dsimp only at h
  rw [hparts] at h
  obtain ⟨r, w₂, hloop, h⟩ := m_bind h
  rw [← Array.forIn_toList] at hloop
  have hL := forIn_m_idx _
    (fun i (b : Option (Option (ℕ × Part)) × Subarray ℕ) => b.1 = none ∧ b.2.array = (Seq.ofParts ps).starts ∧
      b.2.start = i ∧ b.2.stop = (Seq.ofParts ps).starts.size ∧
      ∀ x < i, ∃ s₁ s₂, (Widths.partOk (srcOf B below) t (Widths.homo t) ((Seq.ofParts ps).starts[x]!)
        ps[x]!).run s₁ = .ok true s₂)
    (fun b => ∃ y, b.1 = some (some y)) ps ?step ps.size 0 _ r w w₂ (by omega)
    ⟨rfl, hsa, hss, hse, fun x hx => absurd hx (by omega)⟩ (by rw [List.drop_zero]; exact hloop)
  case step =>
    rintro i hi' ⟨b1, strm⟩ s st s' ⟨-, h1, h2, h3, h4⟩ hf
    dsimp only at h1 h2 h3 hf
    obtain ⟨strm', hnext, g1, g2, g3⟩ := BodiesOk.next_at strm (Seq.ofParts ps).starts i h1 h2 (by omega)
    rw [hnext] at hf
    dsimp only at hf
    obtain ⟨ok, s₁, hok, hf⟩ := m_bind hf
    cases ok with
    | false =>
      rw [ite_eq_left (by decide)] at hf
      obtain ⟨rfl, -⟩ := m_pure hf
      exact ⟨_, rfl⟩
    | true =>
      rw [ite_eq_right (by decide)] at hf
      obtain ⟨rfl, -⟩ := m_pure hf
      refine ⟨rfl, g1, g2, by rw [g3, h3], fun x hx => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hx with hx' | rfl
      · exact h4 x hx'
      · exact ⟨s, s₁, hok⟩
  obtain ⟨r1, r2⟩ := r
  dsimp only at h hL
  rcases hL with ⟨rfl, -, -, -, hall⟩ | ⟨y, rfl⟩
  · intro j hj S hS
    obtain ⟨i, hi', h1, h2, hget⟩ := Walk.seq_get ps hj
    obtain ⟨s₁, s₂, hpart⟩ := hall i hi'
    have hmem : ps[i]! ∈ ps := by
      rw [getElem!_pos ps i hi']
      exact Array.getElem_mem hi'
    have hx : j - (Seq.ofParts ps).starts[i]! < ps[i]!.count := by omega
    have key := partOk_sound B below t ((Seq.ofParts ps).starts[i]!) ps[i]! s₁ s₂ (hq _ hmem) (hi _ hmem)
      (by have := (get_at ps hi' hx).1; omega)
      (fun k hk => by
        have h3 := get_at ps hi' hk
        rw [← h3.2]
        exact hin _ (by have := h3.1; omega))
      hpart (j - (Seq.ofParts ps).starts[i]!) hx S (by rw [← hget]; exact hS)
    rw [hget, key]
    congr 1
    omega
  · exact absurd (m_pure h).1 (by simp)

/-! ## A body -/

/-- **A body the width check passes reads every leaf at its width.** -/
theorem bodyBad_sound (defs defs' : Array Defn) (i : ℕ) (key : String) (B : Body) (w w' : ℕ)
    (hB : BodiesOk.BodyFacts defs i B) (hI : BodyInner B) (hret : B.ret.count = B.retType.leaves)
    (h : (Widths.bodyBad defs' key B).run w = .ok none w') : BodyWide B := by
  unfold Widths.bodyBad at h
  dsimp only at h
  obtain ⟨r, w₂, hloop, h⟩ := m_bind h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  have hL := forIn_m_inv _
    (fun k : ℕ => ∀ a < (B.nodes[k]!).args.size, ∃ s₁ s₂,
      (Widths.seqBad (srcOf B k) ((B.nodes[k]!).args[a]!).ty ((B.nodes[k]!).args[a]!).seq).run s₁ = .ok none s₂)
    (fun b : Option (Option String) × Unit => b.1 = none) (fun b => ∃ y, b.1 = some (some y)) _ (none, ()) r w w₂
    rfl ?step hloop
  case step =>
    rintro k - bb s st s' - hf
    obtain ⟨r', s₁, hin, hf⟩ := m_bind hf
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hin
    have hL' := forIn_m_inv _
      (fun a : ℕ => ∃ s₁ s₂,
        (Widths.seqBad (srcOf B k) ((B.nodes[k]!).args[a]!).ty ((B.nodes[k]!).args[a]!).seq).run s₁ = .ok none s₂)
      (fun b : Option (Option String) × Unit => b.1 = none) (fun b => ∃ y, b.1 = some (some y)) _ (none, ()) r' s s₁
      rfl ?inner hin
    case inner =>
      rintro a - bb' u st' u' - hg
      obtain ⟨res, u₁, hres, hg⟩ := m_bind hg
      cases res with
      | none =>
        obtain ⟨rfl, -⟩ := m_pure hg
        exact ⟨⟨u, u₁, hres⟩, rfl⟩
      | some pq =>
        obtain ⟨rfl, -⟩ := m_pure hg
        exact ⟨_, rfl⟩
    obtain ⟨r1, r2⟩ := r'
    dsimp only at hf hL'
    rcases hL' with ⟨hall, rfl⟩ | ⟨y, rfl⟩
    · obtain ⟨rfl, -⟩ := m_pure hf
      exact ⟨fun a ha => hall a (by simp [List.mem_range'_1, Std.Legacy.Range.size]; omega), rfl⟩
    · obtain ⟨rfl, -⟩ := m_pure hf
      exact ⟨_, rfl⟩
  obtain ⟨r1, r2⟩ := r
  dsimp only at h hL
  rcases hL with ⟨hall, rfl⟩ | ⟨y, rfl⟩
  · dsimp only at h
    obtain ⟨res, w₃, hres, h⟩ := m_bind h
    cases res with
    | some pq => exact absurd (m_pure h).1 (by simp)
    | none =>
      obtain ⟨hnodes, hrefs, ps, hps, hnn, -⟩ := hB
      refine ⟨fun k hk a ha => ?_, ?_⟩
      · obtain ⟨-, hnode, hargs⟩ := hnodes k hk
        obtain ⟨⟨qs, hqs, hqn⟩, hqr⟩ := hargs a ha
        obtain ⟨-, hcnt, -⟩ := hnode
        obtain ⟨s₁, s₂, hsb⟩ := hall k (by simp [List.mem_range'_1, Std.Legacy.Range.size]; omega) a ha
        have hinner := hI.1 k hk a ha
        have hc := hcnt a ha
        rw [hqs] at hsb hinner hc hqr ⊢
        rw [(Walk.ofParts_spec qs).1] at hinner
        exact seqBad_sound B k _ qs s₁ s₂ hqn hinner hc hqr hsb
      · have hinner := hI.2
        rw [hps] at hres hinner hret hrefs ⊢
        rw [(Walk.ofParts_spec ps).1] at hinner
        exact seqBad_sound B B.nodes.size _ ps w₂ w₃ hnn hinner hret hrefs hres
  · exact absurd (m_pure h).1 (by simp)

end NciCheck.Properties.SecurityProofs.LeafWidths
