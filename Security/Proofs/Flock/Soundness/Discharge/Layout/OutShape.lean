import Proofs.Flock.Soundness.Discharge.Layout.OutRegion

set_option Elab.async false

/-!
# The returned outputs' shape, from the parse

`keyProg_hout` reads `OutShape` off the parse: the unit's returned-output group's words are a power of two, and each copy's leaf's
outputs are 16-bit ports, output words below `nOut`, that fit in the group's words. The verifier checks all of it, in an
untyped circuit whose unit returns the outputs (`outShape_of_parse`):

* `built_out`: `Net.ofRows` (`Built`) has a first output group, each of an aligned power of two of words whose ports'
  bits fit in them;
* `check_out`: `HmRow.check` has the unit's returned outputs 16-bit ports, and each copy's leaf's outputs one per port,
  below `nOut`.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3

/-- **The returned outputs' shape**: the output group's words are a power of two, and each copy's leaf's outputs are
16-bit ports, output words below `nOut`, that fit in the group's words. -/
def OutShape (c : Flock.Circuit) : Prop :=
  (c.unit.outGroups.getD 0 default).words = 2 ^ (c.unit.outGroups.getD 0 default).words.log2 ∧
  ∀ u' < c.upv, (∀ j < (c.leavesOut.getD u' #[]).size,
      outBits c j = 16 ∧ (c.leavesOut.getD u' #[])[j]! < c.outWords) ∧
    16 * (c.leavesOut.getD u' #[]).size ≤ 128 * (c.unit.outGroups.getD 0 default).words

/-- `ite_throw_ok`, keeping the condition that did not throw. -/
theorem ite_throw_cond {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rw [ite_eq_right hc] at h; exact ⟨hc, h⟩

/-- **What `Net.ofRows` checks of the output groups**: there is one, and each is a power of two of words whose ports'
bits fit in them. -/
theorem built_out {n : Net} (hU : Built n) :
    n.outGroups.size ≠ 0 ∧ ∀ g ∈ n.outGroups.toList, isPow2 g.words = true ∧ g.bits.foldl (· + ·) 0 ≤ g.words * WORD := by
  obtain ⟨name, useful, constPos, ig, og, ra, rb, h⟩ := hU
  have hog := (Net.ofRows_ok h).2.2.2.2.2.2.1
  subst hog
  unfold Net.ofRows at h
  simp only at h
  obtain ⟨h1, h⟩ := ite_throw_cond h
  obtain ⟨h2, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨h4, -⟩ := ite_throw_cond h
  simp only [Array.any_eq_true', Array.mem_append, Bool.not_eq_true', Bool.and_eq_false_iff, not_exists,
    not_and] at h1
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true', not_exists, not_and, decide_eq_true_eq, not_lt] at h4
  refine ⟨fun h0 => h2 (by simp [Array.eq_empty_of_size_eq_zero h0]), fun g hg => ⟨?_, h4.2 g (Array.mem_def.2 hg)⟩⟩
  have := h1 g (.inr (Array.mem_def.2 hg))
  simp only [not_or, Bool.not_eq_false] at this
  exact this.1

/-- **What `HmRow.check` checks of the returned outputs**, in an untyped circuit whose unit returns them: the unit's
first output group's ports are 16 bits; there is one leaf map per unit of a template instance, each with one output per
port, each below `nOut`. -/
theorem check_out {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) :
    (∀ b ∈ (c.unit.outGroups.getD 0 default).bits.toList, b = 16) ∧ c.leavesOut.size = c.upv ∧
      ∀ u < c.upv, (c.leavesOut[u]!).size = (c.unit.outGroups.getD 0 default).bits.size ∧
        ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨hug, h⟩ := ite_throw_cond h
  replace h := ite_join h fun ha => (ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, -, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨hlv, h⟩ := ite_throw_cond h
  obtain ⟨seen, hseen, -⟩ := except_bind_ok h
  have leaves := forIn_range_forall
    (P := fun u => (c.leavesOut[u]!).size = (c.unit.outGroups.getD 0 default).bits.size ∧
      ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords)
    (fun u _ _ s t hs => by
      obtain ⟨hc, hs⟩ := ite_throw_cond hs
      obtain ⟨s', hin, hs⟩ := except_bind_ok hs
      cases hs
      refine ⟨⟨_, rfl⟩, ?_, ?_⟩
      · simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not, ho, bne_self_eq_false, Bool.not_false,
          Bool.true_and] at hc
        exact hc.2
      · refine forIn_array_forall (P := fun o => o < c.outWords) (fun o _ s t hs => ?_) hin
        obtain ⟨hc, hs⟩ := ite_throw_cond hs
        cases hs
        refine ⟨⟨_, rfl⟩, ?_⟩
        simp only [Bool.or_eq_true, not_or, decide_eq_true_eq, not_le] at hc
        exact hc.1) hseen
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, not_not] at hlv
  simp only [Bool.or_eq_true, not_or, ho, bne_self_eq_false, Bool.not_false, Bool.true_and, Array.any_eq_true',
    not_exists, not_and, bne_iff_ne, ne_eq, not_not] at hug
  exact ⟨fun b hb => hug.2 b (Array.mem_def.2 hb), hlv.2, fun u hu => leaves u (Nat.zero_le _) hu⟩

theorem foldl_add_const {l : List ℕ} {k : ℕ} (h : ∀ b ∈ l, b = k) (a : ℕ) :
    l.foldl (· + ·) a = a + k * l.length := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih =>
    rw [List.foldl_cons, ih (fun b hb => h b (List.mem_cons_of_mem _ hb)), h x List.mem_cons_self,
      List.length_cons]
    ring

/-- **The returned outputs' shape is the parse's**: in an untyped circuit whose unit returns the outputs, `OutShape` is
what `Net.ofRows` (`built_out`) and `HmRow.check` (`check_out`) checked. -/
theorem outShape_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) (hU : Built c.unit) (ho : c.outNet = c.unitNet) : OutShape c := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨h16, hsz, hlo⟩ := check_out P.typed ho hls
  obtain ⟨hne, hg⟩ := built_out hU
  have hmem : c.unit.outGroups.getD 0 default ∈ c.unit.outGroups.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (Nat.pos_of_ne_zero hne), Option.getD_some]
    exact Array.getElem_mem_toList _
  obtain ⟨hp, hfit⟩ := hg _ hmem
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  refine ⟨hp.2.symm, fun u' hu' => ?_⟩
  have hget : c.leavesOut.getD u' #[] = c.leavesOut[u']! := by
    rw [Array.getD_eq_getD_getElem?, getElem!_pos c.leavesOut u' (by omega),
      Array.getElem?_eq_getElem (by omega), Option.getD_some]
  rw [hget]
  obtain ⟨hl, hlt⟩ := hlo u' hu'
  refine ⟨fun j hj => ⟨?_, ?_⟩, ?_⟩
  · unfold outBits
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
    exact h16 _ (Array.getElem_mem_toList _)
  · rw [getElem!_pos _ j hj]
    exact hlt _ (Array.getElem_mem_toList _)
  · have hsum := foldl_add_const (l := (c.unit.outGroups.getD 0 default).bits.toList) h16 0
    rw [Array.foldl_toList] at hsum
    rw [hsum, Array.length_toList] at hfit
    rw [hl]
    simp only [WORD] at hfit
    omega

end FlockSoundness.Discharge.Layout
