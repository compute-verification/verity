module

public import Definitions.Core.LevelsCredited
public import Proofs.Core.Levels

/-!
# The credited blocks of levels: what the predicates give

* One level constrains nothing: every set of its blocks is credited at it (`Levels.creditedUnsplit_one`,
  `Levels.creditedFirst_one`).
* The sample draws only blocks of the last level (`Levels.design_draw`, from `Level.sampleBelow_draw`), and a credited
  block is one of them (`Levels.CreditedUnsplit.mem_last`), so a drawn block that meets a credited block is that block:
  the sample checks a credited block only whole (`Levels.CreditedUnsplit.drawn_eq`).
* Fewer credited blocks are still credited (`Levels.CreditedFirst.mono`, `Levels.CreditedUnsplit.mono`).
* Being a block of every later level is no later level having a block strictly inside it
  (`Levels.creditedUnsplit_iff_none_inside`), since each level's blocks lie inside the first level's
  (`Level.nestedBelow_sub_first`).
-/

@[expose] public section

namespace Verity

namespace Level

variable {G : Type} [Fintype G] [DecidableEq G]

/-- Two blocks of one partition that share a gate are one block. -/
theorem part_eq_of_mem {P : Finpartition (Finset.univ : Finset G)} {a b : Finset G} {g : G}
    (ha : a ∈ P.parts) (hb : b ∈ P.parts) (hga : g ∈ a) (hgb : g ∈ b) : a = b := by
  by_contra hne
  exact Finset.disjoint_left.1 (P.disjoint ha hb hne) hga hgb

/-- The sample of the levels `l :: ls` inside a block `X` of the level above draws only blocks of the last level,
inside `X`. -/
theorem sampleBelow_draw {P : Finset (Finset G)} {l : Level G} {ls : List (Level G)}
    (hi : InsideBelow P (l :: ls)) {X : Finset G} (hX : X ∈ P) :
    ∀ ω, ∀ c ∈ (sampleBelow (l :: ls) X).draw ω,
      c ∈ ((l :: ls).getLast (List.cons_ne_nil l ls)).blocks.parts ∧ c ⊆ X := by
  induction ls generalizing P l X with
  | nil => exact hi.1 X hX
  | cons l' ls ih =>
    intro ω c hc
    obtain ⟨Y, hY, hcY⟩ := Finset.mem_biUnion.1 hc
    obtain ⟨hYl, hYX⟩ := hi.1 X hX ω.1 Y hY
    obtain ⟨h₁, h₂⟩ := ih hi.2 hYl (ω.2 Y) c hcY
    exact ⟨by simpa only [List.getLast_cons_cons] using h₁, h₂.trans hYX⟩

/-- Below the first of the levels `l :: ls`, every block of a later level lies inside a block of `l`. -/
theorem nestedBelow_sub_first {P : Finset (Finset G)} {l : Level G} {ls : List (Level G)}
    (hn : NestedBelow P (l :: ls)) : ∀ l' ∈ ls, ∀ c ∈ l'.blocks.parts, ∃ X ∈ l.blocks.parts, c ⊆ X := by
  induction ls generalizing P l with
  | nil => intro l' hl'; exact absurd hl' List.not_mem_nil
  | cons l₂ ls ih =>
    intro l' hl' c hc
    rcases List.mem_cons.1 hl' with rfl | hl'
    · exact hn.2.1 c hc
    · obtain ⟨X, hX, hcX⟩ := ih hn.2 l' hl' c hc
      obtain ⟨Y, hY, hXY⟩ := hn.2.1 X hX
      exact ⟨Y, hY, hcX.trans hXY⟩

end Level

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **The sample draws only blocks of the last level.** -/
theorem design_draw (L : Levels G) (h : L.levels ≠ []) :
    ∀ ω, ∀ c ∈ L.design.draw ω, c ∈ (L.levels.getLast h).blocks.parts := by
  obtain ⟨levels, hn, hi⟩ := L
  cases levels with
  | nil => exact absurd rfl h
  | cons l ls =>
    intro ω c hc
    exact (Level.sampleBelow_draw hi (Finset.mem_singleton_self _) ω c hc).1

/-- **One level constrains nothing**: every set of its blocks is credited first. -/
theorem creditedFirst_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {Cr : Finset (Finset G)}
    (hCr : Cr ⊆ l.blocks.parts) : L.CreditedFirst Cr :=
  ⟨l, [], hL, fun _ hb => hCr hb⟩

/-- **One level constrains nothing**: every set of its blocks is credited and unsplit. -/
theorem creditedUnsplit_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {Cr : Finset (Finset G)}
    (hCr : Cr ⊆ l.blocks.parts) : L.CreditedUnsplit Cr :=
  ⟨L.creditedFirst_one hL hCr, by simp [hL]⟩

theorem CreditedFirst.mono {L : Levels G} {Cr Cr' : Finset (Finset G)} (h : L.CreditedFirst Cr) (hs : Cr' ⊆ Cr) :
    L.CreditedFirst Cr' :=
  let ⟨l, ls, hL, h₁⟩ := h
  ⟨l, ls, hL, fun b hb => h₁ b (hs hb)⟩

/-- **Monotone in the credited set.** -/
theorem CreditedUnsplit.mono {L : Levels G} {Cr Cr' : Finset (Finset G)} (h : L.CreditedUnsplit Cr)
    (hs : Cr' ⊆ Cr) : L.CreditedUnsplit Cr' :=
  ⟨h.1.mono hs, fun l hl b hb => h.2 l hl b (hs hb)⟩

theorem CreditedUnsplit.ne_nil {L : Levels G} {Cr : Finset (Finset G)} (h : L.CreditedUnsplit Cr) :
    L.levels ≠ [] := by
  obtain ⟨⟨l, ls, hL, -⟩, -⟩ := h
  simp [hL]

/-- A credited block is a block of every level. -/
theorem CreditedUnsplit.mem_levels {L : Levels G} {Cr : Finset (Finset G)} (h : L.CreditedUnsplit Cr)
    {b : Finset G} (hb : b ∈ Cr) : ∀ l ∈ L.levels, b ∈ l.blocks.parts := by
  obtain ⟨⟨l₁, ls, hL, h₁⟩, h₂⟩ := h
  intro l hl
  rw [hL] at hl
  rcases List.mem_cons.1 hl with rfl | hl
  · exact h₁ b hb
  · exact h₂ l (by simpa [hL] using hl) b hb

/-- **A credited block is a block of the last level.** -/
theorem CreditedUnsplit.mem_last {L : Levels G} {Cr : Finset (Finset G)} (h : L.CreditedUnsplit Cr)
    {b : Finset G} (hb : b ∈ Cr) (hne : L.levels ≠ []) : b ∈ (L.levels.getLast hne).blocks.parts :=
  h.mem_levels hb _ (List.getLast_mem hne)

/-- **The sample checks a credited block only whole**: a block the level-by-level sample draws that meets a credited
block is that block. -/
theorem CreditedUnsplit.drawn_eq {L : Levels G} {Cr : Finset (Finset G)} (h : L.CreditedUnsplit Cr)
    {b : Finset G} (hb : b ∈ Cr) : ∀ ω, ∀ c ∈ L.design.draw ω, (c ∩ b).Nonempty → c = b := by
  intro ω c hc ⟨g, hg⟩
  rw [Finset.mem_inter] at hg
  exact Level.part_eq_of_mem (L.design_draw h.ne_nil ω c hc) (h.mem_last hb h.ne_nil) hg.1 hg.2

/-- **Unsplit is no later level drawing inside**: a credited block of the first level is a block of every later
level exactly when no later level has a block strictly inside it. -/
theorem creditedUnsplit_iff_none_inside (L : Levels G) (Cr : Finset (Finset G)) :
    L.CreditedUnsplit Cr ↔
      L.CreditedFirst Cr ∧ ∀ l ∈ L.levels.tail, ∀ b ∈ Cr, ∀ c ∈ l.blocks.parts, c ⊆ b → c = b := by
  constructor
  · rintro ⟨hF, hU⟩
    refine ⟨hF, fun l hl b hb c hc hcb => ?_⟩
    obtain ⟨g, hg⟩ := l.blocks.nonempty_of_mem_parts hc
    exact Level.part_eq_of_mem hc (hU l hl b hb) hg (hcb hg)
  · rintro ⟨hF, hI⟩
    refine ⟨hF, fun l hl b hb => ?_⟩
    obtain ⟨l₁, ls, hL, h₁⟩ := hF
    have hb₁ := h₁ b hb
    obtain ⟨g, hg⟩ := l₁.blocks.nonempty_of_mem_parts hb₁
    obtain ⟨c, hc, hgc⟩ := l.blocks.exists_mem (Finset.mem_univ g)
    have hn := L.nested
    rw [hL] at hn
    have hl' : l ∈ ls := by simpa [hL] using hl
    obtain ⟨X, hX, hcX⟩ := Level.nestedBelow_sub_first hn l hl' c hc
    have hXb : X = b := Level.part_eq_of_mem hX hb₁ (hcX hgc) hg
    subst hXb
    rw [← hI l hl X hb c hc hcX]
    exact hc

end Levels

end Verity
