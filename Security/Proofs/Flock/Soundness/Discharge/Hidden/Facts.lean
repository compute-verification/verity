import Proofs.Flock.Soundness.Discharge.Hidden.Spec

/-!
# The facts read off `Spec`

What `Discharge/HiddenExec` reads of `Stmt.setupHidden`, from `setupHidden_spec`: its tags, its leaf scheme
(`setupHidden_hm96`), its well-formedness (`setupHidden_wf`: `StmtWF`, `RegionsWF`) and the drawn public file; and of
`HmOut.parse`, that an untyped one is `HmRow.parse`'s circuit then `HmOut.checkRows` (`parse_untyped`).
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-! ## The parse -/

/-- **`HmOut.parse`** is `HmRow.parse`'s circuit (a typed statement's: on its template, with its block and class, or
on its unit, with its class), then `HmOut.checkRows`. -/
theorem parse_cases {c : Circuit} (h : HmOut.parse tags cf tables = .ok c) :
    HmOut.checkRows c = .ok () ∧
      ((tags.typed = false ∧ HmRow.parse tags cf tables = .ok c) ∨
        (∃ k b tm c₀, HmRow.parse tags cf tables (tmpl := some tm) = .ok c₀ ∧
          c = { c₀ with typed := some b, cls := some k }) ∨
        ∃ k unit c₀, HmRow.parse tags cf tables (pre := #[("unit", unit)]) = .ok c₀ ∧ c = { c₀ with cls := some k }) := by
  unfold HmOut.parse at h
  dsimp only at h
  cases ht : tags.typed
  · simp only [ht, Bool.not_false, ↓reduceIte] at h
    obtain ⟨c₀, hc₀, h⟩ := ebind_ok h
    obtain ⟨u, hu, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hu, .inl ⟨rfl, hc₀⟩⟩
  · simp only [ht, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨⟨k, r⟩, -, h⟩ := ebind_ok h
    dsimp only at h
    cases r with
    | inr t =>
        obtain ⟨c', hc', h⟩ := ebind_ok h
        simp only [pure_bind] at h
        obtain ⟨u, hu, h⟩ := ebind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨hu, .inr (.inl ⟨k, _, _, c', hc', rfl⟩)⟩
    | inl unit =>
        obtain ⟨c', hc', h⟩ := ebind_ok h
        simp only [pure_bind] at h
        obtain ⟨u, hu, h⟩ := ebind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨hu, .inr (.inr ⟨k, unit, c', hc', rfl⟩)⟩

/-- **`parse_untyped`**: an untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows`. -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () := by
  unfold HmOut.parse at h
  simp only [ht, Bool.not_false, ↓reduceIte] at h
  obtain ⟨c₀, hc₀, h⟩ := ebind_ok h
  obtain ⟨u, hu, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact ⟨hc₀, hu⟩

/-- An `HmOut.parse` circuit has the tags' leaf scheme. -/
theorem parse_leafScheme {c : Circuit} (h : HmOut.parse tags cf tables = .ok c) : c.leafScheme = tags.merkleLeaf := by
  obtain ⟨-, ⟨-, hc⟩ | ⟨k, b, tm, c₀, hc, he⟩ | ⟨k, unit, c₀, hc, he⟩⟩ := parse_cases h
  · exact Exec.parse_leafScheme_pre hc
  · have := Exec.parse_leafScheme_tmpl hc; rw [he]; exact this
  · have := Exec.parse_leafScheme_pre hc; rw [he]; exact this

/-- An `HmOut.parse` circuit's slots are `HmRow.check`'s: `slot_log` in 7..k_log. -/
theorem parse_slots {c : Circuit} (h : HmOut.parse tags cf tables = .ok c) :
    ∀ r ∈ c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog := by
  obtain ⟨-, ⟨-, hc⟩ | ⟨k, b, tm, c₀, hc, he⟩ | ⟨k, unit, c₀, hc, he⟩⟩ := parse_cases h
  · obtain ⟨_, hls⟩ := hm_parse_check_none hc
    exact hm_check_slots hls
  · obtain ⟨_, hls⟩ := hm_parse_check_some hc
    have := hm_check_slots hls; rw [he]; exact this
  · obtain ⟨_, hls⟩ := hm_parse_check_none hc
    have := hm_check_slots hls; rw [he]; exact this

/-! ## The regions -/

/-- Every region `HmOut.regions` builds is `mkRegion`'s, so well formed. -/
theorem regions_wf {c : Circuit} {pub : Public} {regs : Array Region} (h : HmOut.regions c pub = .ok regs) :
    ∀ x ∈ regs.toList, RegionWF c.kLog x := by
  unfold HmOut.regions at h
  obtain ⟨-, hall⟩ := Exec.range_mapM_ok h
  intro x hx
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
  exact mkRegion_wf (hall j (by simpa using hj))

/-! ## The facts read off `Spec` -/

/-- `Spec.tagsEq`. -/
theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags :=
  (setupHidden_spec' h).tagsEq

/-- **The leaf scheme**: under `hm96-sha512/v1` tags, a statement `Stmt.setupHidden` accepts commits its rows with
`hm96Sha512`. -/
theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  have s := setupHidden_spec' h
  have hf := s.merkle
  rw [parse_leafScheme s.parse, hl, Exec.find_hm96] at hf
  exact (Option.some.inj hf).symm

/-- **The leaf scheme, plain**: under tags with plain SHA-512 leaves, a statement `Stmt.setupHidden` accepts commits its
rows with `MerkleScheme.sha512 "flock-leaf/sha512-unsalted"`. -/
theorem setupHidden_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
  have s := setupHidden_spec' h
  have hf := s.merkle
  rw [parse_leafScheme s.parse, hl, Exec.find_plain] at hf
  exact (Option.some.inj hf).symm

/-- **The statement is well formed** (`StmtWF`, `RegionsWF`), as `setupH_wf` for `Stmt.setupH`. -/
theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st := by
  have s := setupHidden_spec' h
  exact ⟨wf_of_facts (parse_slots s.parse) (hm_pin_ok s.pin) s.m, s.m, s.mPts, regions_wf s.regions⟩

/-- `Spec.pub`, at a draw. -/
theorem setupHidden_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  obtain ⟨pop, hl, -, hd⟩ := (setupHidden_spec' h).pub
  exact ⟨pop, hl, hd d rfl⟩

end FlockSoundness.Discharge.Hidden
