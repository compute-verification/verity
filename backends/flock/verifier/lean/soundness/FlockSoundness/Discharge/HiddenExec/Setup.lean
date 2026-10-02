import FlockSoundness.Discharge.HiddenExec.Statements
import FlockSoundness.Discharge.Hidden.Theorems

/-!
# What `Stmt.setupHidden` returns, as this area reads it

`setupOne_hidden`, `setupOne_public`: `setupOne` is `setupHidden` under hidden outputs and `setupH` otherwise.

The rest are what this area reads of `Stmt.setupHidden`, all from [wave2-hidden]'s `Hidden.setupHidden_spec`
(`Discharge/Hidden/`). `setupHidden_tags`, `setupHidden_drawn`, `setupHidden_parseC` and `setupHidden_n` are projections
of `Spec` (`tagsEq`, `pub`, `parse`, and `n`, `blocksReal`, `nbl`). `setupHidden_hm96` and `setupHidden_wf` also read
what `HmOut.parse` and `HmOut.regions` check (`hmOut_parse_facts`, `hmOut_regions_wf`), and `parse_untyped` is
`HmOut.parse` without typed rows: `HmRow.parse`, then `HmOut.checkRows`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

theorem setupOne_hidden (hh : tags.hiddenOutputs = true) :
    setupOne tags cf pf coins tables draw partition program =
      Stmt.setupHidden tags cf pf coins tables draw partition program := by
  simp [setupOne, hh]

theorem setupOne_public (hh : tags.hiddenOutputs = false) :
    setupOne tags cf pf coins tables draw partition program =
      Stmt.setupH tags cf pf coins tables draw partition program := by
  simp [setupOne, hh]

set_option maxHeartbeats 4000000 in
/-- What `HmOut.parse` checks of its circuit, typed or not: the leaf scheme is the tags', and every range's slots are
`7 ≤ slotLog ≤ kLog` (`HmRow.check`). -/
theorem hmOut_parse_facts {bytes : ByteArray} {c : Circuit} (h : HmOut.parse tags bytes tables = .ok c) :
    c.leafScheme = tags.merkleLeaf ∧ (∀ r ∈ c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog) := by
  unfold HmOut.parse at h
  dsimp only at h
  cases ht : tags.typed
  · simp only [ht, Bool.not_false, ↓reduceIte] at h
    obtain ⟨c₀, h0, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨_, hc⟩ := hm_parse_check_none h0
    exact ⟨parse_leafScheme_pre h0, hm_check_slots hc⟩
  · simp only [ht, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨⟨k, r⟩, -, h⟩ := ebind_ok h
    dsimp only at h
    cases r with
    | inr t =>
      obtain ⟨c₁, h1, h⟩ := ebind_ok h
      simp only [pure_bind] at h
      obtain ⟨_, -, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨_, hc⟩ := hm_parse_check_some h1
      have e1 := parse_leafScheme_tmpl h1
      have e2 := hm_check_slots hc
      exact ⟨e1, e2⟩
    | inl unit =>
      obtain ⟨c₁, h1, h⟩ := ebind_ok h
      simp only [pure_bind] at h
      obtain ⟨_, -, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨_, hc⟩ := hm_parse_check_none h1
      have e1 := parse_leafScheme_pre h1
      have e2 := hm_check_slots hc
      exact ⟨e1, e2⟩

/-- Every region `HmOut.regions` builds is a `mkRegion`, so WF at the circuit's `kLog`. -/
theorem hmOut_regions_wf {c : Circuit} {pub : Public} {regs : Array Region} (h : HmOut.regions c pub = .ok regs) :
    ∀ x ∈ regs.toList, RegionWF c.kLog x := by
  unfold HmOut.regions at h
  dsimp only at h
  intro x hx
  obtain ⟨p, hp⟩ := mapM_ok_mem h x hx
  exact mkRegion_wf hp

/-- `Spec.tagsEq`. -/
theorem setupHidden_tags (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.tags = tags :=
  (Hidden.setupHidden_spec h).tagsEq

/-- `Spec.merkle` at `Spec.parse`'s leaf scheme, which is the tags'. -/
theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 := by
  have s := Hidden.setupHidden_spec h
  have h2 := s.merkle
  rw [(hmOut_parse_facts s.parse).1, hl, find_hm96] at h2
  exact (Option.some.inj h2).symm

/-- `StmtWF` from `Spec.parse`'s slots, `Spec.pin` and `Spec.m`; `RegionsWF` from `Spec.m`, `Spec.mPts` and
`Spec.regions`. -/
theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st := by
  have s := Hidden.setupHidden_spec h
  exact ⟨wf_of_facts (hmOut_parse_facts s.parse).2 (hm_pin_ok s.pin) s.m, s.m, s.mPts, hmOut_regions_wf s.regions⟩

/-- `Spec.pub`, at a draw. -/
theorem setupHidden_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ HmOut.drawn st.c pop d = .ok st.pub := by
  obtain ⟨pop, hl, -, hd⟩ := (Hidden.setupHidden_spec h).pub
  exact ⟨pop, hl, hd d rfl⟩

/-- `Spec.parse`: the statement's circuit is the one `HmOut.parse` accepts. -/
theorem setupHidden_parseC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parse tags cf tables = .ok st.c :=
  (Hidden.setupHidden_spec h).parse

/-- `Spec.n`, `Spec.blocksReal`, `Spec.nbl`: `n` is the (drawn) public file's, and `nbl = max 3 (log2ceil ⌈n / c.g⌉)`,
as setupH's (`Integrate.setupH_n`). -/
theorem setupHidden_n (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  have s := Hidden.setupHidden_spec h
  exact ⟨s.n, by rw [s.nbl, s.blocksReal]⟩

set_option maxHeartbeats 4000000 in
/-- An untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows`. -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () := by
  unfold HmOut.parse at h
  dsimp only at h
  simp only [ht, Bool.not_false, ↓reduceIte] at h
  obtain ⟨c₀, h0, h⟩ := ebind_ok h
  obtain ⟨u, hu, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact ⟨h0, hu⟩

end FlockSoundness.Discharge.HiddenExec
