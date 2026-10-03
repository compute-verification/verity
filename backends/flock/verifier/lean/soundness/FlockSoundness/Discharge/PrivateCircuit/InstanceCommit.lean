import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs

/-!
# The program commitment binds, and the session opens it

* `progCommit_binding`: two openings of one registered root to two programs' leaves conflict at the first position
  where the leaves differ, so the registered reads' extractor collides (`TwoOpenings.collides`). `rowCommit_binding`: the
  same for the rows, through hm96's leaf where two rows have one leaf (`ExplicitLeaf.pair_spec`).
* `ProgSites.opens_of_checked`: where the verifier's registered check holds at a registration (`Reads.Checked`), the
  session's leaves at the program's sites open the program commitment. `ProgSites.opens_reg`: so do the registrant's
  (`Reads.committed`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Registered Refine Binding Audit

theorem ProgOpening.diff_spec {L : ℕ} {op op' : ProgOpening L} (h : ∃ i, op.leaf i ≠ op'.leaf i) :
    ∃ i, op.diff op' = some i ∧ op.leaf i ≠ op'.leaf i :=
  ⟨Classical.choose h, by simp only [ProgOpening.diff, h, ↓reduceDIte], Classical.choose_spec h⟩

theorem RowOpening.diff_spec {L : ℕ} {Val : Type} {op op' : RowOpening L Val} (h : ∃ i, op.row i ≠ op'.row i) :
    ∃ i, op.diff op' = some i ∧ op.row i ≠ op'.row i :=
  ⟨Classical.choose h, by simp only [RowOpening.diff, h, ↓reduceDIte], Classical.choose_spec h⟩

/-- **The program commitment binds under SHA-512**: two openings of one registered root to two programs' leaves give
the registered reads' explicit collision, at the first position where the leaves differ. -/
theorem progCommit_binding (L : ℕ) : (progCommit L).Binding := by
  rintro rt p p' op op' ⟨hp, hl, ho⟩ ⟨hp', hl', ho'⟩ hne
  subst hp hl hl'
  have hd : ∃ i, op.leaf i ≠ op'.leaf i := by
    by_contra hc
    exact hne (funext fun i => not_not.1 fun h => hc ⟨i, h⟩)
  obtain ⟨i, hi, hd⟩ := ProgOpening.diff_spec hd
  show Binding.Collides (match op.diff op' with | some i => (op.at op' i).pair | none => ([], []))
  rw [hi]
  exact (op.at op' i).collides ⟨hd, ho i, ho' i⟩

/-- **The row commitment binds under SHA-512**: two openings of one registered root to two programs' rows give an
explicit collision, at the first position where the rows differ: hm96's when the two leaves agree, else the registered
reads'. -/
theorem rowCommit_binding (L : ℕ) {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) :
    (rowCommit L enc hinj).Binding := by
  rintro rt p p' op op' ⟨hp, hr, ho⟩ ⟨hp', hr', ho'⟩ hne
  subst hp hr hr'
  have hd : ∃ i, op.row i ≠ op'.row i := by
    by_contra hc
    exact hne (funext fun i => not_not.1 fun h => hc ⟨i, h⟩)
  obtain ⟨i, hi, hd⟩ := RowOpening.diff_spec hd
  classical
  show Binding.Collides (match op.diff op' with
    | some i =>
      if (ExplicitLeaf.hm96 enc hinj).leaf (op.row i) (op.salt i) = (ExplicitLeaf.hm96 enc hinj).leaf (op'.row i) (op'.salt i)
      then (ExplicitLeaf.hm96 enc hinj).pair (op.row i) (op.salt i) (op'.row i) (op'.salt i)
      else ((op.toProg (ExplicitLeaf.hm96 enc hinj)).at (op'.toProg (ExplicitLeaf.hm96 enc hinj)) i).pair
    | none => ([], []))
  rw [hi]
  dsimp only
  split_ifs with hleaf
  · exact (ExplicitLeaf.hm96 enc hinj).pair_spec _ _ _ _ hd hleaf
  · exact ((op.toProg _).at (op'.toProg _) i).collides ⟨hleaf, ho i, ho' i⟩

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}
  {rd : Registered.Reads lay rs} {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)

/-- **The session opens the program commitment** where the verifier's registered check holds at the registration `R`
through `path`: its leaves at the program's sites open `rt` at the program's positions. -/
theorem ProgSites.opens_of_checked {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) :
    (progCommit L).Opens rt (ps.leaves R) (ps.opening R path) := by
  refine ⟨rfl, rfl, fun i => ?_⟩
  have h := hc _ (ps.isRead i)
  rwa [ps.port i, ps.readAt i] at h

/-- **The registrant opens the program commitment**: its leaves at the program's sites, hm96's leaves of the rows and
salts it committed, open `rt` at the program's positions (`Reads.committed`). -/
theorem ProgSites.opens_reg : (progCommit L).Opens rt ps.regLeaves ps.regOpening := by
  refine ⟨rfl, rfl, fun i => ?_⟩
  have h := rd.committed _ (ps.isRead i)
  rwa [ps.port i, ps.readAt i] at h

end Sites

end FlockSoundness.Discharge.PrivateCircuit
